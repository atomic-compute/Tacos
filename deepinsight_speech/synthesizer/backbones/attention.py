# ==== ## ==== ## ==== ## ==== ## ==== ## ==== #
#           SELF ATTENTION LAYERS
# ==== ## ==== ## ==== ## ==== ## ==== ## ==== #

from typing import Tuple, List
import tensorflow.keras.layers as KL
import tensorflow as tf
from functools import reduce
from .embedding import PreNet, ZoneoutLSTMCell, Conv1d, HighwayNet
from ..modules.helpers import (TransformerTrainingHelper, TrainingMgcLf0Helper,
                               StopTokenBasedInferenceHelper, StopTokenBasedMgcLf0InferenceHelper)
import tensorflow_addons as tfa
from tensorflow_addons.seq2seq import BasicDecoder, dynamic_decode
import numpy as np
from collections import namedtuple
import tensorflow.keras.backend as K
from easydict import EasyDict
import enum
from abc import ABCMeta, abstractmethod
from six import add_metaclass


class AttentionOptions(EasyDict):
    def __init__(self, **kwargs) -> None:
        assert set(kwargs.keys()) == set(["num_units",
                                          "memory",
                                          "memory_sequence_length",
                                          "attention_kernel",
                                          "attention_filters",
                                          "smoothing",
                                          "cumulative_weights",
                                          "use_transition_agent",
                                          "teacher_alignments"])
        super().__init__(**kwargs)


class AttentionFactory(enum.Enum):
    forward = 'ForwardAttention'
    location_sensitive = 'LocationSensitiveAttention'
    teacher_forcing_forward = 'TeacherForcingForwardAttention'
    teacher_forcing_additive = 'TeacherForcingAdditiveAttention'
    additive = 'tfa.seq2seq.BahdanauAttention'


class ForwardAttentionState(
    namedtuple("ForwardAttentionState", ["alignments", "alpha", "u"])
):
    pass


@add_metaclass(ABCMeta)
class TransparentRNNCellLike:
    """ RNNCell-like base class that do not create scopes
    """
    @abstractmethod
    def __call__(self, inputs, state, **kwargs):
        """ Dummy RNN CELL like class that represent RNN cell

        Args:
            inputs ([type]): [description]
            state ([type]): [description]

        Returns:
            [type]: [description]
        """
        pass


class RNNStateHistoryWrapperState(
        namedtuple("RNNStateHistoryWrapperState", ["rnn_state", "rnn_state_history", "time"])):
    pass


class TransformerWrapperState(namedtuple(
    "TransformerWrapperState", ["rnn_state", "alignments"])
):
    pass


class RNNStateHistoryWrapper(TransparentRNNCellLike):

    def __init__(self, cell: KL.LSTMCell, max_iter):
        self._cell = cell
        self._max_iter = max_iter
        self.state_size = RNNStateHistoryWrapperState(
            self._cell.state_size,
            tf.TensorShape([None, None, self.output_size]), tf.TensorShape([])
        )
        self.output_size = self._cell.output_size

    def zero_state(self, batch_size, dtype):
        rnn_state = self._cell.zero_state(batch_size, dtype)
        history = tf.zeros(shape=[batch_size, 0, self.output_size], dtype=dtype)
        # avoid Tensor#set_shape which merge unknown shape with known shape
        history._shape_val = tf.TensorShape([None, None, self.output_size])  # pylint: disable=protected-access
        time = tf.zeros([], dtype=tf.int32)
        return RNNStateHistoryWrapperState(rnn_state, history, time)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def __call__(self, inputs, state: RNNStateHistoryWrapperState):
        output, new_rnn_state = self._cell(inputs, state.rnn_state)
        new_history = tf.concat([state.rnn_state_history,
                                 tf.expand_dims(output, axis=1)], axis=1)
        new_history.set_shape([None, None, self.output_size])
        new_state = RNNStateHistoryWrapperState(new_rnn_state, new_history, state.time + 1)
        return output, new_state


class TransformerWrapper(TransparentRNNCellLike):

    def __init__(self, cell: RNNStateHistoryWrapper, transformers):
        self._cell = cell
        self._transformers = transformers
        self.state_size = TransformerWrapperState(
            self._cell.state_size, [(None, None) for _ in self._transformers]
        )
        self.output_size = TransformerWrapperState(
            self._cell.output_size, [(None, None) for _ in self._transformers]
        )

    def get_initial_state(self, batch_size, dtype=None):
        def initial_alignment(num_heads):
            ia = tf.zeros([batch_size, 0, 0], dtype)
            ia._shape_val = tf.TensorShape([None, None, None])  # pylint: disable=protected-access
            return [ia] * num_heads

        return TransformerWrapperState(
            self._cell.zero_state(batch_size, dtype), [
                ia for ia in initial_alignment(2)
                for _ in self._transformers
            ]
        )

    def __call__(self, inputs, state: TransformerWrapperState, training=None, **kwargs):
        memory_sequence_length = kwargs.pop('memory_sequence_length', None)
        output, new_rnn_state = self._cell(inputs, state.rnn_state, training=training, **kwargs)
        history = new_rnn_state.rnn_state_history

        def self_attend(input, alignments, layer):
            output, alignment = layer(input, memory_sequence_length=memory_sequence_length)
            return output, alignments + alignment

        transformed, alignments = reduce(lambda acc, sa: self_attend(acc[0], acc[1], sa),
                                         self._transformers,
                                         (history, []))
        output_element = transformed[:, -1, :]
        new_state = TransformerWrapperState(new_rnn_state, alignments)
        return output_element, new_state


class OutputMgcLf0AndStopTokenWrapper(KL.SimpleRNNCell):

    def __init__(self, cell, mgc_out_units, lf0_out_units, dtype=None):
        super(OutputMgcLf0AndStopTokenWrapper, self).__init__()
        self._mgc_out_units = mgc_out_units
        self._lf0_out_units = lf0_out_units
        self._cell = cell
        self.mgc_out_projection1 = tf.layers.Dense(cell.output_size, activation=tf.nn.tanh, dtype=dtype)
        self.mgc_out_projection2 = tf.layers.Dense(mgc_out_units, dtype=dtype)
        self.lf0_out_projection = tf.layers.Dense(lf0_out_units, dtype=dtype)
        self.stop_token_projection = tf.layers.Dense(1, dtype=dtype)
        self.state_size = self._cell.state_size
        self.output_size = (self._mgc_out_units, self._lf0_out_units, 1)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state, **kwargs):
        output, res_state = self._cell(inputs, state, **kwargs)
        mgc_output = self.mgc_out_projection2(self.mgc_out_projection1(output))
        lf0_output = self.lf0_out_projection(output)
        stop_token = self.stop_token_projection(output)
        return (mgc_output, lf0_output, stop_token), res_state


class OutputAndStopTokenTransparentWrapper(TransparentRNNCellLike):

    def __init__(self, cell, out_units, out_projection, stop_token_projection):
        self._cell = cell
        self._out_units = out_units
        self.out_projection = out_projection
        self.stop_token_projection = stop_token_projection
        self.state_size = self._cell.state_size
        self.output_size = (self._out_units, 1)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def __call__(self, inputs, state, training=None, **kwargs):
        output, res_state = self._cell(inputs, state, training=training, **kwargs)
        mel_output = self.out_projection(output)
        stop_token = self.stop_token_projection(output)
        return (mel_output, stop_token), res_state


class OutputMgcLf0AndStopTokenTransparentWrapper(TransparentRNNCellLike):

    def __init__(
        self,
        cell,
        mgc_out_units, lf0_out_units, mgc_out_projection,
        lf0_out_projection, stop_token_projection
    ):
        self._mgc_out_units = mgc_out_units
        self._lf0_out_units = lf0_out_units
        self._cell = cell
        self.mgc_out_projection = mgc_out_projection
        self.lf0_out_projection = lf0_out_projection
        self.stop_token_projection = stop_token_projection

        self.state_size = self._cell.state_size
        self.output_size = (self._mgc_out_units, self._lf0_out_units, 1)
        self.get_initial_state = (lambda self, batch_size, dtype: self._cell.zero_state(batch_size, dtype))

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def __call__(self, inputs, state, **kwargs):
        output, res_state = self._cell(inputs, state, **kwargs)
        mgc_output = self.mgc_out_projection(output)
        lf0_output = self.lf0_out_projection(output)
        stop_token = self.stop_token_projection(output)
        return (mgc_output, lf0_output, stop_token), res_state


def _location_sensitive_score(W_query, W_fill, W_keys):
    # TODO: Verify and Fix this
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or tf.shape(W_keys)[-1]

    # attention_variable
    v_a = tf.Variable(
        shape=[num_units],
        dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer()
    )
    # attention_bias
    b_a = tf.Variable(
        shape=[num_units],
        dtype=dtype,
        initializer=tf.zeros_initializer()
    )

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fill + b_a), axis=[2])


def _calculate_context(alignments, values):
    '''
    This is a duplication of tensorflow.contrib.seq2seq.attention_wrapper._compute_attention.
    ToDo: Avoid the redundant computation. This requires abstraction of AttentionWrapper itself.
    :param alignments: [batch_size, 1, memory_time]
    :param values: [batch_size, memory_time, memory_size]
    :return:
    '''
    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    context = tf.matmul(expanded_alignments, values)  # [batch_size, 1, memory_size]
    context = tf.squeeze(context, [1])  # [batch_size, memory_size]
    return context


class SelfAttention(KL.Layer):

    def __init__(
            self,
            model_dim,
            num_heads,
            drop_rate,
            use_padding_mask=False, use_subsequent_mask=False,
            name=None, dtype=None, **kwargs
    ):
        super(SelfAttention, self).__init__(name=name, dtype=dtype, **kwargs)
        self.attention = MultiHeadAttention(
            model_dim, num_heads, drop_rate,
            use_padding_mask=use_padding_mask,
            use_subsequent_mask=use_subsequent_mask,
            dtype=dtype
        )

    def call(self, inputs, training=None, memory_sequence_length=None, **kwargs):
        key, value, query = (inputs, inputs, inputs)
        return self.attention((key, value, query), memory_sequence_length=memory_sequence_length)


class MultiHeadAttention(KL.Layer):

    def __init__(
        self,
        model_dim,
        num_heads,
        drop_rate,
        use_padding_mask=False,
        use_subsequent_mask=False,
        name=None, dtype=None, **kwargs
    ):
        super(MultiHeadAttention, self).__init__(name=name, dtype=dtype, **kwargs)
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.drop_rate = drop_rate
        self.use_padding_mask = use_padding_mask
        self.use_subsequent_mask = use_subsequent_mask
        # ToDo: remove bias from projections
        self.key_projection = KL.Dense(model_dim, dtype=dtype)
        self.value_projection = KL.Dense(model_dim, dtype=dtype)
        self.query_projection = KL.Dense(model_dim, dtype=dtype)
        self.output_projection = KL.Dense(model_dim, dtype=dtype)
        self.attention_mechanism = ScaledDotProductAttentionMechanism(
            self.key_projected, self.value_projected, self.num_heads,
            drop_rate=self.drop_rate,
            use_padding_mask=self.use_padding_mask,
            use_subsequent_mask=self.use_subsequent_mask
        )

    def call(self, inputs, training=None, memory_sequence_length=None, **kwargs):
        key, value, query = inputs
        shape = tf.shape(key)
        head_shape = [shape[0], shape[1], self.num_heads, self.head_dim]
        # (B, T, model_dim) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        _key_projected = tf.transpose(tf.reshape(self.key_projection(key), shape=head_shape), perm=[0, 2, 1, 3])
        _key_projected.set_shape([None, self.num_heads, None, self.head_dim])
        _value_projected = tf.transpose(tf.reshape(self.value_projection(value), shape=head_shape), perm=[0, 2, 1, 3])
        _value_projected.set_shape([None, self.num_heads, None, self.head_dim])
        _query_projected = tf.transpose(tf.reshape(self.query_projection(query), shape=head_shape), perm=[0, 2, 1, 3])
        _query_projected.set_shape([None, self.num_heads, None, self.head_dim])
        x, alignment = self.attention_mechanism(_query_projected, memory_sequence_length=memory_sequence_length)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), shape=[shape[0], shape[1], self.num_heads * self.head_dim])
        output = self.output_projection(x)
        alignment = [alignment[:, i, :, :] for i in range(self.num_heads)]
        return output, alignment


class SelfAttentionTransformer(KL.Layer):

    def __init__(
        self,
        out_units=32,
        num_conv_layers=1,
        kernel_size=5,
        self_attention_out_units=256,
        self_attention_num_heads=2,
        self_attention_drop_rate=0.05,
        use_subsequent_mask=False,
        name=None, dtype=None, **kwargs
    ):
        super(SelfAttentionTransformer, self).__init__(name=name, dtype=dtype, **kwargs)
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.self_attention = SelfAttention(
            self_attention_out_units,
            self_attention_num_heads,
            self_attention_drop_rate,
            use_subsequent_mask=use_subsequent_mask,
            dtype=dtype
        )

        self.transform_layers = [tf.layers.Dense(out_units, activation=tf.nn.tanh, dtype=dtype)]

    def build(self, _):
        self.built = True

    def call(self, inputs, memory_sequence_length=None, **kwargs):
        self_attention_output, self_attention_alignment = self.self_attention(
            inputs, memory_sequence_length=memory_sequence_length
        )

        transformed = reduce(lambda acc, l: l(acc), self.transform_layers, self_attention_output)
        residual = inputs + transformed
        return residual, self_attention_alignment


class AttentionRNN(KL.SimpleRNNCell):

    def __init__(
        self,
        cell,
        prenets: Tuple[PreNet],
        attention_mechanism: List[tfa.seq2seq.AttentionMechanism],
        name=None, dtype=None, **kwargs
    ):
        super(AttentionRNN, self).__init__(name=name, dtype=dtype, **kwargs)
        attention_cell = tfa.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            alignment_history=True,
            output_attention=False)
        # prenet -> attention
        prenet_cell = DecoderPreNetWrapper(attention_cell, prenets)
        # prenet -> attention -> concat
        concat_cell = ConcatOutputAndAttentionWrapper(prenet_cell)
        self._cell = concat_cell
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        return self._cell(inputs, state, training=training, **kwargs)


class DecoderRNNV2(KL.SimpleRNNCell):

    def __init__(self,
                 out_units,
                 attention_cell: AttentionRNN,
                 is_training, zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
                 lstm_impl=KL.LSTMCell,
                 name=None, dtype=None, **kwargs):
        super(DecoderRNNV2, self).__init__(name=name, **kwargs)

        self._cell = KL.StackedRNNCells([
            attention_cell,
            ZoneoutLSTMCell(out_units, is_training, zoneout_factor_cell, zoneout_factor_output, lstm_impl=lstm_impl,
                            dtype=dtype),
            ZoneoutLSTMCell(out_units, is_training, zoneout_factor_cell, zoneout_factor_output, lstm_impl=lstm_impl,
                            dtype=dtype),
        ], state_is_tuple=True)

        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        return self._cell(inputs, state, training=training, **kwargs)


# TODO: Look tf.compat.v1.nn.rnn_cell.RNNCell for implementation and replicate here
class DecoderRNNV1(KL.SimpleRNNCell):
    def __init__(
        self,
        units,
        attention_cell: AttentionRNN,
        activation=None,
        use_bias=True,
        name=None, dtype=None, **kwargs
    ):
        super(DecoderRNNV1, self).__init__(
            units,
            use_bias=use_bias,
            name=name, activation=activation, dtype=dtype, **kwargs
        )

        self._cell = KL.StackedRNNCells([
            OutputProjectionWrapper(attention_cell, units),
            tf.nn.rnn_cell.ResidualWrapper(KL.GRUCell(units, dtype=dtype)),
            tf.nn.rnn_cell.ResidualWrapper(KL.GRUCell(units, dtype=dtype)),
        ])
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        return self._cell(inputs, state, training=training, **kwargs)


class OutputProjectionWrapper(KL.SimpleRNNCell):
    """ Compatible with tensorflow.contrib.rnn.OutputProjectionWrapper.
    Support dtype argument as other RNNCells do.
    """

    def __init__(
        self,
        cell,
        output_size,
        activation=None,
        use_bias=True,
        name=None, dtype=None, **kwargs
    ):
        super(OutputProjectionWrapper, self).__init__(name=name, dtype=dtype, **kwargs)
        assert output_size >= 1, ValueError("Parameter output_size must be > 0: %d." % output_size)
        self._cell = cell
        self._output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def build(self, _):
        input_dim = self._cell.output_size
        self._kernel = self.add_weight("kernel", shape=[input_dim, self._output_size])
        self._bias = self.add_weight("bias", shape=[self._output_size])
        self.built = True

    def call(self, inputs, state, training=None, **kwargs):
        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self._cell(inputs, state, training=training)
        projected = tf.tensordot(output, self._kernel, [[len(output.shape.as_list()) - 1], [0]])
        projected = tf.nn.bias_add(projected, self._bias) if self.use_bias else projected
        projected = self.activation(projected) if self.activation else projected
        return projected, res_state


class DecoderMgcLf0PreNetWrapper(KL.SimpleRNNCell):

    def __init__(self, cell: KL.SimpleRNNCell, mgc_prenets: Tuple[PreNet], lf0_prenets: Tuple[PreNet]):
        super(DecoderMgcLf0PreNetWrapper, self).__init__()
        self._cell = cell
        self.mgc_prenets = mgc_prenets
        self.lf0_prenets = lf0_prenets
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        speaker_embed = kwargs.pop('speaker_embed', None)
        mgc_input, lf0_input = inputs
        mgc_prenet_output = reduce(lambda acc, pn: pn(acc, speaker_embed), self.mgc_prenets, mgc_input)
        lf0_prenet_output = reduce(lambda acc, pn: pn(acc, speaker_embed), self.lf0_prenets, lf0_input)
        prenet_output = tf.concat([mgc_prenet_output, lf0_prenet_output], axis=-1)
        return self._cell(prenet_output, state, training=training)


class ConcatOutputAndAttentionWrapper(KL.SimpleRNNCell):

    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        speaker_embed = kwargs.pop('speaker_embed', None)
        output, res_state = self._cell(inputs, state, training=training, speaker_embed=speaker_embed)
        return tf.concat([output, res_state.attention], axis=-1), res_state


class DualSourceAttentionRNN(KL.SimpleRNNCell):

    def __init__(self, cell, prenets: Tuple[PreNet],
                 attention_mechanisms: List[tfa.seq2seq.AttentionMechanism],
                 name=None, **kwargs):
        super(DualSourceAttentionRNN, self).__init__(name=name, **kwargs)
        attention_cell = tfa.seq2seq.AttentionWrapper(
            cell,
            attention_mechanisms,
            alignment_history=True,
            output_attention=False)
        prenet_cell = DecoderPreNetWrapper(attention_cell, prenets)
        concat_cell = ConcatOutputAndAttentionWrapper(prenet_cell)
        self._cell = concat_cell
        self.output_size = self._cell.output_size
        self.state_size = self._cell.state_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        return self._cell(inputs, state, training=training, **kwargs)


class MgcLf0AttentionRNN(KL.SimpleRNNCell):

    def __init__(
        self, cell,
        mgc_prenets: Tuple[PreNet],
        lf0_prenets: Tuple[PreNet],
        attention_mechanism,
        name=None, **kwargs
    ):
        super(MgcLf0AttentionRNN, self).__init__(name=name, **kwargs)
        attention_cell = tfa.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            alignment_history=True,
            output_attention=False)
        prenet_cell = DecoderMgcLf0PreNetWrapper(attention_cell, mgc_prenets, lf0_prenets)
        concat_cell = ConcatOutputAndAttentionWrapper(prenet_cell)
        self._cell = concat_cell
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        speaker_embed = kwargs.pop('speaker_embed', None)
        return self._cell(
            inputs, state,
            speaker_embed=speaker_embed, training=training
        )


class SelfAttentionCBHG(KL.Layer):

    def __init__(
        self,
        out_units,
        conv_channels,
        max_filter_width,
        projection1_out_channels,
        projection2_out_channels,
        num_highway, self_attention_out_units,
        self_attention_num_heads,
        input_lengths=None,
        zoneout_factor_cell=0.0, zoneout_factor_output=0.0, self_attention_drop_rate=0.0,
        lstm_impl=None,
        name=None, dtype=None, **kwargs
    ):
        half_out_units = out_units // 2
        assert out_units % 2 == 0
        assert num_highway == 4
        super(SelfAttentionCBHG, self).__init__(name=name, dtype=dtype, **kwargs)

        self.out_units = out_units
        self._zoneout_factor_cell = zoneout_factor_cell
        self._zoneout_factor_output = zoneout_factor_output
        self._self_attention_out_units = self_attention_out_units
        self._lstm_impl = lstm_impl

        self.convolution_banks = [
            Conv1d(kernel_size,
                   conv_channels,
                   activation=tf.nn.relu,
                   name=f"conv1d_K{kernel_size}",
                   dtype=dtype)
            for kernel_size in range(1, max_filter_width + 1)]
        self.maxpool = KL.MaxPooling1D(pool_size=2, strides=1, padding="SAME", dtype=dtype)

        self.projection1 = Conv1d(
            kernel_size=3,
            out_channels=projection1_out_channels,
            activation=tf.nn.relu,
            name="proj1",
            dtype=dtype
        )

        self.projection2 = Conv1d(
            kernel_size=3,
            out_channels=projection2_out_channels,
            activation=tf.identity,
            name="proj2",
            dtype=dtype
        )

        self.adjustment_layer = KL.Dense(half_out_units, dtype=dtype)

        self.highway_nets = [HighwayNet(half_out_units, dtype=dtype, )
                             for i in range(1, num_highway + 1)]

        self.self_attention_adjustment_layer = KL.Dense(self_attention_out_units, dtype=dtype)

        self.self_attention_highway_nets = [HighwayNet(self_attention_out_units, dtype=dtype)
                                            for i in range(1, num_highway + 1)]

        self.self_attention = SelfAttention(self_attention_out_units,
                                            self_attention_num_heads,
                                            self_attention_drop_rate,
                                            memory_sequence_length=input_lengths,
                                            dtype=dtype)
        self.bidirectional_dynamic_rnn = KL.Bidirectional(
            ZoneoutLSTMCell(self.out_units // 2,
                            zoneout_factor_cell=self._zoneout_factor_cell,
                            zoneout_factor_output=self._zoneout_factor_output,
                            dtype=self.dtype),
            ZoneoutLSTMCell(self.out_units // 2,
                            zoneout_factor_cell=self._zoneout_factor_cell,
                            zoneout_factor_output=self._zoneout_factor_output,
                            dtype=self.dtype),
            input_shape=(),  # TODO: Fix input shape and highway output
            dtype=None
        )

    def build(self, _):
        self.built = True

    def call(self, inputs, positional_encoding=None, training=None, **kwargs):
        conv_outputs = tf.concat([conv1d(inputs) for conv1d in self.convolution_banks], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)

        proj1_output = self.projection1(maxpool_output)
        proj2_output = self.projection2(proj1_output)

        # residual connection
        highway_input = proj2_output + inputs

        if highway_input.shape[2] != self.out_units // 2:
            highway_input = self.adjustment_layer(highway_input)

        highway_output = reduce(lambda acc, hw: hw(acc), self.highway_nets, highway_input)

        self_attention_highway_input = self.self_attention_adjustment_layer(highway_input)

        self_attention_highway_output = reduce(lambda acc, hw: hw(acc), self.self_attention_highway_nets,
                                               self_attention_highway_input)

        self_attention_input = self_attention_highway_output + positional_encoding

        self_attention_output, self_attention_alignments = self.self_attention(self_attention_input)
        self_attention_output = self_attention_output + self_attention_highway_output

        bilstm_outputs, bilstm_states = self.bidirectional_dynamic_rnn(highway_output)

        bilstm_outputs = tf.concat(bilstm_outputs, axis=-1)
        return bilstm_outputs, self_attention_output, self_attention_alignments

    def compute_output_shape(self, input_shape):
        return (tf.TensorShape([input_shape[0], input_shape[1], self.out_units]),
                tf.TensorShape([input_shape[0], input_shape[1], self.self._self_attention_out_units]))


class DualSourceMgcLf0AttentionRNN(KL.SimpleRNNCell):

    def __init__(
        self,
        cell,
        mgc_prenets: Tuple[PreNet],
        lf0_prenets: Tuple[PreNet],
        attention_mechanism: List[tfa.seq2seq.AttentionMechanism],
        name=None, **kwargs
    ):
        super(DualSourceMgcLf0AttentionRNN, self).__init__(name=name, **kwargs)
        attention_cell = tfa.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            alignment_history=True,
            output_attention=False)
        prenet_cell = DecoderMgcLf0PreNetWrapper(attention_cell, mgc_prenets, lf0_prenets)
        self._cell = ConcatOutputAndAttentionWrapper(prenet_cell)
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        return self._cell(inputs, state, training=training, **kwargs)


class DecoderPreNetWrapper(KL.SimpleRNNCell):

    def __init__(self, cell: KL.SimpleRNNCell, prenets: Tuple[PreNet]):
        super(DecoderPreNetWrapper, self).__init__()
        self._cell = cell
        self.prenets = prenets
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        speaker_embed = kwargs.pop('speaker_embed', None)
        prenet_output = reduce(lambda acc, pn: pn(acc, speaker_embed), self.prenets, inputs)
        return self._cell(prenet_output, state, training=training)


class OutputAndStopTokenWrapper(KL.SimpleRNNCell):

    def __init__(
        self,
        cell,
        out_units,
        name=None, dtype=None, **kwargs
    ):
        super(OutputAndStopTokenWrapper, self).__init__(name=name, dtype=dtype, **kwargs)
        self._out_units = out_units
        self._cell = cell
        self.out_projection = tf.layers.Dense(out_units, dtype=dtype)
        self.stop_token_projectioon = tf.layers.Dense(1, dtype=dtype)

        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state, training=None, **kwargs):
        output, res_state = self._cell(inputs, state, training=training, **kwargs)
        mel_output = self.out_projection(output)
        stop_token = self.stop_token_projectioon(output)
        return (mel_output, stop_token), res_state


class ScaledDotProductAttentionMechanism(tfa.seq2seq.AttentionMechanism):
    def __init__(
        self,
        keys, values, num_heads,
        drop_rate=0.0,
        use_padding_mask=True,
        use_subsequent_mask=False
    ):
        self.keys = keys  # (B, num_heads, T, C)
        self.values = values
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.use_padding_mask = use_padding_mask
        self.use_subsequent_mask = use_subsequent_mask
        self.state_size = self.alignments_size
        self.alignments_size = [
            self.num_heads, self.keys.shape[-2].value
        ] or tf.shape(self.keys)[1:3]

    def initial_alignment(self, batch_size, dtype):
        key_length = self.alignments_size
        size = tf.stack([batch_size, tf.ones(shape=(), dtype=tf.int32), key_length], axis=0)
        return tf.zeros(size, dtype=dtype)

    def call(self, query, memory_sequence_length=None, training=False, **kwargs):
        # Q K^\top
        x = tf.matmul(query, self.keys, transpose_b=True)

        # scale attention output
        s = tf.cast(tf.shape(query)[-1], dtype=query.dtype)
        x = x / tf.sqrt(s)

        x = self.apply_padding_mask(x, memory_sequence_length) if self.use_padding_mask else x
        x = self.apply_subsequent_mask(x) if self.use_subsequent_mask else x

        # softmax over last dim
        # (B, num_heads, T_query, T_memory)
        x = tf.nn.softmax(x, axis=-1)
        alignment_scores = x

        x = tf.keras.layers.Dropout(x, rate=self.drop_rate, training=self.is_training)

        x = tf.matmul(x, self.values)

        return x, alignment_scores

    def apply_padding_mask(self, score, memory_sequence_length, score_mask_value=-tf.constant(np.inf)):
        max_length = tf.shape(self.keys)[2]
        score_mask = tf.sequence_mask(memory_sequence_length, maxlen=max_length)
        # (B, T) -> (B, 1, T)
        score_mask = tf.expand_dims(score_mask, axis=1)
        # (B, 1, T) -> (B, T, T)
        score_mask = score_mask & tf.transpose(score_mask, perm=[0, 2, 1])
        # (B, 1, T, T) -> (B, num_heads, T, T)
        score_mask = tf.stack([score_mask] * self.num_heads, axis=1)
        score_mask_values = score_mask_value * tf.ones_like(score)
        return tf.where(score_mask, score, score_mask_values)

    def apply_subsequent_mask(self, score, score_mask_value=-np.inf):
        batch_size = tf.shape(self.keys)[0]
        max_length = tf.shape(self.keys)[2]
        score_mask = tf.ones([batch_size, self.num_heads, 1, 1], dtype=tf.bool) & tf.matrix_band_part(
            tf.ones([max_length, max_length], dtype=tf.bool), -1, 0)
        score_mask_values = score_mask_value * tf.ones_like(score)
        return tf.where(score_mask, score, score_mask_values)


class TeacherForcingForwardAttention(tfa.seq2seq.BahdanauAttention):

    def __init__(
        self,
        num_units,
        teacher_alignments,
        name="ForwardAttention"
    ):
        super(TeacherForcingForwardAttention, self).__init__(
            num_units=num_units,
            probability_fn=None,
            name=name)
        self.teacher_alignments = teacher_alignments
        self.state_size = self._alignments_size, 1

    def call(
        self, query, state,
        memory=None, memory_sequence_length=None, **kwargs
    ):
        previous_alignments, prev_index = state

        index = prev_index + 1
        alignments = self.teacher_alignments[:, index]
        next_state = (alignments, index)
        return alignments, next_state

    def initial_state(self, batch_size, dtype):
        initial_alignments = self.initial_alignments(batch_size, dtype)
        initial_index = tf.to_int64(-1)
        return initial_alignments, initial_index


class TeacherForcingAdditiveAttention(tfa.seq2seq.BahdanauAttention):

    def __init__(
        self,
        num_units,
        teacher_alignments,
        name="BahdanauAttention"
    ):
        super(TeacherForcingAdditiveAttention, self).__init__(
            num_units=num_units,
            probability_fn=None,
            name=name)
        self.teacher_alignments = teacher_alignments
        self.state_size = self._alignments_size, 1

    def call(
        self,
        query, state,
        memory=None, memory_sequence_length=None, **kwargs
    ):
        previous_alignments, prev_index = state

        index = prev_index + 1
        alignments = self.teacher_alignments[:, index]
        next_state = (alignments, index)
        return alignments, next_state

    def initial_state(self, batch_size, dtype):
        initial_alignments = self.initial_alignments(batch_size, dtype)
        initial_index = tf.to_int64(-1)
        return initial_alignments, initial_index


class LocationSensitiveAttention(tfa.seq2seq.BahdanauAttention):

    def __init__(
        self,
        units,
        attention_kernel,
        attention_filters,
        smoothing=False,
        cumulative_weights=True,
        dtype=None,
        name="LocationSensitiveAttention"
    ):
        probability_fn = self._smoothing_normalization if smoothing else None

        super(LocationSensitiveAttention, self).__init__(
            units=units,
            probability_fn=probability_fn,
            dtype=dtype,
            name=name
        )
        self._cumulative_weights = cumulative_weights
        self._dtype = dtype or K.floatx()

        self.location_convolution = tf.layers.Conv1D(
            filters=attention_filters,
            kernel_size=attention_kernel,
            padding="SAME", use_bias=True,
            bias_initializer=tf.zeros_initializer(dtype=memory.dtype),
            name="location_features_convolution", dtype=dtype
        )

        self.location_layer = tf.layers.Dense(
            units=units,
            use_bias=False, name="location_features_layer", dtype=dtype
        )

    def call(
        self,
        query, state,
        memory=None, memory_sequence_length=None, **kwargs
    ):
        previous_alignments = state
        processed_query = self.query_layer(query) if self.query_layer else query

        # -> [batch_size, 1, attention_dim]
        processed_query = tf.expand_dims(processed_query, 1)

        # [batch_size, max_time] -> [batch_size, max_time, 1]
        expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
        # location features [batch_size, max_time, filters]
        f = self.location_convolution(expanded_alignments)
        processed_location_features = self.location_layer(f)

        energy = _location_sensitive_score(processed_query, processed_location_features, self.keys, dtype=self._dtype)

        alignments = self._probability_fn(energy, state)
        if self._cumulative_weights:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments
        return alignments, next_state

    def _smoothing_normalization(e):
        return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keep_dims=True)


class ForwardAttention(tfa.seq2seq.BahdanauAttention):

    def __init__(
        self,
        num_units,
        attention_kernel,
        attention_filters,
        use_transition_agent=False, cumulative_weights=True,
        name="ForwardAttention", dtype=None
    ):
        super(ForwardAttention, self).__init__(
            num_units=num_units,
            probability_fn=None,
            name=name)
        self._use_transition_agent = use_transition_agent
        self._cumulative_weights = cumulative_weights
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=attention_filters,
            kernel_size=attention_kernel,
            padding="SAME",
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            name="location_features_convolution"
        )

        self.location_layer = tf.keras.layers.Dense(
            units=num_units, use_bias=False,
            dtype=dtype, name="location_features_layer"
        )

        if use_transition_agent:
            # ToDo: support speed control bias
            self.transition_factor_projection = tf.keras.layers.Dense(
                units=1,
                use_bias=True,
                dtype=dtype,
                activation=tf.nn.sigmoid,
                name="transition_factor_projection"
            )
        self.state_size = ForwardAttentionState(self._alignments_size, self._alignments_size, 1)

    def call(
        self,
        query, state,
        memory=None, memory_sequence_length=None, **kwargs
    ):
        previous_alignments, prev_alpha, prev_u = state

        # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
        processed_query = self.query_layer(query) if self.query_layer else query

        # -> [batch_size, 1, attention_dim]
        expanded_processed_query = tf.expand_dims(processed_query, 1)

        # [batch_size, max_time] -> [batch_size, max_time, 1]
        expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
        # location features [batch_size, max_time, filters]
        f = self.location_convolution(expanded_alignments)
        processed_location_features = self.location_layer(f)

        energy = _location_sensitive_score(expanded_processed_query, processed_location_features, self.keys)

        alignments = self.probability_fn(energy, state)

        # forward attention
        prev_alpha_n_minus_1 = tf.pad(prev_alpha[:, :-1], paddings=[[0, 0], [1, 0]])
        alpha = ((1 - prev_u) * prev_alpha + prev_u * prev_alpha_n_minus_1 + 1e-7) * alignments
        alpha_normalized = alpha / tf.reduce_sum(alpha, axis=1, keep_dims=True)
        if self._use_transition_agent:
            context = _calculate_context(alpha_normalized, self.values)
            transition_factor_input = tf.concat([context, processed_query], axis=-1)
            transition_factor = self.transition_factor_projection(transition_factor_input)
        else:
            transition_factor = prev_u

        if self._cumulative_weights:
            next_state = ForwardAttentionState(alignments + previous_alignments, alpha_normalized, transition_factor)
        else:
            next_state = ForwardAttentionState(alignments, alpha_normalized, transition_factor)
        return alpha_normalized, next_state

    def initial_state(self, batch_size, dtype):
        initial_alignments = self.initial_alignments(batch_size, dtype)
        # alpha_0 = 1, alpha_n = 0 where n = 2, 3, ..., N
        initial_alpha = tf.concat([
            tf.ones([batch_size, 1], dtype=dtype),
            tf.zeros_like(initial_alignments, dtype=dtype)[:, 1:]], axis=1)
        # transition factor
        initial_u = 0.5 * tf.ones([batch_size, 1], dtype=dtype)
        return ForwardAttentionState(initial_alignments, initial_alpha, initial_u)


class Projection(KL.Layer):
    # TODO: Fix this class w.r.t TF2
    def __init__(self, in_units, out_units, dtype=None, name="projection"):
        super().__init__(dtype=dtype)
        self.in_units = in_units
        self.out_units = out_units
        self.dtype = dtype

    def build(self):
        self.kernel = tf.Variable(
            'kernel',
            shape=[self.in_units, self.out_units],
            dtype=self.dtype
        )

        self.bias = tf.Variable(
            'bias',
            shape=[self.out_units, ],
            dtype=self.dtype,
            initializer=tf.zeros_initializer(dtype=self.dtype)
        )

    def call(self, inputs, **kwargs):
        shape = inputs.get_shape().as_list()
        matmul = tf.tensordot(inputs, self.kernel, axes=[[len(shape) - 1], [0]])
        output = tf.nn.bias_add(matmul, self.bias)
        return output


class MGCProjection(KL.Layer):
    # TODO: Fix this class w.r.t TF2
    def __init__(self, in_units, out_units, dtype=None, name="mgc_projection"):
        with tf.variable_scope(name):
            self.dense_kernel1 = tf.get_variable(
                'dense_kernel1',
                shape=[in_units, in_units],
                dtype=dtype
            )

            self.dense_bias1 = tf.get_variable(
                'dense_bias1',
                shape=[in_units, ],
                dtype=dtype,
                initializer=tf.zeros_initializer(dtype=dtype)
            )

            self.dense_kernel2 = tf.get_variable(
                'dense_kernel2',
                shape=[in_units, out_units],
                dtype=dtype
            )

            self.dense_bias2 = tf.get_variable(
                'dense_bias2',
                shape=[out_units, ],
                dtype=dtype,
                initializer=tf.zeros_initializer(dtype=dtype)
            )

    def call(self, inputs, **kwargs):
        shape = inputs.get_shape().as_list()
        matmul1 = tf.tensordot(inputs, self.dense_kernel1, axes=[[len(shape) - 1], [0]])
        dense_output1 = tf.nn.bias_add(matmul1, self.dense_bias1)
        dense_output1 = tf.nn.tanh(dense_output1)
        matmul2 = tf.tensordot(dense_output1, self.dense_kernel2, axes=[[len(shape) - 1], [0]])
        dense_output2 = tf.nn.bias_add(matmul2, self.dense_bias2)
        return dense_output2


class RNNTransformer(KL.Layer):

    def __init__(
            self,
            decoder_cell,
            self_attention_out_units,
            self_attention_transformer_num_conv_layers,
            self_attention_transformer_kernel_size,
            self_attention_num_heads,
            self_attention_num_hop,
            self_attention_drop_rate,
            num_mels,
            outputs_per_step,
            n_feed_frame,
            max_iters,
            dtype=None,
    ):
        self.decoder_cell = decoder_cell
        self.self_attention_transformer_num_conv_layers = self_attention_transformer_num_conv_layers
        self.self_attention_transformer_kernel_size = self_attention_transformer_kernel_size
        self._out_units = num_mels * outputs_per_step
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.n_feed_frame = n_feed_frame
        self.max_iters = max_iters
        self._dtype = dtype

        self.transformers = [
            SelfAttentionTransformer(
                out_units=self_attention_out_units,
                self_attention_out_units=self_attention_out_units,
                self_attention_num_heads=self_attention_num_heads,
                self_attention_drop_rate=self_attention_drop_rate,
                use_subsequent_mask=True,
                dtype=dtype
            )
            for i in range(1, self_attention_num_hop + 1)
        ]

        # at inference time, outputs are evaluated within dynamic_decode (dynamic_decode has "decoder" scope)
        self.out_projection = Projection(
            self_attention_out_units, self._out_units,
            dtype=self._dtype,
            name="out_projection"
        )
        self.stop_token_projection = Projection(
            self_attention_out_units, 1,
            dtype=self._dtype,
            name="stop_token_projection"
        )

        self.transformer_cell = TransformerWrapper(
            RNNStateHistoryWrapper(self.decoder_cell, self.max_iters), self.transformers
        )
        self.output_and_done_cell = OutputAndStopTokenTransparentWrapper(
            self.transformer_cell, self._out_units,
            self.out_projection,
            self.stop_token_projection
        )

    def call(
        self,
        target,
        training=False,
        teacher_forcing=False,
        batch_size=32,
        memory_sequence_length=None,
        _dtype=None, **kwargs

    ):
        decoder_initial_state = self.output_and_done_cell.get_initial_state(
            batch_size=batch_size, dtype=target.dtype
        )
        params = {"memory_sequence_length": memory_sequence_length}

        if training:
            helper = TransformerTrainingHelper(
                target,
                self.num_mels,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                teacher_forcing=teacher_forcing
            )
            (decoder_outputs, _), final_decoder_state, _ = dynamic_decode(
                BasicDecoder(self.decoder_cell, helper, decoder_initial_state),
                maximum_iterations=self.max_iters,
                scope=self._decoder_scope, **params
            )

            def self_attend(input, alignments, layer):
                output, alignment = layer(input, memory_sequence_length=memory_sequence_length)
                return output, alignments + alignment

            # at inference time, transformers are evaluated within dynamic_decode (dynamic_decode has "decoder" scope)
            transformed, alignments = reduce(
                lambda acc, sa: self_attend(acc[0], acc[1], sa),
                self.transformers,
                (decoder_outputs, [])
            )
            mel_output = self.out_projection(transformed)
            stop_token = self.stop_token_projection(transformed)
            return mel_output, stop_token, final_decoder_state

        else:
            helper = StopTokenBasedInferenceHelper(
                batch_size,
                self.num_mels,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                dtype=_dtype
            )

            basic_decoder = BasicDecoder(self.output_and_done_cell, helper, decoder_initial_state)
            ((decoder_outputs, stop_token), _), final_decoder_state, _ = dynamic_decode(
                basic_decoder,
                scope=self._decoder_scope,
                maximum_iterations=self.max_iters,
                parallel_iterations=10,
                swap_memory=True, **params
            )  # Huge memory consumption at inference time
            return decoder_outputs, stop_token, final_decoder_state


class MgcLf0RNNTransformer(KL.Layer):
    def __init__(
        self,
        decoder_cell,
        decoder_initial_state,
        self_attention_out_units,
        self_attention_transformer_num_conv_layers,
        self_attention_transformer_kernel_size,
        self_attention_num_heads,
        self_attention_num_hop,
        self_attention_drop_rate,
        num_mgcs,
        num_lf0s,
        outputs_per_step,
        n_feed_frame,
        max_iters,
        batch_size,
        dtype,
    ) -> None:
        self.self_attention_transformer_num_conv_layers = self_attention_transformer_num_conv_layers
        self.self_attention_transformer_kernel_size = self_attention_transformer_kernel_size
        self.decoder_cell = decoder_cell
        self.decoder_initial_state = decoder_initial_state
        self._mgc_out_units = num_mgcs * outputs_per_step
        self._lf0_out_units = num_lf0s * outputs_per_step
        self.num_mgcs = num_mgcs
        self.num_lf0s = num_lf0s
        self.outputs_per_step = outputs_per_step
        self.n_feed_frame = n_feed_frame
        self.max_iters = max_iters
        self._batch_size = batch_size
        self._dtype = dtype

        self.transformers = [
            SelfAttentionTransformer(
                out_units=self_attention_out_units,
                self_attention_out_units=self_attention_out_units,
                self_attention_num_heads=self_attention_num_heads,
                self_attention_drop_rate=self_attention_drop_rate,
                use_subsequent_mask=True,
                dtype=dtype
            ) for i in range(1, self_attention_num_hop + 1)
        ]

        # at inference time, outputs are evaluated within dynamic_decode
        # (dynamic_decode has "decoder" scope)
        self.mgc_out_projection = MGCProjection(
            self_attention_out_units, self._mgc_out_units,
            dtype=self._dtype,
            name="mgc_out_projection"
        )
        self.lf0_out_projection = Projection(
            self_attention_out_units, self._lf0_out_units,
            dtype=self._dtype,
            name="lf0_out_projection"
        )
        self.stop_token_projection = Projection(
            self_attention_out_units, 1,
            dtype=self._dtype,
            name="stop_token_projection"
        )
        transformer_cell = TransformerWrapper(
            RNNStateHistoryWrapper(self.decoder_cell, self.max_iters), self.transformers
        )
        self.output_and_done_cell = OutputMgcLf0AndStopTokenTransparentWrapper(
            transformer_cell,
            self._mgc_out_units,
            self._lf0_out_units,
            self.mgc_out_projection,
            self.lf0_out_projection,
            self.stop_token_projection
        )

    def call(
        self,
        target,
        training=False,
        teacher_forcing=False,
        batch_size=32,
        memory_sequence_length=None,
        _dtype=None, **kwargs
    ):
        mgc_targets, lf0_targets = target
        decoder_initial_state = self.output_and_done_cell.get_initial_state(
            batch_size=batch_size, dtype=target.dtype
        )
        params = {"memory_sequence_length": memory_sequence_length}

        if training:
            helper = TrainingMgcLf0Helper(
                mgc_targets,
                lf0_targets,
                self.num_mgcs,
                self.num_lf0s,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                teacher_forcing=teacher_forcing
            )
            (decoder_outputs, _), final_decoder_state, _ = dynamic_decode(
                BasicDecoder(self.decoder_cell, helper, self.decoder_initial_state),
                maximum_iterations=self.max_iters,
                scope=self._decoder_scope, **params
            )

            def self_attend(input, alignments, layer):
                output, alignment = layer(input, memory_sequence_length=memory_sequence_length)
                return output, alignments + alignment

            # at inference time, transformers are evaluated within dynamic_decode (dynamic_decode has "decoder" scope)
            transformed, alignments = reduce(
                lambda acc, sa: self_attend(acc[0], acc[1], sa),
                self.transformers,
                (decoder_outputs, [])
            )
            mgc_output = self.mgc_out_projection(transformed)
            lf0_output = self.lf0_out_projection(transformed)
            stop_token = self.stop_token_projection(transformed)
            return mgc_output, lf0_output, stop_token, final_decoder_state

        else:
            helper = StopTokenBasedMgcLf0InferenceHelper(
                self._batch_size,
                self.num_mgcs,
                self.num_lf0s,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                dtype=self._output_dtype
            )

            ((decoder_mgc_outputs, decoder_lf0_outputs, stop_token),
             _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(self.output_and_done_cell, helper, decoder_initial_state),
                scope=self._decoder_scope,
                maximum_iterations=self.max_iters,
                parallel_iterations=10,
                swap_memory=True, **params
            )  # Huge memory consumption at inference time

            return decoder_mgc_outputs, decoder_lf0_outputs, stop_token, final_decoder_state
