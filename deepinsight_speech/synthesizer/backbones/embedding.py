import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.python.keras import backend
from functools import reduce
# from tacotron.rnn_impl import lstm_cell_factory, LSTMImpl
from collections import namedtuple


class LSTMStateTuple(namedtuple("LSTMStateTuple", ("c", "h"))):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

    Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
    and `h` is the output.

    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


class Embedding(KL.Layer):

    def __init__(
        self,
        num_symbols,
        embedding_dim,
        index_offset=0,
        output_dtype=None,
        name=None, dtype=None, **kwargs
    ) -> None:
        self._dtype = dtype or backend.floatx()
        # To ensure self.dtype is float type, set dtype explicitly.
        super(Embedding, self).__init__(name=name, dtype=self._dtype, **kwargs)
        self._num_symbols = num_symbols
        self._embedding_dim = embedding_dim
        self._output_dtype = output_dtype or backend.floatx()
        self.index_offset = tf.convert_to_tensor(index_offset, dtype=tf.int64)

    def build(self, _):
        self._embedding = tf.cast(
            self.add_weight("embedding", shape=[self._num_symbols, self._embedding_dim], dtype=self._dtype),
            dtype=self._output_dtype
        )
        self.built = True

    def call(self, inputs, **kwargs):
        with tf.control_dependencies(
            [
                tf.assert_greater_equal(inputs, self.index_offset),
                tf.assert_less(inputs, self.index_offset + self._num_symbols)
            ]
        ):
            return tf.nn.embedding_lookup(self._embedding, inputs - self.index_offset)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self._embedding_dim])


class PreNet(KL.Layer):

    def __init__(
        self,
        out_units,
        drop_rate=0.5,
        apply_dropout_on_inference=False,
        name=None, dtype=None, **kwargs
    ) -> None:
        super(PreNet, self).__init__(name=name, dtype=dtype, **kwargs)
        self.out_units = out_units
        self.drop_rate = drop_rate
        self.apply_dropout_on_inference = apply_dropout_on_inference
        self.dense = KL.Dense(out_units, activation=tf.nn.relu, dtype=dtype)
        self.dropout = KL.Dropout(rate=self.drop_rate, )

    def build(self, _):
        self.built = True

    def call(self, inputs, training=False, **kwargs):
        output = self.dense(inputs)
        output = self.dropout(output, training=training)
        return output

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


class HighwayNet(KL.Layer):

    def __init__(
            self,
            out_units,
            h_kernel_initializer=None,
            h_bias_initializer=None,
            t_kernel_initializer=None,
            t_bias_initializer=tf.constant_initializer(-1.0),
            name=None, dtype=None, **kwargs
    ):
        super(HighwayNet, self).__init__(name=name, dtype=dtype, **kwargs)
        self.out_units = out_units
        self.H = KL.Dense(
            out_units, activation=tf.nn.relu, name="H",
            kernel_initializer=h_kernel_initializer,
            bias_initializer=h_bias_initializer,
            dtype=dtype
        )
        self.T = KL.Dense(
            out_units, activation=tf.nn.sigmoid, name="T",
            kernel_initializer=t_kernel_initializer,
            bias_initializer=t_bias_initializer,
            dtype=dtype
        )

    def build(self, input_shape):
        with tf.control_dependencies([tf.assert_equal(self.out_units, input_shape[-1])]):
            self.built = True

    def call(self, inputs, **kwargs):
        h = self.H(inputs)
        t = self.T(inputs)
        return h * t + inputs * (1.0 - t)

    def compute_output_shape(self, input_shape):
        return input_shape


class Conv1d(KL.Layer):

    def __init__(
            self,
            kernel_size, out_channels, activation,
            use_bias=False,
            drop_rate=0.0,
            name=None, dtype=None, **kwargs
    ):
        super(Conv1d, self).__init__(name=name, dtype=dtype, **kwargs)
        self.activation = activation
        self.drop_rate = drop_rate
        self.batch_normalization = KL.BatchNormalization()
        self.dropout = KL.Dropout(self.drop_rate)
        self.conv1d = KL.Conv1D(
            out_channels, kernel_size,
            use_bias=use_bias, activation=None,
            padding="SAME", dtype=dtype
        )

    def build(self, _):
        self.built = True

    def call(self, inputs, training=False, **kwargs):
        conv1d = self.conv1d(inputs)
        # fused_batch_norm (and 16bit precision) is only supported for 4D tensor
        conv1d_rank4 = tf.expand_dims(conv1d, axis=2)
        batch_normalization_rank4 = self.batch_normalization(conv1d_rank4, training=training)
        batch_normalization = tf.squeeze(batch_normalization_rank4, axis=2)
        output = self.activation(batch_normalization) if self.activation is not None else batch_normalization
        output = self.dropout(output, training=training)
        return output

    def compute_output_shape(self, input_shape):
        return self.conv1d.compute_output_shape(input_shape)


class CBHG(KL.Layer):

    def __init__(
        self,
        out_units,
        conv_channels,
        max_filter_width,
        projection1_out_channels,
        projection2_out_channels,
        num_highway, name=None,
        dtype=None, input_lengths=None, **kwargs
    ):
        half_out_units = out_units // 2
        assert out_units % 2 == 0
        super(CBHG, self).__init__(name=name, dtype=dtype, **kwargs)

        self.out_units = out_units

        self.convolution_banks = [
            Conv1d(kernel_size,
                   conv_channels,
                   activation=tf.nn.relu,
                   name=f"conv1d_K{kernel_size}",
                   dtype=dtype)
            for kernel_size in range(1, max_filter_width + 1)]
        self.maxpool = KL.MaxPooling1D(pool_size=2, strides=1, padding="SAME", dtype=dtype)

        self.projection1 = Conv1d(kernel_size=3,
                                  out_channels=projection1_out_channels,
                                  activation=tf.nn.relu,
                                  name="proj1",
                                  dtype=dtype)

        self.projection2 = Conv1d(kernel_size=3,
                                  out_channels=projection2_out_channels,
                                  activation=tf.identity,
                                  name="proj2",
                                  dtype=dtype)
        self.adjustment_layer = KL.Dense(half_out_units, dtype=dtype)

        self.highway_nets = [HighwayNet(half_out_units, dtype=dtype) for i in range(1, num_highway + 1)]
        self.bidirectional_dynamic_rnn = KL.Bidirectional(
            KL.GRUCell(self.out_units // 2, dtype=self.dtype),
            backward_layer=KL.GRUCell(self.out_units // 2, dtype=self.dtype),
            weights=self.highway_nets,
            input_shape=input_lengths
        )

    def build(self, _):
        self.built = True

    def call(self, inputs, training=False, **kwargs):
        conv_outputs = tf.concat([conv1d(inputs, training=training) for conv1d in self.convolution_banks], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)

        proj1_output = self.projection1(maxpool_output)
        proj2_output = self.projection2(proj1_output)

        # residual connection
        highway_input = proj2_output + inputs

        if highway_input.shape[2] != self.out_units // 2:
            highway_input = self.adjustment_layer(highway_input)

        highway_output = reduce(lambda acc, hw: hw(acc), self.highway_nets, highway_input)

        # ToDo: use factory from rnn_impl once rnn_impl support bidirectional RNN
        outputs, states = self.bidirectional_dynamic_rnn(highway_output)

        return tf.concat(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.out_units])


class ZoneoutCBHG(KL.Layer):

    def __init__(
            self,
            out_units,
            conv_channels,
            max_filter_width,
            projection1_out_channels,
            projection2_out_channels,
            num_highway,
            zoneout_factor_cell=0.0,
            zoneout_factor_output=0.0,
            input_lengths=None,
            name=None,
            dtype=None, **kwargs
    ):
        half_out_units = out_units // 2
        assert out_units % 2 == 0
        super(ZoneoutCBHG, self).__init__(name=name, dtype=dtype, **kwargs)

        self.out_units = out_units
        self._zoneout_factor_cell = zoneout_factor_cell
        self._zoneout_factor_output = zoneout_factor_output

        self.convolution_banks = [
            Conv1d(kernel_size,
                   conv_channels,
                   activation=tf.nn.relu,
                   name=f"conv1d_K{kernel_size}",
                   dtype=dtype
                   )
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
        self.highway_nets = [HighwayNet(half_out_units, dtype=dtype) for i in range(1, num_highway + 1)]

        self.bidirectional_dynamic_rnn = KL.Bidirectional(
            ZoneoutLSTMCell(
                self.out_units // 2,
                zoneout_factor_cell=self._zoneout_factor_cell,
                zoneout_factor_output=self._zoneout_factor_output,
                dtype=self.dtype
            ),
            backward_layer=ZoneoutLSTMCell(
                self.out_units // 2,
                zoneout_factor_cell=self._zoneout_factor_cell,
                zoneout_factor_output=self._zoneout_factor_output,
                dtype=self.dtype,
            ),
            weights=self.highway_nets,
            input_shape=input_lengths
        )

    def build(self, _):
        self.built = True

    def call(self, inputs, training=False, **kwargs):
        conv_outputs = tf.concat([conv1d(inputs) for conv1d in self.convolution_banks], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)

        proj1_output = self.projection1(maxpool_output)
        proj2_output = self.projection2(proj1_output)

        # residual connection
        highway_input = proj2_output + inputs

        if highway_input.shape[2] != self.out_units // 2:
            highway_input = self.adjustment_layer(highway_input)

        highway_output = reduce(lambda acc, hw: hw(acc), self.highway_nets, highway_input)

        outputs, states = self.bidirectional_dynamic_rnn(highway_output,)

        return tf.concat(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.out_units])


class ZoneoutLSTMCell(KL.SimpleRNNCell):

    def __init__(
        self,
        num_units,
        zoneout_factor_cell=0.0,
        zoneout_factor_output=0.0,
        name=None, dtype=None, **kwargs
    ):
        super(ZoneoutLSTMCell, self).__init__(name=name, dtype=dtype, **kwargs)
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')
        # TODO: Check if LSTM block cell is required from config
        self._cell = KL.LSTMCell(num_units, dtype=dtype)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        # Apply zoneout
        self.keep_rate_cell = 1.0 - self._zoneout_cell
        self.keep_rate_output = 1.0 - self._zoneout_outputs
        self.dropout_c = KL.Dropout(rate=self.keep_rate_cell)
        self.dropout_h = KL.Dropout(rate=self.keep_rate_output)
        self.state_size = self._cell.state_size
        self.output_size = self._cell.output_size

    def call(self, inputs, state, scope=None, training=False, **kwargs):
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope, **kwargs)

        assert self.state_is_tuple, NotImplementedError("non-tuple state is not implemented")

        (prev_c, prev_h) = state
        (new_c, new_h) = new_state

        if training:
            c = self.keep_rate_cell * self.dropout(new_c - prev_c) + prev_c
            h = self.keep_rate_output * self.dropout(new_h - prev_h) + prev_h
        else:
            c = (1.0 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1.0 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

        # TODO: verify is LSTMStateTuple is needed
        new_state = tf.concat([c, h], axis=1)
        return output, new_state


class ChannelEncoderPostNet(KL.Layer):

    def __init__(
            self,
            out_units,
            num_postnet_layers,
            kernel_size,
            out_channels,
            drop_rate=0.5, name=None, dtype=None, **kwargs
    ):
        super(ChannelEncoderPostNet, self).__init__(name=name, dtype=dtype, **kwargs)

        final_conv_layer = Conv1d(
            kernel_size, out_channels, activation=None,
            drop_rate=drop_rate,
            name=f"conv1d_{num_postnet_layers}",
            dtype=dtype
        )

        self.convolutions = [
            Conv1d(kernel_size, out_channels, activation=tf.nn.tanh,
                   drop_rate=drop_rate,
                   name=f"conv1d_{i}",
                   dtype=dtype) for i in range(1, num_postnet_layers)
        ] + [final_conv_layer]

        self.projection_layer = KL.Dense(out_units, dtype=dtype)
        self.speaker_projection = KL.Dense(out_channels, activation=tf.nn.softsign, dtype=dtype)

    def call(self, channel_code, inputs, training=False, **kwargs):
        channel_code = tf.expand_dims(self.speaker_projection(channel_code), axis=1)
        output = reduce(lambda acc, conv: conv(acc, training=training) + channel_code, self.convolutions, inputs)
        projected = self.projection_layer(output)
        summed = inputs + projected
        return summed

    def compute_output_shape(self, input_shape):
        return self.projection_layer.compute_output_shape(input_shape)


class ExternalEmbedding(KL.Layer):
    def __init__(
        self,
        fname, num_symbols,
        embedding_dim,
        index_offset=0,
        name=None, **kwargs
    ):
        super(ExternalEmbedding, self).__init__(name=name, **kwargs)
        self._fname = fname
        self._num_symbols = num_symbols
        self._embedding_dim = embedding_dim
        self.index_offset = tf.convert_to_tensor(index_offset, dtype=tf.int64)

    def build(self, _):
        self._embedding = self.load_embedding_from_file(self._fname)

    def call(self, inputs, **kwargs):
        with tf.control_dependencies([tf.assert_greater_equal(inputs, self.index_offset),
                                      tf.assert_less(inputs, self.index_offset + self._num_symbols)]):
            return tf.nn.embedding_lookup(self._embedding, inputs - self.index_offset)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self._embedding_dim])

    def load_embedding_from_file(self, fname):
        spk_embs = {}
        min = 9999999999  # please do not have a speaker ID larger than this.
        max = -9999999999  # we do not assume that embeddings are listed in order.
        vecsize = 0

        f = open(fname, 'r')
        for line in f:
            parts = line.strip().split('  ')
            spkr = int(parts[0][1:])  # assuming vctk 0.91 speaker ID format for now.
            xvec = [float(x) for x in parts[1].strip('[]').strip().split(' ')]
            if vecsize == 0:
                vecsize = len(xvec)
            if spkr < min:
                min = spkr
            if spkr > max:
                max = spkr
            spk_embs[spkr] = xvec

        xv_table = []
        for i in range(min, max + 1):
            if i in spk_embs.keys():
                xv_table.append(spk_embs[i])
            else:
                empty = [0 for x in range(0, vecsize)]  # placeholder for skipped speaker IDs
                xv_table.append(empty)

        return tf.constant(xv_table)


class PostNetV2(KL.Layer):

    def __init__(
        self,
        out_units,
        num_postnet_layers,
        kernel_size,
        out_channels,
        drop_rate=0.5, name=None, dtype=None, **kwargs
    ):
        super(PostNetV2, self).__init__(name=name, **kwargs)

        final_conv_layer = Conv1d(kernel_size, out_channels,
                                  activation=None,
                                  drop_rate=drop_rate,
                                  name=f"conv1d_{num_postnet_layers}",
                                  dtype=dtype)

        self.convolutions = [
            Conv1d(kernel_size, out_channels,
                   activation=tf.nn.tanh,
                   drop_rate=drop_rate,
                   name=f"conv1d_{i}",
                   dtype=dtype) for i in range(1, num_postnet_layers)
        ] + [final_conv_layer]

        self.projection_layer = KL.Dense(out_units, dtype=dtype)

    def call(self, inputs, training=False, **kwargs):
        output = reduce(lambda acc, conv: conv(acc, training=training), self.convolutions, inputs)
        projected = self.projection_layer(output)
        summed = inputs + projected
        return summed


class MultiSpeakerPostNet(KL.Layer):

    def __init__(
        self,
        out_units,
        num_postnet_layers,
        kernel_size,
        out_channels,
        drop_rate=0.5,
        name=None, dtype=None, **kwargs
    ):
        super(MultiSpeakerPostNet, self).__init__(name=name, dtype=dtype, **kwargs)

        final_conv_layer = Conv1d(kernel_size, out_channels, activation=None,
                                  drop_rate=drop_rate,
                                  name=f"conv1d_{num_postnet_layers}",
                                  dtype=dtype)

        self.convolutions = [Conv1d(kernel_size, out_channels, activation=tf.nn.tanh,
                                    drop_rate=drop_rate,
                                    name=f"conv1d_{i}",
                                    dtype=dtype) for i in
                             range(1, num_postnet_layers)] + [final_conv_layer]

        self.projection_layer = KL.Dense(out_units, dtype=dtype)
        self.speaker_projection = KL.Dense(out_channels, activation=tf.nn.softsign, dtype=dtype)

    def call(self, inputs, speaker_embed=None, training=False, **kwargs):
        speaker_embed = tf.expand_dims(self.speaker_projection(speaker_embed), axis=1)

        output = reduce(lambda acc, conv: conv(acc, training=training) + speaker_embed, self.convolutions, inputs)
        projected = self.projection_layer(output)
        summed = inputs + projected
        return summed

    def compute_output_shape(self, input_shape):
        return self.projection_layer.compute_output_shape(input_shape)


class MultiSpeakerPreNet(KL.Layer):

    def __init__(
        self,
        out_units,
        drop_rate=0.5,
        name=None, dtype=None, **kwargs
    ):
        super(MultiSpeakerPreNet, self).__init__(name=name, dtype=dtype, **kwargs)
        self.out_units = out_units
        self.drop_rate = drop_rate
        self.dense0 = KL.Dense(out_units, activation=tf.nn.relu, dtype=dtype)
        self.dense = KL.Dense(out_units, activation=tf.nn.relu, dtype=dtype)
        self.speaker_projection = KL.Dense(out_units, activation=tf.nn.softsign, dtype=dtype)
        self.dropout = KL.Dropout(self.drop_rate)

    def build(self, _):
        self.built = True

    def call(self, inputs, speaker_embed=None, training=None, **kwargs):
        dense0 = self.dense0(inputs)
        dense0 += self.speaker_projection(speaker_embed)
        dense = self.dense(dense0)
        dropout = self.dropout(dense, training=training)
        return dropout

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)
