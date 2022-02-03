import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from typing import Tuple, List, Union, Type
from .embedding import PreNet, ZoneoutLSTMCell, MultiSpeakerPreNet
from .attention import (
    OutputMgcLf0AndStopTokenWrapper, DualSourceAttentionRNN,
    AttentionRNN, DecoderRNNV1, DecoderRNNV2,
    DualSourceMgcLf0AttentionRNN, MgcLf0AttentionRNN,
    OutputAndStopTokenWrapper, MgcLf0RNNTransformer, RNNTransformer
)
from ..modules.helpers import (
    TrainingHelper, TrainingMgcLf0Helper,
    StopTokenBasedInferenceHelper, StopTokenBasedMgcLf0InferenceHelper
)
import tensorflow.keras.layers as KL
from functools import reduce
import enum
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from easydict import EasyDict

_PreNet_Type = Union[Tuple[PreNet], List[Tuple[PreNet]]]


class DecoderOption(EasyDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DecoderFactory(enum.Enum):
    ExtendedDecoder = 'ExtendedDecoder'
    TransformerDecoder = 'TransformerDecoder'
    DualSourceDecoder = 'DualSourceDecoder'
    DualSourceTransformerDecoder = 'DualSourceTransformerDecoder'
    DualSourceMgcLf0TransformerDecoder = 'DualSourceMgcLf0TransformerDecoder'
    MgcLf0Decoder = 'MgcLf0Decoder'
    MgcLf0DualSourceDecoder = 'MgcLf0DualSourceDecoder'


@add_metaclass(ABCMeta)
class BaseDecoder(KL.Layer):
    def __init__(
        self,
        teacher_alignments=None,
        apply_dropout_on_inference=False,
        prenet_out_units=(256, 128),
        drop_rate=0.5,
        attention_out_units=256,
        decoder_version="v1",  # v1 | v2
        decoder_out_units=256,
        num_mels=80,
        outputs_per_step=2,
        max_iters=200,
        n_feed_frame=1,
        zoneout_factor_cell=0.0,
        zoneout_factor_output=0.0,
        teacher_forcing=False,
        self_attention_out_units=None,
        self_attention_num_heads=None,
        self_attention_num_hop=None,
        use_speaker_embed=False,
        name=None, dtype=None, **kwargs
    ) -> None:

        self.teacher_forcing = teacher_forcing
        self.teacher_alignments = teacher_alignments
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_out_units = attention_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self.dtype = dtype
        self.name = name
        self.kwargs = kwargs
        self.use_speaker_embed = use_speaker_embed
        self.apply_dropout_on_inference = apply_dropout_on_inference
        self.self_attention_out_units = self_attention_out_units
        self.self_attention_num_heads = self_attention_num_heads
        self.self_attention_num_hop = self_attention_num_hop

    @abstractmethod
    def _set_prenets(self) -> _PreNet_Type:
        """ Define prenets over here, return Tuple or single prenets """
        raise NotImplementedError

    @abstractmethod
    def call(self, sources, **kwargs):
        pass


class ExtendedDecoder(BaseDecoder):

    def __init__(self, attention_func: tfa.seq2seq.AttentionMechanism, **kwargs):
        super(ExtendedDecoder, self).__init__(**kwargs)
        self.prenets = self._set_prenets()
        self.attention_mechanism = attention_func
        lstmcell = ZoneoutLSTMCell(
            self.attention_out_units,
            zoneout_factor_cell=self.zoneout_factor_cell,
            zoneout_factor_output=self.zoneout_factor_output,
            dtype=self.dtype
        )
        self.attention_cell = AttentionRNN(
            lstmcell, self.prenets, self.attention_mechanism, dtype=self.dtype
        )
        if self.decoder_version == "v1":
            self.decoder_cell = DecoderRNNV1(
                self.decoder_out_units, self.attention_cell, dtype=self.dtype
            )
        else:
            self.decoder_cell = DecoderRNNV2(
                self.decoder_out_units,
                self.attention_cell,
                zoneout_factor_cell=self.zoneout_factor_cell,
                zoneout_factor_output=self.zoneout_factor_output,
                dtype=self.dtype
            )
        self.output_and_done_cell = OutputAndStopTokenWrapper(
            self.decoder_cell,
            self.num_mels * self.outputs_per_step,
            dtype=self.dtype
        )

    def _set_prenets(self):
        if self.use_speaker_embed is not None:
            prenets = (
                MultiSpeakerPreNet(
                    self._prenet_out_units[0],
                    self._drop_rate,
                    dtype=self.dtype,
                ),
                PreNet(
                    self._prenet_out_units[1],
                    self._drop_rate,
                    self.apply_dropout_on_inference,
                    dtype=self.dtype,
                )
            )
        else:
            prenets = tuple(
                [
                    PreNet(
                        out_unit, self._drop_rate,
                        self.apply_dropout_on_inference, dtype=self.dtype,
                    )
                    for out_unit in self._prenet_out_units
                ]
            )

        return prenets

    def call(
        self,
        source,
        speaker_embed=None,
        target=None,
        memory_sequence_length=None,
        target_sequence_length=None,
        training=None, **kwargs
    ):
        # TODO: Explore RNN network
        batch_size = tf.shape(source)[0]
        decoder_initial_state = self.output_and_done_cell.get_initial_state(
            inputs=source, batch_size=batch_size, dtype=source.dtype
        )
        # TODO: FixMe
        # cell_output = self.output_and_done_cell(
        #     source, decoder_initial_state, speaker_embed
        # )
        params = {
            "speaker_embed": speaker_embed,
            "memory_sequence_length": memory_sequence_length,
            "target_sequence_length": target_sequence_length
        }
        if training:
            helper = TrainingHelper(
                target,
                self.num_mels,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame
            )
        else:
            helper = StopTokenBasedInferenceHelper(
                batch_size,
                self.num_mels,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                dtype=source.dtype
            )
        decoder = tfa.seq2seq.BasicDecoder(self.output_and_done_cell, helper, decoder_initial_state)
        ((decoder_outputs, stop_token), _), final_decoder_state, _ = tfa.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=self.max_iters, **params
        )

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class MgcLf0DualSourceDecoder(BaseDecoder):

    def __init__(
        self,
        attention_mechanisms: List[tfa.seq2seq.AttentionMechanism],
        num_mgcs=60,
        num_lf0s=256,
        **kwargs
    ):
        super(MgcLf0DualSourceDecoder, self).__init__(**kwargs)
        self.num_mgcs = num_mgcs
        self.num_lf0s = num_lf0s
        mgc_prenets, lf0_prenets = self._set_prenets()

        self.attention_mechanisms = attention_mechanisms
        lstmcell = ZoneoutLSTMCell(self.attention_rnn_out_units,
                                   self.zoneout_factor_cell,
                                   self.zoneout_factor_output,
                                   dtype=self.dtype)
        self.attention_cell = DualSourceMgcLf0AttentionRNN(
            lstmcell,
            mgc_prenets, lf0_prenets,
            self.attention_mechanisms
        )
        if self.decoder_version == "v1":
            self.decoder_cell = DecoderRNNV1(
                self.decoder_out_units, self.attention_cell, dtype=self.dtype
            )
        else:
            self.decoder_cell = DecoderRNNV2(
                self.decoder_out_units,
                self.attention_cell,
                zoneout_factor_cell=self.zoneout_factor_cell,
                zoneout_factor_output=self.zoneout_factor_output,
                dtype=self.dtype
            )

        self.output_and_done_cell = OutputMgcLf0AndStopTokenWrapper(
            self.decoder_cell,
            self.num_mgcs * self.outputs_per_step,
            self.num_lf0s * self.outputs_per_step,
            dtype=self.dtype
        )

    def _set_prenets(self) -> _PreNet_Type:
        if self.use_speaker_embed is not None:
            mgc_prenets = (
                MultiSpeakerPreNet(
                    self._prenet_out_units[0],
                    self._drop_rate,
                    dtype=self.dtype,
                ),
                PreNet(
                    self._prenet_out_units[1],
                    self._drop_rate,
                    self.apply_dropout_on_inference,
                    dtype=self.dtype,
                )
            )
            lf0_prenets = (
                MultiSpeakerPreNet(
                    self._prenet_out_units[0],
                    self._drop_rate,
                    dtype=self.dtype,
                ),
                PreNet(
                    self._prenet_out_units[1],
                    self._drop_rate,
                    self.apply_dropout_on_inference, dtype=self.dtype,
                )
            )
        else:
            mgc_prenets = tuple([
                PreNet(
                    out_unit, self._drop_rate,
                    self.apply_dropout_on_inference, dtype=self.dtype
                )
                for out_unit in self._prenet_out_units
            ])
            lf0_prenets = tuple([
                PreNet(out_unit, self._drop_rate,
                       self.apply_dropout_on_inference, dtype=self.dtype)
                for out_unit in self._prenet_out_units
            ])

        return mgc_prenets, lf0_prenets

    def build(self, _):
        self.built = True

    def call(
        self,
        sources,
        speaker_embed=None,
        target=None,
        memory_sequence_length=None,
        target_sequence_length=None,
        training=None, **kwargs
    ):
        # TODO: Fix here
        source1, _ = sources
        batch_size = tf.shape(source1)[0]
        params = {
            "speaker_embed": speaker_embed,
            "memory_sequence_length": memory_sequence_length,
            "target_sequence_length": target_sequence_length
        }

        decoder_initial_state = self.output_and_done_cell.get_initial_state(
            inputs=sources, batch_size=batch_size, dtype=source1.dtype
        )
        if training:
            helper = TrainingMgcLf0Helper(
                target[0],
                target[1],
                self.num_mgcs,
                self.num_lf0s,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame
            )
        else:
            helper = StopTokenBasedMgcLf0InferenceHelper(
                batch_size,
                self.num_mgcs,
                self.num_lf0s,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                dtype=source1.dtype
            )

        basic_decoder = tfa.seq2seq.BasicDecoder(self.output_and_done_cell, helper, decoder_initial_state)
        ((decoder_mgc_outputs, decoder_lf0_outputs, stop_token),
         _), final_decoder_state, _ = tfa.seq2seq.dynamic_decode(
            basic_decoder,
            maximum_iterations=self.max_iters, **params
        )

        mgc_output = tf.reshape(decoder_mgc_outputs, [batch_size, -1, self.num_mgcs])
        lf0_output = tf.reshape(decoder_lf0_outputs, [batch_size, -1, self.num_lf0s])
        return mgc_output, lf0_output, stop_token, final_decoder_state


class DualSourceTransformerDecoder(BaseDecoder):

    def __init__(
        self, 
        attention_mechanisms: List[tfa.seq2seq.AttentionMechanism], 
        **kwargs
    ):
        super(DualSourceTransformerDecoder, self).__init__(**kwargs)

        prenets = self._set_prenets()
        lstmcell = ZoneoutLSTMCell(
            self.attention_rnn_out_units,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            dtype=self.dtype
        )
        self.attention_cell = DualSourceAttentionRNN(lstmcell, prenets, attention_mechanisms)
        if self.decoder_version == "v1":
            self.decoder_cell = DecoderRNNV1(
                self.decoder_out_units, self.attention_cell, dtype=self.dtype
            )
        else:
            self.decoder_cell = DecoderRNNV2(
                self.decoder_out_units,
                self.attention_cell,
                zoneout_factor_cell=self.zoneout_factor_cell,
                zoneout_factor_output=self.zoneout_factor_output,
                dtype=self.dtype
            )

        self.rnn_transformer = RNNTransformer(
            self.decoder_cell,
            self.self_attention_out_units,
            self.self_attention_transformer_num_conv_layers,
            self.self_attention_transformer_kernel_size,
            self.self_attention_num_heads,
            self.self_attention_num_hop,
            self.self_attention_drop_rate,
            self.num_mels,
            self.outputs_per_step,
            self.n_feed_frame,
            self.max_iters,
            self.dtype,
        )

    def build(self, _):
        self.built = True

    def _set_prenets(self) -> _PreNet_Type:
        if self.use_speaker_embed is not None:
            prenets = (
                MultiSpeakerPreNet(
                    self._prenet_out_units[0],
                    self._drop_rate,
                ),
                PreNet(
                    self._prenet_out_units[1],
                    self._drop_rate, self.apply_dropout_on_inference,
                )
            )
        else:
            prenets = tuple([PreNet(
                out_unit, self._drop_rate, self.apply_dropout_on_inference,
                dtype=self.dtype
            ) for out_unit in self._prenet_out_units])
        return prenets

    def call(
        self,
        sources,
        speaker_embed=None,
        target=None,
        memory_sequence_lengths: List[tf.Tensor] = None,
        target_sequence_length=None,
        training=None, **kwargs
    ):
        # TODO: FixMe
        source1, _ = sources
        params = {
            "speaker_embed": speaker_embed,
            "memory_sequence_lengths": memory_sequence_lengths
        }
        batch_size = tf.shape(source1)[0]
        # TODO: FixMe: Parameter that need to be passed as a input to the xformer
        decoder_initial_state = self.decoder_cell.get_initial_state(batch_size, dtype=source1.dtype)
        decoder_outputs, stop_token, final_decoder_state = self.rnn_transformer(
            target,
            decoder_initial_state,
            training=training,
            teacher_forcing=self.teacher_forcing,
            memory_sequence_length=target_sequence_length, **params
        )

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class DualSourceMgcLf0TransformerDecoder(BaseDecoder):

    def __init__(
        self,
        attention_mechanisms: List[tfa.seq2seq.AttentionMechanism],
        **kwargs
    ):
        super(DualSourceMgcLf0TransformerDecoder, self).__init__(**kwargs)

        mgc_prenets, lf0_prenets = self._set_prenets()
        lstmcell = ZoneoutLSTMCell(
            self.attention_rnn_out_units,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            dtype=self.dtype
        )
        self.attention_cell = DualSourceMgcLf0AttentionRNN(lstmcell, mgc_prenets, lf0_prenets, attention_mechanisms)
        if self.decoder_version == "v1":
            self.decoder_cell = DecoderRNNV1(
                self.decoder_out_units, self.attention_cell, dtype=self.dtype
            )
        else:
            self.decoder_cell = DecoderRNNV2(
                self.decoder_out_units,
                self.attention_cell,
                zoneout_factor_cell=self.zoneout_factor_cell,
                zoneout_factor_output=self.zoneout_factor_output,
                dtype=self.dtype
            )

        self.rnn_transformer = MgcLf0RNNTransformer(
            self.decoder_cell,
            self.self_attention_out_units,
            self.self_attention_transformer_num_conv_layers,
            self.self_attention_transformer_kernel_size,
            self.self_attention_num_heads,
            self.self_attention_num_hop,
            self.self_attention_drop_rate,
            self.num_mgcs,
            self.num_lf0s,
            self.outputs_per_step,
            self.n_feed_frame,
            self.max_iters,
            self.dtype,
        )

    def _set_prenets(self) -> _PreNet_Type:
        if self.use_speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            mgc_prenets = (MultiSpeakerPreNet(
                self._prenet_out_units[0],
                self._drop_rate, dtype=self.dtype
            ),
                PreNet(
                self._prenet_out_units[1], self._drop_rate,
                self.apply_dropout_on_inference, dtype=self.dtype
            ))
            lf0_prenets = (MultiSpeakerPreNet(
                self._prenet_out_units[0],
                self._drop_rate, dtype=self.dtype
            ),
                PreNet(
                    self._prenet_out_units[1], self._drop_rate,
                    self.apply_dropout_on_inference, dtype=self.dtype))
        else:
            mgc_prenets = tuple([PreNet(out_unit, self._drop_rate,
                                        self.apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])
            lf0_prenets = tuple([PreNet(out_unit, self._drop_rate,
                                        self.apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])

        return mgc_prenets, lf0_prenets

    def build(self, _):
        self.built = True

    def call(
        self,
        sources,
        speaker_embed=None,
        target=None,
        memory_sequence_lengths: List[tf.Tensor] = None,
        target_sequence_length=None,
        training=None, **kwargs
    ):
        # TODO: (FixMe) use input to attention
        source1, _ = sources
        params = {
            "speaker_embed": speaker_embed,
            "memory_sequence_lengths": memory_sequence_lengths
        }
        batch_size = tf.shape(source1)[0]
        # TODO: FixMe: Parameter that need to be passed as a input to the xformer
        decoder_initial_state = self.decoder_cell.zero_state(batch_size, dtype=source1.dtype)
        decoder_mgc_outputs, decoder_lf0_outputs, stop_token, final_decoder_state = self.rnn_transformer(
            target,
            decoder_initial_state,
            training=training,
            teacher_forcing=self.teacher_forcing,
            memory_sequence_length=target_sequence_length, **params
        )

        mgc_output = tf.reshape(decoder_mgc_outputs, [batch_size, -1, self.num_mgcs])
        lf0_output = tf.reshape(decoder_lf0_outputs, [batch_size, -1, self.num_lf0s])
        return mgc_output, lf0_output, stop_token, final_decoder_state


class TransformerDecoder(BaseDecoder):

    def __init__(
        self,
        attention_mechanisms: List[tfa.seq2seq.AttentionMechanism],
        **kwargs
    ):

        super(TransformerDecoder, self).__init__(**kwargs)
        prenets = self._set_prenets()

        lstmcell = ZoneoutLSTMCell(
            self.attention_out_units,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            dtype=self.dtype
        )
        self.attention_cell = AttentionRNN(lstmcell, prenets, attention_mechanisms, dtype=self.dtype)

        if self.decoder_version == "v1":
            self.decoder_cell = DecoderRNNV1(
                self.decoder_out_units, self.attention_cell, dtype=self.dtype
            )
        else:
            self.decoder_cell = DecoderRNNV2(
                self.decoder_out_units,
                self.attention_cell,
                zoneout_factor_cell=self.zoneout_factor_cell,
                zoneout_factor_output=self.zoneout_factor_output,
                dtype=self.dtype
            )

        self.rnn_transformer = RNNTransformer(
            self.decoder_cell,
            self.self_attention_out_units,
            self.self_attention_transformer_num_conv_layers,
            self.self_attention_transformer_kernel_size,
            self.self_attention_num_heads,
            self.self_attention_num_hop,
            self.self_attention_drop_rate,
            self.num_mels,
            self.outputs_per_step,
            self.n_feed_frame,
            self.max_iters,
            self.dtype,
        )

    def _set_prenets(self):
        if self.use_speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            prenets = (MultiSpeakerPreNet(
                self._prenet_out_units[0],
                self._drop_rate, dtype=self.dtype
            ),
                PreNet(
                self._prenet_out_units[1], self._drop_rate,
                self.apply_dropout_on_inference, dtype=self.dtype
            ))
        else:
            prenets = tuple([PreNet(out_unit, self._drop_rate,
                                    self.apply_dropout_on_inference,
                                    dtype=self.dtype)
                             for out_unit in self._prenet_out_units])
            return prenets

    def build(self, _):
        self.built = True

    def call(
        self,
        source,
        speaker_embed=None,
        target=None,
        memory_sequence_lengths: List[tf.Tensor] = None,
        target_sequence_length=None,
        training=None, **kwargs
    ):
        batch_size = tf.shape(source)[0]
        params = {
            "speaker_embed": speaker_embed,
            "memory_sequence_lengths": memory_sequence_lengths
        }
        # TODO: FixMe: Parameter that need to be passed as a input to the xformer
        decoder_initial_state = self.decoder_cell.get_initial_state(batch_size, dtype=source.dtype)
        decoder_outputs, stop_token, final_decoder_state = self.rnn_transformer(
            target,
            decoder_initial_state,
            training=training,
            teacher_forcing=self.teacher_forcing,
            memory_sequence_length=target_sequence_length, **params
        )

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class DualSourceDecoder(BaseDecoder):

    def __init__(
        self, 
        attention_mechanism: List[tfa.seq2seq.AttentionMechanism], 
        **kwargs
    ):
        super(DualSourceDecoder, self).__init__(**kwargs)

        prenets = self._set_prenets()
        lstmcell = ZoneoutLSTMCell(
            self.attention_rnn_out_units,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            dtype=self.dtype
        )
        self.attention_cell = DualSourceAttentionRNN(lstmcell, prenets, attention_mechanism, dtype=self.dtype)
        if self.decoder_version == "v1":
            self.decoder_cell = DecoderRNNV1(
                self.decoder_out_units, self.attention_cell, dtype=self.dtype
            )
        else:
            self.decoder_cell = DecoderRNNV2(
                self.decoder_out_units,
                self.attention_cell,
                zoneout_factor_cell=self.zoneout_factor_cell,
                zoneout_factor_output=self.zoneout_factor_output,
                dtype=self.dtype
            )
        self.output_and_done_cell = OutputAndStopTokenWrapper(
            self.decoder_cell, self.num_mels * self.outputs_per_step,
            dtype=self.dtype
        )

    def build(self, _):
        self.built = True

    def _set_prenets(self):
        if self.use_speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            prenets = (MultiSpeakerPreNet(
                self._prenet_out_units[0],
                self._drop_rate, dtype=self.dtype
            ),
                PreNet(
                    self._prenet_out_units[1],
                    self._drop_rate,
                    self.apply_dropout_on_inference, dtype=self.dtype
            ))
        else:
            prenets = tuple([
                PreNet(
                    out_unit, self._drop_rate,
                    self.apply_dropout_on_inference,
                    dtype=self.dtype
                ) for out_unit in self._prenet_out_units
            ])

        return prenets

    def call(
        self,
        sources,
        speaker_embed=None,
        target=None,
        memory_sequence_lengths: List[tf.Tensor] = None,
        target_sequence_length=None,
        training=None, **kwargs
    ):
        params = {
            "speaker_embed": speaker_embed,
            "memory_sequence_lengths": memory_sequence_lengths
        }
        source1, _ = sources

        batch_size = tf.shape(source1)[0]
        decoder_initial_state = self.decoder_cell.get_initial_state(
            batch_size, dtype=source1.dtype
        )

        if training:
            helper = TrainingHelper(
                target,
                self.num_mels,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                teacher_forcing=self.teacher_forcing
            )

        else:
            helper = StopTokenBasedInferenceHelper(
                batch_size,
                self.num_mels,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                dtype=source1.dtype
            )
            
        basic_decoder = tfa.seq2seq.BasicDecoder(self.output_and_done_cell, helper, decoder_initial_state)
        ((decoder_outputs, stop_token), _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            basic_decoder, maximum_iterations=self.max_iters
        )

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class MgcLf0Decoder(KL.Layer):

    def __init__(self, attention_mechanism: List[tfa.seq2seq.AttentionMechanism], **kwargs):
        super(MgcLf0Decoder, self).__init__(**kwargs)
        mgc_prenets, lf0_prenets = self._set_prenets()

        lstmcell = ZoneoutLSTMCell(
            self.attention_rnn_out_units,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            dtype=self.dtype,
        )

        self.attention_cell = MgcLf0AttentionRNN(lstmcell, mgc_prenets, lf0_prenets, attention_mechanism)
        if self.decoder_version == "v1":
            self.decoder_cell = DecoderRNNV1(
                self.decoder_out_units, self.attention_cell, dtype=self.dtype
            )
        else:
            self.decoder_cell = DecoderRNNV2(
                self.decoder_out_units,
                self.attention_cell,
                zoneout_factor_cell=self.zoneout_factor_cell,
                zoneout_factor_output=self.zoneout_factor_output,
                dtype=self.dtype
            )
        self.output_and_done_cell = OutputMgcLf0AndStopTokenWrapper(
            self.decoder_cell,
            self.num_mgcs * self.outputs_per_step,
            self.num_lf0s * self.outputs_per_step,
            dtype=self.dtype
        )

    def build(self, _):
        self.built = True

    def _set_prenets(self):
        if self.use_speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            mgc_prenets = (MultiSpeakerPreNet(
                self._prenet_out_units[0],
                self._drop_rate, dtype=self.dtype
            ),
                PreNet(
                self._prenet_out_units[1], self._drop_rate,
                self.apply_dropout_on_inference, dtype=self.dtype
            ))
            lf0_prenets = (MultiSpeakerPreNet(
                self._prenet_out_units[0],
                self._drop_rate, dtype=self.dtype
            ),
                PreNet(
                self._prenet_out_units[1], self._drop_rate,
                self.apply_dropout_on_inference, dtype=self.dtype
            ))
        else:
            mgc_prenets = tuple([PreNet(out_unit, self._drop_rate, self.apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])
            lf0_prenets = tuple([PreNet(out_unit, self._drop_rate, self.apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])
        return mgc_prenets, lf0_prenets

    def call(
        self,
        sources,
        speaker_embed=None,
        target=None,
        memory_sequence_length=None,
        target_sequence_length=None,
        training=None, **kwargs
    ):
        source = sources[0]
        batch_size = tf.shape(source)[0]
        params = {
            "speaker_embed": speaker_embed,
            "memory_sequence_length": memory_sequence_length
        }
        # TODO: FixMe: Parameter that need to be passed as a input to the xformer
        decoder_initial_state = self.decoder_cell.get_initial_state(batch_size, dtype=source.dtype)
        if training:
            helper = TrainingMgcLf0Helper(
                target[0],
                target[1],
                self.num_mgcs,
                self.num_lf0s,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                teacher_forcing=self.teacher_forcing
            )
        else:
            helper = StopTokenBasedMgcLf0InferenceHelper(
                batch_size,
                self.num_mgcs,
                self.num_lf0s,
                self.outputs_per_step,
                n_feed_frame=self.n_feed_frame,
                dtype=source.dtype
            )
        basic_decoder = tfa.seq2seq.BasicDecoder(self.output_and_done_cell, helper, decoder_initial_state)        
        ((decoder_mgc_outputs, decoder_lf0_outputs, stop_token),
         _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            basic_decoder, 
            maximum_iterations=self.max_iters, **params
        )

        mgc_output = tf.reshape(decoder_mgc_outputs, [batch_size, -1, self.num_mgcs])
        lf0_output = tf.reshape(decoder_lf0_outputs, [batch_size, -1, self.num_lf0s])
        return mgc_output, lf0_output, stop_token, final_decoder_state
