# ==== ## ==== ## ==== ## ==== ## ==== ## ==== ## ==== ## ==== #
#       SELF ATTENTION TACOTRON MODEL IMPLEMENTATION
# ==== ## ==== ## ==== ## ==== ## ==== ## ==== ## ==== ## ==== #
import sys
import abc
import numpy as np
import typing as t
from six import add_metaclass
import tensorflow as tf
from tensorflow_tts.models import BaseModel
from .encoder import EncoderOption
from .decoder import DecoderOption
from .attention import AttentionOptions
from .embedding import (
    ExternalEmbedding, MultiSpeakerPostNet, 
    ChannelEncoderPostNet, Embedding, PostNetV2
)
from deepinsight_speech.synthesizer.backbones.encoder import EncoderFactory, BaseEncoder
from deepinsight_speech.synthesizer.backbones.attention import AttentionFactory
from deepinsight_speech.synthesizer.backbones.decoder import DecoderFactory, BaseDecoder
import tensorflow_addons as tfa
from dataclasses import dataclass


@dataclass(init=False)
class ModelOutput:
    mgc_output: tf.Tensor
    postnet_v2_mgc_output: t.Any
    stop_token: t.Any
    alignments: t.List

    def __init__(
        self,
        mgc_output, postnet_v2_mgc_output, stop_token, alignments,
        **kwargs
    ):
        self.mgc_output = mgc_output
        self.postnet_v2_mgc_output = postnet_v2_mgc_output
        self.stop_token = stop_token
        self.alignments = alignments
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, k: str):
        assert hasattr(self, k), AttributeError(f"Invalid attribute: {k}")
        return getattr(self, k)


@add_metaclass(abc.ABCMeta)
class TacotronAttentionModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.use_window_mask = False
        self.maximum_iterations = 4000
        # self.enable_tflite_convertible = enable_tflite_convertible
        self.config = config

    def set_postnet_v2(self):
        config = self.config
        if self.config.speaker_embedd_to_postnet:
            return MultiSpeakerPostNet(
                out_units=config.num_mels,
                num_postnet_layers=config.num_postnet_v2_layers,
                kernel_size=config.postnet_v2_kernel_size,
                out_channels=config.postnet_v2_out_channels,
                drop_rate=config.postnet_v2_drop_rate
            )
        elif self.config.channel_id_to_postnet:
            return ChannelEncoderPostNet(
                out_units=config.num_mels,
                num_postnet_layers=config.num_postnet_v2_layers,
                kernel_size=config.postnet_v2_kernel_size,
                out_channels=config.postnet_v2_out_channels,
                drop_rate=config.postnet_v2_drop_rate
            )
        else:
            return PostNetV2(
                out_units=config.num_mels,
                num_postnet_layers=config.num_postnet_v2_layers,
                kernel_size=config.postnet_v2_kernel_size,
                out_channels=config.postnet_v2_out_channels,
                drop_rate=config.postnet_v2_drop_rate
            )

    def get_attention_mechanism(self, config) -> tfa.seq2seq.BahdanauAttention:
        cfg = AttentionOptions(
            num_units=config.attention1_out_units,
            memory=config.memory,
            memory_sequence_length=config.memory_sequence_length,
            attention_kernel=config.attention_kernel,
            attention_filters=config.attention_filters,
            smoothing=False,
            cumulative_weights=config.cumulative_weights,
            use_transition_agent=config.use_forward_attention_transition_agent,
            teacher_alignments=False,
        )

        return [
            getattr(AttentionFactory, attention_type)(**cfg)
            for attention_type in config["attention_type"]
        ]

    def get_encoder(self, config) -> BaseEncoder:
        # TODO: Get/Set input_length in config or from input params
        opts = {
            "cbhg_out_units": config.cbhg_out_units,
            "conv_channels": config.conv_channels,
            "max_filter_width": config.max_filter_width,
            "prenet_out_units": config.encoder_prenet_out_units,
            "projection1_out_channels": config.projection1_out_channels,
            "projection2_out_channels": config.projection2_out_channels,
            "num_highway": config.num_highway,
            "drop_rate": config.encoder_prenet_drop_rate,
            "use_zoneout": config.use_zoneout_at_encoder,
            "zoneout_factor_cell": config.zoneout_factor_cell,
            "zoneout_factor_output": config.zoneout_factor_output,
            "self_attention_out_units": config.encoder_self_attention_out_units,
            "self_attention_num_heads": config.encoder_self_attention_num_heads,
            "self_attention_num_hop": config.encoder_self_attention_num_hop,
            "self_attention_transformer_num_conv_layers": config.encoder_self_attention_transformer_num_conv_layers,
            "self_attention_transformer_kernel_size": config.self_attention_transformer_kernel_size,
            "self_attention_drop_rate": config.encoder_self_attention_drop_rate,
        }

        cfg = EncoderOption(**opts)

        assert hasattr(EncoderFactory, config.encoder), \
            AttributeError(f"Invalid Encoder option: {config.encoder}")
        return getattr(EncoderFactory, config.encoder)(**cfg)  # BaseEncoder

    def get_decoder(self, attention_mech, config) -> BaseDecoder:
        # TODO: Fix Decoder option and nn parameter
        cfg = DecoderOption(
            # teacher_alignments=config.teacher_alignments,
            apply_dropout_on_inference=config.apply_dropout_on_inference,
            prenet_out_units=config.decoder_prenet_out_units,
            drop_rate=config.decoder_prenet_drop_rate,
            attention_out_units=256,
            decoder_version=config.decoder_version,  # v1 | v2
            decoder_out_units=config.decoder_out_units,
            num_mels=config.num_mels,
            outputs_per_step=config.outputs_per_step,
            max_iters=config.max_iters,
            n_feed_frame=1,
            zoneout_factor_cell=config.zoneout_factor_cell,
            zoneout_factor_output=config.zoneout_factor_output,
            teacher_forcing=config.use_forced_alignment_mode,
            self_attention_out_units=config.decoder_self_attention_out_units,
            self_attention_num_heads=config.decoder_self_attention_num_heads,
            self_attention_num_hop=config.decoder_self_attention_num_hop,
        )

        assert hasattr(DecoderFactory, config.decoder), \
            AttributeError(f"Invalid decoder option: {config.decoder}")
        return getattr(DecoderFactory, config.decoder)(
            attention_func=attention_mech, **cfg)

    def setup_window(self, win_front, win_back):
        """Call only for inference."""
        self.use_window_mask = True
        self.win_front = win_front
        self.win_back = win_back

    def setup_maximum_iterations(self, maximum_iterations):
        """Call only for inference."""
        self.maximum_iterations = maximum_iterations

    def _build(self):
        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        input_lengths = np.array([9])
        speaker_ids = np.array([0])
        mel_outputs = np.random.normal(size=(1, 50, 80)).astype(np.float32)
        mel_lengths = np.array([50])
        self(
            input_ids,
            input_lengths,
            speaker_ids,
            mel_outputs,
            mel_lengths,
            10,
            training=True,
        )


class ExtendedTacotronV1Model(TacotronAttentionModel):

    def __init__(self, config, **kwargs):
        """Initalize tacotron-2 layers.
        """
        super().__init__(self, config, **kwargs)
        self.embedding = Embedding(
            self.config.num_symbols, embedding_dim=self.config.embedding_dim)

        if self.config.use_accent_type:
            self.accent_embedding = Embedding(
                self.config.num_accent_type,
                embedding_dim=config.accent_type_embedding_dim,
                index_offset=config.accent_type_offset
            )
        if self.config.use_speaker_embedding:
            self.speaker_embedding = Embedding(
                self.config.num_speakers,
                embedding_dim=self.config.speaker_embedding_dim,
                index_offset=self.config.speaker_embedding_offset
            )

        self.encoder = self.get_encoder(config)
        self.attentions = self.get_attention_mechanism(config)  # AttentionMechanism
        self.decoder = self.get_decoder(self.attentions, config)  # BaseDecoder

        self.postnet = self.set_postnet_v2()

    def call(
        self,
        input_ids,
        input_lengths,
        speaker_ids,
        mel_gts,
        mel_lengths,
        mel_dims=None,
        accent_type=None,
        training=False,
        maximum_iterations=None,
        use_window_mask=False,
        win_front=2,
        win_back=3,
        **kwargs,
    ):
        alignments = []
        mel_lengths, mel_widths = mel_dims
        
        embedding_output = self.embedding(input_ids)
        if self.config.use_accent_type:
            accent_embed = self.accent_embedding(accent_type)
            encoder_output = self.encoder(
                [embedding_output, accent_embed],
                input_lengths=input_lengths
            )
        else:
            encoder_output = self.encoder([embedding_output], input_lengths=input_lengths)

        speaker_embedding_output = self.speaker_embedding(speaker_ids)

        mel_output, stop_token, decoder_state = self.decoder(
            encoder_output,
            speaker_embed=speaker_embedding_output,
            memory_sequence_length=input_lengths,
            target_sequence_length=mel_lengths if training else None,
            target=mel_gts if training else None,
            training=training,
        )
        if self.config.decoder == "TransformerDecoder" and not training:
            decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
            alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
            decoder_self_attention_alignment = [
                tf.transpose(a, perm=[0, 2, 1]) for a in decoder_state.alignments
            ]
        else:
            decoder_rnn_state = decoder_state[0]
            alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
            decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

        if self.config.use_forced_alignment_mode:
            mel_output, stop_token, decoder_state = self.decoder(
                encoder_output,
                speaker_embed=speaker_embedding_output,
                training=training,
                teacher_forcing=False,
                memory_sequence_length=input_lengths,
                target_sequence_length=mel_lengths,
                target=mel_gts if training else None,
                teacher_alignments=tf.transpose(alignment, [0, 2, 1]),
                apply_dropout_on_inference=self.config.apply_dropout_on_inference
            )

            alignment = (
                tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history.stack(), [1, 2, 0])
                if self.config.decoder == "TransformerDecoder" and not training
                else tf.transpose(decoder_state[0].alignment_history.stack(), [1, 2, 0])
            )
        postnet_v2_mel_output = self.postnet(mel_output)
        alignments += [alignment]
        alignments += decoder_self_attention_alignment

        return ModelOutput(mel_output, postnet_v2_mel_output, stop_token, alignments)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="input_lengths"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="speaker_ids"),
        ],
    )
    def inference(self, input_ids, input_lengths, speaker_ids, **kwargs):
        """ Call logic. """
        return self.call(
            input_ids,
            input_lengths,
            speaker_ids,
            training=False,
            **kwargs
        )


class DualSourceSelfAttentionTacotronModel(TacotronAttentionModel):

    def __init__(self, config, **kwargs):
        super().__init__(self, config, **kwargs)
        self.embedding = Embedding(
            self.config.num_symbols,
            embedding_dim=self.config.embedding_dim
        )
        self.speaker_embedding = None
        self._compose = (lambda f, g: lambda arg, *args, **kwargs: f(g(arg, *args, **kwargs)))
        if self.config.use_accent_type:
            self.accent_embedding = Embedding(
                self.config.num_accent_type,
                embedding_dim=self.config.accent_type_embedding_dim,
                index_offset=self.config.accent_type_offset
            )

        # make sure that only one of (external_speaker_embedding, speaker_embedding) has been chosen
        assert not (self.config.use_speaker_embedding and self.config.use_external_speaker_embedding), \
            ValueError("Either speaker-embedding or external-speaker-embedding should be available")

        if self.config.use_external_speaker_embedding:
            self.speaker_embedding = ExternalEmbedding(
                self.config.embedding_file,
                self.config.num_speakers,
                embedding_dim=self.config.speaker_embedding_dim,
                index_offset=config.speaker_embedding_offset
            )
        else:
            self.speaker_embedding = Embedding(
                self.config.num_speakers,
                embedding_dim=self.config.speaker_embedding_dim,
                index_offset=self.config.speaker_embedding_offset
            )

        # resize speaker embedding with a projection layer
        if self.config.speaker_embedding_projection_out_dim > -1:
            resize = tf.layers.Dense(self.config.speaker_embedding_projection_out_dim, activation=tf.nn.relu)
            self.speaker_embedding = self._compose(resize, self.speaker_embedding)

        # language (dialect) embedding
        if self.config.use_language_embedding:
            self.language_embedding = ExternalEmbedding(
                self.config.language_embedding_file,
                self.config.num_speakers,
                embedding_dim=self.config.language_embedding_dim,
                index_offset=self.config.speaker_embedding_offset
            )

        # resize language embedding with a projection layer
        if self.config.language_embedding_projection_out_dim > -1:
            resize = tf.layers.Dense(
                self.config.language_embedding_projection_out_dim, activation=tf.nn.relu
            )
            self.language_embedding = self._compose(resize, self.language_embedding)

        # channel label
        if self.config.channel_id_to_postnet:
            self.channel_code = ExternalEmbedding(
                self.config.channel_id_file, self.config.num_speakers,
                embedding_dim=self.config.channel_id_dim, index_offset=self.config.speaker_embedding_offset
            )

        self.attentions = self.get_attention_mechanism(config)

        assert self.config.decoder in ["DualSourceDecoder", "DualSourceTransformerDecoder"]
        self.encoder = self.get_encoder(config)
        self.decoder = self.get_decoder(self.attentions, config)  # BaseDecoder

        if self.config.use_forced_alignment_mode:
            self.attentions = self.get_attention_mechanism(config)
            self.force_aligned_decoder = self.get_decoder(self.attentions, config)

        self.postnet = self.set_postnet_v2()

    def call(
        self,
        input_ids,
        input_lengths,
        speaker_ids,
        mel_gts,
        mel_lengths,
        training=False,
        maximum_iterations=None,
        use_window_mask=False,
        win_front=2,
        win_back=3,
        **kwargs,
    ):
        alignments = []
        # choose a speaker ID to synthesize as
        x = self.config.speaker_for_synthesis
        if x > -1:
            speaker_embedding_output = self.speaker_embedding(x)
        # default is to just use the speaker ID associated with the test utterance
        elif self.config.use_speaker_embedding or self.config.use_external_speaker_embedding:
            speaker_embedding_output = self.speaker_embedding(speaker_ids)

        if x > -1:  # -1 is default (just use the speaker ID associated with the test utterance)
            language_embedding_output = self.language_embedding(x)
        elif self.config.use_language_embedding:
            language_embedding_output = self.language_embedding(speaker_ids)

        channel_code_output = self.channel_code(speaker_ids) if self.config.channel_id_to_postnet else None

        # get phone/letter embeddings
        embedding_output = self.embedding(input_ids)  # phone/letter embedding
        # add language embedding as bias along the time axis to embedding_output
        if self.config.language_embedd_to_input:
            language_embedd_input_projection_layer = tf.layers.Dense(self.config.embedding_dim)
            language_embedd_input_projected = language_embedd_input_projection_layer(language_embedding_output)
            expand_language_embedding_input = tf.tile(tf.expand_dims(language_embedd_input_projected, axis=1),
                                                      [1, tf.shape(embedding_output)[1], 1])
            embedding_output = embedding_output + expand_language_embedding_input  # as bias

        # pass input embeddings to encoder
        encoder_lstm_output, encoder_self_attention_output, self_attention_alignment = self.encoder(
            (embedding_output, self.accent_embedding(self.config.accent_type)),
            input_lengths=input_lengths
        ) if self.config.use_accent_type else self.encoder(
            embedding_output, input_lengths=input_lengths
        )

        # concatenate encoder outputs with speaker embedding along the time axis
        if self.config.speaker_embedd_to_decoder:
            expand_speaker_embedding_output = tf.tile(tf.expand_dims(speaker_embedding_output, axis=1),
                                                      [1, tf.shape(encoder_lstm_output)[1], 1])
            encoder_lstm_output = tf.concat((encoder_lstm_output, expand_speaker_embedding_output), axis=-1)
            encoder_self_attention_output = tf.concat(
                (encoder_self_attention_output, expand_speaker_embedding_output), axis=-1)

        # concatenate encoder outputs with language embedding along the time axis
        if self.config.language_embedd_to_decoder:
            expand_language_embedding_output = tf.tile(tf.expand_dims(language_embedding_output, axis=1),
                                                       [1, tf.shape(encoder_lstm_output)[1], 1])
            encoder_lstm_output = tf.concat((encoder_lstm_output, expand_language_embedding_output), axis=-1)
            encoder_self_attention_output = tf.concat(
                (encoder_self_attention_output, expand_language_embedding_output), axis=-1)

        # arrange to (B, T_memory, T_query)
        mel_output, stop_token, decoder_state = self.decoder(
            (encoder_lstm_output, encoder_self_attention_output),
            speaker_embed=speaker_embedding_output,
            teacher_forcing=self.config.use_forced_alignment_mode,
            memory_sequence_length=input_lengths,
            memory2_sequence_length=input_lengths,
            target_sequence_length=mel_lengths if training else None,
            target=mel_gts if training else None,
            apply_dropout_on_inference=self.config.apply_dropout_on_inference
        )
        self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in self.config.self_attention_alignment]
        if self.config.decoder == "DualSourceTransformerDecoder" and not training:
            decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
            alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
            alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
            decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                decoder_state.alignments]
        else:
            decoder_rnn_state = decoder_state[0]
            alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
            alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
            decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

        if self.config.use_forced_alignment_mode:
            mel_output, stop_token, decoder_state = self.force_aligned_decoder(
                (encoder_lstm_output, encoder_self_attention_output),
                speaker_embed=speaker_embedding_output if self.config.speaker_embedd_to_prenet else None,
                training=training,
                validation=not training,
                teacher_forcing=False,
                memory_sequence_length=input_lengths,
                memory2_sequence_length=input_lengths,
                target_sequence_length=mel_lengths if training else None,
                target=mel_gts if training else None,
                teacher_alignments=(
                    tf.transpose(alignment1, perm=[0, 2, 1]),
                    tf.transpose(alignment2, perm=[0, 2, 1])),
                apply_dropout_on_inference=self.config.apply_dropout_on_inference
            )

            if self.config.decoder == "DualSourceTransformerDecoder" and not training:
                alignment1 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[0].stack(),
                                          [1, 2, 0])
                alignment2 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[1].stack(),
                                          [1, 2, 0])
                decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                    decoder_state.alignments]
            else:
                alignment1 = tf.transpose(decoder_state[0].alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_state[0].alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

        postnet_v2_mel_output = self.postnet(mel_output)
        alignments += [alignment1, alignment2]
        alignments += self_attention_alignment
        alignments += decoder_self_attention_alignment

        return ModelOutput(mel_output, postnet_v2_mel_output, stop_token, alignments)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="input_lengths"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="speaker_ids"),
        ],
    )
    def inference(self, input_ids, input_lengths, speaker_ids, **kwargs):
        return self.call(
            input_ids,
            input_lengths,
            speaker_ids,
            training=False,
            **kwargs
        )


class DualSourceSelfAttentionMgcLf0TacotronModel(TacotronAttentionModel):

    def __init__(self, config, **kwargs):
        super().__init__(self, config, **kwargs)
        self.speaker_embedding = None
        self.accent_embedding = None
        self.postnet_v2_mgc_output = None
        assert config.decoder in ["MgcLf0DualSourceDecoder", "DualSourceMgcLf0TransformerDecoder"]

        self.embedding = Embedding(config.num_symbols, embedding_dim=config.embedding_dim)

        if config.use_accent_type:
            self.accent_embedding = Embedding(
                config.num_accent_type,
                embedding_dim=config.accent_type_embedding_dim,
                index_offset=config.accent_type_offset
            )

        self.attentions = self.get_attention_mechanism(config)
        self.encoder = self.get_encoder(config)
        self.decoder = self.get_decoder(self.attentions, config)

        if config.use_speaker_embedding:
            self.speaker_embedding = Embedding(
                config.num_speakers,
                embedding_dim=config.speaker_embedding_dim,
                index_offset=config.speaker_embedding_offset
            )

        self.postnet_v2_mgc_output = PostNetV2(
            out_units=config.num_mgcs,
            num_postnet_layers=config.num_postnet_v2_layers,
            kernel_size=config.postnet_v2_kernel_size,
            out_channels=config.postnet_v2_out_channels,
            drop_rate=config.postnet_v2_drop_rate
        )

    def call(
        self,
        input_ids,
        input_lengths,
        speaker_ids,
        mel_gts,
        mel_lengths,
        training=False,
        maximum_iterations=None,
        use_window_mask=False,
        win_front=2,
        win_back=3,
        **kwargs,
    ):
        alignments = []
        embedding_output = self.embedding(input_ids)
        accent_emdoutput = self.accent_embedding(self.config.accent_type)
        encoder_lstm_output, encoder_self_attention_output, self_attention_alignment = self.encoder(
            (embedding_output, accent_emdoutput), input_lengths=input_lengths
        ) if self.config.use_accent_type else self.encoder(
            embedding_output, input_lengths=input_lengths
        )

        speaker_embedding_output = self.speaker_embedding(speaker_ids)

        mgc_output, lf0_output, stop_token, decoder_state = self.decoder(
            (encoder_lstm_output, encoder_self_attention_output),
            speaker_embed=speaker_embedding_output,
            training=training,
            validation=not training or self.config.use_forced_alignment_mode,
            teacher_forcing=self.config.use_forced_alignment_mode,
            memory_sequence_length=input_lengths,
            memory2_sequence_length=input_lengths,
            target=mel_gts if training else None,
            target_sequence_length=mel_lengths if training else None,
            apply_dropout_on_inference=self.config.apply_dropout_on_inference
        )

        # arrange to (B, T_memory, T_query)
        self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in self_attention_alignment]
        if self.config.decoder == "DualSourceMgcLf0TransformerDecoder" and not training:
            decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
            alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
            alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
            decoder_self_attention_alignment = [
                tf.transpose(a, perm=[0, 2, 1]) for a in decoder_state.alignments
            ]
        else:
            decoder_rnn_state = decoder_state[0]
            alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
            alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
            decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

        if self.config.use_forced_alignment_mode:
            mgc_output, lf0_output, stop_token, decoder_state = self.decoder(
                (encoder_lstm_output, encoder_self_attention_output),
                speaker_embed=speaker_embedding_output,
                training=training,
                validation=not training,
                teacher_forcing=False,
                memory_sequence_length=input_lengths,
                memory2_sequence_length=input_lengths,
                target_sequence_length=mel_lengths if training else None,
                target=mel_gts if training else None,
                teacher_alignments=(
                    tf.transpose(alignment1, perm=[0, 2, 1]),
                    tf.transpose(alignment2, perm=[0, 2, 1])),
                apply_dropout_on_inference=self.config.apply_dropout_on_inference)

            if self.config.decoder == "DualSourceMgcLf0TransformerDecoder" and not training:
                alignment1 = tf.transpose(
                    decoder_state.rnn_state.rnn_state[0].alignment_history[0].stack(),
                    [1, 2, 0]
                )
                alignment2 = tf.transpose(
                    decoder_state.rnn_state.rnn_state[0].alignment_history[1].stack(),
                    [1, 2, 0]
                )
                decoder_self_attention_alignment = [
                    tf.transpose(a, perm=[0, 2, 1]) for a in decoder_state.alignments
                ]
            else:
                alignment1 = tf.transpose(decoder_state[0].alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_state[0].alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

        alignments += [alignment1, alignment2]
        alignments += self_attention_alignment
        alignments += decoder_self_attention_alignment
        postnet_v2_mgc_output = self.postnet(mgc_output)
        return ModelOutput(mgc_output, postnet_v2_mgc_output, stop_token, alignments, lf0_output=lf0_output)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="input_lengths"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="speaker_ids"),
        ],
    )
    def inference(self, input_ids, input_lengths, speaker_ids, **kwargs):
        return self.call(
            input_ids,
            input_lengths,
            speaker_ids,
            training=False,
            **kwargs
        )


class MgcLf0TacotronModel(TacotronAttentionModel):

    def __init__(self, config, model_dir=None, warm_start_from=None):

        self.embedding = Embedding(config.num_symbols, embedding_dim=config.embedding_dim)
        self.accent_embedding, self.speaker_embedding = None, None
        if config.use_accent_type:
            self.accent_embedding = Embedding(
                config.num_accent_type,
                embedding_dim=config.accent_type_embedding_dim,
                index_offset=config.accent_type_offset
            )

        self.attention = self.get_attention_mechanism(config)
        self.encoder = self.get_encoder(self.config)
        self.decoder = self.get_decoder(self.attention, self.config)

        if config.use_speaker_embedding:
            self.speaker_embedding = Embedding(
                config.num_speakers,
                embedding_dim=config.speaker_embedding_dim,
                index_offset=config.speaker_embedding_offset
            )

        self.postnet = PostNetV2(
            out_units=config.num_mgcs,
            num_postnet_layers=config.num_postnet_v2_layers,
            kernel_size=config.postnet_v2_kernel_size,
            out_channels=config.postnet_v2_out_channels,
            drop_rate=config.postnet_v2_drop_rate
        )

    def call(
        self,
        input_ids,
        input_lengths,
        speaker_ids,
        mel_gts,
        mel_lengths,
        training=False,
        maximum_iterations=None,
        use_window_mask=False,
        win_front=2,
        win_back=3,
        **kwargs,
    ):
        alignments = []
        embout = self.embedding(input_ids)
        if self.config.use_accent_type:
            accent_embout = self.accent_embedding(self.config.accent_type)
            encoder_output = self.encoder([embout, accent_embout], input_lengths=input_lengths)
        else:
            encoder_output = self.encoder(embout, input_lengths=input_lengths)

        speaker_embedding_output = self.speaker_embedding(speaker_ids)

        mgc_output, lf0_output, stop_token, decoder_state = self.decoder(
            encoder_output,
            speaker_embed=speaker_embedding_output,
            training=training,
            validation=(not training or self.config.use_forced_alignment_mode),
            teacher_forcing=self.config.use_forced_alignment_mode,
            memory_sequence_length=input_lengths,
            target_sequence_length=mel_lengths if training else None,
            target=mel_gts,
            apply_dropout_on_inference=self.config.apply_dropout_on_inference
        )

        # arrange to (B, T_memory, T_query)
        if self.config.decoder == "TransformerDecoder" and not training:
            decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
            alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
            decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                decoder_state.alignments]
        else:
            decoder_rnn_state = decoder_state[0]
            alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
            decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

        if self.config.use_forced_alignment_mode:
            mgc_output, lf0_output, stop_token, decoder_state = self.decoder(
                encoder_output,
                speaker_embed=speaker_embedding_output,
                training=training,
                validation=None,
                teacher_forcing=False,
                memory_sequence_length=input_lengths,
                target_sequence_length=mel_lengths if training else None,
                target=mel_gts,
                teacher_alignments=tf.transpose(alignment, perm=[0, 2, 1]),
                apply_dropout_on_inference=self.config.apply_dropout_on_inference
            )

        if self.config.decoder == "TransformerDecoder" and not training:
            alignment = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history.stack(), [1, 2, 0])
            decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                decoder_state.alignments]
        else:
            alignment = tf.transpose(decoder_state[0].alignment_history.stack(), [1, 2, 0])
            decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

        postnet_v2_mgc_output = self.postnet(mgc_output)
        alignments.append(alignment)
        alignments += decoder_self_attention_alignment
        return ModelOutput(mgc_output, postnet_v2_mgc_output, stop_token, alignments, lf0_output=lf0_output)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="input_lengths"),
            tf.TensorSpec([None, ], dtype=tf.int32, name="speaker_ids"),
        ],
    )
    def inference(self, input_ids, input_lengths, speaker_ids, **kwargs):
        return self.call(
            input_ids,
            input_lengths,
            speaker_ids,
            training=False,
            **kwargs
        )
