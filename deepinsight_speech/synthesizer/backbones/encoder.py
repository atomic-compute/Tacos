from typing import Tuple, List
import tensorflow as tf
import tensorflow.keras.layers as KL
from .embedding import PreNet, CBHG, Conv1d, ZoneoutLSTMCell, ZoneoutCBHG
from .attention import SelfAttentionTransformer
from functools import reduce
import enum
from easydict import EasyDict


class EncoderOption(EasyDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EncoderFactory(enum.Enum):
    SelfAttentionCBHGEncoderWithAccentType = 'SelfAttentionCBHGEncoderWithAccentType'
    SelfAttentionCBHGEncoder = 'SelfAttentionCBHGEncoder'
    EncoderV1WithAccentType = 'EncoderV1WithAccentType'
    ZoneoutEncoderV1 = 'ZoneoutEncoderV1'
    EncoderV2 = 'EncoderV2'


class BaseEncoder(KL.Layer):
    def __init__(
        self,
        cbhg_out_units=256,
        conv_channels=128,
        max_filter_width=16,
        prenet_out_units=(256, 128),
        projection1_out_channels=128,
        projection2_out_channels=128,
        num_highway=4,
        drop_rate=0.5,
        use_zoneout=False,
        zoneout_factor_cell=0.0,
        zoneout_factor_output=0.0,
        self_attention_out_units=32,
        self_attention_num_heads=2,
        self_attention_num_hop=1,
        self_attention_transformer_num_conv_layers=1,
        self_attention_transformer_kernel_size=5,
        self_attention_drop_rate=0.1,
        name=None, dtype=None, **kwargs
    ):
        self.cbhg_out_units = cbhg_out_units
        self.conv_channels = conv_channels
        self.max_filter_width = max_filter_width
        self.prenet_out_units = prenet_out_units
        self.projection1_out_channels = projection1_out_channels
        self.projection2_out_channels = projection2_out_channels
        self.num_highway = num_highway
        self.drop_rate = drop_rate
        self.use_zoneout = use_zoneout
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self.self_attention_out_units = self_attention_out_units
        self.self_attention_num_heads = self_attention_num_heads
        self.self_attention_num_hop = self_attention_num_hop
        self.self_attention_transformer_num_conv_layers = self_attention_transformer_num_conv_layers
        self.self_attention_transformer_kernel_size = self_attention_transformer_kernel_size
        self.self_attention_drop_rate = self_attention_drop_rate
        super().__init__(name=name, dtype=dtype, **kwargs)

        self.set_prenets = (lambda prenet_out_units: [PreNet(out_unit, drop_rate, dtype=self.dtype)
                                                      for out_unit in prenet_out_units])

        if self.use_zoneout:
            self.set_cbhg = ZoneoutCBHG(
                self.cbhg_out_units,
                self.conv_channels,
                self.max_filter_width,
                self.projection1_out_channels,
                self.projection2_out_channels,
                self.num_highway,
                self.zoneout_factor_cell,
                self.zoneout_factor_output,
                dtype=self.dtype,
            )
        else:
            self.set_cbhg = CBHG(
                self.cbhg_out_units,
                self.conv_channels,
                self.max_filter_width,
                self.projection1_out_channels,
                self.projection2_out_channels,
                self.num_highway,
                dtype=self.dtype,
            )


class SelfAttentionCBHGEncoder(BaseEncoder):

    def __init__(
        self,
        cbhg_out_units=224,
        conv_channels=128,
        max_filter_width=16,
        projection1_out_channels=128,
        projection2_out_channels=128,
        num_highway=4,
        self_attention_out_units=32,
        self_attention_num_heads=2,
        self_attention_num_hop=1,
        drop_rate=0.5,
        use_zoneout=False,
        prenet_out_units=(256, 128),
        zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
        self_attention_drop_rate=0.1, dtype=None,
        **kwargs
    ):

        super().__init__(
            cbhg_out_units=cbhg_out_units,
            conv_channels=conv_channels,
            max_filter_width=max_filter_width,
            projection1_out_channels=projection1_out_channels,
            projection2_out_channels=projection2_out_channels,
            num_highway=num_highway,
            self_attention_out_units=self_attention_out_units,
            self_attention_num_heads=self_attention_num_heads,
            self_attention_num_hop=self_attention_num_hop,
            drop_rate=drop_rate,
            use_zoneout=use_zoneout,
            prenet_out_units=prenet_out_units,
            zoneout_factor_cell=zoneout_factor_cell,
            zoneout_factor_output=zoneout_factor_output,
            self_attention_drop_rate=self_attention_drop_rate,
            **kwargs
        )

        self.prenets = self.set_prenets(prenet_out_units)
        self.cbhg = self.set_cbhg()
        self.self_attention_projection_layer = KL.Dense(self.self_attention_out_units, dtype=self.dtype)

        self.self_attention = [
            SelfAttentionTransformer(
                out_units=self_attention_out_units,
                self_attention_out_units=self_attention_out_units,
                self_attention_num_heads=self_attention_num_heads,
                self_attention_drop_rate=self_attention_drop_rate,
                use_subsequent_mask=False,
                dtype=dtype
            ) for i in range(1, self_attention_num_hop + 1)
        ]

    def build(self, input_shape):
        embed_dim = input_shape[2].value
        self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputs)
        lstm_output = self.cbhg(prenet_output, input_lengths=input_lengths)

        self_attention_input = self.self_attention_projection_layer(lstm_output)

        def self_attend(input, alignments, layer):
            output, alignment = layer(input, memory_sequence_length=input_lengths)
            return output, alignments + alignment

        self_attention_output, self_attention_alignments = reduce(
            lambda acc, sa: self_attend(acc[0], acc[1], sa),
            self.self_attention, (self_attention_input, [])
        )
        return lstm_output, self_attention_output, self_attention_alignments

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class SelfAttentionCBHGEncoderWithAccentType(BaseEncoder):

    def __init__(
        self,
        cbhg_out_units=224,
        conv_channels=128,
        max_filter_width=16,
        projection1_out_channels=128,
        projection2_out_channels=128,
        num_highway=4,
        self_attention_out_units=32,
        self_attention_num_heads=2,
        self_attention_num_hop=1,
        prenet_out_units=(224, 112),
        accent_type_prenet_out_units=(32, 16),
        drop_rate=0.5,
        use_zoneout=False,
        zoneout_factor_cell=0.0,
        zoneout_factor_output=0.0,
        self_attention_drop_rate=0.1,
        dtype=None, **kwargs
    ):
        super(SelfAttentionCBHGEncoderWithAccentType, self).__init__(
            cbhg_out_units,
            conv_channels,
            max_filter_width,
            projection1_out_channels,
            projection2_out_channels,
            num_highway,
            self_attention_out_units,
            self_attention_num_heads,
            self_attention_num_hop,
            prenet_out_units,
            accent_type_prenet_out_units,
            drop_rate=drop_rate,
            use_zoneout=use_zoneout,
            zoneout_factor_cell=zoneout_factor_cell,
            zoneout_factor_output=zoneout_factor_output,
            self_attention_drop_rate=self_attention_drop_rate,
            **kwargs
        )

        self.prenets = [
            PreNet(out_unit, drop_rate, dtype=dtype)
            for out_unit in prenet_out_units
        ]
        self.accent_type_prenets = self.set_prenets(accent_type_prenet_out_units)

        self.cbhg = ZoneoutCBHG(
            cbhg_out_units,
            conv_channels,
            max_filter_width,
            projection1_out_channels,
            projection2_out_channels,
            num_highway,
            zoneout_factor_cell,
            zoneout_factor_output,
            dtype=dtype
        )

        self.self_attention_projection_layer = tf.layers.Dense(self_attention_out_units, dtype=self.dtype)
        self.self_attention = [
            SelfAttentionTransformer(
                out_units=self_attention_out_units,
                self_attention_out_units=self_attention_out_units,
                self_attention_num_heads=self_attention_num_heads,
                self_attention_drop_rate=self_attention_drop_rate,
                use_subsequent_mask=False,
                dtype=dtype
            )
            for i in range(1, self_attention_num_hop + 1)
        ]

    def build(self, input_shape):
        (phoneme_input_shape, accent_type_shape) = input_shape
        embed_dim = phoneme_input_shape[2].value
        accent_type_embed_dim = accent_type_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim),
                                      tf.assert_equal(self.accent_type_prenet_out_units[0], accent_type_embed_dim),
                                      tf.assert_equal(self.cbhg_out_units + self.self_attention_out_units,
                                                      embed_dim + accent_type_embed_dim)]):
            self.built = True

    def call(self, inputs: Tuple[tf.Tensor], input_lengths=None, training=False, **kwargs):
        inputt, accent_type = inputs
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputt)
        accent_type_prenet_output = reduce(lambda acc, pn: pn(acc), self.accent_type_prenets, accent_type)
        concatenated = tf.concat([prenet_output, accent_type_prenet_output], axis=-1)
        lstm_output = self.cbhg(concatenated, input_lengths=input_lengths)

        self_attention_input = self.self_attention_projection_layer(lstm_output)

        def self_attend(input, alignments, layer):
            output, alignment = layer(input, memory_sequence_length=input_lengths)
            return output, alignments + alignment

        self_attention_output, self_attention_alignments = reduce(
            lambda acc, sa: self_attend(acc[0], acc[1], sa), self.self_attention, (self_attention_input, [])
        )
        return lstm_output, self_attention_output, self_attention_alignments

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class EncoderV1WithAccentType(BaseEncoder):

    def __init__(
        self,
        cbhg_out_units=256,
        conv_channels=128,
        max_filter_width=16,
        projection1_out_channels=128,
        projection2_out_channels=128,
        num_highway=4,
        prenet_out_units=(224, 112),
        accent_type_prenet_out_units=(32, 16),
        drop_rate=0.5,
        use_zoneout=False,
        zoneout_factor_cell=0.0,
        zoneout_factor_output=0.0,
        name=None, dtype=None, **kwargs
    ):
        super(EncoderV1WithAccentType, self).__init__(
            cbhg_out_units,
            conv_channels,
            max_filter_width,
            projection1_out_channels,
            projection2_out_channels,
            num_highway,
            zoneout_factor_cell,
            zoneout_factor_output,
            use_zoneout=use_zoneout,
            name=name, dtype=dtype, **kwargs
        )
        self.prenet_out_units = prenet_out_units
        self.accent_type_prenet_out_units = accent_type_prenet_out_units
        self.cbhg_out_units = cbhg_out_units
        self.prenets = [PreNet(out_unit, drop_rate, dtype=dtype) for out_unit in prenet_out_units]
        self.accent_type_prenets = [
            PreNet(out_unit, drop_rate, dtype=dtype)
            for out_unit in accent_type_prenet_out_units
        ]

        self.cbhg = self.set_cbhg()

    def build(self, input_shape):
        (phoneme_input_shape, accent_type_shape) = input_shape
        embed_dim = phoneme_input_shape[2].value
        accent_type_embed_dim = accent_type_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim),
                                      tf.assert_equal(self.accent_type_prenet_out_units[0], accent_type_embed_dim),
                                      tf.assert_equal(self.cbhg_out_units, embed_dim + accent_type_embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        input, accent_type = inputs
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, input)
        accent_type_prenet_output = reduce(lambda acc, pn: pn(acc), self.accent_type_prenets, accent_type)
        concatenated = tf.concat([prenet_output, accent_type_prenet_output], axis=-1)
        cbhg_output = self.cbhg(concatenated, input_lengths=input_lengths)
        return cbhg_output

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class ZoneoutEncoderV1(BaseEncoder):

    def __init__(
        self,
        cbhg_out_units=256,
        conv_channels=128,
        max_filter_width=16,
        projection1_out_channels=128,
        projection2_out_channels=128,
        num_highway=4,
        prenet_out_units=(256, 128),
        drop_rate=0.5,
        use_zoneout=False,
        zoneout_factor_cell=0.0,
        zoneout_factor_output=0.0,
        name=None, dtype=None, **kwargs
    ):
        super(ZoneoutEncoderV1, self).__init__(
            cbhg_out_units=cbhg_out_units,
            conv_channels=conv_channels,
            max_filter_width=max_filter_width,
            prenet_out_units=prenet_out_units,
            projection1_out_channels=projection1_out_channels,
            projection2_out_channels=projection2_out_channels,
            num_highway=num_highway,
            drop_rate=drop_rate,
            use_zoneout=use_zoneout,
            zoneout_factor_cell=zoneout_factor_cell,
            zoneout_factor_output=zoneout_factor_output,
            name=name, dtype=dtype, **kwargs
        )
        self.prenet_out_units = prenet_out_units
        self.cbhg_out_units = cbhg_out_units

        self.prenets = [PreNet(out_unit, drop_rate, dtype=dtype) for out_unit in prenet_out_units]

    def build(self, input_shape):
        embed_dim = input_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputs)
        cbhg_output = self.cbhg(prenet_output, input_lengths=input_lengths)
        return cbhg_output

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class EncoderV2(BaseEncoder):

    def __init__(
        self,
        num_conv_layers,
        kernel_size,
        out_units,
        drop_rate,
        zoneout_factor_cell=0.0,
        zoneout_factor_output=0.0,
        use_zoneout=False,
        name=None, dtype=None, **kwargs
    ):
        super(EncoderV2, self).__init__(
            drop_rate=drop_rate,
            use_zoneout=use_zoneout,
            zoneout_factor_cell=zoneout_factor_cell,
            zoneout_factor_output=zoneout_factor_output,
            name=name, dtype=dtype, **kwargs
        )
        assert out_units % 2 == 0
        self.out_units = out_units

        self.convolutions = [
            Conv1d(kernel_size, out_units, activation=tf.nn.relu,
                   drop_rate=self.drop_rate,
                   name=f"conv1d_{i}",
                   dtype=self.dtype)
            for i in range(0, num_conv_layers)
        ]
        self.bidirectional = KL.Bidirectional(
            ZoneoutLSTMCell(
                self.out_units // 2,
                self.zoneout_factor_cell,
                self.zoneout_factor_output,
                dtype=self.dtype
            ),
            backward_layer=ZoneoutLSTMCell(
                self.out_units // 2,
                self.zoneout_factor_cell,
                self.zoneout_factor_output,
                dtype=self.dtype
            ),
            input_shape=(), # TODO: Fix this
            dtype=self.dtype
        )

    def call(self, inputs, input_lengths=None, training=False, **kwargs):
        conv_output = reduce(lambda acc, conv: conv(acc), self.convolutions, inputs)
        self.outputs, self.states = self.bidirectional(inputs, initial_state=conv_output, training=training)
        return tf.concat(self.outputs, axis=-1)
