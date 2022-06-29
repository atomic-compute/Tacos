# ====#=========#=========#=========#=========#=========#=========#===== #
#               SELF-ATTENTION TACOTRON TRAINER
# ====#=========#=========#=========#=========#=========#=========#===== #
import os
import click
import yaml
import logging
from typing import List
from six import add_metaclass
from abc import ABCMeta
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_tts.configs import Tacotron2Config
import deepinsight_speech
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy
from deepinsight_speech.synthesizer.modules.losses import ComputeWeightedLoss
from deepinsight_speech.synthesizer.modules.metrics import MetricsSaver
from deepinsight_speech.apps.selfattention_tacotron2.dataset import CharactorMelDataset
from deepinsight_speech.synthesizer.backbones.tacotronv2 import (
    TacotronAttentionModel,
    MgcLf0TacotronModel,
    ExtendedTacotronV1Model,
    DualSourceSelfAttentionTacotronModel,
    DualSourceSelfAttentionMgcLf0TacotronModel
)
import tensorflow as tf
import dataclasses
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError


@dataclasses.dataclass(frozen=True)
class trainer_model_cls:
    """Pair of model-class and trainer-class
    """
    model: TacotronAttentionModel
    trainer: Seq2SeqBasedTrainer


@add_metaclass(ABCMeta)
class SelfAttentionTacotronTrainer(Seq2SeqBasedTrainer):
    def __init__(
        self, config, strategy, steps=0, epochs=0, is_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.

        """
        super(Seq2SeqBasedTrainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "stop_token_loss",
            "mel_loss_before",
            "mel_loss_after",
            "guided_attention_loss",
        ]
        self.config = config
        # TODO: Fix this loss function
        self._spec_loss = {'l1': ComputeWeightedLoss(), 'mse': MeanSquaredError()}
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

    def _one_step_predict_per_replica(self, batch):
        """One step predict per GPU

        Tacotron-2 used teacher-forcing when training and evaluation.
        So we need pass `training=True` for inference step.
        """
        outputs = self._model(**batch, training=True)
        return outputs

    def _one_step_evaluate_per_replica(self, batch):
        """One step evaluate per GPU

        Tacotron-2 used teacher-forcing when training and evaluation.
        So we need pass `training=True` for inference step.
        """
        outputs = self._model(**batch, training=True)
        _, dict_metrics_losses = self.compute_per_example_losses(batch, outputs)
        self.update_eval_metrics(dict_metrics_losses)

    def init_train_eval_metrics(self, list_metrics_name: List[str]):
        """Init train and eval metrics to save it to tensorboard."""
        # TODO: FixMe
        alignment_saver = MetricsSaver(

            # [alignment] + decoder_self_attention_alignment, global_step, mel_output,
            # labels.mel,
            # labels.target_length,
            # features.id,
            # features.text,
            # 1,
            # mode, summary_writer,
            save_training_time_metrics=self.config.save_training_time_metrics,
            keep_eval_results_max_epoch=self.config.keep_eval_results_max_epoch
        )
        self.train_metrics = {}
        self.eval_metrics = {}
        for name in list_metrics_name:
            self.train_metrics.update(
                {name: tf.keras.metrics.Mean(name="train_" + name, dtype=tf.float32)}
            )
            self.eval_metrics.update(
                {name: tf.keras.metrics.Mean(name="eval_" + name, dtype=tf.float32)}
            )

    def _train_step(self, batch):
        """Here we re-define _train_step because apply input_signature make
        the training progress slower on my experiment. Note that input_signature
        is apply on based_trainer by default.
        """
        # from tensorflow_tts.optimizers import GradientAccumulator
        if self._already_apply_input_signature is False:
            self.one_step_forward = tf.function(
                self._one_step_forward, experimental_relax_shapes=True
            )
            self.one_step_evaluate = tf.function(
                self._one_step_evaluate, experimental_relax_shapes=True
            )
            self.one_step_predict = tf.function(
                self._one_step_predict, experimental_relax_shapes=True
            )
            self._already_apply_input_signature = True

        # run one_step_forward
        self.one_step_forward(batch)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _calculate_gradient_per_batch(self, batch):
        outputs = self._model(**batch, training=True)
        loss = self.compute_per_example_losses(batch, outputs)
        gradients, variables = zip(*self._optimizer.compute_gradients(loss))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, variables))

    def compute_per_example_losses(self, batch, outputs):
        """Compute per example losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model

        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        mgc_output, postnet_v2_mgc_output, stop_token, alignments = (
            outputs.mgc_output,
            outputs.postnet_v2_mgc_output,
            outputs.stop_token, outputs.alignments
        )

        # loss mask
        spec_loss_mask = tf.ones(shape=batch['padded_target_length'], dtype=tf.float32)
        binary_loss_mask = tf.ones(
            shape=np.floor(batch['padded_target_length'] / self.config['tacotron2_params']['outputs_per_step']),
            dtype=tf.float32
        )

        mel_loss = self.mel_loss(mgc_output, batch['mel_gts'], sample_weight=spec_loss_mask)
        done_loss = self.done_loss(stop_token, batch['done'], sample_weight=binary_loss_mask)

        regularization_loss = 0

        if self.config['use_l2_regularization']:
            # TODO: FixMe
            blacklist = [
                "embedding",
                "bias",
                "batch_normalization",
                "output_projection_wrapper/kernel",
                "lstm_cell",
                "output_and_stop_token_wrapper/dense/",
                "output_and_stop_token_wrapper/dense_1/"
            ]
            # regularization_loss = l2_regularization_loss(
            #     tf.trainable_variables(),
            #     self.config['l2_regularization_weight'], blacklist
            # )

        postnet_v2_mel_loss = self.postnet_v2_mel_loss(
            postnet_v2_mgc_output,
            batch['mel_gts'],
            sample_weight=spec_loss_mask
        ) if self.config['tacotron2_params']['use_postnet_v2'] else 0

        loss = mel_loss + done_loss + regularization_loss + postnet_v2_mel_loss
        return loss

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # predict with tf.function for faster.
        outputs = self.one_step_predict(batch)
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs
        mel_gts = batch["mel_gts"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = decoder_output.values[0].numpy()
            mels_after = mel_outputs.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            alignment_historys = alignment_historys.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception as e:
            mels_before = decoder_output.numpy()
            mels_after = mel_outputs.numpy()
            mel_gts = mel_gts.numpy()
            alignment_historys = alignment_historys.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_before, mel_after, alignment_history) in enumerate(
            zip(mel_gts, mels_before, mels_after, alignment_historys), 0
        ):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_before = tf.reshape(mel_before, (-1, 80)).numpy()  # [length, 80]
            mel_after = tf.reshape(mel_after, (-1, 80)).numpy()  # [length, 80]

            # plot figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title(f"Predicted Mel-before-Spectrogram @ {self.steps} steps")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title(f"Predicted Mel-after-Spectrogram @ {self.steps} steps")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # plot alignment
            figname = os.path.join(dirname, f"{idx}_alignment.png")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment_history, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, ax=ax)
            xlabel = "Decoder timestep"
            plt.xlabel(xlabel)
            plt.ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()


class ExtendedTacotronV1Trainer(SelfAttentionTacotronTrainer):
    def compile(self, model: ExtendedTacotronV1Model, optimizer: AdamWeightDecay):
        super().compile(model, optimizer)
        self.mel_loss = self._spec_loss[self.config.spec_loss_type]
        self.done_loss = BinaryCrossentropy()

        self.regularization_loss = 0
        if self.config.use_l2_regularization:
            # TODO: FixMe
            blacklist = [
                "embedding",
                "bias",
                "batch_normalization",
                "output_projection_wrapper/kernel",
                "lstm_cell",
                "output_and_stop_token_wrapper/dense/",
                "output_and_stop_token_wrapper/dense_1/"
            ]
            # self.regularization_loss = l2_regularization_loss(
            #     model.postnet.trainable_variables,
            #     self.config.l2_regularization_weight, blacklist
            # )

        self.postnet_v2_mel_loss = 0
        if self.config.use_postnet_v2:
            self.postnet_v2_mel_loss = self._spec_loss[self.config.spec_loss_type]


class DualSourceSelfAttentionTacotronTrainer(SelfAttentionTacotronTrainer):
    def compile(self, model: DualSourceSelfAttentionTacotronModel, optimizer: AdamWeightDecay):
        super().compile(model, optimizer)
        self.mel_loss = self._spec_loss[self.config.spec_loss_type]
        self.done_loss = BinaryCrossentropy()

        self.regularization_loss = None
        if self.config.use_l2_regularization:
            # TODO: FixMe
            blacklist = [
                "embedding",
                "bias",
                "batch_normalization",
                "output_projection_wrapper/kernel",
                "lstm_cell",
                "output_and_stop_token_wrapper/dense/",
                "output_and_stop_token_wrapper/dense_1/",
                "stop_token_projection/kernel"
            ]
            # self.regularization_loss = l2_regularization_loss(
            #     model.postnet.trainable_variables,
            #     self.config.l2_regularization_weight, blacklist
            # )

        self.postnet_v2_mel_loss = None
        if self.config.use_postnet_v2:
            self.postnet_v2_mel_loss = self._spec_loss[self.config.spec_loss_type]

        self.loss = self.mel_loss + self.done_loss + self.regularization_loss + self.postnet_v2_mel_loss


class DualSourceSelfAttentionMgcLf0TacotronTrainer(SelfAttentionTacotronTrainer):
    def compile(self, model: DualSourceSelfAttentionMgcLf0TacotronModel, optimizer: AdamWeightDecay):
        super().compile(model, optimizer)

        self.mgc_loss = self._spec_loss[self.config.spec_loss_type]
        self.lf0_loss = CategoricalCrossentropy()
        self.done_loss = BinaryCrossentropy()

        self.postnet_v2_mgc_loss = None
        if self.config.use_postnet_v2:
            self.postnet_v2_mgc_loss = self._spec_loss[self.config.spec_loss_type]

        self.loss = self.mgc_loss + self.lf0_loss * self.config.lf0_loss_factor + \
            self.done_loss + self.postnet_v2_mgc_loss


class MgcLf0TacotronTrainer(SelfAttentionTacotronTrainer):
    def compile(self, model: MgcLf0TacotronModel, optimizer: AdamWeightDecay):
        super().compile(model, optimizer)

        self.mgc_loss = self._spec_loss[self.config.spec_loss_type]
        self.lf0_loss = CategoricalCrossentropy()
        self.done_loss = BinaryCrossentropy()

        self.postnet_v2_mgc_loss = None
        if self.config.use_postnet_v2:
            self.postnet_v2_mgc_loss = self._spec_loss[self.config.spec_loss_type]

        self.loss = self.mgc_loss + self.lf0_loss * self.config.lf0_loss_factor + \
            self.done_loss + self.postnet_v2_mgc_loss


_TACO_TRAINER = {
    'extendedv1': trainer_model_cls(
        ExtendedTacotronV1Model,
        ExtendedTacotronV1Trainer
    ),

    'mgclf0': trainer_model_cls(
        MgcLf0TacotronModel,
        MgcLf0TacotronTrainer
    ),

    'dualsource_atten': trainer_model_cls(
        DualSourceSelfAttentionTacotronModel,
        DualSourceSelfAttentionTacotronTrainer
    ),

    'dualsource_atten_mgclf0': trainer_model_cls(
        DualSourceSelfAttentionMgcLf0TacotronModel,
        DualSourceSelfAttentionMgcLf0TacotronTrainer
    )
}


@click.command(description="Train FastSpeech (See detail in tensorflow_tts/bin/train-fastspeech.py)")
@click.option(
    "--train-dir",
    default=None,
    type=str,
    help="directory including training data. ",
)
@click.option(
    "--dev-dir",
    default=None,
    type=str,
    help="directory including development data. ",
)
@click.option(
    "--taco_name",
    default="extendedv1",
    type=str,
    choices=_TACO_TRAINER.keys(),
    help="Tacotron trainer"
)
@click.option(
    "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
)
@click.option(
    "--outdir", type=str, required=True, help="directory to save checkpoints."
)
@click.option(
    "--config", type=str, required=True, help="yaml format configuration file."
)
@click.option(
    "--resume",
    default=[],
    type=str,
    nargs="?",
    help='checkpoint file path to resume training. (default=[])',
)
@click.option(
    "--verbose",
    type=int,
    default=1,
    help="logging level. higher is more logging. (default=1)",
)
@click.option(
    "--mixed_precision",
    default=0,
    type=int,
    help="using mixed precision for generator or not.",
)
@click.option(
    "--pretrained",
    default=[],
    type=str,
    nargs="?",
    help="pretrained weights .h5 file to load weights from. Auto-skips non-matching layers",
)
def main(
    train_dir: str, dev_dir: str, taco_name: str,
    use_norm: int, outdir: str, config: str, resume: List[str],
    verbose: int, mixed_precision: int, pretrained: List[str]
):
    STRATEGY = return_strategy()
    # check directory existence
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # check arguments
    if train_dir is None:
        raise ValueError("Please specify --train-dir")
    if dev_dir is None:
        raise ValueError("Please specify --valid-dir")

    # load and save config
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update({})
    config["version"] = deepinsight_speech.__version__

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["mel_length_threshold"]
    else:
        mel_length_threshold = 0

    if config["format"] == "npy":
        charactor_query = "*-ids.npy"
        mel_query = "*-raw-feats.npy" if use_norm is False else "*-norm-feats.npy"
        charactor_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy are supported.")

    train_dataset = CharactorMelDataset(
        dataset=config["tacotron2_params"]["dataset"],
        root_dir=train_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        use_fixed_shapes=config["use_fixed_shapes"],
    )

    # update max_mel_length and max_char_length to config
    config.update({"max_mel_length": int(train_dataset.max_mel_length)})
    config.update({"max_char_length": int(train_dataset.max_char_length)})

    with open(os.path.join(outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    train_dataset = train_dataset.create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=(config["batch_size"] * STRATEGY.num_replicas_in_sync * config["gradient_accumulation_steps"]),
    )

    valid_dataset = CharactorMelDataset(
        dataset=config["tacotron2_params"]["dataset"],
        root_dir=dev_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        use_fixed_shapes=False,  # don't need apply fixed shape for evaluation.
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    # ##===================================
    # NOTE: Uncomment to debug dataset here
    # ##===================================
    iterator = train_dataset.batch(1).as_numpy_iterator()
    data = iterator.next()

    model_cls, trainer_cls = _TACO_TRAINER[taco_name].model, _TACO_TRAINER[taco_name].trainer
    trainer = trainer_cls(
        config=config,
        strategy=STRATEGY,
        steps=0,
        epochs=0,
        is_mixed_precision=mixed_precision,
    )  # type: Seq2SeqBasedTrainer

    with STRATEGY.scope():
        # define model.
        tacotron_config = Tacotron2Config(**config["tacotron2_params"])
        tacotron2 = model_cls(config=tacotron_config, name="selfattention-tacotron2")  # type: TacotronAttentionModel
        tacotron2._build()
        tacotron2.summary()

        if len(pretrained) > 1:
            tacotron2.load_weights(pretrained, by_name=True, skip_mismatch=True)
            logging.info(
                f"Successfully loaded pretrained weight from {pretrained}."
            )

        # AdamW for tacotron2
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
            decay_steps=config["optimizer_params"]["decay_steps"],
            end_learning_rate=config["optimizer_params"]["end_learning_rate"],
        )

        learning_rate_fn = WarmUp(
            initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=int(
                config["train_max_steps"] * config["optimizer_params"]["warmup_proportion"]
            ),
        )

        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=config["optimizer_params"]["weight_decay"],
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )

        _ = optimizer.iterations

    # compile trainer
    trainer.compile(model=tacotron2, optimizer=optimizer)

    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(config["outdir"], "checkpoints/"),
            resume=resume,
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
