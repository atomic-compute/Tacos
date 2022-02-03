import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_alignment(alignments, text, _id, global_step, path):
    num_alignment = len(alignments)
    fig = plt.figure(figsize=(12, 16))
    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_alignment, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(path, format='png')
    plt.close()


def plot_mel(mel, mel_predicted, text, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(mel.T, origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_spec(spec, spec_predicted, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(spec.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(spec_predicted[:spec.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_predictions(alignments, mel, mel_predicted, spec, spec_predicted, text, _id, filename):
    fig = plt.figure(figsize=(12, 24))

    num_alignment = len(alignments)
    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_alignment + 4, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 1)
    im = ax.imshow(mel.T, origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)

    if spec is not None and spec_predicted is not None:
        ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 3)
        im = ax.imshow(spec.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax)
        ax = fig.add_subplot(num_alignment + 4, 1, num_alignment + 4)
        im = ax.imshow(spec_predicted[:spec.shape[0], :].T,
                       origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"record ID: {_id}\ninput text: {str(text)}")

    fig.savefig(filename, format='png')
    plt.close()


def plot_predictions(alignments, mel, mel_predicted, mel_predicted_postnet, text, key, filename):
    from matplotlib import pylab as plt
    num_alignment = len(alignments)
    num_rows = num_alignment + 3
    if mel_predicted_postnet is not None:
        num_rows += 1
    fig = plt.figure(figsize=(14, num_rows * 3))

    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_rows, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap='jet')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))

    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    ax = fig.add_subplot(num_rows, 1, num_alignment + 1)
    im = ax.imshow(mel.T, origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(num_rows, 1, num_alignment + 2, sharex=ax)
    im = ax.imshow(mel_predicted.T,
                   origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)

    if mel_predicted_postnet is not None:
        ax = fig.add_subplot(num_rows, 1, num_alignment + 3, sharex=ax)
        im = ax.imshow(mel_predicted_postnet.T,
                       origin="lower bottom", aspect="auto", cmap="magma")
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"record ID: {key}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_mgc_lf0(mgc, mgc_predicted, lf0, lf0_predicted, text, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 20))
    ax = fig.add_subplot(4, 1, 1)
    im = ax.imshow(mgc.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=-4.0, vmax=4.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(4, 1, 2)
    im = ax.imshow(mgc_predicted[:mgc.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=-4.0, vmax=4.0)
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(4, 1, 3)
    im = ax.imshow(lf0.T, origin="lower bottom", aspect="auto", cmap="binary", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(4, 1, 4)
    im = ax.imshow(lf0_predicted[:mgc.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="binary", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


class MgcLf0MetricsSaver(tf.keras.callbacks.Callback):
    
    def __init__(
            self,
            alignment_tensors,
            global_step_tensor,
            predicted_mgc_tensor,
            ground_truth_mgc_tensor,
            predicted_lf0_tensor,
            ground_truth_lf0_tensor,
            target_length_tensor,
            id_tensor,
            text_tensor,
            save_steps,
            mode, hparams, writer
    ):
        self.alignment_tensors = alignment_tensors
        self.global_step_tensor = global_step_tensor
        self.predicted_mgc_tensor = predicted_mgc_tensor
        self.ground_truth_mgc_tensor = ground_truth_mgc_tensor
        self.predicted_lf0_tensor = predicted_lf0_tensor
        self.ground_truth_lf0_tensor = ground_truth_lf0_tensor
        self.target_length_tensor = target_length_tensor
        self.id_tensor = id_tensor
        self.text_tensor = text_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer

    def on_epoch_end(self, epoch, logs):
        # TODO: Fix this
        return super().on_epoch_end(epoch, logs=logs)

    def on_train_batch_end(self, batch, run_context, run_values, logs=None, ):
        # TODO: Fix this metric callback
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, alignments, predicted_mgcs, ground_truth_mgcs, predicted_lf0s, ground_truth_lf0s, target_length, ids, texts = run_context.session.run(
                (self.global_step_tensor, self.alignment_tensors, self.predicted_mgc_tensor,
                    self.ground_truth_mgc_tensor, self.predicted_lf0_tensor, self.ground_truth_lf0_tensor,
                    self.target_length_tensor, self.id_tensor, self.text_tensor))
            id_strings = ",".join([str(i) for i in ids])
            result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
            tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)

            alignments = [[a[i] for a in alignments] for i in range(alignments[0].shape[0])]
            for _id, text, align, pred_mgc, gt_mgc, pred_lf0, gt_lf0 in zip(ids, texts, alignments, predicted_mgcs,
                                                                            ground_truth_mgcs, predicted_lf0s,
                                                                            ground_truth_lf0s):
                output_filename = "{}_result_step{:09d}_{:d}.png".format(self.mode,
                                                                         global_step_value, _id)
                plot_alignment(align, text.decode('utf-8'), _id, global_step_value,
                               os.path.join(self.writer.get_logdir(), "alignment_" + output_filename))
                plot_mgc_lf0(gt_mgc, pred_mgc, gt_lf0, pred_lf0,
                             text.decode('utf-8'), _id, global_step_value,
                             os.path.join(self.writer.get_logdir(), "mgc_lf0_" + output_filename))


class MetricsSaver(tf.keras.callbacks.Callback):
    def __init__(
        self,
        alignment_tensors,
        global_step_tensor,
        predicted_mel_tensor,
        ground_truth_mel_tensor,
        mel_length_tensor,
        id_tensor,
        text_tensor,
        save_steps,
        mode, 
        writer,
        save_training_time_metrics=True,
        keep_eval_results_max_epoch=10
    ):
        self.alignment_tensors = alignment_tensors
        self.global_step_tensor = global_step_tensor
        self.predicted_mel_tensor = predicted_mel_tensor
        self.ground_truth_mel_tensor = ground_truth_mel_tensor
        self.mel_length_tensor = mel_length_tensor
        self.id_tensor = id_tensor
        self.text_tensor = text_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer
        self.save_training_time_metrics = save_training_time_metrics
        self.keep_eval_results_max_epoch = keep_eval_results_max_epoch
        self.checkpoint_pattern = re.compile('all_model_checkpoint_paths: "model.ckpt-(\d+)"')

    def on_epoch_end(self, epoch, logs):
        pass

    def on_train_batch_end(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, alignments, predicted_mels, ground_truth_mels, mel_length, ids, texts = run_context.session.run(
                (self.global_step_tensor, self.alignment_tensors, self.predicted_mel_tensor,
                 self.ground_truth_mel_tensor, self.mel_length_tensor, self.id_tensor, self.text_tensor))
            alignments = [a.astype(np.float32) for a in alignments]
            predicted_mels = [m.astype(np.float32) for m in list(predicted_mels)]
            ground_truth_mels = [m.astype(np.float32) for m in list(ground_truth_mels)]
            if self.mode == tf.estimator.ModeKeys.EVAL or self.save_training_time_metrics:
                id_strings = ",".join([str(i) for i in ids][:10])
                result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
                tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)
                write_training_result(global_step_value, list(ids), list(texts), predicted_mels,
                                      ground_truth_mels, list(mel_length),
                                      alignments,
                                      filename=os.path.join(self.writer.get_logdir(), result_filename))
            if self.mode == tf.estimator.ModeKeys.EVAL:
                alignments = [[a[i] for a in alignments] for i in range(alignments[0].shape[0])]
                for _id, text, align, pred_mel, gt_mel in zip(ids, texts, alignments, predicted_mels,
                                                              ground_truth_mels):
                    output_filename = "{}_result_step{:09d}_{:d}.png".format(self.mode,
                                                                             global_step_value, _id)
                    plot_alignment(align, text.decode('utf-8'), _id, global_step_value,
                                   os.path.join(self.writer.get_logdir(), "alignment_" + output_filename))
                    plot_mel(gt_mel, pred_mel, text.decode('utf-8'), _id, global_step_value,
                             os.path.join(self.writer.get_logdir(), "mel_" + output_filename))

    def end(self, session):
        current_global_step = session.run(self.global_step_tensor)
        with open(os.path.join(self.writer.get_logdir(), "checkpoint")) as f:
            checkpoints = [ckpt for ckpt in f]
            checkpoints = [self.extract_global_step(ckpt) for ckpt in checkpoints[1:]]
            checkpoints = list(filter(lambda gs: gs < current_global_step, checkpoints))
            if len(checkpoints) > self.keep_eval_results_max_epoch:
                checkpoint_to_delete = checkpoints[-self.keep_eval_results_max_epoch]
                tf.logging.info("Deleting %s results at the step %d", self.mode, checkpoint_to_delete)
                tfrecord_filespec = os.path.join(self.writer.get_logdir(),
                                                 "eval_result_step{:09d}_*.tfrecord".format(checkpoint_to_delete))
                alignment_filespec = os.path.join(self.writer.get_logdir(),
                                                  "alignment_eval_result_step{:09d}_*.png".format(
                    checkpoint_to_delete))
                mel_filespec = os.path.join(self.writer.get_logdir(),
                                            "mel_eval_result_step{:09d}_*.png".format(checkpoint_to_delete))
                for pathname in tf.gfile.Glob([tfrecord_filespec, alignment_filespec, mel_filespec]):
                    file_io.delete_file(pathname)

    def extract_global_step(self, checkpoint_str):
        return int(self.checkpoint_pattern.match(checkpoint_str)[1])


