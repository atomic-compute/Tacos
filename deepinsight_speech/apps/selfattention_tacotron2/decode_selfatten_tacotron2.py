import os
import yaml
import click
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from deepinsight_speech.apps.tacotron2.tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs import Tacotron2Config


@click.command(
    description="Decode mel-spectrogram from folder ids with trained Tacotron-2 "
    "(See detail in tensorflow_tts/example/tacotron2/decode_tacotron2.py)."
)
@click.option(
    "--rootdir",
    default=None,
    type=str,
    required=True,
    help="directory including ids/durations files.",
)
@click.option(
    "--outdir", type=str, required=True, help="directory to save generated speech."
)
@click.option(
    "--checkpoint", type=str, required=True, help="checkpoint file to be loaded."
)
@click.option(
    "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
)
@click.option("--batch-size", default=8, type=int, help="batch size.")
@click.option("--win-front", default=3, type=int, help="win-front.")
@click.option("--win-back", default=3, type=int, help="win-front.")
@click.option(
    "--config",
    default=None,
    type=str,
    required=True,
    help="yaml format configuration file. if not explicitly provided, "
    "it will be searched in the checkpoint directory. (default=None)",
)
@click.option(
    "--verbose",
    type=int,
    default=1,
    help="logging level. higher is more logging. (default=1)",
)
def main(
    rootdir: str, outdir: str, checkpoint: str, 
    use_norm: int, batch_size: int, win_front: int, 
    win_back: int, config: str, verbose: int
):
    # check directory existence
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # load config
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update({})

    if config["format"] == "npy":
        char_query = "*-ids.npy"
        mel_query = "*-raw-feats.npy" if use_norm is False else "*-norm-feats.npy"
        char_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    # define data-loader
    dataset = CharactorMelDataset(
        dataset=config["tacotron2_params"]["dataset"],
        root_dir=rootdir,
        charactor_query=char_query,
        mel_query=mel_query,
        charactor_load_fn=char_load_fn,
        mel_load_fn=mel_load_fn,
        reduction_factor=config["tacotron2_params"]["reduction_factor"]
    )
    dataset = dataset.create(allow_cache=True, batch_size=batch_size)
    # TODO: FixME
    # - Use different SATacotron based on configuration
    # define model and load checkpoint
    tacotron2 = TFTacotron2(
        config=Tacotron2Config(**config["tacotron2_params"]),
        name="tacotron2",
    )
    tacotron2._build()  # build model to be able load_weights.
    tacotron2.load_weights(checkpoint)

    # setup window
    tacotron2.setup_window(win_front=win_front, win_back=win_back)

    for data in tqdm(dataset, desc="[Decoding]"):
        utt_ids = data["utt_ids"]
        utt_ids = utt_ids.numpy()

        # tacotron2 inference.
        (
            mel_outputs,
            post_mel_outputs,
            stop_outputs,
            alignment_historys,
        ) = tacotron2.inference(
            input_ids=data["input_ids"], 
            input_lengths=data["input_lengths"], 
            speaker_ids=data["speaker_ids"],
        )

        # convert to numpy
        post_mel_outputs = post_mel_outputs.numpy()

        for i, post_mel_output in enumerate(post_mel_outputs):
            stop_token = tf.math.round(tf.nn.sigmoid(stop_outputs[i]))  # [T]
            real_length = tf.math.reduce_sum(
                tf.cast(tf.math.equal(stop_token, 0.0), tf.int32), -1
            )
            post_mel_output = post_mel_output[:real_length, :]

            saved_name = utt_ids[i].decode("utf-8")

            # save D to folder.
            np.save(
                os.path.join(outdir, f"{saved_name}-norm-feats.npy"),
                post_mel_output.astype(np.float32),
                allow_pickle=False,
            )


if __name__ == '__main__':
    main()
