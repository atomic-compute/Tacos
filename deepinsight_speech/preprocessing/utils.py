import os
import numpy as np
import tensorflow as tf
from deepinsight_speech.utils.tfrecord import write_tfrecord, int64_feature, bytes_feature


def save_features_to_file(features, subdir, config):
    """Save transformed dataset features in disk.
    Args:
        features (Dict): dictionary containing the attributes to save.
        subdir (str): data split folder where features will be saved.
        config (Dict): configuration dictionary.
    """
    utt_id = features["utt_id"]

    if config["format"] == "npy":
        save_list = [
            (features["audio"], "wavs", "wave", np.float32),
            (features["mel"], "raw-feats", "raw-feats", np.float32),
            (features["text_ids"], "ids", "ids", np.int32),
            (features["f0"], "raw-f0", "raw-f0", np.float32),
            (features["energy"], "raw-energies", "raw-energy", np.float32),
        ]
        for item, name_dir, name_file, fmt in save_list:
            np.save(
                os.path.join(
                    config["outdir"], subdir, name_dir, f"{utt_id}-{name_file}.npy"
                ),
                item.astype(fmt),
                allow_pickle=False,
            )
    else:
        raise ValueError("'npy' is the only supported format.")


def write_preprocessed_target_data(_id: int, key: str, mel: np.ndarray, filename: str):
    raw_mel = mel.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([_id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'mel': bytes_feature([raw_mel]),
        'target_length': int64_feature([len(mel)]),
        'mel_width': int64_feature([mel.shape[1]]),
    }))
    write_tfrecord(example, filename)


def write_preprocessed_source_data(_id: int, key: str, source: np.ndarray, text, speaker_id, age, gender,
                                   filename: str):
    raw_source = source.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([_id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'source': bytes_feature([raw_source]),
        'source_length': int64_feature([len(source)]),
        'text': bytes_feature([text.encode('utf-8')]),
        'speaker_id': int64_feature([speaker_id]),
        'age': int64_feature([age]),
        'gender': int64_feature([gender]),
    }))
    write_tfrecord(example, filename)
