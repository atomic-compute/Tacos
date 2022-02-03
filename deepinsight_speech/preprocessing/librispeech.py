"""Perform preprocessing and raw feature extraction for LibreSpeech dataset.
"""

import re
import csv
import tensorflow as tf
import numpy as np
import os
from dataclasses import dataclass
import soundfile as sf
from g2p_en import g2p as grapheme_to_phonem
from tensorflow_tts.utils import cleaners
from tensorflow_tts.processor import BaseProcessor
from tensorflow_tts.utils.utils import PROCESSOR_FILE_NAME


g2p = grapheme_to_phonem.G2p()

valid_symbols = g2p.phonemes
valid_symbols.append("SIL")
valid_symbols.append("END")

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]

_pad = "pad"
_eos = "eos"
_punctuation = "!'(),.:;? "
# _punctuation = '!\'\"()[],-.:;?` '
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_characters = _letters + _punctuation

# Export all symbols:
LIBRISPEECH_SYMBOLS = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + [_eos]
)

symbols = list(_characters)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


@dataclass
class LibriSpeechProcessor(BaseProcessor):
    """LJSpeech processor."""

    mode: str = "inference"
    cleaner_names: str = "english_cleaners"
    positions = {
        "wave_file": 0,
        "text": 1,
        "speaker_name": 3,
    }
    train_f_name: str = "metadata.csv"
    f_extension: str = ".flac"

    def create_items(self):
        if self.data_dir:
            fieldnames = ["wave_file", "text", "speaker_id", "speaker_name", "speaker_gender"]
            with open(os.path.join(self.data_dir, self.train_f_name), encoding="utf-8"
                      ) as f:
                next(f)
                reader = csv.DictReader(f, fieldnames)
                self.items = [self.split_line(self.data_dir, line) for line in reader]

    def split_line(self, data_dir, parts):
        wave_file = parts["wave_file"]
        text_norm = parts["text"]
        wav_path = os.path.join(data_dir, f"{wave_file}")
        speaker_name = parts['speaker_name']
        return text_norm, wav_path, speaker_name

    def setup_eos_token(self):
        return _eos

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, PROCESSOR_FILE_NAME), {})

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_path)[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def setup_eos_token(self):
        return None  # because we do not use this

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, PROCESSOR_FILE_NAME), {})

    def text_to_sequence(self, text):
        if self.mode == "train":  # in train mode text should be already transformed to phonemes
            return self.symbols_to_ids(self.clean_g2p(self.text_to_ph(text.split(" "))))
        else:
            return self.inference_text_to_seq(text)

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

    def text_to_ph(self, text: str):
        return self.clean_g2p(g2p(text))

    def clean_g2p(self, g2p_text: list):
        data = []
        for i, txt in enumerate(g2p_text):
            if i == len(g2p_text) - 1:
                if txt != " " and txt != "SIL":
                    data.append("@" + txt)
                else:
                    # TODO try learning without end token and compare results
                    data.append("@END")
                break
            if txt != " ":
                data.append("@" + txt)
        return data
