import argparse
import glob
import os
import csv
import re
import easydict
from collections import defaultdict, namedtuple
from pathlib import Path


class speaker_info(
    namedtuple('speaker_info', ['speaker_id', 'age', 'gender', 'accent'])
):
    pass


SPEAKER_MAP = defaultdict(speaker_info)
TEXT_MAP = defaultdict(str)
SRC_DIRPATH = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str, help="Source directory")
    parser.add_argument("--in_fname", default="speaker-info.txt", type=str, help="Input csv filename")
    parser.add_argument("--out_fname", default="metadata.csv", type=str, help="Output csv filename")
    argv = parser.parse_args()
    return argv


def prepare_row(wav):
    ac_wavefname = Path(wav).name
    wavfname = Path(wav).relative_to(SRC_DIRPATH)
    file_id, mic_num = str(wavfname.stem).rsplit("_", 1)
    speaker_id = wavfname.parent.name
    # If mic2 is present and mic1 is also present -> return
    if file_id not in TEXT_MAP:
        return None

    if mic_num == "mic2" and (
        Path(wav).parent.
        joinpath(ac_wavefname.replace("mic2", "mic1")).exists()
    ):
        return None

    output = easydict.EasyDict({
        "wave_file": str(wavfname),
        "text": TEXT_MAP[file_id],
        "speaker_name": speaker_id,
        "speaker_age": SPEAKER_MAP[speaker_id].age,
        "speaker_gender": SPEAKER_MAP[speaker_id].gender,
        "speaker_accent": SPEAKER_MAP[speaker_id].accent
    })
    return output


def main():
    global SPEAKER_MAP, TEXT_MAP, SRC_DIRPATH
    argv = parse_args()
    SRC_DIRPATH = Path(argv.source_dir)
    textfiles = glob.glob(str(SRC_DIRPATH / "txt/**/*.txt"))
    wavs = glob.glob(str(SRC_DIRPATH / "wav48_silence_trimmed/**/*.flac"))
    inp_fname = SRC_DIRPATH / argv.in_fname  # type: Path
    with inp_fname.open('r') as fp:

        next(fp)

        for line in fp:
            row = line.strip().split()
            SPEAKER_MAP[row[0]] = speaker_info(
                speaker_id=row[0], age=row[1], gender=row[2], accent=row[3]
            )

    for textfp in textfiles:
        textfp = Path(textfp)
        if textfp.exists() and textfp.parent.name in SPEAKER_MAP:
            fileid = os.path.basename(textfp).split(".")[0]
            TEXT_MAP[fileid] = open(textfp).readline().strip()

    rows = [prepare_row(wav) for wav in wavs]
    rows = [row for row in rows if row]
    keys = rows[0].keys()
    
    with Path(argv.source_dir, argv.out_fname).open('w') as fp:
        dict_writer = csv.DictWriter(fp, keys)
        dict_writer.writeheader()
        dict_writer.writerows(rows)


if __name__ == "__main__":
    main()
