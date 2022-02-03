import argparse
import glob
import os
import easydict
import collections
import csv
from pathlib import Path


class speaker_info(collections.namedtuple(
    'speaker_info', ['speaker_id', 'gender', 'name', 'subset'])
):
    pass


SRC_DIRECTORY = None
TEXT_MAP = collections.defaultdict(str)
SPEAKER_MAP = collections.defaultdict(speaker_info)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str, help="Source directory")
    parser.add_argument("--in_fname", default="speaker-info.csv", type=str, help="Input csv filename")
    parser.add_argument("--out_fname", default="metadata.csv", type=str, help="Output csv filename")
    argv = parser.parse_args()
    return argv


def prepare_row(wav):
    wavfname = Path(wav).relative_to(SRC_DIRECTORY)
    file_id = str(wavfname.stem)
    speaker_id = file_id.split("-")[0]
    if file_id not in TEXT_MAP:
        return None
    
    output = easydict.EasyDict({
        "wave_file": str(wavfname),
        "text": TEXT_MAP[file_id],
        "speaker_id": speaker_id,
        "speaker_name": SPEAKER_MAP[speaker_id].name,
        "speaker_gender": SPEAKER_MAP[speaker_id].gender,
    })
    return output


def main():
    global SRC_DIRECTORY, TEXT_MAP, SPEAKER_MAP
    argv = parse_args()
    SRC_DIRECTORY = Path(argv.source_dir)
    datadirs = ["train-clean-100", "dev-clean", "test-clean"]
    
    assert all(map(lambda subdir: os.path.exists(
        SRC_DIRECTORY.joinpath(subdir)), datadirs)), FileExistsError("Not all file exists")
    textfiles = []
    wavs = []
    
    for directory in datadirs:
        textfiles += glob.glob(str(SRC_DIRECTORY.joinpath(
            f"{directory}/**/**/*.txt")))
        
        wavs += glob.glob(str(SRC_DIRECTORY.joinpath(
            f"{directory}/**/**/*.flac")))

    with open(SRC_DIRECTORY.joinpath(argv.in_fname), 'r') as fp:
        next(fp)

        for line in fp:
            row = [col.strip() for col in line.strip().split("|")]
            SPEAKER_MAP[row[0]] = speaker_info(
                speaker_id=row[0],
                gender=row[1],
                name=row[4],
                subset=row[2]
            )

    for textfp in textfiles:
        speaker_id = Path(textfp).stem.split("-")[0]
        if Path(textfp).exists() and speaker_id in SPEAKER_MAP:
            for line in open(textfp).readlines():
                line = line.strip().split()
                TEXT_MAP[line[0]] = " ".join(line[1:])

    rows = [prepare_row(wav) for wav in wavs]
    rows = [row for row in rows if row]

    keys = rows[0].keys()
    out_fname = SRC_DIRECTORY.joinpath(argv.out_fname)
    with open(out_fname, 'w') as fp:
        dict_writer = csv.DictWriter(fp, keys)
        dict_writer.writeheader()
        dict_writer.writerows(rows)


if __name__ == "__main__":
    main()
