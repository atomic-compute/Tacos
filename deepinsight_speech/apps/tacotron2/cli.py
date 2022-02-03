import sys
import click
from deepinsight_speech.apps.tacotron2 import train_tacotron2
from deepinsight_speech.apps.tacotron2 import decode_tacotron2
from deepinsight_speech.apps.tacotron2 import extract_duration


@click.group()
def main():
    return 0

# TODO: Add sub-command into group for argparser module
# main.command(train_tacotron2.main, "train")
# main.command(extract_duration.main, "extract_duration")
# main.command(decode_tacotron2.main, "decode")


if __name__ == "__main__":
    sys.exit(main())
