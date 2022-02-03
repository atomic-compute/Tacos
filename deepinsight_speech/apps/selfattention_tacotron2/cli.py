import sys
import click
from deepinsight_speech.apps.selfattention_tacotron2 import train_selfatten_tacotron2
from deepinsight_speech.apps.selfattention_tacotron2 import decode_selfatten_tacotron2
from deepinsight_speech.apps.selfattention_tacotron2 import extract_duration


@click.group()
def main():
    return 0


main.command(train_selfatten_tacotron2.main, "train")
main.command(extract_duration.main, "extract_duration")
main.command(decode_selfatten_tacotron2.main, "decode")


if __name__ == "__main__":
    sys.exit(main())
