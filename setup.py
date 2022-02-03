from setuptools import setup, find_packages


setup(
    name="deepinsight-mystic-sound",
    entry_points={
        'console_script': [
            "insight2s-processing=deepinsight_speech.preprocessing.preprocess:preprocess",
            "insight2s-processing-compute-statistics=deepinsight_speech.preprocessing.preprocess:compute_statistics",
            "insight2s-processing-normalize=deepinsight_speech.preprocessing.preprocess:normalize",
            "insight2s-selfattention_tacotron2=deepinsight_speech.apps.selfattention_tacotron2:cli",
            "insight2s-tacotron2=deepinsight_speech.apps.tacotron2:cli",
            "insight2s-melgan=deepinsight_speech.apps.melgan:cli",
            "insight2s-fastspeech=deepinsight_speech.apps.fastspeech:cli",
            "insight2s-fastspeech2=deepinsight_speech.apps.fastspeech2:cli",
            "insight2s-melgan_stft=deepinsight_speech.apps.melgan_stft:cli",
            "insight2s-multiband_melgan=deepinsight_speech.apps.multiband_melgan:cli",
            "insight2s-multiband_pwgan=deepinsight_speech.apps.multiband_pwgan:cli",
            "insight2s-parallel_wavegen=deepinsight_speech.apps.parallel_wavegen:cli",
        ],
    },
    version="0.1.0",
    description="Synthetic Voice Cloning and Text to Speech Synthesis",
    long_description=open("README.md").read(),
    license='APACHE',
    long_description_content_type="text/markdown",
    author="Sandip Dey",
    author_email="sandip.dey1988@yahoo.com",
    packages=['deepinsight_speech'],
    include_package_data=True,
    install_requires=open("requirements.txt", 'r').readlines(),
    platforms=["linux", "unix"],
    python_requires=">3.5.2",
)
