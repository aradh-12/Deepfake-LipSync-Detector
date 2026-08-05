import librosa
import numpy as np
from pathlib import Path

from configs.config import (
    AUDIO_OUTPUT,
    MFCC_OUTPUT
)


def extract_mfcc(audio_file: Path, output_file: Path):
    """
    Extract MFCC features from one audio file.
    """

    y, sr = librosa.load(audio_file, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, mfcc)

    return mfcc.shape


def process_all_audio(input_root: Path, output_root: Path):

    audio_files = sorted(input_root.glob("*/*.wav"))

    if not audio_files:
        print("❌ No WAV files found.")
        return

    processed = 0

    print(f"\nFound {len(audio_files)} audio file(s).\n")

    for audio in audio_files:

        video_name = audio.parent.name

        output_file = (
            output_root /
            video_name /
            f"{video_name}.npy"
        )

        shape = extract_mfcc(audio, output_file)

        processed += 1

        print(f"✅ {audio.name}")
        print(f"   MFCC Shape : {shape}")
        print(f"   Saved -> {output_file}")

    print("\n==============================")
    print(f"Audio Files Processed : {processed}")
    print("==============================")


if __name__ == "__main__":

    process_all_audio(
        AUDIO_OUTPUT,
        MFCC_OUTPUT
    )