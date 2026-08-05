import numpy as np
from pathlib import Path

# ---------------- CONFIG ---------------- #

from configs.config import (
    SYNCHRONIZED_OUTPUT,
    SEQUENCE_OUTPUT,
    SEQUENCE_LENGTH
)

SYNC_ROOT = SYNCHRONIZED_OUTPUT

OUTPUT_ROOT = SEQUENCE_OUTPUT

STRIDE = 1

# ---------------------------------------- #


def build_sequences(video_folder: Path):
    """
    Build fixed-length sequences from synchronized frame features.
    """

    feature_files = sorted(video_folder.glob("*.npy"))

    if len(feature_files) < SEQUENCE_LENGTH:
        print(f"⚠️ Not enough frames in {video_folder.name}")
        return 0

    features = []

    for file in feature_files:
        features.append(np.load(file))

    features = np.array(features, dtype=np.float32)

    save_folder = OUTPUT_ROOT / video_folder.name
    save_folder.mkdir(parents=True, exist_ok=True)

    sequence_count = 0

    for start in range(
        0,
        len(features) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):

        sequence = features[
            start:start + SEQUENCE_LENGTH
        ]

        output_file = (
            save_folder /
            f"sequence_{sequence_count:04d}.npy"
        )

        np.save(output_file, sequence)

        sequence_count += 1

    print(f"✅ {video_folder.name}")
    print(f"   Sequences Created : {sequence_count}")

    return sequence_count


def main():

    total_sequences = 0
    total_videos = 0

    for video_folder in sorted(SYNC_ROOT.iterdir()):

        if not video_folder.is_dir():
            continue

        total_videos += 1

        total_sequences += build_sequences(video_folder)

    print("\n==============================")
    print(f"Videos Processed  : {total_videos}")
    print(f"Sequences Created : {total_sequences}")
    print("==============================")


if __name__ == "__main__":
    main()