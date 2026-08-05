import numpy as np
from pathlib import Path

from .utils import (
    load_lip_coordinates,
    load_mfcc,
    save_feature_vector
)

from configs.config import (
    LIP_COORDINATE_OUTPUT,
    MFCC_OUTPUT,
    SYNCHRONIZED_OUTPUT
)


def synchronize_video(
    lip_folder: Path,
    mfcc_file: Path,
    output_folder: Path
):
    """
    Synchronize each lip frame with its corresponding MFCC frame.
    """

    frame_files = sorted(lip_folder.glob("*.csv"))

    if not frame_files:
        print(f"❌ No CSV files found in {lip_folder.name}")
        return 0

    mfcc = load_mfcc(mfcc_file)

    output_folder.mkdir(parents=True, exist_ok=True)

    synchronized = 0
    total_mfcc_frames = mfcc.shape[1]

    for index, frame in enumerate(frame_files):

        lips = load_lip_coordinates(frame)

        # Flatten lip coordinates (40 x 2 -> 80 values)
        lip_vector = lips.astype(np.float32).flatten()

        # Match each frame with an MFCC frame
        mfcc_index = min(index, total_mfcc_frames - 1)
        mfcc_vector = mfcc[:, mfcc_index].astype(np.float32)

        # Final feature vector (80 + 13 = 93 values)
        feature_vector = np.concatenate(
            (lip_vector, mfcc_vector)
        ).astype(np.float32)

        output_file = output_folder / f"{frame.stem}.npy"

        save_feature_vector(feature_vector, output_file)

        synchronized += 1

    print(f"✅ {lip_folder.name}")
    print(f"   Frames Synchronized : {synchronized}")

    return synchronized


def process_all_videos():

    lip_root = LIP_COORDINATE_OUTPUT
    mfcc_root = MFCC_OUTPUT
    output_root = SYNCHRONIZED_OUTPUT

    total_frames = 0
    total_videos = 0

    for video_folder in sorted(lip_root.iterdir()):

        if not video_folder.is_dir():
            continue

        video_name = video_folder.name

        mfcc_file = mfcc_root / video_name / f"{video_name}.npy"

        if not mfcc_file.exists():
            print(f"❌ Missing MFCC for {video_name}")
            continue

        total_videos += 1

        output_folder = output_root / video_name

        total_frames += synchronize_video(
            video_folder,
            mfcc_file,
            output_folder
        )

    print("\n==============================")
    print(f"Videos Processed : {total_videos}")
    print(f"Frames Synced    : {total_frames}")
    print("==============================")


if __name__ == "__main__":
    process_all_videos()