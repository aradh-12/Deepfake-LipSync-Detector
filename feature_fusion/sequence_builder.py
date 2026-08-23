from pathlib import Path
import numpy as np
import pandas as pd



LIP_FOLDER = Path("outputs/lip_coordinates")
MFCC_FOLDER = Path("outputs/mfcc")
OUTPUT_FOLDER = Path("outputs/sequences")


def process_all_videos():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        video
        for video in LIP_FOLDER.iterdir()
        if video.is_dir()
        and video.name.startswith("FakeAVCeleb_")
    )

    print(f"\nFound {len(videos)} videos.\n")

    for video in videos:

        print(f"Processing : {video.name}")

        csv_files = sorted(video.glob("*.csv"))

        lip_sequence = []

        for csv_file in csv_files:

            df = pd.read_csv(csv_file)

            coordinates = (
                df[["X", "Y"]]
                .values
                .flatten()
            )

            lip_sequence.append(coordinates)

        lip_sequence = np.array(
            lip_sequence,
            dtype=np.float32
        )

        mfcc_name = video.name.replace(
            "FakeAVCeleb_id00076_",
            "",
            1
        )

        mfcc_file = (
            MFCC_FOLDER
            / mfcc_name
            / f"{mfcc_name}.npy"
        )

        if not mfcc_file.exists():

            print(
                f"⚠️ Missing MFCC: {mfcc_file}"
            )

            print(
                f"Skipping : {video.name}"
            )

            print("-" * 40)

            continue

        mfcc = np.load(mfcc_file)

        output_file = (
            OUTPUT_FOLDER
            / f"{video.name}.npz"
        )

        np.savez(
            output_file,
            lip=lip_sequence,
            mfcc=mfcc
        )

        print(
            f"Saved : {output_file.name}"
        )

        print("-" * 40)


if __name__ == "__main__":
    process_all_videos()