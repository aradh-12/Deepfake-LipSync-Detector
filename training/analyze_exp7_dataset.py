from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

DATASET_FOLDER = Path(
    "outputs/dataset_exp7_source_aware"
)


# ============================================================
# Analyze Split
# ============================================================

def analyze_split(name, file_path):

    data = np.load(
        file_path,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]

    unique_videos = np.unique(
        video_ids
    )

    real_videos = 0
    fake_videos = 0

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Sequences:",
        len(X)
    )

    print(
        "Videos:",
        len(unique_videos)
    )

    print(
        "Real sequences:",
        np.sum(y == 0)
    )

    print(
        "Fake sequences:",
        np.sum(y == 1)
    )

    # --------------------------------------------------------
    # Count video labels
    # --------------------------------------------------------

    for video in unique_videos:

        indices = np.where(
            video_ids == video
        )[0]

        labels = np.unique(
            y[indices]
        )

        if len(labels) != 1:

            raise RuntimeError(
                f"Multiple labels for: "
                f"{video}"
            )

        if labels[0] == 0:

            real_videos += 1

        else:

            fake_videos += 1

    print("\nVIDEO-LEVEL DISTRIBUTION")

    print(
        "Real videos:",
        real_videos
    )

    print(
        "Fake videos:",
        fake_videos
    )

    print("\nSEQUENCES PER VIDEO")

    for video in unique_videos:

        indices = np.where(
            video_ids == video
        )[0]

        label = y[indices[0]]

        label_name = (
            "Fake"
            if label == 1
            else "Real"
        )

        print(
            f"{video} | "
            f"{label_name} | "
            f"{len(indices)} sequences"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("EXP7 DATASET ANALYSIS")
    print("=" * 60)

    analyze_split(

        "TRAIN",

        DATASET_FOLDER /
        "train.npz"
    )

    analyze_split(

        "VALIDATION",

        DATASET_FOLDER /
        "validation.npz"
    )

    analyze_split(

        "TEST",

        DATASET_FOLDER /
        "test.npz"
    )

    print("\n" + "=" * 60)
    print("EXP7 DATASET ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()
