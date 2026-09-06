from pathlib import Path

import numpy as np

from utils.multidataset_manager import (
    get_fakeavceleb_all
)


# ============================================================
# Configuration
# ============================================================

SEQUENCE_FOLDER = Path(
    "outputs/synchronized_aligned_exp5"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 199


# ============================================================
# Video Folder Name
# ============================================================

def get_video_folder_name(video):

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


# ============================================================
# Get ALL usable FakeAVCeleb videos
# ============================================================

def get_usable_videos():

    real_videos, fake_videos = (
        get_fakeavceleb_all()
    )

    dataset = []

    seen_video_ids = set()

    # --------------------------------------------------------
    # Real videos
    # --------------------------------------------------------

    for video in real_videos:

        video_id = (
            get_video_folder_name(video)
        )

        folder = (
            SEQUENCE_FOLDER /
            video_id
        )

        if not folder.exists():
            continue

        if video_id in seen_video_ids:

            print(
                f"⚠️ Duplicate skipped: "
                f"{video_id}"
            )

            continue

        seen_video_ids.add(video_id)

        dataset.append(
            (video, 0)
        )

    # --------------------------------------------------------
    # Fake videos
    # --------------------------------------------------------

    for video in fake_videos:

        video_id = (
            get_video_folder_name(video)
        )

        folder = (
            SEQUENCE_FOLDER /
            video_id
        )

        if not folder.exists():
            continue

        if video_id in seen_video_ids:

            print(
                f"⚠️ Duplicate skipped: "
                f"{video_id}"
            )

            continue

        seen_video_ids.add(video_id)

        dataset.append(
            (video, 1)
        )

    return dataset
# ============================================================
# Build Dataset
# ============================================================

def build_dataset():

    dataset = get_usable_videos()

    print("\n==============================")
    print("Usable FakeAVCeleb Dataset")
    print("==============================")

    print(
        "Total usable videos :",
        len(dataset)
    )

    print(
        "Real videos         :",
        sum(label == 0 for _, label in dataset)
    )

    print(
        "Fake videos         :",
        sum(label == 1 for _, label in dataset)
    )

    print("==============================")

    X = []
    y = []
    video_ids = []

    processed_videos = 0
    skipped_videos = 0

    # ========================================================
    # Process every usable video
    # ========================================================

    for video, label in dataset:

        video_folder_name = (
            get_video_folder_name(video)
        )

        video_folder = (
            SEQUENCE_FOLDER /
            video_folder_name
        )

        feature_files = sorted(
            video_folder.glob("*.npy")
        )

        if len(feature_files) < SEQUENCE_LENGTH:

            print(
                f"⚠️ Too few features: "
                f"{video_folder_name} "
                f"({len(feature_files)})"
            )

            skipped_videos += 1
            continue

        features = []

        for feature_file in feature_files:

            try:

                feature = np.load(
                    feature_file
                )

            except Exception as error:

                print(
                    f"❌ Cannot load: "
                    f"{feature_file}"
                )

                print(
                    f"   Error: {error}"
                )

                continue

            if feature.shape != (
                FEATURE_SIZE,
            ):

                print(
                    f"❌ Wrong feature shape: "
                    f"{feature_file}"
                )

                print(
                    f"   Expected: "
                    f"({FEATURE_SIZE},)"
                )

                print(
                    f"   Got: "
                    f"{feature.shape}"
                )

                continue

            features.append(
                feature.astype(
                    np.float32
                )
            )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if len(features) < SEQUENCE_LENGTH:

            print(
                f"⚠️ Not enough valid features: "
                f"{video_folder_name}"
            )

            skipped_videos += 1
            continue

        features = np.asarray(
            features,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Create non-overlapping sequences
        # ----------------------------------------------------

        sequence_count = (
            len(features)
            // SEQUENCE_LENGTH
        )

        usable_length = (
            sequence_count
            * SEQUENCE_LENGTH
        )

        features = features[
            :usable_length
        ]

        for start in range(
            0,
            usable_length,
            SEQUENCE_LENGTH
        ):

            sequence = features[
                start:
                start + SEQUENCE_LENGTH
            ]

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_SIZE
            ):

                continue

            X.append(
                sequence
            )

            y.append(
                label
            )

            # IMPORTANT:
            # Every sequence from the same video
            # gets the same video ID.
            video_ids.append(
                video_folder_name
            )

        processed_videos += 1

        label_name = (
            "Real"
            if label == 0
            else "Fake"
        )

        print(
            f"✅ {video_folder_name} | "
            f"Label: {label_name} | "
            f"Features: {len(features)} | "
            f"Sequences: {sequence_count}"
        )

    # ========================================================
    # Convert to NumPy
    # ========================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    video_ids = np.asarray(
        video_ids
    )

    # ========================================================
    # Dataset Summary
    # ========================================================

    print("\n==============================")
    print("Dataset Built")
    print("==============================")

    print(
        "Videos processed :",
        processed_videos
    )

    print(
        "Videos skipped   :",
        skipped_videos
    )

    print(
        "X shape          :",
        X.shape
    )

    print(
        "y shape          :",
        y.shape
    )

    print(
        "Video IDs        :",
        video_ids.shape
    )

    print(
        "Unique Videos    :",
        len(np.unique(video_ids))
    )

    print(
        "Real sequences   :",
        np.sum(y == 0)
    )

    print(
        "Fake sequences   :",
        np.sum(y == 1)
    )

    print("==============================")

    # ========================================================
    # Safety Checks
    # ========================================================

    if len(X) == 0:

        raise RuntimeError(
            "Dataset is empty."
        )

    if X.ndim != 3:

        raise RuntimeError(
            f"Expected X to have 3 dimensions, "
            f"got {X.ndim}"
        )

    if X.shape[1:] != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):

        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )

    if len(X) != len(y):

        raise RuntimeError(
            "X and y length mismatch."
        )

    if len(X) != len(video_ids):

        raise RuntimeError(
            "X and video_ids length mismatch."
        )

    if np.isnan(X).any():

        raise RuntimeError(
            "Dataset contains NaN values."
        )

    if np.isinf(X).any():

        raise RuntimeError(
            "Dataset contains infinite values."
        )

    unique_labels = np.unique(y)

    if len(unique_labels) != 2:

        raise RuntimeError(
            f"Expected both classes, "
            f"got labels: {unique_labels}"
        )

    print(
        "Dataset validation : PASSED"
    )

    print("==============================")

    return (
        X,
        y,
        video_ids
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    X, y, video_ids = build_dataset()

    print(
        "\nDataset builder completed successfully."
    )

    output_folder = Path(
        "outputs/dataset_exp5_full"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_folder /
        "dataset.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        video_ids=video_ids
    )

    print(
        f"Dataset saved to: {output_file}"
    )