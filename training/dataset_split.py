from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATASET_FILE = Path(
    "outputs/dataset/dataset.npz"
)

OUTPUT_FOLDER = Path(
    "outputs/dataset"
)

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# Validation
# ============================================================

def validate_split(
    name,
    X,
    y,
    video_ids
):

    print("\n" + "=" * 50)
    print(f"{name} SET")
    print("=" * 50)

    print("X shape       :", X.shape)
    print("y shape       :", y.shape)
    print("Video IDs     :", video_ids.shape)

    print(
        "Unique videos :",
        len(np.unique(video_ids))
    )

    print(
        "Real samples  :",
        np.sum(y == 0)
    )

    print(
        "Fake samples  :",
        np.sum(y == 1)
    )

    if len(X) != len(y):
        raise RuntimeError(
            f"{name}: X/y length mismatch."
        )

    if len(X) != len(video_ids):
        raise RuntimeError(
            f"{name}: X/video_ids length mismatch."
        )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}\n"
            "Run dataset_builder first."
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = np.load(
        DATASET_FILE,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]

    print("\n" + "=" * 50)
    print("LOADED DATASET")
    print("=" * 50)

    print("X shape       :", X.shape)
    print("y shape       :", y.shape)
    print("Video IDs     :", video_ids.shape)

    original_sequence_count = len(X)
    original_video_count = len(np.unique(video_ids))

    print(
        "Original sequences :",
        original_sequence_count
    )

    print(
        "Original videos    :",
        original_video_count
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT remove sequences here.
    #
    # Multiple sequences from the same video are valid.
    # We keep every sequence and split by VIDEO ID.
    # --------------------------------------------------------

    unique_videos = np.unique(
        video_ids
    )

    # --------------------------------------------------------
    # Determine one label for every unique video
    # --------------------------------------------------------

    video_labels = []

    for video_id in unique_videos:

        indices = np.where(
            video_ids == video_id
        )[0]

        labels = np.unique(
            y[indices]
        )

        if len(labels) != 1:

            raise RuntimeError(
                f"Video {video_id} has multiple labels: "
                f"{labels}"
            )

        video_labels.append(
            labels[0]
        )

    video_labels = np.asarray(
        video_labels,
        dtype=np.int32
    )

    print(
        "Real videos        :",
        np.sum(video_labels == 0)
    )

    print(
        "Fake videos        :",
        np.sum(video_labels == 1)
    )

    # --------------------------------------------------------
    # First split:
    #
    # 70% TRAIN
    # 30% TEMPORARY
    # --------------------------------------------------------

    train_videos, temp_videos = train_test_split(
        unique_videos,
        test_size=(
            VALIDATION_SIZE +
            TEST_SIZE
        ),
        random_state=RANDOM_STATE,
        stratify=video_labels
    )

    # --------------------------------------------------------
    # Labels for temporary videos
    # --------------------------------------------------------

    video_to_label = {
        video_id: label
        for video_id, label
        in zip(unique_videos, video_labels)
    }

    temp_labels = np.asarray(
        [
            video_to_label[video_id]
            for video_id in temp_videos
        ],
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Second split:
    #
    # TEMPORARY → VALIDATION + TEST
    #
    # 15% / 15%
    # --------------------------------------------------------

    test_fraction_of_temp = (
        TEST_SIZE /
        (VALIDATION_SIZE + TEST_SIZE)
    )

    validation_videos, test_videos = train_test_split(
        temp_videos,
        test_size=test_fraction_of_temp,
        random_state=RANDOM_STATE,
        stratify=temp_labels
    )

    # --------------------------------------------------------
    # Convert VIDEO IDs into sequence indices
    #
    # Every sequence belonging to a video follows that video
    # into exactly one split.
    # --------------------------------------------------------

    def get_sequence_indices(
        selected_videos
    ):

        selected_videos = set(
            selected_videos
        )

        return np.asarray(
            [
                i
                for i, video_id
                in enumerate(video_ids)
                if video_id in selected_videos
            ],
            dtype=np.int64
        )

    train_indices = get_sequence_indices(
        train_videos
    )

    validation_indices = get_sequence_indices(
        validation_videos
    )

    test_indices = get_sequence_indices(
        test_videos
    )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    X_train = X[train_indices]
    y_train = y[train_indices]
    ids_train = video_ids[train_indices]

    X_validation = X[validation_indices]
    y_validation = y[validation_indices]
    ids_validation = video_ids[validation_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]
    ids_test = video_ids[test_indices]

    # --------------------------------------------------------
    # Safety check:
    #
    # No video may appear in multiple splits.
    # --------------------------------------------------------

    train_set = set(ids_train)
    validation_set = set(ids_validation)
    test_set = set(ids_test)

    train_validation_overlap = (
        train_set &
        validation_set
    )

    train_test_overlap = (
        train_set &
        test_set
    )

    validation_test_overlap = (
        validation_set &
        test_set
    )

    if train_validation_overlap:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "train/validation overlap detected."
        )

    if train_test_overlap:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "train/test overlap detected."
        )

    if validation_test_overlap:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "validation/test overlap detected."
        )

    # --------------------------------------------------------
    # Important conservation check
    #
    # ALL original sequences must be present.
    # --------------------------------------------------------

    total_split_sequences = (
        len(X_train)
        + len(X_validation)
        + len(X_test)
    )

    if total_split_sequences != original_sequence_count:

        raise RuntimeError(
            "SEQUENCE LOSS DETECTED: "
            f"Original={original_sequence_count}, "
            f"Split total={total_split_sequences}"
        )

    # --------------------------------------------------------
    # Validate each split
    # --------------------------------------------------------

    validate_split(
        "TRAIN",
        X_train,
        y_train,
        ids_train
    )

    validate_split(
        "VALIDATION",
        X_validation,
        y_validation,
        ids_validation
    )

    validate_split(
        "TEST",
        X_test,
        y_test,
        ids_test
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FOLDER / "train.npz",
        X=X_train,
        y=y_train,
        video_ids=ids_train
    )

    np.savez_compressed(
        OUTPUT_FOLDER / "validation.npz",
        X=X_validation,
        y=y_validation,
        video_ids=ids_validation
    )

    np.savez_compressed(
        OUTPUT_FOLDER / "test.npz",
        X=X_test,
        y=y_test,
        video_ids=ids_test
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("VIDEO-LEVEL SPLIT COMPLETED")
    print("=" * 50)

    print(
        "Train videos      :",
        len(train_set)
    )

    print(
        "Validation videos :",
        len(validation_set)
    )

    print(
        "Test videos       :",
        len(test_set)
    )

    print()

    print(
        "Train sequences      :",
        len(X_train)
    )

    print(
        "Validation sequences :",
        len(X_validation)
    )

    print(
        "Test sequences       :",
        len(X_test)
    )

    print()

    print(
        "Total split sequences:",
        total_split_sequences
    )

    print(
        "Original sequences   :",
        original_sequence_count
    )

    print()

    print(
        "Data leakage check : PASSED"
    )

    print(
        "Sequence conservation : PASSED"
    )

    print(
        "Files saved in    :",
        OUTPUT_FOLDER
    )

    print("=" * 50)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()