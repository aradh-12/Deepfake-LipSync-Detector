from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MULTIMODAL_FEATURE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "multimodal_features"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

FEATURE_DIMENSION = 119

SEQUENCE_LENGTH = 150


# ============================================================
# GET SPLIT DIRECTORY
# ============================================================

def get_split_directory(
    split
):

    split_path = (
        MULTIMODAL_FEATURE_ROOT
        / split
    )

    if not split_path.exists():

        raise FileNotFoundError(

            f"Split directory not found:\n"
            f"{split_path}"
        )

    return split_path


# ============================================================
# GET FEATURE FILES
# ============================================================

def get_feature_files(
    split
):

    split_directory = (
        get_split_directory(
            split
        )
    )

    files = sorted(

        split_directory.glob(
            "*.npz"
        )
    )

    if len(files) == 0:

        raise ValueError(

            f"No feature files found "
            f"for split: {split}"
        )

    return files


# ============================================================
# LOAD SINGLE SAMPLE
# ============================================================

def load_sample(
    file_path
):

    with np.load(
        file_path
    ) as data:

        features = np.array(

            data["features"],

            dtype=np.float32
        )

        label = int(

            data["label"]
        )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if features.ndim != 2:

        raise ValueError(

            f"Invalid feature dimensions: "
            f"{features.shape}"
        )

    if features.shape[1] != FEATURE_DIMENSION:

        raise ValueError(

            f"Expected feature dimension "
            f"{FEATURE_DIMENSION}, "
            f"got {features.shape[1]}"
        )

    if features.shape[0] == 0:

        raise ValueError(

            "Feature sequence is empty"
        )

    if not np.isfinite(
        features
    ).all():

        raise ValueError(

            "Features contain NaN or Inf"
        )

    if label not in [

        0,
        1

    ]:

        raise ValueError(

            f"Invalid label: {label}"
        )

    return (

        features,

        label
    )


# ============================================================
# PAD OR TRUNCATE SEQUENCE
# ============================================================

def pad_or_truncate(
    features
):

    current_length = (

        features.shape[0]
    )

    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------

    if current_length > SEQUENCE_LENGTH:

        features = (

            features[
                :SEQUENCE_LENGTH
            ]
        )

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    elif current_length < SEQUENCE_LENGTH:

        padding_length = (

            SEQUENCE_LENGTH
            -
            current_length
        )

        padding = np.zeros(

            (

                padding_length,

                FEATURE_DIMENSION

            ),

            dtype=np.float32
        )

        features = np.concatenate(

            [

                features,

                padding

            ],

            axis=0
        )

    return features


# ============================================================
# LOAD COMPLETE SPLIT
# ============================================================

def load_split(
    split
):

    files = (

        get_feature_files(
            split
        )
    )

    features_list = []

    labels_list = []

    total = len(
        files
    )

    print()

    print(
        "=" * 70
    )

    print(
        f"LOADING {split.upper()} DATASET"
    )

    print(
        "=" * 70
    )

    print()

    for index, file_path in enumerate(

        files,

        start=1

    ):

        features, label = (

            load_sample(
                file_path
            )
        )

        features = (

            pad_or_truncate(
                features
            )
        )

        features_list.append(

            features
        )

        labels_list.append(

            label
        )

        if (

            index % 100 == 0

            or

            index == total

        ):

            print(

                f"[{index}/{total}] "
                f"Loaded"
            )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    X = np.array(

        features_list,

        dtype=np.float32
    )

    y = np.array(

        labels_list,

        dtype=np.int32
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(

            f"Expected 3D dataset, "
            f"got {X.shape}"
        )

    if X.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(

            f"Invalid sequence length: "
            f"{X.shape}"
        )

    if X.shape[2] != FEATURE_DIMENSION:

        raise ValueError(

            f"Invalid feature dimension: "
            f"{X.shape}"
        )

    if len(X) != len(y):

        raise ValueError(

            "Feature and label counts "
            "do not match"
        )

    print()

    print(
        "Dataset shape:"
    )

    print(
        X.shape
    )

    print()

    print(
        "Labels shape:"
    )

    print(
        y.shape
    )

    print()

    print(
        "Real samples:",
        int(
            np.sum(
                y == 0
            )
        )
    )

    print(
        "Fake samples:",
        int(
            np.sum(
                y == 1
            )
        )
    )

    return (

        X,

        y
    )


# ============================================================
# CREATE TENSORFLOW DATASET
# ============================================================

def create_tf_dataset(
    X,
    y,
    batch_size=16,
    shuffle=False
):

    dataset = (

        tf.data.Dataset.from_tensor_slices(

            (

                X,

                y

            )
        )
    )

    if shuffle:

        dataset = dataset.shuffle(

            buffer_size=len(X),

            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(

        batch_size
    )

    dataset = dataset.prefetch(

        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# LOAD ALL DATASETS
# ============================================================

def load_datasets(
    batch_size=16
):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    X_train, y_train = (

        load_split(
            "train"
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    X_validation, y_validation = (

        load_split(
            "validation"
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    X_test, y_test = (

        load_split(
            "test"
        )
    )

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_dataset = (

        create_tf_dataset(

            X_train,

            y_train,

            batch_size=batch_size,

            shuffle=True
        )
    )

    validation_dataset = (

        create_tf_dataset(

            X_validation,

            y_validation,

            batch_size=batch_size,

            shuffle=False
        )
    )

    test_dataset = (

        create_tf_dataset(

            X_test,

            y_test,

            batch_size=batch_size,

            shuffle=False
        )
    )

    return (

        train_dataset,

        validation_dataset,

        test_dataset
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "MULTIMODAL DATASET LOADER"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_dataset, validation_dataset, test_dataset = (

        load_datasets(

            batch_size=16
        )
    )

    # --------------------------------------------------------
    # Display one batch
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "TRAINING BATCH TEST"
    )

    print(
        "=" * 70
    )

    print()

    for batch_features, batch_labels in train_dataset.take(1):

        print(
            "Batch features shape:",
            batch_features.shape
        )

        print(
            "Batch labels shape:",
            batch_labels.shape
        )

        print()

        print(
            "Feature dtype:",
            batch_features.dtype
        )

        print(
            "Label dtype:",
            batch_labels.dtype
        )

    print()

    print(
        "=" * 70
    )

    print(
        "DATASET LOADER COMPLETE"
    )

    print(
        "=" * 70
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()