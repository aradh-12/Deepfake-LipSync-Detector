from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "outputs/dataset_exp7_source_aware"
)

OUTPUT_DIR = Path(
    "outputs/dataset_exp8_temporal"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEMPORAL FEATURE CREATION
# ============================================================

def create_temporal_features(X):

    """
    Input shape:

        (samples, sequence_length, features)

    For every frame we create:

        1. Original features
        2. Velocity features
        3. Acceleration features

    Final feature size:

        original_features * 3
    """

    print("Original shape:", X.shape)

    # --------------------------------------------------------
    # ORIGINAL FEATURES
    # --------------------------------------------------------

    original = X

    # --------------------------------------------------------
    # VELOCITY
    #
    # Difference between consecutive frames
    # --------------------------------------------------------

    velocity = np.diff(
        X,
        axis=1,
        prepend=X[:, 0:1, :]
    )

    # --------------------------------------------------------
    # ACCELERATION
    #
    # Difference between consecutive velocity vectors
    # --------------------------------------------------------

    acceleration = np.diff(
        velocity,
        axis=1,
        prepend=velocity[:, 0:1, :]
    )

    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------

    X_temporal = np.concatenate(
        [
            original,
            velocity,
            acceleration
        ],
        axis=2
    )

    print(
        "Temporal shape:",
        X_temporal.shape
    )

    return X_temporal


# ============================================================
# PROCESS DATASET SPLIT
# ============================================================

def process_split(split_name):

    print("\n" + "=" * 60)
    print(
        f"PROCESSING {split_name.upper()} DATASET"
    )
    print("=" * 60)

    input_file = INPUT_DIR / (
        f"{split_name}.npz"
    )

    output_file = OUTPUT_DIR / (
        f"{split_name}.npz"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading dataset...")

    data = np.load(
        input_file,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --------------------------------------------------------
    # CREATE TEMPORAL FEATURES
    # --------------------------------------------------------

    print(
        "\nCreating temporal features..."
    )

    X_temporal = create_temporal_features(
        X
    )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    print(
        "\nSaving temporal dataset..."
    )

    np.savez_compressed(
        output_file,
        X=X_temporal,
        y=y,
        video_ids=video_ids
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("EXP8 TEMPORAL DATASET CREATION")
    print("=" * 60)

    print("\nInput directory:")
    print(INPUT_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    process_split("train")

    process_split("validation")

    process_split("test")

    print("\n" + "=" * 60)
    print(
        "EXP8 TEMPORAL DATASET CREATION COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":

    main()
