from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp8_temporal"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split_name):

    file_path = DATASET_DIR / (
        f"{split_name}.npz"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    data = np.load(
        file_path,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]

    video_ids = data["video_ids"]

    return (
        X,
        y,
        video_ids
    )


# ============================================================
# ANALYZE DATASET SPLIT
# ============================================================

def analyze_split(split_name):

    print("\n" + "=" * 70)
    print(
        f"EXP8 {split_name.upper()} TEMPORAL FEATURE ANALYSIS"
    )
    print("=" * 70)


    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset...")

    X, y, video_ids = load_dataset(
        split_name
    )


    print("\nDataset loaded successfully.")


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)


    print(
        "\nX shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Total sequences:",
        len(y)
    )

    print(
        "Total videos:",
        len(np.unique(video_ids))
    )

    print(
        "Sequence length:",
        X.shape[1]
    )

    print(
        "Total feature size:",
        X.shape[2]
    )


    print("\nClass distribution:")

    print(
        "Real sequences:",
        np.sum(y == 0)
    )

    print(
        "Fake sequences:",
        np.sum(y == 1)
    )


    # ========================================================
    # EXP8 FEATURE STRUCTURE
    #
    # Original features
    # Velocity features
    # Acceleration features
    #
    # Total = Original × 3
    # ========================================================

    total_features = X.shape[2]

    if total_features % 3 != 0:

        raise ValueError(
            "EXP8 feature size must be divisible by 3. "
            f"Found: {total_features}"
        )


    original_feature_size = (
        total_features // 3
    )


    print("\n" + "=" * 70)
    print("EXP8 FEATURE STRUCTURE")
    print("=" * 70)


    print(
        "\nOriginal feature size:",
        original_feature_size
    )

    print(
        "Velocity feature size:",
        original_feature_size
    )

    print(
        "Acceleration feature size:",
        original_feature_size
    )

    print(
        "Total feature size:",
        total_features
    )


    # ========================================================
    # SPLIT FEATURES
    # ========================================================

    original = X[
        :,
        :,
        :original_feature_size
    ]

    velocity = X[
        :,
        :,
        original_feature_size:
        original_feature_size * 2
    ]

    acceleration = X[
        :,
        :,
        original_feature_size * 2:
    ]


    # ========================================================
    # VALIDATE FEATURE SHAPES
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE SHAPE VALIDATION")
    print("=" * 70)


    print(
        "\nOriginal shape:",
        original.shape
    )

    print(
        "Velocity shape:",
        velocity.shape
    )

    print(
        "Acceleration shape:",
        acceleration.shape
    )


    if (
        original.shape[2]
        != original_feature_size
    ):

        raise ValueError(
            "Invalid original feature size."
        )


    if (
        velocity.shape[2]
        != original_feature_size
    ):

        raise ValueError(
            "Invalid velocity feature size."
        )


    if (
        acceleration.shape[2]
        != original_feature_size
    ):

        raise ValueError(
            "Invalid acceleration feature size."
        )


    print(
        "\nFeature structure validated successfully."
    )


    # ========================================================
    # SEPARATE REAL AND FAKE
    # ========================================================

    X_real = X[y == 0]

    X_fake = X[y == 1]


    original_real = original[y == 0]
    original_fake = original[y == 1]


    velocity_real = velocity[y == 0]
    velocity_fake = velocity[y == 1]


    acceleration_real = acceleration[y == 0]
    acceleration_fake = acceleration[y == 1]


    # ========================================================
    # OVERALL FEATURE STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("OVERALL FEATURE STATISTICS")
    print("=" * 70)


    print("\nORIGINAL FEATURES")

    print(
        "Mean:",
        f"{np.mean(original):.6f}"
    )

    print(
        "Std:",
        f"{np.std(original):.6f}"
    )

    print(
        "Minimum:",
        f"{np.min(original):.6f}"
    )

    print(
        "Maximum:",
        f"{np.max(original):.6f}"
    )


    print("\nVELOCITY FEATURES")

    print(
        "Mean:",
        f"{np.mean(velocity):.6f}"
    )

    print(
        "Std:",
        f"{np.std(velocity):.6f}"
    )

    print(
        "Minimum:",
        f"{np.min(velocity):.6f}"
    )

    print(
        "Maximum:",
        f"{np.max(velocity):.6f}"
    )


    print("\nACCELERATION FEATURES")

    print(
        "Mean:",
        f"{np.mean(acceleration):.6f}"
    )

    print(
        "Std:",
        f"{np.std(acceleration):.6f}"
    )

    print(
        "Minimum:",
        f"{np.min(acceleration):.6f}"
    )

    print(
        "Maximum:",
        f"{np.max(acceleration):.6f}"
    )


    # ========================================================
    # ORIGINAL FEATURE ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("ORIGINAL FEATURE ANALYSIS")
    print("=" * 70)


    real_original_mean = np.mean(
        original_real
    )

    fake_original_mean = np.mean(
        original_fake
    )


    print("\nREAL")

    print(
        "Mean:",
        f"{real_original_mean:.6f}"
    )

    print(
        "Std:",
        f"{np.std(original_real):.6f}"
    )


    print("\nFAKE")

    print(
        "Mean:",
        f"{fake_original_mean:.6f}"
    )

    print(
        "Std:",
        f"{np.std(original_fake):.6f}"
    )


    print("\nDifference:")

    print(
        f"{abs(fake_original_mean - real_original_mean):.6f}"
    )


    # ========================================================
    # VELOCITY ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("VELOCITY ANALYSIS")
    print("=" * 70)


    # --------------------------------------------------------
    # Absolute velocity measures movement magnitude
    # --------------------------------------------------------

    real_velocity_abs = np.abs(
        velocity_real
    )

    fake_velocity_abs = np.abs(
        velocity_fake
    )


    real_velocity_motion = np.mean(
        real_velocity_abs
    )

    fake_velocity_motion = np.mean(
        fake_velocity_abs
    )


    print("\nREAL VELOCITY")

    print(
        "Mean absolute velocity:",
        f"{real_velocity_motion:.6f}"
    )

    print(
        "Velocity standard deviation:",
        f"{np.std(velocity_real):.6f}"
    )


    print("\nFAKE VELOCITY")

    print(
        "Mean absolute velocity:",
        f"{fake_velocity_motion:.6f}"
    )

    print(
        "Velocity standard deviation:",
        f"{np.std(velocity_fake):.6f}"
    )


    velocity_difference = abs(
        fake_velocity_motion
        -
        real_velocity_motion
    )


    print("\nREAL VS FAKE VELOCITY")

    print(
        "Real mean motion:",
        f"{real_velocity_motion:.6f}"
    )

    print(
        "Fake mean motion:",
        f"{fake_velocity_motion:.6f}"
    )

    print(
        "Absolute difference:",
        f"{velocity_difference:.6f}"
    )


    # ========================================================
    # VELOCITY SEQUENCE-WISE ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("SEQUENCE-WISE VELOCITY ANALYSIS")
    print("=" * 70)


    real_sequence_velocity = np.mean(
        np.abs(velocity_real),
        axis=(1, 2)
    )

    fake_sequence_velocity = np.mean(
        np.abs(velocity_fake),
        axis=(1, 2)
    )


    print("\nREAL SEQUENCES")

    print(
        "Mean:",
        f"{np.mean(real_sequence_velocity):.6f}"
    )

    print(
        "Median:",
        f"{np.median(real_sequence_velocity):.6f}"
    )

    print(
        "Minimum:",
        f"{np.min(real_sequence_velocity):.6f}"
    )

    print(
        "Maximum:",
        f"{np.max(real_sequence_velocity):.6f}"
    )


    print("\nFAKE SEQUENCES")

    print(
        "Mean:",
        f"{np.mean(fake_sequence_velocity):.6f}"
    )

    print(
        "Median:",
        f"{np.median(fake_sequence_velocity):.6f}"
    )

    print(
        "Minimum:",
        f"{np.min(fake_sequence_velocity):.6f}"
    )

    print(
        "Maximum:",
        f"{np.max(fake_sequence_velocity):.6f}"
    )


    # ========================================================
    # VELOCITY SEPARATION SCORE
    # ========================================================

    real_velocity_std = np.std(
        real_sequence_velocity
    )

    fake_velocity_std = np.std(
        fake_sequence_velocity
    )


    pooled_velocity_std = np.sqrt(

        (
            real_velocity_std ** 2
            +
            fake_velocity_std ** 2
        ) / 2

    )


    if pooled_velocity_std < 1e-8:

        pooled_velocity_std = 1e-8


    velocity_separation_score = (

        velocity_difference
        /
        pooled_velocity_std

    )


    print("\nVelocity separation score:")

    print(
        f"{velocity_separation_score:.6f}"
    )


    # ========================================================
    # ACCELERATION ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("ACCELERATION ANALYSIS")
    print("=" * 70)


    real_acceleration_abs = np.abs(
        acceleration_real
    )

    fake_acceleration_abs = np.abs(
        acceleration_fake
    )


    real_acceleration_motion = np.mean(
        real_acceleration_abs
    )

    fake_acceleration_motion = np.mean(
        fake_acceleration_abs
    )


    print("\nREAL ACCELERATION")

    print(
        "Mean absolute acceleration:",
        f"{real_acceleration_motion:.6f}"
    )

    print(
        "Acceleration standard deviation:",
        f"{np.std(acceleration_real):.6f}"
    )


    print("\nFAKE ACCELERATION")

    print(
        "Mean absolute acceleration:",
        f"{fake_acceleration_motion:.6f}"
    )

    print(
        "Acceleration standard deviation:",
        f"{np.std(acceleration_fake):.6f}"
    )


    acceleration_difference = abs(

        fake_acceleration_motion
        -
        real_acceleration_motion

    )


    print("\nREAL VS FAKE ACCELERATION")

    print(
        "Real mean acceleration:",
        f"{real_acceleration_motion:.6f}"
    )

    print(
        "Fake mean acceleration:",
        f"{fake_acceleration_motion:.6f}"
    )

    print(
        "Absolute difference:",
        f"{acceleration_difference:.6f}"
    )


    # ========================================================
    # SEQUENCE-WISE ACCELERATION
    # ========================================================

    print("\n" + "=" * 70)
    print("SEQUENCE-WISE ACCELERATION ANALYSIS")
    print("=" * 70)


    real_sequence_acceleration = np.mean(

        np.abs(acceleration_real),

        axis=(1, 2)

    )


    fake_sequence_acceleration = np.mean(

        np.abs(acceleration_fake),

        axis=(1, 2)

    )


    print("\nREAL SEQUENCES")

    print(
        "Mean:",
        f"{np.mean(real_sequence_acceleration):.6f}"
    )

    print(
        "Median:",
        f"{np.median(real_sequence_acceleration):.6f}"
    )

    print(
        "Minimum:",
        f"{np.min(real_sequence_acceleration):.6f}"
    )

    print(
        "Maximum:",
        f"{np.max(real_sequence_acceleration):.6f}"
    )


    print("\nFAKE SEQUENCES")

    print(
        "Mean:",
        f"{np.mean(fake_sequence_acceleration):.6f}"
    )

    print(
        "Median:",
        f"{np.median(fake_sequence_acceleration):.6f}"
    )

    print(
        "Minimum:",
        f"{np.min(fake_sequence_acceleration):.6f}"
    )

    print(
        "Maximum:",
        f"{np.max(fake_sequence_acceleration):.6f}"
    )


    # ========================================================
    # ACCELERATION SEPARATION SCORE
    # ========================================================

    real_acceleration_std = np.std(
        real_sequence_acceleration
    )

    fake_acceleration_std = np.std(
        fake_sequence_acceleration
    )


    pooled_acceleration_std = np.sqrt(

        (
            real_acceleration_std ** 2
            +
            fake_acceleration_std ** 2
        ) / 2

    )


    if pooled_acceleration_std < 1e-8:

        pooled_acceleration_std = 1e-8


    acceleration_separation_score = (

        acceleration_difference
        /
        pooled_acceleration_std

    )


    print("\nAcceleration separation score:")

    print(
        f"{acceleration_separation_score:.6f}"
    )


    # ========================================================
    # FEATURE-WISE VELOCITY ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE-WISE VELOCITY ANALYSIS")
    print("=" * 70)


    real_feature_velocity = np.mean(

        np.abs(velocity_real),

        axis=(0, 1)

    )


    fake_feature_velocity = np.mean(

        np.abs(velocity_fake),

        axis=(0, 1)

    )


    feature_velocity_difference = np.abs(

        fake_feature_velocity
        -
        real_feature_velocity

    )


    print(
        "\nAverage feature velocity difference:",
        f"{np.mean(feature_velocity_difference):.6f}"
    )

    print(
        "Maximum feature velocity difference:",
        f"{np.max(feature_velocity_difference):.6f}"
    )


    # ========================================================
    # FEATURE-WISE VELOCITY SEPARATION
    # ========================================================

    real_feature_sequence_velocity = np.mean(

        np.abs(velocity_real),

        axis=1

    )


    fake_feature_sequence_velocity = np.mean(

        np.abs(velocity_fake),

        axis=1

    )


    real_feature_velocity_std = np.std(

        real_feature_sequence_velocity,

        axis=0

    )


    fake_feature_velocity_std = np.std(

        fake_feature_sequence_velocity,

        axis=0

    )


    pooled_feature_velocity_std = np.sqrt(

        (
            real_feature_velocity_std ** 2
            +
            fake_feature_velocity_std ** 2
        ) / 2

    )


    pooled_feature_velocity_std = np.where(

        pooled_feature_velocity_std < 1e-8,

        1e-8,

        pooled_feature_velocity_std

    )


    feature_velocity_separation = (

        feature_velocity_difference
        /
        pooled_feature_velocity_std

    )


    print(
        "\nAverage feature velocity separation:",
        f"{np.mean(feature_velocity_separation):.6f}"
    )

    print(
        "Maximum feature velocity separation:",
        f"{np.max(feature_velocity_separation):.6f}"
    )


    # ========================================================
    # TOP 20 VELOCITY FEATURES
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 20 FEATURES BY VELOCITY SEPARATION")
    print("=" * 70)


    top_velocity_indices = np.argsort(

        feature_velocity_separation

    )[::-1][:20]


    for rank, feature_index in enumerate(

        top_velocity_indices,

        start=1

    ):

        print(f"\nRank {rank}")

        print(
            "Feature index:",
            feature_index
        )

        print(
            "Real velocity:",
            f"{real_feature_velocity[feature_index]:.6f}"
        )

        print(
            "Fake velocity:",
            f"{fake_feature_velocity[feature_index]:.6f}"
        )

        print(
            "Difference:",
            f"{feature_velocity_difference[feature_index]:.6f}"
        )

        print(
            "Separation:",
            f"{feature_velocity_separation[feature_index]:.6f}"
        )


    # ========================================================
    # FEATURE-WISE ACCELERATION ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE-WISE ACCELERATION ANALYSIS")
    print("=" * 70)


    real_feature_acceleration = np.mean(

        np.abs(acceleration_real),

        axis=(0, 1)

    )


    fake_feature_acceleration = np.mean(

        np.abs(acceleration_fake),

        axis=(0, 1)

    )


    feature_acceleration_difference = np.abs(

        fake_feature_acceleration
        -
        real_feature_acceleration

    )


    print(
        "\nAverage feature acceleration difference:",
        f"{np.mean(feature_acceleration_difference):.6f}"
    )

    print(
        "Maximum feature acceleration difference:",
        f"{np.max(feature_acceleration_difference):.6f}"
    )


    # ========================================================
    # FEATURE-WISE ACCELERATION SEPARATION
    # ========================================================

    real_feature_sequence_acceleration = np.mean(

        np.abs(acceleration_real),

        axis=1

    )


    fake_feature_sequence_acceleration = np.mean(

        np.abs(acceleration_fake),

        axis=1

    )


    real_feature_acceleration_std = np.std(

        real_feature_sequence_acceleration,

        axis=0

    )


    fake_feature_acceleration_std = np.std(

        fake_feature_sequence_acceleration,

        axis=0

    )


    pooled_feature_acceleration_std = np.sqrt(

        (
            real_feature_acceleration_std ** 2
            +
            fake_feature_acceleration_std ** 2
        ) / 2

    )


    pooled_feature_acceleration_std = np.where(

        pooled_feature_acceleration_std < 1e-8,

        1e-8,

        pooled_feature_acceleration_std

    )


    feature_acceleration_separation = (

        feature_acceleration_difference
        /
        pooled_feature_acceleration_std

    )


    print(
        "\nAverage feature acceleration separation:",
        f"{np.mean(feature_acceleration_separation):.6f}"
    )

    print(
        "Maximum feature acceleration separation:",
        f"{np.max(feature_acceleration_separation):.6f}"
    )


    # ========================================================
    # TOP 20 ACCELERATION FEATURES
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 20 FEATURES BY ACCELERATION SEPARATION")
    print("=" * 70)


    top_acceleration_indices = np.argsort(

        feature_acceleration_separation

    )[::-1][:20]


    for rank, feature_index in enumerate(

        top_acceleration_indices,

        start=1

    ):

        print(f"\nRank {rank}")

        print(
            "Feature index:",
            feature_index
        )

        print(
            "Real acceleration:",
            f"{real_feature_acceleration[feature_index]:.6f}"
        )

        print(
            "Fake acceleration:",
            f"{fake_feature_acceleration[feature_index]:.6f}"
        )

        print(
            "Difference:",
            f"{feature_acceleration_difference[feature_index]:.6f}"
        )

        print(
            "Separation:",
            f"{feature_acceleration_separation[feature_index]:.6f}"
        )


    # ========================================================
    # TEMPORAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("EXP8 TEMPORAL FEATURE SUMMARY")
    print("=" * 70)


    print("\nVELOCITY")

    print(
        "Real mean:",
        f"{real_velocity_motion:.6f}"
    )

    print(
        "Fake mean:",
        f"{fake_velocity_motion:.6f}"
    )

    print(
        "Difference:",
        f"{velocity_difference:.6f}"
    )

    print(
        "Separation score:",
        f"{velocity_separation_score:.6f}"
    )


    print("\nACCELERATION")

    print(
        "Real mean:",
        f"{real_acceleration_motion:.6f}"
    )

    print(
        "Fake mean:",
        f"{fake_acceleration_motion:.6f}"
    )

    print(
        "Difference:",
        f"{acceleration_difference:.6f}"
    )

    print(
        "Separation score:",
        f"{acceleration_separation_score:.6f}"
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "velocity_difference":
        velocity_difference,

        "velocity_separation":
        velocity_separation_score,

        "acceleration_difference":
        acceleration_difference,

        "acceleration_separation":
        acceleration_separation_score

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("EXP8 COMPLETE TEMPORAL FEATURE ANALYSIS")
    print("=" * 70)


    print("\nDataset directory:")

    print(
        DATASET_DIR
    )


    # ========================================================
    # ANALYZE TRAIN DATASET
    # ========================================================

    train_results = analyze_split(
        "train"
    )


    # ========================================================
    # ANALYZE VALIDATION DATASET
    # ========================================================

    validation_results = analyze_split(
        "validation"
    )


    # ========================================================
    # ANALYZE TEST DATASET
    # ========================================================

    test_results = analyze_split(
        "test"
    )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL EXP8 TEMPORAL SEPARATION COMPARISON")
    print("=" * 70)


    print("\nTRAIN")

    print(
        "Velocity separation:",
        f"{train_results['velocity_separation']:.6f}"
    )

    print(
        "Acceleration separation:",
        f"{train_results['acceleration_separation']:.6f}"
    )


    print("\nVALIDATION")

    print(
        "Velocity separation:",
        f"{validation_results['velocity_separation']:.6f}"
    )

    print(
        "Acceleration separation:",
        f"{validation_results['acceleration_separation']:.6f}"
    )


    print("\nTEST")

    print(
        "Velocity separation:",
        f"{test_results['velocity_separation']:.6f}"
    )

    print(
        "Acceleration separation:",
        f"{test_results['acceleration_separation']:.6f}"
    )


    print("\n" + "=" * 70)
    print("EXP8 TEMPORAL FEATURE ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
