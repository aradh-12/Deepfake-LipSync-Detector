import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp7_source_aware"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("EXP7 TEMPORAL FEATURE ANALYSIS")
print("=" * 70)


print("\nLoading training dataset...")

train_data = np.load(
    DATASET_DIR / "train.npz",
    allow_pickle=True
)

X_train = train_data["X"]
y_train = train_data["y"]


print("\nDataset loaded successfully.")

print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"\nTotal sequences: {len(y_train)}")
print(f"Sequence length: {X_train.shape[1]}")
print(f"Feature size: {X_train.shape[2]}")

print(f"\nReal sequences: {np.sum(y_train == 0)}")
print(f"Fake sequences: {np.sum(y_train == 1)}")


# ============================================================
# SEPARATE REAL AND FAKE
# ============================================================

X_real = X_train[y_train == 0]
X_fake = X_train[y_train == 1]


# ============================================================
# FRAME-TO-FRAME DIFFERENCES
#
# Shape:
# (samples, 29, 199)
# ============================================================

print("\nCalculating frame-to-frame differences...")

real_diff = np.diff(
    X_real,
    axis=1
)

fake_diff = np.diff(
    X_fake,
    axis=1
)


# ============================================================
# ABSOLUTE TEMPORAL MOTION
#
# Measures how much features change between frames.
# ============================================================

real_abs_diff = np.abs(real_diff)
fake_abs_diff = np.abs(fake_diff)


# ============================================================
# AVERAGE MOTION PER SEQUENCE
# ============================================================

real_sequence_motion = np.mean(
    real_abs_diff,
    axis=(1, 2)
)

fake_sequence_motion = np.mean(
    fake_abs_diff,
    axis=(1, 2)
)


print("\n" + "=" * 70)
print("AVERAGE TEMPORAL MOTION")
print("=" * 70)

print("\nREAL SEQUENCES")

print(
    f"Mean motion: "
    f"{np.mean(real_sequence_motion):.6f}"
)

print(
    f"Median motion: "
    f"{np.median(real_sequence_motion):.6f}"
)

print(
    f"Minimum motion: "
    f"{np.min(real_sequence_motion):.6f}"
)

print(
    f"Maximum motion: "
    f"{np.max(real_sequence_motion):.6f}"
)


print("\nFAKE SEQUENCES")

print(
    f"Mean motion: "
    f"{np.mean(fake_sequence_motion):.6f}"
)

print(
    f"Median motion: "
    f"{np.median(fake_sequence_motion):.6f}"
)

print(
    f"Minimum motion: "
    f"{np.min(fake_sequence_motion):.6f}"
)

print(
    f"Maximum motion: "
    f"{np.max(fake_sequence_motion):.6f}"
)


# ============================================================
# TEMPORAL MOTION DIFFERENCE
# ============================================================

motion_difference = abs(
    np.mean(fake_sequence_motion)
    -
    np.mean(real_sequence_motion)
)


print("\n" + "=" * 70)
print("REAL VS FAKE TEMPORAL MOTION")
print("=" * 70)

print(
    f"\nReal mean motion: "
    f"{np.mean(real_sequence_motion):.6f}"
)

print(
    f"Fake mean motion: "
    f"{np.mean(fake_sequence_motion):.6f}"
)

print(
    f"Absolute difference: "
    f"{motion_difference:.6f}"
)


# ============================================================
# STANDARDIZED TEMPORAL MOTION SEPARATION
# ============================================================

real_motion_std = np.std(
    real_sequence_motion
)

fake_motion_std = np.std(
    fake_sequence_motion
)


pooled_motion_std = np.sqrt(
    (
        real_motion_std ** 2
        +
        fake_motion_std ** 2
    ) / 2
)


if pooled_motion_std == 0:

    pooled_motion_std = 1e-8


motion_separation_score = (
    motion_difference
    /
    pooled_motion_std
)


print(
    f"\nTemporal motion separation score: "
    f"{motion_separation_score:.6f}"
)


# ============================================================
# FEATURE-WISE TEMPORAL MOTION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-WISE TEMPORAL MOTION")
print("=" * 70)


# Average absolute change per feature

real_feature_motion = np.mean(
    real_abs_diff,
    axis=(0, 1)
)

fake_feature_motion = np.mean(
    fake_abs_diff,
    axis=(0, 1)
)


feature_motion_difference = np.abs(
    fake_feature_motion
    -
    real_feature_motion
)


print(
    "\nAverage feature motion difference:",
    f"{np.mean(feature_motion_difference):.6f}"
)

print(
    "Maximum feature motion difference:",
    f"{np.max(feature_motion_difference):.6f}"
)


# ============================================================
# FEATURE-WISE TEMPORAL SEPARATION
# ============================================================

# Motion per sequence for each feature

real_feature_sequence_motion = np.mean(
    real_abs_diff,
    axis=1
)

fake_feature_sequence_motion = np.mean(
    fake_abs_diff,
    axis=1
)


real_feature_motion_std = np.std(
    real_feature_sequence_motion,
    axis=0
)

fake_feature_motion_std = np.std(
    fake_feature_sequence_motion,
    axis=0
)


pooled_feature_motion_std = np.sqrt(
    (
        real_feature_motion_std ** 2
        +
        fake_feature_motion_std ** 2
    ) / 2
)


pooled_feature_motion_std = np.where(
    pooled_feature_motion_std == 0,
    1e-8,
    pooled_feature_motion_std
)


feature_motion_separation = (
    feature_motion_difference
    /
    pooled_feature_motion_std
)


print(
    "\nAverage feature temporal separation:",
    f"{np.mean(feature_motion_separation):.6f}"
)

print(
    "Maximum feature temporal separation:",
    f"{np.max(feature_motion_separation):.6f}"
)


# ============================================================
# TOP 20 TEMPORALLY DIFFERENT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURES BY TEMPORAL SEPARATION")
print("=" * 70)


top_indices = np.argsort(
    feature_motion_separation
)[::-1][:20]


for rank, feature_index in enumerate(
    top_indices,
    start=1
):

    print(f"\nRank {rank}")

    print(
        f"Feature index: "
        f"{feature_index}"
    )

    print(
        f"Real motion: "
        f"{real_feature_motion[feature_index]:.6f}"
    )

    print(
        f"Fake motion: "
        f"{fake_feature_motion[feature_index]:.6f}"
    )

    print(
        f"Motion difference: "
        f"{feature_motion_difference[feature_index]:.6f}"
    )

    print(
        f"Temporal separation: "
        f"{feature_motion_separation[feature_index]:.6f}"
    )


# ============================================================
# TEMPORAL VARIABILITY
#
# Standard deviation across time.
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL VARIABILITY")
print("=" * 70)


real_temporal_std = np.std(
    X_real,
    axis=1
)

fake_temporal_std = np.std(
    X_fake,
    axis=1
)


real_variability = np.mean(
    real_temporal_std
)

fake_variability = np.mean(
    fake_temporal_std
)


print(
    f"\nReal average temporal variability: "
    f"{real_variability:.6f}"
)

print(
    f"Fake average temporal variability: "
    f"{fake_variability:.6f}"
)

print(
    f"Difference: "
    f"{abs(fake_variability - real_variability):.6f}"
)


# ============================================================
# INITIAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INITIAL INTERPRETATION")
print("=" * 70)


print("\nStatic feature analysis previously showed:")
print("Weak Real/Fake separation.")


print(
    f"\nTemporal motion separation score: "
    f"{motion_separation_score:.6f}"
)

print(
    f"Maximum feature temporal separation: "
    f"{np.max(feature_motion_separation):.6f}"
)


max_temporal_separation = np.max(
    feature_motion_separation
)


if (
    motion_separation_score < 0.2
    and max_temporal_separation < 0.2
):

    print(
        "\nRESULT:"
    )

    print(
        "Very weak temporal separation detected."
    )

    print(
        "Both static and temporal features appear "
        "weak for Real/Fake discrimination."
    )


elif max_temporal_separation < 0.5:

    print(
        "\nRESULT:"
    )

    print(
        "Weak temporal separation detected."
    )

    print(
        "Temporal information provides limited "
        "additional discrimination."
    )


elif max_temporal_separation < 1.0:

    print(
        "\nRESULT:"
    )

    print(
        "Moderate temporal separation detected."
    )

    print(
        "Some temporal features may help distinguish "
        "Real and Fake sequences."
    )


else:

    print(
        "\nRESULT:"
    )

    print(
        "Strong temporal separation detected."
    )

    print(
        "Temporal patterns contain potentially useful "
        "Real/Fake discrimination information."
    )


print("\n" + "=" * 70)
print("EXP7 TEMPORAL FEATURE ANALYSIS COMPLETED")
print("=" * 70)
