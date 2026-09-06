import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp7_source_aware"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("EXP7 FEATURE SEPARATION ANALYSIS")
print("=" * 70)


print("\nLoading training dataset...")

train_data = np.load(
    DATASET_DIR / "train.npz",
    allow_pickle=True
)

X_train = train_data["X"]
y_train = train_data["y"]


print("\nTraining data loaded.")

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

print(
    f"\nReal sequences: {np.sum(y_train == 0)}"
)

print(
    f"Fake sequences: {np.sum(y_train == 1)}"
)


# ============================================================
# SEPARATE REAL AND FAKE
# ============================================================

X_real = X_train[y_train == 0]

X_fake = X_train[y_train == 1]


# ============================================================
# CONVERT SEQUENCES TO FEATURE VECTORS
#
# Average across time dimension.
# ============================================================

real_features = np.mean(
    X_real,
    axis=1
)

fake_features = np.mean(
    X_fake,
    axis=1
)


# ============================================================
# FEATURE MEAN COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MEAN COMPARISON")
print("=" * 70)


real_mean = np.mean(
    real_features,
    axis=0
)

fake_mean = np.mean(
    fake_features,
    axis=0
)


mean_difference = np.abs(
    fake_mean - real_mean
)


print(
    "\nAverage absolute feature difference:",
    f"{np.mean(mean_difference):.6f}"
)

print(
    "Maximum feature difference:",
    f"{np.max(mean_difference):.6f}"
)

print(
    "Minimum feature difference:",
    f"{np.min(mean_difference):.6f}"
)


# ============================================================
# FEATURE STANDARD DEVIATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VARIABILITY")
print("=" * 70)


real_std = np.std(
    real_features,
    axis=0
)

fake_std = np.std(
    fake_features,
    axis=0
)


print(
    "\nAverage Real feature standard deviation:",
    f"{np.mean(real_std):.6f}"
)

print(
    "Average Fake feature standard deviation:",
    f"{np.mean(fake_std):.6f}"
)


# ============================================================
# TOP DIFFERENT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 MOST DIFFERENT FEATURES")
print("=" * 70)


top_indices = np.argsort(
    mean_difference
)[::-1][:20]


for rank, feature_index in enumerate(
    top_indices,
    start=1
):

    print(
        f"\nRank {rank}"
    )

    print(
        f"Feature index: {feature_index}"
    )

    print(
        f"Real mean: {real_mean[feature_index]:.6f}"
    )

    print(
        f"Fake mean: {fake_mean[feature_index]:.6f}"
    )

    print(
        f"Difference: "
        f"{mean_difference[feature_index]:.6f}"
    )


# ============================================================
# FEATURE SEPARATION SCORE
#
# Cohen-like standardized mean difference.
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZED FEATURE SEPARATION")
print("=" * 70)


pooled_std = np.sqrt(
    (
        real_std ** 2
        +
        fake_std ** 2
    ) / 2
)


# Avoid division by zero
pooled_std = np.where(
    pooled_std == 0,
    1e-8,
    pooled_std
)


separation_score = (
    mean_difference / pooled_std
)


print(
    "\nAverage separation score:",
    f"{np.mean(separation_score):.6f}"
)

print(
    "Maximum separation score:",
    f"{np.max(separation_score):.6f}"
)


# ============================================================
# TOP 20 FEATURES BY SEPARATION SCORE
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURES BY SEPARATION SCORE")
print("=" * 70)


top_separation_indices = np.argsort(
    separation_score
)[::-1][:20]


for rank, feature_index in enumerate(
    top_separation_indices,
    start=1
):

    print(
        f"\nRank {rank}"
    )

    print(
        f"Feature index: {feature_index}"
    )

    print(
        f"Real mean: "
        f"{real_mean[feature_index]:.6f}"
    )

    print(
        f"Fake mean: "
        f"{fake_mean[feature_index]:.6f}"
    )

    print(
        f"Mean difference: "
        f"{mean_difference[feature_index]:.6f}"
    )

    print(
        f"Separation score: "
        f"{separation_score[feature_index]:.6f}"
    )


# ============================================================
# OVERALL FEATURE MAGNITUDE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL FEATURE MAGNITUDE")
print("=" * 70)


print(
    "\nReal feature mean magnitude:",
    f"{np.mean(np.abs(real_features)):.6f}"
)

print(
    "Fake feature mean magnitude:",
    f"{np.mean(np.abs(fake_features)):.6f}"
)


# ============================================================
# SIMPLE INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INITIAL INTERPRETATION")
print("=" * 70)


average_score = np.mean(
    separation_score
)

maximum_score = np.max(
    separation_score
)


if maximum_score < 0.2:

    print(
        "\nWARNING:"
    )

    print(
        "Very weak feature separation detected."
    )

    print(
        "Real and Fake samples appear very similar "
        "based on the extracted features."
    )


elif maximum_score < 0.5:

    print(
        "\nRESULT:"
    )

    print(
        "Weak feature separation detected."
    )

    print(
        "The features contain limited information "
        "for distinguishing Real and Fake samples."
    )


elif maximum_score < 1.0:

    print(
        "\nRESULT:"
    )

    print(
        "Moderate feature separation detected."
    )

    print(
        "Some features may contain useful "
        "Real/Fake discrimination information."
    )


else:

    print(
        "\nRESULT:"
    )

    print(
        "Strong feature separation detected."
    )

    print(
        "Some features show meaningful differences "
        "between Real and Fake samples."
    )


print("\n" + "=" * 70)
print("EXP7 FEATURE SEPARATION ANALYSIS COMPLETED")
print("=" * 70)
