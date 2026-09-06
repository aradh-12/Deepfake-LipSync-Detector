from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATASET = Path(
    "outputs/dataset_exp7_source_aware/test.npz"
)

MODEL_FILE = Path(
    "saved_models/lstm_exp7.keras"
)

NORMALIZATION_FILE = Path(
    "saved_models/lstm_exp7_normalization.npz"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("EXP7 TEST PROBABILITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    data = np.load(
        TEST_DATASET,
        allow_pickle=True
    )

    X_test = data["X"]
    y_test = data["y"]
    video_ids = data["video_ids"]

    print(
        "Test sequences:",
        len(X_test)
    )

    print(
        "Test videos:",
        len(np.unique(video_ids))
    )

    print(
        "Real sequences:",
        np.sum(y_test == 0)
    )

    print(
        "Fake sequences:",
        np.sum(y_test == 1)
    )


    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    print("\nLoading normalization statistics...")

    normalization = np.load(
        NORMALIZATION_FILE
    )

    mean = normalization["mean"]
    std = normalization["std"]

    # Prevent division by zero
    std = np.where(
        std == 0,
        1,
        std
    )

    print(
        "Mean shape:",
        mean.shape
    )

    print(
        "Std shape:",
        std.shape
    )


    # --------------------------------------------------------
    # Normalize test data
    # --------------------------------------------------------

    print("\nNormalizing test data...")

    X_test_normalized = (
        X_test - mean
    ) / std


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Exp7 model...")

    model = tf.keras.models.load_model(
        MODEL_FILE
    )


    # --------------------------------------------------------
    # Run predictions
    # --------------------------------------------------------

    print("\nRunning predictions...")

    probabilities = model.predict(
        X_test_normalized,
        verbose=0
    ).flatten()


    # ========================================================
    # SEQUENCE-LEVEL PROBABILITY ANALYSIS
    # ========================================================

    real_probabilities = probabilities[
        y_test == 0
    ]

    fake_probabilities = probabilities[
        y_test == 1
    ]


    print("\n" + "=" * 70)
    print("SEQUENCE-LEVEL PROBABILITY DISTRIBUTION")
    print("=" * 70)


    # --------------------------------------------------------
    # Real sequences
    # --------------------------------------------------------

    print("\nREAL SEQUENCES")

    print(
        "Count:",
        len(real_probabilities)
    )

    print(
        "Mean probability:",
        f"{np.mean(real_probabilities):.4f}"
    )

    print(
        "Median probability:",
        f"{np.median(real_probabilities):.4f}"
    )

    print(
        "Minimum probability:",
        f"{np.min(real_probabilities):.4f}"
    )

    print(
        "Maximum probability:",
        f"{np.max(real_probabilities):.4f}"
    )

    print(
        "Standard deviation:",
        f"{np.std(real_probabilities):.4f}"
    )


    # --------------------------------------------------------
    # Fake sequences
    # --------------------------------------------------------

    print("\nFAKE SEQUENCES")

    print(
        "Count:",
        len(fake_probabilities)
    )

    print(
        "Mean probability:",
        f"{np.mean(fake_probabilities):.4f}"
    )

    print(
        "Median probability:",
        f"{np.median(fake_probabilities):.4f}"
    )

    print(
        "Minimum probability:",
        f"{np.min(fake_probabilities):.4f}"
    )

    print(
        "Maximum probability:",
        f"{np.max(fake_probabilities):.4f}"
    )

    print(
        "Standard deviation:",
        f"{np.std(fake_probabilities):.4f}"
    )


    # ========================================================
    # CLASS SEPARATION ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("CLASS SEPARATION ANALYSIS")
    print("=" * 70)

    real_mean = np.mean(
        real_probabilities
    )

    fake_mean = np.mean(
        fake_probabilities
    )

    difference = (
        fake_mean - real_mean
    )

    print(
        "\nMean Real probability:",
        f"{real_mean:.4f}"
    )

    print(
        "Mean Fake probability:",
        f"{fake_mean:.4f}"
    )

    print(
        "Difference (Fake - Real):",
        f"{difference:.4f}"
    )


    # --------------------------------------------------------
    # Range overlap
    # --------------------------------------------------------

    real_min = np.min(
        real_probabilities
    )

    real_max = np.max(
        real_probabilities
    )

    fake_min = np.min(
        fake_probabilities
    )

    fake_max = np.max(
        fake_probabilities
    )

    overlap_start = max(
        real_min,
        fake_min
    )

    overlap_end = min(
        real_max,
        fake_max
    )

    print("\nREAL RANGE")

    print(
        f"{real_min:.4f} → "
        f"{real_max:.4f}"
    )

    print("\nFAKE RANGE")

    print(
        f"{fake_min:.4f} → "
        f"{fake_max:.4f}"
    )

    print("\nOVERLAP RANGE")

    if overlap_start <= overlap_end:

        print(
            f"{overlap_start:.4f} → "
            f"{overlap_end:.4f}"
        )

        print(
            "\nWARNING: "
            "Real and Fake probabilities overlap."
        )

    else:

        print(
            "NO RANGE OVERLAP"
        )

        print(
            "\nGOOD: "
            "Classes are separated by probability."
        )


    # ========================================================
    # SORTED SEQUENCE PROBABILITIES
    # ========================================================

    print("\n" + "=" * 70)
    print("SORTED SEQUENCE PREDICTIONS")
    print("=" * 70)

    sorted_indices = np.argsort(
        probabilities
    )

    print()

    for index in sorted_indices:

        label = y_test[index]

        actual = (
            "Fake"
            if label == 1
            else "Real"
        )

        probability = probabilities[
            index
        ]

        print(
            f"{actual:4s} | "
            f"{probability:.4f} | "
            f"{video_ids[index]}"
        )


    # ========================================================
    # VIDEO-LEVEL PROBABILITY ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("VIDEO-LEVEL PROBABILITY DISTRIBUTION")
    print("=" * 70)

    unique_videos = np.unique(
        video_ids
    )

    video_probabilities = []
    video_labels = []
    video_names = []


    # --------------------------------------------------------
    # Average sequence probabilities per video
    # --------------------------------------------------------

    for video_id in unique_videos:

        indices = np.where(
            video_ids == video_id
        )[0]

        video_probability = np.mean(
            probabilities[indices]
        )

        video_label = y_test[
            indices[0]
        ]

        video_probabilities.append(
            video_probability
        )

        video_labels.append(
            video_label
        )

        video_names.append(
            video_id
        )


    video_probabilities = np.asarray(
        video_probabilities
    )

    video_labels = np.asarray(
        video_labels
    )


    # --------------------------------------------------------
    # Separate Real and Fake videos
    # --------------------------------------------------------

    real_video_probabilities = (
        video_probabilities[
            video_labels == 0
        ]
    )

    fake_video_probabilities = (
        video_probabilities[
            video_labels == 1
        ]
    )


    print("\nREAL VIDEOS")

    print(
        "Count:",
        len(real_video_probabilities)
    )

    print(
        "Mean probability:",
        f"{np.mean(real_video_probabilities):.4f}"
    )

    print(
        "Minimum probability:",
        f"{np.min(real_video_probabilities):.4f}"
    )

    print(
        "Maximum probability:",
        f"{np.max(real_video_probabilities):.4f}"
    )


    print("\nFAKE VIDEOS")

    print(
        "Count:",
        len(fake_video_probabilities)
    )

    print(
        "Mean probability:",
        f"{np.mean(fake_video_probabilities):.4f}"
    )

    print(
        "Minimum probability:",
        f"{np.min(fake_video_probabilities):.4f}"
    )

    print(
        "Maximum probability:",
        f"{np.max(fake_video_probabilities):.4f}"
    )


    # ========================================================
    # SORTED VIDEO PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("SORTED VIDEO PREDICTIONS")
    print("=" * 70)

    sorted_video_indices = np.argsort(
        video_probabilities
    )

    print()

    for index in sorted_video_indices:

        actual = (
            "Fake"
            if video_labels[index] == 1
            else "Real"
        )

        probability = (
            video_probabilities[index]
        )

        print(
            f"{actual:4s} | "
            f"{probability:.4f} | "
            f"{video_names[index]}"
        )


    # ========================================================
    # THRESHOLD POSITION ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("THRESHOLD POSITION ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60
    ]

    for threshold in thresholds:

        real_predicted_fake = np.sum(
            real_video_probabilities
            >= threshold
        )

        fake_predicted_fake = np.sum(
            fake_video_probabilities
            >= threshold
        )

        print(
            f"\nThreshold: {threshold:.2f}"
        )

        print(
            "Real videos predicted Fake:",
            f"{real_predicted_fake}"
            f"/{len(real_video_probabilities)}"
        )

        print(
            "Fake videos predicted Fake:",
            f"{fake_predicted_fake}"
            f"/{len(fake_video_probabilities)}"
        )


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        "\nThe model outputs a Fake probability."
    )

    print(
        "Ideally:"
    )

    print(
        "Real videos → probabilities close to 0"
    )

    print(
        "Fake videos → probabilities close to 1"
    )

    print(
        "\nIf the Real and Fake probability ranges "
        "strongly overlap, the model does not have "
        "strong class separation."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This analysis is for understanding the model."
    )

    print(
        "It must NOT be used to select a new final "
        "threshold because this is TEST data."
    )


    print("\n" + "=" * 70)
    print("EXP7 TEST PROBABILITY ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()