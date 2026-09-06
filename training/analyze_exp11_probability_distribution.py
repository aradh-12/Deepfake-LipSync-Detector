from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp8_temporal"
)

MODEL_DIR = Path(
    "saved_models"
)

MODEL_PATH = (
    MODEL_DIR /
    "lstm_exp11.keras"
)

NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp11_normalization.npz"
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    file_path = (
        DATASET_DIR /
        "test.npz"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Test dataset not found: {file_path}"
        )

    data = np.load(
        file_path,
        allow_pickle=True
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.int32
    )

    video_ids = data[
        "video_ids"
    ]

    return X, y, video_ids


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" +
        "=" * 60
    )

    print(
        "EXP11 PROBABILITY DISTRIBUTION ANALYSIS"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print(
        "\nLoading test dataset..."
    )

    X_test, y_test, video_ids = (
        load_test_dataset()
    )

    print(
        f"Test sequences: {len(X_test)}"
    )

    print(
        f"Real sequences: {np.sum(y_test == 0)}"
    )

    print(
        f"Fake sequences: {np.sum(y_test == 1)}"
    )


    # ========================================================
    # LOAD NORMALIZATION
    # ========================================================

    print(
        "\nLoading normalization statistics..."
    )

    normalization_data = np.load(
        NORMALIZATION_PATH
    )

    mean = normalization_data[
        "mean"
    ]

    std = normalization_data[
        "std"
    ]


    # ========================================================
    # NORMALIZE TEST DATA
    # ========================================================

    print(
        "\nNormalizing test data..."
    )

    X_test = (
        X_test - mean
    ) / (
        std + 1e-8
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading Exp11 model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(
        "\nRunning predictions..."
    )

    probabilities = model.predict(
        X_test,
        verbose=0
    ).flatten()


    # ========================================================
    # SEPARATE REAL AND FAKE
    # ========================================================

    real_probabilities = probabilities[
        y_test == 0
    ]

    fake_probabilities = probabilities[
        y_test == 1
    ]


    # ========================================================
    # SEQUENCE LEVEL ANALYSIS
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        "SEQUENCE-LEVEL PROBABILITY DISTRIBUTION"
    )

    print(
        "=" * 60
    )


    print(
        "\nREAL SEQUENCES"
    )

    print(
        f"Count   : {len(real_probabilities)}"
    )

    print(
        f"Mean    : {np.mean(real_probabilities):.4f}"
    )

    print(
        f"Median  : {np.median(real_probabilities):.4f}"
    )

    print(
        f"Minimum : {np.min(real_probabilities):.4f}"
    )

    print(
        f"Maximum : {np.max(real_probabilities):.4f}"
    )


    print(
        "\nFAKE SEQUENCES"
    )

    print(
        f"Count   : {len(fake_probabilities)}"
    )

    print(
        f"Mean    : {np.mean(fake_probabilities):.4f}"
    )

    print(
        f"Median  : {np.median(fake_probabilities):.4f}"
    )

    print(
        f"Minimum : {np.min(fake_probabilities):.4f}"
    )

    print(
        f"Maximum : {np.max(fake_probabilities):.4f}"
    )


    # ========================================================
    # PERCENTILES
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        "PROBABILITY PERCENTILES"
    )

    print(
        "=" * 60
    )

    percentiles = [
        0,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        100
    ]

    print(
        "\nPercentile | Real     | Fake"
    )

    print(
        "-" * 38
    )

    for percentile in percentiles:

        real_value = np.percentile(
            real_probabilities,
            percentile
        )

        fake_value = np.percentile(
            fake_probabilities,
            percentile
        )

        print(
            f"{percentile:>10}% | "
            f"{real_value:.4f} | "
            f"{fake_value:.4f}"
        )


    # ========================================================
    # THRESHOLD 0.5 ANALYSIS
    # ========================================================

    threshold = 0.5

    real_above_threshold = np.sum(
        real_probabilities >= threshold
    )

    fake_below_threshold = np.sum(
        fake_probabilities < threshold
    )


    print(
        "\n" +
        "=" * 60
    )

    print(
        "OVERLAP ANALYSIS AT THRESHOLD 0.50"
    )

    print(
        "=" * 60
    )


    print(
        "\nReal sequences predicted as Fake:"
    )

    print(
        f"{real_above_threshold} / "
        f"{len(real_probabilities)}"
    )


    print(
        "\nFake sequences predicted as Real:"
    )

    print(
        f"{fake_below_threshold} / "
        f"{len(fake_probabilities)}"
    )


    # ========================================================
    # VIDEO LEVEL AGGREGATION
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        "VIDEO-LEVEL PROBABILITY DISTRIBUTION"
    )

    print(
        "=" * 60
    )


    unique_video_ids = np.unique(
        video_ids
    )

    video_probabilities = []
    video_labels = []


    for video_id in unique_video_ids:

        mask = (
            video_ids == video_id
        )

        video_probability = np.mean(
            probabilities[mask]
        )

        video_label = y_test[
            mask
        ][0]

        video_probabilities.append(
            video_probability
        )

        video_labels.append(
            video_label
        )


    video_probabilities = np.array(
        video_probabilities
    )

    video_labels = np.array(
        video_labels
    )


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


    print(
        f"\nTotal videos: {len(unique_video_ids)}"
    )

    print(
        f"Real videos : "
        f"{len(real_video_probabilities)}"
    )

    print(
        f"Fake videos : "
        f"{len(fake_video_probabilities)}"
    )


    print(
        "\nREAL VIDEOS"
    )

    print(
        f"Mean    : "
        f"{np.mean(real_video_probabilities):.4f}"
    )

    print(
        f"Median  : "
        f"{np.median(real_video_probabilities):.4f}"
    )

    print(
        f"Minimum : "
        f"{np.min(real_video_probabilities):.4f}"
    )

    print(
        f"Maximum : "
        f"{np.max(real_video_probabilities):.4f}"
    )


    print(
        "\nFAKE VIDEOS"
    )

    print(
        f"Mean    : "
        f"{np.mean(fake_video_probabilities):.4f}"
    )

    print(
        f"Median  : "
        f"{np.median(fake_video_probabilities):.4f}"
    )

    print(
        f"Minimum : "
        f"{np.min(fake_video_probabilities):.4f}"
    )

    print(
        f"Maximum : "
        f"{np.max(fake_video_probabilities):.4f}"
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    output_path = (
        MODEL_DIR /
        "lstm_exp11_test_predictions.npz"
    )


    np.savez(

        output_path,

        sequence_probabilities=probabilities,

        sequence_labels=y_test,

        sequence_video_ids=video_ids,

        video_probabilities=video_probabilities,

        video_labels=video_labels,

        video_ids=unique_video_ids
    )


    print(
        "\n" +
        "=" * 60
    )

    print(
        "PREDICTIONS SAVED"
    )

    print(
        "=" * 60
    )

    print(
        f"\nSaved to:"
    )

    print(
        output_path
    )


    print(
        "\n" +
        "=" * 60
    )

    print(
        "EXP11 PROBABILITY DISTRIBUTION ANALYSIS COMPLETED"
    )

    print(
        "=" * 60 +
        "\n"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()