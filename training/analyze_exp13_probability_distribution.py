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
    "lstm_exp13.keras"
)


NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp13_normalization.npz"
)


PREDICTIONS_PATH = (
    MODEL_DIR /
    "lstm_exp13_test_predictions.npz"
)


THRESHOLD = 0.50


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split_name):

    file_path = (
        DATASET_DIR /
        f"{split_name}.npz"
    )


    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )


    data = np.load(
        file_path,
        allow_pickle=True
    )


    X = data[
        "X"
    ].astype(
        np.float32
    )


    y = data[
        "y"
    ].astype(
        np.int32
    )


    video_ids = data[
        "video_ids"
    ]


    return (
        X,
        y,
        video_ids
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(
    X,
    mean,
    std
):

    X_normalized = (
        X - mean
    ) / std


    return X_normalized.astype(
        np.float32
    )


# ============================================================
# VIDEO LEVEL PREDICTIONS
# ============================================================

def get_video_predictions(
    probabilities,
    labels,
    video_ids
):

    unique_videos = np.unique(
        video_ids
    )


    video_probabilities = []

    video_labels = []

    video_names = []


    for video_id in unique_videos:


        indices = (
            video_ids == video_id
        )


        video_probability = np.mean(
            probabilities[indices]
        )


        video_label = labels[
            indices
        ][0]


        video_probabilities.append(
            video_probability
        )


        video_labels.append(
            video_label
        )


        video_names.append(
            video_id
        )


    return (

        np.array(
            video_probabilities
        ),

        np.array(
            video_labels
        ),

        np.array(
            video_names
        )

    )


# ============================================================
# PRINT DISTRIBUTION STATISTICS
# ============================================================

def print_statistics(
    probabilities,
    title
):

    print(
        f"\n{title}"
    )


    print(
        f"Count   : {len(probabilities)}"
    )


    print(
        f"Mean    : {np.mean(probabilities):.4f}"
    )


    print(
        f"Median  : {np.median(probabilities):.4f}"
    )


    print(
        f"Minimum : {np.min(probabilities):.4f}"
    )


    print(
        f"Maximum : {np.max(probabilities):.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():


    print(
        "\n" + "=" * 60
    )


    print(
        "EXP13 PROBABILITY DISTRIBUTION ANALYSIS"
    )


    print(
        "=" * 60
    )


    # ========================================================
    # LOAD TEST DATASET
    # ========================================================

    print(
        "\nLoading test dataset..."
    )


    (
        X_test,
        y_test,
        video_ids
    ) = load_dataset(
        "test"
    )


    print(
        "Test sequences:",
        len(X_test)
    )


    print(
        "Real sequences:",
        np.sum(y_test == 0)
    )


    print(
        "Fake sequences:",
        np.sum(y_test == 1)
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
    # NORMALIZE DATA
    # ========================================================

    print(
        "\nNormalizing test data..."
    )


    X_test = normalize_data(
        X_test,
        mean,
        std
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading Exp13 model..."
    )


    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


    # ========================================================
    # RUN PREDICTIONS
    # ========================================================

    print(
        "\nRunning predictions..."
    )


    probabilities = model.predict(
        X_test,
        verbose=0
    ).flatten()


    # ========================================================
    # SEQUENCE LEVEL DISTRIBUTION
    # ========================================================

    real_probabilities = probabilities[
        y_test == 0
    ]


    fake_probabilities = probabilities[
        y_test == 1
    ]


    print(
        "\n" + "=" * 60
    )


    print(
        "SEQUENCE-LEVEL PROBABILITY DISTRIBUTION"
    )


    print(
        "=" * 60
    )


    print_statistics(
        real_probabilities,
        "REAL SEQUENCES"
    )


    print_statistics(
        fake_probabilities,
        "FAKE SEQUENCES"
    )


    # ========================================================
    # PERCENTILES
    # ========================================================

    print(
        "\n" + "=" * 60
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
    # OVERLAP ANALYSIS
    # ========================================================

    real_as_fake = np.sum(
        real_probabilities >= THRESHOLD
    )


    fake_as_real = np.sum(
        fake_probabilities < THRESHOLD
    )


    print(
        "\n" + "=" * 60
    )


    print(
        f"OVERLAP ANALYSIS AT THRESHOLD {THRESHOLD:.2f}"
    )


    print(
        "=" * 60
    )


    print(
        "\nReal sequences predicted as Fake:"
    )


    print(
        f"{real_as_fake} / "
        f"{len(real_probabilities)}"
    )


    print(
        "\nFake sequences predicted as Real:"
    )


    print(
        f"{fake_as_real} / "
        f"{len(fake_probabilities)}"
    )


    # ========================================================
    # VIDEO LEVEL PREDICTIONS
    # ========================================================

    (
        video_probabilities,
        video_labels,
        video_names
    ) = get_video_predictions(

        probabilities,
        y_test,
        video_ids

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


    # ========================================================
    # VIDEO LEVEL DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 60
    )


    print(
        "VIDEO-LEVEL PROBABILITY DISTRIBUTION"
    )


    print(
        "=" * 60
    )


    print(
        "\nTotal videos:",
        len(video_probabilities)
    )


    print(
        "Real videos :",
        len(real_video_probabilities)
    )


    print(
        "Fake videos :",
        len(fake_video_probabilities)
    )


    print_statistics(
        real_video_probabilities,
        "REAL VIDEOS"
    )


    print_statistics(
        fake_video_probabilities,
        "FAKE VIDEOS"
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    np.savez(

        PREDICTIONS_PATH,

        sequence_probabilities=probabilities,

        sequence_labels=y_test,

        sequence_video_ids=video_ids,

        video_probabilities=video_probabilities,

        video_labels=video_labels,

        video_names=video_names

    )


    print(
        "\n" + "=" * 60
    )


    print(
        "PREDICTIONS SAVED"
    )


    print(
        "=" * 60
    )


    print(
        "\nSaved to:"
    )


    print(
        PREDICTIONS_PATH
    )


    # ========================================================
    # COMPLETED
    # ========================================================

    print(
        "\n" + "=" * 60
    )


    print(
        "EXP13 PROBABILITY DISTRIBUTION ANALYSIS COMPLETED"
    )


    print(
        "=" * 60
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()