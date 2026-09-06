from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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


THRESHOLD_OUTPUT_PATH = (
    MODEL_DIR /
    "lstm_exp13_validation_thresholds.npz"
)


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

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
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
# NORMALIZATION
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


    return (

        np.array(
            video_probabilities
        ),

        np.array(
            video_labels
        )

    )


# ============================================================
# FIND BEST THRESHOLD
#
# Selection criterion:
# Maximum F1 Score
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )


    best_threshold = 0.50

    best_f1 = -1

    best_accuracy = 0

    best_precision = 0

    best_recall = 0

    best_predictions = None


    for threshold in thresholds:


        predictions = (

            probabilities >= threshold

        ).astype(
            np.int32
        )


        accuracy = accuracy_score(
            y_true,
            predictions
        )


        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )


        # Select threshold with highest F1

        if f1 > best_f1:


            best_f1 = f1

            best_threshold = threshold

            best_accuracy = accuracy

            best_precision = precision

            best_recall = recall

            best_predictions = predictions


    return {

        "threshold": best_threshold,

        "accuracy": best_accuracy,

        "precision": best_precision,

        "recall": best_recall,

        "f1": best_f1,

        "predictions": best_predictions

    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results,
    y_true,
    title
):

    print(
        "\n" + "=" * 60
    )

    print(
        title
    )

    print(
        "=" * 60
    )


    print(
        f"\nBest threshold: "
        f"{results['threshold']:.2f}"
    )


    print(
        f"Accuracy       : "
        f"{results['accuracy'] * 100:.2f}%"
    )


    print(
        f"Precision      : "
        f"{results['precision'] * 100:.2f}%"
    )


    print(
        f"Recall         : "
        f"{results['recall'] * 100:.2f}%"
    )


    print(
        f"F1 Score       : "
        f"{results['f1'] * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(

        confusion_matrix(

            y_true,

            results[
                "predictions"
            ]

        )

    )


# ============================================================
# MAIN
# ============================================================

def main():


    print(
        "\n" + "=" * 60
    )

    print(
        "EXP13 VALIDATION THRESHOLD ANALYSIS"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print(
        "\nLoading validation dataset..."
    )


    (

        X_validation,

        y_validation,

        video_ids

    ) = load_dataset(
        "validation"
    )


    print(
        "Validation sequences:",
        len(X_validation)
    )


    print(
        "Validation videos:",
        len(
            np.unique(video_ids)
        )
    )


    print(
        "Real sequences:",
        np.sum(
            y_validation == 0
        )
    )


    print(
        "Fake sequences:",
        np.sum(
            y_validation == 1
        )
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
    # NORMALIZE VALIDATION DATA
    # ========================================================

    print(
        "\nNormalizing validation data..."
    )


    X_validation = normalize_data(

        X_validation,

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
    # RUN VALIDATION PREDICTIONS
    # ========================================================

    print(
        "\nRunning validation predictions..."
    )


    probabilities = model.predict(

        X_validation,

        verbose=0

    ).flatten()


    # ========================================================
    # SEQUENCE LEVEL THRESHOLD
    # ========================================================

    sequence_results = find_best_threshold(

        y_validation,

        probabilities

    )


    print_results(

        sequence_results,

        y_validation,

        "SEQUENCE-LEVEL VALIDATION THRESHOLD SEARCH"

    )


    # ========================================================
    # VIDEO LEVEL PREDICTIONS
    # ========================================================

    (

        video_probabilities,

        video_labels

    ) = get_video_predictions(

        probabilities,

        y_validation,

        video_ids

    )


    # ========================================================
    # VIDEO LEVEL THRESHOLD
    # ========================================================

    video_results = find_best_threshold(

        video_labels,

        video_probabilities

    )


    print_results(

        video_results,

        video_labels,

        "VIDEO-LEVEL VALIDATION THRESHOLD SEARCH"

    )


    # ========================================================
    # SAVE THRESHOLDS
    # ========================================================

    np.savez(

        THRESHOLD_OUTPUT_PATH,

        sequence_threshold=

        sequence_results[
            "threshold"
        ],

        video_threshold=

        video_results[
            "threshold"
        ]

    )


    print(
        "\n" + "=" * 60
    )

    print(
        "VALIDATION THRESHOLDS SAVED"
    )

    print(
        "=" * 60
    )


    print(
        "\nSaved to:"
    )

    print(
        THRESHOLD_OUTPUT_PATH
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "EXP13 VALIDATION THRESHOLD ANALYSIS COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()