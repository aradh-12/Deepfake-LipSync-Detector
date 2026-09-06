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
# Configuration
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp7_source_aware"
)

MODEL_PATH = Path(
    "saved_models/lstm_exp7.keras"
)

NORMALIZATION_PATH = Path(
    "saved_models/lstm_exp7_normalization.npz"
)


# ============================================================
# Thresholds to Analyze
# ============================================================

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


# ============================================================
# Load Validation Dataset
# ============================================================

def load_validation_data():

    dataset_path = (
        DATASET_DIR /
        "validation.npz"
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{dataset_path}"
        )

    data = np.load(

        dataset_path,

        allow_pickle=True

    )

    X = data["X"]

    y = data["y"]

    video_ids = data["video_ids"]

    return X, y, video_ids


# ============================================================
# Normalize Data
# ============================================================

def normalize_data(X):

    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(
            f"Normalization file not found: "
            f"{NORMALIZATION_PATH}"
        )

    normalization_data = np.load(
        NORMALIZATION_PATH
    )

    mean = normalization_data["mean"]

    std = normalization_data["std"]

    X_normalized = (
        X - mean
    ) / std

    return X_normalized.astype(
        np.float32
    )


# ============================================================
# Sequence-Level Threshold Evaluation
# ============================================================

def evaluate_threshold(

    y_true,

    probabilities,

    threshold

):

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

    cm = confusion_matrix(

        y_true,

        predictions

    )

    return (

        accuracy,

        precision,

        recall,

        f1,

        cm

    )


# ============================================================
# Video-Level Aggregation
# ============================================================

def aggregate_video_predictions(

    video_ids,

    y_true,

    probabilities

):

    unique_videos = np.unique(
        video_ids
    )

    video_labels = []

    video_probabilities = []

    for video in unique_videos:

        indices = np.where(

            video_ids == video

        )[0]

        labels = y_true[
            indices
        ]

        probabilities_for_video = (
            probabilities[
                indices
            ]
        )

        unique_labels = np.unique(
            labels
        )

        if len(unique_labels) != 1:

            raise RuntimeError(

                f"Video {video} has "
                f"multiple labels: "
                f"{unique_labels}"

            )

        video_labels.append(

            unique_labels[0]

        )

        # Average probability across
        # all sequences of this video

        video_probability = np.mean(

            probabilities_for_video

        )

        video_probabilities.append(

            video_probability

        )

    return (

        np.asarray(
            video_labels,
            dtype=np.int32
        ),

        np.asarray(
            video_probabilities,
            dtype=np.float32
        ),

        unique_videos

    )


# ============================================================
# Print Threshold Results
# ============================================================

def print_results(

    threshold,

    sequence_results,

    video_results

):

    (

        seq_accuracy,

        seq_precision,

        seq_recall,

        seq_f1,

        _

    ) = sequence_results

    (

        video_accuracy,

        video_precision,

        video_recall,

        video_f1,

        _

    ) = video_results

    print(

        f"\nThreshold: {threshold:.2f}"

    )

    print("-" * 50)

    print("SEQUENCE LEVEL")

    print(

        f"Accuracy  : "
        f"{seq_accuracy * 100:.2f}%"

    )

    print(

        f"Precision : "
        f"{seq_precision * 100:.2f}%"

    )

    print(

        f"Recall    : "
        f"{seq_recall * 100:.2f}%"

    )

    print(

        f"F1 Score  : "
        f"{seq_f1 * 100:.2f}%"

    )

    print("\nVIDEO LEVEL")

    print(

        f"Accuracy  : "
        f"{video_accuracy * 100:.2f}%"

    )

    print(

        f"Precision : "
        f"{video_precision * 100:.2f}%"

    )

    print(

        f"Recall    : "
        f"{video_recall * 100:.2f}%"

    )

    print(

        f"F1 Score  : "
        f"{video_f1 * 100:.2f}%"

    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)

    print(
        "EXP7 VALIDATION THRESHOLD ANALYSIS"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load Validation Data
    # --------------------------------------------------------

    print(
        "\nLoading validation dataset..."
    )

    X_validation, y_validation, video_ids = (

        load_validation_data()

    )

    print(

        "Validation sequences:",

        len(X_validation)

    )

    print(

        "Validation videos:",

        len(np.unique(video_ids))

    )

    print(

        "Real sequences:",

        np.sum(y_validation == 0)

    )

    print(

        "Fake sequences:",

        np.sum(y_validation == 1)

    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print(
        "\nNormalizing validation data..."
    )

    X_validation = normalize_data(
        X_validation
    )

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    print(
        "\nLoading Exp7 model..."
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model not found: "

            f"{MODEL_PATH}"

        )

    model = tf.keras.models.load_model(

        MODEL_PATH

    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        "\nRunning predictions..."
    )

    probabilities = model.predict(

        X_validation,

        verbose=0

    ).flatten()

    # --------------------------------------------------------
    # Aggregate Video Predictions
    # --------------------------------------------------------

    (

        video_labels,

        video_probabilities,

        unique_videos

    ) = aggregate_video_predictions(

        video_ids,

        y_validation,

        probabilities

    )

    # --------------------------------------------------------
    # Store Results
    # --------------------------------------------------------

    all_results = []

    # --------------------------------------------------------
    # Test Every Threshold
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "THRESHOLD ANALYSIS"
    )

    print("=" * 60)

    for threshold in THRESHOLDS:

        sequence_results = (

            evaluate_threshold(

                y_validation,

                probabilities,

                threshold

            )

        )

        video_results = (

            evaluate_threshold(

                video_labels,

                video_probabilities,

                threshold

            )

        )

        print_results(

            threshold,

            sequence_results,

            video_results

        )

        all_results.append({

            "threshold": threshold,

            "sequence_accuracy":
                sequence_results[0],

            "sequence_precision":
                sequence_results[1],

            "sequence_recall":
                sequence_results[2],

            "sequence_f1":
                sequence_results[3],

            "video_accuracy":
                video_results[0],

            "video_precision":
                video_results[1],

            "video_recall":
                video_results[2],

            "video_f1":
                video_results[3]

        })

    # ========================================================
    # Find Best Threshold
    #
    # PRIMARY METRIC:
    # VIDEO-LEVEL F1 SCORE
    # ========================================================

    best_result = max(

        all_results,

        key=lambda result:

        result["video_f1"]

    )

    print("\n" + "=" * 60)

    print(
        "BEST VALIDATION THRESHOLD"
    )

    print("=" * 60)

    print(

        f"\nSelected threshold: "

        f"{best_result['threshold']:.2f}"

    )

    print(

        "\nSelection metric:"

    )

    print(

        "VIDEO-LEVEL F1 SCORE"

    )

    print(

        f"\nVideo Accuracy  : "

        f"{best_result['video_accuracy'] * 100:.2f}%"

    )

    print(

        f"Video Precision : "

        f"{best_result['video_precision'] * 100:.2f}%"

    )

    print(

        f"Video Recall    : "

        f"{best_result['video_recall'] * 100:.2f}%"

    )

    print(

        f"Video F1 Score  : "

        f"{best_result['video_f1'] * 100:.2f}%"

    )

    print("\n" + "=" * 60)

    print(

        "IMPORTANT:"
    )

    print(

        "Threshold selected using "
        "VALIDATION DATA ONLY."

    )

    print(

        "Do NOT use test data "
        "to select the threshold."

    )

    print("=" * 60)

    print(
        "\nEXP7 VALIDATION THRESHOLD "
        "ANALYSIS COMPLETED"
    )

    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
