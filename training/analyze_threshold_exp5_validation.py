from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Configuration
# ============================================================

DATASET_FOLDER = Path(
    "outputs/dataset_exp5_full"
)


VALIDATION_FILE = (
    DATASET_FOLDER /
    "validation.npz"
)


MODEL_FOLDER = Path(
    "models"
)


MODEL_FILE = (
    MODEL_FOLDER /
    "lstm_exp5_best.keras"
)


NORMALIZATION_FILE = (
    MODEL_FOLDER /
    "lstm_exp5_normalization.npz"
)


# ============================================================
# Model Configuration
# ============================================================

SEQUENCE_LENGTH = 30

FEATURE_SIZE = 199


# ============================================================
# Thresholds to Test
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
# Load Dataset
# ============================================================

def load_dataset(
    dataset_file
):

    if not dataset_file.exists():

        raise FileNotFoundError(
            f"Dataset file not found: "
            f"{dataset_file}"
        )


    data = np.load(
        dataset_file,
        allow_pickle=True
    )


    X = data["X"].astype(
        np.float32
    )


    y = data["y"].astype(
        np.int32
    )


    video_ids = data["video_ids"]


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            f"Expected X to have 3 dimensions, "
            f"got {X.ndim}"
        )


    if X.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"got {X.shape[1]}"
        )


    if X.shape[2] != FEATURE_SIZE:

        raise ValueError(
            f"Expected feature size "
            f"{FEATURE_SIZE}, "
            f"got {X.shape[2]}"
        )


    if len(X) != len(y):

        raise ValueError(
            "X and y length mismatch."
        )


    if len(X) != len(video_ids):

        raise ValueError(
            "X and video_ids length mismatch."
        )


    # --------------------------------------------------------
    # Check for NaN
    # --------------------------------------------------------

    if np.isnan(X).any():

        raise ValueError(
            "NaN values found in X."
        )


    if np.isinf(X).any():

        raise ValueError(
            "Inf values found in X."
        )


    return (
        X,
        y,
        video_ids
    )


# ============================================================
# Load Normalization
# ============================================================

def load_normalization():

    if not NORMALIZATION_FILE.exists():

        raise FileNotFoundError(
            f"Normalization file not found: "
            f"{NORMALIZATION_FILE}"
        )


    data = np.load(
        NORMALIZATION_FILE
    )


    mean = data["mean"].astype(
        np.float32
    )


    std = data["std"].astype(
        np.float32
    )


    return (
        mean,
        std
    )


# ============================================================
# Normalize Features
# ============================================================

def normalize_features(
    X,
    mean,
    std
):

    std = np.maximum(
        std,
        1e-6
    )


    X_normalized = (
        X - mean
    ) / std


    return X_normalized.astype(
        np.float32
    )


# ============================================================
# Get Video-Level Probabilities
# ============================================================

def get_video_level_data(
    probabilities,
    y_true,
    video_ids
):

    unique_video_ids = np.unique(
        video_ids
    )


    video_true_labels = []

    video_probabilities = []


    for video_id in unique_video_ids:


        indices = np.where(
            video_ids == video_id
        )[0]


        video_probs = probabilities[
            indices
        ]


        video_labels = y_true[
            indices
        ]


        # ----------------------------------------------------
        # Average probabilities from all sequences
        # belonging to the same video.
        # ----------------------------------------------------

        average_probability = np.mean(
            video_probs
        )


        unique_labels = np.unique(
            video_labels
        )


        if len(unique_labels) != 1:

            raise ValueError(
                f"Inconsistent labels found "
                f"for video: {video_id}"
            )


        true_label = int(
            unique_labels[0]
        )


        video_true_labels.append(
            true_label
        )


        video_probabilities.append(
            float(average_probability)
        )


    return (

        np.array(
            video_true_labels
        ),

        np.array(
            video_probabilities
        )

    )


# ============================================================
# Calculate Metrics
# ============================================================

def calculate_metrics(
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


    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# Print Results
# ============================================================

def print_results(
    threshold_results
):

    print(
        "\n" + "=" * 105
    )


    print(
        "EXP5 VALIDATION THRESHOLD ANALYSIS"
    )


    print(
        "=" * 105
    )


    print(
        f"{'Threshold':<12}"
        f"{'Seq Acc':<12}"
        f"{'Seq Prec':<12}"
        f"{'Seq Recall':<12}"
        f"{'Seq F1':<12}"
        f"{'Vid Acc':<12}"
        f"{'Vid F1':<12}"
    )


    print(
        "-" * 105
    )


    for result in threshold_results:


        threshold = result["threshold"]


        sequence_accuracy = result[
            "sequence_accuracy"
        ]


        sequence_precision = result[
            "sequence_precision"
        ]


        sequence_recall = result[
            "sequence_recall"
        ]


        sequence_f1 = result[
            "sequence_f1"
        ]


        video_accuracy = result[
            "video_accuracy"
        ]


        video_f1 = result[
            "video_f1"
        ]


        print(

            f"{threshold:<12.2f}"
            f"{sequence_accuracy * 100:<12.2f}"
            f"{sequence_precision * 100:<12.2f}"
            f"{sequence_recall * 100:<12.2f}"
            f"{sequence_f1 * 100:<12.2f}"
            f"{video_accuracy * 100:<12.2f}"
            f"{video_f1 * 100:<12.2f}"

        )


    print(
        "=" * 105
    )


# ============================================================
# Find Best Threshold
# ============================================================

def find_best_threshold(
    threshold_results
):

    # --------------------------------------------------------
    # Primary metric:
    # Video-level F1 score
    #
    # Secondary metric:
    # Video-level accuracy
    # --------------------------------------------------------

    best_result = max(

        threshold_results,

        key=lambda result: (

            result["video_f1"],

            result["video_accuracy"]

        )

    )


    return best_result


# ============================================================
# Main
# ============================================================

def main():


    # --------------------------------------------------------
    # Load Validation Dataset
    # --------------------------------------------------------

    print(
        "\nLoading Exp5 validation dataset..."
    )


    (
        X_validation,
        y_validation,
        validation_video_ids

    ) = load_dataset(
        VALIDATION_FILE
    )


    print(
        "Validation sequences:",
        len(X_validation)
    )


    print(
        "Validation videos:",

        len(
            np.unique(
                validation_video_ids
            )
        )

    )


    print(
        "Real sequences:",
        int(
            np.sum(
                y_validation == 0
            )
        )
    )


    print(
        "Fake sequences:",
        int(
            np.sum(
                y_validation == 1
            )
        )
    )


    # --------------------------------------------------------
    # Load Normalization
    # --------------------------------------------------------

    print(
        "\nLoading Exp5 normalization..."
    )


    (
        feature_mean,
        feature_std

    ) = load_normalization()


    # --------------------------------------------------------
    # Normalize Validation Data
    # --------------------------------------------------------

    print(
        "Normalizing validation data..."
    )


    X_validation = normalize_features(

        X_validation,

        feature_mean,

        feature_std

    )


    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model file not found: "
            f"{MODEL_FILE}"
        )


    print(
        "\nLoading Exp5 model..."
    )


    model = tf.keras.models.load_model(
        MODEL_FILE
    )


    # --------------------------------------------------------
    # Run Predictions Once
    # --------------------------------------------------------

    print(
        "\nRunning predictions..."
    )


    sequence_probabilities = model.predict(

        X_validation,

        verbose=0

    )


    sequence_probabilities = sequence_probabilities.reshape(
        -1
    )


    # --------------------------------------------------------
    # Get Video-Level Probabilities
    # --------------------------------------------------------

    (
        video_true_labels,
        video_probabilities

    ) = get_video_level_data(

        sequence_probabilities,

        y_validation,

        validation_video_ids

    )


    # --------------------------------------------------------
    # Analyze Thresholds
    # --------------------------------------------------------

    threshold_results = []


    for threshold in THRESHOLDS:


        # ----------------------------------------------------
        # Sequence-Level Metrics
        # ----------------------------------------------------

        (

            sequence_accuracy,
            sequence_precision,
            sequence_recall,
            sequence_f1

        ) = calculate_metrics(

            y_validation,

            sequence_probabilities,

            threshold

        )


        # ----------------------------------------------------
        # Video-Level Metrics
        # ----------------------------------------------------

        (

            video_accuracy,
            video_precision,
            video_recall,
            video_f1

        ) = calculate_metrics(

            video_true_labels,

            video_probabilities,

            threshold

        )


        result = {

            "threshold":
                threshold,

            "sequence_accuracy":
                sequence_accuracy,

            "sequence_precision":
                sequence_precision,

            "sequence_recall":
                sequence_recall,

            "sequence_f1":
                sequence_f1,

            "video_accuracy":
                video_accuracy,

            "video_precision":
                video_precision,

            "video_recall":
                video_recall,

            "video_f1":
                video_f1

        }


        threshold_results.append(
            result
        )


    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print_results(
        threshold_results
    )


    # --------------------------------------------------------
    # Find Best Threshold
    # --------------------------------------------------------

    best_result = find_best_threshold(
        threshold_results
    )


    # --------------------------------------------------------
    # Print Best Threshold
    # --------------------------------------------------------

    print(
        "\nBEST VALIDATION THRESHOLD"
    )


    print(
        "=" * 60
    )


    print(
        f"Threshold           : "
        f"{best_result['threshold']:.2f}"
    )


    print(
        f"Sequence Accuracy   : "
        f"{best_result['sequence_accuracy'] * 100:.2f}%"
    )


    print(
        f"Sequence F1 Score   : "
        f"{best_result['sequence_f1'] * 100:.2f}%"
    )


    print(
        f"Video Accuracy      : "
        f"{best_result['video_accuracy'] * 100:.2f}%"
    )


    print(
        f"Video Precision     : "
        f"{best_result['video_precision'] * 100:.2f}%"
    )


    print(
        f"Video Recall        : "
        f"{best_result['video_recall'] * 100:.2f}%"
    )


    print(
        f"Video F1 Score      : "
        f"{best_result['video_f1'] * 100:.2f}%"
    )


    print(
        "=" * 60
    )


    print(
        "\nVALIDATION THRESHOLD ANALYSIS COMPLETED"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
    