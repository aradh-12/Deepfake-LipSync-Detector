from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
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


THRESHOLD_PATH = (
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
# CALCULATE VIDEO-LEVEL PREDICTIONS
# ============================================================

def calculate_video_predictions(

    probabilities,

    labels,

    video_ids

):

    unique_video_ids = []

    video_probabilities = []

    video_labels = []


    for video_id in np.unique(
        video_ids
    ):


        indices = (

            video_ids == video_id

        )


        video_probability = np.mean(

            probabilities[indices]

        )


        video_label = labels[
            indices
        ][0]


        unique_video_ids.append(
            video_id
        )


        video_probabilities.append(
            video_probability
        )


        video_labels.append(
            video_label
        )


    return (

        np.array(
            unique_video_ids
        ),

        np.array(
            video_probabilities
        ),

        np.array(
            video_labels
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
        "EXP13 FINAL TEST EVALUATION"
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
        "Test videos:",
        len(
            np.unique(
                video_ids
            )
        )
    )


    print(
        "Real sequences:",
        np.sum(
            y_test == 0
        )
    )


    print(
        "Fake sequences:",
        np.sum(
            y_test == 1
        )
    )


    # ========================================================
    # LOAD NORMALIZATION STATISTICS
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


    X_test = normalize_data(

        X_test,

        mean,

        std

    )


    # ========================================================
    # LOAD VALIDATION-SELECTED THRESHOLDS
    # ========================================================

    print(
        "\nLoading validation-selected thresholds..."
    )


    threshold_data = np.load(

        THRESHOLD_PATH

    )


    sequence_threshold = float(

        threshold_data[
            "sequence_threshold"
        ]

    )


    video_threshold = float(

        threshold_data[
            "video_threshold"
        ]

    )


    print(
        "Sequence threshold:",
        f"{sequence_threshold:.2f}"
    )


    print(
        "Video threshold   :",
        f"{video_threshold:.2f}"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading Exp13 model..."
    )


    model = tf.keras.models.load_model(

        MODEL_PATH

    )


    # ========================================================
    # RUN TEST PREDICTIONS
    # ========================================================

    print(
        "\nRunning test predictions..."
    )


    probabilities = model.predict(

        X_test,

        verbose=0

    ).flatten()


    # ========================================================
    # SEQUENCE-LEVEL PREDICTIONS
    # ========================================================

    sequence_predictions = (

        probabilities >= sequence_threshold

    ).astype(
        int
    )


    sequence_accuracy = accuracy_score(

        y_test,

        sequence_predictions

    )


    sequence_precision = precision_score(

        y_test,

        sequence_predictions,

        zero_division=0

    )


    sequence_recall = recall_score(

        y_test,

        sequence_predictions,

        zero_division=0

    )


    sequence_f1 = f1_score(

        y_test,

        sequence_predictions,

        zero_division=0

    )


    sequence_confusion = confusion_matrix(

        y_test,

        sequence_predictions

    )


    sequence_auc = roc_auc_score(

        y_test,

        probabilities

    )


    # ========================================================
    # PRINT SEQUENCE RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL SEQUENCE-LEVEL TEST RESULTS"
    )

    print(
        "=" * 60
    )


    print(
        f"\nThreshold: {sequence_threshold:.2f}"
    )


    print(
        f"\nAccuracy  : {sequence_accuracy * 100:.2f}%"
    )


    print(
        f"Precision : {sequence_precision * 100:.2f}%"
    )


    print(
        f"Recall    : {sequence_recall * 100:.2f}%"
    )


    print(
        f"F1 Score  : {sequence_f1 * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        sequence_confusion
    )


    print(
        "\nClassification Report:"
    )


    print(

        classification_report(

            y_test,

            sequence_predictions,

            target_names=[

                "Real",

                "Fake"

            ],

            zero_division=0

        )

    )


    print(
        f"ROC-AUC   : {sequence_auc:.4f}"
    )


    # ========================================================
    # VIDEO-LEVEL PREDICTIONS
    # ========================================================

    (

        unique_video_ids,

        video_probabilities,

        video_labels

    ) = calculate_video_predictions(

        probabilities,

        y_test,

        video_ids

    )


    video_predictions = (

        video_probabilities >= video_threshold

    ).astype(
        int
    )


    video_accuracy = accuracy_score(

        video_labels,

        video_predictions

    )


    video_precision = precision_score(

        video_labels,

        video_predictions,

        zero_division=0

    )


    video_recall = recall_score(

        video_labels,

        video_predictions,

        zero_division=0

    )


    video_f1 = f1_score(

        video_labels,

        video_predictions,

        zero_division=0

    )


    video_confusion = confusion_matrix(

        video_labels,

        video_predictions

    )


    video_auc = roc_auc_score(

        video_labels,

        video_probabilities

    )


    # ========================================================
    # PRINT VIDEO RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL VIDEO-LEVEL TEST RESULTS"
    )

    print(
        "=" * 60
    )


    print(
        f"\nThreshold: {video_threshold:.2f}"
    )


    print(
        f"\nAccuracy  : {video_accuracy * 100:.2f}%"
    )


    print(
        f"Precision : {video_precision * 100:.2f}%"
    )


    print(
        f"Recall    : {video_recall * 100:.2f}%"
    )


    print(
        f"F1 Score  : {video_f1 * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        video_confusion
    )


    print(
        "\nClassification Report:"
    )


    print(

        classification_report(

            video_labels,

            video_predictions,

            target_names=[

                "Real",

                "Fake"

            ],

            zero_division=0

        )

    )


    print(
        f"ROC-AUC   : {video_auc:.4f}"
    )


    # ========================================================
    # VIDEO-WISE RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL VIDEO-WISE TEST PREDICTIONS"
    )

    print(
        "=" * 60
    )


    for (

        video_id,

        probability,

        actual,

        predicted

    ) in zip(

        unique_video_ids,

        video_probabilities,

        video_labels,

        video_predictions

    ):


        actual_label = (

            "Fake"

            if actual == 1

            else "Real"

        )


        predicted_label = (

            "Fake"

            if predicted == 1

            else "Real"

        )


        result = (

            "CORRECT"

            if actual == predicted

            else "WRONG"

        )


        print(
            f"\nVideo: {video_id}"
        )


        print(
            f"Actual: {actual_label}"
        )


        print(
            f"Probability: {probability:.4f}"
        )


        print(
            f"Predicted: {predicted_label}"
        )


        print(
            f"Result: {result}"
        )


    # ========================================================
    # COMPLETED
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "EXP13 FINAL TEST EVALUATION COMPLETED"
    )

    print(
        "=" * 60
    )


    print(
        "\nIMPORTANT:"
    )


    print(
        "Thresholds were selected using the validation dataset."
    )


    print(
        "The test dataset was used only for final evaluation."
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()