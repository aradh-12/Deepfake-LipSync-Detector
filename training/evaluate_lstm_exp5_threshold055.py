from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Configuration
# ============================================================

DATASET_FOLDER = Path(
    "outputs/dataset_exp5_full"
)

TEST_FILE = (
    DATASET_FOLDER /
    "test.npz"
)


MODEL_FILE = Path(
    "models/lstm_exp5_best.keras"
)


NORMALIZATION_FILE = Path(
    "models/lstm_exp5_normalization.npz"
)


# ============================================================
# Locked Threshold
#
# IMPORTANT:
#
# This threshold was selected ONLY
# using the validation dataset.
#
# It must not be changed after
# looking at test results.
# ============================================================

THRESHOLD = 0.55


# ============================================================
# Load Dataset
# ============================================================

def load_dataset():

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found: "
            f"{TEST_FILE}"
        )


    data = np.load(

        TEST_FILE,

        allow_pickle=True

    )


    X = data["X"].astype(
        np.float32
    )


    y = data["y"].astype(
        np.int32
    )


    video_ids = data["video_ids"]


    return X, y, video_ids


# ============================================================
# Normalize Features
# ============================================================

def normalize_features(
    X
):

    if not NORMALIZATION_FILE.exists():

        raise FileNotFoundError(

            f"Normalization file not found: "
            f"{NORMALIZATION_FILE}"

        )


    normalization_data = np.load(

        NORMALIZATION_FILE

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
# Calculate Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(

        y_true,

        y_pred

    )


    precision = precision_score(

        y_true,

        y_pred,

        zero_division=0

    )


    recall = recall_score(

        y_true,

        y_pred,

        zero_division=0

    )


    f1 = f1_score(

        y_true,

        y_pred,

        zero_division=0

    )


    return (

        accuracy,

        precision,

        recall,

        f1

    )


# ============================================================
# Video-Level Aggregation
#
# Average all sequence probabilities
# belonging to the same video.
# ============================================================

def aggregate_video_predictions(

    probabilities,

    labels,

    video_ids

):

    video_probabilities = {}

    video_labels = {}


    for probability, label, video_id in zip(

        probabilities,

        labels,

        video_ids

    ):


        if video_id not in video_probabilities:

            video_probabilities[
                video_id
            ] = []


            video_labels[
                video_id
            ] = label


        video_probabilities[
            video_id
        ].append(

            float(probability)

        )


    final_video_ids = []

    final_probabilities = []

    final_labels = []


    for video_id in sorted(

        video_probabilities.keys()

    ):


        average_probability = np.mean(

            video_probabilities[
                video_id
            ]

        )


        final_video_ids.append(

            video_id

        )


        final_probabilities.append(

            average_probability

        )


        final_labels.append(

            video_labels[
                video_id
            ]

        )


    return (

        np.array(
            final_video_ids
        ),

        np.array(
            final_probabilities
        ),

        np.array(
            final_labels
        )

    )


# ============================================================
# Print Metrics
# ============================================================

def print_metrics(

    title,

    y_true,

    y_pred

):

    (

        accuracy,

        precision,

        recall,

        f1

    ) = calculate_metrics(

        y_true,

        y_pred

    )


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

        "Accuracy  :",
        f"{accuracy * 100:.2f}%"

    )


    print(

        "Precision :",
        f"{precision * 100:.2f}%"

    )


    print(

        "Recall    :",
        f"{recall * 100:.2f}%"

    )


    print(

        "F1 Score  :",
        f"{f1 * 100:.2f}%"

    )


    print(

        "\nConfusion Matrix:"

    )


    print(

        confusion_matrix(

            y_true,

            y_pred

        )

    )


    print(

        "\nClassification Report:"

    )


    print(

        classification_report(

            y_true,

            y_pred,

            target_names=[

                "Real",

                "Fake"

            ],

            zero_division=0

        )

    )


# ============================================================
# Main
# ============================================================

def main():

    print(

        "\nLoading Exp5 test dataset..."

    )


    (

        X_test,

        y_test,

        test_video_ids

    ) = load_dataset()


    print(

        "Test sequences:",

        len(X_test)

    )


    print(

        "Test videos:",

        len(

            np.unique(
                test_video_ids
            )

        )

    )


    print(

        "Real sequences:",

        int(

            np.sum(
                y_test == 0
            )

        )

    )


    print(

        "Fake sequences:",

        int(

            np.sum(
                y_test == 1
            )

        )

    )


    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    print(

        "\nLoading Exp5 normalization..."

    )


    print(

        "Normalizing test data..."

    )


    X_test = normalize_features(

        X_test

    )


    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    print(

        "\nLoading Exp5 model..."

    )


    if not MODEL_FILE.exists():

        raise FileNotFoundError(

            f"Model file not found: "
            f"{MODEL_FILE}"

        )


    model = tf.keras.models.load_model(

        MODEL_FILE

    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(

        "\nRunning predictions..."

    )


    sequence_probabilities = model.predict(

        X_test,

        verbose=0

    ).flatten()


    sequence_predictions = (

        sequence_probabilities >= THRESHOLD

    ).astype(
        np.int32
    )


    # --------------------------------------------------------
    # Sequence-Level Results
    # --------------------------------------------------------

    print(

        "\n" + "=" * 60

    )


    print(

        f"LOCKED THRESHOLD: {THRESHOLD}"

    )


    print(

        "Selected using VALIDATION DATA ONLY"

    )


    print(

        "=" * 60

    )


    print_metrics(

        "SEQUENCE-LEVEL TEST RESULTS",

        y_test,

        sequence_predictions

    )


    # --------------------------------------------------------
    # Video-Level Aggregation
    # --------------------------------------------------------

    (

        video_ids,

        video_probabilities,

        video_labels

    ) = aggregate_video_predictions(

        sequence_probabilities,

        y_test,

        test_video_ids

    )


    video_predictions = (

        video_probabilities >= THRESHOLD

    ).astype(
        np.int32
    )


    # --------------------------------------------------------
    # Video-Level Predictions
    # --------------------------------------------------------

    print(

        "\n" + "=" * 60

    )


    print(

        "VIDEO-LEVEL PREDICTIONS"

    )


    print(

        "=" * 60

    )


    for (

        video_id,

        probability,

        true_label,

        predicted_label

    ) in zip(

        video_ids,

        video_probabilities,

        video_labels,

        video_predictions

    ):


        true_name = (

            "Fake"

            if true_label == 1

            else "Real"

        )


        predicted_name = (

            "Fake"

            if predicted_label == 1

            else "Real"

        )


        print()

        print(
            video_id
        )


        print(

            "True       :",

            true_name

        )


        print(

            "Probability:",

            f"{probability:.4f}"

        )


        print(

            "Predicted  :",

            predicted_name

        )


        print(
            "-" * 50
        )


    # --------------------------------------------------------
    # Video-Level Results
    # --------------------------------------------------------

    print_metrics(

        "VIDEO-LEVEL FINAL TEST RESULTS",

        video_labels,

        video_predictions

    )


    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    (

        sequence_accuracy,

        sequence_precision,

        sequence_recall,

        sequence_f1

    ) = calculate_metrics(

        y_test,

        sequence_predictions

    )


    (

        video_accuracy,

        video_precision,

        video_recall,

        video_f1

    ) = calculate_metrics(

        video_labels,

        video_predictions

    )


    print(

        "\n" + "=" * 60

    )


    print(

        "FINAL EXP5 EVALUATION SUMMARY"

    )


    print(

        "=" * 60

    )


    print(

        "Locked Threshold:",

        THRESHOLD

    )


    print()


    print(

        "Sequence-Level"

    )


    print(

        f"Accuracy  : {sequence_accuracy * 100:.2f}%"

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


    print()


    print(

        "Video-Level"

    )


    print(

        f"Accuracy  : {video_accuracy * 100:.2f}%"

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

        "\n" + "=" * 60

    )


    print(

        "FINAL EXP5 TEST EVALUATION COMPLETED"

    )


    print(

        "=" * 60

    )


if __name__ == "__main__":

    main()
