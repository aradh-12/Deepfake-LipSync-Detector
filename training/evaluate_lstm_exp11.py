from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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
    "lstm_exp11.keras"
)


NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp11_normalization.npz"
)


THRESHOLD = 0.5


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

    return (

        X - mean

    ) / (

        std + 1e-8

    )


# ============================================================
# VIDEO LEVEL AGGREGATION
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

        video_id = str(

            video_id

        )


        if video_id not in video_probabilities:

            video_probabilities[

                video_id

            ] = []


        video_probabilities[

            video_id

        ].append(

            float(probability)

        )


        video_labels[

            video_id

        ] = int(label)


    aggregated_probabilities = []

    aggregated_labels = []

    aggregated_video_ids = []


    for video_id in sorted(

        video_probabilities.keys()

    ):

        probabilities_for_video = (

            video_probabilities[

                video_id

            ]

        )


        average_probability = np.mean(

            probabilities_for_video

        )


        aggregated_probabilities.append(

            average_probability

        )


        aggregated_labels.append(

            video_labels[

                video_id

            ]

        )


        aggregated_video_ids.append(

            video_id

        )


    return (

        np.array(

            aggregated_probabilities

        ),

        np.array(

            aggregated_labels

        ),

        aggregated_video_ids

    )


# ============================================================
# CALCULATE METRICS
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


    try:

        auc = roc_auc_score(

            y_true,

            probabilities

        )


    except ValueError:

        auc = float(

            "nan"

        )


    matrix = confusion_matrix(

        y_true,

        predictions

    )


    return (

        predictions,

        accuracy,

        precision,

        recall,

        f1,

        auc,

        matrix

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(

        "\n" + "=" * 60

    )


    print(

        "EXP11 MODEL EVALUATION"

    )


    print(

        "=" * 60

    )


    print(

        "\nThreshold:",

        THRESHOLD

    )


    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print(

        "\nLoading test dataset..."

    )


    (

        X_test,

        y_test,

        video_ids

    ) = load_test_dataset()


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


    # ========================================================
    # LOAD NORMALIZATION
    # ========================================================

    print(

        "\nLoading normalization statistics..."

    )


    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(

            f"Normalization file not found: {NORMALIZATION_PATH}"

        )


    normalization = np.load(

        NORMALIZATION_PATH

    )


    mean = normalization[

        "mean"

    ]


    std = normalization[

        "std"

    ]


    print(

        "Mean shape:",

        mean.shape

    )


    print(

        "Std shape:",

        std.shape

    )


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
    # LOAD MODEL
    # ========================================================

    print(

        "\nLoading Exp11 model..."

    )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model not found: {MODEL_PATH}"

        )


    model = tf.keras.models.load_model(

        MODEL_PATH

    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(

        "\nRunning test predictions..."

    )


    probabilities = model.predict(

        X_test,

        verbose=0

    ).flatten()


    # ========================================================
    # SEQUENCE LEVEL RESULTS
    # ========================================================

    (

        sequence_predictions,

        sequence_accuracy,

        sequence_precision,

        sequence_recall,

        sequence_f1,

        sequence_auc,

        sequence_matrix

    ) = calculate_metrics(

        y_test,

        probabilities,

        THRESHOLD

    )


    print(

        "\n" + "=" * 60

    )


    print(

        "SEQUENCE-LEVEL TEST RESULTS"

    )


    print(

        "=" * 60

    )


    print(

        f"\nThreshold: {THRESHOLD:.2f}"

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

        sequence_matrix

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
    # VIDEO LEVEL AGGREGATION
    # ========================================================

    (

        video_probabilities,

        video_labels,

        unique_video_ids

    ) = aggregate_video_predictions(

        probabilities,

        y_test,

        video_ids

    )


    (

        video_predictions,

        video_accuracy,

        video_precision,

        video_recall,

        video_f1,

        video_auc,

        video_matrix

    ) = calculate_metrics(

        video_labels,

        video_probabilities,

        THRESHOLD

    )


    print(

        "\n" + "=" * 60

    )


    print(

        "VIDEO-LEVEL TEST RESULTS"

    )


    print(

        "=" * 60

    )


    print(

        f"\nThreshold: {THRESHOLD:.2f}"

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

        video_matrix

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

        "VIDEO-WISE TEST PREDICTIONS"

    )


    print(

        "=" * 60

    )


    for (

        video_id,

        actual,

        probability,

        prediction

    ) in zip(

        unique_video_ids,

        video_labels,

        video_probabilities,

        video_predictions

    ):


        actual_label = (

            "Fake"

            if actual == 1

            else "Real"

        )


        predicted_label = (

            "Fake"

            if prediction == 1

            else "Real"

        )


        result = (

            "CORRECT"

            if actual == prediction

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
    # PROBABILITY STATISTICS
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

        "PROBABILITY STATISTICS"

    )


    print(

        "=" * 60

    )


    print(

        "\nREAL"

    )


    print(

        f"Mean: {np.mean(real_probabilities):.4f}"

    )


    print(

        f"Median: {np.median(real_probabilities):.4f}"

    )


    print(

        f"Minimum: {np.min(real_probabilities):.4f}"

    )


    print(

        f"Maximum: {np.max(real_probabilities):.4f}"

    )


    print(

        "\nFAKE"

    )


    print(

        f"Mean: {np.mean(fake_probabilities):.4f}"

    )


    print(

        f"Median: {np.median(fake_probabilities):.4f}"

    )


    print(

        f"Minimum: {np.min(fake_probabilities):.4f}"

    )


    print(

        f"Maximum: {np.max(fake_probabilities):.4f}"

    )


    # ========================================================
    # COMPLETED
    # ========================================================

    print(

        "\n" + "=" * 60

    )


    print(

        "EXP11 TEST EVALUATION COMPLETED"

    )


    print(

        "=" * 60

    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()