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

            probabilities[
                indices
            ]

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
# PRINT METRICS
# ============================================================

def print_metrics(

    y_true,

    probabilities,

    threshold,

    level_name

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


    print(

        "\n" + "=" * 60

    )


    print(

        f"{level_name} TEST RESULTS"

    )


    print(

        "=" * 60

    )


    print(

        f"\nThreshold: {threshold:.2f}"

    )


    print(

        f"\nAccuracy  : {accuracy * 100:.2f}%"

    )


    print(

        f"Precision : {precision * 100:.2f}%"

    )


    print(

        f"Recall    : {recall * 100:.2f}%"

    )


    print(

        f"F1 Score  : {f1 * 100:.2f}%"

    )


    print(

        "\nConfusion Matrix:"

    )


    print(

        confusion_matrix(

            y_true,

            predictions

        )

    )


    print(

        "\nClassification Report:"

    )


    print(

        classification_report(

            y_true,

            predictions,

            target_names=[

                "Real",

                "Fake"

            ],

            zero_division=0

        )

    )


    print(

        f"ROC-AUC   : {auc:.4f}"

    )


# ============================================================
# MAIN
# ============================================================

def main():


    print(

        "\n" + "=" * 60

    )


    print(

        "EXP13 MODEL EVALUATION"

    )


    print(

        "=" * 60

    )


    print(

        f"\nThreshold: {THRESHOLD}"

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

        len(
            X_test
        )

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


    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(

            f"Normalization file not found: "

            f"{NORMALIZATION_PATH}"

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

        "\nRunning test predictions..."

    )


    probabilities = model.predict(

        X_test,

        verbose=0

    ).flatten()


    # ========================================================
    # SEQUENCE LEVEL RESULTS
    # ========================================================

    print_metrics(

        y_test,

        probabilities,

        THRESHOLD,

        "SEQUENCE-LEVEL"

    )


    # ========================================================
    # VIDEO LEVEL RESULTS
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


    print_metrics(

        video_labels,

        video_probabilities,

        THRESHOLD,

        "VIDEO-LEVEL"

    )


    # ========================================================
    # VIDEO-WISE PREDICTIONS
    # ========================================================

    video_predictions = (

        video_probabilities >= THRESHOLD

    ).astype(

        np.int32

    )


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

        video_name,

        actual,

        probability,

        predicted

    ) in zip(

        video_names,

        video_labels,

        video_probabilities,

        video_predictions

    ):


        actual_name = (

            "Fake"

            if actual == 1

            else "Real"

        )


        predicted_name = (

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

            f"\nVideo: {video_name}"

        )


        print(

            f"Actual: {actual_name}"

        )


        print(

            f"Probability: {probability:.4f}"

        )


        print(

            f"Predicted: {predicted_name}"

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

        "EXP13 TEST EVALUATION COMPLETED"

    )


    print(

        "=" * 60

    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()