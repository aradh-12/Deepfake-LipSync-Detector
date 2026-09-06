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

MODEL_PATH = Path(
    "saved_models/lstm_exp10.keras"
)

NORMALIZATION_PATH = Path(
    "saved_models/lstm_exp10_normalization.npz"
)


# ============================================================
# THRESHOLD
#
# Initially use 0.50.
#
# IMPORTANT:
# Later we should select the final threshold using
# VALIDATION data only.
# ============================================================

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
# LOAD NORMALIZATION
# ============================================================

def load_normalization():

    print(
        "\nLoading normalization statistics..."
    )

    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(
            f"Normalization file not found: "
            f"{NORMALIZATION_PATH}"
        )

    data = np.load(
        NORMALIZATION_PATH
    )

    mean = data["mean"]

    std = data["std"]

    print(
        "Mean shape:",
        mean.shape
    )

    print(
        "Std shape:",
        std.shape
    )

    return (
        mean,
        std
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(
    X,
    mean,
    std
):

    X = (
        X - mean
    ) / std

    return X.astype(
        np.float32
    )


# ============================================================
# VIDEO LEVEL PREDICTIONS
# ============================================================

def get_video_predictions(
    probabilities,
    y,
    video_ids
):

    video_results = {}

    for probability, label, video_id in zip(

        probabilities,

        y,

        video_ids

    ):

        video_id = str(
            video_id
        )

        if video_id not in video_results:

            video_results[
                video_id
            ] = {

                "probabilities": [],

                "label": int(
                    label
                )

            }

        video_results[
            video_id
        ][
            "probabilities"
        ].append(

            float(
                probability
            )

        )


    video_probabilities = []

    video_labels = []

    video_names = []


    for video_id, data in (
        video_results.items()
    ):

        average_probability = np.mean(

            data[
                "probabilities"
            ]

        )


        video_probabilities.append(

            average_probability

        )


        video_labels.append(

            data[
                "label"
            ]

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

        video_names

    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
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


    matrix = confusion_matrix(

        y_true,

        predictions

    )


    print(
        "\n" + "=" * 60
    )

    print(
        f"{level_name} RESULTS"
    )

    print(
        "=" * 60
    )


    print(

        f"\nThreshold: "
        f"{threshold:.2f}"

    )


    print(

        f"\nAccuracy  : "
        f"{accuracy * 100:.2f}%"

    )


    print(

        f"Precision : "
        f"{precision * 100:.2f}%"

    )


    print(

        f"Recall    : "
        f"{recall * 100:.2f}%"

    )


    print(

        f"F1 Score  : "
        f"{f1 * 100:.2f}%"

    )


    print(

        "\nConfusion Matrix:"

    )


    print(
        matrix
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


    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    if len(
        np.unique(
            y_true
        )
    ) == 2:

        auc = roc_auc_score(

            y_true,

            probabilities

        )


        print(

            f"ROC-AUC   : "
            f"{auc:.4f}"

        )


    return (

        predictions,

        accuracy,

        precision,

        recall,

        f1

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "EXP10 MODEL EVALUATION"
    )

    print(
        "=" * 60
    )


    print(

        f"\nThreshold: "
        f"{THRESHOLD}"

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

    ) = load_dataset(

        "test"

    )


    print(

        "Test sequences:",
        len(y_test)

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
    # LOAD NORMALIZATION
    # ========================================================

    (

        mean,

        std

    ) = load_normalization()


    # ========================================================
    # NORMALIZE
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
        "\nLoading Exp10 model..."
    )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model not found: "
            f"{MODEL_PATH}"

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

    sequence_predictions = (

        probabilities >= THRESHOLD

    ).astype(
        np.int32
    )


    print_results(

        y_test,

        probabilities,

        THRESHOLD,

        "SEQUENCE-LEVEL TEST"

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


    video_predictions = (

        video_probabilities >= THRESHOLD

    ).astype(
        np.int32
    )


    print_results(

        video_labels,

        video_probabilities,

        THRESHOLD,

        "VIDEO-LEVEL TEST"

    )


    # ========================================================
    # VIDEO-WISE PREDICTIONS
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

        video_name,

        actual,

        probability,

        prediction

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

            if prediction == 1

            else "Real"

        )


        result = (

            "CORRECT"

            if actual == prediction

            else "WRONG"

        )


        print(

            f"\nVideo: "
            f"{video_name}"

        )


        print(

            f"Actual: "
            f"{actual_name}"

        )


        print(

            f"Probability: "
            f"{probability:.4f}"

        )


        print(

            f"Predicted: "
            f"{predicted_name}"

        )


        print(

            f"Result: "
            f"{result}"

        )


    # ========================================================
    # PROBABILITY STATISTICS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PROBABILITY STATISTICS"
    )

    print(
        "=" * 60
    )


    real_probabilities = probabilities[
        y_test == 0
    ]


    fake_probabilities = probabilities[
        y_test == 1
    ]


    print(
        "\nREAL"
    )


    print(

        f"Mean: "
        f"{np.mean(real_probabilities):.4f}"

    )


    print(

        f"Median: "
        f"{np.median(real_probabilities):.4f}"

    )


    print(

        f"Minimum: "
        f"{np.min(real_probabilities):.4f}"

    )


    print(

        f"Maximum: "
        f"{np.max(real_probabilities):.4f}"

    )


    print(
        "\nFAKE"
    )


    print(

        f"Mean: "
        f"{np.mean(fake_probabilities):.4f}"

    )


    print(

        f"Median: "
        f"{np.median(fake_probabilities):.4f}"

    )


    print(

        f"Minimum: "
        f"{np.min(fake_probabilities):.4f}"

    )


    print(

        f"Maximum: "
        f"{np.max(fake_probabilities):.4f}"

    )


    # ========================================================
    # COMPLETED
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "EXP10 TEST EVALUATION COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()