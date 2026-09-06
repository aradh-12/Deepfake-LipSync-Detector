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
    "saved_models/lstm_exp12.keras"
)


NORMALIZATION_PATH = Path(
    "saved_models/lstm_exp12_normalization.npz"
)


THRESHOLD_PATH = Path(
    "saved_models/lstm_exp12_validation_thresholds.npz"
)


# ============================================================
# TEMPORAL ATTENTION LAYER
# ============================================================

class TemporalAttention(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


    def build(
        self,
        input_shape
    ):

        self.attention_dense = (
            tf.keras.layers.Dense(
                1
            )
        )


        super().build(
            input_shape
        )


    def call(
        self,
        inputs
    ):

        attention_scores = (
            self.attention_dense(
                inputs
            )
        )


        attention_weights = (
            tf.nn.softmax(
                attention_scores,
                axis=1
            )
        )


        weighted_inputs = (

            inputs *

            attention_weights

        )


        context_vector = tf.reduce_sum(

            weighted_inputs,

            axis=1

        )


        return context_vector


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    file_path = (
        DATASET_DIR /
        "test.npz"
    )


    if not file_path.exists():

        raise FileNotFoundError(

            f"Test dataset not found: "
            f"{file_path}"

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

    X_normalized = (

        X - mean

    ) / std


    return X_normalized.astype(
        np.float32
    )


# ============================================================
# CREATE VIDEO-LEVEL PREDICTIONS
# ============================================================

def create_video_predictions(

    probabilities,

    labels,

    video_ids

):

    unique_video_ids = np.unique(
        video_ids
    )


    video_probabilities = []

    video_labels = []

    video_names = []


    for video_id in unique_video_ids:


        mask = (

            video_ids == video_id

        )


        video_probs = probabilities[
            mask
        ]


        video_label_values = labels[
            mask
        ]


        video_probability = np.mean(
            video_probs
        )


        video_label = int(
            video_label_values[0]
        )


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

    predictions,

    probabilities,

    threshold

):

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


    confusion = confusion_matrix(

        y_true,

        predictions

    )


    try:

        auc = roc_auc_score(

            y_true,

            probabilities

        )

    except ValueError:

        auc = None


    print(
        f"\nThreshold: {threshold:.2f}"
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
        confusion
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


    if auc is not None:

        print(
            f"ROC-AUC   : "
            f"{auc:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():


    print(
        "\n" + "=" * 60
    )


    print(
        "EXP12 FINAL TEST EVALUATION"
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


    (

        X_test,

        y_test,

        video_ids

    ) = load_test_data()


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
    # LOAD NORMALIZATION
    # ========================================================

    print(
        "\nLoading normalization statistics..."
    )


    (

        mean,

        std

    ) = load_normalization()


    print(
        "\nNormalizing test data..."
    )


    X_test = normalize_data(

        X_test,

        mean,

        std

    )


    # ========================================================
    # LOAD THRESHOLDS
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
        f"Sequence threshold: "
        f"{sequence_threshold:.2f}"
    )


    print(
        f"Video threshold   : "
        f"{video_threshold:.2f}"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading Exp12 model..."
    )


    model = tf.keras.models.load_model(

        MODEL_PATH,

        custom_objects={

            "TemporalAttention":
            TemporalAttention

        }

    )


    # ========================================================
    # RUN PREDICTIONS
    # ========================================================

    print(
        "\nRunning test predictions..."
    )


    sequence_probabilities = (

        model.predict(

            X_test,

            verbose=0

        )

        .flatten()

    )


    # ========================================================
    # SEQUENCE PREDICTIONS
    # ========================================================

    sequence_predictions = (

        sequence_probabilities

        >=

        sequence_threshold

    ).astype(
        np.int32
    )


    print(
        "\n" + "=" * 60
    )


    print(
        "FINAL SEQUENCE-LEVEL TEST RESULTS"
    )


    print(
        "=" * 60
    )


    print_metrics(

        y_test,

        sequence_predictions,

        sequence_probabilities,

        sequence_threshold

    )


    # ========================================================
    # VIDEO PREDICTIONS
    # ========================================================

    (

        video_probabilities,

        video_labels,

        video_names

    ) = create_video_predictions(

        sequence_probabilities,

        y_test,

        video_ids

    )


    video_predictions = (

        video_probabilities

        >=

        video_threshold

    ).astype(
        np.int32
    )


    print(
        "\n" + "=" * 60
    )


    print(
        "FINAL VIDEO-LEVEL TEST RESULTS"
    )


    print(
        "=" * 60
    )


    print_metrics(

        video_labels,

        video_predictions,

        video_probabilities,

        video_threshold

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
            f"\nVideo: {video_name}"
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
    # COMPLETED
    # ========================================================

    print(
        "\n" + "=" * 60
    )


    print(
        "EXP12 FINAL TEST EVALUATION COMPLETED"
    )


    print(
        "=" * 60
    )


    print(
        "\nIMPORTANT:"
    )


    print(
        "Thresholds were selected using "
        "the validation dataset."
    )


    print(
        "The test dataset was used only "
        "for final evaluation."
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()