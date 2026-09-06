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
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    file_path = (
        DATASET_DIR /
        "validation.npz"
    )


    if not file_path.exists():

        raise FileNotFoundError(

            f"Validation dataset not found: "
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


        # Average sequence probabilities

        video_probability = np.mean(
            video_probs
        )


        # Every sequence from a video
        # should have the same label

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
# FIND BEST THRESHOLD
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

    best_accuracy = -1


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


        f1 = f1_score(

            y_true,

            predictions,

            zero_division=0

        )


        # Primary criterion:
        # highest F1 score
        #
        # Tie breaker:
        # highest accuracy

        if (

            f1 > best_f1

            or (

                f1 == best_f1

                and accuracy > best_accuracy

            )

        ):


            best_f1 = f1

            best_accuracy = accuracy

            best_threshold = threshold


    predictions = (

        probabilities >= best_threshold

    ).astype(
        np.int32
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


    confusion = confusion_matrix(

        y_true,

        predictions

    )


    return {

        "threshold": best_threshold,

        "accuracy": best_accuracy,

        "precision": precision,

        "recall": recall,

        "f1": best_f1,

        "confusion": confusion

    }


# ============================================================
# MAIN
# ============================================================

def main():


    print(
        "\n" + "=" * 60
    )


    print(
        "EXP12 VALIDATION THRESHOLD ANALYSIS"
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

    ) = load_validation_data()


    print(
        "Validation sequences:",
        len(X_validation)
    )


    print(
        "Validation videos:",
        len(
            np.unique(
                video_ids
            )
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


    (

        mean,

        std

    ) = load_normalization()


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
        "\nLoading Exp12 model..."
    )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model not found: "
            f"{MODEL_PATH}"

        )


    model = tf.keras.models.load_model(

        MODEL_PATH,

        custom_objects={

            "TemporalAttention":
            TemporalAttention

        }

    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(
        "\nRunning validation predictions..."
    )


    sequence_probabilities = (

        model.predict(

            X_validation,

            verbose=0

        )

        .flatten()

    )


    # ========================================================
    # SEQUENCE THRESHOLD
    # ========================================================

    print(
        "\n" + "=" * 60
    )


    print(
        "SEQUENCE-LEVEL VALIDATION THRESHOLD SEARCH"
    )


    print(
        "=" * 60
    )


    sequence_results = (

        find_best_threshold(

            y_validation,

            sequence_probabilities

        )

    )


    print(
        f"\nBest threshold: "
        f"{sequence_results['threshold']:.2f}"
    )


    print(
        f"Accuracy       : "
        f"{sequence_results['accuracy'] * 100:.2f}%"
    )


    print(
        f"Precision      : "
        f"{sequence_results['precision'] * 100:.2f}%"
    )


    print(
        f"Recall         : "
        f"{sequence_results['recall'] * 100:.2f}%"
    )


    print(
        f"F1 Score       : "
        f"{sequence_results['f1'] * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        sequence_results[
            "confusion"
        ]
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

        y_validation,

        video_ids

    )


    # ========================================================
    # VIDEO THRESHOLD
    # ========================================================

    print(
        "\n" + "=" * 60
    )


    print(
        "VIDEO-LEVEL VALIDATION THRESHOLD SEARCH"
    )


    print(
        "=" * 60
    )


    video_results = (

        find_best_threshold(

            video_labels,

            video_probabilities

        )

    )


    print(
        f"\nBest threshold: "
        f"{video_results['threshold']:.2f}"
    )


    print(
        f"Accuracy       : "
        f"{video_results['accuracy'] * 100:.2f}%"
    )


    print(
        f"Precision      : "
        f"{video_results['precision'] * 100:.2f}%"
    )


    print(
        f"Recall         : "
        f"{video_results['recall'] * 100:.2f}%"
    )


    print(
        f"F1 Score       : "
        f"{video_results['f1'] * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        video_results[
            "confusion"
        ]
    )


    # ========================================================
    # SAVE THRESHOLDS
    # ========================================================

    np.savez(

        THRESHOLD_PATH,

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
        THRESHOLD_PATH
    )


    print(
        "\n" + "=" * 60
    )


    print(
        "EXP12 VALIDATION THRESHOLD ANALYSIS COMPLETED"
    )


    print(
        "=" * 60
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()