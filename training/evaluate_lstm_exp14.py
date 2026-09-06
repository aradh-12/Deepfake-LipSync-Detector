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
    "saved_models/lstm_exp14.keras"
)


NORMALIZATION_PATH = Path(
    "saved_models/lstm_exp14_normalization.npz"
)


VALIDATION_PREDICTIONS_PATH = Path(
    "saved_models/lstm_exp14_validation_predictions.npz"
)


TEST_PREDICTIONS_PATH = Path(
    "saved_models/lstm_exp14_test_predictions.npz"
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
            f"Dataset not found: "
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
# VIDEO-LEVEL AGGREGATION
# ============================================================

def aggregate_video_predictions(
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


    for video_name in unique_videos:


        indices = np.where(

            video_ids == video_name

        )[0]


        video_probs = (

            probabilities[indices]

        )


        video_probability = np.mean(

            video_probs

        )


        video_label_values = (

            labels[indices]

        )


        unique_labels = np.unique(

            video_label_values

        )


        if len(unique_labels) != 1:

            raise ValueError(

                f"Video has inconsistent labels: "
                f"{video_name}"

            )


        video_label = (

            unique_labels[0]

        )


        video_probabilities.append(

            video_probability

        )


        video_labels.append(

            video_label

        )


        video_names.append(

            video_name

        )


    return (

        np.asarray(
            video_probabilities,
            dtype=np.float32
        ),

        np.asarray(
            video_labels,
            dtype=np.int32
        ),

        np.asarray(
            video_names
        )

    )


# ============================================================
# FIND BEST THRESHOLD
#
# Threshold is selected ONLY using validation data.
# ============================================================

def find_best_threshold(
    probabilities,
    labels
):

    thresholds = np.arange(

        0.01,

        1.00,

        0.01

    )


    best_threshold = None

    best_f1 = -1.0

    best_accuracy = -1.0


    for threshold in thresholds:


        predictions = (

            probabilities >= threshold

        ).astype(

            np.int32

        )


        f1 = f1_score(

            labels,

            predictions,

            zero_division=0

        )


        accuracy = accuracy_score(

            labels,

            predictions

        )


        if (

            f1 > best_f1

            or

            (
                f1 == best_f1

                and

                accuracy > best_accuracy
            )

        ):


            best_f1 = f1

            best_accuracy = accuracy

            best_threshold = threshold


    return (

        float(best_threshold),

        float(best_f1),

        float(best_accuracy)

    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    title,
    labels,
    probabilities,
    threshold
):

    predictions = (

        probabilities >= threshold

    ).astype(

        np.int32

    )


    accuracy = accuracy_score(

        labels,

        predictions

    )


    precision = precision_score(

        labels,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        labels,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        labels,

        predictions,

        zero_division=0

    )


    matrix = confusion_matrix(

        labels,

        predictions

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

        f"\nThreshold: "
        f"{threshold:.2f}"

    )


    print(

        f"Accuracy : "
        f"{accuracy * 100:.2f} %"

    )


    print(

        f"Precision: "
        f"{precision * 100:.2f} %"

    )


    print(

        f"Recall   : "
        f"{recall * 100:.2f} %"

    )


    print(

        f"F1 Score : "
        f"{f1 * 100:.2f} %"

    )


    print(

        "\nConfusion Matrix:"

    )


    print(

        matrix

    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    output_path,
    sequence_probabilities,
    sequence_labels,
    sequence_video_ids,
    video_probabilities,
    video_labels,
    video_names
):

    np.savez_compressed(

        output_path,

        sequence_probabilities=
        sequence_probabilities,

        sequence_labels=
        sequence_labels,

        sequence_video_ids=
        sequence_video_ids,

        video_probabilities=
        video_probabilities,

        video_labels=
        video_labels,

        video_names=
        video_names

    )


# ============================================================
# MAIN
# ============================================================

def main():


    print(

        "\n" + "=" * 60

    )


    print(

        "EXP14 EVALUATION"

    )


    print(

        "=" * 60

    )


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model not found: "
            f"{MODEL_PATH}"

        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(

        "\nLoading model..."

    )


    model = tf.keras.models.load_model(

        MODEL_PATH

    )


    print(

        "Model loaded successfully."

    )


    # ========================================================
    # LOAD NORMALIZATION
    # ========================================================

    print(

        "\nLoading normalization..."

    )


    (

        mean,

        std

    ) = load_normalization()


    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print(

        "\nLoading validation dataset..."

    )


    (

        X_validation,

        y_validation,

        video_ids_validation

    ) = load_dataset(

        "validation"

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

        video_ids_test

    ) = load_dataset(

        "test"

    )


    # ========================================================
    # SELECT ORIGINAL 199 FEATURES
    # ========================================================

    X_validation = (

        X_validation[:, :, :199]

    )


    X_test = (

        X_test[:, :, :199]

    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    print(

        "\nNormalizing datasets..."

    )


    X_validation = normalize_data(

        X_validation,

        mean,

        std

    )


    X_test = normalize_data(

        X_test,

        mean,

        std

    )


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    print(

        "\nGenerating validation predictions..."

    )


    validation_probabilities = (

        model.predict(

            X_validation,

            verbose=0

        )

        .flatten()

    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print(

        "Generating test predictions..."

    )


    test_probabilities = (

        model.predict(

            X_test,

            verbose=0

        )

        .flatten()

    )


    # ========================================================
    # VIDEO AGGREGATION
    # ========================================================

    (

        validation_video_probabilities,

        validation_video_labels,

        validation_video_names

    ) = aggregate_video_predictions(

        validation_probabilities,

        y_validation,

        video_ids_validation

    )


    (

        test_video_probabilities,

        test_video_labels,

        test_video_names

    ) = aggregate_video_predictions(

        test_probabilities,

        y_test,

        video_ids_test

    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    save_predictions(

        VALIDATION_PREDICTIONS_PATH,

        validation_probabilities,

        y_validation,

        video_ids_validation,

        validation_video_probabilities,

        validation_video_labels,

        validation_video_names

    )


    save_predictions(

        TEST_PREDICTIONS_PATH,

        test_probabilities,

        y_test,

        video_ids_test,

        test_video_probabilities,

        test_video_labels,

        test_video_names

    )


    print(

        "\nPredictions saved."

    )


    # ========================================================
    # SELECT SEQUENCE THRESHOLD
    #
    # VALIDATION ONLY
    # ========================================================

    (

        sequence_threshold,

        _,

        _

    ) = find_best_threshold(

        validation_probabilities,

        y_validation

    )


    # ========================================================
    # SELECT VIDEO THRESHOLD
    #
    # VALIDATION ONLY
    # ========================================================

    (

        video_threshold,

        _,

        _

    ) = find_best_threshold(

        validation_video_probabilities,

        validation_video_labels

    )


    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    print_results(

        "EXP14 VALIDATION SEQUENCE-LEVEL RESULTS",

        y_validation,

        validation_probabilities,

        sequence_threshold

    )


    print_results(

        "EXP14 VALIDATION VIDEO-LEVEL RESULTS",

        validation_video_labels,

        validation_video_probabilities,

        video_threshold

    )


    # ========================================================
    # TEST RESULTS
    #
    # IMPORTANT:
    # Use thresholds selected on validation.
    # ========================================================

    print_results(

        "EXP14 TEST SEQUENCE-LEVEL RESULTS",

        y_test,

        test_probabilities,

        sequence_threshold

    )


    print_results(

        "EXP14 TEST VIDEO-LEVEL RESULTS",

        test_video_labels,

        test_video_probabilities,

        video_threshold

    )


    # ========================================================
    # SAVE THRESHOLDS
    # ========================================================

    threshold_path = Path(

        "saved_models/"
        "lstm_exp14_validation_thresholds.npz"

    )


    np.savez(

        threshold_path,

        sequence_threshold=
        sequence_threshold,

        video_threshold=
        video_threshold

    )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print(

        "\n" + "=" * 60

    )


    print(

        "EXP14 EVALUATION COMPLETED"

    )


    print(

        "=" * 60

    )


    print(

        "\nSequence threshold:"

    )


    print(

        sequence_threshold

    )


    print(

        "\nVideo threshold:"

    )


    print(

        video_threshold

    )


    print(

        "\nValidation predictions:"

    )


    print(

        VALIDATION_PREDICTIONS_PATH

    )


    print(

        "\nTest predictions:"

    )


    print(

        TEST_PREDICTIONS_PATH

    )


    print(

        "\nThresholds:"

    )


    print(

        threshold_path

    )


    print(

        "\n" + "=" * 60

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()