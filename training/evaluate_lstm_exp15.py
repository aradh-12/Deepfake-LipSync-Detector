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

MODEL_DIR = Path(
    "saved_models"
)


MODEL_PATH = (
    MODEL_DIR /
    "lstm_exp15.keras"
)


NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp15_normalization.npz"
)


VALIDATION_PREDICTIONS_PATH = (
    MODEL_DIR /
    "lstm_exp15_validation_predictions.npz"
)


TEST_PREDICTIONS_PATH = (
    MODEL_DIR /
    "lstm_exp15_test_predictions.npz"
)


THRESHOLDS_PATH = (
    MODEL_DIR /
    "lstm_exp15_validation_thresholds.npz"
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


    # ========================================================
    # EXP15 FEATURES
    #
    # Original Features:
    #       0   : 199
    #
    # Velocity Features:
    #       199 : 398
    #
    # Acceleration:
    #       398 : 597
    #
    # EXP15 uses:
    #
    # Original + Velocity
    #
    # Total = 398 features
    # ========================================================

    X = X[:, :, :398]


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
# GET VIDEO-LEVEL PREDICTIONS
#
# Multiple sequences belong to one video.
#
# We average sequence probabilities.
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


        indices = np.where(

            video_ids == video_id

        )[0]


        video_probs = probabilities[
            indices
        ]


        video_label_values = labels[
            indices
        ]


        unique_labels = np.unique(

            video_label_values

        )


        if len(unique_labels) != 1:

            raise RuntimeError(

                f"Video has multiple labels: "
                f"{video_id}"

            )


        # ----------------------------------------------------
        # Average probabilities of all sequences
        # ----------------------------------------------------

        mean_probability = np.mean(

            video_probs

        )


        video_probabilities.append(

            mean_probability

        )


        video_labels.append(

            int(unique_labels[0])

        )


        video_names.append(

            video_id

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
# IMPORTANT:
#
# Threshold is selected ONLY using validation data.
#
# We maximize F1 score.
# ============================================================

def find_best_threshold(

    probabilities,

    labels

):


    best_threshold = None

    best_f1 = -1


    thresholds = np.arange(

        0.01,

        1.00,

        0.01

    )


    for threshold in thresholds:


        predictions = (

            probabilities >= threshold

        ).astype(

            np.int32

        )


        score = f1_score(

            labels,

            predictions,

            zero_division=0

        )


        if score > best_f1:


            best_f1 = score

            best_threshold = threshold


    return (

        float(best_threshold),

        float(best_f1)

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

        predictions,

        labels=[0, 1]

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

        "\nThreshold:",

        round(

            threshold,

            4

        )

    )


    print(

        "Accuracy :",

        round(

            accuracy * 100,

            2

        ),

        "%"

    )


    print(

        "Precision:",

        round(

            precision * 100,

            2

        ),

        "%"

    )


    print(

        "Recall   :",

        round(

            recall * 100,

            2

        ),

        "%"

    )


    print(

        "F1 Score :",

        round(

            f1 * 100,

            2

        ),

        "%"

    )


    print(

        "\nConfusion Matrix:"

    )


    print(

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

        "EXP15 EVALUATION"

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
    # CHECK NORMALIZATION
    # ========================================================

    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(

            f"Normalization file not found: "
            f"{NORMALIZATION_PATH}"

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
    # DATA SHAPE CHECK
    # ========================================================

    print(

        "\nValidation X shape:",

        X_validation.shape

    )


    print(

        "Test X shape:",

        X_test.shape

    )


    # ========================================================
    # MODEL INPUT CHECK
    # ========================================================

    expected_feature_size = (

        model.input_shape[-1]

    )


    validation_feature_size = (

        X_validation.shape[-1]

    )


    test_feature_size = (

        X_test.shape[-1]

    )


    if (

        validation_feature_size
        !=
        expected_feature_size

    ):

        raise ValueError(

            f"Validation feature size mismatch. "
            f"Model expects "
            f"{expected_feature_size}, "
            f"but validation data has "
            f"{validation_feature_size}"

        )


    if (

        test_feature_size
        !=
        expected_feature_size

    ):

        raise ValueError(

            f"Test feature size mismatch. "
            f"Model expects "
            f"{expected_feature_size}, "
            f"but test data has "
            f"{test_feature_size}"

        )


    # ========================================================
    # NORMALIZE DATA
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
    # GENERATE VALIDATION PREDICTIONS
    # ========================================================

    print(

        "\nGenerating validation predictions..."

    )


    validation_probabilities = model.predict(

        X_validation,

        verbose=0

    ).flatten()


    # ========================================================
    # GENERATE TEST PREDICTIONS
    # ========================================================

    print(

        "Generating test predictions..."

    )


    test_probabilities = model.predict(

        X_test,

        verbose=0

    ).flatten()


    # ========================================================
    # VIDEO-LEVEL VALIDATION PREDICTIONS
    # ========================================================

    (

        validation_video_probabilities,

        validation_video_labels,

        validation_video_names

    ) = get_video_predictions(

        validation_probabilities,

        y_validation,

        video_ids_validation

    )


    # ========================================================
    # VIDEO-LEVEL TEST PREDICTIONS
    # ========================================================

    (

        test_video_probabilities,

        test_video_labels,

        test_video_names

    ) = get_video_predictions(

        test_probabilities,

        y_test,

        video_ids_test

    )


    # ========================================================
    # FIND SEQUENCE THRESHOLD
    #
    # ONLY VALIDATION DATA
    # ========================================================

    sequence_threshold, _ = (

        find_best_threshold(

            validation_probabilities,

            y_validation

        )

    )


    # ========================================================
    # FIND VIDEO THRESHOLD
    #
    # ONLY VALIDATION DATA
    # ========================================================

    video_threshold, _ = (

        find_best_threshold(

            validation_video_probabilities,

            validation_video_labels

        )

    )


    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    np.savez_compressed(

        VALIDATION_PREDICTIONS_PATH,


        sequence_probabilities=(

            validation_probabilities

        ),


        sequence_labels=(

            y_validation

        ),


        sequence_video_ids=(

            video_ids_validation

        ),


        video_probabilities=(

            validation_video_probabilities

        ),


        video_labels=(

            validation_video_labels

        ),


        video_names=(

            validation_video_names

        )

    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    np.savez_compressed(

        TEST_PREDICTIONS_PATH,


        sequence_probabilities=(

            test_probabilities

        ),


        sequence_labels=(

            y_test

        ),


        sequence_video_ids=(

            video_ids_test

        ),


        video_probabilities=(

            test_video_probabilities

        ),


        video_labels=(

            test_video_labels

        ),


        video_names=(

            test_video_names

        )

    )


    # ========================================================
    # SAVE THRESHOLDS
    # ========================================================

    np.savez(

        THRESHOLDS_PATH,


        sequence_threshold=(

            sequence_threshold

        ),


        video_threshold=(

            video_threshold

        )

    )


    print(

        "\nPredictions saved."

    )


    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    print_results(

        "EXP15 VALIDATION SEQUENCE-LEVEL RESULTS",


        y_validation,


        validation_probabilities,


        sequence_threshold

    )


    print_results(

        "EXP15 VALIDATION VIDEO-LEVEL RESULTS",


        validation_video_labels,


        validation_video_probabilities,


        video_threshold

    )


    # ========================================================
    # TEST RESULTS
    #
    # Uses thresholds selected from validation data.
    # ========================================================

    print_results(

        "EXP15 TEST SEQUENCE-LEVEL RESULTS",


        y_test,


        test_probabilities,


        sequence_threshold

    )


    print_results(

        "EXP15 TEST VIDEO-LEVEL RESULTS",


        test_video_labels,


        test_video_probabilities,


        video_threshold

    )


    # ========================================================
    # COMPLETED
    # ========================================================

    print(

        "\n" + "=" * 60

    )


    print(

        "EXP15 EVALUATION COMPLETED"

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

        THRESHOLDS_PATH

    )


    print(

        "\n" + "=" * 60

    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()