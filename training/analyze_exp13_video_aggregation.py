from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
    "lstm_exp13.keras"
)


NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp13_normalization.npz"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split_name):

    file_path = (
        DATASET_DIR /
        f"{split_name}.npz"
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

    return X, y, video_ids


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(X, mean, std):

    X_normalized = (
        X - mean
    ) / std

    return X_normalized.astype(
        np.float32
    )


# ============================================================
# VIDEO AGGREGATION METHODS
# ============================================================

def aggregate_mean(probabilities):

    return np.mean(
        probabilities
    )


def aggregate_median(probabilities):

    return np.median(
        probabilities
    )


def aggregate_max(probabilities):

    return np.max(
        probabilities
    )


def aggregate_top_k_mean(
    probabilities,
    k=3
):

    sorted_probs = np.sort(
        probabilities
    )[::-1]

    k = min(
        k,
        len(sorted_probs)
    )

    return np.mean(
        sorted_probs[:k]
    )


# ============================================================
# CREATE VIDEO-LEVEL DATA
# ============================================================

def create_video_predictions(
    probabilities,
    labels,
    video_ids,
    aggregation_method
):

    unique_video_ids = np.unique(
        video_ids
    )

    video_probabilities = []
    video_labels = []


    for video_id in unique_video_ids:

        indices = (
            video_ids == video_id
        )

        sequence_probs = probabilities[
            indices
        ]

        video_probability = (
            aggregation_method(
                sequence_probs
            )
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


    return (

        unique_video_ids,

        np.array(
            video_probabilities
        ),

        np.array(
            video_labels
        )

    )


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    probabilities,
    labels
):

    best_threshold = 0.5
    best_f1 = -1

    best_metrics = None


    thresholds = np.arange(

        0.05,

        0.96,

        0.01

    )


    for threshold in thresholds:

        predictions = (

            probabilities >= threshold

        ).astype(
            int
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


        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold

            best_metrics = {

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1": f1

            }


    return (

        best_threshold,

        best_metrics

    )


# ============================================================
# EVALUATE AGGREGATION METHOD
# ============================================================

def evaluate_method(

    method_name,

    aggregation_method,

    test_probabilities,

    test_labels,

    test_video_ids,

    validation_probabilities,

    validation_labels,

    validation_video_ids

):


    # ========================================================
    # CREATE VALIDATION VIDEO PROBABILITIES
    # ========================================================

    (

        _,

        validation_video_probs,

        validation_video_labels

    ) = create_video_predictions(

        validation_probabilities,

        validation_labels,

        validation_video_ids,

        aggregation_method

    )


    # ========================================================
    # FIND THRESHOLD USING VALIDATION DATA ONLY
    # ========================================================

    (

        threshold,

        validation_metrics

    ) = find_best_threshold(

        validation_video_probs,

        validation_video_labels

    )


    # ========================================================
    # CREATE TEST VIDEO PROBABILITIES
    # ========================================================

    (

        test_unique_video_ids,

        test_video_probs,

        test_video_labels

    ) = create_video_predictions(

        test_probabilities,

        test_labels,

        test_video_ids,

        aggregation_method

    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    test_predictions = (

        test_video_probs >= threshold

    ).astype(
        int
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(

        test_video_labels,

        test_predictions

    )


    precision = precision_score(

        test_video_labels,

        test_predictions,

        zero_division=0

    )


    recall = recall_score(

        test_video_labels,

        test_predictions,

        zero_division=0

    )


    f1 = f1_score(

        test_video_labels,

        test_predictions,

        zero_division=0

    )


    auc = roc_auc_score(

        test_video_labels,

        test_video_probs

    )


    cm = confusion_matrix(

        test_video_labels,

        test_predictions

    )


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        method_name.upper()
    )

    print(
        "=" * 60
    )


    print(
        "\nValidation-selected threshold:",
        f"{threshold:.2f}"
    )


    print(
        "\nVALIDATION RESULTS"
    )


    print(
        f"Accuracy  : "
        f"{validation_metrics['accuracy'] * 100:.2f}%"
    )


    print(
        f"Precision : "
        f"{validation_metrics['precision'] * 100:.2f}%"
    )


    print(
        f"Recall    : "
        f"{validation_metrics['recall'] * 100:.2f}%"
    )


    print(
        f"F1 Score  : "
        f"{validation_metrics['f1'] * 100:.2f}%"
    )


    print(
        "\nFINAL TEST RESULTS"
    )


    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
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
        f"ROC-AUC   : {auc:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        cm
    )


    return {

        "method": method_name,

        "threshold": threshold,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "auc": auc

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "EXP13 VIDEO AGGREGATION ANALYSIS"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # LOAD DATASETS
    # ========================================================

    print(
        "\nLoading validation dataset..."
    )


    (

        X_validation,

        y_validation,

        validation_video_ids

    ) = load_dataset(
        "validation"
    )


    print(
        "Validation sequences:",
        len(X_validation)
    )


    print(
        "\nLoading test dataset..."
    )


    (

        X_test,

        y_test,

        test_video_ids

    ) = load_dataset(
        "test"
    )


    print(
        "Test sequences:",
        len(X_test)
    )


    # ========================================================
    # LOAD NORMALIZATION
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
    # NORMALIZE DATA
    # ========================================================

    print(
        "\nNormalizing data..."
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
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading EXP13 model..."
    )


    model = tf.keras.models.load_model(

        MODEL_PATH

    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(
        "\nRunning validation predictions..."
    )


    validation_probabilities = model.predict(

        X_validation,

        verbose=0

    ).flatten()


    print(
        "Running test predictions..."
    )


    test_probabilities = model.predict(

        X_test,

        verbose=0

    ).flatten()


    # ========================================================
    # AGGREGATION METHODS
    # ========================================================

    methods = [

        (

            "Mean Aggregation",

            aggregate_mean

        ),

        (

            "Median Aggregation",

            aggregate_median

        ),

        (

            "Maximum Aggregation",

            aggregate_max

        ),

        (

            "Top-3 Mean Aggregation",

            lambda probabilities:

                aggregate_top_k_mean(

                    probabilities,

                    k=3

                )

        )

    ]


    results = []


    # ========================================================
    # EVALUATE METHODS
    # ========================================================

    for (

        method_name,

        aggregation_method

    ) in methods:


        result = evaluate_method(

            method_name,

            aggregation_method,

            test_probabilities,

            y_test,

            test_video_ids,

            validation_probabilities,

            y_validation,

            validation_video_ids

        )


        results.append(
            result
        )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL AGGREGATION COMPARISON"
    )

    print(
        "=" * 60
    )


    print(

        "\nMethod"

        + " " * 18

        + "Threshold"

        + "   Accuracy"

        + "   Precision"

        + "   Recall"

        + "   F1"

        + "     AUC"

    )


    print(
        "-" * 90
    )


    for result in results:


        print(

            f"{result['method']:<28}"

            f"{result['threshold']:<12.2f}"

            f"{result['accuracy'] * 100:<11.2f}"

            f"{result['precision'] * 100:<12.2f}"

            f"{result['recall'] * 100:<10.2f}"

            f"{result['f1'] * 100:<9.2f}"

            f"{result['auc']:<.4f}"

        )


    # ========================================================
    # BEST METHOD
    # ========================================================

    best_result = max(

        results,

        key=lambda result:

            result["f1"]

    )


    print(
        "\n" + "=" * 60
    )

    print(
        "BEST VIDEO AGGREGATION METHOD"
    )

    print(
        "=" * 60
    )


    print(
        "\nMethod:",
        best_result["method"]
    )


    print(
        "Threshold:",
        f"{best_result['threshold']:.2f}"
    )


    print(
        "Test Accuracy:",
        f"{best_result['accuracy'] * 100:.2f}%"
    )


    print(
        "Test F1 Score:",
        f"{best_result['f1'] * 100:.2f}%"
    )


    print(
        "Test ROC-AUC:",
        f"{best_result['auc']:.4f}"
    )


    print(
        "\nIMPORTANT:"
    )


    print(
        "Thresholds were selected using validation data only."
    )


    print(
        "Test data was used only for final evaluation."
    )


    print(
        "\nEXP13 VIDEO AGGREGATION ANALYSIS COMPLETED"
    )


if __name__ == "__main__":

    main()