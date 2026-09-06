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
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp7_source_aware"
)

MODEL_PATH = Path(
    "saved_models/lstm_exp7.keras"
)

NORMALIZATION_PATH = Path(
    "saved_models/lstm_exp7_normalization.npz"
)


# ============================================================
# LOCKED THRESHOLD
#
# Selected using VALIDATION DATA ONLY
# ============================================================

THRESHOLD = 0.35


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    dataset_path = DATASET_DIR / "test.npz"

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Test dataset not found: "
            f"{dataset_path}"
        )

    data = np.load(
        dataset_path,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]

    return X, y, video_ids


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(X):

    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(
            f"Normalization file not found: "
            f"{NORMALIZATION_PATH}"
        )

    normalization_data = np.load(
        NORMALIZATION_PATH
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
# VIDEO-LEVEL AGGREGATION
# ============================================================

def aggregate_video_predictions(
    video_ids,
    y_true,
    probabilities
):

    unique_videos = np.unique(
        video_ids
    )

    video_labels = []
    video_probabilities = []

    for video in unique_videos:

        indices = np.where(
            video_ids == video
        )[0]

        labels = y_true[indices]

        probs = probabilities[indices]

        unique_labels = np.unique(
            labels
        )

        if len(unique_labels) != 1:

            raise RuntimeError(
                f"Video {video} has "
                f"multiple labels: "
                f"{unique_labels}"
            )

        video_labels.append(
            unique_labels[0]
        )

        # Average sequence probabilities
        # to obtain one probability per video

        video_probability = np.mean(
            probs
        )

        video_probabilities.append(
            video_probability
        )

    return (

        np.asarray(
            video_labels,
            dtype=np.int32
        ),

        np.asarray(
            video_probabilities,
            dtype=np.float32
        ),

        unique_videos

    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions
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

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)

    print(
        "EXP7 FINAL TEST EVALUATION"
    )

    print(
        "SOURCE-AWARE DATASET"
    )

    print("=" * 60)

    print(
        f"\nLocked threshold: "
        f"{THRESHOLD:.2f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This threshold was selected "
        "using validation data only."
    )

    # --------------------------------------------------------
    # Load Test Dataset
    # --------------------------------------------------------

    print(
        "\nLoading test dataset..."
    )

    X_test, y_test, video_ids = (
        load_test_data()
    )

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

    # --------------------------------------------------------
    # Normalize Test Data
    # --------------------------------------------------------

    print(
        "\nNormalizing test data..."
    )

    X_test = normalize_data(
        X_test
    )

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    print(
        "\nLoading Exp7 model..."
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        "\nRunning test predictions..."
    )

    probabilities = model.predict(
        X_test,
        verbose=0
    ).flatten()

    sequence_predictions = (
        probabilities >= THRESHOLD
    ).astype(
        np.int32
    )

    # --------------------------------------------------------
    # Sequence-Level Evaluation
    # --------------------------------------------------------

    sequence_results = (
        calculate_metrics(
            y_test,
            sequence_predictions
        )
    )

    (
        sequence_accuracy,
        sequence_precision,
        sequence_recall,
        sequence_f1,
        sequence_cm
    ) = sequence_results

    # --------------------------------------------------------
    # Video-Level Aggregation
    # --------------------------------------------------------

    (
        video_labels,
        video_probabilities,
        unique_videos
    ) = aggregate_video_predictions(

        video_ids,
        y_test,
        probabilities

    )

    video_predictions = (
        video_probabilities >= THRESHOLD
    ).astype(
        np.int32
    )

    # --------------------------------------------------------
    # Video-Level Evaluation
    # --------------------------------------------------------

    video_results = (
        calculate_metrics(
            video_labels,
            video_predictions
        )
    )

    (
        video_accuracy,
        video_precision,
        video_recall,
        video_f1,
        video_cm
    ) = video_results

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "SEQUENCE-LEVEL TEST RESULTS"
    )

    print("=" * 60)

    print(
        f"\nAccuracy  : "
        f"{sequence_accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{sequence_precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{sequence_recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{sequence_f1 * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        sequence_cm
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

    # ========================================================

    print("\n" + "=" * 60)

    print(
        "VIDEO-LEVEL TEST RESULTS"
    )

    print("=" * 60)

    print(
        f"\nAccuracy  : "
        f"{video_accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{video_precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{video_recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{video_f1 * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        video_cm
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

    # ========================================================
    # VIDEO-WISE RESULTS
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "VIDEO-WISE TEST PREDICTIONS"
    )

    print("=" * 60)

    for video, label, probability, prediction in zip(

        unique_videos,

        video_labels,

        video_probabilities,

        video_predictions

    ):

        actual_name = (
            "Fake"
            if label == 1
            else "Real"
        )

        predicted_name = (
            "Fake"
            if prediction == 1
            else "Real"
        )

        result = (
            "CORRECT"
            if label == prediction
            else "WRONG"
        )

        print(

            f"\nVideo: {video}"

        )

        print(

            f"Actual: {actual_name}"

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

            f"Result: {result}"

        )

    # ========================================================

    print("\n" + "=" * 60)

    print(
        "EXP7 FINAL TEST EVALUATION COMPLETED"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()