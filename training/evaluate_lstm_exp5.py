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


MODEL_FOLDER = Path(
    "models"
)


MODEL_FILE = (
    MODEL_FOLDER /
    "lstm_exp6_best.keras"
)


NORMALIZATION_FILE = (
    MODEL_FOLDER /
    "lstm_exp6_normalization.npz"
)


# ============================================================
# Model Configuration
# ============================================================

SEQUENCE_LENGTH = 30

FEATURE_SIZE = 199


# ============================================================
# Classification Threshold
# ============================================================

THRESHOLD = 0.5


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(
    dataset_file
):

    if not dataset_file.exists():

        raise FileNotFoundError(

            f"Dataset file not found: "
            f"{dataset_file}"

        )


    data = np.load(

        dataset_file,

        allow_pickle=True

    )


    X = data["X"].astype(

        np.float32

    )


    y = data["y"].astype(

        np.int32

    )


    video_ids = data["video_ids"]


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(

            f"Expected X to have 3 dimensions, "
            f"got {X.ndim}"

        )


    if X.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(

            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"got {X.shape[1]}"

        )


    if X.shape[2] != FEATURE_SIZE:

        raise ValueError(

            f"Expected feature size "
            f"{FEATURE_SIZE}, "
            f"got {X.shape[2]}"

        )


    if len(X) != len(y):

        raise ValueError(

            "X and y length mismatch."

        )


    if len(X) != len(video_ids):

        raise ValueError(

            "X and video_ids length mismatch."

        )


    # --------------------------------------------------------
    # Check NaN
    # --------------------------------------------------------

    if np.isnan(X).any():

        raise ValueError(

            "NaN values found in X."

        )


    # --------------------------------------------------------
    # Check Inf
    # --------------------------------------------------------

    if np.isinf(X).any():

        raise ValueError(

            "Inf values found in X."

        )


    return (

        X,

        y,

        video_ids

    )


# ============================================================
# Load Normalization
# ============================================================

def load_normalization():

    if not NORMALIZATION_FILE.exists():

        raise FileNotFoundError(

            f"Normalization file not found: "
            f"{NORMALIZATION_FILE}"

        )


    data = np.load(

        NORMALIZATION_FILE

    )


    mean = data["mean"].astype(

        np.float32

    )


    std = data["std"].astype(

        np.float32

    )


    return (

        mean,

        std

    )


# ============================================================
# Normalize Features
# ============================================================

def normalize_features(
    X,
    mean,
    std
):

    std = np.maximum(

        std,

        1e-6

    )


    X_normalized = (

        X - mean

    ) / std


    return X_normalized.astype(

        np.float32

    )


# ============================================================
# Print Sequence-Level Results
# ============================================================

def print_sequence_results(
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


    matrix = confusion_matrix(

        y_true,

        y_pred

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

            y_pred,

            target_names=[

                "Real",

                "Fake"

            ],

            zero_division=0

        )

    )


# ============================================================
# Video-Level Prediction
# ============================================================

def get_video_predictions(
    probabilities,
    y_true,
    video_ids
):

    unique_video_ids = np.unique(

        video_ids

    )


    video_true_labels = []

    video_probabilities = []

    video_predictions = []

    video_names = []


    for video_id in unique_video_ids:


        indices = np.where(

            video_ids == video_id

        )[0]


        video_probs = probabilities[

            indices

        ]


        video_labels = y_true[

            indices

        ]


        # ----------------------------------------------------
        # Average probabilities of all sequences
        # belonging to the same video
        # ----------------------------------------------------

        average_probability = np.mean(

            video_probs

        )


        # ----------------------------------------------------
        # All sequences from same video should
        # have same label
        # ----------------------------------------------------

        unique_labels = np.unique(

            video_labels

        )


        if len(unique_labels) != 1:

            raise ValueError(

                f"Video {video_id} "
                f"has inconsistent labels."

            )


        true_label = int(

            unique_labels[0]

        )


        predicted_label = int(

            average_probability >= THRESHOLD

        )


        video_names.append(

            video_id

        )


        video_true_labels.append(

            true_label

        )


        video_probabilities.append(

            float(average_probability)

        )


        video_predictions.append(

            predicted_label

        )


    return (

        np.array(

            video_true_labels

        ),

        np.array(

            video_probabilities

        ),

        np.array(

            video_predictions

        ),

        video_names

    )


# ============================================================
# Print Individual Video Predictions
# ============================================================

def print_video_predictions(
    video_names,
    video_true_labels,
    video_probabilities,
    video_predictions
):

    print(

        "\n" + "=" * 60

    )

    print(

        "VIDEO-LEVEL PREDICTIONS"

    )

    print(

        "=" * 60 + "\n"

    )


    for (

        video_name,
        true_label,
        probability,
        predicted_label

    ) in zip(

        video_names,

        video_true_labels,

        video_probabilities,

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


        print(

            video_name

        )


        print(

            f"True       : {true_name}"

        )


        print(

            f"Probability: {probability:.4f}"

        )


        print(

            f"Predicted  : {predicted_name}"

        )


        print(

            "-" * 50

        )


# ============================================================
# Print Video-Level Results
# ============================================================

def print_video_results(
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


    matrix = confusion_matrix(

        y_true,

        y_pred

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


    # --------------------------------------------------------
    # Load Test Dataset
    # --------------------------------------------------------

    print(

        "\nLoading test dataset..."

    )


    (

        X_test,
        y_test,
        test_video_ids

    ) = load_dataset(

        TEST_FILE

    )


    # --------------------------------------------------------
    # Print Dataset Information
    # --------------------------------------------------------

    print()


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
    # Load Normalization
    # --------------------------------------------------------

    print(

        "\nLoading normalization..."

    )


    (

        feature_mean,
        feature_std

    ) = load_normalization()


    # --------------------------------------------------------
    # Normalize Test Data
    # --------------------------------------------------------

    print(

        "Normalizing test data..."

    )


    X_test = normalize_features(

        X_test,

        feature_mean,

        feature_std

    )


    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(

            f"Model file not found: "
            f"{MODEL_FILE}"

        )


    print(

        "\nLoading model..."

    )


    model = tf.keras.models.load_model(

        MODEL_FILE

    )


    # --------------------------------------------------------
    # Run Predictions
    # --------------------------------------------------------

    print(

        "\nRunning predictions..."

    )


    probabilities = model.predict(

        X_test,

        verbose=0

    )


    probabilities = probabilities.reshape(

        -1

    )


    # --------------------------------------------------------
    # Convert Probabilities to Predictions
    # --------------------------------------------------------

    y_pred = (

        probabilities >= THRESHOLD

    ).astype(

        np.int32

    )


    # ========================================================
    # Sequence-Level Evaluation
    # ========================================================

    print_sequence_results(

        y_test,

        y_pred

    )


    # ========================================================
    # Video-Level Evaluation
    # ========================================================

    (

        video_true_labels,
        video_probabilities,
        video_predictions,
        video_names

    ) = get_video_predictions(

        probabilities,

        y_test,

        test_video_ids

    )


    print_video_predictions(

        video_names,

        video_true_labels,

        video_probabilities,

        video_predictions

    )


    print_video_results(

        video_true_labels,

        video_predictions

    )


    # ========================================================
    # Completed
    # ========================================================

    print(

        "\n" + "=" * 60

    )

    print(

        "EXP6 TEST EVALUATION COMPLETED"

    )

    print(

        "=" * 60

    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()