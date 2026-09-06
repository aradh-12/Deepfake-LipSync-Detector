from pathlib import Path
import csv

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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


FINAL_PIPELINE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
)


MULTIMODAL_FEATURE_ROOT = (
    FINAL_PIPELINE_ROOT
    / "multimodal_features"
)


MODEL_ROOT = (
    FINAL_PIPELINE_ROOT
    / "models"
)


TEST_FEATURE_ROOT = (
    MULTIMODAL_FEATURE_ROOT
    / "test"
)


MODEL_PATH = (
    MODEL_ROOT
    / "best_lstm_multimodal_model.keras"
)


NORMALIZATION_PATH = (
    MODEL_ROOT
    / "normalization_stats.npz"
)


EVALUATION_ROOT = (
    FINAL_PIPELINE_ROOT
    / "evaluation"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SEQUENCE_LENGTH = 150

FEATURE_SIZE = 119

CLASS_NAMES = {
    0: "REAL",
    1: "FAKE"
}


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    print()

    print("=" * 70)

    print(
        "CHECKING REQUIRED FILES"
    )

    print("=" * 70)

    print()

    print(
        "Test features:"
    )

    print(
        TEST_FEATURE_ROOT
    )

    print()

    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Normalization statistics:"
    )

    print(
        NORMALIZATION_PATH
    )

    print()

    if not TEST_FEATURE_ROOT.exists():

        raise FileNotFoundError(

            "Test feature directory not found:\n"
            f"{TEST_FEATURE_ROOT}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            "Model file not found:\n"
            f"{MODEL_PATH}"
        )

    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(

            "Normalization statistics not found:\n"
            f"{NORMALIZATION_PATH}"
        )

    print(
        "✅ All required files found."
    )


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

def load_normalization_statistics():

    print()

    print("=" * 70)

    print(
        "LOADING NORMALIZATION STATISTICS"
    )

    print("=" * 70)

    print()

    with np.load(
        NORMALIZATION_PATH
    ) as data:

        mean = data["mean"]

        std = data["std"]

    mean = mean.astype(
        np.float32
    )

    std = std.astype(
        np.float32
    )

    if mean.shape != (
        FEATURE_SIZE,
    ):

        raise ValueError(

            f"Invalid mean shape: "
            f"{mean.shape}"
        )

    if std.shape != (
        FEATURE_SIZE,
    ):

        raise ValueError(

            f"Invalid std shape: "
            f"{std.shape}"
        )

    std = np.maximum(

        std,

        1e-6
    )

    print(
        "Mean shape:",
        mean.shape
    )

    print(
        "Std shape:",
        std.shape
    )

    print()

    return mean, std


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    print()

    print("=" * 70)

    print(
        "LOADING TEST DATASET"
    )

    print("=" * 70)

    print()

    feature_files = sorted(

        TEST_FEATURE_ROOT.glob(
            "*.npz"
        )
    )

    if not feature_files:

        raise ValueError(

            "No test feature files found."
        )

    print(

        "Test samples found:",

        len(feature_files)
    )

    print()

    features_list = []

    labels_list = []

    video_ids = []

    for index, file_path in enumerate(

        feature_files,

        start=1
    ):

        with np.load(
            file_path
        ) as data:

            features = data[
                "features"
            ]

            label = int(
                data["label"]
            )

            video_id = str(
                data["video_id"]
            )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if features.ndim != 2:

            raise ValueError(

                f"Invalid feature dimensions "
                f"in {file_path}: "
                f"{features.shape}"
            )

        if features.shape[1] != FEATURE_SIZE:

            raise ValueError(

                f"Invalid feature size "
                f"in {file_path}: "
                f"{features.shape}"
            )

        if features.shape[0] == 0:

            raise ValueError(

                f"Empty feature sequence "
                f"in {file_path}"
            )

        if not np.isfinite(
            features
        ).all():

            raise ValueError(

                f"NaN or Inf found "
                f"in {file_path}"
            )

        if label not in [

            0,

            1

        ]:

            raise ValueError(

                f"Invalid label {label} "
                f"in {file_path}"
            )

        # ----------------------------------------------------
        # Pad sequence
        # ----------------------------------------------------

        padded = np.zeros(

            (
                MAX_SEQUENCE_LENGTH,
                FEATURE_SIZE
            ),

            dtype=np.float32
        )

        length = min(

            features.shape[0],

            MAX_SEQUENCE_LENGTH
        )

        padded[
            :length
        ] = features[
            :length
        ]

        features_list.append(

            padded
        )

        labels_list.append(

            label
        )

        video_ids.append(

            video_id
        )

        if (

            index % 100 == 0

            or

            index == len(
                feature_files
            )

        ):

            print(

                f"[{index}/"
                f"{len(feature_files)}] "
                f"Loaded"
            )

    X = np.asarray(

        features_list,

        dtype=np.float32
    )

    y = np.asarray(

        labels_list,

        dtype=np.int32
    )

    print()

    print(
        "Dataset shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    print()

    print(
        "Real samples:",
        int(
            np.sum(y == 0)
        )
    )

    print(
        "Fake samples:",
        int(
            np.sum(y == 1)
        )
    )

    print()

    return X, y, video_ids


# ============================================================
# NORMALIZE DATASET
# ============================================================

def normalize_dataset(

    X,

    mean,

    std

):

    print()

    print("=" * 70)

    print(
        "NORMALIZING TEST DATASET"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Detect padded frames
    #
    # Padded frames are completely zero.
    # --------------------------------------------------------

    valid_mask = np.any(

        X != 0,

        axis=2
    )

    X_normalized = X.copy()

    # --------------------------------------------------------
    # Normalize only valid frames
    # --------------------------------------------------------

    valid_features = (

        X[
            valid_mask
        ]

        - mean

    ) / std

    X_normalized[
        valid_mask
    ] = valid_features

    # --------------------------------------------------------
    # Keep padding zero
    # --------------------------------------------------------

    X_normalized[
        ~valid_mask
    ] = 0.0

    if not np.isfinite(
        X_normalized
    ).all():

        raise ValueError(

            "NaN or Inf found "
            "after normalization"
        )

    print(
        "Normalization complete."
    )

    print()

    return X_normalized.astype(

        np.float32
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()

    print("=" * 70)

    print(
        "LOADING TRAINED MODEL"
    )

    print("=" * 70)

    print()

    model = tf.keras.models.load_model(

        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    print()

    print(
        "Model input shape:",
        model.input_shape
    )

    print(
        "Model output shape:",
        model.output_shape
    )

    print()

    return model


# ============================================================
# MAKE PREDICTIONS
# ============================================================

def make_predictions(

    model,

    X

):

    print()

    print("=" * 70)

    print(
        "RUNNING MODEL PREDICTIONS"
    )

    print("=" * 70)

    print()

    probabilities = model.predict(

        X,

        batch_size=16,

        verbose=1
    )

    probabilities = probabilities.reshape(

        -1
    )

    predictions = (

        probabilities >= 0.5

    ).astype(

        np.int32
    )

    print()

    print(
        "Predictions complete."
    )

    print()

    return (

        probabilities,

        predictions
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(

    y_true,

    y_pred,

    probabilities

):

    print()

    print("=" * 70)

    print(
        "MODEL EVALUATION RESULTS"
    )

    print("=" * 70)

    print()

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

    try:

        roc_auc = roc_auc_score(

            y_true,

            probabilities
        )

    except ValueError:

        roc_auc = None

    print(
        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1:.4f}"
    )

    if roc_auc is not None:

        print(
            f"ROC-AUC  : "
            f"{roc_auc:.4f}"
        )

    print()

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(

    y_true,

    y_pred

):

    print()

    print("=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)

    print()

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]
    )

    tn = matrix[0, 0]

    fp = matrix[0, 1]

    fn = matrix[1, 0]

    tp = matrix[1, 1]

    print()

    print(
        "                 PREDICTED"
    )

    print(
        "              REAL     FAKE"
    )

    print(
        f"ACTUAL REAL   {tn:5d}    {fp:5d}"
    )

    print(
        f"ACTUAL FAKE   {fn:5d}    {tp:5d}"
    )

    print()

    print(
        "True Negatives :",
        tn
    )

    print(
        "False Positives:",
        fp
    )

    print(
        "False Negatives:",
        fn
    )

    print(
        "True Positives :",
        tp
    )

    print()

    return matrix


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def print_classification_report(

    y_true,

    y_pred

):

    print()

    print("=" * 70)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 70)

    print()

    report = classification_report(

        y_true,

        y_pred,

        target_names=[

            "REAL",

            "FAKE"

        ],

        digits=4,

        zero_division=0
    )

    print(
        report
    )

    return report


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(

    video_ids,

    y_true,

    probabilities,

    y_pred

):

    EVALUATION_ROOT.mkdir(

        parents=True,

        exist_ok=True
    )

    output_path = (

        EVALUATION_ROOT
        / "test_predictions.csv"
    )

    with open(

        output_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.writer(

            file
        )

        writer.writerow(

            [

                "video_id",

                "actual_label",

                "actual_class",

                "fake_probability",

                "predicted_label",

                "predicted_class",

                "correct"

            ]
        )

        for (

            video_id,

            actual,

            probability,

            predicted

        ) in zip(

            video_ids,

            y_true,

            probabilities,

            y_pred

        ):

            writer.writerow(

                [

                    video_id,

                    int(actual),

                    CLASS_NAMES[
                        int(actual)
                    ],

                    float(probability),

                    int(predicted),

                    CLASS_NAMES[
                        int(predicted)
                    ],

                    int(actual)
                    ==
                    int(predicted)

                ]
            )

    print()

    print(
        "Predictions saved:"
    )

    print(
        output_path
    )

    print()

    return output_path


# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

def save_metrics(

    metrics

):

    EVALUATION_ROOT.mkdir(

        parents=True,

        exist_ok=True
    )

    output_path = (

        EVALUATION_ROOT
        / "evaluation_metrics.txt"
    )

    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            "MULTIMODAL DEEPFAKE "
            "LIP-SYNC DETECTOR\n"
        )

        file.write(

            "=" * 60
        )

        file.write(

            "\n\n"
        )

        file.write(

            "TEST SET EVALUATION\n\n"
        )

        file.write(

            f"Accuracy: "
            f"{metrics['accuracy']:.6f}\n"
        )

        file.write(

            f"Precision: "
            f"{metrics['precision']:.6f}\n"
        )

        file.write(

            f"Recall: "
            f"{metrics['recall']:.6f}\n"
        )

        file.write(

            f"F1 Score: "
            f"{metrics['f1_score']:.6f}\n"
        )

        if (

            metrics["roc_auc"]
            is not None

        ):

            file.write(

                f"ROC-AUC: "
                f"{metrics['roc_auc']:.6f}\n"
            )

    print(
        "Metrics saved:"
    )

    print(
        output_path
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "MULTIMODAL LSTM MODEL EVALUATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    mean, std = (

        load_normalization_statistics()
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    X_test, y_test, video_ids = (

        load_test_dataset()
    )

    # --------------------------------------------------------
    # Normalize test dataset
    # --------------------------------------------------------

    X_test = (

        normalize_dataset(

            X_test,

            mean,

            std
        )
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = (

        load_model()
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities, predictions = (

        make_predictions(

            model,

            X_test
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = (

        calculate_metrics(

            y_test,

            predictions,

            probabilities
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(

        y_test,

        predictions
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print_classification_report(

        y_test,

        predictions
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_predictions(

        video_ids,

        y_test,

        probabilities,

        predictions
    )

    save_metrics(

        metrics
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "MODEL EVALUATION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "Final Test Accuracy:"
    )

    print(

        f"{metrics['accuracy']:.4f}"
    )

    print()

    print(
        "Evaluation directory:"
    )

    print(
        EVALUATION_ROOT
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()