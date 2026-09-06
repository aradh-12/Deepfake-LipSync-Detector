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
    "lstm_exp11.keras"
)


NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp11_normalization.npz"
)


THRESHOLD_PATH = (
    MODEL_DIR /
    "lstm_exp11_validation_thresholds.npz"
)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 60)
print("EXP11 FINAL TEST EVALUATION")
print("=" * 60)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print()
print("Loading test dataset...")


test_path = (
    DATASET_DIR /
    "test.npz"
)


if not test_path.exists():

    raise FileNotFoundError(
        f"Test dataset not found: "
        f"{test_path}"
    )


data = np.load(
    test_path,
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


print(
    f"Test sequences: {len(X)}"
)


print(
    f"Test videos: "
    f"{len(np.unique(video_ids))}"
)


print(
    f"Real sequences: "
    f"{np.sum(y == 0)}"
)


print(
    f"Fake sequences: "
    f"{np.sum(y == 1)}"
)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

print()
print(
    "Loading normalization statistics..."
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


# ============================================================
# NORMALIZE TEST DATA
# ============================================================

print()
print(
    "Normalizing test data..."
)


X_normalized = (
    X - mean
) / std


# ============================================================
# LOAD VALIDATION THRESHOLD
# ============================================================

print()
print(
    "Loading validation-selected thresholds..."
)


threshold_data = np.load(
    THRESHOLD_PATH
)


SEQUENCE_THRESHOLD = float(
    threshold_data[
        "sequence_threshold"
    ]
)


VIDEO_THRESHOLD = float(
    threshold_data[
        "video_threshold"
    ]
)


print(
    f"Sequence threshold: "
    f"{SEQUENCE_THRESHOLD:.2f}"
)


print(
    f"Video threshold   : "
    f"{VIDEO_THRESHOLD:.2f}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Loading Exp11 model..."
)


model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# RUN TEST PREDICTIONS
# ============================================================

print()
print(
    "Running test predictions..."
)


probabilities = model.predict(
    X_normalized,
    verbose=0
).flatten()


# ============================================================
# SEQUENCE-LEVEL EVALUATION
# ============================================================

sequence_predictions = (
    probabilities >= SEQUENCE_THRESHOLD
).astype(int)


sequence_accuracy = accuracy_score(
    y,
    sequence_predictions
)


sequence_precision = precision_score(
    y,
    sequence_predictions,
    zero_division=0
)


sequence_recall = recall_score(
    y,
    sequence_predictions,
    zero_division=0
)


sequence_f1 = f1_score(
    y,
    sequence_predictions,
    zero_division=0
)


sequence_cm = confusion_matrix(
    y,
    sequence_predictions
)


sequence_auc = roc_auc_score(
    y,
    probabilities
)


print()
print("=" * 60)
print("FINAL SEQUENCE-LEVEL TEST RESULTS")
print("=" * 60)

print()

print(
    f"Threshold: "
    f"{SEQUENCE_THRESHOLD:.2f}"
)

print()

print(
    f"Accuracy  : "
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

print()

print("Confusion Matrix:")

print(sequence_cm)

print()

print("Classification Report:")

print(
    classification_report(
        y,
        sequence_predictions,
        target_names=[
            "Real",
            "Fake"
        ],
        zero_division=0
    )
)

print(
    f"ROC-AUC   : "
    f"{sequence_auc:.4f}"
)


# ============================================================
# VIDEO-LEVEL AGGREGATION
# ============================================================

unique_videos = np.unique(
    video_ids
)


video_probabilities = []

video_labels = []


for video_id in unique_videos:

    indices = np.where(
        video_ids == video_id
    )[0]


    video_probability = np.mean(
        probabilities[indices]
    )


    video_label = y[
        indices[0]
    ]


    video_probabilities.append(
        video_probability
    )


    video_labels.append(
        video_label
    )


video_probabilities = np.array(
    video_probabilities
)


video_labels = np.array(
    video_labels
)


# ============================================================
# VIDEO-LEVEL EVALUATION
# ============================================================

video_predictions = (
    video_probabilities >= VIDEO_THRESHOLD
).astype(int)


video_accuracy = accuracy_score(
    video_labels,
    video_predictions
)


video_precision = precision_score(
    video_labels,
    video_predictions,
    zero_division=0
)


video_recall = recall_score(
    video_labels,
    video_predictions,
    zero_division=0
)


video_f1 = f1_score(
    video_labels,
    video_predictions,
    zero_division=0
)


video_cm = confusion_matrix(
    video_labels,
    video_predictions
)


video_auc = roc_auc_score(
    video_labels,
    video_probabilities
)


print()
print("=" * 60)
print("FINAL VIDEO-LEVEL TEST RESULTS")
print("=" * 60)

print()

print(
    f"Threshold: "
    f"{VIDEO_THRESHOLD:.2f}"
)

print()

print(
    f"Accuracy  : "
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

print()

print("Confusion Matrix:")

print(video_cm)

print()

print("Classification Report:")

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

print(
    f"ROC-AUC   : "
    f"{video_auc:.4f}"
)


# ============================================================
# VIDEO-WISE PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("FINAL VIDEO-WISE TEST PREDICTIONS")
print("=" * 60)


for i, video_id in enumerate(
    unique_videos
):

    actual_label = (
        "Fake"
        if video_labels[i] == 1
        else "Real"
    )


    predicted_label = (
        "Fake"
        if video_predictions[i] == 1
        else "Real"
    )


    result = (
        "CORRECT"
        if video_labels[i]
        == video_predictions[i]
        else "WRONG"
    )


    print()

    print(
        f"Video: {video_id}"
    )

    print(
        f"Actual: {actual_label}"
    )

    print(
        f"Probability: "
        f"{video_probabilities[i]:.4f}"
    )

    print(
        f"Predicted: "
        f"{predicted_label}"
    )

    print(
        f"Result: {result}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("EXP11 FINAL TEST EVALUATION COMPLETED")
print("=" * 60)

print()

print(
    "IMPORTANT:"
)

print(
    "Thresholds were selected using "
    "the validation dataset."
)

print(
    "The test dataset was used only "
    "for final evaluation."
)