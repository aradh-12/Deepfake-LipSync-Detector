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
    "lstm_exp11.keras"
)

NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp11_normalization.npz"
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print()

print("=" * 60)
print("EXP11 VALIDATION THRESHOLD ANALYSIS")
print("=" * 60)

print()
print("Loading validation dataset...")

validation_path = (
    DATASET_DIR /
    "validation.npz"
)

if not validation_path.exists():

    raise FileNotFoundError(
        f"Validation dataset not found: "
        f"{validation_path}"
    )


data = np.load(
    validation_path,
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
    f"Validation sequences: {len(X)}"
)

print(
    f"Validation videos: "
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
# NORMALIZE DATA
# ============================================================

print()
print(
    "Normalizing validation data..."
)


X_normalized = (
    X - mean
) / std


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
# RUN PREDICTIONS
# ============================================================

print()
print(
    "Running validation predictions..."
)


probabilities = model.predict(
    X_normalized,
    verbose=0
).flatten()


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
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50

    best_f1 = -1

    best_accuracy = 0

    best_precision = 0

    best_recall = 0


    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )


    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)


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

            best_accuracy = accuracy

            best_precision = precision

            best_recall = recall


    return (
        best_threshold,
        best_accuracy,
        best_precision,
        best_recall,
        best_f1
    )


# ============================================================
# SEQUENCE-LEVEL THRESHOLD
# ============================================================

print()
print("=" * 60)
print(
    "SEQUENCE-LEVEL VALIDATION THRESHOLD SEARCH"
)
print("=" * 60)


(
    sequence_threshold,
    sequence_accuracy,
    sequence_precision,
    sequence_recall,
    sequence_f1
) = find_best_threshold(
    y,
    probabilities
)


sequence_predictions = (
    probabilities >= sequence_threshold
).astype(int)


sequence_cm = confusion_matrix(
    y,
    sequence_predictions
)


print()

print(
    f"Best threshold: "
    f"{sequence_threshold:.2f}"
)

print(
    f"Accuracy       : "
    f"{sequence_accuracy * 100:.2f}%"
)

print(
    f"Precision      : "
    f"{sequence_precision * 100:.2f}%"
)

print(
    f"Recall         : "
    f"{sequence_recall * 100:.2f}%"
)

print(
    f"F1 Score       : "
    f"{sequence_f1 * 100:.2f}%"
)

print()

print(
    "Confusion Matrix:"
)

print(
    sequence_cm
)


# ============================================================
# VIDEO-LEVEL THRESHOLD
# ============================================================

print()
print("=" * 60)
print(
    "VIDEO-LEVEL VALIDATION THRESHOLD SEARCH"
)
print("=" * 60)


(
    video_threshold,
    video_accuracy,
    video_precision,
    video_recall,
    video_f1
) = find_best_threshold(
    video_labels,
    video_probabilities
)


video_predictions = (
    video_probabilities >= video_threshold
).astype(int)


video_cm = confusion_matrix(
    video_labels,
    video_predictions
)


print()

print(
    f"Best threshold: "
    f"{video_threshold:.2f}"
)

print(
    f"Accuracy       : "
    f"{video_accuracy * 100:.2f}%"
)

print(
    f"Precision      : "
    f"{video_precision * 100:.2f}%"
)

print(
    f"Recall         : "
    f"{video_recall * 100:.2f}%"
)

print(
    f"F1 Score       : "
    f"{video_f1 * 100:.2f}%"
)

print()

print(
    "Confusion Matrix:"
)

print(
    video_cm
)


# ============================================================
# SAVE VALIDATION THRESHOLDS
# ============================================================

threshold_path = (
    MODEL_DIR /
    "lstm_exp11_validation_thresholds.npz"
)


np.savez(

    threshold_path,

    sequence_threshold=
    sequence_threshold,

    video_threshold=
    video_threshold

)


print()
print("=" * 60)

print(
    "VALIDATION THRESHOLDS SAVED"
)

print("=" * 60)

print()

print(
    f"Saved to:"
)

print(
    threshold_path
)


print()

print("=" * 60)

print(
    "EXP11 VALIDATION THRESHOLD ANALYSIS COMPLETED"
)

print("=" * 60)