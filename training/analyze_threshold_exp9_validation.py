from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = Path(
    "outputs/dataset_exp8_temporal/validation.npz"
)

MODEL_FILE = Path(
    "saved_models/lstm_exp9.keras"
)

NORMALIZATION_FILE = Path(
    "saved_models/lstm_exp9_normalization.npz"
)


# Thresholds to test
THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("EXP9 VALIDATION THRESHOLD ANALYSIS")
print("=" * 60)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

data = np.load(
    DATASET_FILE,
    allow_pickle=True
)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int32)
video_ids = data["video_ids"]

unique_videos = np.unique(
    video_ids
)

print(
    f"Validation sequences: {len(X)}"
)

print(
    f"Validation videos: {len(unique_videos)}"
)

print(
    f"Real sequences: {np.sum(y == 0)}"
)

print(
    f"Fake sequences: {np.sum(y == 1)}"
)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

print("\nLoading normalization statistics...")

normalization = np.load(
    NORMALIZATION_FILE
)

mean = normalization["mean"]
std = normalization["std"]

print(
    f"Mean shape: {mean.shape}"
)

print(
    f"Std shape: {std.shape}"
)


# ============================================================
# NORMALIZE VALIDATION DATA
# ============================================================

print("\nNormalizing validation data...")

X = (
    X - mean
) / (
    std + 1e-8
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Exp9 model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning predictions...")

sequence_probabilities = model.predict(
    X,
    verbose=0
).reshape(-1)


# ============================================================
# VIDEO AGGREGATION FUNCTION
# ============================================================

def get_video_predictions(
    probabilities,
    labels,
    video_ids,
    threshold
):

    video_probabilities = []
    video_labels = []

    unique_videos = np.unique(
        video_ids
    )

    for video_id in unique_videos:

        indices = (
            video_ids == video_id
        )

        video_probs = (
            probabilities[indices]
        )

        video_label_values = (
            labels[indices]
        )

        # Average probability across sequences
        video_probability = np.mean(
            video_probs
        )

        # All sequences of a video
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

    video_probabilities = np.array(
        video_probabilities
    )

    video_labels = np.array(
        video_labels
    )

    video_predictions = (
        video_probabilities >= threshold
    ).astype(int)

    return (
        video_probabilities,
        video_labels,
        video_predictions
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)


best_threshold = None
best_video_f1 = -1

best_results = None


for threshold in THRESHOLDS:

    # --------------------------------------------------------
    # SEQUENCE LEVEL
    # --------------------------------------------------------

    sequence_predictions = (

        sequence_probabilities >= threshold

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


    # --------------------------------------------------------
    # VIDEO LEVEL
    # --------------------------------------------------------

    (
        video_probabilities,
        video_labels,
        video_predictions

    ) = get_video_predictions(

        sequence_probabilities,
        y,
        video_ids,
        threshold

    )


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


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print("-" * 50)

    print("SEQUENCE LEVEL")

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


    print("\nVIDEO LEVEL")

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


    # --------------------------------------------------------
    # SELECT BEST THRESHOLD
    # --------------------------------------------------------

    if video_f1 > best_video_f1:

        best_video_f1 = video_f1

        best_threshold = threshold

        best_results = {

            "video_accuracy":
                video_accuracy,

            "video_precision":
                video_precision,

            "video_recall":
                video_recall,

            "video_f1":
                video_f1

        }


# ============================================================
# BEST VALIDATION THRESHOLD
# ============================================================

print("\n" + "=" * 60)
print("BEST VALIDATION THRESHOLD")
print("=" * 60)


print(
    f"\nSelected threshold: "
    f"{best_threshold:.2f}"
)


print(
    "\nSelection metric:"
)

print(
    "VIDEO-LEVEL F1 SCORE"
)


print()

print(
    f"Video Accuracy  : "
    f"{best_results['video_accuracy'] * 100:.2f}%"
)

print(
    f"Video Precision : "
    f"{best_results['video_precision'] * 100:.2f}%"
)

print(
    f"Video Recall    : "
    f"{best_results['video_recall'] * 100:.2f}%"
)

print(
    f"Video F1 Score  : "
    f"{best_results['video_f1'] * 100:.2f}%"
)


# ============================================================
# IMPORTANT NOTE
# ============================================================

print("\n" + "=" * 60)

print(
    "IMPORTANT:"
)

print(
    "Threshold selected using VALIDATION DATA ONLY."
)

print(
    "Do NOT use test data to select the threshold."
)

print("=" * 60)


# ============================================================
# COMPLETED
# ============================================================

print(
    "\nEXP9 VALIDATION THRESHOLD ANALYSIS COMPLETED"
)

print("=" * 60)
