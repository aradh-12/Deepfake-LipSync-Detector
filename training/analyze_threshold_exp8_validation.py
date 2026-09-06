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
    "saved_models/lstm_exp8.keras"
)

NORMALIZATION_FILE = Path(
    "saved_models/lstm_exp8_normalization.npz"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("EXP8 VALIDATION THRESHOLD ANALYSIS")
print("=" * 60)

print("\nLoading validation dataset...")

data = np.load(
    DATASET_FILE,
    allow_pickle=True
)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int32)
video_ids = data["video_ids"]

unique_videos = np.unique(video_ids)

print(f"Validation sequences: {len(X)}")
print(f"Validation videos: {len(unique_videos)}")
print(f"Real sequences: {np.sum(y == 0)}")
print(f"Fake sequences: {np.sum(y == 1)}")


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print("\nLoading normalization statistics...")

normalization = np.load(
    NORMALIZATION_FILE
)

mean = normalization["mean"]
std = normalization["std"]

print(f"Mean shape: {mean.shape}")
print(f"Std shape: {std.shape}")


# ============================================================
# NORMALIZE DATA
# ============================================================

print("\nNormalizing validation data...")

X = (X - mean) / (std + 1e-8)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Exp8 model...")

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
# VIDEO-LEVEL PROBABILITIES
# ============================================================

video_probabilities = []
video_labels = []

for video_id in unique_videos:

    indices = np.where(
        video_ids == video_id
    )[0]

    probabilities = sequence_probabilities[
        indices
    ]

    labels = y[indices]

    # All sequences belonging to one video
    # should have the same label
    unique_labels = np.unique(labels)

    if len(unique_labels) != 1:

        raise RuntimeError(
            f"Multiple labels found for video: "
            f"{video_id}"
        )

    # Average sequence probabilities
    video_probability = np.mean(
        probabilities
    )

    video_probabilities.append(
        video_probability
    )

    video_labels.append(
        unique_labels[0]
    )


video_probabilities = np.array(
    video_probabilities
)

video_labels = np.array(
    video_labels
)


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

thresholds = [
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
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    true_labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

best_threshold = None
best_video_f1 = -1

best_video_metrics = None


for threshold in thresholds:

    # --------------------------------------------------------
    # Sequence-level metrics
    # --------------------------------------------------------

    (
        sequence_accuracy,
        sequence_precision,
        sequence_recall,
        sequence_f1
    ) = calculate_metrics(
        y,
        sequence_probabilities,
        threshold
    )

    # --------------------------------------------------------
    # Video-level metrics
    # --------------------------------------------------------

    (
        video_accuracy,
        video_precision,
        video_recall,
        video_f1
    ) = calculate_metrics(
        video_labels,
        video_probabilities,
        threshold
    )

    print(f"\nThreshold: {threshold:.2f}")

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
    # Select best threshold using VIDEO F1
    # --------------------------------------------------------

    if video_f1 > best_video_f1:

        best_video_f1 = video_f1

        best_threshold = threshold

        best_video_metrics = (
            video_accuracy,
            video_precision,
            video_recall,
            video_f1
        )


# ============================================================
# BEST THRESHOLD
# ============================================================

(
    best_accuracy,
    best_precision,
    best_recall,
    best_f1
) = best_video_metrics


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
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Video Precision : "
    f"{best_precision * 100:.2f}%"
)

print(
    f"Video Recall    : "
    f"{best_recall * 100:.2f}%"
)

print(
    f"Video F1 Score  : "
    f"{best_f1 * 100:.2f}%"
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

print(
    "\nEXP8 VALIDATION THRESHOLD ANALYSIS COMPLETED"
)

print("=" * 60)
