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

DATASET_FILE = Path(
    "outputs/dataset_exp8_temporal/test.npz"
)

MODEL_FILE = Path(
    "saved_models/lstm_exp8.keras"
)

NORMALIZATION_FILE = Path(
    "saved_models/lstm_exp8_normalization.npz"
)

THRESHOLD = 0.65


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("EXP8 FINAL TEST EVALUATION")
print("TEMPORAL FEATURE DATASET")
print("=" * 60)

print(
    f"\nLocked threshold: {THRESHOLD:.2f}"
)

print(
    "\nIMPORTANT:"
)

print(
    "This threshold was selected using validation data only."
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

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

print(f"Test sequences: {len(X)}")
print(f"Test videos: {len(unique_videos)}")
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


# ============================================================
# NORMALIZE TEST DATA
# ============================================================

print("\nNormalizing test data...")

X = (
    X - mean
) / (
    std + 1e-8
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Exp8 model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# RUN TEST PREDICTIONS
# ============================================================

print("\nRunning test predictions...")

sequence_probabilities = model.predict(
    X,
    verbose=0
).reshape(-1)


sequence_predictions = (
    sequence_probabilities >= THRESHOLD
).astype(int)


# ============================================================
# SEQUENCE-LEVEL RESULTS
# ============================================================

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


print("\n" + "=" * 60)
print("SEQUENCE-LEVEL TEST RESULTS")
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


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y,
        sequence_predictions
    )
)


print("\nClassification Report:")

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


# ============================================================
# VIDEO-LEVEL PREDICTIONS
# ============================================================

video_probabilities = []
video_labels = []
video_names = []


for video_id in unique_videos:

    indices = np.where(
        video_ids == video_id
    )[0]

    probabilities = sequence_probabilities[
        indices
    ]

    labels = y[indices]

    unique_labels = np.unique(
        labels
    )

    if len(unique_labels) != 1:

        raise RuntimeError(
            f"Multiple labels found for "
            f"video: {video_id}"
        )

    # --------------------------------------------------------
    # Average sequence probabilities for one video
    # --------------------------------------------------------

    video_probability = np.mean(
        probabilities
    )

    video_probabilities.append(
        video_probability
    )

    video_labels.append(
        unique_labels[0]
    )

    video_names.append(
        video_id
    )


video_probabilities = np.array(
    video_probabilities
)

video_labels = np.array(
    video_labels
)

video_predictions = (
    video_probabilities >= THRESHOLD
).astype(int)


# ============================================================
# VIDEO-LEVEL RESULTS
# ============================================================

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


print("\n" + "=" * 60)
print("VIDEO-LEVEL TEST RESULTS")
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


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        video_labels,
        video_predictions
    )
)


print("\nClassification Report:")

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


# ============================================================
# VIDEO-WISE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("VIDEO-WISE TEST PREDICTIONS")
print("=" * 60)


for (
    video_name,
    actual_label,
    probability,
    prediction
) in zip(

    video_names,
    video_labels,
    video_probabilities,
    video_predictions

):

    actual_name = (
        "Fake"
        if actual_label == 1
        else "Real"
    )

    predicted_name = (
        "Fake"
        if prediction == 1
        else "Real"
    )

    result = (
        "CORRECT"
        if actual_label == prediction
        else "WRONG"
    )

    print(
        f"\nVideo: {video_name}"
    )

    print(
        f"Actual: {actual_name}"
    )

    print(
        f"Probability: "
        f"{probability:.4f}"
    )

    print(
        f"Predicted: {predicted_name}"
    )

    print(
        f"Result: {result}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXP8 FINAL TEST EVALUATION COMPLETED")
print("=" * 60)
