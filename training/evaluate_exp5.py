import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Configuration
# ============================================================
DATASET_DIR = "outputs/dataset_exp5_full"

MODEL_PATH = (
    "models/deepfake_lipsync_lstm_exp5.keras"
)

SCALER_PATH = (
    "models/deepfake_lipsync_feature_scaler_exp5.npz"
)


# ============================================================
# Load Test Dataset
# ============================================================

test_data = np.load(
    f"{DATASET_DIR}/test.npz",
    allow_pickle=True
)

X_test = test_data["X"]
y_test = test_data["y"]
video_ids = test_data["video_ids"]


print("\n" + "=" * 70)
print("EXP5 MODEL EVALUATION")
print("=" * 70)

print("Test X shape :", X_test.shape)
print("Test y shape :", y_test.shape)
print("Test videos  :", len(np.unique(video_ids)))

print(
    "Real samples :",
    int(np.sum(y_test == 0))
)

print(
    "Fake samples :",
    int(np.sum(y_test == 1))
)


# ============================================================
# Load Scaler
# ============================================================

scaler_data = np.load(
    SCALER_PATH
)

mean = scaler_data["mean"]
scale = scaler_data["scale"]

original_shape = X_test.shape

X_test_flat = X_test.reshape(
    -1,
    original_shape[-1]
)

X_test_flat = (
    X_test_flat - mean
) / scale

X_test = X_test_flat.reshape(
    original_shape
).astype(np.float32)


# ============================================================
# Load Model
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Model loaded.")


# ============================================================
# Sequence-Level Predictions
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=0
).reshape(-1)

predictions = (
    probabilities >= 0.5
).astype(np.int32)


# ============================================================
# Sequence-Level Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n" + "=" * 70)
print("SEQUENCE-LEVEL RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")


print("\nConfusion Matrix")
print("----------------")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)


# ============================================================
# Video-Level Aggregation
# ============================================================
#
# Multiple sequences belong to the same video.
#
# We average all sequence probabilities belonging
# to each video.
#
# This gives ONE prediction per video.
# ============================================================

unique_videos = np.unique(
    video_ids
)

video_true = []
video_probabilities = []
video_predictions = []


for video in unique_videos:

    indices = (
        video_ids == video
    )

    video_sequence_probs = (
        probabilities[indices]
    )

    video_sequence_labels = (
        y_test[indices]
    )

    # --------------------------------------------------------
    # Every sequence from a video must have the same label.
    # --------------------------------------------------------

    if len(
        np.unique(
            video_sequence_labels
        )
    ) != 1:

        raise RuntimeError(
            f"Inconsistent labels for video: {video}"
        )

    true_label = int(
        video_sequence_labels[0]
    )

    # Average sequence probabilities
    video_probability = float(
        np.mean(
            video_sequence_probs
        )
    )

    video_prediction = int(
        video_probability >= 0.5
    )

    video_true.append(
        true_label
    )

    video_probabilities.append(
        video_probability
    )

    video_predictions.append(
        video_prediction
    )


video_true = np.asarray(
    video_true,
    dtype=np.int32
)

video_probabilities = np.asarray(
    video_probabilities,
    dtype=np.float32
)

video_predictions = np.asarray(
    video_predictions,
    dtype=np.int32
)


# ============================================================
# Video-Level Metrics
# ============================================================

video_accuracy = accuracy_score(
    video_true,
    video_predictions
)

video_precision = precision_score(
    video_true,
    video_predictions,
    zero_division=0
)

video_recall = recall_score(
    video_true,
    video_predictions,
    zero_division=0
)

video_f1 = f1_score(
    video_true,
    video_predictions,
    zero_division=0
)

video_auc = roc_auc_score(
    video_true,
    video_probabilities
)


print("\n" + "=" * 70)
print("VIDEO-LEVEL RESULTS")
print("=" * 70)

print(
    "Videos evaluated :",
    len(unique_videos)
)

print(
    f"Accuracy         : {video_accuracy:.4f}"
)

print(
    f"Precision        : {video_precision:.4f}"
)

print(
    f"Recall           : {video_recall:.4f}"
)

print(
    f"F1 Score         : {video_f1:.4f}"
)

print(
    f"AUC              : {video_auc:.4f}"
)


print("\nVideo-Level Confusion Matrix")
print("-----------------------------")

video_cm = confusion_matrix(
    video_true,
    video_predictions
)

print(video_cm)


print("\nVideo-Level Classification Report")
print("----------------------------------")

print(
    classification_report(
        video_true,
        video_predictions,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)


# ============================================================
# Individual Video Predictions
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL VIDEO PREDICTIONS")
print("=" * 70)

for (
    video,
    true_label,
    probability,
    prediction
) in zip(
    unique_videos,
    video_true,
    video_probabilities,
    video_predictions
):

    print(
        f"{video}"
    )

    print(
        f"  Actual     : "
        f"{'Fake' if true_label == 1 else 'Real'}"
    )

    print(
        f"  Probability: "
        f"{probability:.4f}"
    )

    print(
        f"  Predicted  : "
        f"{'Fake' if prediction == 1 else 'Real'}"
    )


print("\n" + "=" * 70)
print("EXP5 EVALUATION COMPLETED")
print("=" * 70)
