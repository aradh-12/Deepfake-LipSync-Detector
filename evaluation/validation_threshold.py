import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


MODEL_PATH = Path(
    "models/deepfake_lipsync_lstm.keras"
)

SCALER_PATH = Path(
    "models/deepfake_lipsync_feature_scaler.npz"
)

VALIDATION_PATH = Path(
    "outputs/dataset/validation.npz"
)

THRESHOLD_PATH = Path(
    "models/threshold.json"
)

FEATURE_SIZE = 173


def load_scaler():

    data = np.load(SCALER_PATH)

    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)

    safe_scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale
    )

    return mean, safe_scale


def scale_validation_features(X):

    mean, scale = load_scaler()

    original_shape = X.shape

    X_flat = X.reshape(
        -1,
        FEATURE_SIZE
    )

    X_flat = (
        X_flat - mean
    ) / scale

    return X_flat.reshape(
        original_shape
    ).astype(np.float32)


def aggregate_by_video(
    probabilities,
    labels,
    video_ids
):

    unique_videos = np.unique(
        video_ids
    )

    video_probabilities = []
    video_labels = []

    for video in unique_videos:

        mask = (
            video_ids == video
        )

        video_probs = probabilities[mask]
        video_y = labels[mask]

        # Every sequence from one video
        # must have the same label.
        if len(np.unique(video_y)) != 1:

            raise ValueError(
                f"Inconsistent labels for video: {video}"
            )

        # IMPORTANT:
        # This matches inference/predict_video.py
        # where sequence probabilities are averaged.
        video_probability = float(
            np.mean(video_probs)
        )

        video_probabilities.append(
            video_probability
        )

        video_labels.append(
            int(video_y[0])
        )

    return (
        np.asarray(video_probabilities),
        np.asarray(video_labels)
    )


def main():

    print("\n==============================")
    print("Validation Threshold Analysis")
    print("==============================")

    # ==================================================
    # Load validation dataset
    # ==================================================

    print("\nLoading saved validation dataset...")

    data = np.load(
        VALIDATION_PATH,
        allow_pickle=True
    )

    X_val = data["X"]
    y_val = data["y"]
    video_ids = data["video_ids"]

    print(
        "X_val shape       :",
        X_val.shape
    )

    print(
        "y_val shape       :",
        y_val.shape
    )

    print(
        "Validation videos :",
        len(np.unique(video_ids))
    )

    print(
        "Real sequences    :",
        np.sum(y_val == 0)
    )

    print(
        "Fake sequences    :",
        np.sum(y_val == 1)
    )

    # ==================================================
    # Scale using training scaler
    # ==================================================

    print("\nLoading training scaler...")

    X_val = scale_validation_features(
        X_val
    )

    print(
        "Validation features scaled using training statistics."
    )

    # ==================================================
    # Load model
    # ==================================================

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # ==================================================
    # Sequence predictions
    # ==================================================

    print("\nGenerating validation predictions...")

    sequence_probabilities = (
        model.predict(
            X_val,
            verbose=1
        ).reshape(-1)
    )

    sequence_probabilities = np.clip(
        sequence_probabilities,
        0.0,
        1.0
    )

    # ==================================================
    # VIDEO-LEVEL AGGREGATION
    # ==================================================

    (
        probabilities,
        labels
    ) = aggregate_by_video(
        sequence_probabilities,
        y_val,
        video_ids
    )

    print("\n==============================")
    print("Video-Level Validation Data")
    print("==============================")

    print(
        "Videos        :",
        len(labels)
    )

    print(
        "Real videos   :",
        np.sum(labels == 0)
    )

    print(
        "Fake videos   :",
        np.sum(labels == 1)
    )

    # ==================================================
    # Threshold analysis
    # ==================================================

    print("\n==============================")
    print("Video-Level Threshold Results")
    print("==============================")

    print(
        "\nThreshold | Accuracy | Precision | Recall | F1"
    )

    print(
        "-------------------------------------------------------"
    )

    results = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
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

        results.append(
            (
                threshold,
                accuracy,
                precision,
                recall,
                f1
            )
        )

        print(
            f"{threshold:9.2f} | "
            f"{accuracy:8.4f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

    # ==================================================
    # Best threshold
    # ==================================================

    best_result = max(
        results,
        key=lambda x: x[4]
    )

    (
        best_threshold,
        best_accuracy,
        best_precision,
        best_recall,
        best_f1
    ) = best_result

    print("\n==============================")
    print("BEST VIDEO-LEVEL THRESHOLD")
    print("==============================")

    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"Accuracy  : {best_accuracy:.4f}"
    )

    print(
        f"Precision : {best_precision:.4f}"
    )

    print(
        f"Recall    : {best_recall:.4f}"
    )

    print(
        f"F1 Score  : {best_f1:.4f}"
    )

    # ==================================================
    # Save threshold
    # ==================================================

    THRESHOLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        THRESHOLD_PATH,
        "w"
    ) as f:

        json.dump(
            {
                "threshold": float(
                    best_threshold
                ),
                "selection_metric": "video_level_f1",
                "validation_videos": int(
                    len(labels)
                )
            },
            f,
            indent=2
        )

    print(
        f"\nThreshold saved to: {THRESHOLD_PATH}"
    )

    print(
        "\n=============================="
    )

    print(
        "Validation Threshold Analysis Complete"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()
