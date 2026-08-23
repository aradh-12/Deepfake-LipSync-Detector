import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from training.dataset_builder import build_dataset


MODEL_PATH = "models/deepfake_lipsync_lstm.keras"

RANDOM_STATE = 42


def main():

    print("\nLoading dataset...\n")

    X, y, video_ids = build_dataset()

    # ==================================================
    # Recreate the SAME train / validation / test split
    # ==================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    train_indices, temp_indices = next(
        splitter.split(
            X,
            y,
            groups=video_ids
        )
    )

    X_temp = X[temp_indices]
    y_temp = y[temp_indices]
    temp_videos = video_ids[temp_indices]

    splitter_val_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    val_indices, test_indices = next(
        splitter_val_test.split(
            X_temp,
            y_temp,
            groups=temp_videos
        )
    )

    X_val = X_temp[val_indices]
    y_val = y_temp[val_indices]

    print("\n==============================")
    print("Validation Dataset")
    print("==============================")

    print("X_val shape :", X_val.shape)
    print("y_val shape :", y_val.shape)

    print("Real :", np.sum(y_val == 0))
    print("Fake :", np.sum(y_val == 1))

    # ==================================================
    # Load model
    # ==================================================

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # ==================================================
    # Predictions
    # ==================================================

    probabilities = (
        model.predict(
            X_val,
            verbose=1
        ).ravel()
    )

    # ==================================================
    # Threshold analysis
    # ==================================================

    print("\n==============================")
    print("Validation Threshold Analysis")
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
            y_val,
            predictions
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
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
    # Best threshold based on validation F1
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
    print("Best Validation Threshold")
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

    print("\n==============================")
    print("Validation Threshold Analysis Complete")
    print("==============================")


if __name__ == "__main__":
    main()
