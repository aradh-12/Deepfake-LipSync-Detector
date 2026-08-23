import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


MODEL_PATH = "models/deepfake_lipsync_lstm.keras"
TEST_DATASET = "outputs/dataset/test.npz"


def main():

    print("\n==============================")
    print("Threshold Analysis")
    print("==============================")

    # --------------------------------------------------
    # Load EXACT saved test set
    # --------------------------------------------------

    print("\nLoading saved test dataset...")

    data = np.load(
        TEST_DATASET,
        allow_pickle=True
    )

    X_test = data["X"]
    y_test = data["y"]
    video_ids = data["video_ids"]

    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)
    print("Test videos  :", len(np.unique(video_ids)))

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")
    print("Model input  :", model.input_shape)
    print("Model output :", model.output_shape)

    # --------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    ).ravel()

    # --------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------

    print("\n==============================")
    print("Threshold Results")
    print("==============================")

    print(
        "\nThreshold | Accuracy | Precision | Recall | F1"
    )

    print("-" * 55)

    best_threshold = None
    best_f1 = -1.0

    for threshold in np.arange(
        0.10,
        0.91,
        0.05
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        print(
            f"{threshold:9.2f} | "
            f"{accuracy:8.4f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n==============================")
    print("BEST THRESHOLD")
    print("==============================")

    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"Best F1   : {best_f1:.4f}"
    )

    print("==============================")


if __name__ == "__main__":
    main()