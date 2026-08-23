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

from training.dataset_builder import (
    SEQUENCE_FOLDER,
    SEQUENCE_LENGTH,
    FEATURE_SIZE,
    get_video_folder_name
)

from utils.multidataset_manager import FA_ROOT


MODEL_PATH = "models/deepfake_lipsync_lstm.keras"

THRESHOLD = 0.35

DEVELOPMENT_VIDEOS = 25
UNSEEN_VIDEOS = 25


def get_all_fakeavceleb_videos():

    all_videos = sorted(
        FA_ROOT.rglob("*.mp4")
    )

    real_videos = [
        video
        for video in all_videos
        if "RealVideo-RealAudio" in str(video)
    ]

    fake_videos = [
        video
        for video in all_videos
        if "RealVideo-RealAudio" not in str(video)
    ]

    return real_videos, fake_videos


def build_unseen_dataset():

    real_videos, fake_videos = (
        get_all_fakeavceleb_videos()
    )

    # --------------------------------------------------
    # DEVELOPMENT SET
    # First 25 Real + First 25 Fake
    # --------------------------------------------------

    development_real = real_videos[
        :DEVELOPMENT_VIDEOS
    ]

    development_fake = fake_videos[
        :DEVELOPMENT_VIDEOS
    ]

    # --------------------------------------------------
    # UNSEEN SET
    # Next 25 Real + Next 25 Fake
    # --------------------------------------------------

    unseen_real = real_videos[
        DEVELOPMENT_VIDEOS:
        DEVELOPMENT_VIDEOS + UNSEEN_VIDEOS
    ]

    unseen_fake = fake_videos[
        DEVELOPMENT_VIDEOS:
        DEVELOPMENT_VIDEOS + UNSEEN_VIDEOS
    ]

    unseen_dataset = (
        [(video, 0) for video in unseen_real]
        +
        [(video, 1) for video in unseen_fake]
    )

    print("\n==============================")
    print("Unseen Video Dataset")
    print("==============================")

    print(
        "Development Real :",
        len(development_real)
    )

    print(
        "Development Fake :",
        len(development_fake)
    )

    print(
        "Unseen Real      :",
        len(unseen_real)
    )

    print(
        "Unseen Fake      :",
        len(unseen_fake)
    )

    print(
        "Total Unseen     :",
        len(unseen_dataset)
    )

    return unseen_dataset


def load_video_sequences(video):

    video_folder_name = (
        get_video_folder_name(video)
    )

    video_folder = (
        SEQUENCE_FOLDER /
        video_folder_name
    )

    if not video_folder.exists():

        print(
            f"⚠️ Missing synchronized folder: "
            f"{video_folder_name}"
        )

        return None

    frame_files = sorted(
        video_folder.glob("*.npy")
    )

    if len(frame_files) < SEQUENCE_LENGTH:

        print(
            f"⚠️ Too few frames: "
            f"{video_folder_name}"
        )

        return None

    features = []

    for frame_file in frame_files:

        feature = np.load(frame_file)

        if feature.shape != (
            FEATURE_SIZE,
        ):

            print(
                f"⚠️ Wrong feature shape: "
                f"{frame_file.name} "
                f"{feature.shape}"
            )

            continue

        features.append(feature)

    if len(features) < SEQUENCE_LENGTH:

        return None

    features = np.asarray(
        features,
        dtype=np.float32
    )

    sequences = []

    for start in range(
        0,
        len(features) - SEQUENCE_LENGTH + 1,
        SEQUENCE_LENGTH
    ):

        sequence = features[
            start:start + SEQUENCE_LENGTH
        ]

        sequences.append(sequence)

    if not sequences:

        return None

    return np.asarray(
        sequences,
        dtype=np.float32
    )


def main():

    dataset = build_unseen_dataset()

    if not dataset:

        print(
            "\n❌ No unseen videos found."
        )

        return

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    all_predictions = []
    all_probabilities = []
    all_labels = []

    print("\n==============================")
    print("Testing Unseen Videos")
    print("==============================")

    for video, label in dataset:

        sequences = load_video_sequences(
            video
        )

        if sequences is None:

            continue

        probabilities = (
            model.predict(
                sequences,
                verbose=0
            ).ravel()
        )

        # Average all sequence predictions
        # to obtain one video-level score.

        video_probability = float(
            np.mean(probabilities)
        )

        video_prediction = int(
            video_probability >= THRESHOLD
        )

        all_probabilities.append(
            video_probability
        )

        all_predictions.append(
            video_prediction
        )

        all_labels.append(
            label
        )

        actual = (
            "Fake"
            if label == 1
            else "Real"
        )

        predicted = (
            "Fake"
            if video_prediction == 1
            else "Real"
        )

        print(
            f"{video.name}"
            f" | Actual: {actual}"
            f" | Predicted: {predicted}"
            f" | Score: "
            f"{video_probability:.4f}"
        )

    if not all_labels:

        print(
            "\n❌ No videos could be evaluated."
        )

        return

    y_true = np.asarray(
        all_labels,
        dtype=np.int32
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=np.int32
    )

    y_prob = np.asarray(
        all_probabilities,
        dtype=np.float32
    )

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

        auc = roc_auc_score(
            y_true,
            y_prob
        )

    except ValueError:

        auc = 0.0

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n==============================")
    print("UNSEEN VIDEO RESULTS")
    print("==============================")

    print(
        "Videos Evaluated :",
        len(y_true)
    )

    print(
        "Real Videos      :",
        np.sum(y_true == 0)
    )

    print(
        "Fake Videos      :",
        np.sum(y_true == 1)
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision        : "
        f"{precision:.4f}"
    )

    print(
        f"Recall           : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score         : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC          : "
        f"{auc:.4f}"
    )

    print("\n==============================")
    print("Confusion Matrix")
    print("==============================")

    print(
        "              Predicted"
    )

    print(
        "              Real  Fake"
    )

    print(
        f"Actual Real   "
        f"{cm[0][0]:4d}  "
        f"{cm[0][1]:4d}"
    )

    print(
        f"Actual Fake   "
        f"{cm[1][0]:4d}  "
        f"{cm[1][1]:4d}"
    )

    print("\n==============================")
    print("Classification Report")
    print("==============================")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Real",
                "Fake"
            ],
            zero_division=0
        )
    )

    print(
        "=============================="
    )

    print(
        "Generalization Test Complete"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()