import json
from pathlib import Path

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

from utils.multidataset_manager import (
    get_fakeavceleb_unseen
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/deepfake_lipsync_lstm_exp3.keras"
)

SCALER_PATH = Path(
    "models/deepfake_lipsync_feature_scaler_exp3.npz"
)

THRESHOLD_PATH = Path(
    "models/threshold.json"
)

OUTPUT_DIR = Path(
    "outputs/evaluation"
)


# ============================================================
# Headers
# ============================================================

def print_header(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# Load threshold
# ============================================================

def load_threshold():

    print_header(
        "Loading Validation-Selected Threshold"
    )

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as file:

        data = json.load(file)

    threshold = float(
        data["threshold"]
    )

    if not 0.0 < threshold < 1.0:

        raise ValueError(
            f"Invalid threshold: {threshold}"
        )

    print(
        f"Threshold : {threshold:.4f}"
    )

    print(
        "Selection metric:",
        data.get(
            "selection_metric",
            "unknown"
        )
    )

    print(
        "Validation videos:",
        data.get(
            "validation_videos",
            "unknown"
        )
    )

    return threshold


# ============================================================
# Load training scaler
# ============================================================

def load_scaler():

    print_header(
        "Loading Training Feature Scaler"
    )

    if not SCALER_PATH.exists():

        raise FileNotFoundError(
            f"Scaler not found: "
            f"{SCALER_PATH}"
        )

    data = np.load(
        SCALER_PATH
    )

    mean = data["mean"].astype(
        np.float32
    )

    scale = data["scale"].astype(
        np.float32
    )

    if mean.shape != (
        FEATURE_SIZE,
    ):

        raise ValueError(
            f"Scaler mean shape {mean.shape} "
            f"does not match "
            f"{FEATURE_SIZE}."
        )

    if scale.shape != (
        FEATURE_SIZE,
    ):

        raise ValueError(
            f"Scaler scale shape {scale.shape} "
            f"does not match "
            f"{FEATURE_SIZE}."
        )

    safe_scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale
    )

    print(
        "Scaler loaded successfully."
    )

    print(
        "Feature count:",
        len(mean)
    )

    return mean, safe_scale


# ============================================================
# Apply training scaler
# ============================================================

def apply_scaler(
    X,
    mean,
    scale
):

    original_shape = X.shape

    X_flat = X.reshape(
        -1,
        FEATURE_SIZE
    )

    X_scaled = (
        X_flat - mean
    ) / scale

    return X_scaled.reshape(
        original_shape
    ).astype(
        np.float32
    )


# ============================================================
# Load synchronized video sequences
# ============================================================

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

    feature_files = sorted(
        video_folder.glob("*.npy")
    )

    if len(feature_files) < SEQUENCE_LENGTH:

        print(
            f"⚠️ Too few features: "
            f"{video_folder_name} "
            f"({len(feature_files)})"
        )

        return None

    features = []

    for feature_file in feature_files:

        try:

            feature = np.load(
                feature_file
            )

        except Exception as error:

            print(
                f"⚠️ Could not load "
                f"{feature_file.name}: "
                f"{error}"
            )

            continue

        if feature.shape != (
            FEATURE_SIZE,
        ):

            print(
                f"⚠️ Wrong feature shape: "
                f"{feature_file.name} "
                f"{feature.shape}"
            )

            continue

        if not np.isfinite(
            feature
        ).all():

            print(
                f"⚠️ NaN/Inf detected: "
                f"{feature_file.name}"
            )

            continue

        features.append(
            feature.astype(
                np.float32
            )
        )

    if len(features) < SEQUENCE_LENGTH:

        print(
            f"⚠️ Not enough valid features: "
            f"{video_folder_name}"
        )

        return None

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Non-overlapping 30-frame sequences.
    #
    # This matches training/dataset_builder.py.
    # --------------------------------------------------------

    sequence_count = (
        len(features)
        //
        SEQUENCE_LENGTH
    )

    usable_length = (
        sequence_count
        *
        SEQUENCE_LENGTH
    )

    features = features[
        :usable_length
    ]

    sequences = []

    for start in range(
        0,
        usable_length,
        SEQUENCE_LENGTH
    ):

        sequence = features[
            start:
            start + SEQUENCE_LENGTH
        ]

        if sequence.shape == (
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        ):

            sequences.append(
                sequence
            )

    if not sequences:

        return None

    return np.asarray(
        sequences,
        dtype=np.float32
    )


# ============================================================
# Build reproducible unseen dataset
# ============================================================

def build_unseen_dataset():

    print_header(
        "Building Reproducible FakeAVCeleb Unseen Set"
    )

    dataset = get_fakeavceleb_unseen(
        real_count=25,
        fake_count=25
    )

    print(
        "Unseen videos selected:",
        len(dataset)
    )

    real_count = sum(
        label == 0
        for _, label in dataset
    )

    fake_count = sum(
        label == 1
        for _, label in dataset
    )

    print(
        "Unseen Real:",
        real_count
    )

    print(
        "Unseen Fake:",
        fake_count
    )

    return dataset


# ============================================================
# Main
# ============================================================

def main():

    print_header(
        "MULTIMODAL DEEPFAKE LIP-SYNC DETECTOR"
    )

    print(
        "UNSEEN GENERALIZATION EVALUATION"
    )

    print(
        "Feature size    :",
        FEATURE_SIZE
    )

    print(
        "Sequence length :",
        SEQUENCE_LENGTH
    )

    # ========================================================
    # STEP 1
    # Validate required files
    # ========================================================

    print_header(
        "Checking Evaluation Assets"
    )

    for path in (
        MODEL_PATH,
        SCALER_PATH,
        THRESHOLD_PATH
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: "
                f"{path}"
            )

        print(
            "✓",
            path
        )

    # ========================================================
    # STEP 2
    # Load model
    # ========================================================

    print_header(
        "Loading Exp3 Model"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Model input shape :",
        model.input_shape
    )

    print(
        "Model output shape:",
        model.output_shape
    )

    if model.input_shape[1:] != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):

        raise ValueError(
            "Model input shape does not match "
            "expected sequence configuration."
        )

    # ========================================================
    # STEP 3
    # Load training scaler
    # ========================================================

    mean, scale = load_scaler()

    # ========================================================
    # STEP 4
    # Load validation threshold
    # ========================================================

    threshold = load_threshold()

    # ========================================================
    # STEP 5
    # Build unseen set
    # ========================================================

    dataset = build_unseen_dataset()

    if not dataset:

        print(
            "\n❌ No unseen videos found."
        )

        return

    # ========================================================
    # STEP 6
    # Evaluate
    # ========================================================

    print_header(
        "Testing Unseen Videos"
    )

    all_labels = []
    all_probabilities = []
    all_predictions = []
    evaluated_videos = []

    skipped_videos = []

    for video, label in dataset:

        sequences = load_video_sequences(
            video
        )

        if sequences is None:

            skipped_videos.append(
                str(video)
            )

            continue

        # ----------------------------------------------------
        # Apply training-only normalization.
        # ----------------------------------------------------

        sequences_scaled = apply_scaler(
            sequences,
            mean,
            scale
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        probabilities = (
            model.predict(
                sequences_scaled,
                batch_size=16,
                verbose=0
            )
            .reshape(-1)
        )

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Video-level aggregation.
        #
        # Same strategy used by final test evaluation.
        # ----------------------------------------------------

        video_probability = float(
            np.mean(probabilities)
        )

        video_prediction = int(
            video_probability >= threshold
        )

        all_labels.append(
            label
        )

        all_probabilities.append(
            video_probability
        )

        all_predictions.append(
            video_prediction
        )

        evaluated_videos.append(
            str(video)
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
            f" | Score: {video_probability:.4f}"
        )

    # ========================================================
    # Check results
    # ========================================================

    if not all_labels:

        print(
            "\n❌ No unseen videos could be evaluated."
        )

        return

    y_true = np.asarray(
        all_labels,
        dtype=np.int32
    )

    y_prob = np.asarray(
        all_probabilities,
        dtype=np.float32
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=np.int32
    )

    # ========================================================
    # STEP 7
    # Metrics
    # ========================================================

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

        auc = None

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # ========================================================
    # STEP 8
    # Results
    # ========================================================

    print_header(
        "UNSEEN VIDEO RESULTS"
    )

    print(
        "Videos Evaluated :",
        len(y_true)
    )

    print(
        "Videos Skipped   :",
        len(skipped_videos)
    )

    print(
        "Real Videos      :",
        int(np.sum(y_true == 0))
    )

    print(
        "Fake Videos      :",
        int(np.sum(y_true == 1))
    )

    print()

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

    if auc is not None:

        print(
            f"ROC-AUC          : "
            f"{auc:.4f}"
        )

    else:

        print(
            "ROC-AUC          : N/A"
        )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print_header(
        "Confusion Matrix"
    )

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

    # ========================================================
    # Classification report
    # ========================================================

    print_header(
        "Classification Report"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Real",
                "Fake"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # Save results
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "model": str(MODEL_PATH),
        "scaler": str(SCALER_PATH),
        "threshold": threshold,
        "selection_metric": "video_level_f1",
        "evaluated_videos": len(y_true),
        "skipped_videos": len(skipped_videos),
        "real_videos": int(np.sum(y_true == 0)),
        "fake_videos": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (
            float(auc)
            if auc is not None
            else None
        ),
        "confusion_matrix": cm.tolist(),
        "evaluated_video_paths": evaluated_videos,
        "skipped_video_paths": skipped_videos
    }

    results_path = (
        OUTPUT_DIR /
        "unseen_generalization_results.json"
    )

    with open(
        results_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print_header(
        "GENERALIZATION TEST COMPLETE"
    )

    print(
        "Results saved to:",
        results_path
    )

    print(
        "=" * 60
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()