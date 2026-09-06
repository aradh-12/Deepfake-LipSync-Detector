import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 173

MODEL_PATH = (
    "models/deepfake_lipsync_lstm_exp3.keras"
)

SCALER_PATH = (
    "models/deepfake_lipsync_feature_scaler_exp3.npz"
)

THRESHOLD_PATH = (
    "models/threshold.json"
)

DATASET_DIR = (
    "outputs/dataset"
)

OUTPUT_DIR = (
    "outputs/evaluation"
)


# ============================================================
# Utility
# ============================================================

def print_header(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# Load saved dataset split
# ============================================================

def load_saved_split():

    print_header(
        "Loading Saved Video-Level Dataset Split"
    )

    train_path = os.path.join(
        DATASET_DIR,
        "train.npz"
    )

    validation_path = os.path.join(
        DATASET_DIR,
        "validation.npz"
    )

    test_path = os.path.join(
        DATASET_DIR,
        "test.npz"
    )

    for path in (
        train_path,
        validation_path,
        test_path
    ):

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required dataset file not found: {path}"
            )

    train_data = np.load(
        train_path,
        allow_pickle=True
    )

    validation_data = np.load(
        validation_path,
        allow_pickle=True
    )

    test_data = np.load(
        test_path,
        allow_pickle=True
    )

    X_train = train_data["X"]
    y_train = train_data["y"]
    train_videos = train_data["video_ids"]

    X_val = validation_data["X"]
    y_val = validation_data["y"]
    val_videos = validation_data["video_ids"]

    X_test = test_data["X"]
    y_test = test_data["y"]
    test_videos = test_data["video_ids"]

    print(
        "Training shape   :",
        X_train.shape
    )

    print(
        "Validation shape :",
        X_val.shape
    )

    print(
        "Test shape       :",
        X_test.shape
    )

    print()

    print(
        "Training videos   :",
        len(np.unique(train_videos))
    )

    print(
        "Validation videos :",
        len(np.unique(val_videos))
    )

    print(
        "Test videos       :",
        len(np.unique(test_videos))
    )

    return (
        X_train,
        y_train,
        train_videos,

        X_val,
        y_val,
        val_videos,

        X_test,
        y_test,
        test_videos
    )


# ============================================================
# Validate split
# ============================================================

def validate_split(
    X,
    y,
    video_ids,
    split_name
):

    print_header(
        f"Validating {split_name} Split"
    )

    if X.ndim != 3:

        raise ValueError(
            f"{split_name}: X must be 3D. "
            f"Got {X.shape}"
        )

    if X.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(
            f"{split_name}: expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != FEATURE_SIZE:

        raise ValueError(
            f"{split_name}: expected feature size "
            f"{FEATURE_SIZE}, "
            f"got {X.shape[2]}"
        )

    if len(X) != len(y):

        raise ValueError(
            f"{split_name}: X and y length mismatch."
        )

    if len(X) != len(video_ids):

        raise ValueError(
            f"{split_name}: X and video_ids length mismatch."
        )

    if np.isnan(X).any():

        raise ValueError(
            f"{split_name}: NaN values detected."
        )

    if np.isinf(X).any():

        raise ValueError(
            f"{split_name}: infinite values detected."
        )

    print(
        "Sequences :",
        len(X)
    )

    print(
        "Videos    :",
        len(np.unique(video_ids))
    )

    print(
        "Real      :",
        int(np.sum(y == 0))
    )

    print(
        "Fake      :",
        int(np.sum(y == 1))
    )

    print(
        "Shape     :",
        X.shape
    )

    print(
        "Validation: PASSED"
    )


# ============================================================
# Verify no video leakage
# ============================================================

def verify_no_video_leakage(
    train_videos,
    val_videos,
    test_videos
):

    print_header(
        "Video-Level Leakage Check"
    )

    train_set = set(
        train_videos
    )

    val_set = set(
        val_videos
    )

    test_set = set(
        test_videos
    )

    train_val = (
        train_set & val_set
    )

    train_test = (
        train_set & test_set
    )

    val_test = (
        val_set & test_set
    )

    print(
        "Train ∩ Validation :",
        len(train_val)
    )

    print(
        "Train ∩ Test       :",
        len(train_test)
    )

    print(
        "Validation ∩ Test  :",
        len(val_test)
    )

    if train_val:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "Training and validation videos overlap."
        )

    if train_test:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "Training and test videos overlap."
        )

    if val_test:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "Validation and test videos overlap."
        )

    print(
        "Data leakage check : PASSED"
    )


# ============================================================
# Load trained scaler
# ============================================================

def load_scaler():

    print_header(
        "Loading Training Feature Scaler"
    )

    if not os.path.exists(
        SCALER_PATH
    ):

        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
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
            f"does not match feature size "
            f"{FEATURE_SIZE}."
        )

    if scale.shape != (
        FEATURE_SIZE,
    ):

        raise ValueError(
            f"Scaler scale shape {scale.shape} "
            f"does not match feature size "
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

    return (
        mean,
        safe_scale
    )


# ============================================================
# Apply scaler
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

    X_flat = (
        X_flat - mean
    ) / scale

    return X_flat.reshape(
        original_shape
    ).astype(
        np.float32
    )


# ============================================================
# Load threshold selected on validation data
# ============================================================

def load_threshold():

    print_header(
        "Loading Validation-Selected Threshold"
    )

    if not os.path.exists(
        THRESHOLD_PATH
    ):

        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as f:

        data = json.load(f)

    threshold = float(
        data["threshold"]
    )

    print(
        "Selected threshold:",
        threshold
    )

    print(
        "Selection metric:",
        data.get(
            "selection_metric",
            "unknown"
        )
    )

    print(
        "Validation videos used:",
        data.get(
            "validation_videos",
            "unknown"
        )
    )

    if not (
        0.0 < threshold < 1.0
    ):

        raise ValueError(
            f"Invalid threshold: {threshold}"
        )

    return threshold


# ============================================================
# Aggregate sequence predictions to video level
# ============================================================

def aggregate_video_predictions(
    labels,
    probabilities,
    video_ids
):

    unique_videos = np.unique(
        video_ids
    )

    video_names = []
    video_labels = []
    video_scores = []

    for video in unique_videos:

        mask = (
            video_ids == video
        )

        video_probs = (
            probabilities[mask]
        )

        video_y = (
            labels[mask]
        )

        if len(
            np.unique(video_y)
        ) != 1:

            raise ValueError(
                f"Inconsistent labels for video: "
                f"{video}"
            )

        # Same aggregation used by
        # validation_threshold.py
        video_score = float(
            np.mean(video_probs)
        )

        video_names.append(
            video
        )

        video_labels.append(
            int(video_y[0])
        )

        video_scores.append(
            video_score
        )

    return (
        np.asarray(video_names),
        np.asarray(video_labels),
        np.asarray(video_scores)
    )


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(
        int
    )

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

    try:

        auc = roc_auc_score(
            labels,
            probabilities
        )

    except ValueError:

        auc = None

    metrics = {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": (
            None
            if auc is None
            else float(auc)
        )
    }

    return (
        metrics,
        predictions
    )


# ============================================================
# Print metrics
# ============================================================

def print_metrics(
    metrics
):

    for name, value in metrics.items():

        if value is None:

            print(
                f"{name:12s}: N/A"
            )

        else:

            print(
                f"{name:12s}: "
                f"{value:.4f}"
            )


# ============================================================
# Save evaluation results
# ============================================================

def save_results(
    sequence_metrics,
    video_metrics,
    threshold,
    video_names,
    video_labels,
    video_scores
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = {

        "threshold": float(
            threshold
        ),

        "sequence_level": (
            sequence_metrics
        ),

        "video_level": (
            video_metrics
        ),

        "test_videos": int(
            len(video_names)
        )
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "final_test_results.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save video-level predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({

        "video_id": video_names,

        "true_label": video_labels,

        "probability": video_scores,

        "prediction": (
            video_scores >= threshold
        ).astype(int)
    })

    csv_path = os.path.join(
        OUTPUT_DIR,
        "video_level_test_predictions.csv"
    )

    prediction_df.to_csv(
        csv_path,
        index=False
    )

    print()
    print(
        "Results saved:"
    )

    print(
        json_path
    )

    print(
        csv_path
    )


# ============================================================
# Main
# ============================================================

def main():

    print_header(
        "Multimodal Deepfake Lip-Sync Detector"
    )

    print(
        "FINAL TEST EVALUATION"
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
    # Load trained model
    # ========================================================

    print_header(
        "Loading Trained Model"
    )

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
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

    expected_shape = (
        None,
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    )

    if model.input_shape != expected_shape:

        raise ValueError(
            f"Unexpected model input shape: "
            f"{model.input_shape}. "
            f"Expected {expected_shape}"
        )

    # ========================================================
    # STEP 2
    # Load SAVED split
    # ========================================================

    (
        X_train,
        y_train,
        train_videos,

        X_val,
        y_val,
        val_videos,

        X_test,
        y_test,
        test_videos

    ) = load_saved_split()

    # ========================================================
    # STEP 3
    # Validate data
    # ========================================================

    validate_split(
        X_train,
        y_train,
        train_videos,
        "TRAIN"
    )

    validate_split(
        X_val,
        y_val,
        val_videos,
        "VALIDATION"
    )

    validate_split(
        X_test,
        y_test,
        test_videos,
        "TEST"
    )

    verify_no_video_leakage(
        train_videos,
        val_videos,
        test_videos
    )

    # ========================================================
    # STEP 4
    # Load training scaler
    # ========================================================

    mean, scale = load_scaler()

    # ========================================================
    # STEP 5
    # Apply scaler
    # ========================================================

    print_header(
        "Applying Training Feature Normalization"
    )

    X_train = apply_scaler(
        X_train,
        mean,
        scale
    )

    X_val = apply_scaler(
        X_val,
        mean,
        scale
    )

    X_test = apply_scaler(
        X_test,
        mean,
        scale
    )

    print(
        "Training mean:",
        float(X_train.mean())
    )

    print(
        "Training std :",
        float(X_train.std())
    )

    print(
        "Validation mean:",
        float(X_val.mean())
    )

    print(
        "Validation std :",
        float(X_val.std())
    )

    print(
        "Test mean:",
        float(X_test.mean())
    )

    print(
        "Test std :",
        float(X_test.std())
    )

    # ========================================================
    # STEP 6
    # Load threshold
    # ========================================================

    threshold = load_threshold()

    # ========================================================
    # STEP 7
    # Validation prediction
    #
    # This is ONLY a sanity check.
    # Threshold was already selected previously.
    # ========================================================

    print_header(
        "Validation Sanity Check"
    )

    validation_sequence_probabilities = (
        model.predict(
            X_val,
            batch_size=16,
            verbose=1
        )
        .reshape(-1)
    )

    (
        validation_video_names,
        validation_video_labels,
        validation_video_scores

    ) = aggregate_video_predictions(
        y_val,
        validation_sequence_probabilities,
        val_videos
    )

    (
        validation_metrics,
        validation_predictions

    ) = calculate_metrics(
        validation_video_labels,
        validation_video_scores,
        threshold
    )

    print(
        "Validation videos:",
        len(validation_video_names)
    )

    print(
        "Validation metrics:"
    )

    print_metrics(
        validation_metrics
    )

    print()

    print(
        "Validation confusion matrix:"
    )

    print(
        confusion_matrix(
            validation_video_labels,
            validation_predictions,
            labels=[0, 1]
        )
    )

    # ========================================================
    # STEP 8
    # FINAL TEST PREDICTION
    #
    # IMPORTANT:
    # Test data is NEVER used to select threshold.
    # ========================================================

    print_header(
        "FINAL TEST PREDICTION"
    )

    test_sequence_probabilities = (
        model.predict(
            X_test,
            batch_size=16,
            verbose=1
        )
        .reshape(-1)
    )

    test_sequence_probabilities = np.clip(
        test_sequence_probabilities,
        0.0,
        1.0
    )

    print(
        "Test sequences predicted:",
        len(
            test_sequence_probabilities
        )
    )

    # ========================================================
    # STEP 9
    # Sequence-level test results
    # ========================================================

    print_header(
        "Sequence-Level Test Results"
    )

    (
        sequence_metrics,
        sequence_predictions

    ) = calculate_metrics(
        y_test,
        test_sequence_probabilities,
        threshold
    )

    print_metrics(
        sequence_metrics
    )

    print()

    print(
        "Sequence-Level Classification Report"
    )

    print(
        classification_report(
            y_test,
            sequence_predictions,
            labels=[0, 1],
            target_names=[
                "Real",
                "Fake"
            ],
            zero_division=0
        )
    )

    print(
        "Sequence-Level Confusion Matrix"
    )

    print(
        confusion_matrix(
            y_test,
            sequence_predictions,
            labels=[0, 1]
        )
    )

    # ========================================================
    # STEP 10
    # Video-level test aggregation
    # ========================================================

    print_header(
        "Video-Level Test Aggregation"
    )

    (
        video_names,
        video_labels,
        video_scores

    ) = aggregate_video_predictions(
        y_test,
        test_sequence_probabilities,
        test_videos
    )

    print(
        "Unique test videos:",
        len(video_names)
    )

    print(
        "Real test videos:",
        int(
            np.sum(video_labels == 0)
        )
    )

    print(
        "Fake test videos:",
        int(
            np.sum(video_labels == 1)
        )
    )

    # ========================================================
    # STEP 11
    # FINAL VIDEO-LEVEL TEST RESULTS
    #
    # This is the MAIN result of the detector.
    # ========================================================

    print_header(
        "FINAL VIDEO-LEVEL TEST RESULTS"
    )

    (
        video_metrics,
        video_predictions

    ) = calculate_metrics(
        video_labels,
        video_scores,
        threshold
    )

    print_metrics(
        video_metrics
    )

    print()

    print(
        "Video-Level Classification Report"
    )

    print(
        classification_report(
            video_labels,
            video_predictions,
            labels=[0, 1],
            target_names=[
                "Real",
                "Fake"
            ],
            zero_division=0
        )
    )

    print(
        "Video-Level Confusion Matrix"
    )

    print(
        confusion_matrix(
            video_labels,
            video_predictions,
            labels=[0, 1]
        )
    )

    # ========================================================
    # STEP 12
    # Save results
    # ========================================================

    print_header(
        "Saving Final Evaluation Results"
    )

    save_results(
        sequence_metrics,
        video_metrics,
        threshold,
        video_names,
        video_labels,
        video_scores
    )

    # ========================================================
    # Final summary
    # ========================================================

    print_header(
        "FINAL SUMMARY"
    )

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Test videos : {len(video_names)}"
    )

    print(
        f"Video Accuracy : "
        f"{video_metrics['accuracy']:.4f}"
    )

    print(
        f"Video Precision: "
        f"{video_metrics['precision']:.4f}"
    )

    print(
        f"Video Recall   : "
        f"{video_metrics['recall']:.4f}"
    )

    print(
        f"Video F1       : "
        f"{video_metrics['f1']:.4f}"
    )

    if video_metrics["roc_auc"] is not None:

        print(
            f"Video ROC-AUC  : "
            f"{video_metrics['roc_auc']:.4f}"
        )

    print()
    print(
        "FINAL TEST EVALUATION COMPLETE"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()