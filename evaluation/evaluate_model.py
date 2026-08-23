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

from training.dataset_builder import build_dataset
from training.train import create_group_split


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 173

RANDOM_STATE = 42

MODEL_PATH = (
    "models/deepfake_lipsync_lstm.keras"
)

SCALER_PATH = (
    "models/deepfake_lipsync_feature_scaler.npz"
)

OUTPUT_DIR = (
    "outputs/evaluation"
)

DEFAULT_THRESHOLD = 0.50


# ------------------------------------------------------------
# Threshold candidates
#
# IMPORTANT:
# These thresholds are evaluated ONLY on validation data.
# The test set is never used for threshold selection.
# ------------------------------------------------------------

THRESHOLD_VALUES = np.arange(
    0.30,
    0.71,
    0.01
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
# Load saved scaler
# ============================================================

def load_scaler():

    if not os.path.exists(
        SCALER_PATH
    ):

        raise FileNotFoundError(
            f"Scaler not found: "
            f"{SCALER_PATH}"
        )

    scaler_data = np.load(
        SCALER_PATH
    )

    mean = scaler_data[
        "mean"
    ]

    scale = scaler_data[
        "scale"
    ]

    if mean.shape != (
        FEATURE_SIZE,
    ):

        raise ValueError(
            f"Scaler mean has wrong shape: "
            f"{mean.shape}. "
            f"Expected {(FEATURE_SIZE,)}"
        )

    if scale.shape != (
        FEATURE_SIZE,
    ):

        raise ValueError(
            f"Scaler scale has wrong shape: "
            f"{scale.shape}. "
            f"Expected {(FEATURE_SIZE,)}"
        )

    print(
        "Scaler loaded successfully."
    )

    print(
        "Scaler feature count:",
        len(mean)
    )

    return (
        mean,
        scale
    )


# ============================================================
# Apply saved training scaler
# ============================================================

def apply_scaler(
    X,
    mean,
    scale
):

    original_shape = X.shape

    if original_shape[-1] != (
        FEATURE_SIZE
    ):

        raise ValueError(
            f"Expected last dimension "
            f"{FEATURE_SIZE}, "
            f"got {original_shape[-1]}"
        )

    X_flat = X.reshape(
        -1,
        FEATURE_SIZE
    )

    X_scaled = (
        X_flat - mean
    ) / scale

    X_scaled = X_scaled.reshape(
        original_shape
    )

    return X_scaled.astype(
        np.float32
    )


# ============================================================
# Recreate EXACT training / validation / test split
# ============================================================

def create_train_val_test_split(
    X,
    y,
    video_ids
):

    print_header(
        "Recreating Exact Training Split"
    )

    # ========================================================
    # STEP 1
    # 70% training
    # 30% temporary
    # ========================================================

    train_indices, temp_indices = (
        create_group_split(
            X,
            y,
            video_ids,
            test_size=0.30,
            random_state_start=RANDOM_STATE
        )
    )

    X_train = X[
        train_indices
    ]

    y_train = y[
        train_indices
    ]

    train_videos = video_ids[
        train_indices
    ]

    X_temp = X[
        temp_indices
    ]

    y_temp = y[
        temp_indices
    ]

    temp_videos = video_ids[
        temp_indices
    ]

    # ========================================================
    # STEP 2
    #
    # Temporary -> validation + test
    #
    # 50% temporary = 15% total
    # ========================================================

    val_indices, test_indices = (
        create_group_split(
            X_temp,
            y_temp,
            temp_videos,
            test_size=0.50,
            random_state_start=(
                RANDOM_STATE + 100
            )
        )
    )

    X_val = X_temp[
        val_indices
    ]

    y_val = y_temp[
        val_indices
    ]

    val_videos = temp_videos[
        val_indices
    ]

    X_test = X_temp[
        test_indices
    ]

    y_test = y_temp[
        test_indices
    ]

    test_videos = temp_videos[
        test_indices
    ]

    # ========================================================
    # Verify NO video leakage
    # ========================================================

    train_video_set = set(
        train_videos
    )

    val_video_set = set(
        val_videos
    )

    test_video_set = set(
        test_videos
    )

    if (
        train_video_set
        & val_video_set
    ):

        raise RuntimeError(
            "Data leakage detected between "
            "training and validation videos."
        )

    if (
        train_video_set
        & test_video_set
    ):

        raise RuntimeError(
            "Data leakage detected between "
            "training and test videos."
        )

    if (
        val_video_set
        & test_video_set
    ):

        raise RuntimeError(
            "Data leakage detected between "
            "validation and test videos."
        )

    # ========================================================
    # Print split information
    # ========================================================

    print()
    print(
        "=============================="
    )

    print(
        "Evaluation Split"
    )

    print(
        "=============================="
    )

    print(
        "Training videos    :",
        len(train_video_set)
    )

    print(
        "Validation videos  :",
        len(val_video_set)
    )

    print(
        "Test videos        :",
        len(test_video_set)
    )

    print()

    print(
        "Training sequences :",
        len(X_train)
    )

    print(
        "Validation sequences:",
        len(X_val)
    )

    print(
        "Test sequences     :",
        len(X_test)
    )

    print()

    print(
        "Training Real      :",
        int(
            np.sum(
                y_train == 0
            )
        )
    )

    print(
        "Training Fake      :",
        int(
            np.sum(
                y_train == 1
            )
        )
    )

    print(
        "Validation Real    :",
        int(
            np.sum(
                y_val == 0
            )
        )
    )

    print(
        "Validation Fake    :",
        int(
            np.sum(
                y_val == 1
            )
        )
    )

    print(
        "Test Real          :",
        int(
            np.sum(
                y_test == 0
            )
        )
    )

    print(
        "Test Fake          :",
        int(
            np.sum(
                y_test == 1
            )
        )
    )

    print(
        "=============================="
    )

    # ========================================================
    # Leakage check
    # ========================================================

    print()
    print(
        "Leakage Check"
    )

    print(
        "Train ∩ Validation:",
        len(
            train_video_set
            & val_video_set
        )
    )

    print(
        "Train ∩ Test      :",
        len(
            train_video_set
            & test_video_set
        )
    )

    print(
        "Validation ∩ Test :",
        len(
            val_video_set
            & test_video_set
        )
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
# Validate dataset
# ============================================================

def validate_dataset(
    X,
    y,
    video_ids
):

    if X.ndim != 3:

        raise ValueError(
            f"Expected X to be 3D, "
            f"got {X.shape}"
        )

    if X.shape[1] != (
        SEQUENCE_LENGTH
    ):

        raise ValueError(
            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != (
        FEATURE_SIZE
    ):

        raise ValueError(
            f"Expected feature size "
            f"{FEATURE_SIZE}, "
            f"got {X.shape[2]}"
        )

    if len(X) != len(y):

        raise ValueError(
            "X and y lengths do not match."
        )

    if len(X) != len(video_ids):

        raise ValueError(
            "X and video_ids lengths "
            "do not match."
        )

    if np.isnan(X).any():

        raise ValueError(
            "Dataset contains NaN values."
        )

    if np.isinf(X).any():

        raise ValueError(
            "Dataset contains infinite values."
        )

    print(
        "Dataset validation: PASSED"
    )


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=DEFAULT_THRESHOLD
):

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int32
    )

    metrics = {

        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        )
    }

    if len(
        np.unique(y_true)
    ) == 2:

        metrics[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                probabilities
            )
        )

    else:

        metrics[
            "roc_auc"
        ] = None

    return (
        metrics,
        predictions
    )


# ============================================================
# Threshold Selection
# ============================================================

def select_optimal_threshold(
    y_true,
    probabilities,
    thresholds=THRESHOLD_VALUES
):

    print_header(
        "Validation Threshold Selection"
    )

    best_threshold = (
        DEFAULT_THRESHOLD
    )

    best_f1 = -1.0
    best_accuracy = -1.0

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(
            np.int32
        )

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        results.append({

            "threshold": float(
                threshold
            ),

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
            )
        })

        # ----------------------------------------------------
        # Primary objective:
        # F1
        #
        # Tie breaker:
        # Accuracy
        # ----------------------------------------------------

        if (
            f1 > best_f1
            or (
                np.isclose(
                    f1,
                    best_f1
                )
                and accuracy > best_accuracy
            )
        ):

            best_threshold = float(
                threshold
            )

            best_f1 = float(
                f1
            )

            best_accuracy = float(
                accuracy
            )

    results_df = pd.DataFrame(
        results
    )

    print(
        "Best validation threshold:",
        f"{best_threshold:.2f}"
    )

    print(
        "Validation F1:",
        f"{best_f1:.4f}"
    )

    print(
        "Validation accuracy:",
        f"{best_accuracy:.4f}"
    )

    print()
    print(
        "Top validation thresholds:"
    )

    print(
        results_df.sort_values(
            [
                "f1",
                "accuracy"
            ],
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    return (
        best_threshold,
        results_df
    )


# ============================================================
# Video-level aggregation
# ============================================================

def aggregate_video_predictions(
    y_true,
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

        labels = y_true[
            mask
        ]

        scores = probabilities[
            mask
        ]

        # ----------------------------------------------------
        # Every sequence from one video
        # must have the same label.
        # ----------------------------------------------------

        unique_labels = np.unique(
            labels
        )

        if len(
            unique_labels
        ) != 1:

            raise ValueError(
                f"Inconsistent labels for "
                f"video: {video}"
            )

        video_names.append(
            str(video)
        )

        video_labels.append(
            int(
                unique_labels[0]
            )
        )

        # ----------------------------------------------------
        # Mean sequence probability
        #
        # This becomes the final
        # video-level fake probability.
        # ----------------------------------------------------

        video_scores.append(
            float(
                np.mean(scores)
            )
        )

    return (

        np.asarray(
            video_names,
            dtype=str
        ),

        np.asarray(
            video_labels,
            dtype=np.int32
        ),

        np.asarray(
            video_scores,
            dtype=np.float32
        )
    )


# ============================================================
# Save evaluation results
# ============================================================

def save_results(
    y_test,
    sequence_probabilities,
    sequence_predictions,
    test_videos,

    video_names,
    video_labels,
    video_scores,
    video_predictions,

    sequence_metrics,
    video_metrics,

    selected_threshold,
    validation_threshold_results
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # Save video NPZ
    # ========================================================

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "video_predictions.npz"
        ),

        video_names=video_names,

        labels=video_labels,

        probabilities=video_scores,

        predictions=video_predictions
    )

    # ========================================================
    # Save sequence NPZ
    # ========================================================

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "sequence_predictions.npz"
        ),

        video_ids=test_videos,

        labels=y_test,

        probabilities=(
            sequence_probabilities
        ),

        predictions=sequence_predictions
    )

    # ========================================================
    # Save video CSV
    # ========================================================

    video_dataframe = pd.DataFrame({

        "video_name":
            video_names,

        "true_label":
            video_labels,

        "probability_fake":
            video_scores,

        "prediction":
            video_predictions
    })

    video_dataframe.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "video_predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # Save validation threshold analysis
    # ========================================================

    validation_threshold_results.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "validation_threshold_analysis.csv"
        ),

        index=False
    )

    # ========================================================
    # Save metrics JSON
    # ========================================================

    metrics = {

        "threshold_selection": {

            "method":
                "Validation-set F1 maximization",

            "selected_threshold":
                float(
                    selected_threshold
                ),

            "test_set_used_for_selection":
                False
        },

        "sequence_level":
            sequence_metrics,

        "video_level":
            video_metrics
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "evaluation_metrics.json"
        ),
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # ========================================================
    # Save confusion matrices
    # ========================================================

    sequence_cm = confusion_matrix(

        y_test,

        sequence_predictions,

        labels=[
            0,
            1
        ]
    )

    video_cm = confusion_matrix(

        video_labels,

        video_predictions,

        labels=[
            0,
            1
        ]
    )

    np.savetxt(

        os.path.join(
            OUTPUT_DIR,
            "sequence_confusion_matrix.csv"
        ),

        sequence_cm,

        fmt="%d",

        delimiter=","
    )

    np.savetxt(

        os.path.join(
            OUTPUT_DIR,
            "video_confusion_matrix.csv"
        ),

        video_cm,

        fmt="%d",

        delimiter=","
    )

    print()

    print(
        "Evaluation files saved to:",
        OUTPUT_DIR
    )


# ============================================================
# Main
# ============================================================

def main():

    print_header(
        "Multimodal Deepfake Lip-Sync Detector"
    )

    print(
        "Evaluation Pipeline"
    )

    print(
        "Feature size     :",
        FEATURE_SIZE
    )

    print(
        "Sequence length  :",
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
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Input shape:",
        model.input_shape
    )

    print(
        "Output shape:",
        model.output_shape
    )

    expected_input_shape = (
        None,
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    )

    if model.input_shape != (
        expected_input_shape
    ):

        raise ValueError(
            f"Unexpected model input shape: "
            f"{model.input_shape}. "
            f"Expected {expected_input_shape}"
        )

    # ========================================================
    # STEP 2
    # Load complete development dataset
    # ========================================================

    print_header(
        "Loading Development Dataset"
    )

    X, y, video_ids = (
        build_dataset()
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Video IDs:",
        video_ids.shape
    )

    validate_dataset(
        X,
        y,
        video_ids
    )

    # ========================================================
    # STEP 3
    # Recreate exact split
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

    ) = create_train_val_test_split(
        X,
        y,
        video_ids
    )

    # ========================================================
    # STEP 4
    # Load training scaler
    # ========================================================

    print_header(
        "Loading Training Scaler"
    )

    mean, scale = (
        load_scaler()
    )

    # ========================================================
    # STEP 5
    # Apply normalization
    #
    # IMPORTANT:
    # The scaler was fitted during training.
    #
    # We DO NOT fit anything on validation/test.
    # ========================================================

    print_header(
        "Applying Feature Normalization"
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
        "Scaled training shape:",
        X_train.shape
    )

    print(
        "Scaled validation shape:",
        X_val.shape
    )

    print(
        "Scaled test shape:",
        X_test.shape
    )

    print(
        "Training mean:",
        float(
            X_train.mean()
        )
    )

    print(
        "Training std:",
        float(
            X_train.std()
        )
    )

    print(
        "Validation mean:",
        float(
            X_val.mean()
        )
    )

    print(
        "Validation std:",
        float(
            X_val.std()
        )
    )

    print(
        "Test mean:",
        float(
            X_test.mean()
        )
    )

    print(
        "Test std:",
        float(
            X_test.std()
        )
    )

    # ========================================================
    # STEP 6
    # Validation predictions
    # ========================================================

    print_header(
        "Validation Prediction"
    )

    validation_probabilities = (
        model.predict(
            X_val,
            batch_size=16,
            verbose=1
        )
        .reshape(-1)
    )

    print(
        "Validation predictions generated:",
        len(
            validation_probabilities
        )
    )

    # ========================================================
    # STEP 7
    # Aggregate validation predictions
    #
    # Threshold selection happens at VIDEO LEVEL.
    #
    # This is important because the final detector
    # classifies complete videos.
    # ========================================================

    print_header(
        "Validation Video-Level Aggregation"
    )

    (
        validation_video_names,
        validation_video_labels,
        validation_video_scores

    ) = aggregate_video_predictions(

        y_val,

        validation_probabilities,

        val_videos
    )

    print(
        "Validation videos:",
        len(
            validation_video_names
        )
    )

    # ========================================================
    # STEP 8
    # Select optimal threshold
    #
    # ONLY validation data is used here.
    # ========================================================

    (
        selected_threshold,
        validation_threshold_results

    ) = select_optimal_threshold(

        validation_video_labels,

        validation_video_scores
    )

    # ========================================================
    # Show validation result using selected threshold
    # ========================================================

    (
        validation_metrics,
        validation_predictions
    ) = calculate_metrics(

        validation_video_labels,

        validation_video_scores,

        selected_threshold
    )

    print_header(
        "Selected Validation Threshold Results"
    )

    for name, value in (
        validation_metrics.items()
    ):

        if value is not None:

            print(
                f"{name:12s}: "
                f"{value:.4f}"
            )

    print()

    print(
        "Validation Confusion Matrix"
    )

    print(
        confusion_matrix(
            validation_video_labels,
            validation_predictions,
            labels=[
                0,
                1
            ]
        )
    )

    # ========================================================
    # STEP 9
    # TEST prediction
    #
    # IMPORTANT:
    # The test set has NOT been used to select
    # the threshold.
    # ========================================================

    print_header(
        "Test Prediction"
    )

    sequence_probabilities = (

        model.predict(

            X_test,

            batch_size=16,

            verbose=1

        )
        .reshape(-1)
    )

    print(
        "Test predictions generated:",
        len(
            sequence_probabilities
        )
    )

    # ========================================================
    # STEP 10
    # Sequence-level test metrics
    # ========================================================

    print_header(
        "Sequence-Level Results"
    )

    (
        sequence_metrics,
        sequence_predictions

    ) = calculate_metrics(

        y_test,

        sequence_probabilities,

        selected_threshold
    )

    for name, value in (
        sequence_metrics.items()
    ):

        if value is not None:

            print(
                f"{name:12s}: "
                f"{value:.4f}"
            )

    print()

    print(
        "Classification Report"
    )

    print(
        classification_report(

            y_test,

            sequence_predictions,

            labels=[
                0,
                1
            ],

            target_names=[
                "Real",
                "Fake"
            ],

            zero_division=0
        )
    )

    print(
        "Confusion Matrix"
    )

    print(
        confusion_matrix(

            y_test,

            sequence_predictions,

            labels=[
                0,
                1
            ]
        )
    )

    # ========================================================
    # STEP 11
    # Video-level test aggregation
    # ========================================================

    print_header(
        "Video-Level Aggregation"
    )

    (
        video_names,
        video_labels,
        video_scores

    ) = aggregate_video_predictions(

        y_test,

        sequence_probabilities,

        test_videos
    )

    print(
        "Unique test videos:",
        len(video_names)
    )

    # ========================================================
    # STEP 12
    # Video-level test metrics
    #
    # SAME threshold selected from validation.
    # ========================================================

    print_header(
        "Video-Level Results"
    )

    (
        video_metrics,
        video_predictions

    ) = calculate_metrics(

        video_labels,

        video_scores,

        selected_threshold
    )

    for name, value in (
        video_metrics.items()
    ):

        if value is not None:

            print(
                f"{name:12s}: "
                f"{value:.4f}"
            )

    print()

    print(
        "Video Classification Report"
    )

    print(
        classification_report(

            video_labels,

            video_predictions,

            labels=[
                0,
                1
            ],

            target_names=[
                "Real",
                "Fake"
            ],

            zero_division=0
        )
    )

    print(
        "Video Confusion Matrix"
    )

    print(
        confusion_matrix(

            video_labels,

            video_predictions,

            labels=[
                0,
                1
            ]
        )
    )

    # ========================================================
    # STEP 13
    # Save results
    # ========================================================

    print_header(
        "Saving Evaluation Results"
    )

    save_results(

        y_test=y_test,

        sequence_probabilities=(
            sequence_probabilities
        ),

        sequence_predictions=(
            sequence_predictions
        ),

        test_videos=test_videos,

        video_names=video_names,

        video_labels=video_labels,

        video_scores=video_scores,

        video_predictions=video_predictions,

        sequence_metrics=sequence_metrics,

        video_metrics=video_metrics,

        selected_threshold=(
            selected_threshold
        ),

        validation_threshold_results=(
            validation_threshold_results
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "Evaluation Complete"
    )

    print(
        "=" * 60
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Scaler:",
        SCALER_PATH
    )

    print(
        "Selected threshold:",
        f"{selected_threshold:.2f}"
    )

    print(
        "Test videos:",
        len(
            video_names
        )
    )

    print(
        "Test sequences:",
        len(
            y_test
        )
    )

    print()

    print(
        "Video Accuracy :",
        f"{video_metrics['accuracy']:.4f}"
    )

    print(
        "Video Precision:",
        f"{video_metrics['precision']:.4f}"
    )

    print(
        "Video Recall   :",
        f"{video_metrics['recall']:.4f}"
    )

    print(
        "Video F1       :",
        f"{video_metrics['f1']:.4f}"
    )

    print(
        "Video ROC-AUC  :",
        (
            f"{video_metrics['roc_auc']:.4f}"
            if video_metrics["roc_auc"]
            is not None
            else "N/A"
        )
    )

    print()

    print(
        "Results directory:",
        OUTPUT_DIR
    )

    print(
        "=" * 60
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()