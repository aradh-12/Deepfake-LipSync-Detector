import os
import json

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from models.lstm_model_exp5 import build_lstm_model


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 199

RANDOM_STATE = 42

EPOCHS = 50
BATCH_SIZE = 16

MODEL_PATH = "models/deepfake_lipsync_lstm_exp5.keras"
SCALER_PATH = "models/deepfake_lipsync_feature_scaler_exp5.npz"

DATASET_DIR = "outputs/dataset_exp5_full"


# ============================================================
# Reproducibility
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Helper: Get video-level information
# ============================================================

def get_video_level_data(y, video_ids):

    unique_videos = np.unique(video_ids)

    video_labels = []

    for video in unique_videos:

        labels = y[video_ids == video]

        # Every sequence belonging to one video
        # must have the same label.
        if len(np.unique(labels)) != 1:

            raise ValueError(
                f"Video has inconsistent labels: {video}"
            )

        video_labels.append(labels[0])

    return (
        np.asarray(unique_videos),
        np.asarray(video_labels)
    )


# ============================================================
# Helper: Print split information
# ============================================================

def print_split_info(
    name,
    X_split,
    y_split,
    video_ids_split
):

    videos = np.unique(video_ids_split)

    print(f"\n{name}")
    print("-" * 40)

    print(
        "Sequences :",
        len(X_split)
    )

    print(
        "Videos    :",
        len(videos)
    )

    print(
        "Real      :",
        np.sum(y_split == 0)
    )

    print(
        "Fake      :",
        np.sum(y_split == 1)
    )


# ============================================================
# Feature Scaling
# ============================================================

def scale_features(
    X_train,
    X_val,
    X_test
):
    """
    Standardize all 199 multimodal features.

    IMPORTANT:
    The scaler is fitted ONLY on training data.

    Validation and test data are transformed using
    statistics learned from the training data.

    This prevents data leakage.
    """

    scaler = StandardScaler()

    # --------------------------------------------------------
    # Store original shapes
    # --------------------------------------------------------

    train_shape = X_train.shape
    val_shape = X_val.shape
    test_shape = X_test.shape

    # Expected:
    #
    # X_train = (samples, 30, 199)
    # X_val   = (samples, 30, 199)
    # X_test  = (samples, 30, 199)

    # --------------------------------------------------------
    # Flatten temporal dimension
    # --------------------------------------------------------

    X_train_flat = X_train.reshape(
        -1,
        FEATURE_SIZE
    )

    X_val_flat = X_val.reshape(
        -1,
        FEATURE_SIZE
    )

    X_test_flat = X_test.reshape(
        -1,
        FEATURE_SIZE
    )

    # --------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    X_train_flat = scaler.fit_transform(
        X_train_flat
    )

    # --------------------------------------------------------
    # Transform validation/test
    # using training statistics
    # --------------------------------------------------------

    X_val_flat = scaler.transform(
        X_val_flat
    )

    X_test_flat = scaler.transform(
        X_test_flat
    )

    # --------------------------------------------------------
    # Restore original sequence shapes
    # --------------------------------------------------------

    X_train = X_train_flat.reshape(
        train_shape
    ).astype(np.float32)

    X_val = X_val_flat.reshape(
        val_shape
    ).astype(np.float32)

    X_test = X_test_flat.reshape(
        test_shape
    ).astype(np.float32)

    return (
        X_train,
        X_val,
        X_test,
        scaler
    )


# ============================================================
# Save scaler
# ============================================================

def save_scaler(scaler):

    os.makedirs(
        "models",
        exist_ok=True
    )

    np.savez(
        SCALER_PATH,
        mean=scaler.mean_,
        scale=scaler.scale_
    )

    print(
        f"\nFeature scaler saved to : {SCALER_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n==============================")
    print("Loading Dataset")
    print("==============================\n")

    # ========================================================
    # STEP 1
    # Load precomputed video-level dataset split
    # ========================================================

    print("\n==============================")
    print("Loading Precomputed Dataset Split")
    print("==============================")

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

    if not os.path.exists(train_path):

        raise FileNotFoundError(
            f"Training split not found: {train_path}\n"
            "Run: python -m training.dataset_split_exp5"
        )

    if not os.path.exists(validation_path):

        raise FileNotFoundError(
            f"Validation split not found: {validation_path}\n"
            "Run: python -m training.dataset_split_exp5"
        )

    if not os.path.exists(test_path):

        raise FileNotFoundError(
            f"Test split not found: {test_path}\n"
            "Run: python -m training.dataset_split_exp5"
        )

    train_data = np.load(
        train_path,
        allow_pickle=True
    )

    val_data = np.load(
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

    X_val = val_data["X"]
    y_val = val_data["y"]
    val_videos = val_data["video_ids"]

    X_test = test_data["X"]
    y_test = test_data["y"]
    test_videos = test_data["video_ids"]

    print(
        "Training set loaded   :",
        X_train.shape
    )

    print(
        "Validation set loaded :",
        X_val.shape
    )

    print(
        "Test set loaded       :",
        X_test.shape
    )

    # ========================================================
    # STEP 2
    # Verify split shapes
    # ========================================================

    for name, X_split, y_split, ids_split in [

        (
            "Training",
            X_train,
            y_train,
            train_videos
        ),

        (
            "Validation",
            X_val,
            y_val,
            val_videos
        ),

        (
            "Test",
            X_test,
            y_test,
            test_videos
        )

    ]:

        if X_split.ndim != 3:

            raise ValueError(
                f"{name} X must be 3-dimensional. "
                f"Got {X_split.shape}"
            )

        if X_split.shape[1] != SEQUENCE_LENGTH:

            raise ValueError(
                f"{name} sequence length mismatch. "
                f"Expected {SEQUENCE_LENGTH}, "
                f"got {X_split.shape[1]}"
            )

        if X_split.shape[2] != FEATURE_SIZE:

            raise ValueError(
                f"{name} feature size mismatch. "
                f"Expected {FEATURE_SIZE}, "
                f"got {X_split.shape[2]}"
            )

        if len(X_split) != len(y_split):

            raise ValueError(
                f"{name}: X and y lengths differ."
            )

        if len(X_split) != len(ids_split):

            raise ValueError(
                f"{name}: X and video_ids lengths differ."
            )

        if np.isnan(X_split).any():

            raise ValueError(
                f"{name} dataset contains NaN values."
            )

        if np.isinf(X_split).any():

            raise ValueError(
                f"{name} dataset contains infinite values."
            )

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

    if train_video_set & val_video_set:

        raise RuntimeError(
            "Data leakage detected between "
            "training and validation videos."
        )

    if train_video_set & test_video_set:

        raise RuntimeError(
            "Data leakage detected between "
            "training and test videos."
        )

    if val_video_set & test_video_set:

        raise RuntimeError(
            "Data leakage detected between "
            "validation and test videos."
        )

    print(
        "\nData leakage check : PASSED"
    )

    # ========================================================
    # Print split information
    # ========================================================

    print("\n==============================")
    print("Video-Level Dataset Split")
    print("==============================")

    print_split_info(
        "Training",
        X_train,
        y_train,
        train_videos
    )

    print_split_info(
        "Validation",
        X_val,
        y_val,
        val_videos
    )

    print_split_info(
        "Test",
        X_test,
        y_test,
        test_videos
    )

    print("\n==============================")
    print("Leakage Check")
    print("==============================")

    print(
        "Train ∩ Validation :",
        len(
            train_video_set &
            val_video_set
        )
    )

    print(
        "Train ∩ Test       :",
        len(
            train_video_set &
            test_video_set
        )
    )

    print(
        "Validation ∩ Test  :",
        len(
            val_video_set &
            test_video_set
        )
    )

    # ========================================================
    # STEP 3
    # Feature normalization
    # ========================================================

    print("\n==============================")
    print("Feature Normalization")
    print("==============================")

    print(
        "Before scaling:"
    )

    print(
        "Train mean :",
        float(
            X_train.mean()
        )
    )

    print(
        "Train std  :",
        float(
            X_train.std()
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    print("\nAfter scaling:")

    print(
        "Train mean :",
        float(
            X_train.mean()
        )
    )

    print(
        "Train std  :",
        float(
            X_train.std()
        )
    )

    print(
        "Training features normalized   : OK"
    )

    print(
        "Validation features transformed : OK"
    )

    print(
        "Testing features transformed    : OK"
    )

    print(
        "=============================="
    )

    # --------------------------------------------------------
    # Save scaler for future inference
    # --------------------------------------------------------

    save_scaler(
        scaler
    )

    # ========================================================
    # STEP 4
    # Class weights
    # ========================================================

    classes = np.unique(
        y_train
    )

    if len(classes) < 2:

        raise RuntimeError(
            "Training set contains only one class."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(classes[i]): float(weights[i])
        for i in range(len(classes))
    }

    print("\n==============================")
    print("Class Weights")
    print("==============================")

    print(
        class_weights
    )

    # ========================================================
    # STEP 5
    # Build model
    # ========================================================

    print("\n==============================")
    print("Building LSTM Model")
    print("==============================\n")

    model = build_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        feature_size=FEATURE_SIZE
    )

    model.summary()

    # ========================================================
    # STEP 6
    # Callbacks
    # ========================================================

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )

    ]

    # ========================================================
    # STEP 7
    # Train
    # ========================================================

    print("\n==============================")
    print("Starting Training")
    print("==============================\n")

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1
    )

    # ========================================================
    # Save Training History
    # ========================================================

    os.makedirs(
        "outputs/training",
        exist_ok=True
    )

    history_path = (
        "outputs/training/training_history_exp5.json"
    )

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(

            {
                key: [
                    float(value)
                    for value in values
                ]
                for key, values
                in history.history.items()
            },

            f,

            indent=2
        )

    print(
        "Training history saved to:",
        history_path
    )

    # ========================================================
    # STEP 8
    # Final Test Evaluation
    # ========================================================

    print("\n==============================")
    print("Final Test Evaluation")
    print("==============================")

    test_results = model.evaluate(
        X_test,
        y_test,
        verbose=1,
        return_dict=True
    )

    print("\nTest Results")
    print("------------------------------")

    for metric_name, metric_value in test_results.items():

        print(
            f"{metric_name:10s}: "
            f"{metric_value:.4f}"
        )

    # ========================================================
    # STEP 9
    # Save Model
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    print("\n==============================")
    print("Training Complete")
    print("==============================")

    print(
        "Model saved to :",
        MODEL_PATH
    )

    print(
        "Scaler saved to:",
        SCALER_PATH
    )

    print(
        "History saved to:",
        history_path
    )

    print("==============================")

    return history


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()