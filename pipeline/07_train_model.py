# ============================================================
# 07_train_model.py
# Train Multimodal LSTM Deepfake Detector
# ============================================================

from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


MULTIMODAL_FEATURE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "multimodal_features"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "models"
    / "lstm_multimodal_model.keras"
)


TRAINED_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "models"
    / "trained_lstm_multimodal_model.keras"
)


BEST_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "models"
    / "best_lstm_multimodal_model.keras"
)


NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "models"
    / "normalization_stats.npz"
)


MAX_SEQUENCE_LENGTH = 150

FEATURE_DIMENSION = 119

BATCH_SIZE = 16

EPOCHS = 30


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split):

    split_path = (
        MULTIMODAL_FEATURE_ROOT
        / split
    )

    files = sorted(
        split_path.glob("*.npz")
    )

    print()

    print(
        f"Loading {split.upper()} dataset..."
    )

    print(
        f"Samples found: {len(files)}"
    )

    features_list = []

    labels_list = []

    for index, file_path in enumerate(
        files,
        start=1
    ):

        with np.load(file_path) as data:

            features = (
                data["features"]
                .astype(np.float32)
            )

            label = int(
                data["label"]
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if features.ndim != 2:

            raise ValueError(

                f"Invalid feature shape in:\n"
                f"{file_path}\n"
                f"Shape: {features.shape}"

            )

        if features.shape[1] != FEATURE_DIMENSION:

            raise ValueError(

                f"Invalid feature dimension in:\n"
                f"{file_path}\n"
                f"Expected: {FEATURE_DIMENSION}\n"
                f"Found: {features.shape[1]}"

            )

        # ----------------------------------------------------
        # Truncate if longer than maximum
        # ----------------------------------------------------

        features = features[
            :MAX_SEQUENCE_LENGTH
        ]

        # ----------------------------------------------------
        # Pad sequence
        # ----------------------------------------------------

        padded_features = np.zeros(

            (
                MAX_SEQUENCE_LENGTH,
                FEATURE_DIMENSION
            ),

            dtype=np.float32

        )

        sequence_length = min(

            len(features),

            MAX_SEQUENCE_LENGTH

        )

        padded_features[
            :sequence_length
        ] = features[
            :sequence_length
        ]

        features_list.append(
            padded_features
        )

        labels_list.append(
            label
        )

        if (
            index % 100 == 0
            or index == len(files)
        ):

            print(

                f"[{index}/{len(files)}] Loaded"

            )


    X = np.array(

        features_list,

        dtype=np.float32

    )


    y = np.array(

        labels_list,

        dtype=np.int32

    )


    print()

    print(
        "Dataset shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )


    return X, y


# ============================================================
# CALCULATE NORMALIZATION STATISTICS
# ============================================================

def calculate_normalization_statistics(
    X
):

    print()

    print(
        "Calculating normalization statistics..."
    )


    # --------------------------------------------------------
    # Identify non-zero frames
    #
    # Padding frames are completely zero.
    # --------------------------------------------------------

    valid_frames = X[
        np.any(
            X != 0,
            axis=2
        )
    ]


    print(

        "Valid frames used:",
        valid_frames.shape[0]

    )


    mean = np.mean(

        valid_frames,

        axis=0

    )


    std = np.std(

        valid_frames,

        axis=0

    )


    # --------------------------------------------------------
    # Prevent division by zero
    # --------------------------------------------------------

    std[std < 1e-8] = 1.0


    return (

        mean.astype(np.float32),

        std.astype(np.float32)

    )


# ============================================================
# NORMALIZE DATASET
# ============================================================

def normalize_dataset(
    X,
    mean,
    std
):

    print(
        "Normalizing dataset..."
    )


    normalized_X = np.zeros_like(

        X,

        dtype=np.float32

    )


    valid_mask = np.any(

        X != 0,

        axis=2

    )


    normalized_X[
        valid_mask
    ] = (

        X[valid_mask] - mean

    ) / std


    return normalized_X


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

def print_dataset_information(
    name,
    X,
    y
):

    real_samples = int(
        np.sum(y == 0)
    )


    fake_samples = int(
        np.sum(y == 1)
    )


    print()

    print(
        "=" * 70
    )

    print(
        f"{name.upper()} DATASET"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Features shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    print()

    print(
        "Real samples:",
        real_samples
    )

    print(
        "Fake samples:",
        fake_samples
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "MULTIMODAL LSTM MODEL TRAINING"
    )

    print(
        "=" * 70
    )

    print()


    # ========================================================
    # CHECK REQUIRED PATHS
    # ========================================================

    print(
        "Checking required files..."
    )

    print()


    if not MULTIMODAL_FEATURE_ROOT.exists():

        raise FileNotFoundError(

            "Multimodal feature directory not found:\n"
            f"{MULTIMODAL_FEATURE_ROOT}"

        )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            "LSTM model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run:\n"
            "python pipeline/06_build_lstm_model.py"

        )


    print(
        "Multimodal features:"
    )

    print(
        MULTIMODAL_FEATURE_ROOT
    )

    print()


    print(
        "Base model:"
    )

    print(
        MODEL_PATH
    )

    print()


    # ========================================================
    # LOAD DATASETS
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "LOADING DATASETS"
    )

    print(
        "=" * 70
    )


    X_train, y_train = load_dataset(
        "train"
    )


    X_validation, y_validation = load_dataset(
        "validation"
    )


    X_test, y_test = load_dataset(
        "test"
    )


    # ========================================================
    # PRINT DATASET INFORMATION
    # ========================================================

    print_dataset_information(

        "Train",

        X_train,

        y_train

    )


    print_dataset_information(

        "Validation",

        X_validation,

        y_validation

    )


    print_dataset_information(

        "Test",

        X_test,

        y_test

    )


    # ========================================================
    # NORMALIZATION
    #
    # Statistics are calculated ONLY from training data.
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FEATURE NORMALIZATION"
    )

    print(
        "=" * 70
    )


    mean, std = (
        calculate_normalization_statistics(
            X_train
        )
    )


    NORMALIZATION_PATH.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    np.savez(

        NORMALIZATION_PATH,

        mean=mean,

        std=std

    )


    print()

    print(
        "Normalization statistics saved:"
    )

    print(
        NORMALIZATION_PATH
    )

    print()


    # ========================================================
    # NORMALIZE ALL DATASETS
    # ========================================================

    X_train = normalize_dataset(

        X_train,

        mean,

        std

    )


    X_validation = normalize_dataset(

        X_validation,

        mean,

        std

    )


    X_test = normalize_dataset(

        X_test,

        mean,

        std

    )


    print()

    print(
        "Normalization complete."
    )

    print()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "LOADING LSTM MODEL"
    )

    print(
        "=" * 70
    )

    print()


    model = tf.keras.models.load_model(

        MODEL_PATH

    )


    print(
        "Model loaded successfully."
    )

    print()


    print(
        "Model input shape:",
        model.input_shape
    )


    print(
        "Model output shape:",
        model.output_shape
    )


    print()


    # ========================================================
    # CALLBACKS
    # ========================================================

    BEST_MODEL_PATH.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1

    )


    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

        filepath=BEST_MODEL_PATH,

        monitor="val_loss",

        save_best_only=True,

        verbose=1

    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 70
    )

    print()


    print(
        "Epochs:",
        EPOCHS
    )


    print(
        "Batch size:",
        BATCH_SIZE
    )


    print()


    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_validation,

            y_validation

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[

            early_stopping,

            model_checkpoint

        ],

        verbose=1

    )


    # ========================================================
    # SAVE TRAINED MODEL
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SAVING TRAINED MODEL"
    )

    print(
        "=" * 70
    )

    print()


    model.save(

        TRAINED_MODEL_PATH

    )


    print(
        "Trained model saved:"
    )

    print(
        TRAINED_MODEL_PATH
    )


    print()


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "FINAL MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    print()


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_accuracy = model.evaluate(

        X_train,

        y_train,

        verbose=0

    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_loss, validation_accuracy = model.evaluate(

        X_validation,

        y_validation,

        verbose=0

    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_loss, test_accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=0

    )


    print(
        "Training Results"
    )

    print(
        f"Loss:     {train_loss:.4f}"
    )

    print(
        f"Accuracy: {train_accuracy:.4f}"
    )

    print()


    print(
        "Validation Results"
    )

    print(
        f"Loss:     {validation_loss:.4f}"
    )

    print(
        f"Accuracy: {validation_accuracy:.4f}"
    )

    print()


    print(
        "Test Results"
    )

    print(
        f"Loss:     {test_loss:.4f}"
    )

    print(
        f"Accuracy: {test_accuracy:.4f}"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "MODEL TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_validation)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    print()


    print(
        "Final test accuracy:",
        f"{test_accuracy:.4f}"
    )

    print()


    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()


    print(
        "Trained model:"
    )

    print(
        TRAINED_MODEL_PATH
    )

    print()


    print(
        "Normalization statistics:"
    )

    print(
        NORMALIZATION_PATH
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()