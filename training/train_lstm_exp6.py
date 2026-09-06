from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# Configuration
# ============================================================

DATASET_FOLDER = Path(
    "outputs/dataset_exp5_full"
)


TRAIN_FILE = (
    DATASET_FOLDER /
    "train.npz"
)


VALIDATION_FILE = (
    DATASET_FOLDER /
    "validation.npz"
)


MODEL_OUTPUT_FOLDER = Path(
    "models"
)


BEST_MODEL_FILE = (
    MODEL_OUTPUT_FOLDER /
    "lstm_exp6_best.keras"
)


# ============================================================
# Model Configuration
# ============================================================

SEQUENCE_LENGTH = 30

FEATURE_SIZE = 199

NUM_CLASSES = 1


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.001

RANDOM_STATE = 42


# ============================================================
# Set Random Seed
# ============================================================

def set_random_seed():

    np.random.seed(
        RANDOM_STATE
    )

    tf.random.set_seed(
        RANDOM_STATE
    )


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(
    dataset_file
):

    if not dataset_file.exists():

        raise FileNotFoundError(
            f"Dataset file not found: "
            f"{dataset_file}"
        )


    data = np.load(
        dataset_file,
        allow_pickle=True
    )


    X = data["X"].astype(
        np.float32
    )


    y = data["y"].astype(
        np.float32
    )


    video_ids = data["video_ids"]


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            f"Expected X to have 3 dimensions, "
            f"got {X.ndim}"
        )


    if X.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"got {X.shape[1]}"
        )


    if X.shape[2] != FEATURE_SIZE:

        raise ValueError(
            f"Expected feature size "
            f"{FEATURE_SIZE}, "
            f"got {X.shape[2]}"
        )


    if len(X) != len(y):

        raise ValueError(
            "X and y length mismatch."
        )


    if len(X) != len(video_ids):

        raise ValueError(
            "X and video_ids length mismatch."
        )


    # --------------------------------------------------------
    # Check for NaN
    # --------------------------------------------------------

    if np.isnan(X).any():

        raise ValueError(
            "NaN values found in X."
        )


    if np.isinf(X).any():

        raise ValueError(
            "Inf values found in X."
        )


    return X, y, video_ids


# ============================================================
# Normalize Features
# ============================================================

def normalize_features(
    X_train,
    X_validation
):

    """
    Normalize features using ONLY
    training data statistics.

    This prevents validation leakage.
    """


    # --------------------------------------------------------
    # Calculate mean and standard deviation
    #
    # Shape:
    #
    # (num_sequences,
    #  sequence_length,
    #  feature_size)
    #
    # Statistics are calculated from
    # training data only.
    # --------------------------------------------------------

    mean = np.mean(

        X_train,

        axis=(0, 1),

        keepdims=True

    )


    std = np.std(

        X_train,

        axis=(0, 1),

        keepdims=True

    )


    # --------------------------------------------------------
    # Prevent division by zero
    # --------------------------------------------------------

    std = np.maximum(

        std,

        1e-6

    )


    # --------------------------------------------------------
    # Normalize training data
    # --------------------------------------------------------

    X_train_normalized = (

        X_train - mean

    ) / std


    # --------------------------------------------------------
    # Normalize validation data
    #
    # IMPORTANT:
    # Use training mean and std.
    # --------------------------------------------------------

    X_validation_normalized = (

        X_validation - mean

    ) / std


    return (

        X_train_normalized.astype(
            np.float32
        ),

        X_validation_normalized.astype(
            np.float32
        ),

        mean.astype(
            np.float32
        ),

        std.astype(
            np.float32
        )

    )


# ============================================================
# Calculate Class Weights
# ============================================================

def calculate_class_weights(
    y_train
):

    """
    Calculate balanced class weights.

    Real = 0
    Fake = 1

    Formula:

    weight =
        total_samples /
        (number_of_classes *
         samples_in_class)
    """


    total_samples = len(
        y_train
    )


    real_samples = int(

        np.sum(
            y_train == 0
        )

    )


    fake_samples = int(

        np.sum(
            y_train == 1
        )

    )


    if real_samples == 0:

        raise ValueError(
            "No Real samples found."
        )


    if fake_samples == 0:

        raise ValueError(
            "No Fake samples found."
        )


    real_weight = (

        total_samples /

        (
            2 *
            real_samples
        )

    )


    fake_weight = (

        total_samples /

        (
            2 *
            fake_samples
        )

    )


    class_weights = {

        0: float(
            real_weight
        ),

        1: float(
            fake_weight
        )

    }


    return (

        class_weights,

        real_samples,

        fake_samples

    )


# ============================================================
# Build LSTM Model
# ============================================================

def build_model():

    model = Sequential(

        [

            # ------------------------------------------------
            # Input
            # ------------------------------------------------

            Input(
                shape=(

                    SEQUENCE_LENGTH,

                    FEATURE_SIZE

                )
            ),


            # ------------------------------------------------
            # First LSTM Layer
            # ------------------------------------------------

            LSTM(

                64,

                return_sequences=True

            ),


            BatchNormalization(),


            Dropout(

                0.30

            ),


            # ------------------------------------------------
            # Second LSTM Layer
            # ------------------------------------------------

            LSTM(

                32,

                return_sequences=False

            ),


            BatchNormalization(),


            Dropout(

                0.30

            ),


            # ------------------------------------------------
            # Dense Layer
            # ------------------------------------------------

            Dense(

                32,

                activation="relu"

            ),


            Dropout(

                0.20

            ),


            # ------------------------------------------------
            # Output Layer
            #
            # Real = 0
            # Fake = 1
            # ------------------------------------------------

            Dense(

                NUM_CLASSES,

                activation="sigmoid"

            )

        ]

    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(

        learning_rate=LEARNING_RATE

    )


    # --------------------------------------------------------
    # Compile Model
    # --------------------------------------------------------

    model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[

            "accuracy"

        ]

    )


    return model


# ============================================================
# Print Dataset Information
# ============================================================

def print_dataset_info(

    name,

    X,

    y,

    video_ids

):

    print(

        "\n" + "=" * 60

    )


    print(

        f"{name} DATASET"

    )


    print(

        "=" * 60

    )


    print(

        "X shape       :",

        X.shape

    )


    print(

        "y shape       :",

        y.shape

    )


    print(

        "Unique videos :",

        len(

            np.unique(

                video_ids

            )

        )

    )


    print(

        "Real samples  :",

        int(

            np.sum(

                y == 0

            )

        )

    )


    print(

        "Fake samples  :",

        int(

            np.sum(

                y == 1

            )

        )

    )


    print(

        "=" * 60

    )


# ============================================================
# Main
# ============================================================

def main():


    # --------------------------------------------------------
    # Random Seed
    # --------------------------------------------------------

    set_random_seed()


    # --------------------------------------------------------
    # Create Model Directory
    # --------------------------------------------------------

    MODEL_OUTPUT_FOLDER.mkdir(

        parents=True,

        exist_ok=True

    )


    # --------------------------------------------------------
    # Load Training Dataset
    # --------------------------------------------------------

    print(

        "\nLoading training dataset..."

    )


    (

        X_train,

        y_train,

        train_video_ids

    ) = load_dataset(

        TRAIN_FILE

    )


    # --------------------------------------------------------
    # Load Validation Dataset
    # --------------------------------------------------------

    print(

        "Loading validation dataset..."

    )


    (

        X_validation,

        y_validation,

        validation_video_ids

    ) = load_dataset(

        VALIDATION_FILE

    )


    # --------------------------------------------------------
    # Print Dataset Information
    # --------------------------------------------------------

    print_dataset_info(

        "TRAIN",

        X_train,

        y_train,

        train_video_ids

    )


    print_dataset_info(

        "VALIDATION",

        X_validation,

        y_validation,

        validation_video_ids

    )


    # ========================================================
    # Calculate Class Weights
    # ========================================================

    (

        class_weights,

        real_samples,

        fake_samples

    ) = calculate_class_weights(

        y_train

    )


    print(

        "\n" + "=" * 60

    )


    print(

        "CLASS WEIGHTS"

    )


    print(

        "=" * 60

    )


    print(

        "Real samples :",

        real_samples

    )


    print(

        "Fake samples :",

        fake_samples

    )


    print()


    print(

        "Real class weight (0):",

        round(

            class_weights[0],

            6

        )

    )


    print(

        "Fake class weight (1):",

        round(

            class_weights[1],

            6

        )

    )


    print(

        "=" * 60

    )


    # --------------------------------------------------------
    # Normalize Features
    # --------------------------------------------------------

    print(

        "\nNormalizing features..."

    )


    (

        X_train,

        X_validation,

        feature_mean,

        feature_std

    ) = normalize_features(

        X_train,

        X_validation

    )


    # --------------------------------------------------------
    # Save Normalization Statistics
    # --------------------------------------------------------

    normalization_file = (

        MODEL_OUTPUT_FOLDER

        /

        "lstm_exp6_normalization.npz"

    )


    np.savez(

        normalization_file,

        mean=feature_mean,

        std=feature_std

    )


    print(

        "Normalization saved:",

        normalization_file

    )


    # --------------------------------------------------------
    # Build Model
    # --------------------------------------------------------

    print(

        "\nBuilding LSTM model..."

    )


    model = build_model()


    # --------------------------------------------------------
    # Model Summary
    # --------------------------------------------------------

    print(

        "\nMODEL ARCHITECTURE"

    )


    print(

        "=" * 60

    )


    model.summary()


    # ========================================================
    # Callbacks
    # ========================================================

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=8,

        restore_best_weights=True,

        verbose=1

    )


    model_checkpoint = ModelCheckpoint(

        filepath=str(

            BEST_MODEL_FILE

        ),

        monitor="val_loss",

        save_best_only=True,

        verbose=1

    )


    reduce_learning_rate = ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=1e-6,

        verbose=1

    )


    callbacks = [

        early_stopping,

        model_checkpoint,

        reduce_learning_rate

    ]


    # ========================================================
    # Training
    # ========================================================

    print(

        "\n" + "=" * 60

    )


    print(

        "STARTING EXP6 LSTM TRAINING"

    )


    print(

        "=" * 60

    )


    print(

        "Epochs        :",

        EPOCHS

    )


    print(

        "Batch size    :",

        BATCH_SIZE

    )


    print(

        "Learning rate :",

        LEARNING_RATE

    )


    print(

        "Class weights :",

        class_weights

    )


    print(

        "=" * 60

    )


    # --------------------------------------------------------
    # Train Model
    #
    # EXP6 CHANGE:
    #
    # class_weight is applied here.
    # --------------------------------------------------------

    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_validation,

            y_validation

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # Save Final Model
    # ========================================================

    final_model_file = (

        MODEL_OUTPUT_FOLDER

        /

        "lstm_exp6_final.keras"

    )


    model.save(

        final_model_file

    )


    print(

        "\nFinal model saved:"

    )


    print(

        final_model_file

    )


    # ========================================================
    # Save Training History
    # ========================================================

    history_file = (

        MODEL_OUTPUT_FOLDER

        /

        "lstm_exp6_history.npz"

    )


    np.savez(

        history_file,

        loss=np.array(

            history.history["loss"]

        ),

        accuracy=np.array(

            history.history["accuracy"]

        ),

        val_loss=np.array(

            history.history["val_loss"]

        ),

        val_accuracy=np.array(

            history.history["val_accuracy"]

        )

    )


    print(

        "\nTraining history saved:"

    )


    print(

        history_file

    )


    # ========================================================
    # Final Training Evaluation
    # ========================================================

    print(

        "\n" + "=" * 60

    )


    print(

        "FINAL TRAINING EVALUATION"

    )


    print(

        "=" * 60

    )


    train_loss, train_accuracy = model.evaluate(

        X_train,

        y_train,

        verbose=0

    )


    validation_loss, validation_accuracy = model.evaluate(

        X_validation,

        y_validation,

        verbose=0

    )


    print(

        "Training Loss      :",

        round(

            float(train_loss),

            6

        )

    )


    print(

        "Training Accuracy  :",

        round(

            float(

                train_accuracy * 100

            ),

            2

        ),

        "%"

    )


    print()


    print(

        "Validation Loss    :",

        round(

            float(validation_loss),

            6

        )

    )


    print(

        "Validation Accuracy:",

        round(

            float(

                validation_accuracy * 100

            ),

            2

        ),

        "%"

    )


    print(

        "\n" + "=" * 60

    )


    print(

        "EXP6 LSTM TRAINING COMPLETED"

    )


    print(

        "=" * 60

    )


    print(

        "\nBest model:"

    )


    print(

        BEST_MODEL_FILE

    )


    print(

        "\nFinal model:"

    )


    print(

        final_model_file

    )


    print(

        "\nNormalization:"

    )


    print(

        normalization_file

    )


    print(

        "\nHistory:"

    )


    print(

        history_file

    )


if __name__ == "__main__":

    main()