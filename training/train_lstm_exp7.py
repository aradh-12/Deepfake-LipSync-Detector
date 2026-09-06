from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp7_source_aware"
)

MODEL_DIR = Path(
    "saved_models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "lstm_exp7.keras"

NORMALIZATION_PATH = MODEL_DIR / "lstm_exp7_normalization.npz"


# ============================================================
# Training Configuration
# ============================================================

EPOCHS = 50

BATCH_SIZE = 16

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ============================================================
# Set Random Seeds
# ============================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(filename):

    path = DATASET_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]

    return X, y


# ============================================================
# Normalize Dataset
# ============================================================

def normalize_data(
    X_train,
    X_validation
):

    print("\nCalculating normalization statistics...")

    # --------------------------------------------------------
    # Calculate mean and standard deviation
    # ONLY from training data
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

    # Avoid division by zero

    std = np.where(
        std < 1e-8,
        1e-8,
        std
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    X_train_normalized = (
        X_train - mean
    ) / std

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
        mean,
        std
    )


# ============================================================
# Build LSTM Model
# ============================================================

def build_model(
    sequence_length,
    feature_size
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                sequence_length,
                feature_size
            )
        ),

        # ----------------------------------------------------
        # First LSTM Layer
        # ----------------------------------------------------

        tf.keras.layers.LSTM(
            128,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        # ----------------------------------------------------
        # Second LSTM Layer
        # ----------------------------------------------------

        tf.keras.layers.LSTM(
            64
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        # ----------------------------------------------------
        # Dense Layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        # ----------------------------------------------------
        # Output Layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )

    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]

    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("EXP7 LSTM TRAINING")
    print("SOURCE-AWARE DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Load Train Dataset
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    X_train, y_train = load_dataset(
        "train.npz"
    )

    # --------------------------------------------------------
    # Load Validation Dataset
    # --------------------------------------------------------

    print(
        "\nLoading validation dataset..."
    )

    X_validation, y_validation = (
        load_dataset(
            "validation.npz"
        )
    )

    # --------------------------------------------------------
    # Dataset Information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(
        "\nTraining X shape:",
        X_train.shape
    )

    print(
        "Training y shape:",
        y_train.shape
    )

    print(
        "\nValidation X shape:",
        X_validation.shape
    )

    print(
        "Validation y shape:",
        y_validation.shape
    )

    print("\nTraining distribution:")

    print(
        "Real:",
        np.sum(y_train == 0)
    )

    print(
        "Fake:",
        np.sum(y_train == 1)
    )

    print("\nValidation distribution:")

    print(
        "Real:",
        np.sum(y_validation == 0)
    )

    print(
        "Fake:",
        np.sum(y_validation == 1)
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    (
        X_train,
        X_validation,
        mean,
        std
    ) = normalize_data(

        X_train,
        X_validation

    )

    # --------------------------------------------------------
    # Save Normalization Statistics
    # --------------------------------------------------------

    np.savez(

        NORMALIZATION_PATH,

        mean=mean,

        std=std

    )

    print(
        "\nNormalization saved:"
    )

    print(
        NORMALIZATION_PATH
    )

    # --------------------------------------------------------
    # Calculate Class Weights
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASS WEIGHTS")
    print("=" * 60)

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y_train

    )

    class_weights = {

        int(class_label):
        float(weight)

        for class_label, weight

        in zip(
            classes,
            weights
        )

    }

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )

    # --------------------------------------------------------
    # Build Model
    # --------------------------------------------------------

    sequence_length = X_train.shape[1]

    feature_size = X_train.shape[2]

    print("\n" + "=" * 60)
    print("BUILDING LSTM MODEL")
    print("=" * 60)

    print(
        "\nSequence length:",
        sequence_length
    )

    print(
        "Feature size:",
        feature_size
    )

    model = build_model(

        sequence_length,

        feature_size

    )

    print("\nModel Summary:")

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        # ----------------------------------------------------
        # Save Best Model
        # ----------------------------------------------------

        tf.keras.callbacks.ModelCheckpoint(

            filepath=MODEL_PATH,

            monitor="val_loss",

            save_best_only=True,

            mode="min",

            verbose=1

        ),

        # ----------------------------------------------------
        # Stop if Validation Loss Stops Improving
        # ----------------------------------------------------

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=10,

            restore_best_weights=True,

            verbose=1

        ),

        # ----------------------------------------------------
        # Reduce Learning Rate
        # ----------------------------------------------------

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=0.00001,

            verbose=1

        )

    ]

    # --------------------------------------------------------
    # Train Model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

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

    # --------------------------------------------------------
    # Save Final Model
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print("\n" + "=" * 60)
    print("EXP7 TRAINING COMPLETED")
    print("=" * 60)

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nNormalization saved to:"
    )

    print(
        NORMALIZATION_PATH
    )

    print("\n" + "=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()