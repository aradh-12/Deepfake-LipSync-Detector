from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import (
    compute_class_weight
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp8_temporal"
)

MODEL_DIR = Path(
    "saved_models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / (
    "lstm_exp8.keras"
)

NORMALIZATION_PATH = MODEL_DIR / (
    "lstm_exp8_normalization.npz"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(split_name):

    file_path = DATASET_DIR / (
        f"{split_name}.npz"
    )

    data = np.load(
        file_path,
        allow_pickle=True
    )

    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int32),
        data["video_ids"]
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_training_data(X_train):

    print("\nCalculating normalization statistics...")

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

    std = np.maximum(
        std,
        1e-8
    )

    print("\nSaving normalization statistics...")

    np.savez(
        NORMALIZATION_PATH,
        mean=mean,
        std=std
    )

    X_train = (
        X_train - mean
    ) / std

    return X_train, mean, std


def normalize_data(
    X,
    mean,
    std
):

    return (
        X - mean
    ) / std


# ============================================================
# BUILD MODEL
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

        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.20,
            recurrent_dropout=0.0
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.20,
            recurrent_dropout=0.0
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),

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
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("EXP8 TEMPORAL LSTM TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    X_train, y_train, _ = load_dataset(
        "train"
    )

    print("\nLoading validation dataset...")

    X_val, y_val, _ = load_dataset(
        "validation"
    )

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("\nTraining X shape:", X_train.shape)
    print("Training y shape:", y_train.shape)

    print("\nValidation X shape:", X_val.shape)
    print("Validation y shape:", y_val.shape)

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
        np.sum(y_val == 0)
    )
    print(
        "Fake:",
        np.sum(y_val == 1)
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    X_train, mean, std = (
        normalize_training_data(
            X_train
        )
    )

    X_val = normalize_data(
        X_val,
        mean,
        std
    )

    print(
        "\nNormalization saved:"
    )
    print(
        NORMALIZATION_PATH
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
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

        int(label): float(weight)

        for label, weight in zip(
            classes,
            weights
        )

    }

    print("\nClass weights:")
    print(class_weights)

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING LSTM MODEL")
    print("=" * 60)

    sequence_length = (
        X_train.shape[1]
    )

    feature_size = (
        X_train.shape[2]
    )

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
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            MODEL_PATH,

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=10,

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

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=50,

        batch_size=16,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1

    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print("\n" + "=" * 60)
    print(
        "EXP8 TRAINING COMPLETED"
    )
    print("=" * 60)

    print("\nModel saved to:")
    print(
        MODEL_PATH
    )

    print("\nNormalization saved to:")
    print(
        NORMALIZATION_PATH
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":

    main()
