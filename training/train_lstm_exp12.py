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


MODEL_PATH = (
    MODEL_DIR /
    "lstm_exp12.keras"
)


NORMALIZATION_PATH = (
    MODEL_DIR /
    "lstm_exp12_normalization.npz"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 60

BATCH_SIZE = 16

LEARNING_RATE = 0.0005

RANDOM_SEED = 42


# ============================================================
# SET RANDOM SEEDS
# ============================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split_name):

    file_path = (
        DATASET_DIR /
        f"{split_name}.npz"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )


    data = np.load(
        file_path,
        allow_pickle=True
    )


    X = data["X"].astype(
        np.float32
    )


    y = data["y"].astype(
        np.int32
    )


    video_ids = data[
        "video_ids"
    ]


    return (
        X,
        y,
        video_ids
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_training_data(X_train):

    print(
        "\nCalculating normalization statistics..."
    )


    # Calculate statistics ONLY from training data

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


    # Prevent division by zero

    std = np.maximum(

        std,

        1e-8

    )


    # Normalize training data

    X_train_normalized = (

        X_train - mean

    ) / std


    return (

        X_train_normalized.astype(
            np.float32
        ),

        mean,

        std

    )


# ============================================================
# NORMALIZE OTHER DATA
# ============================================================

def normalize_data(

    X,

    mean,

    std

):

    X_normalized = (

        X - mean

    ) / std


    return X_normalized.astype(
        np.float32
    )


# ============================================================
# TEMPORAL ATTENTION LAYER
# ============================================================

class TemporalAttention(
    tf.keras.layers.Layer
):

    def __init__(

        self,

        **kwargs

    ):

        super().__init__(
            **kwargs
        )


    def build(

        self,

        input_shape

    ):

        feature_size = input_shape[-1]


        self.attention_dense = (
            tf.keras.layers.Dense(
                1
            )
        )


        super().build(
            input_shape
        )


    def call(

        self,

        inputs

    ):

        # ----------------------------------------------------
        # inputs shape:
        #
        # (batch_size, time_steps, features)
        # ----------------------------------------------------


        # Calculate attention scores

        attention_scores = (
            self.attention_dense(
                inputs
            )
        )


        # Convert scores into probabilities

        attention_weights = (
            tf.nn.softmax(

                attention_scores,

                axis=1

            )
        )


        # Apply attention weights

        weighted_inputs = (

            inputs *

            attention_weights

        )


        # Sum across temporal dimension

        context_vector = tf.reduce_sum(

            weighted_inputs,

            axis=1

        )


        return context_vector


# ============================================================
# BUILD EXP12 MODEL
#
# BiLSTM + TEMPORAL ATTENTION
# ============================================================

def build_model(

    sequence_length,

    feature_size

):


    # --------------------------------------------------------
    # L2 REGULARIZATION
    # --------------------------------------------------------

    regularizer = (

        tf.keras.regularizers.l2(

            0.0005

        )

    )


    # ========================================================
    # INPUT
    # ========================================================

    inputs = tf.keras.layers.Input(

        shape=(

            sequence_length,

            feature_size

        )

    )


    # ========================================================
    # FIRST BIDIRECTIONAL LSTM
    # ========================================================

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            48,

            return_sequences=True,

            dropout=0.30,

            recurrent_dropout=0.0,

            kernel_regularizer=regularizer,

            recurrent_regularizer=regularizer

        )

    )(inputs)


    # ========================================================
    # SECOND BIDIRECTIONAL LSTM
    # ========================================================

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            24,

            return_sequences=True,

            dropout=0.30,

            recurrent_dropout=0.0,

            kernel_regularizer=regularizer,

            recurrent_regularizer=regularizer

        )

    )(x)


    # ========================================================
    # TEMPORAL ATTENTION
    #
    # Learns which temporal regions are important
    # ========================================================

    x = TemporalAttention()(

        x

    )


    # ========================================================
    # DENSE LAYER
    # ========================================================

    x = tf.keras.layers.Dense(

        16,

        activation="relu",

        kernel_regularizer=regularizer

    )(x)


    # ========================================================
    # DROPOUT
    # ========================================================

    x = tf.keras.layers.Dropout(

        0.40

    )(x)


    # ========================================================
    # OUTPUT
    #
    # 0 = Real
    # 1 = Fake
    # ========================================================

    outputs = tf.keras.layers.Dense(

        1,

        activation="sigmoid"

    )(x)


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = tf.keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="EXP12_BiLSTM_TemporalAttention"

    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = (

        tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE

        )

    )


    # ========================================================
    # COMPILE MODEL
    # ========================================================

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

            ),

            tf.keras.metrics.AUC(

                name="auc",

                curve="ROC"

            )

        ]

    )


    return model


# ============================================================
# MAIN
# ============================================================

def main():


    print(
        "\n" + "=" * 60
    )

    print(
        "EXP12 BiLSTM + TEMPORAL ATTENTION TRAINING"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    print(
        "\nLoading training dataset..."
    )


    (

        X_train,

        y_train,

        _

    ) = load_dataset(

        "train"

    )


    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print(
        "\nLoading validation dataset..."
    )


    (

        X_validation,

        y_validation,

        _

    ) = load_dataset(

        "validation"

    )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "DATASET INFORMATION"
    )

    print(
        "=" * 60
    )


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


    print(
        "\nTraining distribution:"
    )

    print(
        "Real:",
        np.sum(y_train == 0)
    )

    print(
        "Fake:",
        np.sum(y_train == 1)
    )


    print(
        "\nValidation distribution:"
    )

    print(
        "Real:",
        np.sum(y_validation == 0)
    )

    print(
        "Fake:",
        np.sum(y_validation == 1)
    )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "NORMALIZATION"
    )

    print(
        "=" * 60
    )


    (

        X_train,

        mean,

        std

    ) = normalize_training_data(

        X_train

    )


    X_validation = normalize_data(

        X_validation,

        mean,

        std

    )


    # ========================================================
    # SAVE NORMALIZATION STATISTICS
    # ========================================================

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


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CLASS WEIGHTS"
    )

    print(
        "=" * 60
    )


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

        for label, weight

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


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    sequence_length = (

        X_train.shape[1]

    )


    feature_size = (

        X_train.shape[2]

    )


    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING EXP12 MODEL"
    )

    print(
        "=" * 60
    )


    print(
        "\nSequence length:",
        sequence_length
    )

    print(
        "Feature size:",
        feature_size
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model(

        sequence_length,

        feature_size

    )


    print(
        "\nModel Summary:"
    )

    model.summary()


    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = [


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        tf.keras.callbacks.ModelCheckpoint(

            MODEL_PATH,

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        ),


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=12,

            restore_best_weights=True,

            verbose=1

        ),


        # ----------------------------------------------------
        # REDUCE LEARNING RATE
        # ----------------------------------------------------

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=1e-6,

            verbose=1

        )

    ]


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING EXP12 MODEL"
    )

    print(
        "=" * 60
    )


    model.fit(

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
    # SAVE FINAL MODEL
    # ========================================================

    model.save(

        MODEL_PATH

    )


    # ========================================================
    # COMPLETED
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "EXP12 TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )


    print(
        "\nModel saved:"
    )

    print(
        MODEL_PATH
    )


    print(
        "\nNormalization saved:"
    )

    print(
        NORMALIZATION_PATH
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
