import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization
)


SEQUENCE_LENGTH = 30
FEATURE_SIZE = 199


def build_lstm_model(
    sequence_length=SEQUENCE_LENGTH,
    feature_size=FEATURE_SIZE
):
    """
    Robust LSTM model for multimodal lip-sync
    anomaly detection.

    Input:
        (30, 173)

    Features:
        80 lip positions
        80 lip velocities
        13 MFCC features

    Experiment 3:
        Smaller model with stronger regularization
        to improve generalization.
    """

    model = Sequential([

        Input(
            shape=(
                sequence_length,
                feature_size
            )
        ),

        LSTM(
            32,
            return_sequences=False,
            dropout=0.30,
            recurrent_dropout=0.15
        ),

        LayerNormalization(),

        Dense(
            16,
            activation="relu"
        ),

        Dropout(0.40),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model


if __name__ == "__main__":

    model = build_lstm_model()

    model.summary()
