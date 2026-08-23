import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)


SEQUENCE_LENGTH = 30
FEATURE_SIZE = 173


def build_lstm_model(
    sequence_length=SEQUENCE_LENGTH,
    feature_size=FEATURE_SIZE
):
    """
    Regularized LSTM for multimodal lip-sync
    anomaly detection.

    Input:
        (30, 173)

    Features:
        80 lip positions
        80 lip velocities
        13 MFCC features

    Model V2:
        Stronger regularization to reduce overfitting.
    """

    model = Sequential([

        Input(
            shape=(
                sequence_length,
                feature_size
            )
        ),

        LSTM(
            64,
            return_sequences=False,
            dropout=0.25,
            recurrent_dropout=0.10
        ),

        BatchNormalization(),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.35),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002
        ),

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
                name="auc"
            )
        ]
    )

    return model


if __name__ == "__main__":

    model = build_lstm_model()

    model.summary()
