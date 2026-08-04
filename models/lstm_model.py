import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)


def build_lstm_model(
    sequence_length=30,
    feature_size=93
):
    """
    Build LSTM model for deepfake detection.
    """

    model = Sequential([

        Input(shape=(sequence_length, feature_size)),

        LSTM(
            128,
            return_sequences=True
        ),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(
            64,
            return_sequences=False
        ),
        BatchNormalization(),
        Dropout(0.3),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            1,
            activation="sigmoid"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


if __name__ == "__main__":

    model = build_lstm_model()

    model.summary()