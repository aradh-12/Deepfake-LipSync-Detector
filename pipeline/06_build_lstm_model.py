from pathlib import Path

import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "models"
)

MODEL_PATH = (
    MODEL_OUTPUT_DIR
    / "lstm_multimodal_model.keras"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

TIME_STEPS = 150

FEATURE_DIMENSION = 119

LEARNING_RATE = 0.001


# ============================================================
# BUILD MODEL
# ============================================================

def build_lstm_model():

    model = tf.keras.Sequential(

        [

            # ------------------------------------------------
            # Input
            #
            # Shape:
            # (150, 119)
            # ------------------------------------------------

            tf.keras.layers.Input(

                shape=(
                    TIME_STEPS,
                    FEATURE_DIMENSION
                )
            ),

            # ------------------------------------------------
            # Ignore zero padding
            # ------------------------------------------------

            tf.keras.layers.Masking(

                mask_value=0.0
            ),

            # ------------------------------------------------
            # First BiLSTM layer
            # ------------------------------------------------

            tf.keras.layers.Bidirectional(

                tf.keras.layers.LSTM(

                    64,

                    return_sequences=True
                )
            ),

            # ------------------------------------------------
            # Dropout
            # ------------------------------------------------

            tf.keras.layers.Dropout(

                0.3
            ),

            # ------------------------------------------------
            # Second BiLSTM layer
            # ------------------------------------------------

            tf.keras.layers.Bidirectional(

                tf.keras.layers.LSTM(

                    32
                )
            ),

            # ------------------------------------------------
            # Dense layer
            # ------------------------------------------------

            tf.keras.layers.Dense(

                32,

                activation="relu"
            ),

            # ------------------------------------------------
            # Dropout
            # ------------------------------------------------

            tf.keras.layers.Dropout(

                0.3
            ),

            # ------------------------------------------------
            # Output layer
            #
            # Real = 0
            # Fake = 1
            # ------------------------------------------------

            tf.keras.layers.Dense(

                1,

                activation="sigmoid"
            )

        ]
    )


    # ========================================================
    # COMPILE MODEL
    # ========================================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[

            "accuracy"
        ]
    )


    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(

        "=" * 70
    )

    print(

        "MULTIMODAL LSTM MODEL"
    )

    print(

        "=" * 70
    )

    print()


    # --------------------------------------------------------
    # Create model output directory
    # --------------------------------------------------------

    MODEL_OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True
    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = (

        build_lstm_model()
    )


    # --------------------------------------------------------
    # Display model summary
    # --------------------------------------------------------

    print(

        "MODEL ARCHITECTURE"
    )

    print()

    model.summary()


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model.save(

        MODEL_PATH
    )


    print()

    print(

        "=" * 70
    )

    print(

        "MODEL BUILT SUCCESSFULLY"
    )

    print(

        "=" * 70
    )

    print()

    print(

        "Input shape:"
    )

    print(

        f"(batch_size, "
        f"{TIME_STEPS}, "
        f"{FEATURE_DIMENSION})"
    )

    print()

    print(

        "Output:"
    )

    print(

        "Binary classification"
    )

    print(

        "0 = REAL"
    )

    print(

        "1 = FAKE"
    )

    print()

    print(

        "Model saved to:"
    )

    print(

        MODEL_PATH
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()