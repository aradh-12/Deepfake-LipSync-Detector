import os
import json
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from tensorflow.keras import layers, models, callbacks, regularizers


# ============================================================
# EXPERIMENT 4
# ============================================================

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 173

RANDOM_STATE = 42

EPOCHS = 80
BATCH_SIZE = 16

DATASET_DIR = "outputs/dataset"

MODEL_PATH = "models/deepfake_lipsync_lstm_exp4.keras"
SCALER_PATH = "models/deepfake_lipsync_feature_scaler_exp4.npz"
HISTORY_PATH = "outputs/training/training_history_exp4.json"
METRICS_PATH = "outputs/training/test_metrics_exp4.json"


# ============================================================
# Reproducibility
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Load split
# ============================================================

def load_split(filename):

    path = os.path.join(
        DATASET_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True
    )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    video_ids = data["video_ids"]

    return X, y, video_ids


# ============================================================
# Validate split
# ============================================================

def validate_data(
    name,
    X,
    y,
    video_ids
):

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("X shape       :", X.shape)
    print("y shape       :", y.shape)
    print("Unique videos :", len(np.unique(video_ids)))
    print("Real          :", int(np.sum(y == 0)))
    print("Fake          :", int(np.sum(y == 1)))

    if X.ndim != 3:
        raise ValueError(
            f"{name}: expected 3D X, got {X.shape}"
        )

    if X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"{name}: expected {SEQUENCE_LENGTH} frames, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != FEATURE_SIZE:
        raise ValueError(
            f"{name}: expected {FEATURE_SIZE} features, "
            f"got {X.shape[2]}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name}: X/y mismatch."
        )

    if len(X) != len(video_ids):
        raise ValueError(
            f"{name}: X/video_ids mismatch."
        )

    if np.isnan(X).any():
        raise ValueError(
            f"{name}: NaN detected."
        )

    if np.isinf(X).any():
        raise ValueError(
            f"{name}: Inf detected."
        )


# ============================================================
# Check video leakage
# ============================================================

def check_video_leakage(
    train_ids,
    validation_ids,
    test_ids
):

    train_set = set(train_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    print("\n" + "=" * 55)
    print("VIDEO LEAKAGE CHECK")
    print("=" * 55)

    print(
        "Train ∩ Validation :",
        len(train_set & validation_set)
    )

    print(
        "Train ∩ Test       :",
        len(train_set & test_set)
    )

    print(
        "Validation ∩ Test  :",
        len(validation_set & test_set)
    )

    if train_set & validation_set:
        raise RuntimeError(
            "DATA LEAKAGE detected: train/validation."
        )

    if train_set & test_set:
        raise RuntimeError(
            "DATA LEAKAGE detected: train/test."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "DATA LEAKAGE detected: validation/test."
        )

    print("Data leakage check : PASSED")


# ============================================================
# Feature scaling
# ============================================================

def scale_features(
    X_train,
    X_validation,
    X_test
):

    scaler = StandardScaler()

    train_shape = X_train.shape
    validation_shape = X_validation.shape
    test_shape = X_test.shape

    X_train_flat = X_train.reshape(
        -1,
        FEATURE_SIZE
    )

    X_validation_flat = X_validation.reshape(
        -1,
        FEATURE_SIZE
    )

    X_test_flat = X_test.reshape(
        -1,
        FEATURE_SIZE
    )

    # IMPORTANT:
    # Fit ONLY on training data.

    X_train_flat = scaler.fit_transform(
        X_train_flat
    )

    X_validation_flat = scaler.transform(
        X_validation_flat
    )

    X_test_flat = scaler.transform(
        X_test_flat
    )

    X_train = X_train_flat.reshape(
        train_shape
    ).astype(np.float32)

    X_validation = X_validation_flat.reshape(
        validation_shape
    ).astype(np.float32)

    X_test = X_test_flat.reshape(
        test_shape
    ).astype(np.float32)

    return (
        X_train,
        X_validation,
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
        "\nScaler saved:",
        SCALER_PATH
    )


# ============================================================
# Build Exp4 LSTM
# ============================================================

def build_exp4_model():

    inputs = layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        ),
        name="multimodal_sequence"
    )

    x = layers.LayerNormalization(
        name="input_normalization"
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.25,
            recurrent_dropout=0.0
        ),
        name="bilstm_1"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_1"
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=False,
            dropout=0.25,
            recurrent_dropout=0.0
        ),
        name="bilstm_2"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_2"
    )(x)

    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_features"
    )(x)

    x = layers.Dropout(
        0.25,
        name="dropout_3"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="fake_probability"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="DeepfakeLipSync_Exp4"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-5
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
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
    print("DEEPFAKE LIP-SYNC DETECTOR — EXPERIMENT 4")
    print("=" * 60)

    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------

    X_train, y_train, ids_train = load_split(
        "train.npz"
    )

    X_validation, y_validation, ids_validation = load_split(
        "validation.npz"
    )

    X_test, y_test, ids_test = load_split(
        "test.npz"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_data(
        "TRAIN",
        X_train,
        y_train,
        ids_train
    )

    validate_data(
        "VALIDATION",
        X_validation,
        y_validation,
        ids_validation
    )

    validate_data(
        "TEST",
        X_test,
        y_test,
        ids_test
    )

    check_video_leakage(
        ids_train,
        ids_validation,
        ids_test
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("FEATURE SCALING")
    print("=" * 55)

    (
        X_train,
        X_validation,
        X_test,
        scaler
    ) = scale_features(
        X_train,
        X_validation,
        X_test
    )

    save_scaler(
        scaler
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight
        in zip(classes, weights)
    }

    print(
        "\nClass weights:",
        class_weights
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_exp4_model()

    print("\n" + "=" * 55)
    print("MODEL")
    print("=" * 55)

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "outputs/training",
        exist_ok=True
    )

    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("TRAINING")
    print("=" * 55)

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
        shuffle=True,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    with open(
        HISTORY_PATH,
        "w"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=2
        )

    print(
        "\nTraining history saved:",
        HISTORY_PATH
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    if os.path.exists(
        MODEL_PATH
    ):

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    probabilities = (
        model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .ravel()
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int32
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    except ValueError:

        auc = None

    cm = confusion_matrix(
        y_test,
        predictions
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": (
            float(auc)
            if auc is not None
            else None
        ),
        "confusion_matrix": cm.tolist(),
        "test_sequences": int(len(y_test)),
        "test_videos": int(len(np.unique(ids_test))),
        "threshold": 0.5
    }

    with open(
        METRICS_PATH,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXP4 TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    if auc is not None:

        print(
            f"AUC       : {auc:.4f}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )

    print(
        "\nModel saved:",
        MODEL_PATH
    )

    print(
        "Metrics saved:",
        METRICS_PATH
    )

    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
