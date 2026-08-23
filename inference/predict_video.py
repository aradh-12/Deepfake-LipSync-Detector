import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import librosa
import mediapipe as mp
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/deepfake_lipsync_lstm.keras"
)

SCALER_PATH = Path(
    "models/deepfake_lipsync_feature_scaler.npz"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 173

# ------------------------------------------------------------
# IMPORTANT
#
# This threshold should match the threshold selected during
# your threshold-analysis stage.
# ------------------------------------------------------------

THRESHOLD = 0.56

# Audio configuration
SAMPLE_RATE = 16000
N_MFCC = 13


# ============================================================
# LIP LANDMARKS
# ============================================================
#
# 40 landmarks × 2 coordinates
# = 80 lip-coordinate features
#
# These must remain identical to the feature extraction
# pipeline used during training.
# ============================================================

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 185,
    40, 39, 37, 0, 267, 269,
    270, 409, 78, 95, 88, 178,
    87, 14, 317, 402, 318, 324,
    308, 191, 80, 81, 82, 13,
    312, 311, 310, 415
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        print(
            f"❌ Model not found: {MODEL_PATH}"
        )

        print(
            "\nExpected model:"
        )

        print(
            "models/deepfake_lipsync_lstm.keras"
        )

        return None

    print(
        "\nLoading trained model..."
    )

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

    except Exception as error:

        print(
            f"❌ Could not load model: {error}"
        )

        return None

    print(
        "✅ Model loaded successfully."
    )

    print(
        f"Input shape  : {model.input_shape}"
    )

    print(
        f"Output shape : {model.output_shape}"
    )

    return model


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    if not SCALER_PATH.exists():

        print(
            f"❌ Scaler not found: {SCALER_PATH}"
        )

        print(
            "\nRun training first so that the scaler is created."
        )

        return None

    print(
        "\nLoading feature scaler..."
    )

    try:

        data = np.load(
            SCALER_PATH
        )

        mean = data["mean"]
        scale = data["scale"]

    except Exception as error:

        print(
            f"❌ Could not load scaler: {error}"
        )

        return None

    if mean.shape != (FEATURE_SIZE,):

        print(
            "\n❌ Invalid scaler mean shape:"
        )

        print(
            f"Expected : ({FEATURE_SIZE},)"
        )

        print(
            f"Got      : {mean.shape}"
        )

        return None

    if scale.shape != (FEATURE_SIZE,):

        print(
            "\n❌ Invalid scaler scale shape:"
        )

        print(
            f"Expected : ({FEATURE_SIZE},)"
        )

        print(
            f"Got      : {scale.shape}"
        )

        return None

    print(
        "✅ Scaler loaded successfully."
    )

    print(
        f"Scaler features : {len(mean)}"
    )

    return mean.astype(
        np.float32
    ), scale.astype(
        np.float32
    )


# ============================================================
# EXTRACT AUDIO
# ============================================================

def extract_audio(video_path):

    print(
        "\nExtracting audio..."
    )

    audio_path = Path(
        "inference_temp_audio.wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(audio_path),
        "-loglevel",
        "error"
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    except FileNotFoundError:

        print(
            "❌ FFmpeg is not installed "
            "or is not available in PATH."
        )

        return None

    if (
        result.returncode != 0
        or not audio_path.exists()
    ):

        print(
            "❌ Failed to extract audio."
        )

        if result.stderr:

            print(
                result.stderr.decode(
                    errors="ignore"
                )
            )

        return None

    try:

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

    except Exception as error:

        print(
            f"❌ Could not load audio: {error}"
        )

        if audio_path.exists():

            audio_path.unlink()

        return None

    if audio_path.exists():

        audio_path.unlink()

    print(
        f"Audio samples : {len(audio)}"
    )

    print(
        f"Sample rate   : {sr}"
    )

    if len(audio) == 0:

        print(
            "❌ Audio contains no samples."
        )

        return None

    return audio.astype(
        np.float32
    )


# ============================================================
# EXTRACT MFCC
# ============================================================

def extract_mfcc(audio):

    print(
        "\nExtracting MFCC..."
    )

    if audio is None:

        return None

    try:

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC
        )

    except Exception as error:

        print(
            f"❌ MFCC extraction failed: {error}"
        )

        return None

    if mfcc.ndim != 2:

        print(
            f"❌ Invalid MFCC shape: {mfcc.shape}"
        )

        return None

    if mfcc.shape[0] != N_MFCC:

        print(
            f"❌ Expected {N_MFCC} MFCC coefficients."
        )

        print(
            f"Got: {mfcc.shape}"
        )

        return None

    print(
        f"MFCC shape : {mfcc.shape}"
    )

    return mfcc.astype(
        np.float32
    )


# ============================================================
# OPEN VIDEO
# ============================================================

def open_video(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"❌ Unable to open video: {video_path}"
        )

        return None

    return cap


# ============================================================
# NORMALIZE LIP COORDINATES
# ============================================================

def normalize_lip_coordinates(
    coordinates
):
    """
    Normalize lip landmark coordinates.

    Input:
        (40, 2)

    Output:
        (40, 2)

    The normalization is performed relative to the
    bounding box of the detected lip landmarks.

    This makes the feature representation less dependent
    on absolute image position and image dimensions.
    """

    coordinates = coordinates.astype(
        np.float32
    )

    min_xy = coordinates.min(
        axis=0
    )

    max_xy = coordinates.max(
        axis=0
    )

    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]

    if width < 1e-6:

        width = 1.0

    if height < 1e-6:

        height = 1.0

    normalized = coordinates.copy()

    normalized[:, 0] = (
        normalized[:, 0] - min_xy[0]
    ) / width

    normalized[:, 1] = (
        normalized[:, 1] - min_xy[1]
    ) / height

    return normalized.astype(
        np.float32
    )


# ============================================================
# EXTRACT LIP FEATURES
# ============================================================

def extract_lip_features(
    cap,
    face_mesh
):

    print(
        "\nExtracting lip landmarks..."
    )

    lip_features = []

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 25.0

    print(
        f"Video frames : {total_frames}"
    )

    print(
        f"Video FPS    : {fps:.2f}"
    )

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:

            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(
            rgb
        )

        if (
            not results.multi_face_landmarks
        ):

            frame_index += 1

            continue

        face = (
            results.multi_face_landmarks[0]
        )

        coordinates = []

        for idx in LIP_LANDMARKS:

            point = face.landmark[idx]

            # MediaPipe normalized coordinates.
            x = float(point.x)
            y = float(point.y)

            coordinates.append(
                [x, y]
            )

        coordinates = np.asarray(
            coordinates,
            dtype=np.float32
        )

        if coordinates.shape != (
            40,
            2
        ):

            print(
                "❌ Wrong coordinate shape:",
                coordinates.shape
            )

            frame_index += 1

            continue

        # ----------------------------------------------------
        # Normalize coordinates
        # ----------------------------------------------------

        coordinates = (
            normalize_lip_coordinates(
                coordinates
            )
        )

        lip_vector = (
            coordinates.flatten()
        )

        if lip_vector.shape != (
            80,
        ):

            print(
                "❌ Wrong lip feature shape:",
                lip_vector.shape
            )

            frame_index += 1

            continue

        lip_features.append(
            lip_vector
        )

        frame_index += 1

    print(
        f"Lip frames detected : "
        f"{len(lip_features)}"
    )

    if len(lip_features) == 0:

        print(
            "❌ No lip features detected."
        )

        return (
            np.empty(
                (0, 80),
                dtype=np.float32
            ),
            fps
        )

    lip_features = np.asarray(
        lip_features,
        dtype=np.float32
    )

    print(
        f"Lip feature shape : "
        f"{lip_features.shape}"
    )

    return (
        lip_features,
        fps
    )


# ============================================================
# COMPUTE LIP VELOCITY
# ============================================================

def compute_lip_velocity(
    lip_features
):

    if lip_features is None:

        return None

    if len(lip_features) == 0:

        return np.empty(
            (0, 80),
            dtype=np.float32
        )

    velocity = np.zeros_like(
        lip_features,
        dtype=np.float32
    )

    if len(lip_features) > 1:

        velocity[1:] = (
            lip_features[1:]
            -
            lip_features[:-1]
        )

    return velocity


# ============================================================
# SYNCHRONIZE MFCC WITH LIP FRAMES
# ============================================================

def synchronize_features(
    lip_features,
    mfcc
):
    """
    Create the final 173-dimensional feature representation.

    80 lip coordinates
    +
    80 lip velocity
    +
    13 MFCC
    =
    173 features
    """

    if lip_features is None:

        print(
            "❌ Lip features are None."
        )

        return None

    if mfcc is None:

        print(
            "❌ MFCC data is missing."
        )

        return None

    total_video_frames = len(
        lip_features
    )

    total_mfcc_frames = mfcc.shape[1]

    if total_video_frames == 0:

        print(
            "❌ No video frames available."
        )

        return None

    if total_mfcc_frames == 0:

        print(
            "❌ No MFCC frames available."
        )

        return None

    print(
        "\nSynchronizing audio and video..."
    )

    print(
        f"Lip frames  : {total_video_frames}"
    )

    print(
        f"MFCC frames  : {total_mfcc_frames}"
    )

    # --------------------------------------------------------
    # Map MFCC frames to video frames.
    #
    # This matches the synchronization strategy used in
    # the existing inference pipeline.
    # --------------------------------------------------------

    if total_video_frames == 1:

        mfcc_indices = np.array(
            [0],
            dtype=np.int32
        )

    else:

        mfcc_indices = np.linspace(
            0,
            total_mfcc_frames - 1,
            total_video_frames
        ).astype(
            np.int32
        )

    # --------------------------------------------------------
    # Lip velocity
    # --------------------------------------------------------

    lip_velocity = (
        compute_lip_velocity(
            lip_features
        )
    )

    if lip_velocity is None:

        print(
            "❌ Could not compute lip velocity."
        )

        return None

    synchronized = []

    # --------------------------------------------------------
    # Fuse all modalities
    # --------------------------------------------------------

    for index in range(
        total_video_frames
    ):

        lip_vector = (
            lip_features[index]
            .astype(np.float32)
        )

        velocity_vector = (
            lip_velocity[index]
            .astype(np.float32)
        )

        mfcc_vector = (
            mfcc[
                :,
                mfcc_indices[index]
            ].astype(
                np.float32
            )
        )

        feature_vector = np.concatenate(
            (
                lip_vector,
                velocity_vector,
                mfcc_vector
            )
        ).astype(
            np.float32
        )

        if feature_vector.shape != (
            FEATURE_SIZE,
        ):

            print(
                "❌ Wrong fused feature shape:",
                feature_vector.shape
            )

            continue

        synchronized.append(
            feature_vector
        )

    if not synchronized:

        print(
            "❌ No synchronized features created."
        )

        return None

    synchronized = np.asarray(
        synchronized,
        dtype=np.float32
    )

    print(
        f"Synchronized feature shape : "
        f"{synchronized.shape}"
    )

    # --------------------------------------------------------
    # Final safety validation
    # --------------------------------------------------------

    if synchronized.ndim != 2:

        print(
            "❌ Invalid synchronized feature dimensions."
        )

        return None

    if synchronized.shape[1] != FEATURE_SIZE:

        print(
            f"❌ Expected {FEATURE_SIZE} features."
        )

        print(
            f"Got {synchronized.shape[1]}"
        )

        return None

    return synchronized


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(
    features,
    scaler
):
    """
    Apply the training scaler to inference features.

    IMPORTANT:
    No fitting is performed here.

    The mean and scale were learned only from the
    training set.
    """

    if features is None:

        return None

    if scaler is None:

        print(
            "❌ Scaler is missing."
        )

        return None

    mean, scale = scaler

    if features.ndim != 2:

        print(
            f"❌ Expected 2D features."
        )

        print(
            f"Got: {features.shape}"
        )

        return None

    if features.shape[1] != FEATURE_SIZE:

        print(
            f"❌ Expected {FEATURE_SIZE} features."
        )

        print(
            f"Got: {features.shape[1]}"
        )

        return None

    # --------------------------------------------------------
    # Avoid division by zero.
    # --------------------------------------------------------

    safe_scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale
    )

    scaled = (
        features - mean
    ) / safe_scale

    scaled = scaled.astype(
        np.float32
    )

    if np.isnan(scaled).any():

        print(
            "❌ NaN detected after scaling."
        )

        return None

    if np.isinf(scaled).any():

        print(
            "❌ Inf detected after scaling."
        )

        return None

    return scaled


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    features
):
    """
    Create non-overlapping 30-frame sequences.

    This matches training/dataset_builder.py.
    """

    if features is None:

        print(
            "❌ Features are None."
        )

        return None

    if len(features) < SEQUENCE_LENGTH:

        print(
            "\n❌ Not enough frames."
        )

        print(
            f"Available : {len(features)}"
        )

        print(
            f"Required  : {SEQUENCE_LENGTH}"
        )

        return None

    sequences = []

    # --------------------------------------------------------
    # NON-OVERLAPPING sequences
    #
    # 0-29
    # 30-59
    # 60-89
    # ...
    # --------------------------------------------------------

    sequence_count = (
        len(features)
        // SEQUENCE_LENGTH
    )

    usable_length = (
        sequence_count
        * SEQUENCE_LENGTH
    )

    usable_features = features[
        :usable_length
    ]

    for start in range(
        0,
        usable_length,
        SEQUENCE_LENGTH
    ):

        sequence = usable_features[
            start:
            start + SEQUENCE_LENGTH
        ]

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        ):

            continue

        sequences.append(
            sequence
        )

    if not sequences:

        print(
            "❌ No valid sequences created."
        )

        return None

    sequences = np.asarray(
        sequences,
        dtype=np.float32
    )

    print(
        f"\nSequences shape : "
        f"{sequences.shape}"
    )

    return sequences


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    sequences
):

    if model is None:

        return None

    if sequences is None:

        return None

    if len(sequences) == 0:

        return None

    print(
        "\nRunning LSTM prediction..."
    )

    try:

        probabilities = model.predict(
            sequences,
            verbose=1
        ).reshape(
            -1
        )

    except Exception as error:

        print(
            f"❌ Model prediction failed: {error}"
        )

        return None

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Aggregate sequence probabilities
    # --------------------------------------------------------

    fake_probability = float(
        np.mean(
            probabilities
        )
    )

    real_probability = (
        1.0 - fake_probability
    )

    prediction = (
        "FAKE"
        if fake_probability >= THRESHOLD
        else "REAL"
    )

    return (
        fake_probability,
        real_probability,
        prediction,
        probabilities
    )


# ============================================================
# MAIN VIDEO PIPELINE
# ============================================================

def predict_video(
    video_path
):

    video_path = Path(
        video_path
    )

    # --------------------------------------------------------
    # Validate video
    # --------------------------------------------------------

    if not video_path.exists():

        print(
            f"❌ Video not found: {video_path}"
        )

        return

    if not video_path.is_file():

        print(
            f"❌ Path is not a file: {video_path}"
        )

        return

    print(
        "\n========================================"
    )

    print(
        "DEEPFAKE LIP-SYNC DETECTION"
    )

    print(
        "========================================"
    )

    print(
        f"Video : {video_path}"
    )

    print(
        f"Feature size : {FEATURE_SIZE}"
    )

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    # ========================================================
    # STEP 1 — LOAD MODEL
    # ========================================================

    model = load_model()

    if model is None:

        return

    # ========================================================
    # STEP 2 — LOAD SCALER
    # ========================================================

    scaler = load_scaler()

    if scaler is None:

        return

    # ========================================================
    # STEP 3 — EXTRACT AUDIO
    # ========================================================

    audio = extract_audio(
        video_path
    )

    if audio is None:

        return

    # ========================================================
    # STEP 4 — MFCC
    # ========================================================

    mfcc = extract_mfcc(
        audio
    )

    if mfcc is None:

        return

    # ========================================================
    # STEP 5 — OPEN VIDEO
    # ========================================================

    cap = open_video(
        video_path
    )

    if cap is None:

        return

    # ========================================================
    # STEP 6 — MEDIAPIPE FACE MESH
    # ========================================================

    print(
        "\nInitializing MediaPipe Face Mesh..."
    )

    try:

        mp_face_mesh = (
            mp.solutions.face_mesh
        )

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            (
                lip_features,
                fps
            ) = extract_lip_features(
                cap,
                face_mesh
            )

    except Exception as error:

        print(
            f"❌ MediaPipe processing failed: {error}"
        )

        cap.release()

        return

    cap.release()

    # ========================================================
    # STEP 7 — SYNCHRONIZE FEATURES
    # ========================================================

    features = synchronize_features(
        lip_features,
        mfcc
    )

    if features is None:

        return

    print(
        f"\nRaw fused features : "
        f"{features.shape}"
    )

    # ========================================================
    # STEP 8 — SCALE FEATURES
    # ========================================================

    features = scale_features(
        features,
        scaler
    )

    if features is None:

        return

    print(
        f"Scaled features : "
        f"{features.shape}"
    )

    # ========================================================
    # STEP 9 — CREATE 30-FRAME SEQUENCES
    # ========================================================

    sequences = create_sequences(
        features
    )

    if sequences is None:

        return

    # ========================================================
    # STEP 10 — MODEL PREDICTION
    # ========================================================

    result = predict(
        model,
        sequences
    )

    if result is None:

        return

    (
        fake_probability,
        real_probability,
        prediction,
        probabilities
    ) = result

    # ========================================================
    # STEP 11 — RESULT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"Video              : "
        f"{video_path.name}"
    )

    print(
        f"Sequences analyzed : "
        f"{len(sequences)}"
    )

    print(
        f"Fake Probability   : "
        f"{fake_probability:.4f}"
    )

    print(
        f"Fake Probability   : "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"Real Probability   : "
        f"{real_probability:.4f}"
    )

    print(
        f"Real Probability   : "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"Threshold          : "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Prediction         : "
        f"{prediction}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # STEP 12 — PER-SEQUENCE RESULTS
    # ========================================================

    print(
        "\nSequence probabilities:"
    )

    for i, probability in enumerate(
        probabilities,
        start=1
    ):

        sequence_prediction = (
            "FAKE"
            if probability >= THRESHOLD
            else "REAL"
        )

        print(
            f"Sequence {i:02d} : "
            f"{probability:.4f} "
            f"({probability * 100:.2f}%) "
            f"-> {sequence_prediction}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    fake_sequences = int(
        np.sum(
            probabilities >= THRESHOLD
        )
    )

    real_sequences = (
        len(probabilities)
        - fake_sequences
    )

    print(
        "\n========================================"
    )

    print(
        "SEQUENCE SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Fake sequences : "
        f"{fake_sequences}"
    )

    print(
        f"Real sequences : "
        f"{real_sequences}"
    )

    print(
        f"Total sequences: "
        f"{len(probabilities)}"
    )

    print(
        "========================================"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python -m inference.predict_video "<video_path>"'
        )

        print(
            "\nExample:"
        )

        print(
            'python -m inference.predict_video "test.mp4"'
        )

        sys.exit(1)

    predict_video(
        sys.argv[1]
    )