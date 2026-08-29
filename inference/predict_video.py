import sys
import json
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

THRESHOLD_PATH = Path(
    "models/threshold.json"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 173

SAMPLE_RATE = 16000
N_MFCC = 13

# These must match the training MFCC configuration.
MFCC_HOP_LENGTH = 512
MFCC_N_FFT = 2048


# ============================================================
# LIP LANDMARKS
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

    # --------------------------------------------------------
    # Safety validation
    # --------------------------------------------------------

    expected_input = (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    )

    actual_input = (
        model.input_shape[1],
        model.input_shape[2]
    )

    if actual_input != expected_input:

        print(
            "\n❌ Model input shape does not match pipeline."
        )

        print(
            f"Expected : (None, {SEQUENCE_LENGTH}, {FEATURE_SIZE})"
        )

        print(
            f"Got      : {model.input_shape}"
        )

        return None

    return model


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    if not SCALER_PATH.exists():

        print(
            f"❌ Scaler not found: {SCALER_PATH}"
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

    if mean.shape != (
        FEATURE_SIZE,
    ):

        print(
            "\n❌ Invalid scaler mean shape."
        )

        print(
            f"Expected : ({FEATURE_SIZE},)"
        )

        print(
            f"Got      : {mean.shape}"
        )

        return None

    if scale.shape != (
        FEATURE_SIZE,
    ):

        print(
            "\n❌ Invalid scaler scale shape."
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

    return (
        mean.astype(np.float32),
        scale.astype(np.float32)
    )


# ============================================================
# LOAD VALIDATION THRESHOLD
# ============================================================

def load_threshold():

    if not THRESHOLD_PATH.exists():

        print(
            f"❌ Threshold file not found: "
            f"{THRESHOLD_PATH}"
        )

        print(
            "Run validation threshold analysis first."
        )

        return None

    print(
        "\nLoading validation threshold..."
    )

    try:

        with open(
            THRESHOLD_PATH,
            "r"
        ) as file:

            data = json.load(
                file
            )

    except Exception as error:

        print(
            f"❌ Could not read threshold file: "
            f"{error}"
        )

        return None

    if "threshold" not in data:

        print(
            "❌ Threshold field missing."
        )

        return None

    try:

        threshold = float(
            data["threshold"]
        )

    except Exception:

        print(
            "❌ Invalid threshold value."
        )

        return None

    if not 0.0 <= threshold <= 1.0:

        print(
            f"❌ Threshold must be between 0 and 1."
        )

        print(
            f"Got: {threshold}"
        )

        return None

    print(
        f"✅ Threshold loaded: {threshold:.2f}"
    )

    if "selection_metric" in data:

        print(
            f"Selection metric : "
            f"{data['selection_metric']}"
        )

    if "validation_f1" in data:

        print(
            f"Validation F1    : "
            f"{data['validation_f1']}"
        )

    return threshold


# ============================================================
# EXTRACT AUDIO
# ============================================================

def extract_audio(
    video_path
):

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

def extract_mfcc(
    audio
):

    print(
        "\nExtracting MFCC..."
    )

    if audio is None:

        return None

    try:

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=MFCC_N_FFT,
            hop_length=MFCC_HOP_LENGTH
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

    # ========================================================
    # MFCC NORMALIZATION
    #
    # This matches the training-side MFCC normalization.
    # Each coefficient is normalized independently.
    # ========================================================

    mfcc_mean = np.mean(
        mfcc,
        axis=1,
        keepdims=True
    )

    mfcc_std = np.std(
        mfcc,
        axis=1,
        keepdims=True
    )

    mfcc_std = np.where(
        mfcc_std < 1e-8,
        1.0,
        mfcc_std
    )

    mfcc = (
        mfcc
        -
        mfcc_mean
    ) / mfcc_std

    mfcc = mfcc.astype(
        np.float32
    )

    if not np.isfinite(
        mfcc
    ).all():

        print(
            "❌ MFCC contains NaN or Inf."
        )

        return None

    print(
        f"MFCC shape : {mfcc.shape}"
    )

    print(
        "MFCC normalization : ✅"
    )

    return mfcc


# ============================================================
# OPEN VIDEO
# ============================================================

def open_video(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"❌ Unable to open video: "
            f"{video_path}"
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

    width = (
        max_xy[0]
        -
        min_xy[0]
    )

    height = (
        max_xy[1]
        -
        min_xy[1]
    )

    if width < 1e-6:

        width = 1.0

    if height < 1e-6:

        height = 1.0

    normalized = coordinates.copy()

    normalized[:, 0] = (
        normalized[:, 0]
        -
        min_xy[0]
    ) / width

    normalized[:, 1] = (
        normalized[:, 1]
        -
        min_xy[1]
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
    """
    Extract one lip feature vector for EVERY original
    video frame.

    Missing MediaPipe detections are interpolated.

    This preserves the original temporal video timeline.

    Output:
        (total_video_frames, 80)
    """

    print(
        "\nExtracting lip landmarks..."
    )

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

    # --------------------------------------------------------
    # One slot per ORIGINAL video frame.
    # --------------------------------------------------------

    frame_features = [
        None
        for _ in range(total_frames)
    ]

    frame_index = 0
    detected_count = 0

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

        if not results.multi_face_landmarks:

            frame_index += 1

            continue

        face = (
            results.multi_face_landmarks[0]
        )

        coordinates = []

        for idx in LIP_LANDMARKS:

            point = face.landmark[idx]

            coordinates.append(
                [
                    float(point.x),
                    float(point.y)
                ]
            )

        coordinates = np.asarray(
            coordinates,
            dtype=np.float32
        )

        if coordinates.shape != (
            40,
            2
        ):

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

            frame_index += 1

            continue

        if frame_index < total_frames:

            frame_features[
                frame_index
            ] = lip_vector

            detected_count += 1

        frame_index += 1

    # ========================================================
    # DETECTION STATISTICS
    # ========================================================

    print(
        f"Lip frames detected : "
        f"{detected_count}"
    )

    missing_count = (
        total_frames
        -
        detected_count
    )

    print(
        f"Lip frames missing  : "
        f"{missing_count}"
    )

    if detected_count == 0:

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

    # ========================================================
    # CREATE ARRAY
    # ========================================================

    features = np.full(
        (
            total_frames,
            80
        ),
        np.nan,
        dtype=np.float32
    )

    for i, feature in enumerate(
        frame_features
    ):

        if feature is not None:

            features[i] = feature

    # ========================================================
    # FIND VALID FRAMES
    # ========================================================

    valid_indices = np.where(
        np.isfinite(
            features
        ).all(
            axis=1
        )
    )[0]

    if len(valid_indices) == 0:

        print(
            "❌ No valid lip features available."
        )

        return (
            np.empty(
                (0, 80),
                dtype=np.float32
            ),
            fps
        )

    # ========================================================
    # INTERPOLATE MISSING FRAMES
    # ========================================================

    print(
        "\nInterpolating missing lip frames..."
    )

    all_indices = np.arange(
        total_frames
    )

    for feature_index in range(
        80
    ):

        values = features[
            valid_indices,
            feature_index
        ]

        features[
            :,
            feature_index
        ] = np.interp(
            all_indices,
            valid_indices,
            values
        )

    features = features.astype(
        np.float32
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if features.shape != (
        total_frames,
        80
    ):

        print(
            "❌ Wrong final lip feature shape:",
            features.shape
        )

        return (
            np.empty(
                (0, 80),
                dtype=np.float32
            ),
            fps
        )

    if not np.isfinite(
        features
    ).all():

        print(
            "❌ Lip features contain NaN or Inf."
        )

        return (
            np.empty(
                (0, 80),
                dtype=np.float32
            ),
            fps
        )

    print(
        f"Lip frames after interpolation : "
        f"{len(features)}"
    )

    print(
        f"Lip feature shape : "
        f"{features.shape}"
    )

    return (
        features,
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
    mfcc,
    fps
):
    """
    Create final 173-dimensional features.

    80 lip coordinates
    +
    80 lip velocity
    +
    13 MFCC
    =
    173 features

    Audio/video synchronization is timestamp based.
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

    if fps <= 0:

        print(
            f"❌ Invalid FPS: {fps}"
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
        f"Lip frames   : "
        f"{total_video_frames}"
    )

    print(
        f"MFCC frames  : "
        f"{total_mfcc_frames}"
    )

    print(
        f"Video FPS    : "
        f"{fps:.4f}"
    )

    # ========================================================
    # VIDEO TIMESTAMPS
    # ========================================================

    video_times = (
        np.arange(
            total_video_frames,
            dtype=np.float32
        )
        / float(fps)
    )

    # ========================================================
    # MFCC TIMESTAMPS
    # ========================================================

    mfcc_times = (
        (
            np.arange(
                total_mfcc_frames,
                dtype=np.float32
            )
            *
            MFCC_HOP_LENGTH
            +
            MFCC_N_FFT / 2
        )
        /
        SAMPLE_RATE
    )

    # ========================================================
    # FIND NEAREST MFCC FRAME
    # ========================================================

    indices = np.searchsorted(
        mfcc_times,
        video_times,
        side="left"
    )

    indices = np.clip(
        indices,
        0,
        total_mfcc_frames - 1
    )

    previous_indices = np.maximum(
        indices - 1,
        0
    )

    next_distance = np.abs(
        mfcc_times[indices]
        -
        video_times
    )

    previous_distance = np.abs(
        mfcc_times[previous_indices]
        -
        video_times
    )

    use_previous = (
        previous_distance
        <=
        next_distance
    )

    indices[
        use_previous
    ] = previous_indices[
        use_previous
    ]

    # ========================================================
    # ALIGNED MFCC
    # ========================================================

    aligned_mfcc = (
        mfcc[:, indices]
    ).T.astype(
        np.float32
    )

    if aligned_mfcc.shape != (
        total_video_frames,
        N_MFCC
    ):

        print(
            "❌ Wrong aligned MFCC shape:",
            aligned_mfcc.shape
        )

        return None

    # ========================================================
    # LIP VELOCITY
    # ========================================================

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

    # ========================================================
    # FUSE MODALITIES
    # ========================================================

    synchronized = []

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
            aligned_mfcc[index]
            .astype(np.float32)
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

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

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

    if not np.isfinite(
        synchronized
    ).all():

        print(
            "❌ Synchronized features contain NaN or Inf."
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
    Apply training scaler.

    IMPORTANT:
    No fitting is performed during inference.
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
            "❌ Expected 2D features."
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

    safe_scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale
    )

    scaled = (
        features
        -
        mean
    ) / safe_scale

    scaled = scaled.astype(
        np.float32
    )

    if not np.isfinite(
        scaled
    ).all():

        print(
            "❌ NaN or Inf detected after scaling."
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

    Matches the training pipeline.
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

    sequence_count = (
        len(features)
        //
        SEQUENCE_LENGTH
    )

    usable_length = (
        sequence_count
        *
        SEQUENCE_LENGTH
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

    if not np.isfinite(
        sequences
    ).all():

        print(
            "❌ Sequences contain NaN or Inf."
        )

        return None

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
    sequences,
    threshold
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
            f"❌ Model prediction failed: "
            f"{error}"
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
        1.0
        -
        fake_probability
    )

    prediction = (
        "FAKE"
        if fake_probability >= threshold
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

    # ========================================================
    # VALIDATE VIDEO
    # ========================================================

    if not video_path.exists():

        print(
            f"❌ Video not found: "
            f"{video_path}"
        )

        return

    if not video_path.is_file():

        print(
            f"❌ Path is not a file: "
            f"{video_path}"
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
    # STEP 1 — MODEL
    # ========================================================

    model = load_model()

    if model is None:

        return

    # ========================================================
    # STEP 2 — SCALER
    # ========================================================

    scaler = load_scaler()

    if scaler is None:

        return

    # ========================================================
    # STEP 3 — THRESHOLD
    # ========================================================

    threshold = load_threshold()

    if threshold is None:

        return

    # ========================================================
    # STEP 4 — AUDIO
    # ========================================================

    audio = extract_audio(
        video_path
    )

    if audio is None:

        return

    # ========================================================
    # STEP 5 — MFCC
    # ========================================================

    mfcc = extract_mfcc(
        audio
    )

    if mfcc is None:

        return

    # ========================================================
    # STEP 6 — VIDEO
    # ========================================================

    cap = open_video(
        video_path
    )

    if cap is None:

        return

    # ========================================================
    # STEP 7 — MEDIAPIPE
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
            f"❌ MediaPipe processing failed: "
            f"{error}"
        )

        cap.release()

        return

    cap.release()

    # ========================================================
    # STEP 8 — SYNCHRONIZE
    # ========================================================

    features = synchronize_features(
        lip_features,
        mfcc,
        fps
    )

    if features is None:

        return

    print(
        f"\nRaw fused features : "
        f"{features.shape}"
    )

    # ========================================================
    # STEP 9 — SCALE
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
    # STEP 10 — SEQUENCES
    # ========================================================

    sequences = create_sequences(
        features
    )

    if sequences is None:

        return

    # ========================================================
    # STEP 11 — PREDICTION
    # ========================================================

    result = predict(
        model,
        sequences,
        threshold
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
    # STEP 12 — RESULT
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
        f"{threshold:.2f}"
    )

    print(
        f"Prediction         : "
        f"{prediction}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # STEP 13 — PER-SEQUENCE RESULTS
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
            if probability >= threshold
            else "REAL"
        )

        print(
            f"Sequence {i:02d} : "
            f"{probability:.4f} "
            f"({probability * 100:.2f}%) "
            f"-> {sequence_prediction}"
        )

    # ========================================================
    # STEP 14 — SUMMARY
    # ========================================================

    fake_sequences = int(
        np.sum(
            probabilities >= threshold
        )
    )

    real_sequences = (
        len(probabilities)
        -
        fake_sequences
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
            'python -m inference.predict_video "$HOME/Desktop/real.mp4"'
        )

        sys.exit(1)

    predict_video(
        sys.argv[1]
    )