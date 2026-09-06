import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import librosa
import mediapipe as mp
import tensorflow as tf


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "deepfake_lipsync_lstm_exp3.keras"
)

SCALER_PATH = (
    PROJECT_ROOT /
    "models" /
    "deepfake_lipsync_feature_scaler_exp3.npz"
)

THRESHOLD_PATH = (
    PROJECT_ROOT /
    "models" /
    "threshold.json"
)


SEQUENCE_LENGTH = 30

FEATURE_SIZE = 173


SAMPLE_RATE = 16000

N_MFCC = 13


# Must match training configuration

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


    # ========================================================
    # VALIDATE MODEL INPUT
    # ========================================================

    if len(model.input_shape) != 3:

        print(
            "\n❌ Invalid model input dimensions."
        )

        return None


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
            "\n❌ Model input shape does not "
            "match pipeline."
        )

        print(
            f"Expected : "
            f"(None, {SEQUENCE_LENGTH}, {FEATURE_SIZE})"
        )

        print(
            f"Got      : {model.input_shape}"
        )

        return None


    # ========================================================
    # VALIDATE MODEL OUTPUT
    # ========================================================

    if len(model.output_shape) != 2:

        print(
            "\n❌ Invalid model output dimensions."
        )

        return None


    if model.output_shape[-1] != 1:

        print(
            "\n❌ Model must output one probability."
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
            "❌ Invalid scaler mean shape."
        )

        print(
            f"Expected: ({FEATURE_SIZE},)"
        )

        print(
            f"Got: {mean.shape}"
        )

        return None


    if scale.shape != (
        FEATURE_SIZE,
    ):

        print(
            "❌ Invalid scaler scale shape."
        )

        return None


    mean = mean.astype(
        np.float32
    )


    scale = scale.astype(
        np.float32
    )


    if not np.isfinite(mean).all():

        print(
            "❌ Scaler mean contains NaN or Inf."
        )

        return None


    if not np.isfinite(scale).all():

        print(
            "❌ Scaler scale contains NaN or Inf."
        )

        return None


    print(
        "✅ Scaler loaded successfully."
    )


    print(
        f"Scaler features: {len(mean)}"
    )


    return (
        mean,
        scale
    )


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    if not THRESHOLD_PATH.exists():

        print(
            f"❌ Threshold file not found: "
            f"{THRESHOLD_PATH}"
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
            f"❌ Could not read threshold: {error}"
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


    if not np.isfinite(threshold):

        print(
            "❌ Threshold contains NaN or Inf."
        )

        return None


    if not 0.0 <= threshold <= 1.0:

        print(
            "❌ Threshold must be between 0 and 1."
        )

        return None


    print(
        f"✅ Threshold loaded: {threshold:.4f}"
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


    temp_directory = None


    try:

        temp_directory = Path(
            tempfile.mkdtemp(
                prefix="deepfake_lipsync_audio_"
            )
        )


        audio_path = (

            temp_directory /

            "audio.wav"

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

            "-loglevel",

            "error",

            str(audio_path)

        ]


        try:

            result = subprocess.run(

                command,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True

            )


        except FileNotFoundError:

            print(
                "❌ FFmpeg is not installed "
                "or not available in PATH."
            )

            return None


        if (

            result.returncode != 0

            or

            not audio_path.exists()

        ):

            print(
                "❌ Failed to extract audio."
            )


            if result.stderr:

                print(
                    result.stderr
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

            return None


        audio = np.asarray(

            audio,

            dtype=np.float32

        )


        if len(audio) == 0:

            print(
                "❌ Audio contains no samples."
            )

            return None


        if not np.isfinite(audio).all():

            print(
                "❌ Audio contains NaN or Inf."
            )

            return None


        print(
            f"Audio samples: {len(audio)}"
        )


        print(
            f"Sample rate: {sr}"
        )


        return audio


    finally:

        if (

            temp_directory is not None

            and

            temp_directory.exists()

        ):

            shutil.rmtree(

                temp_directory,

                ignore_errors=True

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

        return None


    if mfcc.shape[1] == 0:

        print(
            "❌ MFCC contains no frames."
        )

        return None


    # ========================================================
    # NORMALIZE MFCC
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


    if not np.isfinite(mfcc).all():

        print(
            "❌ MFCC contains NaN or Inf."
        )

        return None


    print(
        f"MFCC shape: {mfcc.shape}"
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

    coordinates = np.asarray(

        coordinates,

        dtype=np.float32

    )


    if coordinates.shape != (

        40,

        2

    ):

        raise ValueError(

            f"Expected (40, 2), "
            f"got {coordinates.shape}"

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

    print(
        "\nExtracting lip landmarks..."
    )


    total_frames = int(

        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

    )


    fps = float(

        cap.get(
            cv2.CAP_PROP_FPS
        )

    )


    if total_frames <= 0:

        print(
            "❌ Could not determine video frame count."
        )

        return (

            np.empty(
                (0, 80),
                dtype=np.float32
            ),

            fps

        )


    if fps <= 0:

        print(
            "⚠️ Invalid FPS. Using 25 FPS."
        )

        fps = 25.0


    print(
        f"Video frames: {total_frames}"
    )


    print(
        f"Video FPS: {fps:.2f}"
    )


    frame_features = [

        None

        for _ in range(
            total_frames
        )

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


        try:

            results = face_mesh.process(
                rgb
            )


        except Exception:

            frame_index += 1

            continue


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


        try:

            coordinates = (
                normalize_lip_coordinates(
                    coordinates
                )
            )


        except Exception:

            frame_index += 1

            continue


        lip_vector = coordinates.flatten()


        if lip_vector.shape != (
            80,
        ):

            frame_index += 1

            continue


        if np.isfinite(
            lip_vector
        ).all():

            if frame_index < total_frames:

                frame_features[
                    frame_index
                ] = lip_vector


                detected_count += 1


        frame_index += 1


    print(
        f"Lip frames detected: {detected_count}"
    )


    missing_count = (

        total_frames

        -

        detected_count

    )


    print(
        f"Lip frames missing: {missing_count}"
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
    # CREATE FEATURE ARRAY
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
            "❌ No valid lip features."
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
        f"Lip feature shape: {features.shape}"
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
# SYNCHRONIZE FEATURES
# ============================================================

def synchronize_features(
    lip_features,
    mfcc,
    fps
):

    if lip_features is None:

        print(
            "❌ Lip features are None."
        )

        return None


    if mfcc is None:

        print(
            "❌ MFCC is None."
        )

        return None


    if fps <= 0:

        print(
            "❌ Invalid FPS."
        )

        return None


    total_video_frames = len(
        lip_features
    )


    total_mfcc_frames = mfcc.shape[1]


    if total_video_frames == 0:

        return None


    if total_mfcc_frames == 0:

        return None


    print(
        "\nSynchronizing audio and video..."
    )


    print(
        f"Lip frames: {total_video_frames}"
    )


    print(
        f"MFCC frames: {total_mfcc_frames}"
    )


    # ========================================================
    # VIDEO TIMESTAMPS
    # ========================================================

    video_times = (

        np.arange(

            total_video_frames,

            dtype=np.float32

        )

        /

        float(fps)

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
    # FIND NEAREST MFCC
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
    # ALIGN MFCC
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
            "❌ Invalid aligned MFCC shape."
        )

        return None


    # ========================================================
    # LIP VELOCITY
    # ========================================================

    lip_velocity = compute_lip_velocity(
        lip_features
    )


    if lip_velocity is None:

        return None


    # ========================================================
    # FUSE FEATURES
    #
    # 80 Lip
    # 80 Velocity
    # 13 MFCC
    # = 173
    # ========================================================

    synchronized = np.concatenate(

        [

            lip_features,

            lip_velocity,

            aligned_mfcc

        ],

        axis=1

    ).astype(
        np.float32
    )


    print(
        f"Synchronized feature shape: "
        f"{synchronized.shape}"
    )


    if synchronized.shape != (

        total_video_frames,

        FEATURE_SIZE

    ):

        print(
            "❌ Invalid synchronized feature shape."
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

        return None


    if features.shape[1] != FEATURE_SIZE:

        print(
            f"❌ Expected {FEATURE_SIZE} features."
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
            "❌ NaN or Inf after scaling."
        )

        return None


    return scaled


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    features
):

    if features is None:

        print(
            "❌ Features are None."
        )

        return None


    if features.ndim != 2:

        print(
            "❌ Expected 2D feature matrix."
        )

        return None


    if features.shape[1] != FEATURE_SIZE:

        print(
            f"❌ Expected {FEATURE_SIZE} features."
        )

        return None


    if len(features) < SEQUENCE_LENGTH:

        print(
            "\n❌ Not enough frames."
        )

        print(
            f"Available: {len(features)}"
        )

        print(
            f"Required: {SEQUENCE_LENGTH}"
        )

        return None


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


    sequences = usable_features.reshape(

        sequence_count,

        SEQUENCE_LENGTH,

        FEATURE_SIZE

    ).astype(
        np.float32
    )


    print(
        f"\nSequences shape: {sequences.shape}"
    )


    if not np.isfinite(
        sequences
    ).all():

        print(
            "❌ Sequences contain NaN or Inf."
        )

        return None


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

            verbose=0

        )


    except Exception as error:

        print(
            f"❌ Model prediction failed: {error}"
        )

        return None


    probabilities = np.asarray(

        probabilities,

        dtype=np.float32

    )


    if probabilities.ndim != 2:

        print(
            f"❌ Unexpected output shape: "
            f"{probabilities.shape}"
        )

        return None


    if probabilities.shape != (

        len(sequences),

        1

    ):

        print(
            f"❌ Unexpected output shape: "
            f"{probabilities.shape}"
        )

        return None


    probabilities = probabilities.reshape(
        -1
    )


    if not np.isfinite(
        probabilities
    ).all():

        print(
            "❌ Model produced NaN or Inf."
        )

        return None


    probabilities = np.clip(

        probabilities,

        0.0,

        1.0

    )


    # ========================================================
    # VIDEO LEVEL PREDICTION
    # ========================================================

    fake_probability = float(

        np.mean(
            probabilities
        )

    )


    real_probability = float(

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
    # VALIDATE VIDEO PATH
    # ========================================================

    if not video_path.exists():

        raise FileNotFoundError(

            f"Video not found: {video_path}"

        )


    if not video_path.is_file():

        raise FileNotFoundError(

            f"Path is not a file: {video_path}"

        )


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
        f"Video: {video_path}"
    )


    print(
        f"Feature size: {FEATURE_SIZE}"
    )


    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )


    # ========================================================
    # STEP 1 — LOAD MODEL
    # ========================================================

    model = load_model()


    if model is None:

        raise RuntimeError(
            "Failed to load trained model."
        )


    # ========================================================
    # STEP 2 — LOAD SCALER
    # ========================================================

    scaler = load_scaler()


    if scaler is None:

        raise RuntimeError(
            "Failed to load feature scaler."
        )


    # ========================================================
    # STEP 3 — LOAD THRESHOLD
    # ========================================================

    threshold = load_threshold()


    if threshold is None:

        raise RuntimeError(
            "Failed to load threshold."
        )


    # ========================================================
    # STEP 4 — EXTRACT AUDIO
    # ========================================================

    audio = extract_audio(
        video_path
    )


    if audio is None:

        raise RuntimeError(
            "Audio extraction failed."
        )


    # ========================================================
    # STEP 5 — EXTRACT MFCC
    # ========================================================

    mfcc = extract_mfcc(
        audio
    )


    if mfcc is None:

        raise RuntimeError(
            "MFCC extraction failed."
        )


    # ========================================================
    # STEP 6 — OPEN VIDEO
    # ========================================================

    cap = open_video(
        video_path
    )


    if cap is None:

        raise RuntimeError(
            "Could not open video."
        )


    # ========================================================
    # STEP 7 — EXTRACT LIP FEATURES
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

        raise RuntimeError(

            f"MediaPipe processing failed: {error}"

        ) from error


    finally:

        cap.release()


    if lip_features is None:

        raise RuntimeError(
            "Lip feature extraction failed."
        )


    if len(lip_features) == 0:

        raise RuntimeError(
            "No usable lip features extracted."
        )


    # ========================================================
    # STEP 8 — SYNCHRONIZE
    # ========================================================

    features = synchronize_features(

        lip_features,

        mfcc,

        fps

    )


    if features is None:

        raise RuntimeError(
            "Feature synchronization failed."
        )


    print(
        f"\nRaw fused features: "
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

        raise RuntimeError(
            "Feature scaling failed."
        )


    print(
        f"Scaled features: "
        f"{features.shape}"
    )


    # ========================================================
    # STEP 10 — CREATE SEQUENCES
    # ========================================================

    sequences = create_sequences(
        features
    )


    if sequences is None:

        raise RuntimeError(
            "Sequence creation failed."
        )


    # ========================================================
    # STEP 11 — PREDICT
    # ========================================================

    prediction_result = predict(

        model,

        sequences,

        threshold

    )


    if prediction_result is None:

        raise RuntimeError(
            "Model prediction failed."
        )


    (

        fake_probability,

        real_probability,

        prediction,

        probabilities

    ) = prediction_result


    # ========================================================
    # STEP 12 — SEQUENCE RESULTS
    # ========================================================

    sequence_results = []


    print(
        "\n========================================"
    )

    print(
        "SEQUENCE PROBABILITIES"
    )

    print(
        "========================================"
    )


    for i, probability in enumerate(

        probabilities,

        start=1

    ):


        probability = float(
            probability
        )


        sequence_prediction = (

            "FAKE"

            if probability >= threshold

            else "REAL"

        )


        confidence = (

            probability * 100.0

            if sequence_prediction == "FAKE"

            else

            (1.0 - probability) * 100.0

        )


        print(

            f"Sequence {i:02d}: "

            f"{probability:.4f} "

            f"({probability * 100:.2f}%) "

            f"-> {sequence_prediction}"

        )


        sequence_results.append(

            {

                "sequence": i,

                "fake_probability": probability,

                "confidence": float(confidence),

                "prediction": sequence_prediction

            }

        )


    # ========================================================
    # STEP 13 — SUMMARY
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
        "FINAL RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"Video: {video_path.name}"
    )

    print(
        f"Sequences analyzed: {len(sequences)}"
    )

    print(
        f"Fake probability: "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"Real probability: "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Fake sequences: {fake_sequences}"
    )

    print(
        f"Real sequences: {real_sequences}"
    )

    print(
        "========================================"
    )


    # ========================================================
    # STEP 14 — RETURN DICTIONARY
    #
    # IMPORTANT:
    #
    # backend/detector.py expects a dictionary.
    #
    # DO NOT RETURN TRUE.
    # ========================================================

    return {

        "prediction":
            prediction,

        "fake_probability":
            float(fake_probability),

        "real_probability":
            float(real_probability),

        "sequences_analyzed":
            int(len(sequences)),

        "threshold":
            float(threshold),

        "sequence_results":
            sequence_results

    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":


    if len(sys.argv) != 2:


        print(
            "\nUsage:"
        )


        print(

            'python -m inference.predict_video '
            '"<video_path>"'

        )


        sys.exit(1)


    try:


        result = predict_video(
            sys.argv[1]
        )


        print(
            "\nReturned result:"
        )


        print(
            result
        )


        sys.exit(0)


    except Exception as error:


        print(
            f"\n❌ Prediction failed: {error}"
        )


        sys.exit(1)