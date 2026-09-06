from pathlib import Path
import sys
import importlib.util

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_DIR = (
    PROJECT_ROOT
    / "pipeline"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "models"
    / "best_lstm_multimodal_model.keras"
)

NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "models"
    / "normalization_stats.npz"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MAX_SEQUENCE_LENGTH = 150
FEATURE_DIMENSION = 119

REAL_LABEL = 0
FAKE_LABEL = 1


# ============================================================
# LOAD PIPELINE MODULE
# ============================================================

def load_pipeline_module(module_name, file_name):

    file_path = (
        PIPELINE_DIR
        / file_name
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Pipeline file not found:\n{file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# LOAD FEATURE EXTRACTION MODULES
# ============================================================

def load_modules():

    print()
    print("=" * 70)
    print("LOADING FEATURE EXTRACTION MODULES")
    print("=" * 70)
    print()

    lip_module = load_pipeline_module(
        "extract_lip_features_module",
        "02_extract_lip_features.py"
    )

    print("Lip feature module loaded.")

    mfcc_module = load_pipeline_module(
        "extract_mfcc_module",
        "03_extract_mfcc.py"
    )

    print("MFCC feature module loaded.")

    fusion_module = load_pipeline_module(
        "build_multimodal_features_module",
        "04_build_multimodal_features.py"
    )

    print("Multimodal fusion module loaded.")

    print()

    return (
        lip_module,
        mfcc_module,
        fusion_module
    )


# ============================================================
# EXTRACT LIP FEATURES
# ============================================================
def extract_lip_features_from_video(
    lip_module,
    video_path
):

    print()
    print("=" * 70)
    print("EXTRACTING LIP FEATURES")
    print("=" * 70)
    print()

    if not hasattr(
        lip_module,
        "extract_lip_features"
    ):

        raise RuntimeError(
            "extract_lip_features() was not found "
            "in 02_extract_lip_features.py"
        )

    # --------------------------------------------------------
    # Create MediaPipe Face Mesh
    # --------------------------------------------------------

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(

        static_image_mode=False,

        max_num_faces=1,

        refine_landmarks=True,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5

    ) as face_mesh:

        # ----------------------------------------------------
        # 02_extract_lip_features.py expects:
        #
        # extract_lip_features(
        #     video_path,
        #     face_mesh
        # )
        # ----------------------------------------------------

        result = (
            lip_module.extract_lip_features(

                video_path,

                face_mesh

            )
        )

    # --------------------------------------------------------
    # Handle return value
    # --------------------------------------------------------

    if isinstance(
        result,
        tuple
    ):

        lip_features = result[0]

        if len(result) > 1:

            fps = result[1]

            print(
                f"Video FPS: {fps:.2f}"
            )

    else:

        lip_features = result

    if lip_features is None:

        raise ValueError(
            "Lip feature extraction failed."
        )

    lip_features = np.asarray(

        lip_features,

        dtype=np.float32

    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if lip_features.ndim != 2:

        raise ValueError(

            f"Invalid lip feature shape: "
            f"{lip_features.shape}"

        )

    print(

        "Lip feature shape:",

        lip_features.shape

    )

    return lip_features


# ============================================================
# EXTRACT MFCC FEATURES
# ============================================================

def extract_mfcc_features_from_video(
    mfcc_module,
    video_path
):

    print()
    print("=" * 70)
    print("EXTRACTING MFCC FEATURES")
    print("=" * 70)
    print()

    if not hasattr(
        mfcc_module,
        "extract_mfcc_features"
    ):

        raise RuntimeError(
            "extract_mfcc_features() was not found "
            "in 03_extract_mfcc.py"
        )

    mfcc_features = (
        mfcc_module.extract_mfcc_features(
            video_path
        )
    )

    if mfcc_features is None:

        raise ValueError(
            "MFCC feature extraction failed."
        )

    mfcc_features = np.asarray(
        mfcc_features,
        dtype=np.float32
    )

    print(
        "MFCC feature shape:",
        mfcc_features.shape
    )

    return mfcc_features


# ============================================================
# BUILD MULTIMODAL FEATURES
# ============================================================

def build_features(
    fusion_module,
    lip_features,
    mfcc_features
):

    print()
    print("=" * 70)
    print("BUILDING MULTIMODAL FEATURES")
    print("=" * 70)
    print()

    if not hasattr(
        fusion_module,
        "build_multimodal_features"
    ):

        raise RuntimeError(
            "build_multimodal_features() was not found "
            "in 04_build_multimodal_features.py"
        )

    features = (
        fusion_module.build_multimodal_features(
            lip_features,
            mfcc_features
        )
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.ndim != 2:

        raise ValueError(
            f"Invalid multimodal feature shape: "
            f"{features.shape}"
        )

    if features.shape[1] != FEATURE_DIMENSION:

        raise ValueError(
            f"Expected {FEATURE_DIMENSION} features, "
            f"got {features.shape[1]}"
        )

    if not np.isfinite(
        features
    ).all():

        raise ValueError(
            "Features contain NaN or Inf."
        )

    print(
        "Multimodal feature shape:",
        features.shape
    )

    return features


# ============================================================
# PAD OR TRUNCATE SEQUENCE
# ============================================================

def pad_sequence(features):

    frames = features.shape[0]

    if frames == 0:

        raise ValueError(
            "No valid frames found."
        )

    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------

    if frames >= MAX_SEQUENCE_LENGTH:

        return features[
            :MAX_SEQUENCE_LENGTH
        ]

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    padded = np.zeros(

        (
            MAX_SEQUENCE_LENGTH,
            FEATURE_DIMENSION
        ),

        dtype=np.float32
    )

    padded[
        :frames
    ] = features

    return padded


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

def load_normalization_statistics():

    print()
    print("=" * 70)
    print("LOADING NORMALIZATION STATISTICS")
    print("=" * 70)
    print()

    if not NORMALIZATION_PATH.exists():

        raise FileNotFoundError(
            f"Normalization file not found:\n"
            f"{NORMALIZATION_PATH}"
        )

    data = np.load(
        NORMALIZATION_PATH
    )

    try:

        if "mean" in data.files:

            mean = data["mean"]

        elif "feature_mean" in data.files:

            mean = data["feature_mean"]

        else:

            raise KeyError(
                "Mean not found."
            )

        if "std" in data.files:

            std = data["std"]

        elif "feature_std" in data.files:

            std = data["feature_std"]

        else:

            raise KeyError(
                "Standard deviation not found."
            )

    finally:

        data.close()

    mean = mean.astype(
        np.float32
    )

    std = std.astype(
        np.float32
    )

    print(
        "Mean shape:",
        mean.shape
    )

    print(
        "Std shape:",
        std.shape
    )

    return (
        mean,
        std
    )


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize_features(
    features,
    mean,
    std,
    valid_frames
):

    normalized = features.copy()

    valid_frames = min(
        valid_frames,
        MAX_SEQUENCE_LENGTH
    )

    normalized[
        :valid_frames
    ] = (

        (
            normalized[:valid_frames]
            - mean
        )

        /

        (
            std
            + 1e-8
        )
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)
    print()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(

        MODEL_PATH,

        compile=False

    )

    print(
        "Model loaded successfully."
    )

    print()

    print(
        "Model input shape:",
        model.input_shape
    )

    print(
        "Model output shape:",
        model.output_shape
    )

    return model


# ============================================================
# PREDICT VIDEO
# ============================================================

def predict_video(video_path):

    video_path = Path(
        video_path
    ).expanduser().resolve()

    # --------------------------------------------------------
    # Validate video
    # --------------------------------------------------------

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    if not video_path.is_file():

        raise ValueError(
            f"Path is not a file:\n"
            f"{video_path}"
        )

    print()
    print("=" * 70)
    print("MULTIMODAL DEEPFAKE LIP-SYNC PREDICTION")
    print("=" * 70)
    print()

    print("Video:")
    print(video_path)

    # --------------------------------------------------------
    # Load modules
    # --------------------------------------------------------

    (
        lip_module,
        mfcc_module,
        fusion_module

    ) = load_modules()

    # --------------------------------------------------------
    # Extract lip features
    # --------------------------------------------------------

    lip_features = (
        extract_lip_features_from_video(

            lip_module,

            video_path

        )
    )

    # --------------------------------------------------------
    # Extract MFCC features
    # --------------------------------------------------------

    mfcc_features = (
        extract_mfcc_features_from_video(

            mfcc_module,

            video_path

        )
    )

    # --------------------------------------------------------
    # Build multimodal features
    # --------------------------------------------------------

    multimodal_features = (
        build_features(

            fusion_module,

            lip_features,

            mfcc_features

        )
    )

    valid_frames = (
        multimodal_features.shape[0]
    )

    print(
        "Valid frames:",
        valid_frames
    )

    # --------------------------------------------------------
    # Pad sequence
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPARING MODEL INPUT")
    print("=" * 70)
    print()

    padded_features = (
        pad_sequence(
            multimodal_features
        )
    )

    print(
        "Padded feature shape:",
        padded_features.shape
    )

    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    (
        mean,
        std

    ) = load_normalization_statistics()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized_features = (
        normalize_features(

            padded_features,

            mean,

            std,

            valid_frames

        )
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    model_input = np.expand_dims(

        normalized_features,

        axis=0

    )

    print()
    print(
        "Final model input shape:",
        model_input.shape
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING PREDICTION")
    print("=" * 70)
    print()

    probability = model.predict(

        model_input,

        verbose=0

    )[0][0]

    probability = float(
        probability
    )

    # --------------------------------------------------------
    # Determine prediction
    # --------------------------------------------------------

    if probability >= 0.5:

        prediction = "FAKE"

        fake_probability = probability

        real_probability = (
            1.0
            - probability
        )

        confidence = (
            probability
            * 100
        )

    else:

        prediction = "REAL"

        fake_probability = probability

        real_probability = (
            1.0
            - probability
        )

        confidence = (
            real_probability
            * 100
        )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print()

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print()

    print(
        f"Real probability: "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"Fake probability: "
        f"{fake_probability * 100:.2f}%"
    )

    print()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print("DEEPFAKE LIP-SYNC DETECTOR")

    print("=" * 70)

    print()

    print(
        "Enter the path of the video "
        "you want to analyze."
    )

    print()

    video_path = input(
        "Video path: "
    ).strip()

    if not video_path:

        print()

        print("No video path entered.")

        return

    try:

        predict_video(
            video_path
        )

    except Exception as error:

        print()

        print("=" * 70)

        print("PREDICTION FAILED")

        print("=" * 70)

        print()

        print(
            type(error).__name__
        )

        print()

        print(
            error
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()