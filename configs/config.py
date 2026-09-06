from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATASET_ROOT = (
    PROJECT_ROOT / "datasets"
)

SAMPLE_VIDEO_ROOT = (
    PROJECT_ROOT / "data" / "sample_videos"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs"
)

SAVED_MODEL_ROOT = (
    PROJECT_ROOT / "saved_models"
)


# ============================================================
# DATASET OUTPUT DIRECTORIES
# ============================================================

FRAMES_OUTPUT = (
    OUTPUT_ROOT / "extracted_frames"
)

FACE_MESH_OUTPUT = (
    OUTPUT_ROOT / "face_mesh"
)

LIP_LANDMARK_OUTPUT = (
    OUTPUT_ROOT / "lip_landmarks"
)

LIP_COORDINATE_OUTPUT = (
    OUTPUT_ROOT / "lip_coordinates"
)

AUDIO_OUTPUT = (
    OUTPUT_ROOT / "audio"
)

MFCC_OUTPUT = (
    OUTPUT_ROOT / "mfcc"
)

MFCC_EXP5_OUTPUT = (
    OUTPUT_ROOT / "mfcc_exp5"
)

SYNCHRONIZED_OUTPUT = (
    OUTPUT_ROOT / "synchronized_aligned"
)

SYNCHRONIZED_EXP5_OUTPUT = (
    OUTPUT_ROOT / "synchronized_aligned_exp5"
)

SEQUENCE_OUTPUT = (
    OUTPUT_ROOT / "sequences"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30


# ------------------------------------------------------------
# Feature dimensions
#
# Lip coordinates:
# 40 landmarks × 2 coordinates
# = 80 features
#
# Lip velocity:
# = 80 features
#
# MFCC:
# 39 features
#
# Total:
# 80 + 80 + 39 = 199
# ------------------------------------------------------------

LIP_FEATURE_SIZE = 80

LIP_VELOCITY_SIZE = 80

MFCC_FEATURE_SIZE = 39


FEATURE_SIZE = (
    LIP_FEATURE_SIZE
    + LIP_VELOCITY_SIZE
    + MFCC_FEATURE_SIZE
)


NUM_CLASSES = 1


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ============================================================
# DATASET SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

SAVED_MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)