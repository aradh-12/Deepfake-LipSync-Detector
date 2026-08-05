from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets"

SAMPLE_VIDEO_ROOT = PROJECT_ROOT / "data" / "sample_videos"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# ==========================
# Output Directories
# ==========================

FRAMES_OUTPUT = OUTPUT_ROOT / "extracted_frames"

FACE_MESH_OUTPUT = OUTPUT_ROOT / "face_mesh"

LIP_LANDMARK_OUTPUT = OUTPUT_ROOT / "lip_landmarks"

LIP_COORDINATE_OUTPUT = OUTPUT_ROOT / "lip_coordinates"

AUDIO_OUTPUT = OUTPUT_ROOT / "audio"

MFCC_OUTPUT = OUTPUT_ROOT / "mfcc"

SYNCHRONIZED_OUTPUT = OUTPUT_ROOT / "synchronized"

SEQUENCE_OUTPUT = OUTPUT_ROOT / "sequences"

# ==========================
# Model Configuration
# ==========================

SEQUENCE_LENGTH = 30

FEATURE_SIZE = 93

NUM_CLASSES = 1

# ==========================
# Training Configuration
# ==========================

BATCH_SIZE = 32

EPOCHS = 25

LEARNING_RATE = 0.001

VALIDATION_SPLIT = 0.2

RANDOM_SEED = 42