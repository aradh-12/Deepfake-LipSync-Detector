from pathlib import Path


# Project root is one level above the scripts directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


ESSENTIAL_FILES = [
    "app.py",
    "requirements.txt",
    "configs/config.py",
    "backend/detector.py",
    "pipeline/predict_video.py",
]


ESSENTIAL_DIRECTORIES = [
    "audio_processing",
    "backend",
    "configs",
    "feature_extraction",
    "feature_fusion",
    "models",
    "pipeline",
    "preprocessing",
    "saved_models",
    "training",
    "evaluation",
]


def check_path(relative_path):
    """Check whether a file or directory exists."""
    return (PROJECT_ROOT / relative_path).exists()


def main():
    print("\nMultimodal Deepfake Lip-Sync Detector")
    print("Project Health Check")
    print("=" * 45)

    print("\nChecking essential files:")

    for file_path in ESSENTIAL_FILES:
        if check_path(file_path):
            print(f"  [OK]      {file_path}")
        else:
            print(f"  [MISSING] {file_path}")

    print("\nChecking essential directories:")

    for directory in ESSENTIAL_DIRECTORIES:
        if check_path(directory):
            print(f"  [OK]      {directory}/")
        else:
            print(f"  [MISSING] {directory}/")

    print("\nHealth check completed.\n")


if __name__ == "__main__":
    main()