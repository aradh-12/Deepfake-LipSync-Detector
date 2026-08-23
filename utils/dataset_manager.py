from pathlib import Path

# Root directory of FaceForensics++
DATASET_ROOT = Path("datasets/FaceForensics++/FaceForensics++_C23")

# Real videos
REAL_FOLDER = DATASET_ROOT / "original"

# Fake videos
FAKE_FOLDERS = [
    DATASET_ROOT / "Deepfakes",
    DATASET_ROOT / "Face2Face",
    DATASET_ROOT / "FaceSwap",
    DATASET_ROOT / "FaceShifter",
    DATASET_ROOT / "NeuralTextures",
]


def get_real_videos():
    """
    Return all real videos.
    """
    return sorted(REAL_FOLDER.glob("*.mp4"))


def get_fake_videos():
    """
    Return all fake videos.
    """
    videos = []

    for folder in FAKE_FOLDERS:
        videos.extend(sorted(folder.glob("*.mp4")))

    return videos


def get_all_videos():
    """
    Return all videos with labels.

    label = 0 -> Real
    label = 1 -> Fake
    """

    dataset = []

    for video in get_real_videos():
        dataset.append((video, 0))

    for video in get_fake_videos():
        dataset.append((video, 1))

    return dataset