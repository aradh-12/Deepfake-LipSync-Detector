from pathlib import Path
import random


# ============================================================
# Dataset Roots
# ============================================================

FF_ROOT = Path(
    "datasets/FaceForensics++/FaceForensics++_C23"
)

FA_ROOT = Path(
    "datasets/FakeAVCeleb/"
    "FakeAVCeleb_v1.2/"
    "FakeAVCeleb_v1.2"
)


# ============================================================
# FaceForensics++
# ============================================================

def get_faceforensics(limit=10):

    videos = []

    for video in sorted(
        (FF_ROOT / "original").glob("*.mp4")
    )[:limit]:

        videos.append((video, 0))

    fake_dirs = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "FaceShifter",
        "NeuralTextures"
    ]

    for folder in fake_dirs:

        for video in sorted(
            (FF_ROOT / folder).glob("*.mp4")
        )[:limit]:

            videos.append((video, 1))

    return videos


# ============================================================
# FakeAVCeleb — All Videos
# ============================================================

def get_fakeavceleb_all():

    all_videos = sorted(
        FA_ROOT.rglob("*.mp4")
    )

    real_videos = [
        video
        for video in all_videos
        if "RealVideo-RealAudio" in str(video)
    ]

    fake_videos = [
    video
    for video in all_videos
    if "wavtolip" in str(video).lower()
]
    return real_videos, fake_videos


# ============================================================
# Create Reproducible FakeAVCeleb Split
# ============================================================

def get_fakeavceleb_split(
    real_count=25,
    fake_count=25,
    seed=42
):

    real_videos, fake_videos = (
        get_fakeavceleb_all()
    )

    rng = random.Random(seed)

    real_videos = real_videos.copy()
    fake_videos = fake_videos.copy()

    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)

    development = []

    development.extend(
        (video, 0)
        for video in real_videos[:real_count]
    )

    development.extend(
        (video, 1)
        for video in fake_videos[:fake_count]
    )

    return development


# ============================================================
# Development Set
# ============================================================

def get_fakeavceleb_development(
    real_count=25,
    fake_count=25
):

    return get_fakeavceleb_split(
        real_count=real_count,
        fake_count=fake_count,
        seed=42
    )


# ============================================================
# Unseen Set
# ============================================================

def get_fakeavceleb_unseen(
    real_count=25,
    fake_count=25
):

    real_videos, fake_videos = (
        get_fakeavceleb_all()
    )

    rng = random.Random(42)

    real_videos = real_videos.copy()
    fake_videos = fake_videos.copy()

    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)

    unseen = []

    unseen.extend(
        (video, 0)
        for video in real_videos[
            real_count:real_count * 2
        ]
    )

    unseen.extend(
        (video, 1)
        for video in fake_videos[
            fake_count:fake_count * 2
        ]
    )

    return unseen


# ============================================================
# Backward-Compatible Function
# ============================================================

def get_fakeavceleb(limit=50):

    half = limit // 2

    return get_fakeavceleb_development(
        real_count=half,
        fake_count=half
    )


# ============================================================
# Combined Dataset
# ============================================================

def get_all_videos():

    dataset = []

    dataset.extend(
        get_faceforensics()
    )

    dataset.extend(
        get_fakeavceleb()
    )

    return dataset