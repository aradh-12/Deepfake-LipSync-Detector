from pathlib import Path
import random


# ============================================================
# DATASET ROOTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


FF_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "FaceForensics++"
    / "FaceForensics++_C23"
)


FA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "FakeAVCeleb"
    / "FakeAVCeleb_v1.2"
    / "FakeAVCeleb_v1.2"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# DATASET SIZE
# ============================================================
#
# FakeAVCeleb contains:
#
# RealVideo-RealAudio
#     500 genuine videos
#
# wavtolip fake videos
#     thousands available
#
# We start with a balanced dataset:
#
# TRAIN:
#     200 REAL
#     200 FAKE
#
# VALIDATION:
#      50 REAL
#      50 FAKE
#
# TEST:
#      50 REAL
#      50 FAKE
#
# TOTAL:
#     300 REAL
#     300 FAKE
# ============================================================


TRAIN_REAL_COUNT = 200
TRAIN_FAKE_COUNT = 200


VALIDATION_REAL_COUNT = 50
VALIDATION_FAKE_COUNT = 50


TEST_REAL_COUNT = 50
TEST_FAKE_COUNT = 50


# ============================================================
# VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov"
)


# ============================================================
# CHECK VIDEO FILE
# ============================================================

def is_video_file(path):

    return (
        path.is_file()
        and path.suffix.lower()
        in VIDEO_EXTENSIONS
    )


# ============================================================
# GET ALL FAKEAVCELEB VIDEOS
# ============================================================

def get_fakeavceleb_all():
    """
    Returns:

        real_videos
        fake_videos

    REAL:
        RealVideo-RealAudio

    FAKE:
        wavtolip videos

    The project focuses on detecting
    audio-lip synchronization anomalies.
    """

    if not FA_ROOT.exists():

        raise FileNotFoundError(
            "FakeAVCeleb dataset root not found:\n"
            f"{FA_ROOT}"
        )


    # --------------------------------------------------------
    # REAL VIDEOS
    # --------------------------------------------------------

    real_root = (
        FA_ROOT
        / "RealVideo-RealAudio"
    )


    if not real_root.exists():

        raise FileNotFoundError(
            "RealVideo-RealAudio directory not found:\n"
            f"{real_root}"
        )


    real_videos = sorted(

        video

        for video in real_root.rglob("*")

        if is_video_file(video)

    )


    # --------------------------------------------------------
    # FAKE VIDEOS
    # --------------------------------------------------------
    #
    # We specifically select wavtolip videos.
    #
    # These are relevant to the project's
    # lip-sync anomaly detection objective.
    # --------------------------------------------------------

    fake_videos = []


    for video in FA_ROOT.rglob("*"):

        if not is_video_file(video):

            continue


        if "wavtolip" in video.stem.lower():

            fake_videos.append(video)


    fake_videos = sorted(fake_videos)


    return real_videos, fake_videos


# ============================================================
# BUILD REPRODUCIBLE SPLITS
# ============================================================

def build_fakeavceleb_splits(
    seed=RANDOM_SEED
):
    """
    Creates reproducible video-level splits.

    IMPORTANT:

    A video can appear in ONLY ONE split.

        TRAIN
        VALIDATION
        TEST

    This prevents data leakage.
    """


    real_videos, fake_videos = (
        get_fakeavceleb_all()
    )


    # --------------------------------------------------------
    # REQUIRED COUNTS
    # --------------------------------------------------------

    required_real = (

        TRAIN_REAL_COUNT
        + VALIDATION_REAL_COUNT
        + TEST_REAL_COUNT

    )


    required_fake = (

        TRAIN_FAKE_COUNT
        + VALIDATION_FAKE_COUNT
        + TEST_FAKE_COUNT

    )


    # --------------------------------------------------------
    # VALIDATE DATASET SIZE
    # --------------------------------------------------------

    if len(real_videos) < required_real:

        raise ValueError(

            "Not enough REAL videos.\n"

            f"Available: {len(real_videos)}\n"

            f"Required: {required_real}"

        )


    if len(fake_videos) < required_fake:

        raise ValueError(

            "Not enough FAKE videos.\n"

            f"Available: {len(fake_videos)}\n"

            f"Required: {required_fake}"

        )


    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    rng = random.Random(seed)


    real_videos = real_videos.copy()

    fake_videos = fake_videos.copy()


    rng.shuffle(real_videos)

    rng.shuffle(fake_videos)


    # ========================================================
    # REAL SPLIT
    # ========================================================


    train_real = real_videos[
        :TRAIN_REAL_COUNT
    ]


    validation_real = real_videos[

        TRAIN_REAL_COUNT:

        TRAIN_REAL_COUNT
        + VALIDATION_REAL_COUNT

    ]


    test_real = real_videos[

        TRAIN_REAL_COUNT
        + VALIDATION_REAL_COUNT:

        required_real

    ]


    # ========================================================
    # FAKE SPLIT
    # ========================================================


    train_fake = fake_videos[
        :TRAIN_FAKE_COUNT
    ]


    validation_fake = fake_videos[

        TRAIN_FAKE_COUNT:

        TRAIN_FAKE_COUNT
        + VALIDATION_FAKE_COUNT

    ]


    test_fake = fake_videos[

        TRAIN_FAKE_COUNT
        + VALIDATION_FAKE_COUNT:

        required_fake

    ]


    # ========================================================
    # CREATE DATASETS
    #
    # Label:
    #
    # REAL = 0
    # FAKE = 1
    # ========================================================


    train = []


    train.extend(

        (video, 0)

        for video in train_real

    )


    train.extend(

        (video, 1)

        for video in train_fake

    )


    validation = []


    validation.extend(

        (video, 0)

        for video in validation_real

    )


    validation.extend(

        (video, 1)

        for video in validation_fake

    )


    test = []


    test.extend(

        (video, 0)

        for video in test_real

    )


    test.extend(

        (video, 1)

        for video in test_fake

    )


    # --------------------------------------------------------
    # SHUFFLE EACH SPLIT
    # --------------------------------------------------------

    rng.shuffle(train)

    rng.shuffle(validation)

    rng.shuffle(test)


    return {

        "train": train,

        "validation": validation,

        "test": test

    }


# ============================================================
# GET TRAIN DATASET
# ============================================================

def get_fakeavceleb_train():

    splits = build_fakeavceleb_splits()

    return splits["train"]


# ============================================================
# GET VALIDATION DATASET
# ============================================================

def get_fakeavceleb_validation():

    splits = build_fakeavceleb_splits()

    return splits["validation"]


# ============================================================
# GET TEST DATASET
# ============================================================

def get_fakeavceleb_test():

    splits = build_fakeavceleb_splits()

    return splits["test"]


# ============================================================
# GET ALL SELECTED DATASET
# ============================================================

def get_fakeavceleb_selected():

    splits = build_fakeavceleb_splits()


    dataset = []


    dataset.extend(
        splits["train"]
    )


    dataset.extend(
        splits["validation"]
    )


    dataset.extend(
        splits["test"]
    )


    return dataset


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_fakeavceleb_development(
    real_count=100,
    fake_count=100
):
    """
    Temporary compatibility function.

    New pipeline should use:

        get_fakeavceleb_train()
        get_fakeavceleb_validation()
        get_fakeavceleb_test()
    """

    real_videos, fake_videos = (
        get_fakeavceleb_all()
    )


    rng = random.Random(
        RANDOM_SEED
    )


    real_videos = real_videos.copy()

    fake_videos = fake_videos.copy()


    rng.shuffle(real_videos)

    rng.shuffle(fake_videos)


    dataset = []


    dataset.extend(

        (video, 0)

        for video in real_videos[:real_count]

    )


    dataset.extend(

        (video, 1)

        for video in fake_videos[:fake_count]

    )


    rng.shuffle(dataset)


    return dataset


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_fakeavceleb_unseen(
    real_count=100,
    fake_count=100
):
    """
    Temporary compatibility function.
    """

    real_videos, fake_videos = (
        get_fakeavceleb_all()
    )


    rng = random.Random(
        RANDOM_SEED
    )


    real_videos = real_videos.copy()

    fake_videos = fake_videos.copy()


    rng.shuffle(real_videos)

    rng.shuffle(fake_videos)


    dataset = []


    dataset.extend(

        (video, 0)

        for video in real_videos[
            real_count:
            real_count * 2
        ]

    )


    dataset.extend(

        (video, 1)

        for video in fake_videos[
            fake_count:
            fake_count * 2
        ]

    )


    rng.shuffle(dataset)


    return dataset


# ============================================================
# FACEFORENSICS++
# ============================================================

def get_faceforensics(limit=10):

    videos = []


    original_root = (
        FF_ROOT / "original"
    )


    for video in sorted(

        original_root.glob("*.mp4")

    )[:limit]:

        videos.append(

            (video, 0)

        )


    fake_dirs = [

        "Deepfakes",

        "Face2Face",

        "FaceSwap",

        "FaceShifter",

        "NeuralTextures"

    ]


    for folder in fake_dirs:


        folder_path = (
            FF_ROOT / folder
        )


        for video in sorted(

            folder_path.glob("*.mp4")

        )[:limit]:

            videos.append(

                (video, 1)

            )


    return videos


# ============================================================
# COMBINED DATASET
# ============================================================

def get_all_videos():

    return get_fakeavceleb_selected()