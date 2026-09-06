import numpy as np
import re


# ============================================================
# EXP7 SOURCE-AWARE DATASET
# ============================================================

DATASET_DIR = "outputs/dataset_exp7_source_aware"


# ============================================================
# Load Video IDs
# ============================================================

def load_video_ids(filename):

    data = np.load(
        f"{DATASET_DIR}/{filename}",
        allow_pickle=True
    )

    return data["video_ids"]


# ============================================================
# Extract Source IDs
# ============================================================

def extract_ids(video_name):

    """
    Extract all IDs from a video name.

    Example:

    FakeAVCeleb_id00103_00241_id04537_wavtolip

    Returns:

    {
        "id00103",
        "id04537"
    }
    """

    return set(
        re.findall(
            r"id\d+",
            str(video_name)
        )
    )


# ============================================================
# Get All Source IDs
# ============================================================

def get_source_ids(video_ids):

    all_ids = set()

    for video in np.unique(video_ids):

        ids = extract_ids(video)

        all_ids.update(ids)

    return all_ids


# ============================================================
# Get Videos Containing Source ID
# ============================================================

def get_videos_for_source_id(
    video_ids,
    source_id
):

    matching_videos = []

    for video in np.unique(video_ids):

        ids = extract_ids(video)

        if source_id in ids:

            matching_videos.append(
                str(video)
            )

    return matching_videos


# ============================================================
# Main
# ============================================================

print("\n" + "=" * 70)
print("EXP7 SOURCE-VIDEO LEAKAGE CHECK")
print("=" * 70)


print("\nDataset directory:")
print(DATASET_DIR)


# ============================================================
# Load Splits
# ============================================================

print("\nLoading dataset splits...")

train_video_ids = load_video_ids(
    "train.npz"
)

val_video_ids = load_video_ids(
    "validation.npz"
)

test_video_ids = load_video_ids(
    "test.npz"
)


# ============================================================
# Exact Video Leakage Check
# ============================================================

train_videos = set(
    np.unique(train_video_ids)
)

val_videos = set(
    np.unique(val_video_ids)
)

test_videos = set(
    np.unique(test_video_ids)
)


train_val_video_overlap = (
    train_videos.intersection(val_videos)
)

train_test_video_overlap = (
    train_videos.intersection(test_videos)
)

val_test_video_overlap = (
    val_videos.intersection(test_videos)
)


# ============================================================
# Source ID Extraction
# ============================================================

print("\nExtracting source IDs...")

train_ids = get_source_ids(
    train_video_ids
)

val_ids = get_source_ids(
    val_video_ids
)

test_ids = get_source_ids(
    test_video_ids
)


# ============================================================
# Source ID Overlap
# ============================================================

train_val_overlap = (
    train_ids.intersection(val_ids)
)

train_test_overlap = (
    train_ids.intersection(test_ids)
)

val_test_overlap = (
    val_ids.intersection(test_ids)
)


# ============================================================
# Dataset Statistics
# ============================================================

print("\n" + "=" * 70)
print("VIDEO STATISTICS")
print("=" * 70)

print(f"\nTrain videos      : {len(train_videos)}")
print(f"Validation videos : {len(val_videos)}")
print(f"Test videos       : {len(test_videos)}")


print("\n" + "=" * 70)
print("SOURCE ID STATISTICS")
print("=" * 70)

print(f"\nTrain source IDs      : {len(train_ids)}")
print(f"Validation source IDs : {len(val_ids)}")
print(f"Test source IDs       : {len(test_ids)}")


# ============================================================
# Exact Video Leakage
# ============================================================

print("\n" + "=" * 70)
print("EXACT VIDEO LEAKAGE CHECK")
print("=" * 70)

print(
    f"\nTrain ∩ Validation : "
    f"{len(train_val_video_overlap)}"
)

print(
    f"Train ∩ Test       : "
    f"{len(train_test_video_overlap)}"
)

print(
    f"Validation ∩ Test  : "
    f"{len(val_test_video_overlap)}"
)


# ============================================================
# Source ID Leakage
# ============================================================

print("\n" + "=" * 70)
print("SOURCE ID OVERLAP")
print("=" * 70)

print(
    f"\nTrain ∩ Validation : "
    f"{len(train_val_overlap)}"
)

print(
    f"Train ∩ Test       : "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test  : "
    f"{len(val_test_overlap)}"
)


# ============================================================
# Print Source Overlap Details
# ============================================================

def print_overlap(
    name,
    overlap,
    split1_name,
    split1_videos,
    split2_name,
    split2_videos
):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    if len(overlap) == 0:

        print(
            "No overlap detected."
        )

    else:

        print(
            "Overlapping source IDs:"
        )

        for source_id in sorted(overlap):

            print(
                f"\n{source_id}"
            )

            print(
                f"  {split1_name}:"
            )

            for video in get_videos_for_source_id(
                split1_videos,
                source_id
            ):

                print(
                    f"    {video}"
                )

            print(
                f"  {split2_name}:"
            )

            for video in get_videos_for_source_id(
                split2_videos,
                source_id
            ):

                print(
                    f"    {video}"
                )


print_overlap(
    "TRAIN ∩ VALIDATION",
    train_val_overlap,
    "Train",
    train_video_ids,
    "Validation",
    val_video_ids
)


print_overlap(
    "TRAIN ∩ TEST",
    train_test_overlap,
    "Train",
    train_video_ids,
    "Test",
    test_video_ids
)


print_overlap(
    "VALIDATION ∩ TEST",
    val_test_overlap,
    "Validation",
    val_video_ids,
    "Test",
    test_video_ids
)


# ============================================================
# Final Result
# ============================================================

print("\n" + "=" * 70)


no_exact_video_leakage = (

    len(train_val_video_overlap) == 0
    and len(train_test_video_overlap) == 0
    and len(val_test_video_overlap) == 0
)


no_source_leakage = (

    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
)


if (
    no_exact_video_leakage
    and no_source_leakage
):

    print(
        "RESULT: NO SOURCE-VIDEO LEAKAGE DETECTED"
    )

else:

    print(
        "WARNING: SOURCE-VIDEO OVERLAP DETECTED"
    )


print("=" * 70)