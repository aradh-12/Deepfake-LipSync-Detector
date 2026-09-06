from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATASET_FILE = Path(
    "outputs/dataset_exp5_full/dataset.npz"
)

OUTPUT_FOLDER = Path(
    "outputs/dataset_exp7_source_aware"
)

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# Extract IDs from Video Name
# ============================================================

def extract_ids(video_id):

    """
    Extract all FakeAVCeleb-style IDs.

    Example:

    FakeAVCeleb_id00345_00243_id00060_wavtolip

    Returns:

    {
        "id00345",
        "id00060"
    }
    """

    ids = re.findall(
        r"id\d+",
        str(video_id)
    )

    return set(ids)


# ============================================================
# Union-Find
#
# Used to group videos connected through shared IDs.
# ============================================================

class UnionFind:

    def __init__(self, items):

        self.parent = {
            item: item
            for item in items
        }

        self.rank = {
            item: 0
            for item in items
        }


    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]


    def union(self, item1, item2):

        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 == root2:
            return

        if self.rank[root1] < self.rank[root2]:

            self.parent[root1] = root2

        elif self.rank[root1] > self.rank[root2]:

            self.parent[root2] = root1

        else:

            self.parent[root2] = root1

            self.rank[root1] += 1


# ============================================================
# Build Source Groups
# ============================================================

def build_source_groups(unique_videos):

    print(
        "\nBuilding source-connected groups..."
    )

    union_find = UnionFind(
        unique_videos
    )

    id_to_videos = {}

    # --------------------------------------------------------
    # Map every ID to all videos containing that ID
    # --------------------------------------------------------

    for video_id in unique_videos:

        source_ids = extract_ids(
            video_id
        )

        for source_id in source_ids:

            if source_id not in id_to_videos:

                id_to_videos[source_id] = []

            id_to_videos[source_id].append(
                video_id
            )

    # --------------------------------------------------------
    # Connect videos sharing an ID
    # --------------------------------------------------------

    for source_id, videos in id_to_videos.items():

        if len(videos) <= 1:
            continue

        first_video = videos[0]

        for video in videos[1:]:

            union_find.union(
                first_video,
                video
            )

    # --------------------------------------------------------
    # Collect connected components
    # --------------------------------------------------------

    groups = {}

    for video_id in unique_videos:

        root = union_find.find(
            video_id
        )

        if root not in groups:

            groups[root] = []

        groups[root].append(
            video_id
        )

    source_groups = list(
        groups.values()
    )

    return source_groups


# ============================================================
# Get Label for Video
# ============================================================

def get_video_labels(
    unique_videos,
    y,
    video_ids
):

    video_to_label = {}

    for video_id in unique_videos:

        indices = np.where(
            video_ids == video_id
        )[0]

        labels = np.unique(
            y[indices]
        )

        if len(labels) != 1:

            raise RuntimeError(
                f"Video has multiple labels: "
                f"{video_id} -> {labels}"
            )

        video_to_label[video_id] = int(
            labels[0]
        )

    return video_to_label


# ============================================================
# Group Statistics
# ============================================================

def print_group_statistics(
    source_groups,
    video_to_label
):

    print("\n" + "=" * 60)
    print("SOURCE GROUP STATISTICS")
    print("=" * 60)

    print(
        "Total source groups:",
        len(source_groups)
    )

    group_sizes = [
        len(group)
        for group in source_groups
    ]

    print(
        "Smallest group:",
        min(group_sizes)
    )

    print(
        "Largest group:",
        max(group_sizes)
    )

    print(
        "Average videos/group:",
        round(
            np.mean(group_sizes),
            2
        )
    )

    mixed_groups = 0

    for group in source_groups:

        labels = {
            video_to_label[video]
            for video in group
        }

        if len(labels) > 1:

            mixed_groups += 1

    print(
        "Groups containing both Real and Fake:",
        mixed_groups
    )


# ============================================================
# Create Group Labels
#
# A group can contain Real and Fake videos.
# For splitting, we calculate the majority label.
# ============================================================

def get_group_labels(
    source_groups,
    video_to_label
):

    group_labels = []

    for group in source_groups:

        labels = [
            video_to_label[video]
            for video in group
        ]

        fake_count = sum(
            label == 1
            for label in labels
        )

        real_count = sum(
            label == 0
            for label in labels
        )

        if fake_count > real_count:

            group_labels.append(1)

        else:

            group_labels.append(0)

    return np.asarray(
        group_labels,
        dtype=np.int32
    )


# ============================================================
# Convert Groups to Videos
# ============================================================

def flatten_groups(groups):

    videos = []

    for group in groups:

        videos.extend(group)

    return np.asarray(
        videos
    )


# ============================================================
# Get Sequence Indices
# ============================================================

def get_sequence_indices(
    selected_videos,
    video_ids
):

    selected_videos = set(
        selected_videos
    )

    return np.asarray(
        [
            i
            for i, video_id
            in enumerate(video_ids)
            if video_id in selected_videos
        ],
        dtype=np.int64
    )


# ============================================================
# Validate Split
# ============================================================

def validate_split(
    name,
    X,
    y,
    video_ids
):

    print("\n" + "=" * 60)
    print(f"{name} SET")
    print("=" * 60)

    print(
        "Sequences:",
        len(X)
    )

    unique_videos = np.unique(
        video_ids
    )

    print(
        "Videos:",
        len(unique_videos)
    )

    print(
        "Real sequences:",
        np.sum(y == 0)
    )

    print(
        "Fake sequences:",
        np.sum(y == 1)
    )

    if len(X) != len(y):

        raise RuntimeError(
            f"{name}: X/y mismatch"
        )

    if len(X) != len(video_ids):

        raise RuntimeError(
            f"{name}: X/video_ids mismatch"
        )


# ============================================================
# Source Overlap Check
# ============================================================

def get_all_source_ids(videos):

    source_ids = set()

    for video in videos:

        source_ids.update(
            extract_ids(video)
        )

    return source_ids


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_FILE}"
        )

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    data = np.load(
        DATASET_FILE,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]

    unique_videos = np.unique(
        video_ids
    )

    print(
        "Total sequences:",
        len(X)
    )

    print(
        "Total videos:",
        len(unique_videos)
    )

    # --------------------------------------------------------
    # Get video labels
    # --------------------------------------------------------

    video_to_label = get_video_labels(
        unique_videos,
        y,
        video_ids
    )

    # --------------------------------------------------------
    # Build source-connected groups
    # --------------------------------------------------------

    source_groups = build_source_groups(
        unique_videos
    )

    print_group_statistics(
        source_groups,
        video_to_label
    )

    # --------------------------------------------------------
    # Group labels
    # --------------------------------------------------------

    group_labels = get_group_labels(
        source_groups,
        video_to_label
    )

    source_groups = np.asarray(
        source_groups,
        dtype=object
    )

    # --------------------------------------------------------
    # First split
    #
    # 70% Train
    # 30% Temporary
    # --------------------------------------------------------

    (
        train_groups,
        temp_groups
    ) = train_test_split(

        source_groups,

        test_size=(
            VALIDATION_SIZE +
            TEST_SIZE
        ),

        random_state=RANDOM_STATE,

        stratify=group_labels
    )

    # --------------------------------------------------------
    # Temporary labels
    # --------------------------------------------------------

    group_to_label = {
        tuple(group): label

        for group, label

        in zip(
            source_groups,
            group_labels
        )
    }

    temp_labels = np.asarray(
        [
            group_to_label[
                tuple(group)
            ]

            for group in temp_groups
        ],

        dtype=np.int32
    )

    # --------------------------------------------------------
    # Validation / Test split
    # --------------------------------------------------------

    test_fraction_of_temp = (
        TEST_SIZE /
        (
            VALIDATION_SIZE +
            TEST_SIZE
        )
    )

    (
        validation_groups,
        test_groups
    ) = train_test_split(

        temp_groups,

        test_size=test_fraction_of_temp,

        random_state=RANDOM_STATE,

        stratify=temp_labels
    )

    # --------------------------------------------------------
    # Flatten groups
    # --------------------------------------------------------

    train_videos = flatten_groups(
        train_groups
    )

    validation_videos = flatten_groups(
        validation_groups
    )

    test_videos = flatten_groups(
        test_groups
    )

    # --------------------------------------------------------
    # Get sequence indices
    # --------------------------------------------------------

    train_indices = get_sequence_indices(
        train_videos,
        video_ids
    )

    validation_indices = get_sequence_indices(
        validation_videos,
        video_ids
    )

    test_indices = get_sequence_indices(
        test_videos,
        video_ids
    )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    X_train = X[train_indices]
    y_train = y[train_indices]
    video_ids_train = video_ids[train_indices]

    X_validation = X[
        validation_indices
    ]

    y_validation = y[
        validation_indices
    ]

    video_ids_validation = video_ids[
        validation_indices
    ]

    X_test = X[test_indices]
    y_test = y[test_indices]
    video_ids_test = video_ids[
        test_indices
    ]

    # --------------------------------------------------------
    # Validate splits
    # --------------------------------------------------------

    validate_split(
        "TRAIN",
        X_train,
        y_train,
        video_ids_train
    )

    validate_split(
        "VALIDATION",
        X_validation,
        y_validation,
        video_ids_validation
    )

    validate_split(
        "TEST",
        X_test,
        y_test,
        video_ids_test
    )

    # --------------------------------------------------------
    # Check exact video leakage
    # --------------------------------------------------------

    train_set = set(
        train_videos
    )

    validation_set = set(
        validation_videos
    )

    test_set = set(
        test_videos
    )

    print("\n" + "=" * 60)
    print("EXACT VIDEO LEAKAGE CHECK")
    print("=" * 60)

    print(
        "Train ∩ Validation:",
        len(
            train_set &
            validation_set
        )
    )

    print(
        "Train ∩ Test:",
        len(
            train_set &
            test_set
        )
    )

    print(
        "Validation ∩ Test:",
        len(
            validation_set &
            test_set
        )
    )

    # --------------------------------------------------------
    # Check source leakage
    # --------------------------------------------------------

    train_source_ids = get_all_source_ids(
        train_videos
    )

    validation_source_ids = (
        get_all_source_ids(
            validation_videos
        )
    )

    test_source_ids = get_all_source_ids(
        test_videos
    )

    train_validation_overlap = (
        train_source_ids &
        validation_source_ids
    )

    train_test_overlap = (
        train_source_ids &
        test_source_ids
    )

    validation_test_overlap = (
        validation_source_ids &
        test_source_ids
    )

    print("\n" + "=" * 60)
    print("SOURCE LEAKAGE CHECK")
    print("=" * 60)

    print(
        "Train ∩ Validation:",
        len(train_validation_overlap)
    )

    print(
        "Train ∩ Test:",
        len(train_test_overlap)
    )

    print(
        "Validation ∩ Test:",
        len(validation_test_overlap)
    )

    if (
        len(train_validation_overlap) != 0
        or
        len(train_test_overlap) != 0
        or
        len(validation_test_overlap) != 0
    ):

        raise RuntimeError(
            "SOURCE LEAKAGE DETECTED!"
        )

    print(
        "\nRESULT: NO SOURCE LEAKAGE DETECTED"
    )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    print("\nSaving datasets...")

    np.savez_compressed(

        OUTPUT_FOLDER / "train.npz",

        X=X_train,
        y=y_train,
        video_ids=video_ids_train
    )

    np.savez_compressed(

        OUTPUT_FOLDER / "validation.npz",

        X=X_validation,
        y=y_validation,
        video_ids=video_ids_validation
    )

    np.savez_compressed(

        OUTPUT_FOLDER / "test.npz",

        X=X_test,
        y=y_test,
        video_ids=video_ids_test
    )

    print("\n" + "=" * 60)
    print("EXP7 SOURCE-AWARE SPLIT COMPLETED")
    print("=" * 60)

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FOLDER}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
