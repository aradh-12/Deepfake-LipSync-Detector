import numpy as np
import re


DATASET_DIR = "outputs/dataset_exp5_full"


def load_video_ids(filename):

    data = np.load(
        f"{DATASET_DIR}/{filename}",
        allow_pickle=True
    )

    return np.unique(data["video_ids"])


def extract_ids(video_name):

    return set(
        re.findall(r"id\d+", str(video_name))
    )


def find_videos_with_id(video_ids, source_id):

    matches = []

    for video in video_ids:

        ids = extract_ids(video)

        if source_id in ids:
            matches.append(video)

    return matches


print("\n" + "=" * 80)
print("SOURCE ID OVERLAP INVESTIGATION")
print("=" * 80)


train_videos = load_video_ids("train.npz")
val_videos = load_video_ids("validation.npz")
test_videos = load_video_ids("test.npz")


def get_all_ids(videos):

    all_ids = set()

    for video in videos:
        all_ids.update(extract_ids(video))

    return all_ids


train_ids = get_all_ids(train_videos)
val_ids = get_all_ids(val_videos)
test_ids = get_all_ids(test_videos)


comparisons = {

    "TRAIN vs VALIDATION":
        (
            train_ids.intersection(val_ids),
            train_videos,
            val_videos
        ),

    "TRAIN vs TEST":
        (
            train_ids.intersection(test_ids),
            train_videos,
            test_videos
        ),

    "VALIDATION vs TEST":
        (
            val_ids.intersection(test_ids),
            val_videos,
            test_videos
        )
}


for comparison_name, data in comparisons.items():

    overlap_ids, split_a_videos, split_b_videos = data

    print("\n" + "=" * 80)
    print(comparison_name)
    print("=" * 80)

    for source_id in sorted(overlap_ids):

        print("\n" + "-" * 80)
        print(f"OVERLAPPING ID: {source_id}")
        print("-" * 80)

        print("\nFirst split videos:")

        videos_a = find_videos_with_id(
            split_a_videos,
            source_id
        )

        for video in videos_a:
            print(video)

        print("\nSecond split videos:")

        videos_b = find_videos_with_id(
            split_b_videos,
            source_id
        )

        for video in videos_b:
            print(video)


print("\n" + "=" * 80)
print("INVESTIGATION COMPLETED")
print("=" * 80)
