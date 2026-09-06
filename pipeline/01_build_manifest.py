from pathlib import Path
import csv
import random


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FAKEAVCELEB_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "FakeAVCeleb"
    / "FakeAVCeleb_v1.2"
    / "FakeAVCeleb_v1.2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "manifests"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

REAL_COUNT = 500
FAKE_COUNT = 500


# ============================================================
# FIND REAL VIDEOS
# ============================================================

def get_real_videos():

    real_root = (
        FAKEAVCELEB_ROOT
        / "RealVideo-RealAudio"
    )

    videos = sorted(
        real_root.rglob("*.mp4")
    )

    return videos


# ============================================================
# FIND WAVTOLIP FAKE VIDEOS
# ============================================================

def get_fake_videos():

    videos = []

    fake_root = (
        FAKEAVCELEB_ROOT
        / "FakeVideo-FakeAudio"
    )

    for video in fake_root.rglob("*.mp4"):

        if "wavtolip" in video.name.lower():

            videos.append(video)

    return sorted(videos)


# ============================================================
# CREATE SPLIT
# ============================================================

def split_videos(videos):

    total = len(videos)

    train_end = int(
        total * 0.70
    )

    validation_end = int(
        total * 0.85
    )

    train = videos[:train_end]

    validation = videos[
        train_end:validation_end
    ]

    test = videos[
        validation_end:
    ]

    return train, validation, test


# ============================================================
# CREATE RECORD
# ============================================================

def create_records(
    videos,
    label,
    split,
    dataset,
    manipulation_type
):

    records = []

    for video in videos:

        relative_path = video.relative_to(
            FAKEAVCELEB_ROOT
        )

        video_id = (
            "_".join(
                relative_path.with_suffix(
                    ""
                ).parts
            )
        )

        record = {

            "video_id":
                video_id,

            "video_path":
                str(
                    video.resolve()
                ),

            "dataset":
                dataset,

            "label":
                label,

            "manipulation_type":
                manipulation_type,

            "split":
                split
        }

        records.append(
            record
        )

    return records

# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FINAL DATASET MANIFEST BUILDER")
    print("=" * 70)


    # --------------------------------------------------------
    # Find videos
    # --------------------------------------------------------

    real_videos = (
        get_real_videos()
    )

    fake_videos = (
        get_fake_videos()
    )


    print()
    print(
        "Available real videos:",
        len(real_videos)
    )

    print(
        "Available fake wavtolip videos:",
        len(fake_videos)
    )


    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        real_videos
    )

    rng.shuffle(
        fake_videos
    )


    # --------------------------------------------------------
    # Balance dataset
    # --------------------------------------------------------

    real_videos = real_videos[
        :REAL_COUNT
    ]

    fake_videos = fake_videos[
        :FAKE_COUNT
    ]


    print()
    print("=" * 70)
    print("SELECTED DATASET")
    print("=" * 70)

    print(
        "REAL:",
        len(real_videos)
    )

    print(
        "FAKE:",
        len(fake_videos)
    )


    # --------------------------------------------------------
    # Split real
    # --------------------------------------------------------

    (
        real_train,
        real_validation,
        real_test
    ) = split_videos(
        real_videos
    )


    # --------------------------------------------------------
    # Split fake
    # --------------------------------------------------------

    (
        fake_train,
        fake_validation,
        fake_test
    ) = split_videos(
        fake_videos
    )


    # --------------------------------------------------------
    # Create records
    # --------------------------------------------------------

    records = []


    records += create_records(

        real_train,

        label=0,

        split="train",

        dataset="FakeAVCeleb",

        manipulation_type="real"
    )


    records += create_records(

        real_validation,

        label=0,

        split="validation",

        dataset="FakeAVCeleb",

        manipulation_type="real"
    )


    records += create_records(

        real_test,

        label=0,

        split="test",

        dataset="FakeAVCeleb",

        manipulation_type="real"
    )


    records += create_records(

        fake_train,

        label=1,

        split="train",

        dataset="FakeAVCeleb",

        manipulation_type="wavtolip"
    )


    records += create_records(

        fake_validation,

        label=1,

        split="validation",

        dataset="FakeAVCeleb",

        manipulation_type="wavtolip"
    )


    records += create_records(

        fake_test,

        label=1,

        split="test",

        dataset="FakeAVCeleb",

        manipulation_type="wavtolip"
    )


    # --------------------------------------------------------
    # Save complete manifest
    # --------------------------------------------------------

    manifest_file = (
        OUTPUT_DIR
        / "dataset_manifest.csv"
    )


    fieldnames = [

        "video_id",

        "video_path",

        "dataset",

        "label",

        "manipulation_type",

        "split"
    ]


    with open(

        manifest_file,

        "w",

        newline=""

    ) as file:


        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames
        )


        writer.writeheader()


        for record in records:

            writer.writerow(
                record
            )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)


    for split in [

        "train",

        "validation",

        "test"

    ]:


        split_records = [

            record

            for record in records

            if record["split"] == split

        ]


        real_count = sum(

            record["label"] == 0

            for record in split_records

        )


        fake_count = sum(

            record["label"] == 1

            for record in split_records

        )


        print()

        print(
            split.upper()
        )

        print(
            "TOTAL:",
            len(split_records)
        )

        print(
            "REAL :",
            real_count
        )

        print(
            "FAKE :",
            fake_count
        )


    print()

    print("=" * 70)
    print("MANIFEST CREATED")
    print("=" * 70)

    print(
        manifest_file
    )


if __name__ == "__main__":

    main()
