from pathlib import Path

import numpy as np

from utils.multidataset_manager import (
    get_fakeavceleb_development,
    get_fakeavceleb_unseen
)


# ============================================================
# Configuration
# ============================================================

FEATURE_ROOT = Path(
    "outputs/synchronized_aligned_exp5"
)

DEVELOPMENT_OUTPUT = Path(
    "outputs/development_sequences_exp5"
)

UNSEEN_OUTPUT = Path(
    "outputs/unseen_sequences"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 199


# ============================================================
# Video Name
# ============================================================

def get_video_name(video):
    """
    Must match the naming convention used by
    feature_fusion.feature_sync.
    """

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


# ============================================================
# Build Dataset Lookup
# ============================================================

def build_video_lookup():

    lookup = {}

    dataset = get_fakeavceleb_development(
        real_count=100,
        fake_count=100
    )

    for video, label in dataset:

        video_name = get_video_name(video)

        lookup[video_name] = video

    return lookup


# ============================================================
# Create Sequences From One Video
# ============================================================

def create_sequences(
    video_folder,
    label
):

    feature_files = sorted(
        video_folder.glob("frame_*.npy")
    )

    if len(feature_files) < SEQUENCE_LENGTH:

        return np.empty(
            (0, SEQUENCE_LENGTH, FEATURE_SIZE),
            dtype=np.float32
        )

    frames = []

    for feature_file in feature_files:

        feature = np.load(
            feature_file
        )

        if feature.shape != (
            FEATURE_SIZE,
        ):

            raise ValueError(
                f"Invalid feature shape in "
                f"{feature_file}: "
                f"{feature.shape}. "
                f"Expected ({FEATURE_SIZE},)"
            )

        if np.isnan(feature).any():

            raise ValueError(
                f"NaN detected in {feature_file}"
            )

        if np.isinf(feature).any():

            raise ValueError(
                f"Inf detected in {feature_file}"
            )

        frames.append(
            feature.astype(np.float32)
        )

    frames = np.asarray(
        frames,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Non-overlapping 30-frame sequences
    # --------------------------------------------------------

    sequence_count = (
        len(frames)
        // SEQUENCE_LENGTH
    )

    usable_frames = (
        sequence_count
        * SEQUENCE_LENGTH
    )

    if sequence_count == 0:

        return np.empty(
            (0, SEQUENCE_LENGTH, FEATURE_SIZE),
            dtype=np.float32
        )

    frames = frames[
        :usable_frames
    ]

    sequences = frames.reshape(
        sequence_count,
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    )

    return sequences


# ============================================================
# Process One Dataset Group
# ============================================================

def process_dataset(
    dataset,
    output_root,
    dataset_name
):
    """
    Build 30-frame sequences from existing Exp5 feature folders.

    dataset format:
        [
            (video_name, label),
            ...
        ]

    The video_name must correspond exactly to a folder
    inside FEATURE_ROOT.
    """

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset is already in:
    #
    # (video_name, label)
    #
    # Do NOT call build_dataset_lookup() here because that
    # function expects Path objects.
    # --------------------------------------------------------

    lookup = {
        video_name: int(label)
        for video_name, label in dataset
    }

    print()
    print("=" * 70)
    print(f"{dataset_name} SEQUENCE GENERATION")
    print("=" * 70)

    print(
        "Videos in dataset :",
        len(lookup)
    )

    print(
        "Real videos       :",
        sum(label == 0 for label in lookup.values())
    )

    print(
        "Fake videos       :",
        sum(label == 1 for label in lookup.values())
    )

    print(
        "Sequence length   :",
        SEQUENCE_LENGTH
    )

    print(
        "Feature size      :",
        FEATURE_SIZE
    )

    print("=" * 70)

    successful_videos = 0
    skipped_videos = 0

    real_sequences = 0
    fake_sequences = 0

    all_sequences = []
    all_labels = []
    all_video_ids = []

    # --------------------------------------------------------
    # Process every existing Exp5 feature folder
    # --------------------------------------------------------

    for index, (video_name, label) in enumerate(
        sorted(lookup.items()),
        start=1
    ):

        print()
        print(
            f"[{index}/{len(lookup)}] "
            f"{video_name}"
        )

        print(
            "Label :",
            "Real" if label == 0 else "Fake"
        )

        video_folder = (
            FEATURE_ROOT
            / video_name
        )

        if not video_folder.exists():

            print(
                "❌ Missing Exp5 feature folder"
            )

            skipped_videos += 1
            continue

        sequences = create_sequences(
            video_folder,
            label
        )

        if len(sequences) == 0:

            print(
                "⚠️ Not enough frames "
                f"for {SEQUENCE_LENGTH}-frame sequence"
            )

            skipped_videos += 1
            continue

        output_file = (
            output_root
            / f"{video_name}.npz"
        )

        np.savez_compressed(
            output_file,
            X=sequences,
            y=np.full(
                len(sequences),
                label,
                dtype=np.int32
            ),
            video_ids=np.array(
                [video_name] * len(sequences),
                dtype=object
            )
        )

        sequence_count = len(sequences)

        successful_videos += 1

        if label == 0:
            real_sequences += sequence_count
        else:
            fake_sequences += sequence_count

        all_sequences.append(sequences)

        all_labels.append(
            np.full(
                sequence_count,
                label,
                dtype=np.int32
            )
        )

        all_video_ids.append(
            np.array(
                [video_name] * sequence_count,
                dtype=object
            )
        )

        print(
            "Frames    :",
            len(
                list(
                    video_folder.glob(
                        "frame_*.npy"
                    )
                )
            )
        )

        print(
            "Sequences :",
            sequence_count
        )

        print(
            "Saved     :",
            output_file
        )

    # --------------------------------------------------------
    # Combine complete dataset
    # --------------------------------------------------------

    if all_sequences:

        X = np.concatenate(
            all_sequences,
            axis=0
        )

        y = np.concatenate(
            all_labels,
            axis=0
        )

        video_ids = np.concatenate(
            all_video_ids,
            axis=0
        )

    else:

        X = np.empty(
            (0, SEQUENCE_LENGTH, FEATURE_SIZE),
            dtype=np.float32
        )

        y = np.empty(
            (0,),
            dtype=np.int32
        )

        video_ids = np.empty(
            (0,),
            dtype=object
        )

    combined_file = (
        output_root
        / "all_sequences.npz"
    )

    np.savez_compressed(
        combined_file,
        X=X,
        y=y,
        video_ids=video_ids
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"{dataset_name} SEQUENCE GENERATION COMPLETE")
    print("=" * 70)

    print(
        "Dataset videos     :",
        len(lookup)
    )

    print(
        "Successful videos  :",
        successful_videos
    )

    print(
        "Skipped videos     :",
        skipped_videos
    )

    print(
        "Total sequences    :",
        len(X)
    )

    print(
        "Real sequences     :",
        real_sequences
    )

    print(
        "Fake sequences     :",
        fake_sequences
    )

    print(
        "Final X shape      :",
        X.shape
    )

    print(
        "Final y shape      :",
        y.shape
    )

    print(
        "Video IDs shape    :",
        video_ids.shape
    )

    print(
        "Saved combined     :",
        combined_file
    )

    print("=" * 70)

    return X, y, video_ids


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Build label lookup from the FULL FakeAVCeleb dataset
    #
    # IMPORTANT:
    # We use the actual Exp5 feature folders that already
    # exist on disk. We do NOT randomly select another
    # development subset.
    # --------------------------------------------------------

    from utils.multidataset_manager import get_fakeavceleb_all

    real_videos, fake_videos = get_fakeavceleb_all()

    dataset_lookup = {}

    for video in real_videos:
        dataset_lookup[get_video_name(video)] = 0

    for video in fake_videos:
        dataset_lookup[get_video_name(video)] = 1

    # --------------------------------------------------------
    # Use only videos for which Exp5 features exist
    # --------------------------------------------------------

    feature_folders = sorted(
        p
        for p in FEATURE_ROOT.iterdir()
        if p.is_dir()
    )

    selected_dataset = []

    for folder in feature_folders:

        video_name = folder.name

        if video_name not in dataset_lookup:

            print(
                f"⚠️ No label found for: {video_name}"
            )

            continue

        label = dataset_lookup[video_name]

        selected_dataset.append(
            (video_name, label)
        )

    print()
    print("=" * 70)
    print("EXP5 DATASET SELECTION")
    print("=" * 70)

    print(
        "Exp5 feature folders :",
        len(feature_folders)
    )

    print(
        "Matched videos       :",
        len(selected_dataset)
    )

    print(
        "Real videos          :",
        sum(label == 0 for _, label in selected_dataset)
    )

    print(
        "Fake videos          :",
        sum(label == 1 for _, label in selected_dataset)
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Process directly using existing feature-folder names
    # --------------------------------------------------------

    lookup = {
        video_name: label
        for video_name, label in selected_dataset
    }

    process_dataset(
        selected_dataset,
        DEVELOPMENT_OUTPUT,
        "DEVELOPMENT EXP5"
    )


if __name__ == "__main__":

    main()
