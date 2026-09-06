from pathlib import Path

import numpy as np
import cv2

from feature_fusion.utils import (
    load_lip_coordinates
)

from feature_fusion.feature_normalization import (
    normalize_lip_coordinates
)

from configs.config import (
    LIP_COORDINATE_OUTPUT,
    OUTPUT_ROOT
)

from utils.multidataset_manager import (
    get_fakeavceleb_development,
    get_fakeavceleb_unseen
)


# ============================================================
# Configuration
# ============================================================

MFCC_HOP_LENGTH = 512
MFCC_N_FFT = 2048
AUDIO_SAMPLE_RATE = 16000

SEQUENCE_LENGTH = 30


# ============================================================
# Feature Dimensions
# ============================================================
#
# Lip coordinates:
#   40 landmarks × 2 coordinates = 80
#
# Lip velocity:
#   difference between consecutive frames = 80
#
# MFCC:
#   13 coefficients
#
# Total:
#   80 + 80 + 39 = 199
# ============================================================

LIP_FEATURE_SIZE = 80
LIP_VELOCITY_SIZE = 80
MFCC_FEATURE_SIZE = 39

FEATURE_SIZE = (
    LIP_FEATURE_SIZE
    + LIP_VELOCITY_SIZE
    + MFCC_FEATURE_SIZE
)


MFCC_FRAME_DURATION = (
    MFCC_HOP_LENGTH /
    AUDIO_SAMPLE_RATE
)


# ============================================================
# Exp5 Output Directories
# ============================================================

MFCC_EXP5_OUTPUT = (
    OUTPUT_ROOT /
    "mfcc_exp5"
)

SYNCHRONIZED_EXP5_OUTPUT = (
    OUTPUT_ROOT /
    "synchronized_aligned_exp5"
)


# ============================================================
# Video Name
# ============================================================

def get_video_name(video):

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


# ============================================================
# Dataset Video Lookup
# ============================================================

def build_video_lookup():

    lookup = {}

    dataset = get_fakeavceleb_development(
        real_count=100,
        fake_count=100
    )

    for video, label in dataset:

        video_name = get_video_name(
            video
        )

        lookup[video_name] = video

    return lookup
# ============================================================
# Get Actual Video FPS
# ============================================================

def get_video_fps(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps is None or fps <= 0:

        raise ValueError(
            f"Invalid FPS: {fps}"
        )

    if frame_count <= 0:

        raise ValueError(
            f"Invalid frame count: {frame_count}"
        )

    return float(fps), frame_count


# ============================================================
# Load MFCC
# ============================================================

def load_mfcc(video_name):

    mfcc_file = (
        MFCC_EXP5_OUTPUT
        / video_name
        / f"{video_name}.npy"
    )

    if not mfcc_file.exists():

        raise FileNotFoundError(
            f"MFCC file not found: {mfcc_file}"
        )

    mfcc = np.load(
        mfcc_file
    ).astype(
        np.float32
    )

    if mfcc.ndim != 2:

        raise ValueError(
            f"Expected MFCC shape (13, T), "
            f"got {mfcc.shape}"
        )

    if mfcc.shape[0] != MFCC_FEATURE_SIZE:

        raise ValueError(
            f"Expected {MFCC_FEATURE_SIZE} "
            f"MFCC coefficients, "
            f"got {mfcc.shape[0]}"
        )

    return mfcc


# ============================================================
# Normalize MFCC
# ============================================================

def normalize_mfcc(mfcc):

    mean = np.mean(
        mfcc,
        axis=1,
        keepdims=True
    )

    std = np.std(
        mfcc,
        axis=1,
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    normalized = (
        mfcc - mean
    ) / std

    return normalized.astype(
        np.float32
    )


# ============================================================
# Timestamp-Based MFCC Alignment
# ============================================================

def align_mfcc_to_video(
    mfcc,
    video_frame_count,
    video_fps
):
    """
    Align MFCC frames to video-frame timestamps.

    Input:
        mfcc:
            (13, T)

    Output:
        aligned MFCC:
            (video_frame_count, 13)
    """

    mfcc_frames = mfcc.shape[1]

    if mfcc_frames == 0:

        raise ValueError(
            "MFCC contains zero frames."
        )

    if video_frame_count <= 0:

        raise ValueError(
            "Video contains zero frames."
        )

    if video_fps <= 0:

        raise ValueError(
            f"Invalid video FPS: {video_fps}"
        )

    # --------------------------------------------------------
    # Video timestamps
    # --------------------------------------------------------

    video_times = (
        np.arange(
            video_frame_count,
            dtype=np.float32
        )
        / float(video_fps)
    )

    # --------------------------------------------------------
    # MFCC timestamps
    #
    # librosa uses centered STFT by default.
    #
    # Approximate center:
    #
    # (i * hop_length + n_fft / 2)
    # / sample_rate
    # --------------------------------------------------------

    mfcc_times = (
        (
            np.arange(
                mfcc_frames,
                dtype=np.float32
            )
            * MFCC_HOP_LENGTH
            + MFCC_N_FFT / 2
        )
        / AUDIO_SAMPLE_RATE
    )

    # --------------------------------------------------------
    # Find nearest MFCC frame
    # --------------------------------------------------------

    indices = np.searchsorted(
        mfcc_times,
        video_times,
        side="left"
    )

    indices = np.clip(
        indices,
        0,
        mfcc_frames - 1
    )

    previous_indices = np.maximum(
        indices - 1,
        0
    )

    next_distance = np.abs(
        mfcc_times[indices]
        -
        video_times
    )

    previous_distance = np.abs(
        mfcc_times[previous_indices]
        -
        video_times
    )

    use_previous = (
        previous_distance
        <=
        next_distance
    )

    indices[
        use_previous
    ] = previous_indices[
        use_previous
    ]

    # --------------------------------------------------------
    # Convert:
    #
    # (13, T)
    #
    # into:
    #
    # (video_frames, 13)
    # --------------------------------------------------------

    aligned = (
        mfcc[:, indices]
    ).T.astype(
        np.float32
    )

    expected_shape = (
        video_frame_count,
        MFCC_FEATURE_SIZE
    )

    if aligned.shape != expected_shape:

        raise ValueError(
            f"MFCC alignment produced "
            f"{aligned.shape}; "
            f"expected {expected_shape}"
        )

    return aligned


# ============================================================
# Load + Normalize Lip Coordinates
# ============================================================

def load_and_normalize_lips(
    video_name
):
    """
    Load all lip-coordinate CSV files.

    Each frame contains:
        40 landmarks × (X,Y)
        = 80 values

    Output:
        (frames, 80)
    """

    lip_folder = (
        LIP_COORDINATE_OUTPUT
        / video_name
    )

    if not lip_folder.exists():

        raise FileNotFoundError(
            "Lip coordinate folder not found: "
            f"{lip_folder}"
        )

    csv_files = sorted(
        lip_folder.glob("*.csv")
    )

    if not csv_files:

        raise ValueError(
            f"No lip CSV files found: "
            f"{lip_folder}"
        )

    lip_features = []

    for csv_file in csv_files:

        coordinates = (
            load_lip_coordinates(
                csv_file
            )
        )

        if coordinates.shape != (
            40,
            2
        ):

            raise ValueError(
                f"Invalid lip coordinate shape "
                f"in {csv_file}: "
                f"{coordinates.shape}"
            )

        normalized = (
            normalize_lip_coordinates(
                coordinates
            )
        )

        flattened = (
            normalized.flatten()
        )

        if flattened.shape != (
            LIP_FEATURE_SIZE,
        ):

            raise ValueError(
                f"Expected "
                f"{LIP_FEATURE_SIZE} lip features, "
                f"got {flattened.shape}"
            )

        lip_features.append(
            flattened.astype(
                np.float32
            )
        )

    return np.asarray(
        lip_features,
        dtype=np.float32
    )


# ============================================================
# Compute Lip Velocity
# ============================================================

def compute_lip_velocity(
    lip_features
):
    """
    Compute frame-to-frame lip movement.

    velocity[t] =
        lip[t] - lip[t-1]

    The first frame has no previous frame,
    therefore its velocity is zero.

    Input:
        (frames, 80)

    Output:
        (frames, 80)
    """

    if lip_features.ndim != 2:

        raise ValueError(
            f"Expected 2D lip features, "
            f"got {lip_features.shape}"
        )

    if lip_features.shape[1] != LIP_FEATURE_SIZE:

        raise ValueError(
            f"Expected {LIP_FEATURE_SIZE} lip "
            f"features, got "
            f"{lip_features.shape[1]}"
        )

    velocity = np.zeros_like(
        lip_features,
        dtype=np.float32
    )

    if len(lip_features) > 1:

        velocity[1:] = (
            lip_features[1:]
            -
            lip_features[:-1]
        )

    return velocity


# ============================================================
# Fuse Position + Velocity + MFCC
# ============================================================

def fuse_features(
    lip_features,
    lip_velocity,
    aligned_mfcc
):
    """
    Create the final multimodal feature vector.

    Each frame contains:

        80 lip positions
        80 lip velocities
        39 audio features (MFCC + Delta + Delta-Delta)

        -----------------
        173 total features
    """

    if len(lip_features) != len(
        lip_velocity
    ):

        raise ValueError(
            "Lip and velocity frame counts "
            "do not match."
        )

    if len(lip_features) != len(
        aligned_mfcc
    ):

        raise ValueError(
            "Lip and MFCC frame counts "
            "do not match."
        )

    if lip_features.shape[1] != (
        LIP_FEATURE_SIZE
    ):

        raise ValueError(
            f"Expected lip shape "
            f"(T, {LIP_FEATURE_SIZE}), "
            f"got {lip_features.shape}"
        )

    if lip_velocity.shape[1] != (
        LIP_VELOCITY_SIZE
    ):

        raise ValueError(
            f"Expected velocity shape "
            f"(T, {LIP_VELOCITY_SIZE}), "
            f"got {lip_velocity.shape}"
        )

    if aligned_mfcc.shape[1] != (
        MFCC_FEATURE_SIZE
    ):

        raise ValueError(
            f"Expected MFCC shape "
            f"(T, {MFCC_FEATURE_SIZE}), "
            f"got {aligned_mfcc.shape}"
        )

    fused = np.concatenate(
        [
            lip_features,
            lip_velocity,
            aligned_mfcc
        ],
        axis=1
    )

    if fused.shape[1] != FEATURE_SIZE:

        raise ValueError(
            f"Expected fused feature size "
            f"{FEATURE_SIZE}, "
            f"got {fused.shape[1]}"
        )

    if not np.isfinite(
        fused
    ).all():

        raise ValueError(
            "Fused features contain "
            "NaN or Inf values."
        )

    return fused.astype(
        np.float32
    )


# ============================================================
# Process One Video
# ============================================================

def process_video(
    video_name,
    video_path
):

    print(
        f"\nProcessing: {video_name}"
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps, original_frame_count = (
        get_video_fps(
            video_path
        )
    )

    print(
        f"Actual FPS   : {fps:.4f}"
    )

    print(
        f"Video frames : {original_frame_count}"
    )

    # --------------------------------------------------------
    # Lip positions
    # --------------------------------------------------------

    lip_features = (
        load_and_normalize_lips(
            video_name
        )
    )

    lip_frame_count = len(
        lip_features
    )

    print(
        f"Lip frames   : "
        f"{lip_frame_count}"
    )

    if lip_frame_count == 0:

        raise ValueError(
            "No lip features found."
        )

    # --------------------------------------------------------
    # Frame count sanity check
    # --------------------------------------------------------

    if abs(
        lip_frame_count
        -
        original_frame_count
    ) > 2:

        print(
            "⚠️ Warning: extracted frame "
            "count differs from source."
        )

        print(
            f"   Source : "
            f"{original_frame_count}"
        )

        print(
            f"   Lips   : "
            f"{lip_frame_count}"
        )

    # --------------------------------------------------------
    # Lip velocity
    # --------------------------------------------------------

    lip_velocity = (
        compute_lip_velocity(
            lip_features
        )
    )

    print(
        f"Lip velocity: "
        f"{lip_velocity.shape}"
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = load_mfcc(
        video_name
    )

    print(
        f"MFCC frames : "
        f"{mfcc.shape[1]}"
    )

    # --------------------------------------------------------
    # Normalize MFCC
    # --------------------------------------------------------

    mfcc = normalize_mfcc(
        mfcc
    )

    print(
        "MFCC normalization : ✅"
    )

    # --------------------------------------------------------
    # Timestamp alignment
    # --------------------------------------------------------

    aligned_mfcc = (
        align_mfcc_to_video(
            mfcc,
            lip_frame_count,
            fps
        )
    )

    print(
        f"Aligned MFCC : "
        f"{aligned_mfcc.shape}"
    )

    # --------------------------------------------------------
    # Feature fusion
    # --------------------------------------------------------

    fused = fuse_features(
        lip_features,
        lip_velocity,
        aligned_mfcc
    )

    print(
        f"Fused shape  : "
        f"{fused.shape}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_folder = (
        SYNCHRONIZED_EXP5_OUTPUT
        / video_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove old feature files
    for old_file in output_folder.glob(
        "*.npy"
    ):

        old_file.unlink()

    # --------------------------------------------------------
    # Save one feature vector per frame
    # --------------------------------------------------------

    for index, feature in enumerate(
        fused
    ):

        output_file = (
            output_folder
            / f"frame_{index:04d}.npy"
        )

        np.save(
            output_file,
            feature
        )

    print(
        f"Saved {len(fused)} feature files."
    )

    return fused.shape


# ============================================================
# Process Development Dataset
# ============================================================

def process_development_dataset():

    dataset = get_fakeavceleb_development(
        real_count=100,
        fake_count=100
    )

    print("\n==============================")
    print("DEVELOPMENT FEATURE FUSION")
    print("==============================")

    print(
        "Videos       :",
        len(dataset)
    )

    print(
        "Real videos  :",
        sum(
            label == 0
            for _, label in dataset
        )
    )

    print(
        "Fake videos  :",
        sum(
            label == 1
            for _, label in dataset
        )
    )

    print(
        "Feature size :",
        FEATURE_SIZE
    )

    print("==============================")

    successful = 0
    failed = 0
    skipped = 0

    for index, (video, label) in enumerate(
        dataset,
        start=1
    ):

        video_name = get_video_name(
            video
        )

        label_name = (
            "Real"
            if label == 0
            else "Fake"
        )

        print(
            f"\n[{index}/{len(dataset)}] "
            f"{video_name}"
        )

        print(
            f"Label : {label_name}"
        )

        # ----------------------------------------------------
        # Check whether this video is already completed
        # ----------------------------------------------------

        output_folder = (
            SYNCHRONIZED_EXP5_OUTPUT
            / video_name
        )

        existing_features = (
            list(
                output_folder.glob(
                    "frame_*.npy"
                )
            )
            if output_folder.exists()
            else []
        )

        if len(existing_features) >= SEQUENCE_LENGTH:

            print(
                f"⏭️ Already completed. "
                f"Skipping ({len(existing_features)} frames)"
            )

            successful += 1
            skipped += 1

            continue

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        try:

            shape = process_video(
                video_name,
                video
            )

            print(
                f"✅ Completed: {shape}"
            )

            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"❌ Failed: {video_name}"
            )

            print(
                f"   Error: {error}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==============================")
    print("DEVELOPMENT FEATURE FUSION COMPLETE")
    print("==============================")

    print(
        "Videos processed :",
        len(dataset)
    )

    print(
        "Successful       :",
        successful
    )

    print(
        "Skipped          :",
        skipped
    )

    print(
        "Failed           :",
        failed
    )

    print(
        "Feature size     :",
        FEATURE_SIZE
    )

    print("==============================")


# ============================================================
# Process Unseen Dataset
# ============================================================

def process_unseen_dataset():

    dataset = get_fakeavceleb_unseen(
        real_count=100,
        fake_count=100
    )

    print("\n==============================")
    print("UNSEEN FEATURE FUSION")
    print("==============================")

    print(
        "Videos       :",
        len(dataset)
    )

    print(
        "Real videos  :",
        sum(
            label == 0
            for _, label in dataset
        )
    )

    print(
        "Fake videos  :",
        sum(
            label == 1
            for _, label in dataset
        )
    )

    print(
        "Feature size :",
        FEATURE_SIZE
    )

    print("==============================")

    successful = 0
    failed = 0
    skipped = 0

    for index, (video, label) in enumerate(
        dataset,
        start=1
    ):

        video_name = get_video_name(
            video
        )

        label_name = (
            "Real"
            if label == 0
            else "Fake"
        )

        print(
            f"\n[{index}/{len(dataset)}] "
            f"{video_name}"
        )

        print(
            f"Label : {label_name}"
        )

        # ----------------------------------------------------
        # Check whether already completed
        # ----------------------------------------------------

        output_folder = (
            SYNCHRONIZED_EXP5_OUTPUT
            / video_name
        )

        existing_features = (
            list(
                output_folder.glob(
                    "frame_*.npy"
                )
            )
            if output_folder.exists()
            else []
        )

        if len(existing_features) >= SEQUENCE_LENGTH:

            print(
                f"⏭️ Already completed. "
                f"Skipping ({len(existing_features)} frames)"
            )

            successful += 1
            skipped += 1

            continue

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        try:

            shape = process_video(
                video_name,
                video
            )

            print(
                f"✅ Completed: {shape}"
            )

            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"❌ Failed: {video_name}"
            )

            print(
                f"   Error: {error}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==============================")
    print("UNSEEN FEATURE FUSION COMPLETE")
    print("==============================")

    print(
        "Videos processed :",
        len(dataset)
    )

    print(
        "Successful       :",
        successful
    )

    print(
        "Skipped          :",
        skipped
    )

    print(
        "Failed           :",
        failed
    )

    print(
        "Feature size     :",
        FEATURE_SIZE
    )

    print("==============================")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    process_development_dataset()



