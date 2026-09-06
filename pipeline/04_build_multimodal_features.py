from pathlib import Path
import csv

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


FINAL_PIPELINE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
)


MANIFEST_PATH = (
    FINAL_PIPELINE_ROOT
    / "manifests"
    / "dataset_manifest.csv"
)


LIP_FEATURE_ROOT = (
    FINAL_PIPELINE_ROOT
    / "lip_features"
)


MFCC_FEATURE_ROOT = (
    FINAL_PIPELINE_ROOT
    / "mfcc_features"
)


MULTIMODAL_OUTPUT_ROOT = (
    FINAL_PIPELINE_ROOT
    / "multimodal_features"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

LIP_FEATURE_DIM = 80

MFCC_FEATURE_DIM = 39

MULTIMODAL_FEATURE_DIM = (
    LIP_FEATURE_DIM
    + MFCC_FEATURE_DIM
)


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(

            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"

        )

    rows = []

    with open(

        MANIFEST_PATH,
        "r",
        newline="",
        encoding="utf-8"

    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            rows.append(
                row
            )

    return rows


# ============================================================
# GET LIP FEATURE PATH
# ============================================================

def get_lip_feature_path(

    split,
    video_id

):

    return (

        LIP_FEATURE_ROOT
        / split
        / f"{video_id}.npz"

    )


# ============================================================
# GET MFCC FEATURE PATH
# ============================================================

def get_mfcc_feature_path(

    split,
    video_id

):

    return (

        MFCC_FEATURE_ROOT
        / split
        / f"{video_id}.npy"

    )


# ============================================================
# GET MULTIMODAL OUTPUT PATH
# ============================================================

def get_multimodal_output_path(

    split,
    video_id

):

    return (

        MULTIMODAL_OUTPUT_ROOT
        / split
        / f"{video_id}.npz"

    )


# ============================================================
# LOAD LIP FEATURES
# ============================================================

def load_lip_features(

    feature_path

):

    data = np.load(
        feature_path
    )

    # --------------------------------------------------------
    # Get first array from NPZ
    # --------------------------------------------------------

    if len(data.files) == 0:

        raise ValueError(
            "Lip feature NPZ is empty"
        )

    # --------------------------------------------------------
    # Most common saved format
    # --------------------------------------------------------

    if "features" in data.files:

        features = data["features"]

    else:

        first_key = data.files[0]

        features = data[
            first_key
        ]

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if features.ndim != 2:

        raise ValueError(

            f"Invalid lip feature dimensions: "
            f"{features.shape}"

        )

    if features.shape[0] == 0:

        raise ValueError(
            "Lip features contain zero frames"
        )

    if features.shape[1] != LIP_FEATURE_DIM:

        raise ValueError(

            f"Expected lip feature dimension "
            f"{LIP_FEATURE_DIM}, "
            f"got {features.shape}"

        )

    if not np.isfinite(
        features
    ).all():

        raise ValueError(
            "Lip features contain NaN or Inf"
        )

    return features


# ============================================================
# LOAD MFCC FEATURES
# ============================================================

def load_mfcc_features(

    feature_path

):

    features = np.load(
        feature_path
    )

    features = np.asarray(

        features,

        dtype=np.float32

    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if features.ndim != 2:

        raise ValueError(

            f"Invalid MFCC dimensions: "
            f"{features.shape}"

        )

    if features.shape[0] == 0:

        raise ValueError(
            "MFCC contains zero frames"
        )

    if features.shape[1] != MFCC_FEATURE_DIM:

        raise ValueError(

            f"Expected MFCC feature dimension "
            f"{MFCC_FEATURE_DIM}, "
            f"got {features.shape}"

        )

    if not np.isfinite(
        features
    ).all():

        raise ValueError(
            "MFCC contains NaN or Inf"
        )

    return features


# ============================================================
# TEMPORAL ALIGNMENT
# ============================================================

def align_mfcc_to_lip(

    mfcc_features,
    target_frames

):

    source_frames = (

        mfcc_features.shape[0]

    )

    # --------------------------------------------------------
    # Same number of frames
    # --------------------------------------------------------

    if source_frames == target_frames:

        return mfcc_features

    # --------------------------------------------------------
    # Single MFCC frame
    # --------------------------------------------------------

    if source_frames == 1:

        aligned = np.repeat(

            mfcc_features,

            target_frames,

            axis=0

        )

        return aligned

    # --------------------------------------------------------
    # Interpolation
    # --------------------------------------------------------

    source_positions = np.linspace(

        0.0,

        1.0,

        source_frames

    )

    target_positions = np.linspace(

        0.0,

        1.0,

        target_frames

    )

    aligned = np.zeros(

        (

            target_frames,

            MFCC_FEATURE_DIM

        ),

        dtype=np.float32

    )

    # --------------------------------------------------------
    # Interpolate every MFCC feature
    # --------------------------------------------------------

    for feature_index in range(

        MFCC_FEATURE_DIM

    ):

        aligned[:, feature_index] = (

            np.interp(

                target_positions,

                source_positions,

                mfcc_features[
                    :,
                    feature_index
                ]

            )

        )

    return aligned


# ============================================================
# BUILD MULTIMODAL FEATURES
# ============================================================

def build_multimodal_features(

    lip_features,
    mfcc_features

):

    lip_frames = (

        lip_features.shape[0]

    )

    # --------------------------------------------------------
    # Align MFCC to lip frames
    # --------------------------------------------------------

    aligned_mfcc = (

        align_mfcc_to_lip(

            mfcc_features,

            lip_frames

        )

    )

    # --------------------------------------------------------
    # Concatenate
    #
    # Lip:
    # (T, 80)
    #
    # MFCC:
    # (T, 39)
    #
    # Result:
    # (T, 119)
    # --------------------------------------------------------

    multimodal_features = (

        np.concatenate(

            [

                lip_features,

                aligned_mfcc

            ],

            axis=1

        )

    )

    multimodal_features = (

        multimodal_features.astype(

            np.float32

        )

    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if multimodal_features.ndim != 2:

        raise ValueError(

            f"Invalid multimodal dimensions: "
            f"{multimodal_features.shape}"

        )

    if multimodal_features.shape[1] != (

        MULTIMODAL_FEATURE_DIM

    ):

        raise ValueError(

            f"Expected multimodal dimension "
            f"{MULTIMODAL_FEATURE_DIM}, "
            f"got "
            f"{multimodal_features.shape}"

        )

    if multimodal_features.shape[0] == 0:

        raise ValueError(

            "Multimodal features contain zero frames"

        )

    if not np.isfinite(

        multimodal_features

    ).all():

        raise ValueError(

            "Multimodal features contain NaN or Inf"

        )

    return multimodal_features


# ============================================================
# PROCESS SINGLE VIDEO
# ============================================================

def process_video(

    row

):

    split = row[
        "split"
    ]

    video_id = row[
        "video_id"
    ]

    label = int(
        row["label"]
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    lip_path = (

        get_lip_feature_path(

            split,

            video_id

        )

    )

    mfcc_path = (

        get_mfcc_feature_path(

            split,

            video_id

        )

    )

    output_path = (

        get_multimodal_output_path(

            split,

            video_id

        )

    )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not lip_path.exists():

        return (

            "NO_LIP",

            "Lip feature missing"

        )

    if not mfcc_path.exists():

        return (

            "NO_MFCC",

            "MFCC feature missing"

        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    # --------------------------------------------------------
    # Check existing output
    # --------------------------------------------------------

    if output_path.exists():

        try:

            existing_data = np.load(
                output_path
            )

            if (

                "features"
                in existing_data.files

            ):

                existing_features = (

                    existing_data[
                        "features"
                    ]

                )

                if (

                    existing_features.ndim == 2

                    and

                    existing_features.shape[1]
                    ==
                    MULTIMODAL_FEATURE_DIM

                    and

                    existing_features.shape[0] > 0

                    and

                    np.isfinite(
                        existing_features
                    ).all()

                ):

                    return (

                        "EXISTS",

                        existing_features.shape

                    )

        except Exception:

            pass

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    lip_features = (

        load_lip_features(

            lip_path

        )

    )

    mfcc_features = (

        load_mfcc_features(

            mfcc_path

        )

    )

    # --------------------------------------------------------
    # Build multimodal features
    # --------------------------------------------------------

    multimodal_features = (

        build_multimodal_features(

            lip_features,

            mfcc_features

        )

    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez_compressed(

        output_path,

        features=multimodal_features,

        label=np.int32(label),

        video_id=video_id,

        lip_frames=np.int32(
            lip_features.shape[0]
        ),

        mfcc_frames=np.int32(
            mfcc_features.shape[0]
        )

    )

    return (

        "SAVED",

        multimodal_features.shape

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL PIPELINE — MULTIMODAL FEATURE BUILDING"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    MULTIMODAL_OUTPUT_ROOT.mkdir(

        parents=True,

        exist_ok=True

    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    rows = load_manifest()

    print(
        "Manifest:"
    )

    print(
        MANIFEST_PATH
    )

    print()

    print(
        "Total manifest videos:",
        len(rows)
    )

    print()

    print(
        "Lip feature directory:"
    )

    print(
        LIP_FEATURE_ROOT
    )

    print()

    print(
        "MFCC feature directory:"
    )

    print(
        MFCC_FEATURE_ROOT
    )

    print()

    print(
        "Multimodal output directory:"
    )

    print(
        MULTIMODAL_OUTPUT_ROOT
    )

    print()

    print(
        "Feature dimensions:"
    )

    print(
        "Lip:",
        LIP_FEATURE_DIM
    )

    print(
        "MFCC:",
        MFCC_FEATURE_DIM
    )

    print(
        "Multimodal:",
        MULTIMODAL_FEATURE_DIM
    )

    print()

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total = len(
        rows
    )

    saved = 0

    already_exists = 0

    no_lip = 0

    no_mfcc = 0

    failed = 0

    failures = []

    # --------------------------------------------------------
    # Process all videos
    # --------------------------------------------------------

    for index, row in enumerate(

        rows,

        start=1

    ):

        split = row[
            "split"
        ]

        video_id = row[
            "video_id"
        ]

        label = int(
            row["label"]
        )

        label_name = (

            "REAL"

            if label == 0

            else "FAKE"

        )

        print()

        print(

            f"[{index}/{total}] "

            f"Processing | "

            f"{split.upper()} | "

            f"{label_name}"

        )

        print(

            f"Video: "

            f"{video_id}"

        )

        try:

            status, info = (

                process_video(

                    row

                )

            )

            if status == "SAVED":

                saved += 1

                print(

                    "✅ SAVED | "

                    f"Shape: {info}"

                )

            elif status == "EXISTS":

                already_exists += 1

                print(

                    "⏭ ALREADY EXISTS | "

                    f"Shape: {info}"

                )

            elif status == "NO_LIP":

                no_lip += 1

                print(

                    "⚠ SKIPPED | "

                    "Lip feature missing"

                )

            elif status == "NO_MFCC":

                no_mfcc += 1

                print(

                    "⚠ SKIPPED | "

                    "MFCC feature missing"

                )

            else:

                failed += 1

                failures.append(

                    (

                        video_id,

                        str(info)

                    )

                )

                print(
                    "❌ FAILED"
                )

                print(
                    str(info)
                )

        except Exception as error:

            failed += 1

            failures.append(

                (

                    video_id,

                    str(error)

                )

            )

            print(
                "❌ FAILED"
            )

            print(
                str(error)
            )

    # ========================================================
    # SAVE FAILURE LOG
    # ========================================================

    failure_log = (

        FINAL_PIPELINE_ROOT
        / "multimodal_failures.csv"

    )

    with open(

        failure_log,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(

            [

                "video_id",

                "error"

            ]

        )

        writer.writerows(
            failures
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "MULTIMODAL FEATURE BUILDING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Total manifest videos:",
        total
    )

    print(
        "Successfully processed:",
        saved
    )

    print(
        "Already existed:",
        already_exists
    )

    print(
        "Skipped (no lip feature):",
        no_lip
    )

    print(
        "Skipped (no MFCC feature):",
        no_mfcc
    )

    print(
        "Failed:",
        failed
    )

    print()

    print(
        "Output directory:"
    )

    print(
        MULTIMODAL_OUTPUT_ROOT
    )

    print()

    print(
        "Failure log:"
    )

    print(
        failure_log
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()