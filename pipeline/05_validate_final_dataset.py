from pathlib import Path
import csv
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

MULTIMODAL_FEATURE_ROOT = (
    FINAL_PIPELINE_ROOT
    / "multimodal_features"
)

VALID_MANIFEST_PATH = (
    FINAL_PIPELINE_ROOT
    / "manifests"
    / "valid_dataset_manifest.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FEATURE_DIM = 119


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest():

    rows = []

    with open(
        MANIFEST_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append(row)

    return rows


# ============================================================
# GET MULTIMODAL FEATURE PATH
# ============================================================

def get_feature_path(
    split,
    video_id
):

    return (
        MULTIMODAL_FEATURE_ROOT
        / split
        / f"{video_id}.npz"
    )


# ============================================================
# VALIDATE FEATURE
# ============================================================

def validate_feature(
    feature_path
):

    try:

        data = np.load(
            feature_path
        )

        # ----------------------------------------------------
        # Get first available array
        # ----------------------------------------------------

        if len(data.files) == 0:

            return (
                False,
                "NPZ contains no arrays"
            )

        # Prefer "features" if available
        if "features" in data.files:

            features = data["features"]

        else:

            features = data[
                data.files[0]
            ]

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if features.ndim != 2:

            return (
                False,
                f"Invalid dimensions: {features.shape}"
            )

        if features.shape[0] == 0:

            return (
                False,
                "Zero temporal frames"
            )

        if features.shape[1] != EXPECTED_FEATURE_DIM:

            return (
                False,
                f"Expected {EXPECTED_FEATURE_DIM} features, "
                f"got {features.shape}"
            )

        if not np.isfinite(
            features
        ).all():

            return (
                False,
                "Contains NaN or Inf"
            )

        return (
            True,
            features.shape
        )

    except Exception as error:

        return (
            False,
            str(error)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "FINAL MULTIMODAL DATASET VALIDATION"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    rows = load_manifest()

    print(
        "Total manifest videos:",
        len(rows)
    )

    print()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_rows = []

    invalid_rows = []

    split_stats = {

        "train": 0,

        "validation": 0,

        "test": 0

    }

    label_stats = {

        "real": 0,

        "fake": 0

    }

    # --------------------------------------------------------
    # Process each sample
    # --------------------------------------------------------

    for index, row in enumerate(
        rows,
        start=1
    ):

        split = row["split"]

        video_id = row["video_id"]

        label = int(
            row["label"]
        )

        print(

            f"[{index}/{len(rows)}] "

            f"Checking | "

            f"{split.upper()} | "

            f"{video_id}"
        )

        feature_path = get_feature_path(
            split,
            video_id
        )

        # ----------------------------------------------------
        # Check file exists
        # ----------------------------------------------------

        if not feature_path.exists():

            invalid_rows.append(

                {
                    "video_id": video_id,
                    "reason": "Feature file missing"
                }
            )

            print(
                "❌ MISSING"
            )

            continue

        # ----------------------------------------------------
        # Validate feature
        # ----------------------------------------------------

        is_valid, info = validate_feature(
            feature_path
        )

        if is_valid:

            valid_rows.append(
                row
            )

            if split in split_stats:

                split_stats[split] += 1

            if label == 0:

                label_stats["real"] += 1

            else:

                label_stats["fake"] += 1

            print(
                f"✅ VALID | Shape: {info}"
            )

        else:

            invalid_rows.append(

                {
                    "video_id": video_id,
                    "reason": str(info)
                }
            )

            print(
                f"❌ INVALID | {info}"
            )

    # ========================================================
    # CREATE VALID MANIFEST
    # ========================================================

    VALID_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if len(valid_rows) > 0:

        with open(

            VALID_MANIFEST_PATH,

            "w",

            newline="",

            encoding="utf-8"

        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=valid_rows[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                valid_rows
            )

    # ========================================================
    # SAVE INVALID LOG
    # ========================================================

    invalid_log_path = (

        FINAL_PIPELINE_ROOT
        / "invalid_multimodal_features.csv"
    )

    with open(

        invalid_log_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[
                "video_id",
                "reason"
            ]
        )

        writer.writeheader()

        writer.writerows(
            invalid_rows
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "VALIDATION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "Original manifest:",
        len(rows)
    )

    print(
        "Valid samples:",
        len(valid_rows)
    )

    print(
        "Invalid samples:",
        len(invalid_rows)
    )

    print()

    print(
        "TRAIN:",
        split_stats["train"]
    )

    print(
        "VALIDATION:",
        split_stats["validation"]
    )

    print(
        "TEST:",
        split_stats["test"]
    )

    print()

    print(
        "REAL:",
        label_stats["real"]
    )

    print(
        "FAKE:",
        label_stats["fake"]
    )

    print()

    print(
        "Valid manifest:"
    )

    print(
        VALID_MANIFEST_PATH
    )

    print()

    print(
        "Invalid feature log:"
    )

    print(
        invalid_log_path
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()