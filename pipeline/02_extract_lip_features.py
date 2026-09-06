from pathlib import Path
import csv
import sys

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "manifests"
    / "dataset_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
    / "lip_features"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Process every Nth frame.
#
# This keeps the pipeline practical while still preserving
# temporal lip movement.
FRAME_STRIDE = 2

# Maximum frames extracted from one video.
MAX_FRAMES = 150

# Minimum valid lip frames required.
MIN_VALID_FRAMES = 30


# ============================================================
# MEDIAPIPE LIP LANDMARKS
# ============================================================

# MediaPipe FaceMesh lip landmark indices.
#
# We use 40 points around the lips.

LIP_INDICES = [
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291, 308, 324, 318, 402,
    317, 14, 87, 178, 88,
    95, 185, 40, 39, 37,
    0, 267, 269, 270, 409,
    415, 310, 311, 312, 13,
    82, 81, 42, 183, 78
]


# ============================================================
# READ MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

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
# NORMALIZE LIP LANDMARKS
# ============================================================

def normalize_lips(lips):
    """
    Normalize lip coordinates.

    Input:
        (40, 2)

    Output:
        (40, 2)

    Method:
        1. Center coordinates around lip centroid.
        2. Scale using maximum distance.

    This reduces sensitivity to:
        - face position
        - camera distance
        - resolution
    """

    lips = lips.astype(np.float32)

    center = np.mean(
        lips,
        axis=0,
        keepdims=True
    )

    normalized = lips - center

    distances = np.linalg.norm(
        normalized,
        axis=1
    )

    scale = np.max(distances)

    if scale < 1e-6:

        return None

    normalized = normalized / scale

    return normalized.astype(np.float32)


# ============================================================
# EXTRACT LIP FEATURES FROM VIDEO
# ============================================================

def extract_lip_features(
    video_path,
    face_mesh
):

    video_path = Path(video_path)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"❌ Cannot open video:\n{video_path}"
        )

        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if total_frames <= 0:

        cap.release()

        print(
            f"❌ Invalid frame count:\n{video_path}"
        )

        return None

    if fps <= 0:

        cap.release()

        print(
            f"❌ Invalid FPS:\n{video_path}"
        )

        return None


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    lip_frames = []

    frame_index = 0

    processed_frames = 0


    # ========================================================
    # READ VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:

            break


        # ----------------------------------------------------
        # Frame stride
        # ----------------------------------------------------

        if frame_index % FRAME_STRIDE != 0:

            frame_index += 1

            continue


        # ----------------------------------------------------
        # Stop after maximum processed frames
        # ----------------------------------------------------

        if processed_frames >= MAX_FRAMES:

            break


        frame_index += 1

        processed_frames += 1


        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe FaceMesh
        # ----------------------------------------------------

        results = face_mesh.process(
            rgb_frame
        )


        if not results.multi_face_landmarks:

            continue


        # ----------------------------------------------------
        # Use first detected face
        # ----------------------------------------------------

        face_landmarks = (
            results.multi_face_landmarks[0]
        )


        # ----------------------------------------------------
        # Extract 40 lip points
        # ----------------------------------------------------

        lips = []

        for index in LIP_INDICES:

            landmark = (
                face_landmarks.landmark[index]
            )

            lips.append(
                [
                    landmark.x,
                    landmark.y
                ]
            )


        lips = np.array(
            lips,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if lips.shape != (40, 2):

            continue


        if not np.isfinite(lips).all():

            continue


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized_lips = normalize_lips(
            lips
        )

        if normalized_lips is None:

            continue


        # ----------------------------------------------------
        # Flatten
        #
        # (40, 2)
        #
        # →
        #
        # (80,)
        # ----------------------------------------------------

        flattened = (
            normalized_lips.flatten()
        )


        if flattened.shape != (80,):

            continue


        lip_frames.append(
            flattened.astype(np.float32)
        )


    cap.release()


    # ========================================================
    # VALIDATE
    # ========================================================

    if len(lip_frames) < MIN_VALID_FRAMES:

        return None


    lip_features = np.array(
        lip_frames,
        dtype=np.float32
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if lip_features.ndim != 2:

        return None


    if lip_features.shape[1] != 80:

        return None


    if not np.isfinite(
        lip_features
    ).all():

        return None


    return lip_features


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    row,
    face_mesh,
    index,
    total
):

    video_id = row["video_id"]

    video_path = Path(
        row["video_path"]
    )

    label = int(
        row["label"]
    )

    split = row["split"]


    # ========================================================
    # Output directory
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_file = (
        output_dir
        / f"{video_id}.npz"
    )


    # ========================================================
    # Skip existing valid files
    # ========================================================

    if output_file.exists():

        try:

            data = np.load(
                output_file,
                allow_pickle=True
            )

            features = data["features"]

            if (
                features.ndim == 2
                and features.shape[1] == 80
                and len(features)
                >= MIN_VALID_FRAMES
            ):

                print(
                    f"[{index}/{total}] "
                    f"⏭️  EXISTS | "
                    f"{split.upper()} | "
                    f"{video_id}"
                )

                return "skipped"

        except Exception:

            pass


    # ========================================================
    # Check video exists
    # ========================================================

    if not video_path.exists():

        print(
            f"[{index}/{total}] "
            f"❌ MISSING VIDEO | "
            f"{video_id}"
        )

        return "failed"


    # ========================================================
    # Extract
    # ========================================================

    print(
        f"[{index}/{total}] "
        f"Processing | "
        f"{split.upper()} | "
        f"{'FAKE' if label == 1 else 'REAL'}"
    )

    print(
        f"Video: {video_id}"
    )


    features = extract_lip_features(
        video_path,
        face_mesh
    )


    # ========================================================
    # Failed
    # ========================================================

    if features is None:

        print(
            "⚠️  SKIPPED — "
            "Insufficient valid lip frames"
        )

        return "failed"


    # ========================================================
    # Save
    # ========================================================

    np.savez_compressed(

        output_file,

        features=features,

        label=np.int32(label),

        video_id=np.array(video_id),

        split=np.array(split)

    )


    print(
        f"✅ SAVED | "
        f"Frames: {len(features)} | "
        f"Shape: {features.shape}"
    )


    return "success"


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "FINAL PIPELINE — STEP 2"
    )

    print(
        "LIP FEATURE EXTRACTION"
    )

    print("=" * 70)


    # ========================================================
    # Load manifest
    # ========================================================

    rows = load_manifest()

    print()

    print(
        "Manifest:"
    )

    print(
        MANIFEST_PATH
    )

    print()

    print(
        "Total videos:",
        len(rows)
    )


    # ========================================================
    # Split summary
    # ========================================================

    print()

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        split_rows = [

            row

            for row in rows

            if row["split"]
            == split_name

        ]

        print(

            f"{split_name.upper():12}:",
            len(split_rows)

        )


    print()

    print(
        "Frame stride:",
        FRAME_STRIDE
    )

    print(
        "Maximum frames per video:",
        MAX_FRAMES
    )

    print(
        "Minimum valid frames:",
        MIN_VALID_FRAMES
    )

    print(
        "Lip feature size:",
        80
    )


    print()

    print("=" * 70)

    print(
        "STARTING EXTRACTION"
    )

    print("=" * 70)


    # ========================================================
    # Statistics
    # ========================================================

    successful = 0

    skipped = 0

    failed = 0


    # ========================================================
    # Create FaceMesh
    # ========================================================

    mp_face_mesh = mp.solutions.face_mesh


    with mp_face_mesh.FaceMesh(

        static_image_mode=False,

        max_num_faces=1,

        refine_landmarks=True,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5

    ) as face_mesh:


        # ====================================================
        # Process videos
        # ====================================================

        for index, row in enumerate(

            rows,

            start=1

        ):


            result = process_video(

                row,

                face_mesh,

                index,

                len(rows)

            )


            if result == "success":

                successful += 1


            elif result == "skipped":

                skipped += 1


            else:

                failed += 1


    # ========================================================
    # Final Summary
    # ========================================================

    print()

    print("=" * 70)

    print(
        "LIP FEATURE EXTRACTION COMPLETE"
    )

    print("=" * 70)

    print(
        "Total videos:",
        len(rows)
    )

    print(
        "Successfully processed:",
        successful
    )

    print(
        "Already existed:",
        skipped
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
        OUTPUT_ROOT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print(
            "⚠️ Process interrupted by user."
        )

        sys.exit(1)

    except Exception as error:

        print()

        print("=" * 70)

        print(
            "❌ FATAL ERROR"
        )

        print("=" * 70)

        print(error)

        sys.exit(1)