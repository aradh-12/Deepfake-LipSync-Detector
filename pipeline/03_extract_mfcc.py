from pathlib import Path
import csv
import subprocess
import tempfile
import shutil

import numpy as np
import librosa


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

LIP_FEATURE_ROOT = (
    FINAL_PIPELINE_ROOT
    / "lip_features"
)

MFCC_OUTPUT_ROOT = (
    FINAL_PIPELINE_ROOT
    / "mfcc_features"
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 13

N_FFT = 512

HOP_LENGTH = 160


# ============================================================
# CHECK FFMPEG
# ============================================================

def check_ffmpeg():

    ffmpeg_path = shutil.which(
        "ffmpeg"
    )

    if ffmpeg_path is None:

        raise RuntimeError(
            "\nFFmpeg was not found.\n"
            "Install it using:\n\n"
            "brew install ffmpeg\n"
        )

    print(
        "FFmpeg found:"
    )

    print(
        ffmpeg_path
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
# GET MFCC OUTPUT PATH
# ============================================================

def get_mfcc_output_path(
    split,
    video_id
):

    return (
        MFCC_OUTPUT_ROOT
        / split
        / f"{video_id}.npy"
    )


# ============================================================
# EXTRACT AUDIO USING FFMPEG
# ============================================================

def extract_audio_to_wav(
    video_path,
    wav_path
):

    command = [

        "ffmpeg",

        "-y",

        "-i",
        str(video_path),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(SAMPLE_RATE),

        "-loglevel",
        "error",

        str(wav_path)
    ]

    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(

            result.stderr.strip()
        )


# ============================================================
# EXTRACT MFCC FEATURES
# ============================================================

def extract_mfcc_features(
    video_path
):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        wav_path = (
            temp_dir
            / "audio.wav"
        )

        extract_audio_to_wav(

            video_path,

            wav_path
        )

        audio, sample_rate = librosa.load(

            wav_path,

            sr=SAMPLE_RATE,

            mono=True
        )

    # --------------------------------------------------------
    # Validate audio
    # --------------------------------------------------------

    if audio is None:

        raise ValueError(
            "Audio loading failed"
        )

    if len(audio) == 0:

        raise ValueError(
            "Audio contains zero samples"
        )

    if not np.isfinite(
        audio
    ).all():

        raise ValueError(
            "Audio contains NaN or Inf"
        )

    # --------------------------------------------------------
    # MFCC
    #
    # Shape:
    #
    # (13, T)
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sample_rate,

        n_mfcc=N_MFCC,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # Delta MFCC
    #
    # Shape:
    #
    # (13, T)
    # --------------------------------------------------------

    delta = librosa.feature.delta(

        mfcc,

        order=1
    )

    # --------------------------------------------------------
    # Delta-Delta MFCC
    #
    # Shape:
    #
    # (13, T)
    # --------------------------------------------------------

    delta2 = librosa.feature.delta(

        mfcc,

        order=2
    )

    # --------------------------------------------------------
    # Combine
    #
    # 13 + 13 + 13
    #
    # = 39 features
    # --------------------------------------------------------

    features = np.concatenate(

        [

            mfcc,

            delta,

            delta2
        ],

        axis=0
    )

    # --------------------------------------------------------
    # Convert
    #
    # (39, T)
    #
    # to
    #
    # (T, 39)
    # --------------------------------------------------------

    features = (

        features.T
    )

    features = features.astype(

        np.float32
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if features.ndim != 2:

        raise ValueError(

            f"Invalid MFCC dimensions: "
            f"{features.shape}"
        )

    if features.shape[1] != 39:

        raise ValueError(

            f"Expected 39 features, "
            f"got {features.shape}"
        )

    if features.shape[0] == 0:

        raise ValueError(

            "MFCC contains zero frames"
        )

    if not np.isfinite(
        features
    ).all():

        raise ValueError(

            "MFCC contains NaN or Inf"
        )

    return features


# ============================================================
# PROCESS SINGLE VIDEO
# ============================================================

# ============================================================
# PROCESS SINGLE VIDEO
# ============================================================

def process_video(
    row
):

    split = row["split"]

    video_id = row["video_id"]

    video_path = Path(
        row["video_path"]
    )

    output_path = get_mfcc_output_path(
        split,
        video_id
    )

    # --------------------------------------------------------
    # Validate video
    # --------------------------------------------------------

    if not video_path.exists():

        return (
            "FAILED",
            "Video not found"
        )

    # --------------------------------------------------------
    # Only process videos that have lip features
    #
    # This keeps the multimodal dataset consistent.
    # --------------------------------------------------------

    lip_path = get_lip_feature_path(
        split,
        video_id
    )

    if not lip_path.exists():

        return (
            "NO_LIP",
            "Lip feature missing"
        )

    # --------------------------------------------------------
    # Skip existing valid MFCC
    # --------------------------------------------------------

    if output_path.exists():

        try:

            existing = np.load(
                output_path
            )

            if (

                existing.ndim == 2

                and

                existing.shape[1] == 39

                and

                existing.shape[0] > 0

                and

                np.isfinite(
                    existing
                ).all()

            ):

                return (
                    "EXISTS",
                    existing.shape
                )

        except Exception:

            pass

    # --------------------------------------------------------
    # Extract MFCC features
    # --------------------------------------------------------

    features = extract_mfcc_features(
        video_path
    )

    # --------------------------------------------------------
    # Create split output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        output_path,
        features
    )

    return (
        "SAVED",
        features.shape
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
        "FINAL PIPELINE — MFCC FEATURE EXTRACTION"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # Check FFmpeg
    # --------------------------------------------------------

    check_ffmpeg()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    MFCC_OUTPUT_ROOT.mkdir(

        parents=True,

        exist_ok=True
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    rows = (

        load_manifest()
    )

    print()

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
        "MFCC output directory:"
    )

    print(
        MFCC_OUTPUT_ROOT
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

    failed = 0

    processed = 0

    failures = []

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for index, row in enumerate(

        rows,

        start=1
    ):

        video_id = (

            row["video_id"]
        )

        split = (

            row["split"]
        )

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

        processed += 1

    # ========================================================
    # SAVE FAILURE LOG
    # ========================================================

    failure_log = (

        FINAL_PIPELINE_ROOT
        / "mfcc_failures.csv"
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
        "MFCC FEATURE EXTRACTION COMPLETE"
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
        "Failed:",
        failed
    )

    print()

    print(
        "Output directory:"
    )

    print(
        MFCC_OUTPUT_ROOT
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
