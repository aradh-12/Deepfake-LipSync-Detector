import librosa
import numpy as np
from pathlib import Path

from utils.multidataset_manager import (
    get_fakeavceleb_unseen
)


# ============================================================
# PATHS
# ============================================================

AUDIO_ROOT = Path(
    "outputs/audio"
)

MFCC_ROOT = Path(
    "outputs/mfcc"
)


# ============================================================
# MFCC CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 13

N_FFT = 2048

HOP_LENGTH = 512


# ============================================================
# EXTRACT MFCC
# ============================================================

def extract_mfcc(
    audio_file: Path,
    output_file: Path
):

    """
    Extract 13-dimensional MFCC features.

    Input:
        WAV audio file

    Output:
        NumPy array with shape:

            (13, T)

    where T is the number of temporal
    MFCC frames.
    """

    if not audio_file.exists():

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_file}"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    y, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    if y is None or len(y) == 0:

        raise ValueError(
            f"Empty audio file: "
            f"{audio_file}"
        )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=True
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if mfcc.ndim != 2:

        raise ValueError(
            f"Expected 2D MFCC array, "
            f"got {mfcc.shape}"
        )

    if mfcc.shape[0] != N_MFCC:

        raise ValueError(
            f"Expected {N_MFCC} MFCC coefficients, "
            f"got {mfcc.shape[0]}"
        )

    if not np.isfinite(mfcc).all():

        raise ValueError(
            f"MFCC contains NaN or Inf: "
            f"{audio_file}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_file,
        mfcc.astype(np.float32)
    )

    return mfcc.shape


# ============================================================
# PROCESS UNSEEN DATASET
# ============================================================

def process_unseen_videos():

    dataset = get_fakeavceleb_unseen(
        real_count=25,
        fake_count=25
    )

    print()
    print("==============================")
    print("UNSEEN MFCC EXTRACTION")
    print("==============================")

    print(
        "Videos found:",
        len(dataset)
    )

    print(
        "Real:",
        sum(
            label == 0
            for _, label in dataset
        )
    )

    print(
        "Fake:",
        sum(
            label == 1
            for _, label in dataset
        )
    )

    print("==============================")

    successful = 0

    failed = 0

    # ========================================================
    # PROCESS EACH VIDEO
    # ========================================================

    for index, (video, label) in enumerate(
        dataset,
        start=1
    ):

        video_name = (
            f"FakeAVCeleb_"
            f"{video.parent.name}_"
            f"{video.stem}"
        )

        label_name = (
            "Real"
            if label == 0
            else "Fake"
        )

        print("------------------------------")

        print(
            f"[{index}/{len(dataset)}] "
            f"{video_name}"
        )

        print(
            f"Label : {label_name}"
        )

        # ----------------------------------------------------
        # Audio input
        # ----------------------------------------------------

        audio_file = (
            AUDIO_ROOT
            / video_name
            / f"{video_name}.wav"
        )

        if not audio_file.exists():

            print(
                f"❌ Missing audio:"
            )

            print(
                f"   {audio_file}"
            )

            failed += 1

            continue

        # ----------------------------------------------------
        # MFCC output
        # ----------------------------------------------------

        output_file = (
            MFCC_ROOT
            / video_name
            / f"{video_name}.npy"
        )

        try:

            shape = extract_mfcc(
                audio_file,
                output_file
            )

            print(
                f"✅ MFCC shape : {shape}"
            )

            print(
                f"Saved : {output_file}"
            )

            successful += 1

        except Exception as error:

            print(
                f"❌ Failed: {video_name}"
            )

            print(
                f"   Error: {error}"
            )

            failed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("==============================")
    print("UNSEEN MFCC EXTRACTION COMPLETE")
    print("==============================")

    print(
        "Videos processed:",
        len(dataset)
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print("==============================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_unseen_videos()
