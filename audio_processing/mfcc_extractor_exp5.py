import librosa
import numpy as np
from pathlib import Path

from configs.config import (
    AUDIO_OUTPUT,
    OUTPUT_ROOT
)


# ============================================================
# Exp5 Configuration
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 13

N_FFT = 2048

HOP_LENGTH = 512


# ============================================================
# Exp5 Output
# ============================================================

MFCC_EXP5_OUTPUT = (
    OUTPUT_ROOT /
    "mfcc_exp5"
)


# ============================================================
# Extract MFCC + Delta + Delta-Delta
# ============================================================

def extract_mfcc_exp5(
    audio_file: Path,
    output_file: Path
):
    """
    Extract enhanced audio features.

    Features:
        13 MFCC
        13 MFCC Delta
        13 MFCC Delta-Delta

    Total:
        39 audio features per time frame.

    Output shape:
        (39, T)
    """

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    y, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE
    )

    if y is None or len(y) == 0:

        raise ValueError(
            f"Audio file is empty: {audio_file}"
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
    # Delta
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    # --------------------------------------------------------
    # Delta-Delta
    # --------------------------------------------------------

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if mfcc.shape[0] != N_MFCC:

        raise ValueError(
            f"Invalid MFCC shape: "
            f"{mfcc.shape}"
        )

    if delta.shape != mfcc.shape:

        raise ValueError(
            f"Invalid delta shape: "
            f"{delta.shape}"
        )

    if delta_delta.shape != mfcc.shape:

        raise ValueError(
            f"Invalid delta-delta shape: "
            f"{delta_delta.shape}"
        )

    # --------------------------------------------------------
    # Combine
    #
    # 13 MFCC
    # +13 Delta
    # +13 Delta-Delta
    #
    # = 39
    # --------------------------------------------------------

    features = np.concatenate(
        [
            mfcc,
            delta,
            delta_delta
        ],
        axis=0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Check finite values
    # --------------------------------------------------------

    if not np.isfinite(
        features
    ).all():

        raise ValueError(
            f"MFCC features contain "
            f"NaN or Inf: {audio_file}"
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
        features
    )

    return features.shape


# ============================================================
# Process All Audio
# ============================================================

def process_all_audio_exp5(
    input_root: Path,
    output_root: Path
):
    """
    Process every WAV file available in the
    existing audio output directory.

    Existing Exp3 MFCC files are NOT modified.
    """

    audio_files = sorted(
        input_root.glob(
            "*/*.wav"
        )
    )

    if not audio_files:

        print(
            "❌ No WAV files found."
        )

        return

    print("\n" + "=" * 60)
    print("EXP5 MFCC FEATURE EXTRACTION")
    print("=" * 60)

    print(
        "Input audio directory :",
        input_root
    )

    print(
        "Output directory      :",
        output_root
    )

    print(
        "Audio files found     :",
        len(audio_files)
    )

    print(
        "MFCC features         :",
        N_MFCC
    )

    print(
        "Delta features        :",
        N_MFCC
    )

    print(
        "Delta-delta features  :",
        N_MFCC
    )

    print(
        "Total audio features  :",
        N_MFCC * 3
    )

    print("=" * 60)

    processed = 0
    failed = 0

    # --------------------------------------------------------
    # Process every WAV
    # --------------------------------------------------------

    for index, audio in enumerate(
        audio_files,
        start=1
    ):

        video_name = audio.parent.name

        output_file = (
            output_root /
            video_name /
            f"{video_name}.npy"
        )

        print(
            f"\n[{index}/{len(audio_files)}] "
            f"{video_name}"
        )

        try:

            shape = extract_mfcc_exp5(
                audio,
                output_file
            )

            processed += 1

            print(
                "   Shape :",
                shape
            )

            print(
                "   Saved :",
                output_file
            )

        except Exception as error:

            failed += 1

            print(
                "   ❌ Failed"
            )

            print(
                "   Error :",
                error
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXP5 MFCC EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        "Total audio files :",
        len(audio_files)
    )

    print(
        "Processed         :",
        processed
    )

    print(
        "Failed            :",
        failed
    )

    print(
        "Output directory  :",
        output_root
    )

    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    process_all_audio_exp5(
        AUDIO_OUTPUT,
        MFCC_EXP5_OUTPUT
    )