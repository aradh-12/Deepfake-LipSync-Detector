from pathlib import Path
import subprocess
import cv2
import librosa
import numpy as np

from utils.multidataset_manager import (
    get_fakeavceleb_development
)


# ============================================================
# Paths
# ============================================================

FRAME_OUTPUT = Path(
    "outputs/extracted_frames"
)

AUDIO_OUTPUT = Path(
    "outputs/audio"
)

MFCC_OUTPUT = Path(
    "outputs/mfcc"
)


# ============================================================
# Create project video name
# ============================================================

def get_video_name(video):

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


# ============================================================
# Extract Frames
# ============================================================

def extract_frames(
    video,
    output_folder
):

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():

        print(
            f"❌ Cannot open video: {video}"
        )

        return 0

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_path = (
            output_folder /
            f"frame_{frame_count:04d}.jpg"
        )

        cv2.imwrite(
            str(frame_path),
            frame
        )

        frame_count += 1

    cap.release()

    return frame_count


# ============================================================
# Extract Audio
# ============================================================

def extract_audio(
    video,
    output_audio
):

    output_audio.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [

        "ffmpeg",

        "-y",

        "-i",
        str(video),

        "-vn",

        "-acodec",
        "pcm_s16le",

        "-ar",
        "16000",

        "-ac",
        "1",

        str(output_audio)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.returncode == 0


# ============================================================
# Extract MFCC
# ============================================================

def extract_mfcc(
    audio_file,
    output_file
):

    y, sr = librosa.load(
        audio_file,
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_file,
        mfcc
    )

    return mfcc.shape


# ============================================================
# Main
# ============================================================

def main():

    dataset = (
        get_fakeavceleb_development()
    )

    print("\n==============================")
    print("Development Video Preprocessing")
    print("==============================")

    print(
        "Videos to process :",
        len(dataset)
    )

    print(
        "Real videos       :",
        sum(
            label == 0
            for _, label in dataset
        )
    )

    print(
        "Fake videos       :",
        sum(
            label == 1
            for _, label in dataset
        )
    )

    total_frames = 0
    audio_success = 0
    mfcc_success = 0

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

        print("\n------------------------------")

        print(
            f"[{index}/{len(dataset)}] "
            f"{video_name}"
        )

        print(
            f"Label : {label_name}"
        )

        print("------------------------------")

        # ==================================================
        # Frames
        # ==================================================

        frame_folder = (
            FRAME_OUTPUT /
            video_name
        )

        frames = extract_frames(
            video,
            frame_folder
        )

        total_frames += frames

        print(
            f"Frames extracted : {frames}"
        )

        if frames == 0:

            print(
                "⚠️ Skipping audio/MFCC "
                "because no frames were extracted."
            )

            continue

        # ==================================================
        # Audio
        # ==================================================

        audio_file = (
            AUDIO_OUTPUT /
            video_name /
            f"{video_name}.wav"
        )

        if extract_audio(
            video,
            audio_file
        ):

            audio_success += 1

            print(
                "Audio extracted : ✅"
            )

        else:

            print(
                "Audio extracted : ❌"
            )

            continue

        # ==================================================
        # MFCC
        # ==================================================

        mfcc_file = (
            MFCC_OUTPUT /
            video_name /
            f"{video_name}.npy"
        )

        try:

            shape = extract_mfcc(
                audio_file,
                mfcc_file
            )

            mfcc_success += 1

            print(
                f"MFCC extracted : "
                f"{shape} ✅"
            )

        except Exception as error:

            print(
                "MFCC extraction failed : "
                f"{error}"
            )

    # ======================================================
    # Summary
    # ======================================================

    print("\n==============================")
    print("Development Preprocessing Complete")
    print("==============================")

    print(
        "Videos processed :",
        len(dataset)
    )

    print(
        "Total frames     :",
        total_frames
    )

    print(
        "Audio extracted  :",
        audio_success
    )

    print(
        "MFCC extracted   :",
        mfcc_success
    )

    print("==============================")


if __name__ == "__main__":
    main()
