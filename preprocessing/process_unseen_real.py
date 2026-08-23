from pathlib import Path
import subprocess
import cv2
import librosa
import numpy as np


from utils.multidataset_manager import FA_ROOT


# --------------------------------------------------
# Paths
# --------------------------------------------------

FRAME_OUTPUT = Path("outputs/extracted_frames")
AUDIO_OUTPUT = Path("outputs/audio")
MFCC_OUTPUT = Path("outputs/mfcc")


DEVELOPMENT_VIDEOS = 25
UNSEEN_VIDEOS = 25


# --------------------------------------------------
# Get the exact 25 unseen Real videos
# --------------------------------------------------

def get_unseen_real_videos():

    all_videos = sorted(
        video
        for video in FA_ROOT.rglob("*.mp4")
        if "RealVideo-RealAudio" in str(video)
    )

    return all_videos[
        DEVELOPMENT_VIDEOS:
        DEVELOPMENT_VIDEOS + UNSEEN_VIDEOS
    ]


# --------------------------------------------------
# Create project video name
# --------------------------------------------------

def get_video_name(video):

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


# --------------------------------------------------
# Extract frames
# --------------------------------------------------

def extract_frames(video, output_folder):

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


# --------------------------------------------------
# Extract audio
# --------------------------------------------------

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


# --------------------------------------------------
# Extract MFCC
# --------------------------------------------------

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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    videos = get_unseen_real_videos()

    print("\n==============================")
    print("Unseen Real Video Preprocessing")
    print("==============================")

    print(
        "Videos to process :",
        len(videos)
    )

    total_frames = 0
    audio_success = 0
    mfcc_success = 0

    for index, video in enumerate(
        videos,
        start=1
    ):

        video_name = get_video_name(
            video
        )

        print("\n------------------------------")
        print(
            f"[{index}/{len(videos)}] "
            f"{video_name}"
        )
        print("------------------------------")

        # ------------------------------------------
        # Frames
        # ------------------------------------------

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

        # ------------------------------------------
        # Audio
        # ------------------------------------------

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

        # ------------------------------------------
        # MFCC
        # ------------------------------------------

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
                f"MFCC extracted : {shape} ✅"
            )

        except Exception as error:

            print(
                f"MFCC extraction failed : "
                f"{error}"
            )

    print("\n==============================")
    print("Preprocessing Complete")
    print("==============================")

    print(
        "Videos processed :",
        len(videos)
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
