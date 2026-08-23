from pathlib import Path
import numpy as np
import cv2

from configs.config import MFCC_OUTPUT
from utils.multidataset_manager import (
    FA_ROOT,
    get_fakeavceleb_development
)


MFCC_HOP_LENGTH = 512
MFCC_N_FFT = 2048
AUDIO_SAMPLE_RATE = 16000


def get_video_name(video):

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


def get_video_info(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return fps, frame_count, duration


def main():

    dataset = get_fakeavceleb_development(
        real_count=25,
        fake_count=25
    )

    print("\n==============================")
    print("MFCC / Video Alignment Check")
    print("==============================")

    print(
        "Videos:",
        len(dataset)
    )

    print("==============================")

    durations = []
    fps_values = []

    for index, (video, label) in enumerate(
        dataset,
        start=1
    ):

        video_name = get_video_name(
            video
        )

        try:

            fps, frame_count, video_duration = (
                get_video_info(video)
            )

            mfcc_file = (
                MFCC_OUTPUT
                / video_name
                / f"{video_name}.npy"
            )

            if not mfcc_file.exists():

                print(
                    f"\n❌ {video_name}"
                )

                print(
                    "   MFCC missing"
                )

                continue

            mfcc = np.load(
                mfcc_file
            )

            if mfcc.ndim != 2:

                print(
                    f"\n❌ {video_name}"
                )

                print(
                    f"   Invalid MFCC shape: "
                    f"{mfcc.shape}"
                )

                continue

            mfcc_frames = mfcc.shape[1]

            mfcc_times = (
                (
                    np.arange(
                        mfcc_frames
                    )
                    * MFCC_HOP_LENGTH
                    +
                    MFCC_N_FFT / 2
                )
                /
                AUDIO_SAMPLE_RATE
            )

            mfcc_start = mfcc_times[0]
            mfcc_end = mfcc_times[-1]

            coverage_difference = (
                video_duration
                -
                mfcc_end
            )

            print(
                f"\n[{index}/{len(dataset)}] "
                f"{video_name}"
            )

            print(
                f"   Label       : "
                f"{'Real' if label == 0 else 'Fake'}"
            )

            print(
                f"   FPS         : "
                f"{fps:.4f}"
            )

            print(
                f"   Video frames: "
                f"{frame_count}"
            )

            print(
                f"   Video time  : "
                f"{video_duration:.4f}s"
            )

            print(
                f"   MFCC frames : "
                f"{mfcc_frames}"
            )

            print(
                f"   MFCC start  : "
                f"{mfcc_start:.4f}s"
            )

            print(
                f"   MFCC end    : "
                f"{mfcc_end:.4f}s"
            )

            print(
                f"   Difference  : "
                f"{coverage_difference:.4f}s"
            )

            durations.append(
                video_duration
            )

            fps_values.append(
                fps
            )

        except Exception as error:

            print(
                f"\n❌ {video_name}: "
                f"{error}"
            )

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    if durations:

        print(
            f"Video duration min : "
            f"{min(durations):.4f}s"
        )

        print(
            f"Video duration max : "
            f"{max(durations):.4f}s"
        )

        print(
            f"Video duration avg : "
            f"{np.mean(durations):.4f}s"
        )

        print(
            f"FPS min            : "
            f"{min(fps_values):.4f}"
        )

        print(
            f"FPS max            : "
            f"{max(fps_values):.4f}"
        )

        print(
            f"FPS avg            : "
            f"{np.mean(fps_values):.4f}"
        )

    print("==============================\n")


if __name__ == "__main__":
    main()