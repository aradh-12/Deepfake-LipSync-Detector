import subprocess
from pathlib import Path

from utils.multidataset_manager import get_fakeavceleb_unseen


AUDIO_OUTPUT = Path("outputs/audio")


def extract_audio(video_path: Path, output_audio: Path):

    output_audio.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
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

    if result.returncode != 0:

        print(
            result.stderr.decode(
                errors="ignore"
            )
        )

    return result.returncode == 0


def get_video_name(video):

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


def process_unseen_videos():

    dataset = get_fakeavceleb_unseen(
        real_count=25,
        fake_count=25
    )

    success = 0
    failed = 0

    print()
    print("==============================")
    print("EXTRACTING UNSEEN AUDIO")
    print("==============================")
    print(
        "Videos found:",
        len(dataset)
    )

    real = sum(
        label == 0
        for _, label in dataset
    )

    fake = sum(
        label == 1
        for _, label in dataset
    )

    print("Real:", real)
    print("Fake:", fake)
    print()

    for video, label in dataset:

        video_name = get_video_name(video)

        output_audio = (
            AUDIO_OUTPUT
            / video_name
            / f"{video_name}.wav"
        )

        if output_audio.exists():

            print(
                f"⏭️ Already exists: "
                f"{video_name}"
            )

            success += 1
            continue

        print(
            f"Processing: {video_name} | "
            f"{'Real' if label == 0 else 'Fake'}"
        )

        if extract_audio(
            video,
            output_audio
        ):

            success += 1

            print(
                f"✅ Saved: {output_audio}"
            )

        else:

            failed += 1

            print(
                f"❌ Failed: {video_name}"
            )

    print()
    print("==============================")
    print("UNSEEN AUDIO EXTRACTION COMPLETE")
    print("==============================")
    print(
        "Videos processed:",
        len(dataset)
    )
    print(
        "Successful:",
        success
    )
    print(
        "Failed:",
        failed
    )
    print("==============================")


if __name__ == "__main__":

    process_unseen_videos()
