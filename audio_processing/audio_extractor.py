import subprocess
from pathlib import Path


def extract_audio(video_path: Path, output_audio: Path):
    """
    Extract audio from a single video using FFmpeg.
    """

    output_audio.parent.mkdir(parents=True, exist_ok=True)

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

    return result.returncode == 0


def process_all_videos(input_folder: Path, output_folder: Path):

    videos = sorted(input_folder.glob("*.mp4"))

    if not videos:
        print("❌ No videos found.")
        return

    total = 0
    success = 0

    print(f"\nFound {len(videos)} video(s).\n")

    for video in videos:

        total += 1

        video_name = video.stem

        output_audio = (
            output_folder /
            video_name /
            f"{video_name}.wav"
        )

        if extract_audio(video, output_audio):

            success += 1

            print(f"✅ {video.name}")
            print(f"   Audio Saved -> {output_audio}")

        else:

            print(f"❌ Failed -> {video.name}")

    print("\n==============================")
    print(f"Videos Processed : {total}")
    print(f"Audio Extracted  : {success}")
    print("==============================")


if __name__ == "__main__":

    input_folder = Path("data/sample_videos")

    output_folder = Path("outputs/audio")

    process_all_videos(input_folder, output_folder)