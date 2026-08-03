import cv2
from pathlib import Path


def extract_frames(video_path: Path, output_folder: Path):
    """Extract all frames from a single video."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Unable to open {video_path.name}")
        return 0

    output_folder.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        frame_path = output_folder / f"frame_{frame_count:04d}.jpg"

        cv2.imwrite(str(frame_path), frame)

    cap.release()

    print(f"✅ {video_path.name} -> {frame_count} frames extracted.")

    return frame_count


def process_all_videos(input_folder: Path, output_root: Path):
    """Process all MP4 videos inside the input folder."""

    videos = sorted(input_folder.glob("*.mp4"))

    if not videos:
        print("❌ No MP4 videos found.")
        return

    total_frames = 0

    print(f"\nFound {len(videos)} video(s).\n")

    for video in videos:

        video_name = video.stem

        output_folder = output_root / video_name

        frames = extract_frames(video, output_folder)

        total_frames += frames

    print("\n==============================")
    print(f"Videos Processed : {len(videos)}")
    print(f"Total Frames     : {total_frames}")
    print("==============================")


if __name__ == "__main__":

    input_folder = Path("data/sample_videos")

    output_root = Path("outputs/extracted_frames")

    process_all_videos(input_folder, output_root)