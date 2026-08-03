import cv2
from pathlib import Path


def read_video(video_path: Path):
    """Read a single video and print its information."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Unable to open {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if fps > 0 else 0

    print("=" * 50)
    print(f"Video          : {video_path.name}")
    print(f"FPS            : {fps:.2f}")
    print(f"Resolution     : {width} x {height}")
    print(f"Total Frames   : {total_frames}")
    print(f"Duration       : {duration:.2f} seconds")
    print("=" * 50)

    cap.release()


def process_all_videos(folder_path: Path):
    """Read every MP4 video in the folder."""

    videos = sorted(folder_path.glob("*.mp4"))

    if not videos:
        print("❌ No MP4 videos found.")
        return

    print(f"\nFound {len(videos)} video(s).\n")

    for video in videos:
        read_video(video)

    print(f"\n✅ Processed {len(videos)} video(s).")


if __name__ == "__main__":
    input_folder = Path("data/sample_videos")
    process_all_videos(input_folder)