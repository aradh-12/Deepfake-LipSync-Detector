import cv2
from pathlib import Path

from utils.dataset_manager import get_all_videos


def read_video(video_path: Path, label: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("--------------------------------")
    print(f"Video : {video_path.name}")
    print(f"Label : {'Real' if label == 0 else 'Fake'}")
    print(f"FPS   : {fps}")
    print(f"Frames: {frames}")

    cap.release()


def process_videos(limit=10):

    dataset = get_all_videos()

    print(f"\nTotal videos in dataset : {len(dataset)}\n")

    real = [x for x in dataset if x[1] == 0][:5]
    fake = [x for x in dataset if x[1] == 1][:5]

    dataset = real + fake

    for video, label in dataset:
        read_video(video, label)


if __name__ == "__main__":
    process_videos()