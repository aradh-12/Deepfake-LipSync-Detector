import cv2
from pathlib import Path

from utils.multidataset_manager import get_fakeavceleb

OUTPUT_ROOT = Path("outputs/extracted_frames")


def extract_frames(video_path: Path, output_folder: Path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Unable to open {video_path}")
        return 0

    output_folder.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_path = output_folder / f"frame_{frame_count:04d}.jpg"

        cv2.imwrite(str(frame_path), frame)

        frame_count += 1

    cap.release()

    return frame_count


def process_videos():

    dataset = [
    item
    for item in get_fakeavceleb(limit=50)
    if item[1] == 0
]

    print(f"\nProcessing {len(dataset)} videos...\n")

    total_frames = 0

    for video, label in dataset:

        dataset_name = video.parts[1]

        folder_name = (
            dataset_name
            + "_"
            + video.parent.name
            + "_"
            + video.stem
        )

        output_folder = OUTPUT_ROOT / folder_name

        frames = extract_frames(video, output_folder)

        total_frames += frames

        print(f"{folder_name}")
        print(f"Frames : {frames}")
        print("-" * 40)

    print("\nFinished")
    print("Videos :", len(dataset))
    print("Frames :", total_frames)


if __name__ == "__main__":
    process_videos()