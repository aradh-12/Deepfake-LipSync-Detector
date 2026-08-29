import cv2
from pathlib import Path

from utils.multidataset_manager import get_fakeavceleb_unseen


# ============================================================
# PATHS
# ============================================================

OUTPUT_ROOT = Path(
    "outputs/extracted_frames"
)


# ============================================================
# EXTRACT FRAMES FROM ONE VIDEO
# ============================================================

def extract_frames(
    video_path: Path,
    output_folder: Path
):
    """
    Extract every frame from one video.

    Frames are saved as:

        frame_0000.jpg
        frame_0001.jpg
        frame_0002.jpg
        ...

    Returns:
        Number of successfully saved frames.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"❌ Unable to open: {video_path}"
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

        if cv2.imwrite(
            str(frame_path),
            frame
        ):

            frame_count += 1

        else:

            print(
                f"⚠️ Failed to save frame: "
                f"{frame_count}"
            )

    cap.release()

    return frame_count


# ============================================================
# VIDEO FOLDER NAME
# ============================================================

def get_video_name(video):

    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


# ============================================================
# PROCESS UNSEEN DATASET
# ============================================================

def process_unseen_videos():

    dataset = get_fakeavceleb_unseen(
        real_count=25,
        fake_count=25
    )

    print("\n==============================")
    print("Unseen Frame Extraction")
    print("==============================")

    print(
        "Unseen videos :",
        len(dataset)
    )

    print(
        "Real videos   :",
        sum(
            label == 0
            for _, label in dataset
        )
    )

    print(
        "Fake videos   :",
        sum(
            label == 1
            for _, label in dataset
        )
    )

    print("==============================")

    total_frames = 0
    successful_videos = 0
    failed_videos = 0

    # ========================================================
    # Process every unseen video
    # ========================================================

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

        output_folder = (
            OUTPUT_ROOT /
            video_name
        )

        print("------------------------------")

        print(
            f"[{index}/{len(dataset)}] "
            f"{video_name}"
        )

        print(
            f"Label : {label_name}"
        )

        # ----------------------------------------------------
        # Skip only if frames already exist
        # ----------------------------------------------------

        existing_frames = list(
            output_folder.glob("*.jpg")
        ) if output_folder.exists() else []

        if existing_frames:

            print(
                f"⏭️ Already exists: "
                f"{len(existing_frames)} frames"
            )

            total_frames += len(
                existing_frames
            )

            successful_videos += 1

            continue

        # ----------------------------------------------------
        # Extract frames
        # ----------------------------------------------------

        frames = extract_frames(
            video,
            output_folder
        )

        if frames > 0:

            successful_videos += 1

            total_frames += frames

            print(
                f"✅ Frames : {frames}"
            )

        else:

            failed_videos += 1

            print(
                "❌ No frames extracted"
            )

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n==============================")
    print("Unseen Frame Extraction Complete")
    print("==============================")

    print(
        "Videos found       :",
        len(dataset)
    )

    print(
        "Successful videos   :",
        successful_videos
    )

    print(
        "Failed videos       :",
        failed_videos
    )

    print(
        "Total frames        :",
        total_frames
    )

    print("==============================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_unseen_videos()