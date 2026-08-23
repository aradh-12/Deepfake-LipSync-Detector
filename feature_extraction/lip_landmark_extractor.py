import cv2
import mediapipe as mp
import csv
from pathlib import Path

from utils.multidataset_manager import get_fakeavceleb_development


# ============================================================
# PATHS
# ============================================================

INPUT_ROOT = Path("outputs/extracted_frames")

CSV_OUTPUT = Path("outputs/lip_coordinates")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# EXACTLY 40 LIP LANDMARKS
# ============================================================

LIP_LANDMARKS = [
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291, 185, 40, 39, 37,
    0, 267, 269, 270, 409,
    78, 95, 88, 178, 87,
    14, 317, 402, 318, 324,
    308, 191, 80, 81, 82,
    13, 312, 311, 310, 415
]

assert len(LIP_LANDMARKS) == 40
# Safety check
assert len(LIP_LANDMARKS) == 40


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    face_mesh,
    input_image,
    csv_output
):

    image = cv2.imread(
        str(input_image)
    )

    if image is None:
        return False

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return False

    h, w, _ = image.shape

    # Only first detected face
    face = results.multi_face_landmarks[0]

    coordinates = []

    for idx in LIP_LANDMARKS:

        point = face.landmark[idx]

        x = int(point.x * w)
        y = int(point.y * h)

        coordinates.append(
            [idx, x, y]
        )

    # Must contain exactly 40 landmarks
    if len(coordinates) != 40:
        return False

    csv_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        csv_output,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Landmark", "X", "Y"]
        )

        writer.writerows(
            coordinates
        )

    return True


# ============================================================
# PROCESS DEVELOPMENT DATASET
# ============================================================

def process_videos():

    dataset = get_fakeavceleb_development(
        real_count=25,
        fake_count=25
    )

    print("\n==============================")
    print("Development Lip Extraction")
    print("==============================")

    print(
        "Development videos :",
        len(dataset)
    )

    print(
        "Real videos        :",
        sum(label == 0 for _, label in dataset)
    )

    print(
        "Fake videos        :",
        sum(label == 1 for _, label in dataset)
    )

    print("==============================")

    total_frames = 0
    total_detected = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for index, (video, label) in enumerate(
            dataset,
            start=1
        ):

            video_name = (
                f"FakeAVCeleb_"
                f"{video.parent.name}_"
                f"{video.stem}"
            )

            label_name = (
                "Real"
                if label == 0
                else "Fake"
            )

            input_folder = (
                INPUT_ROOT /
                video_name
            )

            output_folder = (
                CSV_OUTPUT /
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

            # ------------------------------------------
            # Check input frames
            # ------------------------------------------

            if not input_folder.exists():

                print(
                    "❌ Missing frame folder"
                )

                continue

            frames = sorted(
                input_folder.glob("*.jpg")
            )

            if not frames:

                print(
                    "❌ No frames found"
                )

                continue

            # ------------------------------------------
            # Rebuild landmark output
            # ------------------------------------------

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            # Remove old CSV files
            for old_file in output_folder.glob(
                "*.csv"
            ):

                old_file.unlink()

            detected = 0

            # ------------------------------------------
            # Process frames
            # ------------------------------------------

            for frame in frames:

                csv_output = (
                    output_folder /
                    frame.with_suffix(".csv").name
                )

                success = process_frame(
                    face_mesh,
                    frame,
                    csv_output
                )

                if success:
                    detected += 1

            total_frames += len(frames)
            total_detected += detected

            print(
                f"Frames : {len(frames)}"
            )

            print(
                f"Lips   : {detected}"
            )

    print("\n==============================")
    print("Development Lip Extraction Complete")
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
        "Lip detections   :",
        total_detected
    )

    print("==============================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_videos()
