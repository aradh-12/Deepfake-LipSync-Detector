import cv2
import mediapipe as mp
import csv
from pathlib import Path

# -----------------------------
# MediaPipe Initialization
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

# Lip landmark indices
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 185, 40, 39, 37, 0,
    267, 269, 270, 409, 78, 95, 88, 178,
    87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310,
    415
]


def process_frame(
    input_image: Path,
    output_image: Path,
    csv_output: Path
):
    image = cv2.imread(str(input_image))

    if image is None:
        return False

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False

        h, w, _ = image.shape

        coordinates = []

        for face_landmarks in results.multi_face_landmarks:

            for idx in LIP_LANDMARKS:

                point = face_landmarks.landmark[idx]

                x = int(point.x * w)
                y = int(point.y * h)

                coordinates.append([idx, x, y])

                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Save landmark image
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image), image)

        # Save coordinates as CSV
        csv_output.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_output, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Landmark", "X", "Y"])
            writer.writerows(coordinates)

        return True


def process_all_videos(input_root: Path, output_root: Path):

    total_frames = 0
    total_detected = 0
    total_videos = 0

    for video_folder in sorted(input_root.iterdir()):

        if not video_folder.is_dir():
            continue

        total_videos += 1

        output_folder = output_root / video_folder.name
        csv_folder = Path("outputs/lip_coordinates") / video_folder.name

        frame_files = sorted(video_folder.glob("*.jpg"))

        detected = 0

        print(f"\nProcessing Video : {video_folder.name}")

        for frame in frame_files:

            output_image = output_folder / frame.name
            csv_output = csv_folder / frame.with_suffix(".csv").name

            if process_frame(frame, output_image, csv_output):
                detected += 1

        print(f"Frames Processed : {len(frame_files)}")
        print(f"Lips Detected    : {detected}")

        total_frames += len(frame_files)
        total_detected += detected

    print("\n==============================")
    print(f"Videos Processed : {total_videos}")
    print(f"Total Frames     : {total_frames}")
    print(f"Lips Detected    : {total_detected}")
    print("==============================")


if __name__ == "__main__":

    input_root = Path("outputs/extracted_frames")
    output_root = Path("outputs/lip_landmarks")

    process_all_videos(input_root, output_root)