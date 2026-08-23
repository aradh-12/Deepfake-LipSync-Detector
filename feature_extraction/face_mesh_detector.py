import cv2
import mediapipe as mp
import csv
from pathlib import Path

INPUT_ROOT = Path("outputs/extracted_frames")
LANDMARK_OUTPUT = Path("outputs/lip_landmarks")
CSV_OUTPUT = Path("outputs/lip_coordinates")

mp_face_mesh = mp.solutions.face_mesh

LIP_LANDMARKS = [
    61,146,91,181,84,17,314,405,
    321,375,291,185,40,39,37,0,
    267,269,270,409,78,95,88,178,
    87,14,317,402,318,324,308,
    191,80,81,82,13,312,311,310,
    415
]


def process_frame(face_mesh, input_image, output_image, csv_output):

    image = cv2.imread(str(input_image))

    if image is None:
        return False

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return False

    h, w, _ = image.shape

    coordinates = []

    for face in results.multi_face_landmarks:

        for idx in LIP_LANDMARKS:

            point = face.landmark[idx]

            x = int(point.x * w)
            y = int(point.y * h)

            coordinates.append([idx, x, y])

            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), image)

    csv_output.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_output, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["Landmark", "X", "Y"])
        writer.writerows(coordinates)

    return True


def process_videos():

    folders = sorted(INPUT_ROOT.iterdir())

    total_frames = 0
    total_detected = 0

    print(f"\nFound {len(folders)} videos.\n")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for folder in folders:

            if not folder.is_dir():
                continue

            print(f"Processing : {folder.name}")

            landmark_folder = LANDMARK_OUTPUT / folder.name
            csv_folder = CSV_OUTPUT / folder.name

            frames = sorted(folder.glob("*.jpg"))

            detected = 0

            for frame in frames:

                output_image = landmark_folder / frame.name
                csv_output = csv_folder / frame.with_suffix(".csv").name

                if process_frame(
                    face_mesh,
                    frame,
                    output_image,
                    csv_output
                ):
                    detected += 1

            total_frames += len(frames)
            total_detected += detected

            print(f"Frames : {len(frames)}")
            print(f"Lips   : {detected}")
            print("-" * 40)

    print("\nFinished")
    print("Frames :", total_frames)
    print("Lips   :", total_detected)


if __name__ == "__main__":
    process_videos()