import cv2
import mediapipe as mp
from pathlib import Path

# -----------------------------
# MediaPipe Initialization
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def process_frame(input_image: Path, output_image: Path):
    """
    Detect face mesh on a single frame.
    Returns True if a face was detected.
    """

    image = cv2.imread(str(input_image))

    if image is None:
        return False

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return False

        for landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        output_image.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_image), image)

        return True


def process_all_videos(input_root: Path, output_root: Path):

    total_videos = 0
    total_frames = 0
    total_faces = 0

    for video_folder in sorted(input_root.iterdir()):

        if not video_folder.is_dir():
            continue

        total_videos += 1

        output_folder = output_root / video_folder.name

        frame_files = sorted(video_folder.glob("*.jpg"))

        detected = 0

        print(f"\nProcessing Video : {video_folder.name}")

        for frame in frame_files:

            output_image = output_folder / frame.name

            if process_frame(frame, output_image):
                detected += 1

        total_frames += len(frame_files)
        total_faces += detected

        print(f"Frames           : {len(frame_files)}")
        print(f"Faces Detected   : {detected}")

    print("\n==============================")
    print(f"Videos Processed : {total_videos}")
    print(f"Total Frames     : {total_frames}")
    print(f"Faces Detected   : {total_faces}")
    print("==============================")


if __name__ == "__main__":

    input_root = Path("outputs/extracted_frames")

    output_root = Path("outputs/face_mesh")

    process_all_videos(input_root, output_root)