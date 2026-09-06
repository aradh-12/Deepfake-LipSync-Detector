from pathlib import Path

import cv2
import mediapipe as mp

# ============================================================
# IMPORT NEW PIPELINE
# ============================================================

from pipeline.predict_video import predict_video


# ============================================================
# CUSTOM VALIDATION ERROR
# ============================================================

class VideoValidationError(Exception):
    """
    Raised when a video cannot be analyzed.
    """

    pass


# ============================================================
# DEEPFAKE DETECTOR
# ============================================================

class DeepfakeDetector:

    """
    Backend service responsible for:

    1. Validating uploaded video
    2. Checking for a visible face
    3. Running the NEW ML inference pipeline
    4. Returning results compatible with Streamlit
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )


    # ========================================================
    # PUBLIC PREDICTION METHOD
    # ========================================================

    def predict(self, video_path):

        """
        Validate the video and run the NEW
        deepfake detection pipeline.
        """

        video_path = Path(video_path)

        # ====================================================
        # CHECK FILE
        # ====================================================

        if not video_path.exists():

            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        if not video_path.is_file():

            raise FileNotFoundError(
                f"Video path is not a file: {video_path}"
            )


        # ====================================================
        # VALIDATE VIDEO
        # ====================================================

        validation = self._validate_video(
            video_path
        )


        # ====================================================
        # RUN NEW INFERENCE PIPELINE
        # ====================================================

        try:

            prediction_result = predict_video(
                video_path
            )

        except Exception as error:

            raise RuntimeError(
                f"Video analysis failed: {error}"
            ) from error


        # ====================================================
        # VALIDATE PIPELINE RESULT
        # ====================================================

        if prediction_result is None:

            raise RuntimeError(
                "Inference pipeline returned no result."
            )


        if not isinstance(
            prediction_result,
            dict
        ):

            raise RuntimeError(
                "Inference pipeline returned "
                "an invalid result."
            )


        # ====================================================
        # REQUIRED NEW PIPELINE FIELDS
        # ====================================================

        required_fields = [

            "prediction",

            "confidence",

            "fake_probability",

            "real_probability"

        ]


        for field in required_fields:

            if field not in prediction_result:

                raise RuntimeError(
                    f"Inference result is missing "
                    f"required field: {field}"
                )


        # ====================================================
        # ADD STREAMLIT COMPATIBILITY FIELDS
        # ====================================================
        #
        # The new pipeline analyzes the complete video as
        # one padded temporal input:
        #
        # (1, 150, 119)
        #
        # Therefore there is currently ONE video-level
        # prediction instead of old sequence-level predictions.
        # ====================================================

        prediction_result[
            "sequences_analyzed"
        ] = 1


        # ====================================================
        # NEW MODEL DECISION THRESHOLD
        # ====================================================

        prediction_result[
            "threshold"
        ] = 0.50


        # ====================================================
        # NEW PIPELINE DOES NOT PRODUCE
        # OLD SEQUENCE-LEVEL RESULTS
        # ====================================================

        prediction_result[
            "sequence_results"
        ] = []


        # ====================================================
        # ADD VALIDATION INFORMATION
        # ====================================================

        prediction_result[
            "validation"
        ] = validation


        # ====================================================
        # RETURN FINAL RESULT
        # ====================================================

        return prediction_result


    # ========================================================
    # VIDEO VALIDATION
    # ========================================================

    def _validate_video(
        self,
        video_path
    ):

        """
        Validate that the uploaded video:

        - Can be opened by OpenCV
        - Contains readable frames
        - Is long enough
        - Contains a visible face
        - Contains a reasonably stable face
        """

        video_path = Path(
            video_path
        )


        # ====================================================
        # CHECK FILE
        # ====================================================

        if not video_path.exists():

            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )


        if not video_path.is_file():

            raise FileNotFoundError(
                f"Video path is not a file: "
                f"{video_path}"
            )


        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            str(video_path)
        )


        if not cap.isOpened():

            cap.release()

            raise VideoValidationError(
                "The uploaded video could not "
                "be opened."
            )


        try:

            # ====================================================
            # VIDEO PROPERTIES
            # ====================================================

            total_frames = int(

                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )

            )


            fps = float(

                cap.get(
                    cv2.CAP_PROP_FPS
                )

            )


            frame_width = int(

                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )

            )


            frame_height = int(

                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )

            )


            # ====================================================
            # VALIDATE FRAME COUNT
            # ====================================================

            if total_frames <= 0:

                raise VideoValidationError(
                    "The video contains no "
                    "readable frames."
                )


            # ====================================================
            # VALIDATE DIMENSIONS
            # ====================================================

            if (

                frame_width <= 0

                or

                frame_height <= 0

            ):

                raise VideoValidationError(
                    "Could not determine "
                    "video dimensions."
                )


            # ====================================================
            # HANDLE INVALID FPS
            # ====================================================

            if fps <= 0:

                fps = 25.0


            # ====================================================
            # CALCULATE DURATION
            # ====================================================

            duration = (

                total_frames

                /

                fps

            )


            # ====================================================
            # MINIMUM DURATION
            # ====================================================

            if duration < 1.0:

                raise VideoValidationError(
                    "Video is too short for analysis. "
                    "Please upload a video at least "
                    "1 second long."
                )


            # ====================================================
            # SAMPLE FRAMES
            # ====================================================

            sample_count = min(

                12,

                total_frames

            )


            if sample_count == 1:

                frame_indices = [

                    0

                ]

            else:

                step = (

                    total_frames - 1

                ) / (

                    sample_count - 1

                )


                frame_indices = [

                    int(
                        round(
                            i * step
                        )
                    )

                    for i in range(
                        sample_count
                    )

                ]


            # ====================================================
            # DETECTION STATISTICS
            # ====================================================

            detected_faces = 0

            processed_frames = 0

            face_sizes = []

            face_centers = []


            # ====================================================
            # INITIALIZE MEDIAPIPE
            # ====================================================

            mp_face_mesh = (

                mp.solutions.face_mesh

            )


            with mp_face_mesh.FaceMesh(

                static_image_mode=False,

                max_num_faces=1,

                refine_landmarks=True,

                min_detection_confidence=0.60,

                min_tracking_confidence=0.60

            ) as face_mesh:


                # ================================================
                # CHECK EACH SAMPLED FRAME
                # ================================================

                for frame_index in frame_indices:


                    # ============================================
                    # JUMP TO FRAME
                    # ============================================

                    cap.set(

                        cv2.CAP_PROP_POS_FRAMES,

                        frame_index

                    )


                    success, frame = cap.read()


                    if not success:

                        continue


                    processed_frames += 1


                    # ============================================
                    # BGR TO RGB
                    # ============================================

                    rgb = cv2.cvtColor(

                        frame,

                        cv2.COLOR_BGR2RGB

                    )


                    # ============================================
                    # MEDIAPIPE PROCESSING
                    # ============================================

                    try:

                        results = (

                            face_mesh.process(
                                rgb
                            )

                        )

                    except Exception:

                        continue


                    # ============================================
                    # NO FACE
                    # ============================================

                    if not results.multi_face_landmarks:

                        continue


                    # ============================================
                    # GET FIRST FACE
                    # ============================================

                    face = (

                        results.multi_face_landmarks[
                            0
                        ]

                    )


                    # ============================================
                    # GET FACE COORDINATES
                    # ============================================

                    xs = [

                        landmark.x

                        for landmark
                        in face.landmark

                    ]


                    ys = [

                        landmark.y

                        for landmark
                        in face.landmark

                    ]


                    if not xs or not ys:

                        continue


                    # ============================================
                    # FACE BOUNDING BOX
                    # ============================================

                    min_x = max(

                        0.0,

                        min(xs)

                    )


                    max_x = min(

                        1.0,

                        max(xs)

                    )


                    min_y = max(

                        0.0,

                        min(ys)

                    )


                    max_y = min(

                        1.0,

                        max(ys)

                    )


                    face_width = (

                        max_x

                        -

                        min_x

                    )


                    face_height = (

                        max_y

                        -

                        min_y

                    )


                    face_area = (

                        face_width

                        *

                        face_height

                    )


                    # ============================================
                    # FACE CENTER
                    # ============================================

                    face_center_x = (

                        min_x

                        +

                        max_x

                    ) / 2.0


                    face_center_y = (

                        min_y

                        +

                        max_y

                    ) / 2.0


                    # ============================================
                    # REJECT TINY FACE
                    # ============================================

                    if face_width < 0.12:

                        continue


                    if face_height < 0.12:

                        continue


                    if face_area < 0.025:

                        continue


                    # ============================================
                    # VALID FACE
                    # ============================================

                    detected_faces += 1


                    face_sizes.append(

                        face_area

                    )


                    face_centers.append(

                        (

                            face_center_x,

                            face_center_y

                        )

                    )


            # ====================================================
            # NO VALID FACE
            # ====================================================

            if detected_faces == 0:

                raise VideoValidationError(
                    "No suitable face was detected. "
                    "Please upload a video containing "
                    "a clearly visible speaking person."
                )


            # ====================================================
            # DETECTION RATIO
            # ====================================================

            detection_ratio = (

                detected_faces

                /

                max(
                    processed_frames,
                    1
                )

            )


            

            

        finally:

            cap.release()


        # ========================================================
        # VALIDATION SUCCESS
        # ========================================================

        return {

            "valid": True,

            "total_frames": total_frames,

            "fps": round(
                fps,
                2
            ),

            "duration": round(
                duration,
                2
            ),

            "frame_width": frame_width,

            "frame_height": frame_height,

            "frames_checked": processed_frames,

            "face_frames": detected_faces,

            "face_detection_ratio": round(
                detection_ratio,
                4
            )

        }