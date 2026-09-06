import tempfile
from pathlib import Path

import cv2
import streamlit as st

from backend.detector import (
    DeepfakeDetector,
    VideoValidationError
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Lip-Sync Detector",
    page_icon="🎭",
    layout="wide"
)


# ============================================================
# INITIALIZE DETECTOR
# ============================================================

detector = DeepfakeDetector()


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_video(video_path):

    """
    Perform basic validation before running
    the deepfake detection pipeline.

    Returns:

        (True, None)

    when the video is suitable.

        (False, error_message)

    otherwise.
    """

    video_path = Path(video_path)


    # ========================================================
    # CHECK FILE
    # ========================================================

    if not video_path.exists():

        return (
            False,
            "Video file could not be found."
        )


    if video_path.stat().st_size == 0:

        return (
            False,
            "The uploaded video is empty."
        )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        return (
            False,
            "The uploaded video could not be opened."
        )


    # ========================================================
    # VIDEO PROPERTIES
    # ========================================================

    frame_count = int(

        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

    )


    fps = float(

        cap.get(
            cv2.CAP_PROP_FPS
        )

    )


    duration = (

        frame_count / fps

        if fps > 0

        else 0

    )


    cap.release()


    # ========================================================
    # VALIDATE FRAME COUNT
    # ========================================================

    if frame_count < 30:

        return (

            False,

            "The video is too short. "
            "At least 30 frames are required."

        )


    # ========================================================
    # VALIDATE DURATION
    # ========================================================

    if duration < 1.0:

        return (

            False,

            "The video is too short for "
            "reliable analysis."

        )


    return (

        True,

        None

    )


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(

    """

    <style>

    .main-title {

        font-size: 42px;

        font-weight: 700;

        margin-bottom: 5px;

    }


    .subtitle {

        font-size: 18px;

        opacity: 0.75;

        margin-bottom: 30px;

    }


    .result-box {

        padding: 25px;

        border-radius: 15px;

        border: 1px solid rgba(
            128,
            128,
            128,
            0.25
        );

        margin-top: 20px;

    }


    .small-text {

        opacity: 0.7;

        font-size: 14px;

    }

    </style>

    """,

    unsafe_allow_html=True

)


# ============================================================
# HEADER
# ============================================================

st.markdown(

    """

    <div class="main-title">

    🎭 Multimodal Deepfake Lip-Sync Detector

    </div>

    """,

    unsafe_allow_html=True

)


st.markdown(

    """

    <div class="subtitle">

    Detect potential deepfake videos by analyzing
    facial lip movement and spoken audio using a
    multimodal temporal deep learning model.

    </div>

    """,

    unsafe_allow_html=True

)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:


    # ========================================================
    # PIPELINE INFORMATION
    # ========================================================

    st.header(
        "⚙️ Detection Pipeline"
    )


    st.markdown(

        """

        **Visual features**

        - MediaPipe Face Mesh
        - Lip movement representation
        - **80 visual features**

        **Audio features**

        - MFCC extraction
        - **39 audio features**

        **Multimodal representation**

        - Audio + visual fusion
        - **119 total features**

        **Temporal model**

        - 150-frame input
        - LSTM classifier

        """

    )


    st.divider()


    # ========================================================
    # MODEL THRESHOLD
    # ========================================================

    detection_threshold = 0.50


    st.metric(

        "Decision Threshold",

        f"{detection_threshold:.2f}"

    )


    st.caption(

        "Model prediction threshold used "
        "by the current inference pipeline."

    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.subheader(
    "📤 Upload Video"
)


uploaded_file = st.file_uploader(

    "Choose a video file",

    type=[

        "mp4",

        "mov",

        "avi",

        "mkv",

        "webm"

    ],

    help=(

        "Upload a video containing a "
        "clearly visible speaking face."

    )

)


# ============================================================
# NO VIDEO UPLOADED
# ============================================================

if uploaded_file is None:


    st.info(

        "Upload a video above to begin "
        "multimodal deepfake lip-sync analysis."

    )


# ============================================================
# VIDEO UPLOADED
# ============================================================

else:


    # ========================================================
    # UPLOADED VIDEO
    # ========================================================

    st.subheader(
        "🎥 Uploaded Video"
    )


    st.video(
        uploaded_file
    )


    st.write(

        f"**File:** {uploaded_file.name}"

    )


    st.write(

        f"**Size:** "

        f"{uploaded_file.size / (1024 * 1024):.2f} MB"

    )


    st.divider()


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(

        "🔍 Analyze Video",

        type="primary",

        use_container_width=True

    )


    # ========================================================
    # ANALYSIS START
    # ========================================================

    if analyze:


        # ====================================================
        # FILE EXTENSION
        # ====================================================

        suffix = Path(

            uploaded_file.name

        ).suffix


        temp_path = None


        try:


            # ====================================================
            # SAVE VIDEO TEMPORARILY
            # ====================================================

            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=suffix

            ) as temp_file:


                temp_file.write(

                    uploaded_file.getbuffer()

                )


                temp_path = Path(

                    temp_file.name

                )


            # ====================================================
            # BASIC VIDEO VALIDATION
            # ====================================================

            valid, validation_error = validate_video(

                temp_path

            )


            if not valid:


                st.warning(

                    f"⚠️ {validation_error}"

                )


                st.info(

                    "Please upload a valid video "
                    "containing a visible speaking "
                    "person."

                )


                st.stop()


            # ====================================================
            # BACKEND INFERENCE
            # ====================================================

            with st.spinner(

                "Analyzing video... "
                "Extracting visual and audio features."

            ):


                result = detector.predict(

                    temp_path

                )


            # ====================================================
            # EXTRACT RESULTS
            # ====================================================

            prediction = result[

                "prediction"

            ]


            confidence = result[

                "confidence"

            ]


            fake_probability = result[

                "fake_probability"

            ]


            real_probability = result[

                "real_probability"

            ]


            threshold = result[

                "threshold"

            ]


            validation = result.get(

                "validation",

                {}

            )


            # ====================================================
            # MAIN RESULT
            # ====================================================

            if prediction == "FAKE":


                st.error(

                    "🚨 RESULT: POTENTIAL DEEPFAKE"

                )


            elif prediction == "REAL":


                st.success(

                    "✅ RESULT: LIKELY REAL"

                )


            else:


                st.warning(

                    "⚠️ Unable to determine prediction."

                )


            # ====================================================
            # RESULT METRICS
            # ====================================================

            col1, col2, col3 = st.columns(

                3

            )


            # ------------------------------------------------
            # FAKE PROBABILITY
            # ------------------------------------------------

            with col1:


                st.metric(

                    "Fake Probability",

                    f"{fake_probability * 100:.2f}%"

                )


            # ------------------------------------------------
            # REAL PROBABILITY
            # ------------------------------------------------

            with col2:


                st.metric(

                    "Real Probability",

                    f"{real_probability * 100:.2f}%"

                )


            # ------------------------------------------------
            # MODEL CONFIDENCE
            # ------------------------------------------------

            with col3:


                st.metric(

                    "Model Confidence",

                    f"{confidence:.2f}%"

                )


            # ====================================================
            # DECISION THRESHOLD
            # ====================================================

            st.caption(

                f"Decision threshold: "

                f"{threshold:.2f}"

            )


            # ====================================================
            # DETECTION CONFIDENCE
            # ====================================================

            st.subheader(

                "📊 Detection Confidence"

            )


            # ====================================================
            # CONFIDENCE VALUE
            # ====================================================

            confidence_value = max(

                fake_probability,

                real_probability

            )


            confidence_value = min(

                max(

                    confidence_value,

                    0.0

                ),

                1.0

            )


            # ====================================================
            # PROGRESS BAR
            # ====================================================

            st.progress(

                confidence_value

            )


            # ====================================================
            # DISPLAY CONFIDENCE
            # ====================================================

            st.write(

                f"Prediction confidence: "

                f"**{confidence_value * 100:.2f}%**"

            )


            # ====================================================
            # PIPELINE INFORMATION
            # ====================================================

            st.subheader(

                "🧠 Analysis Summary"

            )


            summary_col1, summary_col2 = st.columns(

                2

            )


            # ------------------------------------------------
            # MODEL INPUT
            # ------------------------------------------------

            with summary_col1:


                st.markdown(

                    """

                    **Multimodal Feature Pipeline**

                    - Visual features: 80
                    - Audio features: 39
                    - Total features: 119
                    - Temporal input: 150 frames

                    """

                )


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            with summary_col2:


                st.markdown(

                    """

                    **Deep Learning Model**

                    - Architecture: LSTM
                    - Input shape: `(150, 119)`
                    - Output: Real / Fake probability
                    - Analysis: Video-level prediction

                    """

                )


            # ====================================================
            # VIDEO VALIDATION DETAILS
            # ====================================================

            if validation:


                with st.expander(

                    "🔍 Video Validation Details"

                ):


                    validation_col1, validation_col2 = (

                        st.columns(

                            2

                        )

                    )


                    with validation_col1:


                        st.write(

                            f"**Duration:** "

                            f"{validation.get('duration', 'N/A')} seconds"

                        )


                        st.write(

                            f"**FPS:** "

                            f"{validation.get('fps', 'N/A')}"

                        )


                        st.write(

                            f"**Total Frames:** "

                            f"{validation.get('total_frames', 'N/A')}"

                        )


                    with validation_col2:


                        st.write(

                            f"**Resolution:** "

                            f"{validation.get('frame_width', 'N/A')}"

                            f" × "

                            f"{validation.get('frame_height', 'N/A')}"

                        )


                        st.write(

                            f"**Frames Checked:** "

                            f"{validation.get('frames_checked', 'N/A')}"

                        )


                        st.write(

                            f"**Face Detection Ratio:** "

                            f"{validation.get('face_detection_ratio', 'N/A')}"

                        )


        # ========================================================
        # FILE ERROR
        # ========================================================

        except FileNotFoundError as error:


            st.error(

                f"❌ File error: {error}"

            )


        # ========================================================
        # VIDEO VALIDATION ERROR
        # ========================================================

        except VideoValidationError as error:


            st.warning(

                f"⚠️ {error}"

            )


        # ========================================================
        # RUNTIME ERROR
        # ========================================================

        except RuntimeError as error:


            st.error(

                f"❌ Video analysis failed: {error}"

            )


        # ========================================================
        # UNEXPECTED ERROR
        # ========================================================

        except Exception as error:


            st.error(

                f"❌ Unexpected error: {error}"

            )


        # ========================================================
        # DELETE TEMPORARY VIDEO
        # ========================================================

        finally:


            if (

                temp_path is not None

                and temp_path.exists()

            ):


                temp_path.unlink()


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(

    "Multimodal Deepfake Lip-Sync Anomaly Detector • "

    "Python • MediaPipe • Librosa • TensorFlow • LSTM"

)