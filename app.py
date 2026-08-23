import re
import sys
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Lip-Sync Detector",
    page_icon="🎭",
    layout="wide"
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
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 20px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 700;
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
    '<div class="main-title">🎭 Multimodal Deepfake Lip-Sync Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Detect potential deepfake videos by analyzing the synchronization
    between facial lip movements and spoken audio.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Pipeline")

    st.markdown(
        """
        **Visual features**
        - MediaPipe Face Mesh
        - 40 lip landmarks
        - Lip movement velocity

        **Audio features**
        - MFCC
        - 13 coefficients

        **Multimodal representation**
        - 80 lip coordinates
        - 80 velocity features
        - 13 MFCC features
        - **173 total features**

        **Temporal model**
        - 30-frame sequences
        - LSTM classifier
        """
    )

    st.divider()

    st.metric(
        "Detection Threshold",
        "0.56"
    )

    st.caption(
        "Threshold selected using validation-set F1 maximization."
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.subheader("📤 Upload Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm"
    ],
    help="Upload a video containing a visible speaking face."
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a video above to begin deepfake lip-sync analysis."
    )

else:

    st.subheader("🎥 Uploaded Video")

    st.video(
        uploaded_file
    )

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Size:** {uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    st.divider()

    analyze = st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    )

    if analyze:

        suffix = Path(
            uploaded_file.name
        ).suffix

        temp_path = None

        try:

            # ------------------------------------------------
            # Save uploaded video temporarily
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Run existing inference pipeline
            #
            # IMPORTANT:
            # We are using inference/predict_video.py
            # directly instead of duplicating the ML pipeline.
            # ------------------------------------------------

            with st.spinner(
                "Analyzing video... This may take a little while."
            ):

                result = subprocess.run(

                    [
                        sys.executable,
                        "-m",
                        "inference.predict_video",
                        str(temp_path)
                    ],

                    stdout=subprocess.PIPE,

                    stderr=subprocess.PIPE,

                    text=True,

                    cwd=Path.cwd()
                )

            output = result.stdout
            errors = result.stderr

            # ------------------------------------------------
            # Handle inference failure
            # ------------------------------------------------

            if result.returncode != 0:

                st.error(
                    "❌ Video analysis failed."
                )

                with st.expander(
                    "Show technical output"
                ):

                    st.code(
                        output + "\n" + errors
                    )

            else:

                # ============================================
                # PARSE RESULT
                # ============================================

                fake_match = re.search(
                    r"Fake Probability\s+:\s+([0-9.]+)",
                    output
                )

                real_match = re.search(
                    r"Real Probability\s+:\s+([0-9.]+)",
                    output
                )

                prediction_match = re.search(
                    r"Prediction\s+:\s+(FAKE|REAL)",
                    output
                )

                sequence_count_match = re.search(
                    r"Sequences analyzed\s+:\s+(\d+)",
                    output
                )

                threshold_match = re.search(
                    r"Threshold\s+:\s+([0-9.]+)",
                    output
                )

                fake_probability = (
                    float(fake_match.group(1))
                    if fake_match
                    else None
                )

                real_probability = (
                    float(real_match.group(1))
                    if real_match
                    else None
                )

                prediction = (
                    prediction_match.group(1)
                    if prediction_match
                    else None
                )

                sequence_count = (
                    int(sequence_count_match.group(1))
                    if sequence_count_match
                    else None
                )

                threshold = (
                    float(threshold_match.group(1))
                    if threshold_match
                    else 0.56
                )

                # ============================================
                # RESULT DISPLAY
                # ============================================

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

                # ============================================
                # PROBABILITY METRICS
                # ============================================

                col1, col2, col3 = st.columns(3)

                with col1:

                    if fake_probability is not None:

                        st.metric(
                            "Fake Probability",
                            f"{fake_probability * 100:.2f}%"
                        )

                with col2:

                    if real_probability is not None:

                        st.metric(
                            "Real Probability",
                            f"{real_probability * 100:.2f}%"
                        )

                with col3:

                    if sequence_count is not None:

                        st.metric(
                            "Sequences Analyzed",
                            sequence_count
                        )

                st.caption(
                    f"Decision threshold: {threshold:.2f}"
                )

                # ============================================
                # PROBABILITY BAR
                # ============================================

                if fake_probability is not None:

                    st.subheader(
                        "📊 Detection Confidence"
                    )

                    st.progress(
                        min(
                            max(
                                fake_probability,
                                0.0
                            ),
                            1.0
                        )
                    )

                    st.write(
                        f"Fake score: "
                        f"**{fake_probability * 100:.2f}%**"
                    )

                # ============================================
                # SEQUENCE RESULTS
                # ============================================

                sequence_rows = []

                pattern = re.compile(
                    r"Sequence\s+(\d+)\s*:\s*"
                    r"([0-9.]+)\s*"
                    r"\(([0-9.]+)%\)\s*->\s*"
                    r"(FAKE|REAL)"
                )

                for match in pattern.finditer(
                    output
                ):

                    sequence_rows.append(
                        {
                            "Sequence":
                                int(match.group(1)),

                            "Fake Probability":
                                float(match.group(2)),

                            "Confidence":
                                f"{float(match.group(3)):.2f}%",

                            "Prediction":
                                match.group(4)
                        }
                    )

                if sequence_rows:

                    st.subheader(
                        "🔬 Sequence-Level Analysis"
                    )

                    sequence_df = pd.DataFrame(
                        sequence_rows
                    )

                    st.dataframe(
                        sequence_df,
                        use_container_width=True,
                        hide_index=True
                    )

                # ============================================
                # TECHNICAL OUTPUT
                # ============================================

                with st.expander(
                    "🛠️ Technical Inference Log"
                ):

                    st.code(
                        output
                    )

        except Exception as error:

            st.error(
                f"❌ Unexpected error: {error}"
            )

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
    "Python • MediaPipe • Librosa • TensorFlow LSTM"
)
