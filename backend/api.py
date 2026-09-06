from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException

from backend.detector import DeepfakeDetector


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Multimodal Deepfake Lip-Sync Detector API",
    description=(
        "Backend API for detecting potential deepfake videos "
        "using multimodal lip movement and audio analysis."
    ),
    version="1.0.0"
)


# ============================================================
# DETECTOR
# ============================================================

detector = DeepfakeDetector()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Deepfake Lip-Sync Detector",
        "version": "1.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# VIDEO PREDICTION
# ============================================================

@app.post("/predict")
async def predict_video(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No video file provided."
        )

    # --------------------------------------------------------
    # Allowed video extensions
    # --------------------------------------------------------

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm"
    }

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Allowed formats: "
                "mp4, mov, avi, mkv, webm."
            )
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary video file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        # ----------------------------------------------------
        # Run detector
        # ----------------------------------------------------

        result = detector.predict(
            temp_path
        )

        # ----------------------------------------------------
        # Return JSON
        # ----------------------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except RuntimeError as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unexpected server error: {error}"
            )
        )

    finally:

        # ----------------------------------------------------
        # Always delete temporary video
        # ----------------------------------------------------

        if (
            temp_path is not None
            and temp_path.exists()
        ):

            temp_path.unlink()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
