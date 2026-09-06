# Multimodal Deepfake Lip-Sync Anomaly Detector

## Project Overview

The Multimodal Deepfake Lip-Sync Anomaly Detector is a deep learning-based system designed to detect potentially manipulated or deepfake videos by analyzing the relationship between facial lip movements and spoken audio.

Unlike traditional deepfake detection approaches that focus only on visual artifacts, this project uses a multimodal approach that combines visual and audio information.

The system analyzes:

- Facial lip movement
- Temporal visual information
- Audio characteristics
- Audio-visual feature relationships

The extracted multimodal features are processed by a temporal LSTM model to generate a video-level prediction.

The final output classifies a video as:

- Real
- Potential Deepfake

along with confidence and probability scores.

---

# System Architecture

The overall pipeline consists of the following stages:

```text
Input Video
     │
     ▼
Video Validation
     │
     ├── Video readability
     ├── Frame validation
     └── Face detection
     │
     ▼
Visual Feature Extraction
     │
     ├── MediaPipe Face Mesh
     └── Lip movement representation
     │
     ▼
Audio Feature Extraction
     │
     ├── Audio extraction
     └── MFCC features
     │
     ▼
Multimodal Feature Fusion
     │
     ├── 80 Visual Features
     └── 39 Audio Features
     │
     ▼
119-Dimensional Feature Vector
     │
     ▼
Temporal Sequence Processing
     │
     └── 150 Frame Input
     │
     ▼
LSTM Deep Learning Model
     │
     ▼
Real / Fake Prediction