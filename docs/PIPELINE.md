# Detection Pipeline

## Overview

The Multimodal Deepfake Lip-Sync Anomaly Detector processes an uploaded video through a sequence of validation, feature extraction, multimodal fusion, temporal modeling, and prediction stages.

The pipeline combines visual lip movement information with audio features to produce a video-level Real/Fake prediction.

---

# Pipeline Flow

```text
Uploaded Video
      │
      ▼
Video Validation
      │
      ▼
Face Detection
      │
      ▼
Visual Feature Extraction
      │
      ▼
Audio Extraction
      │
      ▼
Audio Feature Extraction
      │
      ▼
Multimodal Feature Fusion
      │
      ▼
Temporal Sequence Preparation
      │
      ▼
LSTM Model Inference
      │
      ▼
Prediction
      │
      ▼
Result Visualization