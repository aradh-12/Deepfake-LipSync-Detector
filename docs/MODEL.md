# Deep Learning Model

## Overview

The Multimodal Deepfake Lip-Sync Anomaly Detector uses a Long Short-Term Memory (LSTM) neural network for temporal classification.

The model analyzes multimodal audio and visual features extracted from an input video.

The purpose of the model is to learn temporal patterns associated with Real and potentially manipulated videos.

---

# Why LSTM?

Video data contains temporal information.

A single frame is not sufficient to understand:

- Lip movement patterns
- Facial movement over time
- Changes in multimodal features
- Temporal relationships between audio and visual information

An LSTM is suitable because it can process sequential data and learn dependencies across time.

---

# Model Input

The model receives a temporal sequence of multimodal features.

The input shape is:

```text
(150, 119)