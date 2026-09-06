# 🎭 Multimodal Deepfake Lip-Sync Anomaly Detector

A multimodal deepfake detection system that analyzes the synchronization between **facial lip movements** and **spoken audio** to identify potential deepfake videos.

The project combines **computer vision**, **audio processing**, **multimodal feature fusion**, and a **Temporal LSTM model** to classify videos as:

- ✅ REAL
- 🚨 FAKE

---

# 📌 Project Overview

Deepfake videos have become increasingly realistic, making traditional visual-artifact-based detection challenging.

This project focuses on a different approach:

> **Does the movement of a person's lips correspond to the audio being spoken?**

The system:

- Extracts lip movement features from video frames
- Extracts MFCC-based audio features
- Combines both modalities
- Builds a multimodal temporal representation
- Uses an LSTM neural network for classification

The final system predicts whether a video is likely **REAL** or **FAKE**.

---

# 🧠 Project Architecture

```text
                    INPUT VIDEO
                         │
                         ▼
              ┌─────────────────────┐
              │   Video Processing  │
              └─────────────────────┘
                         │
                         ▼
              MediaPipe Face Mesh
                         │
                         ▼
                Lip Feature Extraction
                         │
                         ├───────────────┐
                         │               │
                         ▼               ▼
                  Video Features     Audio Extraction
                                         │
                                         ▼
                                      Librosa
                                         │
                                         ▼
                                   MFCC Features
                                         │
                         ┌───────────────┘
                         │
                         ▼
              Multimodal Feature Fusion
                         │
                         ▼
                119 Feature Representation
                         │
                         ▼
                 Temporal Padding / Trimming
                         │
                         ▼
                 Input Shape: (150, 119)
                         │
                         ▼
                    LSTM Classifier
                         │
                         ▼
                   REAL / FAKE