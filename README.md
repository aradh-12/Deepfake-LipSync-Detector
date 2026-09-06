# 🎭 Multimodal Deepfake Lip-Sync Anomaly Detector

A multimodal deepfake detection system that analyzes the synchronization between **facial lip movements** and **spoken audio** to identify potential deepfake videos.

The project combines **Computer Vision**, **Audio Processing**, **Multimodal Feature Fusion**, and a **Temporal LSTM Neural Network** to classify videos as:

- ✅ REAL
- 🚨 FAKE

---

## 📌 Project Overview

Deepfake videos have become increasingly realistic, making traditional visual-artifact-based detection challenging.

Instead of relying only on visual artifacts, this project focuses on a multimodal approach:

> **Do the speaker's lip movements correspond to the spoken audio?**

The system analyzes multiple signals:

- 👄 Visual lip movement
- 🔊 Audio characteristics
- 🎵 MFCC-based speech features
- ⏱ Temporal synchronization between audio and video

These features are combined and processed using an **LSTM-based temporal classification model**.

The final system predicts whether a video is likely:

- **REAL**
- **FAKE**

---

## 🧠 Key Features

- 🎥 Video-based deepfake detection
- 👄 Lip landmark extraction using MediaPipe Face Mesh
- 🔊 Audio feature extraction
- 🎵 MFCC-based audio analysis using Librosa
- 🔗 Multimodal audio-visual feature fusion
- ⏱ Temporal sequence modeling
- 🧠 LSTM-based classification
- 📊 Model evaluation pipeline
- 📈 Threshold and probability analysis
- 🌐 Streamlit-based user interface
- 🔌 Backend detection components
- 🧪 Training and experiment scripts
- 📁 Modular project architecture
- 🧩 Multi-dataset management
- 🧪 Basic project tests

---

# 🏗️ Project Architecture

![Project Architecture](assets/project_architecture.png)

The overall detection pipeline is:

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
