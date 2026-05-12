# 🛡️ Suraksha AI — Women Harassment Detection System

> **Advanced AI Surveillance** designed to detect and record harassment incidents in real-time. Built with a fusion of **Computer Vision**, **Temporal Action Recognition**, and **Forensic Behavioral Analysis**.

---

## 🚀 Core Capabilities
*   **Temporal Action Recognition**: Uses a custom **CNN + LSTM** architecture to detect complex actions like pushing, chasing, and grabbing over time.
*   **Forensic Evidence System**: Automatically records 15-second video clips and high-res screenshots of every incident.
*   **Hybrid Threat Engine**: Fuses multiple AI signals (Gender, Action, Proximity, Weapons) into a single 0–1 Threat Score.
*   **Hardware Optimized**: Special fail-safe logic for **macOS M1/M2 (ARM64)** and high-performance inference on CPU/GPU.

---

## 🧠 AI Training & Models

### 🎬 Action Recognition (CNN + LSTM)
The heart of the system is the **Action Recognizer**. It doesn't just look at a single photo; it looks at **sequences of 16 frames** to understand "movement."
*   **Architecture**: ResNet18 (Feature Extraction) + LSTM (Temporal Memory).
*   **Dataset**: 2,000+ video samples (1,600 training, 200 validation, 200 test).
*   **Performance**: Achieved **94.5% Accuracy** in binary classification (Normal vs. Harassment).
*   **Export**: Available in both PyTorch (`.pt`) and high-performance Web format (`.onnx`).

### ⚙️ The Pipeline Stack
1.  **Person Detection**: YOLOv8n (optimized for speed).
2.  **Tracking**: DeepSORT (maintains persistent identity across frames).
3.  **Gender Detection**: MobileNetV2 (PyTorch) — Classifies suspects and victims.
4.  **Pose Estimation**: MediaPipe (Skeleton Analysis) — Detects sudden aggressive movements.
5.  **Behavior Analyzer**: Geometry-based logic for stalking and cornering detection.

---

## 📂 File Structure

```text
.
├── main.py                 # 🚀 Entry point (Video/Webcam/API modes)
├── configs/
│   └── config.yaml         # ⚙️ Central configuration (Thresholds, Weights, Paths)
├── models/
│   ├── person_detector.py  # YOLOv8 Person detection
│   ├── action_recognizer.py# CNN+LSTM Harassment model
│   ├── harassment_engine.py# 🧠 The Fusion "Brain" (Weighted Scoring)
│   └── pose_estimator.py   # MediaPipe skeleton logic (with M1 Fail-safe)
├── inference/
│   └── inference_pipeline.py# 🏎️ Multi-threaded orchestration logic
├── training/
│   ├── train.py           # Action model training script
│   └── dataset_loader.py   # Temporal video processing for training
├── saved_models/           # 💾 Model weights (.pt and .onnx)
├── outputs/                # 📁 Forensic evidence (Clips, Screenshots, CSVs)
├── database/               # 🗄️ SQLite incident records
├── utils/                  # 🛠️ Video processing, Drawing, and Logging
└── requirements.txt        # 📦 Dependency list
```

---

## 🛠️ Installation & Usage

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Detection
**Webcam Mode:**
```bash
python main.py --mode webcam
```

**Video Mode:**
```bash
python main.py --mode video --source path/to/video.mp4
```

**Backend API Mode:**
```bash
python main.py --mode api
```

---

## 🎨 Configuration (config.yaml)
You can tune the system's sensitivity without touching the code.
*   `threat_score_harassment`: Change to `0.1` for testing or `0.7` for production.
*   `w_action`: Increase this to `0.8+` to prioritize the trained AI model.
*   `buffer_seconds`: How much video to save as evidence when an incident is caught.

---

## 📱 Frontend Integration
The system includes an exported **ONNX** model located at `saved_models/harassment_detector.onnx`. This can be used in React, Vue, or Mobile apps via `onnxruntime-web`. 

Check `artifacts/frontend_integration.md` for the full Javascript implementation guide.

---

## 🛡️ Security & Privacy
*   All processing is done **locally** on the edge (no cloud video upload required).
*   Incident reports are encrypted and stored in `database/incidents.db`.
*   Forensic clips are timestamped and tagged with Camera IDs for legal evidence.
