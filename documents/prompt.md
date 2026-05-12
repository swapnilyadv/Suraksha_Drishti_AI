# PROJECT TITLE
AI-Based Women Harassment Detection System using Python

# PROJECT OVERVIEW
Build an advanced AI-powered surveillance system that detects possible women harassment incidents from CCTV live streams and uploaded videos using Deep Learning, Computer Vision, Human Pose Estimation, Object Detection, Action Recognition, and Behavioral Analysis.

The system must work in real-time and generate alerts whenever suspicious harassment behavior is detected.

The architecture should be modular because frontend and backend will be built separately later.

DO NOT USE DOCKER.

Use pure Python environment and standard installation setup.

---

# MAIN OBJECTIVES

The AI system should:

1. Detect humans from CCTV/live camera/video
2. Detect male and female persons
3. Track movement of people
4. Detect suspicious interaction patterns
5. Detect violence or aggressive body movement
6. Detect forced physical contact
7. Detect stalking/chasing behavior
8. Detect crowding around women
9. Detect unsafe proximity duration
10. Generate harassment confidence score
11. Save incident clips automatically
12. Generate alerts
13. Work with:
   - Webcam
   - CCTV RTSP streams

---

# TECH STACK

Use:

- Python 3.11+
- PyTorch
- OpenCV
- YOLOv8
- DeepSORT tracking
- MediaPipe
- NumPy
- Pandas
- FastAPI
- SQLite
- FFmpeg
- Torchvision

Optional:
- TensorFlow
- Detectron2
- HuggingFace Transformers

---

# AI MODULES REQUIRED

## 1. PERSON DETECTION MODULE

Purpose:
Detect all humans in video frames.

Requirements:
- Use YOLOv8 for person detection
- Draw bounding boxes
- Assign unique tracking IDs
- Track movement continuously

Output:
- Person ID
- Bounding box
- Confidence score

---

## 2. GENDER DETECTION MODULE

Purpose:
Detect whether detected person is male or female.

Requirements:
- Use lightweight CNN model
- Real-time inference
- Store gender label with tracking ID

Output:
- Male/Female
- Confidence %

---

## 3. HUMAN POSE ESTIMATION MODULE

Purpose:
Detect body posture and body movement.

Requirements:
- Use MediaPipe Pose or OpenPose
- Detect:
  - Hand movement
  - Body bending
  - Sudden pulling
  - Physical struggle

Output:
- Pose landmarks
- Skeleton drawing
- Movement vectors


Purpose:
Detect dangerous weapons in real-time from CCTV streams and uploaded videos.

Supported weapon detection:
- Gun
- Knife
- Pistol
- Rifle
- Bat
- Sharp objects
- Suspicious metallic objects

Requirements:
- Use YOLOv8 custom-trained weapon detection model
- Detect weapon location with bounding boxes
- Track weapon movement
- Associate weapon with nearby person ID
- Real-time inference support

Output:
- Weapon type
- Confidence score
- Bounding box coordinates
- Associated suspect tracking ID

Alert logic:
- If weapon detected near aggressive movement:
  Threat level = HIGH
- If weapon + harassment detected:
  Trigger emergency alert immediately

UI Overlay:
- Red bounding box around weapon
- Label weapon name and confidence %
- Show danger indicator on screen

---

## 4. ACTION RECOGNITION MODULE

Purpose:
Detect suspicious actions related to harassment.

Train model to detect:
- Pushing
- Pulling
- Chasing
- Hitting
- Grabbing
- Unwanted touching
- Fighting
- Group surrounding
- Running behind someone

Use:
- LSTM
- CNN + LSTM
- 3D CNN
- Video Transformer

Recommended:
CNN feature extraction + LSTM sequence learning

---

## 5. BEHAVIOR ANALYSIS MODULE

Purpose:
Analyze long-duration suspicious behavior.

Detect:
- One person continuously following another
- Aggressive movement
- Very close interaction for long duration
- Fast movement toward victim
- Cornering behavior

Logic:
- Distance analysis
- Movement trajectory
- Interaction duration
- Motion intensity

---

## 6. HARASSMENT CLASSIFICATION ENGINE

Purpose:
Combine all AI modules and calculate final threat score.

Inputs:
- Action recognition
- Pose analysis
- Distance analysis
- Gender detection
- Movement patterns

Output:
- Safe
- Suspicious
- Harassment Detected

Also generate:
- Threat confidence %
- Incident timestamp

---

# ALERT SYSTEM

When harassment is detected:

1. Show red warning box
2. Play alarm sound
3. Save screenshot
4. Save video clip
5. Generate JSON incident report
6. Store incident in database

Incident data:
- Timestamp
- Threat level
- Person IDs
- Video path
- Screenshot path

---

# VIDEO INPUT SUPPORT

The system must support:

## 1. Live CCTV Stream
- RTSP stream
- IP camera
- Webcam

## 2. Uploaded Video
- MP4
- AVI
- MOV

---

# DATASET REQUIREMENTS

I will provide videos manually for training.

The system must support custom dataset training.

Dataset structure:

dataset/
│
├── train/
│   ├── harassment/
│   ├── normal/
│
├── val/
│   ├── harassment/
│   ├── normal/
│
├── test/
│   ├── harassment/
│   ├── normal/

---

# DATA PREPROCESSING

Implement:
- Frame extraction
- Frame resizing
- Video normalization
- Data augmentation
- Sequence generation

Augmentations:
- Flip
- Brightness
- Blur
- Rotation
- Noise

---

# MODEL TRAINING PIPELINE

Build complete training scripts.

Include:
- Dataset loader
- Training loop
- Validation loop
- Checkpoint saving
- Early stopping
- Accuracy metrics

Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

# REAL-TIME INFERENCE PIPELINE

Pipeline flow:

Video Input
→ Frame Extraction
→ Person Detection
→ Tracking
→ Gender Detection
→ Pose Estimation
→ Action Recognition
→ Behavior Analysis
→ Threat Scoring
→ Alert Generation

---

# FILE STRUCTURE

Create clean production-level architecture:

project/
│
├── models/
├── datasets/
├── training/
├── inference/
├── tracking/
├── alerts/
├── database/
├── api/
├── utils/
├── configs/
├── saved_models/
├── outputs/
│   ├── incidents/
│   ├── screenshots/
│   ├── clips/
│
├── main.py
├── requirements.txt
└── README.md

---

# BACKEND API

Build FastAPI backend.

Required APIs:

1. Upload video
2. Start live detection
3. Stop detection
4. Get incidents
5. Download incident clip
6. Get real-time status

Response format:
JSON

---

# DATABASE

Use SQLite.

Store:
- Incident history
- Threat level
- Video paths
- Screenshot paths
- Detection timestamps

---

# UI OVERLAY REQUIREMENTS

Display:
- Bounding boxes
- Gender labels
- Threat level
- Tracking ID
- Live FPS
- Harassment confidence score

Color scheme:
- Green = Safe
- Yellow = Suspicious
- Red = Harassment

---

# PERFORMANCE OPTIMIZATION

Requirements:
- GPU support
- CUDA acceleration
- Multi-threading
- Frame skipping optimization
- Real-time FPS optimization

---

# SECURITY REQUIREMENTS

- Prevent fake API requests
- Validate uploaded videos
- Secure file storage
- Avoid model crashes on corrupted videos

---

# LOGGING SYSTEM

Create logs for:
- Detection events
- Errors
- API requests
- Incident reports

---

# EXPORT FEATURES

Generate:
- CSV reports
- JSON reports
- Incident summaries

---

# TRAINING REQUIREMENTS

Provide:
- train.py
- evaluate.py
- inference.py

Support:
- Resume training
- Custom epochs
- Batch size settings
- GPU selection

---

# INSTALLATION REQUIREMENTS

Generate:
- requirements.txt
- setup instructions
- CUDA installation guide

NO DOCKER.

Use:
- pip installation only

---

# README REQUIREMENTS

Create professional README with:
- Installation
- Dataset setup
- Training guide
- Inference guide
- API documentation
- Folder structure
- Example screenshots

---

# IMPORTANT REQUIREMENTS

1. Write clean modular code
2. Use OOP architecture
3. Add comments everywhere
4. Use configuration files
5. Make scalable architecture
6. Separate training and inference
7. Save best model automatically
8. Support future frontend integration
9. Real-time optimized system
10. Production-ready architecture

---

# ADVANCED FEATURES (OPTIONAL)

If possible also implement:

- Weapon detection
- Audio scream detection
- Crowd density detection
- Multi-camera support
- Face recognition
- Emotion detection
- SMS/Email alerts
- WhatsApp alerts
- Telegram alerts

---

# FINAL OUTPUT REQUIRED

Generate:
- Complete source code
- AI model training pipeline
- Real-time detection pipeline
- Backend API
- Database integration
- Full documentation
- requirements.txt
- Setup instructions

The project should run locally on Windows/macOS/Linux without Docker.

Use best practices for AI engineering and scalable production architecture.