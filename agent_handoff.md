# 🤝 Antigravity Agent Handoff Document

> **Project:** AI Tennis Visualiser & Analytics Engine  
> **Repository:** `tennis-visualiser`  
> **Workspace Root:** `/Users/andrewpham/Documents/GitHub/tennis-visualiser`  
> **Target Python Environment:** `./venv` (Python 3.14 / 3.9–3.12+ compatible)  
> **Documentation Map:** [`README.md`](README.md) | [`features.md`](features.md) | [`development_blog.md`](development_blog.md) | [`model_and_training.md`](model_and_training.md)

---

## 📌 Executive Summary & Project Goal

The **AI Tennis Visualiser & Analytics Engine** is an end-to-end computer vision and deep learning platform inspired by [`abdullahtarek/tennis_analysis`](https://github.com/abdullahtarek/tennis_analysis). It processes standard broadcast tennis match footage (720p/1080p MP4) and extracts high-precision metric telemetry:
1. Persistent 2-Player Tracking & Re-Identification across the net line.
2. High-Resolution ($1280\text{px}$) micro tennis ball detection & time-series trajectory interpolation.
3. Perspective homography mapping video camera pixels to a 2D top-down mini-court and real-world metric dimensions ($23.77\text{m} \times 10.97\text{m}$).
4. Biomechanical stroke classification (Serve, Forehand, Backhand, Volley), shot speed ($\text{km/h}$), and ITF singles in/out line calling.
5. Player kinetics (instantaneous running speed, cumulative distance covered in meters, and 2D Gaussian density court heatmaps).
6. Multi-format delivery: Annotated broadcast video with HUD & 2D radar overlay, structured JSON telemetry, standalone dark-mode HTML reports, and an interactive Streamlit web dashboard.
7. Local LLM & VLM AI Copilot Suite (powered by Ollama: Qwen 2.5 & Moondream) for training diagnostics, active dataset verification, video triage, and tactical scouting reports.

---

## 🏗️ Architecture & Pipeline Overview

The system operates on a **two-pass cascading architecture**:

```
Broadcast Video (.mp4 in data/inputs/)
       │
       ├──► [Pass 1: Feature Extraction & Detection] (src/detectors/yolo_detector.py)
       │       ├── Player Detector: YOLOv8 person class 0 (PERSON_CONF_THRESHOLD = 0.40)
       │       ├── High-Res Ball Detector: 1280px YOLOv8 (BALL_CONF_THRESHOLD = 0.08)
       │       └── Court Line Detector: 14 ITF Keypoint Regressor (CourtLineDetector)
       │
       ├──► [Time-Series Smoothing & Tracking Guardrails]
       │       ├── Net-Partitioned Persistent 2-Player Re-ID (PlayerTracker)
       │       ├── Bounded Piecewise Linear Interpolation (BallInterpolator, max 8 frames)
       │       └── 4 Safety Checks: 220px Jump Filter, Court ROI Mask, HSV Scene Cuts, Gap Resets
       │
       ├──► [Perspective Homography Engine] (src/mini_court/mini_court.py)
       │       ├── Camera Pixels -> 2D Mini-Court Canvas Pixels
       │       └── Camera Pixels -> Real-World Metric Court (Meters: 23.77m × 10.97m)
       │
       ├──► [Physics, Biomechanics & Kinetics]
       │       ├── Shot Detection: Tralection inflection points Δvy (ShotDetector)
       │       ├── Ball Speed (km/h) & ITF Singles Boundary Calling (is_inside_singles_court)
       │       ├── Stroke Classification: Serve, Forehand, Backhand, Volley (StrokeClassifier)
       │       └── Player Kinetics: Speed km/h, Distance meters, 2D Heatmaps (PlayerAnalytics)
       │
       └──► [Pass 2: Visual Rendering & Multi-Format Delivery]
               ├── Annotated Video with HUD & Mini-Court Radar (data/outputs/output.mp4)
               ├── Structured JSON Match Telemetry (data/analysis/match_summary.json)
               ├── Standalone Dark-Mode HTML Report (data/analysis/match_report.html)
               ├── Interactive Streamlit Web Dashboard (app.py)
               └── Narrative LLM Scouting Report (data/analysis/tactical_scouting_report.md)
```

---

## 📂 Repository File Directory & Module Map

```
tennis-visualiser/
├── data/
│   ├── inputs/                    # Input broadcast match videos (*.mp4)
│   ├── outputs/                   # Annotated output videos with HUD (*_annotated.mp4)
│   └── analysis/                  # Generated JSON summaries, HTML reports & heatmaps
│       ├── match_summary.json
│       ├── match_report.html
│       ├── heatmap_player_1.png
│       ├── heatmap_player_2.png
│       └── tactical_scouting_report.md
├── src/
│   ├── config.py                  # Global thresholds, paths, model candidates & ROI settings
│   ├── utils/
│   │   ├── roi_utils.py           # Polygon scaling & OpenCV pointPolygonTest gating
│   │   └── scene_utils.py         # 2D HSV color histogram correlation for scene cuts
│   ├── detectors/
│   │   ├── ball_detector.py       # High-res 1280px ball detector with kinematic distance checks
│   │   └── yolo_detector.py       # Pass 1 parallel extraction orchestrator
│   ├── trackers/
│   │   ├── ball_interpolator.py   # Piecewise linear interpolation bounded by max 8 frames
│   │   └── player_tracker.py      # Spatial net-split persistent Player 1 & Player 2 Re-ID
│   ├── court_detector/
│   │   └── court_line_detector.py # 14 ITF court intersection keypoint extraction
│   ├── mini_court/
│   │   └── mini_court.py          # Homography engine & 2D top-down radar canvas
│   ├── analysis/
│   │   ├── shot_detector.py       # Inflection hit detection, ball speed (km/h) & line calls
│   │   ├── stroke_classifier.py   # Forehand, Backhand, Serve, Volley classifier
│   │   ├── player_analytics.py    # Running speeds, cumulative distance & 2D Gaussian heatmaps
│   │   ├── coaching_insights.py   # Automated AI coaching heuristics & match tempo profiling
│   │   ├── llm_scout.py           # Local LLM narrative scouting engine (Qwen 2.5 via Ollama)
│   │   └── report_generator.py    # JSON & standalone HTML report generator (Base64 embedded)
│   └── visualizers/
│       └── video_annotator.py     # Pass 2 video compositor with HUD & radar overlays
├── training/
│   ├── extract_frames.py          # Multi-video frame harvester across data/inputs/
│   ├── auto_label.py              # Semi-supervised pseudo-labeling engine (YOLO format)
│   ├── train_ball_detector.py     # YOLOv8 fine-tuner + auto-deployment to best_tennis.pt
│   ├── train_diagnostics.py       # Local LLM training diagnostics copilot (Qwen 2.5)
│   ├── vlm_verifier.py            # Local VLM active label verification (Moondream)
│   └── video_triage.py            # Smart VLM rally segmentation & non-play filtering
├── app.py                         # Interactive Multi-Tab Streamlit Dashboard
├── batch_process.py               # Batch processor for multi-match queues
├── main.py                        # Single-match pipeline CLI entrypoint
├── requirements.txt               # Complete dependencies (ultralytics, torch, cv2, streamlit, etc.)
├── README.md                      # Primary project overview, quickstart & prerequisites
├── features.md                    # 8-Phase completed feature matrix & technical roadmap
├── development_blog.md            # Technical engineering journal (14 chronological entries)
├── model_and_training.md          # Comprehensive model architecture & training guide
└── agent_handoff.md               # [THIS FILE] Context & handoff guide for subsequent agents
```

---

## ⚙️ Key Technical Invariants & Critical Configurations

1. **Model Checkpoint Fallback Hierarchy (`src/config.py`)**:
   ```python
   MODEL_CANDIDATES = [
       'best_tennis.pt',             # 1. Custom fine-tuned weights (Auto-deployed upon training completion)
       'tennis_ball_detector.pt',    # 2. Pre-trained custom checkpoint
       'yolov8x.pt',                 # 3. General COCO high-capacity model
       'yolov8m.pt',                 # 4. General COCO medium model
       'yolov8n.pt'                  # 5. Lightweight nano fallback
   ]
   ```
2. **Ball Detection Resolution & Sensitivity**:
   - `BALL_IMGSZ = 1280`: Required to prevent sub-pixel downsampling of small, fast-moving tennis balls.
   - `BALL_CONF_THRESHOLD = 0.08`: Sensitive floor to detect motion-blurred balls.
3. **Player Identity Attribution**:
   - `PlayerTracker` splits candidates across the net horizon (`net_y_ratio = 0.50`):
     - **Player 1**: Near/bottom court ($Y \ge \text{net}$).
     - **Player 2**: Far/top court ($Y < \text{net}$).
4. **Trajectory Interpolation Safety Guardrails**:
   - `MAX_MISSING_FRAMES = 8`: Ball gaps larger than 8 frames are left as `None` to prevent distortion.
   - `MAX_BALL_SPEED_PIXELS = 220`: Kinematic jump filter rejects implausible cross-frame displacements.
   - `SCENE_CUT_THRESHOLD = 0.60`: 2D HSV histogram correlation resets tracking state across camera cuts.

---

## 🛠️ How to Run & Verify the Work

Always activate the virtual environment first:
```bash
source venv/bin/activate
```

### 1. Run Single-Match Pipeline
```bash
python main.py
```
*Processes `data/inputs/input.mp4` and exports annotated video to `data/outputs/output.mp4`, telemetry to `data/analysis/match_summary.json`, and reports to `data/analysis/match_report.html`.*

### 2. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
*Launches multi-tab dashboard at `http://localhost:8501` featuring Video Playback, AI Coaching, Court Heatmaps, and Tournament Stats.*

### 3. Run Multi-Match Batch Processing
```bash
python batch_process.py
```
*Processes all videos in `data/inputs/` and compiles `data/analysis/tournament_summary.json`.*

### 4. Run Model Training & MLOps Pipeline
```bash
# Step A: Harvest frames from all match videos
python training/extract_frames.py --sample_rate 5 --max_frames 400

# Step B: Auto-label dataset and create train/val split
python training/auto_label.py

# Step C: Fine-tune custom YOLOv8 model (auto-deploys to best_tennis.pt)
python training/train_ball_detector.py --epochs 25 --batch 8 --imgsz 640

# Step D: Generate AI diagnostics on training loss
python training/train_diagnostics.py
```

### 5. Run Local LLM/VLM Copilots (Ollama Required)
Ensure `ollama serve` is active with `qwen2.5` and `moondream`:
```bash
# Executive Tactical Scouting Report
python src/analysis/llm_scout.py

# Active VLM Label Denoising
python training/vlm_verifier.py

# Video Triage & Rally Filter
python training/video_triage.py --video data/inputs/input_grass.mp4
```

---

## 🚀 Recommended Next Steps & Potential Extensions

If you are continuing development, here are high-impact enhancement opportunities:

1. **Player Pose Estimation for Advanced Biomechanics**:
   - Integrate `yolov8n-pose.pt` to extract 17 player skeleton keypoints (wrist, elbow, shoulder, knee).
   - Compute racket preparation angle, knee flexion depth on serve, and kinetic chain metrics.
2. **Shot Heatmaps & Trajectory Clustering**:
   - Extend `PlayerAnalytics` to generate ball landing heatmaps (service box depth, baseline shot depth).
   - Cluster shot trajectories into crosscourt vs. down-the-line distributions.
3. **Automated Scoreboard OCR / Score Tracking**:
   - Add a lightweight OCR module (e.g. EasyOCR / Tesseract) to crop and track live broadcast scoreboards (games, sets, points).
4. **Cloud / Webhook Export**:
   - Add automated export connectors to cloud storage (S3 / GCS) or automated PDF report generation.
