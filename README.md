# 🎾 AI Tennis Visualiser & Analytics Engine

An end-to-end computer vision and deep learning platform designed to track tennis players, estimate ball trajectories, detect court boundaries via perspective homography, classify stroke biomechanics, and deliver automated AI coaching analytics from standard broadcast video feeds.

Inspired by and building upon the architecture of [`abdullahtarek/tennis_analysis`](https://github.com/abdullahtarek/tennis_analysis).

---

## 🌟 Comprehensive Feature Suite

### 1. Multi-Object Tracking & Ball Trajectory
- **Persistent 2-Player Re-ID (`PlayerTracker`)**: Spatially partitions court players across the net line into **Player 1** (near court) and **Player 2** (far court), filtering out referees and ball boys.
- **Dedicated High-Resolution Ball Detector (`TennisBallDetector`)**: Multi-scale 1280px inference to detect small, motion-blurred tennis balls.
- **Missing Frame Interpolation**: Segmented linear interpolation bounded by `MAX_MISSING_FRAMES = 8` to reconstruct occluded ball paths.
- **4 Tracking Safety Guardrails**: Velocity/distance jump filters ($220\text{px}$), track gap resets, strict court ROI polygon masking, and HSV 2D histogram scene cut detection.

### 2. Geometry, Homography & 2D Top-Down Radar
- **14 Court Keypoints Extraction (`CourtLineDetector`)**: Detects standard ITF court intersections with broadcast calibration fallback.
- **Perspective Homography Engine (`MiniCourt`)**: Projective transformation matrices mapping camera pixels $(u, v)$ to top-down 2D canvas pixels and real-world metric meters ($23.77\text{m} \times 10.97\text{m}$).
- **2D Mini-Court Bird's-Eye Radar**: Real-time overlay in the top-right corner tracking Player 1, Player 2, and ball trajectory trails.

### 3. Shot Analytics & Biomechanics
- **Hit Event Detection**: Inflection point analysis ($\Delta v_y$) attributing shots to Player 1 or Player 2 with adaptive debouncing.
- **Physical Ball Velocity ($\text{km/h}$)**: Real-world metric speed measurement over frame deltas.
- **Automated In/Out Line Calling**: Evaluates ball contact against official ITF singles boundary lines.
- **Stroke Classification (`StrokeClassifier`)**: Classifies shots into **Serve**, **Forehand**, **Backhand**, or **Volley/Smash**.
- **Live Rally Counter**: Real-time on-screen counter tracking consecutive shots.

### 4. Player Kinetics & Exertion
- **Instantaneous Running Speed ($\text{km/h}$)**: Rolling 5-frame velocity measurement capturing sprint bursts ($0 - 32\text{ km/h}$).
- **Cumulative Distance Covered ($m$)**: Integrates player movement meters across all rallies.
- **2D Positional Heatmaps**: High-resolution Gaussian density heatmaps (`data/analysis/heatmap_player_*.png`) showing tactical court distribution.

### 5. Automated AI Coaching Intelligence
- **Tactical Profiling (`CoachingInsightsGenerator`)**: Identifies stroke bias (forehand dominance vs two-wing balance), court positioning efficiency, and match tempo rhythm.
- **Actionable Coaching Recommendations**: Generates player-specific tactical takeaways.

### 6. Production Dashboard & Reporting Suite
- **Broadcast Telemetry HUD**: On-screen overlay card showing live rally count, stroke type & ball speed ticker, player sprint speeds, cumulative distance, and `IN`/`OUT` badge.
- **Structured JSON Export (`data/analysis/match_summary.json`)**: Full machine-readable match telemetry.
- **Standalone HTML Report (`data/analysis/match_report.html`)**: Responsive dark-mode report with KPI cards, head-to-head comparison tables, and embedded heatmaps.
- **Interactive Streamlit Web Dashboard (`app.py`)**: Multi-tab web application for video playback, parameter tuning, tactical coaching review, tournament statistics, and report downloads.

### 7. Multi-Video Training Pipeline & Batch MLOps
- **Multi-Video Frame Harvester (`training/extract_frames.py`)**: Slices sampled training frames across multiple videos.
- **Semi-Supervised Auto-Labeler (`training/auto_label.py`)**: Generates YOLO format dataset splits (`train/val`) and `data.yaml`.
- **Custom Model Training CLI (`training/train_ball_detector.py`)**: Fine-tunes YOLOv8 at 1280px and deploys to `best_tennis.pt`.
- **Multi-Match Batch Processor (`batch_process.py`)**: Runs the pipeline over multiple match files and aggregates tournament summaries.

### 8. Local LLM & Vision-Language Model (VLM) Copilot Suite
- **AI Training Diagnostics Copilot (`training/train_diagnostics.py`)**: Connects to local **Qwen 2.5** via Ollama to diagnose training convergence, analyze loss curves, and recommend custom ML optimizations.
- **Executive Tactical Scouting Engine (`src/analysis/llm_scout.py`)**: Converts match telemetry into narrative coaching reports and tactical game plans.
- **VLM Active Label Verifier (`training/vlm_verifier.py`)**: Employs local **Moondream** vision model to audit ambiguous ball crops and eliminate false positives.
- **Smart Video Triage & Rally Segmentation (`training/video_triage.py`)**: Uses local VLM to filter out non-play footage (replays, crowd close-ups, breaks) before dataset extraction.

---

## 📋 Prerequisites & System Requirements

Before getting started, ensure you have the following installed:

- **Python**: `3.9` to `3.12+` (tested on macOS Apple Silicon, Linux, and Windows)
- **Virtual Environment**: `venv` or `conda`
- **FFmpeg** (Recommended for video codec handling):
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
- **Hardware Acceleration** *(Optional but recommended)*:
  - Apple Silicon GPU (Metal/MPS)
  - NVIDIA GPU with CUDA
  - Multi-core CPU fallback supported
- **Local LLM/VLM Copilots** *(Optional - for Phase 8 AI diagnostics & scouting)*:
  - Install [Ollama](https://ollama.com/)
  - Pull models:
    ```bash
    ollama pull qwen2.5:latest
    ollama pull moondream:latest
    ollama serve
    ```

---

## 🚀 Quick Start & CLI Guide

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/water-bear-dev/tennis-visualiser.git
cd tennis-visualiser

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Upgrade pip and install all prerequisites
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Running a Single Match Analysis
Place your match video into `data/inputs/input.mp4` and run:
```bash
python main.py
```
**Generated Outputs**:
- `data/outputs/output.mp4` (Annotated broadcast video with HUD & 2D radar)
- `data/analysis/match_summary.json` (Structured telemetry export)
- `data/analysis/match_report.html` (Standalone HTML match report)
- `data/analysis/heatmap_player_1.png` & `heatmap_player_2.png` (Court heatmaps)

---

### 3. Running Batch Processing on Multiple Match Videos
Place all your `.mp4` match videos into `data/inputs/` and run:
```bash
python batch_process.py
```
This processes all matches sequentially and outputs:
- `data/outputs/{match_name}_annotated.mp4` for each video.
- `data/analysis/{match_name}_summary.json` for each video.
- `data/analysis/tournament_summary.json` aggregating stats across all matches.

---

### 4. Training a Custom Model on Multiple Videos

#### Step A: Extract frames from all videos in `data/inputs/`
```bash
python training/extract_frames.py --sample_rate 5 --max_frames 400
```

#### Step B: Auto-label extracted frames & create dataset splits
```bash
python training/auto_label.py
```

#### Step C: Fine-tune custom YOLOv8 ball detector
```bash
python training/train_ball_detector.py --epochs 25 --batch 16 --imgsz 640
```
*When training finishes, the best checkpoint is automatically deployed as `best_tennis.pt` in the project root and used by the analytics pipeline.*

---

### 5. Local LLM & VLM Copilot Commands

Ensure local Ollama is running (`ollama serve`), then run:

#### AI Training Diagnostics (Powered by Qwen 2.5)
```bash
python training/train_diagnostics.py
```
*Outputs diagnostic insights to `runs/detect/training_diagnostic_report.md`.*

#### Pro Tactical Scouting & Coaching Report (Powered by Qwen 2.5)
```bash
python src/analysis/llm_scout.py
```
*Outputs narrative coaching report to `data/analysis/tactical_scouting_report.md`.*

#### VLM Active Label Verification (Powered by Moondream)
```bash
python training/vlm_verifier.py
```

#### Smart Video Triage & Rally Segmentation (Powered by Moondream)
```bash
python training/video_triage.py --video data/inputs/input_grass.mp4
```

---

### 6. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Features 4 dedicated interactive tabs:
- **📺 Match Video & Telemetry**: Video player and head-to-head metrics.
- **🧠 AI Coaching Insights**: Stroke distribution charts & tactical recommendations.
- **🗺️ Court Heatmaps**: High-resolution 2D tactical court coverage maps.
- **🏆 Tournament & Multi-Match Batch**: Aggregated multi-match comparisons.

---

## 📁 Repository Structure

```
tennis-visualiser/
├── data/
│   ├── inputs/                    # Raw input videos (data/inputs/*.mp4)
│   ├── outputs/                   # Processed annotated videos
│   └── analysis/                  # Post-match JSON/HTML reports & heatmaps
├── training/
│   ├── extract_frames.py          # Multi-video frame harvester
│   ├── auto_label.py              # Semi-supervised pseudo-labeling engine
│   ├── train_ball_detector.py     # Custom YOLOv8 fine-tuner
│   ├── train_diagnostics.py       # Local LLM training diagnostics copilot (Qwen 2.5)
│   ├── vlm_verifier.py            # Local VLM active label verification (Moondream)
│   └── video_triage.py            # Smart VLM video triage & rally segmentation
├── src/
│   ├── config.py                  # Tunable thresholds, paths, ROI coordinates
│   ├── utils/
│   │   ├── roi_utils.py           # Spatial polygon coordinate conversions
│   │   └── scene_utils.py         # HSV histogram comparison for scene cuts
│   ├── detectors/
│   │   ├── ball_detector.py       # Dedicated 1280px high-res ball detector
│   │   └── yolo_detector.py       # Player inference & Pass 1 orchestrator
│   ├── trackers/
│   │   ├── ball_interpolator.py   # Time-series interpolation & gap limits
│   │   └── player_tracker.py      # Net-split persistent 2-player Re-ID
│   ├── court_detector/
│   │   └── court_line_detector.py # 14 court keypoint extraction
│   ├── mini_court/
│   │   └── mini_court.py          # Homography engine & 2D top-down radar
│   ├── analysis/
│   │   ├── shot_detector.py       # Hit detection, ball speed (km/h) & line calls
│   │   ├── stroke_classifier.py   # Forehand, Backhand, Serve, Volley classifier
│   │   ├── player_analytics.py    # Player running speeds, distance & heatmaps
│   │   ├── coaching_insights.py   # Automated AI coaching & tactical profiling
│   │   ├── llm_scout.py           # Local LLM narrative tactical scouting engine (Qwen 2.5)
│   │   └── report_generator.py    # JSON & HTML report compilation
│   └── visualizers/
│       └── video_annotator.py     # Pass 2 rendering, HUD, polylines & radar
├── app.py                         # Multi-tab Streamlit Web Dashboard
├── batch_process.py               # Batch executor for multi-match queues
├── features.md                    # 8-Phase Roadmap & Feature Matrix
├── development_blog.md            # Technical engineering journal (14 entries)
├── main.py                        # Single-match pipeline entrypoint
├── requirements.txt               # Dependencies
└── .gitignore
```

---

## 🗺️ Roadmap & Documentation
- Full implementation breakdown: **[`features.md`](features.md)**
- Engineering journal & decision logs: **[`development_blog.md`](development_blog.md)**
