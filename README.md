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

---

## 🚀 Quick Start & CLI Guide

### 1. Installation
```bash
git clone https://github.com/water-bear-dev/tennis-visualiser.git
cd tennis-visualiser
python3 -m venv venv
source venv/bin/activate
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

#### Step C: Fine-tune custom YOLOv8 ball detector at 1280px resolution
```bash
python training/train_ball_detector.py --epochs 50 --batch 8 --imgsz 1280
```
*When training finishes, the best checkpoint is automatically deployed as `best_tennis.pt` in the project root and used by the analytics pipeline.*

---

### 5. Launch Interactive Web Dashboard
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
│   └── train_ball_detector.py      # Custom 1280px YOLOv8 fine-tuner
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
│   │   └── report_generator.py    # JSON & HTML report compilation
│   └── visualizers/
│       └── video_annotator.py     # Pass 2 rendering, HUD, polylines & radar
├── app.py                         # Multi-tab Streamlit Web Dashboard
├── batch_process.py               # Batch executor for multi-match queues
├── features.md                    # 7-Phase Roadmap & Feature Matrix
├── development_blog.md            # Technical engineering journal (13 entries)
├── main.py                        # Single-match pipeline entrypoint
├── requirements.txt               # Dependencies
└── .gitignore
```

---

## 🗺️ Roadmap & Documentation
- Full implementation breakdown: **[`features.md`](features.md)**
- Engineering journal & decision logs: **[`development_blog.md`](development_blog.md)**
