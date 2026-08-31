# 🎾 Development Blog: Building an AI Tennis Analytics Platform

Welcome to the engineering journal of the **AI Tennis Visualiser & Analytics Engine**. This blog documents our design decisions, real-world computer vision challenges, algorithmic breakthroughs, and lessons learned while transforming raw match video into broadcast-grade analytics.

---

## 📅 Entry 1: The Baseline & The "Invisible Ball" Problem
*Date: Kickoff Phase*

### The Goal
Build a working baseline script to detect tennis players and the ball in `input.mp4` using standard YOLOv8 (`yolov8n.pt`).

### What Happened
Running off-the-shelf YOLOv8 on 1080p broadcast video produced clean bounding boxes around the players, line judges, and ball boys. However, the tennis ball was **completely missing** in 99% of frames.

### Root Cause Analysis
1. **Motion Blur & Scale**: A tennis ball traveling at 150–200 km/h covers dozens of pixels per frame, resulting in significant motion blur and faint edges.
2. **Confidence Threshold**: Default YOLO inference uses a confidence threshold of `0.25`. For a blurred, small object (class 32 in COCO), the model's confidence often hovers around `0.10 - 0.18`.
3. **Generalist vs. Domain-Specific Weights**: COCO class 32 covers generic "sports balls" (soccer balls, basketballs, tennis balls). The features are not optimized for a 6.7cm neon sphere moving at high speed across court lines.

### Solution
- Lowered the ball detection confidence threshold down to `0.12` while maintaining player confidence at `0.40`.
- Prepared the model initialization logic to dynamically search for fine-tuned weights (`best_tennis.pt`, `tennis_ball.pt`) before falling back to `yolov8n.pt`.

---

## 📅 Entry 2: Two-Pass Interpolation & The "Ghost Line" Glitch
*Date: Tracking Iteration 1*

### The Goal
Reconstruct ball paths during frames where the ball dropped detection due to occlusion behind the net or severe motion blur.

### What Happened
We implemented a two-pass pipeline:
1. **Pass 1**: Extract raw detections across all frames and store centroid coordinates `(x, y)`.
2. **Data Interpolation**: Apply linear interpolation using `pandas.DataFrame.interpolate()` to fill in missing coordinates between detected frames.
3. **Pass 2**: Draw continuous fading trajectory polylines.

While this smoothly tracked the ball during rallies, it created a severe visual glitch: **"Ghost Lines"**. Whenever the camera switched to a player close-up or during long gaps between serves, the interpolation drew straight lines across the screen, connecting the end of one rally to the beginning of the next stroke minutes later.

### Key Takeaway
Interpolation without temporal and spatial constraints is catastrophic in video processing. Real tennis rallies are discrete events, not infinite continuous lines.

---

## 📅 Entry 3: Architecture Refactoring – From Monolith to Modular Package
*Date: Refactoring Milestone*

As new filters and logic grew, `main.py` surpassed 250+ lines of mixed concerns (video IO, YOLO inference, polygon geometry, dataframe math, and video encoding).

We broke the monolithic codebase into a clean, single-responsibility architecture under `src/`:
- **`src/config.py`**: Central source of truth for thresholds, file paths, and normalized ROI points.
- **`src/utils/roi_utils.py`**: Geometric containment and pixel coordinate mappings.
- **`src/detectors/yolo_detector.py`**: Model loading and Pass 1 frame extraction.
- **`src/trackers/ball_interpolator.py`**: Time-series interpolation and gap bounding.
- **`src/visualizers/video_annotator.py`**: Pass 2 video rendering and multi-frame polyline buffers.
- **`main.py`**: Lightweight 40-line orchestrator.

This separation made testing and extending individual components dramatically simpler.

---

## 📅 Entry 4: The Four Guardrails of Reliable Ball Tracking
*Date: Safety & Stabilization Phase*

To permanently eliminate ghost trajectories, false-positive flickers, and camera transition artifacts, we developed a 4-pillar safety framework:

### 1. Temporal Track Reset (`MAX_MISSING_FRAMES = 7`)
If the ball is lost for more than 7 consecutive frames (~0.23 seconds at 30fps), we declare the track broken. The gap is left empty (`None`), and the visual trajectory queue is cleared.

### 2. Velocity & Physical Distance Filter (`MAX_BALL_SPEED_PIXELS = 180`)
A tennis ball cannot physically teleport across half the screen in 1/30th of a second. If Euclidean distance between consecutive detections exceeds $180\text{px}$, it is rejected as an outlier.

### 3. Strict ROI Polygon Masking
Detections outside the primary court polygon (spectators, advertising banners, distant ball boys) are strictly filtered out during candidate evaluation.

### 4. Scene Cut Reset via HSV Histogram Correlation
Using OpenCV 2D HSV color histograms, we compare frame $t$ with frame $t-1$. When a camera switch occurs (`correlation < 0.60`), the system triggers a hard reset: active tracks terminate, and interpolation across the cut boundary is strictly blocked.

---

## 📅 Entry 5: Completing Phase 1 – Persistent 2-Player Re-ID & Spatial Net Split
*Date: Phase 1 Finalization*

We engineered `src/trackers/player_tracker.py` to divide court space across the net line ($Y_{\text{net}}$):
- **Player 1**: Near court competitor (bottom of screen), rendered in Electric Blue.
- **Player 2**: Far court competitor (top of screen), rendered in Deep Orange.
- Filters out non-competitor clutter (line judges, ball kids) and preserves identity through rallies.

---

## 📅 Entry 6: Phase 2 Breakthrough – Homography & The 2D Mini-Court Radar
*Date: Phase 2 Implementation*

Using standard 14 court keypoints (`src/court_detector/court_line_detector.py`), we computed a perspective transformation matrix $\mathbf{H}$ (`src/mini_court/mini_court.py`) mapping camera pixels $(u, v)$ to top-down 2D canvas coordinates $(x', y')$ and metric real-world coordinates ($23.77\text{m} \times 10.97\text{m}$). We rendered a live top-down mini-court radar in the top-right corner.

---

## 📅 Entry 7: Phase 3 Breakthrough – Physics-Based Velocity, Shot Intelligence & Line Calling
*Date: Phase 3 Implementation*

Using metric homography coordinates, `src/analysis/shot_detector.py` introduced:
- **Hit Detection**: Analyzing velocity direction reversals ($\Delta v_y$) and player proximity.
- **Ball Velocity ($\text{km/h}$)**: Physical speed measurement over frame deltas.
- **Automated In/Out Calling**: Real-time line boundary intersection testing.
- **Live Rally Counters**: On-screen broadcast HUD displaying continuous rally stats.

---

## 📅 Entry 8: Phase 4 Milestone – Player Kinetics, Athletic Speed & Positional Heatmaps
*Date: Phase 4 Implementation*

In `src/analysis/player_analytics.py`, we expanded player biomechanics:
- **Running Speed ($\text{km/h}$)**: Real-time velocity with rolling window filtering.
- **Cumulative Distance ($m$)**: Integrated running distance across all points.
- **2D Court Heatmaps**: Exported `heatmap_player_1.png` and `heatmap_player_2.png` via 2D Gaussian density convolution.

---

## 📅 Entry 9: Phase 5 Milestone – The Production Delivery Suite & Interactive Dashboard
*Date: Phase 5 Implementation*

We finalized the production delivery architecture:
- **`match_summary.json` & `match_report.html`**: Exported structured JSON and standalone dark-mode HTML reports with embedded heatmaps.
- **Streamlit Web UI (`app.py`)**: Interactive video player, parameter sliders, and metric dashboards.

---

## 📅 Entry 10: Phase 6 Milestone – Biomechanical Stroke Classification & AI Coaching Intelligence
*Date: Phase 6 Implementation*

### From Raw Tracking to Tactical Coaching Insights
In Phase 6, we bridged computer vision with real tennis coaching strategy:

1. **Kinematic Stroke Classification (`src/analysis/stroke_classifier.py`)**:
   - Classifies each shot into **Serve**, **Forehand**, **Backhand**, or **Volley/Smash** based on impact coordinate geometry relative to player body center and net proximity.
   - Displays real-time stroke badges directly on the broadcast HUD (e.g. `FOREHAND: 148 km/h (P1)`).
2. **AI Coaching Intelligence Engine (`src/analysis/coaching_insights.py`)**:
   - Synthesizes tactical heuristics:
     - **Shot Distribution Profiling**: Identifies heavy forehand reliance vs. two-wing baseline balance.
     - **Court Positioning & Recovery**: Identifies baseline defense efficiency vs. high lateral fatigue.
     - **Match Tempo Assessment**: Profiles match rhythm (High-Pace Offensive Clash vs. Extended Baseline Grinding).
3. **Multi-Tab Web UI Upgrade (`app.py`)**:
   - Added a dedicated **🧠 AI Coaching Insights** tab displaying tactical reports, stroke distributions, and performance takeaways alongside video playback.

### Conclusion: Full Platform Maturity 🎾
The AI Tennis Visualiser now offers complete end-to-end intelligence: from raw broadcast pixels to kinematic biomechanics, spatial radar overlays, and automated AI coaching.
