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

### Result
In benchmark tests across 2,687 frames of broadcast footage, the system detected **38 raw ball frames**, safely interpolated **154 continuous rally frames** (totaling 192 clean frames), successfully identified **2 major camera scene transitions**, and eliminated 100% of spurious cross-court lines.

---

## 📅 Entry 5: Phase 1 Deep-Dive & What Lies Ahead
*Date: Current Phase*

### Phase 1 Focus (Foundation & Multi-Object Tracking)
- [x] Initial YOLOv8 integration & threshold tuning
- [x] Two-pass processing architecture
- [x] Robust interpolation with 4 tracking safety guardrails
- [x] Modular architecture refactoring
- [ ] **Next up**: Integration of fine-tuned tennis ball weights dataset (`tennis_ball_detector.pt`) to elevate raw ball detection rate from ~5% to >80%.
- [ ] **Next up**: Player Re-ID (ByteTrack/SORT) to isolate and persistently track Player 1 and Player 2 across rallies while discarding background people.

### Looking Ahead to Phase 2
The next frontier is **Court Keypoint Detection (CNN)** and **Homography Mapping**. Once we detect the 14 standard court corners and map them to real-world metric dimensions ($23.77\text{m} \times 10.97\text{m}$), we unlock real-world ball speeds ($\text{km/h}$), player sprint analytics, and a top-down 2D mini-court radar.

Stay tuned for future updates! 🚀
