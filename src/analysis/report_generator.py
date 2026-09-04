"""
src/analysis/report_generator.py
Match Telemetry and HTML Report Generation Module.
Compiles high-level rally statistics, speed distributions, player physical exertion metrics,
embeds Base64 court heatmaps, and writes structured JSON and standalone dark-mode HTML reports.
"""

import base64
import json
import os
import cv2
import numpy as np
from src.analysis.coaching_insights import CoachingInsightsGenerator
from src.config import MATCH_SUMMARY_PATH, MATCH_REPORT_PATH, HEATMAP_P1_PATH, HEATMAP_P2_PATH


class MatchReportGenerator:
    """
    Compiles aggregate match statistics, stroke distributions, and AI coaching insights
    and saves them to the data/analysis/ directory.
    """

    def __init__(self, fps: int = 30):
        """
        Initializes the report generator.

        Args:
            fps (int): Video frame rate for time calculations.
        """
        self.fps = fps
        self.coaching_generator = CoachingInsightsGenerator()

    def _encode_image_base64(self, filepath: str) -> str:
        """
        Reads an image file from disk and encodes it into a Base64 string for embedding directly in standalone HTML.

        Args:
            filepath (str): Absolute or relative image path.

        Returns:
            str: Base64 UTF-8 encoded image string, or empty string if file missing.
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return ""

    def generate_report(self, frame_detections: list[dict], telemetry_per_frame: list[dict], kinetics_per_frame: list[dict]):
        """
        Processes telemetry streams and exports JSON and HTML reports to data/analysis/.

        Args:
            frame_detections (list[dict]): Pass 1 frame detections.
            telemetry_per_frame (list[dict]): Shot speeds, rally counts, and line calls.
            kinetics_per_frame (list[dict]): Player running speeds and cumulative meters.
        """
        print("\n--- Phase 5 & 6: Generating Match Reports & AI Coaching Insights ---")
        total_frames = len(frame_detections)
        duration_s = total_frames / float(self.fps)

        # 1. Aggregate Statistics
        rally_counts = [t.get('rally_count', 0) for t in telemetry_per_frame]
        max_rally = max(rally_counts) if rally_counts else 0
        total_rallies = sum(1 for i in range(1, len(rally_counts)) if rally_counts[i] == 1 and rally_counts[i-1] == 0)
        total_rallies = max(total_rallies, 1 if max_rally > 0 else 0)

        # Speeds
        speeds = [t.get('shot_speed_kmh', 0.0) for t in telemetry_per_frame if t.get('is_hit', False) and t.get('shot_speed_kmh', 0.0) > 0]
        max_speed = max(speeds) if speeds else 0.0
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        total_shots = sum(1 for t in telemetry_per_frame if t.get('is_hit', False))

        p1_shots = sum(1 for t in telemetry_per_frame if t.get('is_hit', False) and t.get('last_hitter') == 'Player 1')
        p2_shots = sum(1 for t in telemetry_per_frame if t.get('is_hit', False) and t.get('last_hitter') == 'Player 2')

        # Kinetics
        p1_speeds = [k.get('p1_speed_kmh', 0.0) for k in kinetics_per_frame if k.get('p1_speed_kmh', 0.0) > 0]
        p2_speeds = [k.get('p2_speed_kmh', 0.0) for k in kinetics_per_frame if k.get('p2_speed_kmh', 0.0) > 0]
        p1_max_spd = max(p1_speeds) if p1_speeds else 0.0
        p2_max_spd = max(p2_speeds) if p2_speeds else 0.0
        p1_avg_spd = float(np.mean(p1_speeds)) if p1_speeds else 0.0
        p2_avg_spd = float(np.mean(p2_speeds)) if p2_speeds else 0.0

        p1_total_dist = kinetics_per_frame[-1].get('p1_dist_m', 0.0) if kinetics_per_frame else 0.0
        p2_total_dist = kinetics_per_frame[-1].get('p2_dist_m', 0.0) if kinetics_per_frame else 0.0

        in_calls = sum(1 for t in telemetry_per_frame if t.get('call') == 'IN')
        out_calls = sum(1 for t in telemetry_per_frame if t.get('call') == 'OUT')

        base_summary = {
            "match_overview": {
                "total_frames": total_frames,
                "duration_seconds": round(duration_s, 2),
                "fps": self.fps,
                "total_rallies": total_rallies,
                "longest_rally_shots": max_rally,
                "total_shots_played": total_shots,
                "in_calls": in_calls,
                "out_calls": out_calls
            },
            "ball_metrics": {
                "peak_shot_speed_kmh": round(max_speed, 1),
                "average_shot_speed_kmh": round(avg_speed, 1),
                "total_shots_recorded": len(speeds)
            },
            "player_1_near_court": {
                "total_distance_meters": round(p1_total_dist, 1),
                "peak_running_speed_kmh": round(p1_max_spd, 1),
                "average_running_speed_kmh": round(p1_avg_spd, 1),
                "shots_played": p1_shots
            },
            "player_2_far_court": {
                "total_distance_meters": round(p2_total_dist, 1),
                "peak_running_speed_kmh": round(p2_max_spd, 1),
                "average_running_speed_kmh": round(p2_avg_spd, 1),
                "shots_played": p2_shots
            }
        }

        # 2. Generate AI Coaching Insights
        coaching_data = self.coaching_generator.generate_coaching_report(base_summary, telemetry_per_frame)
        base_summary["ai_coaching_intelligence"] = coaching_data

        # 3. Export JSON to data/analysis/
        with open(MATCH_SUMMARY_PATH, 'w') as f:
            json.dump(base_summary, f, indent=4)
        print(f"Exported: '{MATCH_SUMMARY_PATH}'")

        # 4. Export HTML Report to data/analysis/
        p1_heatmap_b64 = self._encode_image_base64(HEATMAP_P1_PATH)
        p2_heatmap_b64 = self._encode_image_base64(HEATMAP_P2_PATH)

        strokes_p1 = coaching_data["stroke_breakdown"]["player_1"]
        strokes_p2 = coaching_data["stroke_breakdown"]["player_2"]
        p1_adv_html = "".join([f"<li>{adv}</li>" for adv in coaching_data["player_1_tactical_insights"]])
        p2_adv_html = "".join([f"<li>{adv}</li>" for adv in coaching_data["player_2_tactical_insights"]])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Tennis Match Analytics & Coaching Report</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #f0f6fc;
            --text-muted: #8b949e;
            --accent-p1: #388bfd;
            --accent-p2: #f0883e;
            --accent-gold: #e3b341;
            --accent-green: #3fb950;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 30px 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{ text-align: center; border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 30px; }}
        h1 {{ margin: 0 0 10px 0; font-size: 2.2rem; color: #ffffff; letter-spacing: -0.5px; }}
        .subtitle {{ color: var(--text-muted); font-size: 1rem; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 30px; }}
        .card {{ background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        .card-label {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 8px; }}
        .card-value {{ font-size: 1.9rem; font-weight: 700; color: #ffffff; }}
        .card-value.gold {{ color: var(--accent-gold); }}
        .card-value.green {{ color: var(--accent-green); }}
        
        .section-title {{ font-size: 1.4rem; margin: 30px 0 16px 0; }}
        table.h2h-table {{ width: 100%; border-collapse: collapse; background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
        table.h2h-table th, table.h2h-table td {{ padding: 16px 20px; text-align: center; border-bottom: 1px solid var(--border); }}
        table.h2h-table th {{ background-color: #21262d; font-size: 1rem; }}
        table.h2h-table tr:last-child td {{ border-bottom: none; }}
        .p1-header {{ color: var(--accent-p1); font-weight: 700; }}
        .p2-header {{ color: var(--accent-p2); font-weight: 700; }}
        .stat-name {{ color: var(--text-muted); font-weight: 600; text-align: left; }}
        
        .coaching-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .coaching-card {{ background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
        .coaching-card ul {{ margin: 10px 0 0 0; padding-left: 20px; color: #d0d7de; line-height: 1.6; }}
        
        .heatmaps-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .heatmap-box {{ background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 16px; text-align: center; }}
        .heatmap-box img {{ max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border); }}
        footer {{ text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎾 AI Tennis Match Analytics & Coaching Report</h1>
            <div class="subtitle">Broadcast Computer Vision, Biomechanical Stroke Classification & Tactical Intelligence</div>
        </header>

        <div class="grid-4">
            <div class="card">
                <div class="card-label">Total Match Rallies</div>
                <div class="card-value gold">{total_rallies}</div>
            </div>
            <div class="card">
                <div class="card-label">Longest Rally</div>
                <div class="card-value gold">{max_rally} <span style="font-size:1rem;font-weight:400">shots</span></div>
            </div>
            <div class="card">
                <div class="card-label">Peak Ball Shot Speed</div>
                <div class="card-value green">{max_speed:.1f} <span style="font-size:1rem;font-weight:400">km/h</span></div>
            </div>
            <div class="card">
                <div class="card-label">Total Distance Ran</div>
                <div class="card-value">{(p1_total_dist + p2_total_dist):.1f} <span style="font-size:1rem;font-weight:400">meters</span></div>
            </div>
        </div>

        <h2 class="section-title">🧠 AI Tactical Coaching Intelligence</h2>
        <div style="background-color:var(--card-bg);border:1px solid var(--border);border-left:4px solid var(--accent-gold);border-radius:8px;padding:14px 20px;margin-bottom:20px;">
            <strong>Match Tempo Profile:</strong> {coaching_data["match_tempo_assessment"]}
        </div>
        <div class="coaching-grid">
            <div class="coaching-card" style="border-top:3px solid var(--accent-p1);">
                <h3 style="color:var(--accent-p1);margin-top:0;">Player 1 Tactical Report</h3>
                <p style="font-size:0.9rem;color:var(--text-muted);margin:4px 0 10px 0;">
                    Strokes: {strokes_p1.get('FOREHAND',0)} FH | {strokes_p1.get('BACKHAND',0)} BH | {strokes_p1.get('SERVE',0)} Serve | {strokes_p1.get('VOLLEY',0)} Volley
                </p>
                <ul>{p1_adv_html}</ul>
            </div>
            <div class="coaching-card" style="border-top:3px solid var(--accent-p2);">
                <h3 style="color:var(--accent-p2);margin-top:0;">Player 2 Tactical Report</h3>
                <p style="font-size:0.9rem;color:var(--text-muted);margin:4px 0 10px 0;">
                    Strokes: {strokes_p2.get('FOREHAND',0)} FH | {strokes_p2.get('BACKHAND',0)} BH | {strokes_p2.get('SERVE',0)} Serve | {strokes_p2.get('VOLLEY',0)} Volley
                </p>
                <ul>{p2_adv_html}</ul>
            </div>
        </div>

        <h2 class="section-title">⚔️ Head-to-Head Kinetic Breakdown</h2>
        <table class="h2h-table">
            <thead>
                <tr>
                    <th style="width:40%; text-align:left;">Metric</th>
                    <th class="p1-header" style="width:30%;">Player 1 (Near Court)</th>
                    <th class="p2-header" style="width:30%;">Player 2 (Far Court)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="stat-name">Total Distance Covered</td>
                    <td style="font-weight:700;color:var(--accent-p1);">{p1_total_dist:.1f} meters</td>
                    <td style="font-weight:700;color:var(--accent-p2);">{p2_total_dist:.1f} meters</td>
                </tr>
                <tr>
                    <td class="stat-name">Peak Sprint Speed</td>
                    <td>{p1_max_spd:.1f} km/h</td>
                    <td>{p2_max_spd:.1f} km/h</td>
                </tr>
                <tr>
                    <td class="stat-name">Average Running Speed</td>
                    <td>{p1_avg_spd:.1f} km/h</td>
                    <td>{p2_avg_spd:.1f} km/h</td>
                </tr>
                <tr>
                    <td class="stat-name">Total Strokes Played</td>
                    <td>{p1_shots} shots</td>
                    <td>{p2_shots} shots</td>
                </tr>
            </tbody>
        </table>

        <h2 class="section-title">🗺️ 2D Court Spatial Heatmaps</h2>
        <div class="heatmaps-container">
            <div class="heatmap-box">
                <h3 style="color:var(--accent-p1);margin-top:0;">Player 1 Tactical Coverage</h3>
                {'<img src="data:image/png;base64,' + p1_heatmap_b64 + '" alt="P1 Heatmap">' if p1_heatmap_b64 else '<p>Heatmap generated.</p>'}
            </div>
            <div class="heatmap-box">
                <h3 style="color:var(--accent-p2);margin-top:0;">Player 2 Tactical Coverage</h3>
                {'<img src="data:image/png;base64,' + p2_heatmap_b64 + '" alt="P2 Heatmap">' if p2_heatmap_b64 else '<p>Heatmap generated.</p>'}
            </div>
        </div>

        <footer>
            Generated automatically by AI Tennis Visualiser & Analytics Engine. Modeled after abdullahtarek/tennis_analysis.
        </footer>
    </div>
</body>
</html>
"""
        with open(MATCH_REPORT_PATH, 'w') as f:
            f.write(html_content)
        print(f"Exported: '{MATCH_REPORT_PATH}'")
