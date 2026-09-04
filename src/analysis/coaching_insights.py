"""
src/analysis/coaching_insights.py
Automated AI Coaching Insights and Tactical Profiling Engine.
Synthesizes stroke distribution, court coverage metrics, sprint bursts, and rally dynamics
into actionable coaching recommendations and match tempo assessments.
"""

import math
import numpy as np


class CoachingInsightsGenerator:
    """
    Analyzes match telemetry, stroke distribution, and player kinetics to generate
    automated AI coaching takeaways, tactical depth metrics, and performance advice.
    """

    def __init__(self):
        """Initializes the coaching insights generator."""
        pass

    def generate_coaching_report(self, match_summary: dict, telemetry_per_frame: list[dict]) -> dict:
        """
        Synthesizes technical and tactical coaching insights from match metrics.

        Args:
            match_summary (dict): High-level statistical match summary dictionary.
            telemetry_per_frame (list[dict]): Full per-frame telemetry history from shot detection.

        Returns:
            dict: Structured coaching intelligence report containing:
                - 'match_tempo_assessment' (str): Tactical pace classification.
                - 'stroke_breakdown' (dict): Forehand/Backhand/Serve/Volley counts for P1 & P2.
                - 'player_1_tactical_insights' (list[str]): Actionable coaching recommendations for P1.
                - 'player_2_tactical_insights' (list[str]): Actionable coaching recommendations for P2.
        """
        p1 = match_summary.get("player_1_near_court", {})
        p2 = match_summary.get("player_2_far_court", {})
        ball = match_summary.get("ball_metrics", {})
        overview = match_summary.get("match_overview", {})

        # ----------------------------------------------------------------------
        # 1. Aggregate Stroke Distribution per Player
        # ----------------------------------------------------------------------
        strokes_p1 = {"FOREHAND": 0, "BACKHAND": 0, "SERVE": 0, "VOLLEY": 0}
        strokes_p2 = {"FOREHAND": 0, "BACKHAND": 0, "SERVE": 0, "VOLLEY": 0}

        for t in telemetry_per_frame:
            if t.get('is_hit', False):
                st_type = t.get('stroke_type', 'FOREHAND')
                hitter = t.get('last_hitter')
                if hitter == "Player 1":
                    strokes_p1[st_type] = strokes_p1.get(st_type, 0) + 1
                elif hitter == "Player 2":
                    strokes_p2[st_type] = strokes_p2.get(st_type, 0) + 1

        # ----------------------------------------------------------------------
        # 2. Extract Key Tactical Heuristics & Metrics
        # ----------------------------------------------------------------------
        p1_dist = p1.get('total_distance_meters', 0.0)
        p2_dist = p2.get('total_distance_meters', 0.0)
        p1_speed = p1.get('peak_running_speed_kmh', 0.0)
        p2_speed = p2.get('peak_running_speed_kmh', 0.0)
        peak_ball_spd = ball.get('peak_shot_speed_kmh', 0.0)

        # Tactical Takeaways
        p1_advice = []
        p2_advice = []

        # ----------------------------------------------------------------------
        # Player 1 Tactical Analysis
        # ----------------------------------------------------------------------
        if strokes_p1.get('FOREHAND', 0) > strokes_p1.get('BACKHAND', 0) * 1.5:
            p1_advice.append("Heavy Forehand Bias: Strong tactical tendency to dictate play on the forehand wing.")
        else:
            p1_advice.append("Balanced Groundstrokes: Solid baseline consistency between forehand and backhand wings.")

        if p1_dist > p2_dist * 1.15:
            p1_advice.append("High Court Coverage: Covered significantly more court distance; consider aiming deeper to push opponent back.")
        else:
            p1_advice.append("Efficient Court Positioning: Controlled the center of the court with minimal wasted lateral movement.")

        if strokes_p1.get('VOLLEY', 0) >= 2:
            p1_advice.append("Aggressive Net Transition: Successfully transitioned into the forecourt for finishing volleys.")

        # ----------------------------------------------------------------------
        # Player 2 Tactical Analysis
        # ----------------------------------------------------------------------
        if strokes_p2.get('FOREHAND', 0) > strokes_p2.get('BACKHAND', 0) * 1.5:
            p2_advice.append("Forehand Dominance: Aggressive offensive patterns created when attacking off the forehand.")
        else:
            p2_advice.append("Two-Wing Baseline Defense: Stable rally tolerance with reliable backhand placement.")

        if p2_speed > 18.0:
            p2_advice.append(f"Explosive Sprint Recovery: Clocked peak burst speed of {p2_speed:.1f} km/h on defensive scramble.")
        else:
            p2_advice.append("Steady Recovery Pace: Good baseline anchoring during extended rallies.")

        # ----------------------------------------------------------------------
        # Match Tempo & Style Classification
        # ----------------------------------------------------------------------
        avg_rally = overview.get('longest_rally_shots', 1)
        if avg_rally > 8:
            tempo = "High-Endurance Baseline War (Extended rallies with heavy baseline exchanges)."
        elif peak_ball_spd > 150.0:
            tempo = "High-Pace Offensive Clash (Fast aggressive points driven by pace of shot)."
        else:
            tempo = "Tactical Placement Match (Strategic point construction with moderate rally lengths)."

        return {
            "match_tempo_assessment": tempo,
            "stroke_breakdown": {
                "player_1": strokes_p1,
                "player_2": strokes_p2
            },
            "player_1_tactical_insights": p1_advice,
            "player_2_tactical_insights": p2_advice
        }
