import argparse
import json
import os
import sys
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import MATCH_SUMMARY_PATH, ANALYSIS_DIR

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


def query_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Queries local Ollama instance with tactical tennis analysis prompt."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1200
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Ollama Error (Status {response.status_code}): {response.text}"
    except Exception as e:
        return f"Could not connect to Ollama at {OLLAMA_API_URL}. Error: {str(e)}"


def generate_scouting_report(summary_path: str = MATCH_SUMMARY_PATH,
                             model: str = DEFAULT_MODEL) -> str:
    """
    Ingests match telemetry summary and prompts local Qwen model to generate
    an executive pro-level tennis scouting & coaching report.
    """
    print(f"\n=======================================================")
    print(f"🎾 AI Match & Tactical Scouting Engine (Powered by {model})")
    print(f"=======================================================")

    if not os.path.exists(summary_path):
        print(f"Error: Match summary '{summary_path}' not found. Run main.py or batch_process.py first.")
        return ""

    with open(summary_path, 'r') as f:
        match_data = json.load(f)

    prompt = f"""You are an elite ATP/WTA Tour Tennis Coach and Tactical Data Analyst.

Analyze the following structured match telemetry and produce an executive tactical coaching breakdown:

MATCH TELEMETRY:
```json
{json.dumps(match_data, indent=2)}
```

Please structure your coaching report with these 4 clear sections:
1. Match Summary & Dominance Overview: Who controlled the court geometry and pace?
2. Player 1 (Top Court) Profile & Shot Breakdown: Key strengths, shot patterns (forehand vs backhand vs serve), physical output.
3. Player 2 (Bottom Court) Profile & Shot Breakdown: Key strengths, shot patterns, physical output.
4. Strategic Coaching Directives: 3 specific tactical adjustments Player 1 and Player 2 should make for the next match.

Use clear, professional tennis terminology (e.g. rally tolerance, court depth, crosscourt angles, recovery kinetics).
"""

    print("🧠 Analyzing match kinetics and tactical patterns with Qwen...")
    scouting_report = query_ollama(prompt, model=model)

    print("\n--- Executive Scouting Report ---")
    print(scouting_report)
    print("---------------------------------\n")

    # Save scouting report
    report_file = os.path.join(ANALYSIS_DIR, "tactical_scouting_report.md")
    with open(report_file, "w") as f:
        f.write(f"# Professional Tennis Tactical & Scouting Report\n\n**Analyst:** {model}\n\n{scouting_report}\n")

    print(f"Tactical report saved to: '{report_file}'")
    return scouting_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate tactical scouting report with local LLM.")
    parser.add_argument('--summary', type=str, default=MATCH_SUMMARY_PATH, help="Path to match_summary.json")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help="Local Ollama model name")
    args = parser.parse_args()

    generate_scouting_report(args.summary, model=args.model)
