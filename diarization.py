import json
import sys
import os
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

load_dotenv()

AUDIO_FILE = sys.argv[1]
HF_TOKEN   = os.getenv("HF_TOKEN")
DEVICE     = "cpu"
LOW_CONF   = 0.6

base = os.path.splitext(AUDIO_FILE)[0]
json_path = base + "_transcript.json"

if not os.path.exists(json_path):
    print(f"❌ Could not find {json_path} — run the main script first.")
    sys.exit(1)

# ── STEP 1: Load existing transcript ─────────────────────────
print(f"Loading existing transcript from {json_path}...")
with open(json_path, "r", encoding="utf-8") as f:
    enriched = json.load(f)

# ── STEP 2: Run diarization ───────────────────────────────────
print("Loading audio for diarization...")
audio = whisperx.load_audio(AUDIO_FILE)

print("Diarizing (identifying speakers)...")
try:
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio, num_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, {"segments": enriched})
    enriched = result["segments"]
    print("  Diarization complete.")
except Exception as e:
    print(f"  Diarization failed: {e}")
    print("  Continuing with UNKNOWN speakers.")

# ── STEP 3: Save JSON ─────────────────────────────────────────
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(enriched, f, ensure_ascii=False, indent=2)
print(f"  JSON updated: {json_path}")

# ── STEP 4: Save TXT ──────────────────────────────────────────
txt_path = base + "_transcript.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for seg in enriched:
        flags = []
        if seg.get("low_confidence"): flags.append("LOW-CONF")
        if seg.get("code_switch"):    flags.append("CODE-SWITCH")
        flag_str = f"  ⚠ {', '.join(flags)}" if flags else ""
        f.write(
            f"[{seg['start']:.1f}s → {seg['end']:.1f}s] "
            f"[{seg['speaker']}] "
            f"(conf: {seg['confidence']})"
            f"{flag_str}\n"
            f"{seg['text']}\n\n"
        )
print(f"  TXT updated: {txt_path}")

# ── STEP 5: Save HTML ─────────────────────────────────────────

# Compute summary stats
total     = len(enriched)
low_conf  = sum(1 for s in enriched if s.get("low_confidence"))
code_sw   = sum(1 for s in enriched if s.get("code_switch"))
speakers  = sorted(set(s.get("speaker", "UNKNOWN") for s in enriched))
duration  = max((s["end"] for s in enriched), default=0)

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

# Build speaker legend
speaker_legend = ""
for i, sp in enumerate(speakers):
    speaker_legend += f'<span class="sp-legend sp-{i}">{sp}</span> '

# Build rows
rows = ""
for seg in enriched:
    sp       = seg.get("speaker", "UNKNOWN")
    sp_index = speakers.index(sp) if sp in speakers else 0
    classes  = []
    if seg.get("low_confidence"): classes.append("low-conf")
    if seg.get("code_switch"):    classes.append("code-switch")
    cls = " ".join(classes)

    badges = ""
    if seg.get("low_confidence"): badges += '<span class="badge lc">LOW CONF</span>'
    if seg.get("code_switch"):    badges += '<span class="badge cs">CODE SWITCH</span>'

    rows += f"""
    <tr class="{cls}">
      <td class="time">{fmt_time(seg['start'])} → {fmt_time(seg['end'])}</td>
      <td class="speaker sp-{sp_index}">{sp}</td>
      <td class="conf">{seg['confidence']}</td>
      <td class="dari" dir="rtl">{seg['text']} {badges}</td>
    </tr>"""

html_path = base + "_transcript.html"
html = f"""<!DOCTYPE html>
<html lang="fa">
<head>
  <meta charset="UTF-8">
  <title>Transcript — {os.path.basename(AUDIO_FILE)}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Segoe UI', Tahoma, sans-serif;
      background: #f0f2f5;
      padding: 32px 24px;
      color: #222;
    }}

    .header {{
      max-width: 1100px;
      margin: 0 auto 24px;
    }}

    .header h1 {{
      font-size: 1.4em;
      font-weight: 600;
      color: #1a1a2e;
      margin-bottom: 4px;
    }}

    .header p {{
      font-size: 0.85em;
      color: #666;
      margin-bottom: 16px;
    }}

    /* Stats bar */
    .stats {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}

    .stat {{
      background: white;
      border-radius: 8px;
      padding: 10px 18px;
      font-size: 0.85em;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}

    .stat strong {{
      display: block;
      font-size: 1.3em;
      color: #1a1a2e;
    }}

    /* Legend */
    .legend {{
      font-size: 0.82em;
      margin-bottom: 8px;
      color: #555;
    }}

    .sp-legend {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 12px;
      margin-right: 8px;
      font-weight: 600;
      color: white;
    }}

    /* Table */
    .table-wrap {{
      max-width: 1100px;
      margin: 0 auto;
      overflow-x: auto;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
    }}

    thead th {{
      background: #1a1a2e;
      color: white;
      padding: 12px 16px;
      text-align: left;
      font-size: 0.82em;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}

    td {{
      padding: 10px 16px;
      border-bottom: 1px solid #eee;
      vertical-align: top;
      font-size: 0.92em;
    }}

    tr:last-child td {{ border-bottom: none; }}
    tr:hover {{ background: #fafafa; }}

    .time {{
      white-space: nowrap;
      color: #888;
      font-size: 0.82em;
      font-variant-numeric: tabular-nums;
      min-width: 100px;
    }}

    .speaker {{
      font-weight: 700;
      white-space: nowrap;
      min-width: 110px;
    }}

    .conf {{
      color: #aaa;
      font-size: 0.8em;
      white-space: nowrap;
    }}

    .dari {{
      font-size: 1.05em;
      line-height: 1.7;
      font-family: 'Tahoma', 'Arial', sans-serif;
    }}

    /* Row highlight states */
    .low-conf  {{ background: #fffbea; }}
    .code-switch {{ background: #f0fdf4; }}
    .low-conf.code-switch {{ background: #fff0f3; }}

    /* Speaker colors */
    .sp-0 {{ color: #1565c0; }}
    .sp-1 {{ color: #b71c1c; }}
    .sp-2 {{ color: #1b5e20; }}
    .sp-3 {{ color: #4a148c; }}

    .sp-legend.sp-0 {{ background: #1565c0; }}
    .sp-legend.sp-1 {{ background: #b71c1c; }}
    .sp-legend.sp-2 {{ background: #1b5e20; }}
    .sp-legend.sp-3 {{ background: #4a148c; }}

    /* Badges */
    .badge {{
      display: inline-block;
      font-size: 0.65em;
      padding: 2px 6px;
      border-radius: 4px;
      margin-right: 4px;
      font-weight: 700;
      letter-spacing: 0.03em;
      vertical-align: middle;
    }}

    .lc {{ background: #f59e0b; color: white; }}
    .cs {{ background: #10b981; color: white; }}

    /* Print */
    @media print {{
      body {{ background: white; padding: 0; }}
      .table-wrap {{ box-shadow: none; }}
      .stats {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>📋 {os.path.basename(AUDIO_FILE)}</h1>
    <p>Language: Farsi / Dari &nbsp;|&nbsp; Generated by WhisperX</p>

    <div class="stats">
      <div class="stat"><strong>{total}</strong>Segments</div>
      <div class="stat"><strong>{fmt_time(duration)}</strong>Duration</div>
      <div class="stat"><strong>{len(speakers)}</strong>Speakers</div>
      <div class="stat"><strong>{low_conf}</strong>Low Confidence</div>
      <div class="stat"><strong>{code_sw}</strong>Code Switches</div>
    </div>

    <div class="legend">
      Speakers: {speaker_legend}
      &nbsp;&nbsp;
      <span class="badge lc">LOW CONF</span> confidence &lt; {LOW_CONF}
      &nbsp;
      <span class="badge cs">CODE SWITCH</span> mixed script
    </div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Speaker</th>
          <th>Conf</th>
          <th>Dari / Farsi Text</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</body>
</html>"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  HTML updated: {html_path}")

print(f"\n✅ Done.")
print(f"  JSON: {json_path}")
print(f"  TXT:  {txt_path}")
print(f"  HTML: {html_path}")