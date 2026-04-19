import re
import whisperx
import json
import sys
import os
import nltk
from dotenv import load_dotenv

# Initialize
load_dotenv()
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ─── CONFIGURATION ──────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_audio>")
    sys.exit(1)

AUDIO_FILE = sys.argv[1]
LANG = "fa"  # Works for Farsi and Dari
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cpu"
COMPUTE = "int8"  # Optimized for CPU
BATCH_SIZE = 4    # Adjust based on your RAM (4-8 is safe for CPU)
LOW_CONF = 0.6
MODEL_NAME = "large-v3-turbo" # Faster and often better for Farsi script

# ─── HELPER FUNCTIONS ───────────────────────────────────────

def detect_codeswitching(text):
    """Flag if segment mixes Persian script with Latin characters."""
    has_persian = bool(re.search(r'[\u0600-\u06FF]', text))
    has_latin   = bool(re.search(r'[a-zA-Z]{2,}', text))
    return has_persian and has_latin

def clean_words(words):
    """Convert numpy types to native Python types for JSON compatibility."""
    return [
        {k: (float(v) if hasattr(v, 'item') else v) for k, v in w.items()}
        for w in words
    ]

def avg_word_confidence(segment):
    """Calculate mean confidence across words in a segment."""
    words = segment.get("words", [])
    # Default to 0.0 if score is missing from alignment
    scores = [w.get("score", 0.0) for w in words]
    return round(sum(scores) / len(scores), 3) if scores else None

# ─── EXECUTION PIPELINE ─────────────────────────────────────

# 1. Load Model
print(f"--- Loading {MODEL_NAME} on {DEVICE} ---")
model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE, language=LANG)

# 2. Load Audio
print(f"--- Loading Audio: {os.path.basename(AUDIO_FILE)} ---")
audio = whisperx.load_audio(AUDIO_FILE)

# 3. Transcribe
# WhisperX uses VAD to split audio into segments automatically
print("--- Transcribing (Native Batching) ---")
result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANG)

# 4. Align
print("--- Aligning Timestamps ---")
try:
    model_a, metadata = whisperx.load_align_model(language_code=LANG, device=DEVICE)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE,
        return_char_alignments=False
    )
except Exception as e:
    print(f"Alignment Error: {e}")

# 5. Diarize
print("--- Identifying Speakers ---")
try:
    # Use min/max speakers to be more flexible than a hard 'num_speakers=2'
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=3)
    result = whisperx.assign_word_speakers(diarize_segments, result)
except Exception as e:
    print(f"Diarization Error: {e} (Ensure your HF_TOKEN is valid)")

# 6. Post-Process
enriched = []
for seg in result["segments"]:
    speaker    = seg.get("speaker", "UNKNOWN")
    # Fallback to avg_logprob if word scores aren't available
    confidence = avg_word_confidence(seg) or round(seg.get("avg_logprob", 0), 3)
    text       = seg["text"].strip()

    enriched.append({
        "start":          round(seg["start"], 2),
        "end":            round(seg["end"],   2),
        "speaker":        speaker,
        "text":           text,
        "confidence":     confidence,
        "low_confidence": bool(confidence < LOW_CONF),
        "code_switch":    bool(detect_codeswitching(text)),
        "words":          clean_words(seg.get("words", []))
    })

# ─── EXPORTING ──────────────────────────────────────────────
base = os.path.splitext(AUDIO_FILE)[0]

# Save TXT
with open(base + "_transcript.txt", "w", encoding="utf-8") as f:
    for seg in enriched:
        flags = [f for f, val in [("LOW-CONF", seg["low_confidence"]), ("CODE-SWITCH", seg["code_switch"])] if val]
        flag_str = f"  ⚠ {', '.join(flags)}" if flags else ""
        f.write(f"[{seg['start']}s - {seg['end']}s] [{seg['speaker']}] (conf: {seg['confidence']}){flag_str}\n{seg['text']}\n\n")

# Save JSON
with open(base + "_transcript.json", "w", encoding="utf-8") as f:
    json.dump(enriched, f, ensure_ascii=False, indent=2)

# Save HTML (with RTL support)
rows = ""
for seg in enriched:
    cls = f"{'low-conf' if seg['low_confidence'] else ''} {'code-switch' if seg['code_switch'] else ''}"
    badges = "".join([f'<span class="badge {b[:2].lower()}">{b}</span>' for b, val in [("LOW CONF", seg["low_confidence"]), ("CODE SWITCH", seg["code_switch"])] if val])
    
    rows += f"""
    <tr class="{cls}">
        <td class="time">{seg['start']}s → {seg['end']}s</td>
        <td class="speaker {seg['speaker'].lower().replace('_','-')}">{seg['speaker']}</td>
        <td class="conf">{seg['confidence']}</td>
        <td class="text" dir="rtl">{seg['text']} {badges}</td>
    </tr>"""

html_content = f"""<!DOCTYPE html>
<html lang="fa">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Tahoma', sans-serif; background: #f4f4f9; padding: 40px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: left; }}
        th {{ background: #2c3e50; color: white; }}
        .text {{ font-size: 1.1em; line-height: 1.6; }}
        .low-conf {{ background: #fff9c4; }}
        .code-switch {{ background: #e8f5e9; }}
        .badge {{ font-size: 0.7em; padding: 2px 5px; border-radius: 3px; margin-left: 5px; color: white; }}
        .lo {{ background: #fbc02d; }} .co {{ background: #43a047; }}
        .speaker-0 {{ color: #1976d2; font-weight: bold; }}
        .speaker-1 {{ color: #d32f2f; font-weight: bold; }}
    </style>
</head>
<body>
    <h2>Transcript for {os.path.basename(AUDIO_FILE)}</h2>
    <table>
        <thead><tr><th>Time</th><th>Speaker</th><th>Conf</th><th>Text (RTL)</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
</body>
</html>"""

with open(base + "_transcript.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n✅ Processing complete. Files saved to: {base}_transcript.*")