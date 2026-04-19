import re
import whisperx
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()
#get the file from script
#Configuration
Audio_file = sys.argv[1]
LANG = "fa"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cpu"
COMPUTE = "int8"
NUM_CHUNKS = 4
LOW_CONF = 0.6


#Functions

def detect_codeswitching(text):
    """Flag if segment mixes Persian script with Latin characters."""
    has_persian = bool(re.search(r'[\u0600-\u06FF]', text))
    has_latin   = bool(re.search(r'[a-zA-Z]{2,}', text))  
    return has_persian and has_latin



print("Loading model...")

model = whisperx.load_model("large-v2", "cpu", compute_type="int8",language=LANG)

print(f"Transcribing {Audio_file}...")
audio = whisperx.load_audio(Audio_file)
result = model.transcribe(audio, language=LANG)

txt_path = os.path.splitext(Audio_file)[0] + "trascribed.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    for seg in result["segments"]:
        start = seg["start"]
        end   = seg["end"]
        text  = seg["text"].strip()
        f.write(f"[{start:.1f}s --> {end:.1f}s]  {text}\n")
json_path = os.path.splitext(Audio_file)[0]  + "_transcript.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Transcription completed.")
print(f"  Text: {txt_path}")
print(f"  JSON: {json_path}")