from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Model
model = WhisperModel("tiny", device="cpu", compute_type="int8")

def format_timestamp(seconds: float):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

@app.get("/")
def home():
    return {"status": "AI Subtitle API is live!"}

@app.post("/generate-subtitles")
async def generate_subtitles(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments, _ = model.transcribe(temp_path)

    srt_content = ""
    for i, seg in enumerate(segments, start=1):
        s = format_timestamp(seg.start)
        e = format_timestamp(seg.end)
        srt_content += f"{i}\n{s} --> {e}\n{seg.text.strip()}\n\n"

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {"srt_text": srt_content}
