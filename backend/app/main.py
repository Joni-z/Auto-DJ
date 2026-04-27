from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .audio_engine import process_mix


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AutoDJ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/mix")
async def mix(
    track_a: UploadFile = File(...),
    track_b: UploadFile = File(...),
    crossfade_seconds: float = Form(8.0),
    phrase_bars: int = Form(8),
    harmonic_correction: bool = Form(True),
) -> dict:
    if not track_a.filename or not track_b.filename:
        raise HTTPException(status_code=400, detail="Two audio files are required.")
    if not 2 <= crossfade_seconds <= 30:
        raise HTTPException(status_code=400, detail="crossfade_seconds must be between 2 and 30.")
    if phrase_bars not in {4, 8, 16}:
        raise HTTPException(status_code=400, detail="phrase_bars must be 4, 8, or 16.")

    path_a = await save_upload(track_a, "a")
    path_b = await save_upload(track_b, "b")

    try:
        return process_mix(
            track_a_path=path_a,
            track_b_path=path_b,
            output_dir=OUTPUT_DIR,
            crossfade_seconds=crossfade_seconds,
            phrase_bars=phrase_bars,
            harmonic_correction=harmonic_correction,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc


@app.get("/api/download/{filename}")
def download(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="audio/wav", filename=filename)


async def save_upload(upload: UploadFile, prefix: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or ".audio"
    path = UPLOAD_DIR / f"{prefix}_{Path(upload.filename or 'track').stem}_{id(upload)}{suffix}"
    with path.open("wb") as file:
        shutil.copyfileobj(upload.file, file)
    return path
