# AutoDJ

AutoDJ is a local full-stack automatic DJ transition tool. Upload two audio files, analyze tempo and key, time-stretch the second track to the first track's BPM, optionally apply harmonic correction, and render a crossfaded transition.

## Stack

- Backend: FastAPI, librosa, NumPy, SoundFile
- Frontend: React, Vite, TypeScript
- Report: LaTeX in `report/`

## Run Locally

Terminal 1:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Audio Notes

The backend writes rendered mixes as WAV files so the project works without a system encoder. For MP3 input or MP3 export, install FFmpeg and use audio files supported by your local `librosa`/`soundfile` stack.

On macOS:

```bash
brew install ffmpeg
```

## API

- `GET /api/health` checks server status.
- `POST /api/mix` accepts multipart fields:
  - `track_a`: first audio file
  - `track_b`: second audio file
  - `crossfade_seconds`: crossfade duration
  - `phrase_bars`: phrase alignment in bars
  - `harmonic_correction`: whether to pitch-shift B toward a compatible Camelot key

The response includes the analysis report and a `download_url` for the rendered WAV.
