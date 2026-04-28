# AutoDJ

AutoDJ is a local full-stack automatic DJ transition tool. Upload two audio files, analyze tempo and key, locally morph only the transition region, and render a DJ-style handoff while preserving the body of the incoming track.

## Stack

- Backend: FastAPI, librosa, NumPy, SoundFile
- Frontend: React, Vite, TypeScript
- Report: LaTeX in `report/`

## Run Locally

One-command dev server:

```bash
./run_dev.sh
```

The script starts both backend and frontend, then prints local and LAN URLs.

Terminal 1:

```bash
cd backend
conda activate ml
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
  - `crossfade_seconds`: optional; omitted by the frontend because the backend estimates transition duration from BPM and phrase length
  - `phrase_bars`: phrase alignment in bars, one of 8, 16, or 24
  - `harmonic_correction`: whether to morph Track A's transition segment toward a compatible Camelot key

The response includes the analysis report and a `download_url` for the rendered WAV.
