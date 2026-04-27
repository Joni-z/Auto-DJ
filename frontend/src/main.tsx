import React, { ChangeEvent, DragEvent, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Download,
  FileAudio,
  Loader2,
  Music2,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Waves,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

type MixResponse = {
  filename: string;
  download_url: string;
  mix: {
    duration_seconds: number;
    cut_seconds: number;
    crossfade_seconds: number;
    tempo_rate: number;
    pitch_shift_semitones: number;
  };
  track_a: TrackSummary;
  track_b: TrackSummary;
  track_b_after_processing: {
    estimated_key: string;
    estimated_camelot: string;
    estimated_bpm: number;
  };
  harmonic: {
    compatible_before: boolean;
    compatible_after: boolean;
    suggested_pitch_shift_semitones: number;
  };
};

type TrackSummary = {
  bpm: number;
  key: string;
  camelot: string;
  duration_seconds: number;
};

function App() {
  const [trackA, setTrackA] = useState<File | null>(null);
  const [trackB, setTrackB] = useState<File | null>(null);
  const [crossfade, setCrossfade] = useState(8);
  const [phraseBars, setPhraseBars] = useState(8);
  const [harmonicCorrection, setHarmonicCorrection] = useState(true);
  const [result, setResult] = useState<MixResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const canSubmit = useMemo(() => Boolean(trackA && trackB && !loading), [trackA, trackB, loading]);
  const audioUrl = result ? `${API_BASE}${result.download_url}` : "";

  async function submitMix() {
    if (!trackA || !trackB) return;
    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("track_a", trackA);
    formData.append("track_b", trackB);
    formData.append("crossfade_seconds", String(crossfade));
    formData.append("phrase_bars", String(phraseBars));
    formData.append("harmonic_correction", String(harmonicCorrection));

    try {
      const response = await fetch(`${API_BASE}/api/mix`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Mix failed.");
      }
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mix failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="workspace">
        <header className="topbar">
          <div className="brand">
            <span className="brandMark">
              <Waves size={18} />
            </span>
            <div>
              <h1>AutoDJ</h1>
              <p>MIR transition engine</p>
            </div>
          </div>
          <button className="ghostButton" onClick={submitMix} disabled={!canSubmit}>
            {loading ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
            Render Mix
          </button>
        </header>

        <section className="deckGrid">
          <UploadDeck label="Track A" file={trackA} onChange={setTrackA} accent="a" />
          <UploadDeck label="Track B" file={trackB} onChange={setTrackB} accent="b" />
        </section>

        <section className="controlBand">
          <div className="controlTitle">
            <SlidersHorizontal size={18} />
            <span>Transition</span>
          </div>
          <label className="sliderControl">
            <span>Crossfade</span>
            <input
              type="range"
              min="2"
              max="30"
              step="1"
              value={crossfade}
              onChange={(event) => setCrossfade(Number(event.target.value))}
            />
            <strong>{crossfade}s</strong>
          </label>
          <div className="segmented">
            {[4, 8, 16].map((bars) => (
              <button
                key={bars}
                className={phraseBars === bars ? "active" : ""}
                onClick={() => setPhraseBars(bars)}
                type="button"
              >
                {bars} bars
              </button>
            ))}
          </div>
          <label className="switch">
            <input
              type="checkbox"
              checked={harmonicCorrection}
              onChange={(event) => setHarmonicCorrection(event.target.checked)}
            />
            <span />
            Harmonic shift
          </label>
        </section>

        {error && <div className="error">{error}</div>}

        <section className="resultArea">
          {result ? (
            <>
              <div className="playerPanel">
                <div className="resultHeader">
                  <div>
                    <span className="eyebrow">Rendered</span>
                    <h2>{result.filename}</h2>
                  </div>
                  <a className="downloadButton" href={audioUrl}>
                    <Download size={17} />
                    WAV
                  </a>
                </div>
                <audio controls src={audioUrl} />
              </div>

              <div className="analysisGrid">
                <Metric label="A BPM" value={result.track_a.bpm} />
                <Metric label="B BPM" value={result.track_b.bpm} />
                <Metric label="Rate" value={`${result.mix.tempo_rate}x`} />
                <Metric label="Pitch" value={`${result.mix.pitch_shift_semitones} st`} />
                <Metric label="A Key" value={`${result.track_a.key} · ${result.track_a.camelot}`} />
                <Metric label="B Key" value={`${result.track_b.key} · ${result.track_b.camelot}`} />
                <Metric
                  label="B Rendered"
                  value={`${result.track_b_after_processing.estimated_key} · ${result.track_b_after_processing.estimated_camelot}`}
                />
                <Metric label="Cut" value={`${result.mix.cut_seconds}s`} />
              </div>
            </>
          ) : (
            <div className="emptyState">
              <Music2 size={30} />
              <span>{loading ? "Analyzing tempo, key, and phrase alignment" : "Load two tracks to render a transition"}</span>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function UploadDeck({
  label,
  file,
  onChange,
  accent,
}: {
  label: string;
  file: File | null;
  onChange: (file: File | null) => void;
  accent: "a" | "b";
}) {
  function updateFile(event: ChangeEvent<HTMLInputElement>) {
    onChange(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    onChange(event.dataTransfer.files?.[0] ?? null);
  }

  return (
    <label
      className={`deck deck-${accent}`}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
      <input type="file" accept="audio/*" onChange={updateFile} />
      <div className="deckHead">
        <span>{label}</span>
        <Upload size={18} />
      </div>
      <div className="deckBody">
        <FileAudio size={30} />
        <strong>{file ? file.name : "Drop audio"}</strong>
        <small>{file ? formatBytes(file.size) : "WAV, AIFF, FLAC, MP3 with FFmpeg"}</small>
      </div>
      <div className="fakeWave">
        {Array.from({ length: 42 }).map((_, index) => (
          <i key={index} style={{ height: `${18 + ((index * 17 + accent.charCodeAt(0)) % 54)}%` }} />
        ))}
      </div>
    </label>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatBytes(bytes: number) {
  const mb = bytes / 1024 / 1024;
  return `${mb.toFixed(2)} MB`;
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
