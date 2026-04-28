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
} from "lucide-react";
import "./styles.css";

const API_BASE =
  import.meta.env.VITE_API_URL ?? `${window.location.protocol}//${window.location.hostname}:8000`;

type MixResponse = {
  filename: string;
  download_url: string;
  mix: {
    duration_seconds: number;
    cut_seconds: number;
    crossfade_seconds: number;
    tempo_rate: number;
    a_transition_rate: number;
    b_intro_rate: number;
    b_body_start_seconds: number;
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
    transition_key: string;
    transition_camelot: string;
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
  const [phraseBars, setPhraseBars] = useState(16);
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
    formData.append("phrase_bars", String(phraseBars));
    formData.append("harmonic_correction", String(harmonicCorrection));

    try {
      const response = await fetch(`${API_BASE}/api/mix`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
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
        <section className="deckGrid">
          <UploadDeck label="Track A" file={trackA} onChange={setTrackA} accent="a" />
          <UploadDeck label="Track B" file={trackB} onChange={setTrackB} accent="b" />
        </section>

        <section className="controlBand">
          <div className="controlTitle">
            <SlidersHorizontal size={18} />
            <span>Transition</span>
          </div>
          <div className="fixedValue">
            <span>Transition</span>
            <strong>Auto</strong>
          </div>
          <div className="segmented">
            {[8, 16, 24].map((bars) => (
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
          <button className="ghostButton" onClick={submitMix} disabled={!canSubmit}>
            {loading ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
            Render Mix
          </button>
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
                <Metric label="A Transition" value={`${result.mix.a_transition_rate}x`} />
                <Metric label="B Original" value={`${result.mix.b_intro_rate}x`} />
                <Metric label="Transition" value={`${result.mix.crossfade_seconds}s`} />
                <Metric label="A Key" value={`${result.track_a.key} · ${result.track_a.camelot}`} />
                <Metric label="B Key" value={`${result.track_b.key} · ${result.track_b.camelot}`} />
                <Metric
                  label="Transition Key"
                  value={`${result.harmonic.transition_key} · ${result.harmonic.transition_camelot}`}
                />
                <Metric label="B Body Starts" value={`${result.mix.b_body_start_seconds}s`} />
                <Metric label="A Ends" value={`${result.mix.cut_seconds}s`} />
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
