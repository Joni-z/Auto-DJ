from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf


SR = 44_100
MAX_AUDIO_SECONDS = 12 * 60
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

CAMEL0T_MAJOR_BY_PC = {
    11: "1B",
    6: "2B",
    1: "3B",
    8: "4B",
    3: "5B",
    10: "6B",
    5: "7B",
    0: "8B",
    7: "9B",
    2: "10B",
    9: "11B",
    4: "12B",
}

CAMEL0T_MINOR_BY_PC = {
    8: "1A",
    3: "2A",
    10: "3A",
    5: "4A",
    0: "5A",
    7: "6A",
    2: "7A",
    9: "8A",
    4: "9A",
    11: "10A",
    6: "11A",
    1: "12A",
}


@dataclass
class TrackAnalysis:
    bpm: float
    key: str
    tonic_pc: int
    mode: str
    camelot: str
    duration_seconds: float
    beats: list[float]


def process_mix(
    track_a_path: Path,
    track_b_path: Path,
    output_dir: Path,
    crossfade_seconds: float = 8.0,
    phrase_bars: int = 8,
    harmonic_correction: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_a, sr = load_audio(track_a_path)
    y_b, _ = load_audio(track_b_path)

    analysis_a = analyze_track(y_a, sr)
    analysis_b = analyze_track(y_b, sr)

    tempo_rate = clamp(analysis_a.bpm / max(analysis_b.bpm, 1e-6), 0.65, 1.55)
    suggested_shift = suggest_harmonic_shift(analysis_a, analysis_b)
    applied_shift = suggested_shift if harmonic_correction else 0

    y_b_processed = time_stretch_multichannel(y_b, tempo_rate)
    if applied_shift:
        y_b_processed = pitch_shift_multichannel(y_b_processed, sr, applied_shift)

    shifted_b = analyze_shifted_key(analysis_b, applied_shift)
    cut_seconds = choose_cut_point(
        analysis_a=analysis_a,
        duration_a=samples_to_seconds(y_a.shape[-1], sr),
        crossfade_seconds=crossfade_seconds,
        phrase_bars=phrase_bars,
    )

    mixed = render_crossfade(
        y_a=y_a,
        y_b=y_b_processed,
        sr=sr,
        cut_seconds=cut_seconds,
        crossfade_seconds=crossfade_seconds,
    )
    mixed = normalize_peak(mixed, peak=0.96)

    output_id = uuid4().hex
    output_path = output_dir / f"autodj_mix_{output_id}.wav"
    sf.write(output_path, mixed.T, sr, subtype="PCM_16")

    compatibility_before = camelot_compatible(analysis_a.camelot, analysis_b.camelot)
    compatibility_after = camelot_compatible(analysis_a.camelot, shifted_b["camelot"])

    return {
        "id": output_id,
        "filename": output_path.name,
        "download_url": f"/api/download/{output_path.name}",
        "mix": {
            "duration_seconds": round(samples_to_seconds(mixed.shape[-1], sr), 2),
            "cut_seconds": round(cut_seconds, 2),
            "crossfade_seconds": round(crossfade_seconds, 2),
            "tempo_rate": round(float(tempo_rate), 4),
            "pitch_shift_semitones": int(applied_shift),
        },
        "track_a": analysis_to_dict(analysis_a),
        "track_b": analysis_to_dict(analysis_b),
        "track_b_after_processing": {
            "estimated_key": shifted_b["key"],
            "estimated_camelot": shifted_b["camelot"],
            "estimated_bpm": round(float(analysis_b.bpm * tempo_rate), 2),
        },
        "harmonic": {
            "compatible_before": compatibility_before,
            "compatible_after": compatibility_after,
            "suggested_pitch_shift_semitones": int(suggested_shift),
        },
    }


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SR, mono=False, duration=MAX_AUDIO_SECONDS)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    if y.shape[0] == 1:
        y = np.vstack([y[0], y[0]])
    return normalize_peak(y), sr


def analyze_track(y: np.ndarray, sr: int) -> TrackAnalysis:
    mono = to_mono(y)
    duration = samples_to_seconds(mono.shape[-1], sr)
    bpm, beats = estimate_bpm_and_beats(mono, sr)
    tonic_pc, mode = estimate_key(mono, sr)
    key = f"{PITCH_CLASS_NAMES[tonic_pc]} {mode}"
    camelot = camelot_code(tonic_pc, mode)
    return TrackAnalysis(
        bpm=bpm,
        key=key,
        tonic_pc=tonic_pc,
        mode=mode,
        camelot=camelot,
        duration_seconds=duration,
        beats=beats,
    )


def estimate_bpm_and_beats(y: np.ndarray, sr: int) -> tuple[float, list[float]]:
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", trim=False)
        bpm = float(np.asarray(tempo).reshape(-1)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).astype(float).tolist()
        if not np.isfinite(bpm) or bpm <= 0:
            raise ValueError("invalid tempo")
        return round(bpm, 2), beat_times
    except Exception:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])
        return round(bpm if np.isfinite(bpm) and bpm > 0 else 120.0, 2), []


def estimate_key(y: np.ndarray, sr: int) -> tuple[int, str]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    if np.max(chroma_mean) > 0:
        chroma_mean = chroma_mean / np.linalg.norm(chroma_mean)

    scores: list[tuple[float, int, str]] = []
    for tonic in range(12):
        major = np.roll(MAJOR_PROFILE, tonic)
        minor = np.roll(MINOR_PROFILE, tonic)
        major = major / np.linalg.norm(major)
        minor = minor / np.linalg.norm(minor)
        scores.append((float(np.dot(chroma_mean, major)), tonic, "major"))
        scores.append((float(np.dot(chroma_mean, minor)), tonic, "minor"))

    _, tonic_pc, mode = max(scores, key=lambda item: item[0])
    return tonic_pc, mode


def camelot_code(tonic_pc: int, mode: str) -> str:
    mapping = CAMEL0T_MAJOR_BY_PC if mode == "major" else CAMEL0T_MINOR_BY_PC
    return mapping[tonic_pc % 12]


def camelot_compatible(code_a: str, code_b: str) -> bool:
    num_a, mode_a = parse_camelot(code_a)
    num_b, mode_b = parse_camelot(code_b)
    if num_a == num_b:
        return True
    if mode_a == mode_b and circular_distance(num_a, num_b, 12) == 1:
        return True
    return False


def suggest_harmonic_shift(track_a: TrackAnalysis, track_b: TrackAnalysis) -> int:
    candidates: list[tuple[int, int]] = []
    for shift in range(-6, 7):
        shifted_pc = (track_b.tonic_pc + shift) % 12
        shifted_code = camelot_code(shifted_pc, track_b.mode)
        if camelot_compatible(track_a.camelot, shifted_code):
            candidates.append((abs(shift), shift))
    if not candidates:
        return 0
    _, shift = min(candidates, key=lambda item: (item[0], abs(item[1] - 0.1)))
    return int(shift)


def analyze_shifted_key(track: TrackAnalysis, shift: int) -> dict[str, str]:
    tonic = (track.tonic_pc + shift) % 12
    mode = track.mode
    return {
        "key": f"{PITCH_CLASS_NAMES[tonic]} {mode}",
        "camelot": camelot_code(tonic, mode),
    }


def choose_cut_point(
    analysis_a: TrackAnalysis,
    duration_a: float,
    crossfade_seconds: float,
    phrase_bars: int,
) -> float:
    safe_end = max(duration_a - 1.0, crossfade_seconds + 1.0)
    target = min(duration_a * 0.72, safe_end)
    phrase_beats = max(4, int(phrase_bars) * 4)

    beat_times = [t for t in analysis_a.beats if crossfade_seconds + 1.0 <= t <= safe_end]
    phrase_times = [t for idx, t in enumerate(beat_times) if idx > 0 and idx % phrase_beats == 0]
    candidates = phrase_times or beat_times
    if candidates:
        return min(candidates, key=lambda t: abs(t - target))

    beat_duration = 60.0 / max(analysis_a.bpm, 1.0)
    phrase_duration = beat_duration * phrase_beats
    if phrase_duration <= 0:
        return target
    phrase_index = max(1, round(target / phrase_duration))
    return min(max(phrase_index * phrase_duration, crossfade_seconds + 1.0), safe_end)


def render_crossfade(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
    cut_seconds: float,
    crossfade_seconds: float,
) -> np.ndarray:
    crossfade_samples = max(1, int(crossfade_seconds * sr))
    cut_sample = int(cut_seconds * sr)
    start_b = max(0, cut_sample - crossfade_samples)
    a_end = min(y_a.shape[-1], start_b + crossfade_samples)
    actual_fade = max(1, a_end - start_b)

    output_len = max(a_end, start_b + y_b.shape[-1])
    out = np.zeros((2, output_len), dtype=np.float32)

    out[:, :a_end] += y_a[:, :a_end]

    b_copy_len = min(y_b.shape[-1], output_len - start_b)
    out[:, start_b : start_b + b_copy_len] += y_b[:, :b_copy_len]

    fade = np.linspace(0.0, 1.0, actual_fade, dtype=np.float32)
    fade_in = np.sin(fade * np.pi / 2.0)
    fade_out = np.cos(fade * np.pi / 2.0)

    a_segment = y_a[:, start_b : start_b + actual_fade] * fade_out
    b_segment = y_b[:, :actual_fade] * fade_in
    out[:, start_b : start_b + actual_fade] = a_segment + b_segment
    return out


def time_stretch_multichannel(y: np.ndarray, rate: float) -> np.ndarray:
    channels = [librosa.effects.time_stretch(channel, rate=rate) for channel in y]
    return stack_channels(channels)


def pitch_shift_multichannel(y: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    channels = [librosa.effects.pitch_shift(channel, sr=sr, n_steps=semitones) for channel in y]
    return stack_channels(channels)


def stack_channels(channels: list[np.ndarray]) -> np.ndarray:
    min_len = min(channel.shape[-1] for channel in channels)
    return np.vstack([channel[:min_len] for channel in channels]).astype(np.float32)


def to_mono(y: np.ndarray) -> np.ndarray:
    return np.mean(y, axis=0).astype(np.float32)


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs <= 1e-9:
        return y.astype(np.float32)
    if max_abs <= peak:
        return y.astype(np.float32)
    return (y / max_abs * peak).astype(np.float32)


def samples_to_seconds(samples: int, sr: int) -> float:
    return float(samples) / float(sr)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def circular_distance(a: int, b: int, modulo: int) -> int:
    diff = abs(a - b) % modulo
    return min(diff, modulo - diff)


def parse_camelot(code: str) -> tuple[int, str]:
    return int(code[:-1]), code[-1]


def analysis_to_dict(analysis: TrackAnalysis) -> dict[str, Any]:
    return {
        "bpm": round(float(analysis.bpm), 2),
        "key": analysis.key,
        "camelot": analysis.camelot,
        "duration_seconds": round(float(analysis.duration_seconds), 2),
    }
