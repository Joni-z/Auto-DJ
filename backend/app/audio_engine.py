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
MIN_LATE_TRANSITION_RATIO = 0.72
DEFAULT_END_GUARD_SECONDS = 0.75
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
    crossfade_seconds: float | None = None,
    phrase_bars: int = 16,
    harmonic_correction: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_a, sr = load_audio(track_a_path)
    y_b, _ = load_audio(track_b_path)

    analysis_a = analyze_track(y_a, sr)
    analysis_b = analyze_track(y_b, sr)
    duration_a = samples_to_seconds(y_a.shape[-1], sr)
    transition_seconds = crossfade_seconds or choose_transition_duration(analysis_a.bpm, analysis_b.bpm, phrase_bars)
    transition_seconds = constrain_transition_duration_for_late_entry(
        transition_seconds=transition_seconds,
        duration_a=duration_a,
        bpm_a=analysis_a.bpm,
        bpm_b=analysis_b.bpm,
    )

    suggested_shift = suggest_harmonic_shift(analysis_b, analysis_a)
    applied_shift = suggested_shift if harmonic_correction else 0

    transition_start_seconds = choose_transition_start_point(
        analysis_a=analysis_a,
        duration_a=duration_a,
        transition_seconds=transition_seconds,
        phrase_bars=phrase_bars,
        bpm_b=analysis_b.bpm,
    )

    mixed, transition_meta = render_automix_transition(
        y_a=y_a,
        y_b=y_b,
        sr=sr,
        bpm_a=analysis_a.bpm,
        bpm_b=analysis_b.bpm,
        beats_b=analysis_b.beats,
        transition_start_seconds=transition_start_seconds,
        transition_seconds=transition_seconds,
        key_shift_a=applied_shift,
    )
    mixed = normalize_peak(mixed, peak=0.96)

    output_id = uuid4().hex
    output_path = output_dir / f"autodj_mix_{output_id}.wav"
    sf.write(output_path, mixed.T, sr, subtype="PCM_16")

    compatibility_before = camelot_compatible(analysis_a.camelot, analysis_b.camelot)
    shifted_a = analyze_shifted_key(analysis_a, applied_shift)
    compatibility_after = camelot_compatible(shifted_a["camelot"], analysis_b.camelot)

    return {
        "id": output_id,
        "filename": output_path.name,
        "download_url": f"/api/download/{output_path.name}",
        "mix": {
            "duration_seconds": round(samples_to_seconds(mixed.shape[-1], sr), 2),
            "cut_seconds": round(float(transition_meta["a_source_end_seconds"]), 2),
            "transition_start_seconds": round(float(transition_meta["transition_start_seconds"]), 2),
            "crossfade_seconds": round(transition_seconds, 2),
            "tempo_rate": round(float(transition_meta["b_intro_rate"]), 4),
            "a_transition_rate": round(float(transition_meta["a_transition_rate"]), 4),
            "b_intro_rate": round(float(transition_meta["b_intro_rate"]), 4),
            "b_body_start_seconds": round(float(transition_meta["b_body_start_seconds"]), 2),
            "pitch_shift_semitones": int(applied_shift),
        },
        "track_a": analysis_to_dict(analysis_a),
        "track_b": analysis_to_dict(analysis_b),
        "track_b_after_processing": {
            "estimated_key": analysis_b.key,
            "estimated_camelot": analysis_b.camelot,
            "estimated_bpm": round(float(analysis_b.bpm), 2),
        },
        "harmonic": {
            "compatible_before": compatibility_before,
            "compatible_after": compatibility_after,
            "suggested_pitch_shift_semitones": int(suggested_shift),
            "transition_key": shifted_a["key"],
            "transition_camelot": shifted_a["camelot"],
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


def choose_transition_start_point(
    analysis_a: TrackAnalysis,
    duration_a: float,
    transition_seconds: float,
    phrase_bars: int,
    bpm_b: float,
) -> float:
    end_guard = min(DEFAULT_END_GUARD_SECONDS, max(0.0, duration_a * 0.05))
    safe_end = max(0.0, duration_a - end_guard)
    phrase_beats = max(4, int(phrase_bars) * 4)
    beat_duration = 60.0 / max(analysis_a.bpm, 1.0)
    estimated_source_seconds = estimate_a_transition_source_seconds(
        transition_seconds=transition_seconds,
        bpm_a=analysis_a.bpm,
        bpm_b=bpm_b,
    )
    latest_start = max(0.0, safe_end - estimated_source_seconds)
    late_floor = max(0.0, duration_a * MIN_LATE_TRANSITION_RATIO)
    target = max(late_floor, latest_start)
    target = min(target, latest_start if latest_start > 0 else safe_end)

    beat_times = [t for t in analysis_a.beats if late_floor <= t <= latest_start]
    phrase_times = [
        t
        for idx, t in enumerate(analysis_a.beats)
        if idx > 0 and idx % phrase_beats == 0 and late_floor <= t <= latest_start
    ]
    phrase_duration = beat_duration * phrase_beats
    phrase_is_close = bool(phrase_times) and (target - phrase_times[-1]) <= min(phrase_duration * 0.5, 12.0)
    candidates = phrase_times if phrase_is_close else beat_times
    if candidates:
        return min(candidates, key=lambda t: abs(t - target))

    if phrase_duration <= 0:
        return target
    beat_index = max(1, round(target / beat_duration))
    return min(max(beat_index * beat_duration, late_floor), latest_start if latest_start > 0 else safe_end)


def choose_transition_duration(bpm_a: float, bpm_b: float, phrase_bars: int) -> float:
    reference_bpm = float(np.nanmedian([max(bpm_a, 1.0), max(bpm_b, 1.0)]))
    phrase_seconds = max(1.0, phrase_bars * 4.0 * 60.0 / reference_bpm)
    tempo_gap = abs(np.log2(max(bpm_b, 1.0) / max(bpm_a, 1.0)))
    complexity_bonus = clamp(tempo_gap * 10.0, 0.0, 4.0)
    target = phrase_seconds * 0.55 + complexity_bonus
    return round(clamp(target, 12.0, 32.0), 2)


def constrain_transition_duration_for_late_entry(
    transition_seconds: float,
    duration_a: float,
    bpm_a: float,
    bpm_b: float,
) -> float:
    if duration_a <= 0:
        return transition_seconds

    end_guard = min(DEFAULT_END_GUARD_SECONDS, max(0.0, duration_a * 0.05))
    late_tail_seconds = max(1.5, duration_a * (1.0 - MIN_LATE_TRANSITION_RATIO) - end_guard)
    rate = estimate_a_transition_source_rate(transition_seconds, bpm_a, bpm_b)
    max_transition_seconds = late_tail_seconds / max(rate, 0.1)
    if transition_seconds <= max_transition_seconds:
        return round(max(1.5, transition_seconds), 2)
    return round(max(1.5, max_transition_seconds), 2)


def estimate_a_transition_source_seconds(transition_seconds: float, bpm_a: float, bpm_b: float) -> float:
    return transition_seconds * estimate_a_transition_source_rate(transition_seconds, bpm_a, bpm_b)


def estimate_a_transition_source_rate(transition_seconds: float, bpm_a: float, bpm_b: float) -> float:
    a_transition_rate = clamp(max(bpm_b, 1.0) / max(bpm_a, 1.0), 0.82, 1.18)
    align_seconds = min(transition_seconds * 0.18, 2.0 * 60.0 / max(bpm_a, 1.0))
    align_samples = max(1, int(align_seconds * SR))
    transition_samples = max(1, int(transition_seconds * SR))
    blend_samples = max(1, transition_samples - align_samples)
    return weighted_average_rate(1.0, a_transition_rate, align_samples, blend_samples)


def render_automix_transition(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
    bpm_a: float,
    bpm_b: float,
    beats_b: list[float],
    transition_start_seconds: float,
    transition_seconds: float,
    key_shift_a: int,
) -> tuple[np.ndarray, dict[str, float]]:
    transition_samples = max(1, int(transition_seconds * sr))

    a_transition_rate = clamp(max(bpm_b, 1.0) / max(bpm_a, 1.0), 0.82, 1.18)
    b_intro_rate = 1.0

    align_seconds = min(transition_seconds * 0.18, 2.0 * 60.0 / max(bpm_a, 1.0))
    align_samples = max(1, int(align_seconds * sr))
    blend_samples = max(1, transition_samples - align_samples)
    average_a_rate = weighted_average_rate(1.0, a_transition_rate, align_samples, blend_samples)
    a_source_len = min(y_a.shape[-1], max(1, int(transition_samples * average_a_rate)))
    transition_start = min(max(0, int(transition_start_seconds * sr)), y_a.shape[-1] - 1)
    transition_end = min(max(transition_start + a_source_len, transition_start + 1), y_a.shape[-1])
    b_entry_offset = choose_b_entry_offset(beats_b, sr)
    b_source_len = min(y_b.shape[-1] - b_entry_offset, transition_samples)

    a_prefix = y_a[:, :transition_start]
    a_source = y_a[:, transition_start:transition_end]
    b_source = y_b[:, b_entry_offset : b_entry_offset + b_source_len]

    a_transition = fast_align_time_stretch_multichannel(
        y=a_source,
        start_rate=1.0,
        end_rate=a_transition_rate,
        target_samples=transition_samples,
        align_samples=align_samples,
    )
    b_intro = fit_length(b_source, transition_samples)

    if key_shift_a:
        a_transition = key_morph_transition(a_transition, sr, key_shift_a)

    b_intro = bass_first_intro(b_intro, sr)

    position = np.arange(transition_samples, dtype=np.float32)
    entry_delay = int(align_samples * 0.45)
    early_b = smoothstep(np.clip((position - entry_delay) / max(1, align_samples - entry_delay), 0.0, 1.0))
    main_b = smoothstep(np.clip((position - int(align_samples * 1.18)) / blend_samples, 0.0, 1.0))
    b_fade = np.clip(0.28 * early_b + 0.82 * main_b, 0.0, 1.0)
    fade_in = 0.04 + 0.98 * np.sin(b_fade * np.pi / 2.0)
    a_duck = np.power(np.clip((position - align_samples) / blend_samples, 0.0, 1.0), 0.58).astype(np.float32)
    fade_out = 1.0 - 0.9 * a_duck
    transition = (a_transition * fade_out) + (b_intro * fade_in)

    mixed = np.concatenate([a_prefix, transition], axis=1)
    body_start = b_entry_offset + b_source_len
    mixed = append_continuous_b_body(mixed, y_b, body_start, sr, fade_seconds=0.9)

    return mixed, {
        "a_transition_rate": float(a_transition_rate),
        "b_intro_rate": float(b_intro_rate),
        "transition_start_seconds": samples_to_seconds(transition_start, sr),
        "a_source_end_seconds": samples_to_seconds(transition_end, sr),
        "b_body_start_seconds": samples_to_seconds(body_start, sr),
    }


def key_morph_transition(y: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    shifted = pitch_shift_multichannel(y, sr, semitones)
    shifted = fit_length(shifted, y.shape[-1])
    fade = smoothstep(np.linspace(0.0, 1.0, y.shape[-1], dtype=np.float32))
    return (y * (1.0 - fade)) + (shifted * fade)


def bass_first_intro(y: np.ndarray, sr: int) -> np.ndarray:
    low_channels: list[np.ndarray] = []
    high_channels: list[np.ndarray] = []
    for channel in y:
        low = low_band(channel, sr, cutoff_hz=180.0)
        low_channels.append(low)
        high_channels.append(channel[: low.shape[-1]] - low)

    low = stack_channels(low_channels)
    high = stack_channels(high_channels)
    length = min(low.shape[-1], high.shape[-1])
    low = low[:, :length]
    high = high[:, :length]

    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    low_gain = 0.9 + 0.2 * smoothstep(np.clip(x / 0.3, 0.0, 1.0))
    high_gain = 0.18 + 0.82 * smoothstep(np.clip((x - 0.08) / 0.72, 0.0, 1.0))
    shaped = (low * low_gain) + (high * high_gain)
    return morph_tail_to_original(shaped, y[:, :length], fade_samples=min(length, int(1.4 * sr)))


def append_continuous_b_body(
    head: np.ndarray,
    original_b: np.ndarray,
    body_start: int,
    sr: int,
    fade_seconds: float,
) -> np.ndarray:
    if body_start >= original_b.shape[-1]:
        return head

    fade_samples = min(int(fade_seconds * sr), head.shape[-1], body_start, original_b.shape[-1] - body_start)
    if fade_samples <= 0:
        return np.concatenate([head, original_b[:, body_start:]], axis=1)

    # Reintroduce the same samples from Track B that the transition already used, then
    # continue into the untouched body. This avoids a tiny discontinuity at the join.
    tail_start = body_start - fade_samples
    tail = original_b[:, tail_start:]
    x = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.cos(x * np.pi / 2.0)
    fade_in = np.sin(x * np.pi / 2.0)
    overlap = (head[:, -fade_samples:] * fade_out) + (tail[:, :fade_samples] * fade_in)
    return np.concatenate([head[:, :-fade_samples], overlap, tail[:, fade_samples:]], axis=1)


def append_with_crossfade(head: np.ndarray, tail: np.ndarray, sr: int, fade_seconds: float) -> np.ndarray:
    if tail.size == 0:
        return head
    fade_samples = min(int(fade_seconds * sr), head.shape[-1], tail.shape[-1])
    if fade_samples <= 0:
        return np.concatenate([head, tail], axis=1)

    x = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.cos(x * np.pi / 2.0)
    fade_in = np.sin(x * np.pi / 2.0)
    overlap = (head[:, -fade_samples:] * fade_out) + (tail[:, :fade_samples] * fade_in)
    return np.concatenate([head[:, :-fade_samples], overlap, tail[:, fade_samples:]], axis=1)


def time_stretch_multichannel(y: np.ndarray, rate: float) -> np.ndarray:
    channels = [librosa.effects.time_stretch(channel, rate=rate) for channel in y]
    return stack_channels(channels)


def progressive_time_stretch_multichannel(
    y: np.ndarray,
    start_rate: float,
    end_rate: float,
    target_samples: int,
    chunks: int = 12,
) -> np.ndarray:
    if y.shape[-1] < chunks * 512:
        return fit_length(time_stretch_multichannel(y, (start_rate + end_rate) / 2.0), target_samples)

    boundaries = np.linspace(0, y.shape[-1], chunks + 1, dtype=int)
    stretched_parts: list[np.ndarray] = []
    for index in range(chunks):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end <= start:
            continue
        progress = index / max(1, chunks - 1)
        eased = float(smoothstep(np.array([progress], dtype=np.float32))[0])
        rate = start_rate + (end_rate - start_rate) * eased
        stretched_parts.append(time_stretch_multichannel(y[:, start:end], rate))

    if not stretched_parts:
        return fit_length(y, target_samples)
    return fit_length(concat_with_microfades(stretched_parts, fade_samples=512), target_samples)


def fast_align_time_stretch_multichannel(
    y: np.ndarray,
    start_rate: float,
    end_rate: float,
    target_samples: int,
    align_samples: int,
) -> np.ndarray:
    align_samples = min(max(1, align_samples), target_samples)
    average_align_rate = (start_rate + end_rate) / 2.0
    source_align_len = min(y.shape[-1], max(1, int(align_samples * average_align_rate)))
    aligned = fit_length(time_stretch_multichannel(y[:, :source_align_len], average_align_rate), align_samples)

    remaining_target = target_samples - align_samples
    if remaining_target <= 0:
        return fit_length(aligned, target_samples)

    remaining_source = y[:, source_align_len:]
    if remaining_source.shape[-1] <= 0:
        sustain = fit_length(aligned[:, -1:], remaining_target)
    else:
        sustain = fit_length(time_stretch_multichannel(remaining_source, end_rate), remaining_target)
    return fit_length(append_with_crossfade(aligned, sustain, SR, fade_seconds=0.08), target_samples)


def pitch_shift_multichannel(y: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    channels = [librosa.effects.pitch_shift(channel, sr=sr, n_steps=semitones) for channel in y]
    return stack_channels(channels)


def stack_channels(channels: list[np.ndarray]) -> np.ndarray:
    min_len = min(channel.shape[-1] for channel in channels)
    return np.vstack([channel[:min_len] for channel in channels]).astype(np.float32)


def concat_with_microfades(parts: list[np.ndarray], fade_samples: int) -> np.ndarray:
    output = parts[0]
    for part in parts[1:]:
        output = append_with_crossfade(output, part, SR, fade_seconds=fade_samples / SR)
    return output


def low_band(channel: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    n_fft = 2048
    hop_length = 512
    spectrum = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs <= cutoff_hz).astype(np.float32)[:, np.newaxis]
    low = librosa.istft(spectrum * mask, hop_length=hop_length, length=channel.shape[-1])
    return low.astype(np.float32)


def choose_b_entry_offset(beat_times: list[float], sr: int) -> int:
    if not beat_times:
        return 0
    first_usable = next((time for time in beat_times if 0.05 <= time <= 4.0), None)
    if first_usable is None:
        return 0
    return max(0, int(first_usable * sr))


def morph_tail_to_original(shaped: np.ndarray, original: np.ndarray, fade_samples: int) -> np.ndarray:
    length = min(shaped.shape[-1], original.shape[-1])
    shaped = shaped[:, :length]
    original = original[:, :length]
    fade_samples = min(max(0, fade_samples), length)
    if fade_samples <= 0:
        return shaped.astype(np.float32)

    out = shaped.copy()
    x = smoothstep(np.linspace(0.0, 1.0, fade_samples, dtype=np.float32))
    out[:, -fade_samples:] = (shaped[:, -fade_samples:] * (1.0 - x)) + (original[:, -fade_samples:] * x)
    return out.astype(np.float32)


def weighted_average_rate(start_rate: float, end_rate: float, align_samples: int, blend_samples: int) -> float:
    if align_samples + blend_samples <= 0:
        return end_rate
    align_rate = (start_rate + end_rate) / 2.0
    return ((align_rate * align_samples) + (end_rate * blend_samples)) / (align_samples + blend_samples)


def fit_length(y: np.ndarray, target_samples: int) -> np.ndarray:
    target_samples = max(1, int(target_samples))
    if y.shape[-1] == target_samples:
        return y.astype(np.float32)
    if y.shape[-1] > target_samples:
        return y[:, :target_samples].astype(np.float32)
    pad_width = target_samples - y.shape[-1]
    return np.pad(y, ((0, 0), (0, pad_width)), mode="edge").astype(np.float32)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


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
