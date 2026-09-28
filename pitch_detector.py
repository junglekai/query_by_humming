"""音符起始點偵測，以及以自相關函數進行音高估計"""

import numpy as np
from scipy.signal import butter, filtfilt

import settings


def _high_pass_filter(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    centered = waveform.astype(np.float32, copy=False) - float(np.mean(waveform))
    normalized_cutoff = settings.ONSET.high_pass_hz / (sample_rate / 2.0)
    if not 0.0 < normalized_cutoff < 1.0:
        raise ValueError("高通濾波器截止頻率不適用於目前取樣率")

    numerator, denominator = butter(2, normalized_cutoff, btype="highpass")
    return filtfilt(numerator, denominator, centered).astype(np.float32)


def _onset_detection_curve(filtered: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, float]:
    block_size = max(1, int(0.01 * sample_rate))
    if len(filtered) < block_size:
        return np.empty(0, dtype=np.float32), block_size, 0.0

    amplitudes = np.asarray(
        [
            np.max(np.abs(filtered[start : start + block_size]))
            for start in range(0, len(filtered) - block_size + 1, block_size)
        ],
        dtype=np.float32,
    )

    amplitudes = np.maximum(amplitudes - 0.02, 0.0)
    mean_amplitude = float(np.mean(amplitudes))
    divisor = 0.2 * mean_amplitude + 0.1 if mean_amplitude > 0.0 else 0.1
    shaped = (amplitudes / divisor) ** 0.7

    onset_kernel = np.asarray(
        [3, 3, 4, 4, -1, -1, -2, -2, -2, -2, -2, -2],
        dtype=np.float32,
    )
    curve = np.convolve(shaped, onset_kernel, mode="same")
    median = float(np.median(curve))
    deviation = max(1.4826 * float(np.median(np.abs(curve - median))), 1e-8)
    threshold = median + settings.ONSET.threshold_scale * deviation
    return curve, block_size, threshold


def _refine_onsets(onsets: list[float], filtered: np.ndarray, sample_rate: int) -> list[float]:
    if not onsets:
        return []

    smoothing_size = max(1, int(0.005 * sample_rate))
    smoothing_kernel = np.ones(smoothing_size, dtype=np.float64) / smoothing_size
    envelope = np.convolve(np.abs(filtered), smoothing_kernel, mode="same")
    envelope_change = np.diff(envelope, prepend=envelope[0])

    refined: list[float] = []
    left = int(0.02 * sample_rate)
    right = int(0.03 * sample_rate)
    for onset in onsets:
        center = int(onset * sample_rate)
        start = max(0, center - left)
        stop = min(len(envelope_change), center + right)
        if stop <= start:
            refined.append(onset)
            continue
        sample_index = start + int(np.argmax(envelope_change[start:stop]))
        refined.append(sample_index / sample_rate)
    return refined


def detect_onsets(filtered: np.ndarray, sample_rate: int) -> list[float]:
    """偵測音符起始點，並以秒為單位回傳其時間"""
    curve, hop_size, threshold = _onset_detection_curve(filtered, sample_rate)
    if len(curve) < 5:
        return []

    curve_median = float(np.median(curve))
    curve_deviation = max(1.4826 * float(np.median(np.abs(curve - curve_median))), 1e-8)
    minimum_gap = max(1, int(settings.ONSET.minimum_gap_seconds * sample_rate / hop_size))

    candidates: list[float] = []
    previous_index = -minimum_gap
    for index in range(2, len(curve) - 2):
        value = curve[index]
        is_local_peak = value >= curve[index - 1] and value >= curve[index + 1]
        prominence = value - max(curve[index - 2], curve[index + 2])
        is_prominent = prominence > settings.ONSET.prominence_scale * curve_deviation
        is_separated = index - previous_index >= minimum_gap
        if value > threshold and is_local_peak and is_prominent and is_separated:
            candidates.append(((index + 3) * hop_size) / sample_rate)
            previous_index = index

    return _refine_onsets(candidates, filtered, sample_rate)


def _segments_from_onsets(onsets: list[float], sample_count: int, sample_rate: int) -> list[tuple[int, int]]:
    if not onsets:
        return [(0, sample_count)]

    boundaries = [min(sample_count, max(0, int(time * sample_rate))) for time in onsets]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    segments.append((boundaries[-1], sample_count))
    return segments


def estimate_segment_pitch(segment: np.ndarray, sample_rate: int) -> float | None:
    """利用正規化自相關峰值估計單一音段的基頻"""
    if len(segment) < int(0.03 * sample_rate):
        return None

    frame_size = int(settings.PITCH.frame_milliseconds * sample_rate / 1000.0)
    hop_size = int(settings.PITCH.hop_milliseconds * sample_rate / 1000.0)
    if frame_size <= 1 or hop_size <= 0 or len(segment) < frame_size:
        return None

    first_lag = max(2, int(sample_rate / settings.PITCH.maximum_hz))
    last_lag = min(frame_size - 1, int(sample_rate / settings.PITCH.minimum_hz))
    if first_lag >= last_lag:
        return None

    estimates: list[float] = []
    fft_size = 1 << int(np.ceil(np.log2(2 * frame_size - 1)))
    for start in range(0, len(segment) - frame_size + 1, hop_size):
        frame = segment[start : start + frame_size]
        frame = frame - float(np.mean(frame))
        if np.allclose(frame, 0.0, atol=1e-6):
            continue

        spectrum = np.fft.rfft(frame, n=fft_size)
        autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2, n=fft_size)[:frame_size]
        if autocorrelation[0] <= 0.0:
            continue
        autocorrelation /= autocorrelation[0]

        peak_lag = first_lag + int(np.argmax(autocorrelation[first_lag : last_lag + 1]))
        if autocorrelation[peak_lag] < settings.PITCH.correlation_threshold:
            continue

        interpolated_lag = float(peak_lag)
        if 1 < peak_lag < frame_size - 1:
            left, center, right = autocorrelation[peak_lag - 1 : peak_lag + 2]
            denominator = 2.0 * center - left - right
            if denominator != 0.0:
                offset = 0.5 * (right - left) / denominator
                if -1.0 < offset < 1.0:
                    interpolated_lag += float(offset)

        estimates.append(sample_rate / interpolated_lag)

    return float(np.median(estimates)) if estimates else None


def correct_octave_jumps(pitches: list[float | None]) -> list[float | None]:
    """當音高跳動過大時，從原頻率、半頻與倍頻中選擇較合理者"""
    corrected = list(pitches)
    previous: float | None = None
    for index, pitch in enumerate(corrected):
        if pitch is None:
            continue
        if previous is None:
            previous = pitch
            continue

        alternatives = (pitch, pitch / 2.0, pitch * 2.0)
        best = min(alternatives, key=lambda value: abs(12.0 * np.log2(value / previous)))
        original_jump = abs(12.0 * np.log2(pitch / previous))
        corrected[index] = best if original_jump > settings.PITCH.maximum_semitone_jump else pitch
        previous = corrected[index]
    return corrected


def analyze_audio(
    waveform: np.ndarray,
    sample_rate: int,
) -> tuple[list[float | None], list[float], list[float], list[tuple[int, int]]]:
    """依序執行濾波、起始點偵測、音訊分段與音高估計"""
    print("\n" + "=" * 60)
    print("開始音訊處理")
    print("=" * 60)
    print(f"取樣率: {sample_rate} Hz")
    print(
        "Onset 參數: "
        f"k_thr={settings.ONSET.threshold_scale}, "
        f"min_gap={settings.ONSET.minimum_gap_seconds}s"
    )
    print(f"F0 範圍: {settings.PITCH.minimum_hz}-{settings.PITCH.maximum_hz} Hz")

    filtered = _high_pass_filter(waveform, sample_rate)
    onset_times = detect_onsets(filtered, sample_rate)
    segments = _segments_from_onsets(onset_times, len(filtered), sample_rate)
    pitches = [estimate_segment_pitch(filtered[start:stop], sample_rate) for start, stop in segments]
    pitches = correct_octave_jumps(pitches)
    segment_times = [start / sample_rate for start, _ in segments]
    return pitches, segment_times, onset_times, segments
