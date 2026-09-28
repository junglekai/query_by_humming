"""建立查詢序列、解析歌曲資料庫並進行旋律比對"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from docx import Document

import settings


@dataclass(frozen=True)
class Song:
    name: str
    midi_notes: tuple[int, ...]
    beats: tuple[float, ...]


@dataclass(frozen=True)
class MatchResult:
    song: Song
    distance: float


def pitches_to_query(
    pitches: list[float | None],
    segment_times: list[float],
) -> tuple[list[int], list[float]]:
    """將有效的 F0 估計值轉換成音程差與時間間隔序列"""
    valid = [
        (pitch, time)
        for pitch, time in zip(pitches, segment_times)
        if pitch is not None and pitch > 0.0 and np.isfinite(pitch)
    ]

    midi_notes = [int(round(69.0 + 12.0 * np.log2(pitch / 440.0))) for pitch, _ in valid]
    intervals = [current - previous for previous, current in zip(midi_notes, midi_notes[1:])]
    elapsed_times = [current - previous for (_, previous), (_, current) in zip(valid, valid[1:])]

    print("\nMIDI 計算:")
    print(f"總音段數: {len(pitches)}")
    print(f"有效 F0: {len(valid)}/{len(pitches)}")
    print(f"MIDI 差異數: {len(intervals)}")
    print(f"節奏數: {len(elapsed_times)}")
    return intervals, elapsed_times


def _numeric_values(line: str) -> list[float]:
    values: list[float] = []
    for token in line.split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _is_separator(line: str) -> bool:
    return bool(line) and all(character in "-=－—_ " for character in line)


def load_song_database(document_path: str | Path) -> list[Song]:
    """從 DOCX 檔案解析重複排列的歌名、MIDI 音符與拍值資料"""
    path = Path(document_path)
    if not path.is_file():
        raise FileNotFoundError(f"歌曲資料庫不存在: {path}")

    document = Document(path)
    records: dict[str, dict[str, list[float]]] = {}
    current_name: str | None = None

    for paragraph in document.paragraphs:
        line = paragraph.text.strip()
        if not line or line == "處理結果" or _is_separator(line):
            continue

        numbers = _numeric_values(line)
        if not numbers:
            current_name = line
            records[current_name] = {"notes": [], "beats": []}
            continue

        if current_name is None:
            continue
        if not records[current_name]["notes"]:
            records[current_name]["notes"] = numbers
        else:
            records[current_name]["beats"] = numbers

    songs: list[Song] = []
    for name, record in records.items():
        notes = tuple(int(value) for value in record["notes"])
        if len(notes) < 2:
            continue

        expected_beats = len(notes) - 1
        beats = list(record["beats"][:expected_beats])
        beats.extend([1.0] * (expected_beats - len(beats)))
        songs.append(Song(name=name, midi_notes=notes, beats=tuple(beats)))

    return songs


def _pitch_interval_cost(interval_difference: int) -> float:
    octaves, remainder = divmod(abs(interval_difference), 12)
    if remainder > 6:
        remainder = 12 - remainder
        octaves += 1
    return float(2 * octaves + remainder)


def _rhythm_cost(query_beat: float, song_beat: float) -> float:
    if query_beat <= 0.0 or song_beat <= 0.0:
        return 0.0
    ratio = float(np.clip(query_beat / song_beat, 0.01, 100.0))
    return settings.MATCHING.rhythm_weight * abs(float(np.log2(ratio)))


def _query_to_song_distance(
    query_intervals: list[int],
    query_beats: list[float],
    song: Song,
) -> float:
    song_intervals = [
        current - previous
        for previous, current in zip(song.midi_notes, song.midi_notes[1:])
    ]
    query_rhythm = query_beats if len(query_beats) == len(query_intervals) else [1.0] * len(query_intervals)
    song_rhythm = list(song.beats) if len(song.beats) == len(song_intervals) else [1.0] * len(song_intervals)

    row_count = len(query_intervals) + 1
    column_count = len(song_intervals) + 1
    table = np.zeros((row_count, column_count), dtype=np.float64)

    for row in range(1, row_count):
        table[row, 0] = table[row - 1, 0] + settings.MATCHING.gap_penalty * query_rhythm[row - 1]

    for row in range(1, row_count):
        for column in range(1, column_count):
            delete_query = table[row - 1, column] + settings.MATCHING.gap_penalty * query_rhythm[row - 1]
            skip_song = table[row, column - 1] + settings.MATCHING.gap_penalty * song_rhythm[column - 1]
            substitution = (
                table[row - 1, column - 1]
                + _pitch_interval_cost(query_intervals[row - 1] - song_intervals[column - 1])
                + _rhythm_cost(query_rhythm[row - 1], song_rhythm[column - 1])
            )
            table[row, column] = min(delete_query, skip_song, substitution)

    return float(np.min(table[-1]))


def rank_songs(
    query_intervals: list[int],
    query_beats: list[float],
    songs: list[Song],
) -> list[MatchResult]:
    """使用半全域DP計算距離，並排列資料庫歌曲"""
    print(f"查詢序列長度: {len(query_intervals)}")
    print(
        f"比對參數: gap={settings.MATCHING.gap_penalty}, "
        f"rhythm_weight={settings.MATCHING.rhythm_weight}"
    )

    results: list[MatchResult] = []
    for song in songs:
        distance = _query_to_song_distance(query_intervals, query_beats, song)
        results.append(
            MatchResult(
                song=song,
                distance=distance,
            )
        )

    return sorted(results, key=lambda result: result.distance)
