"""哼唱歌曲查詢系統的集中設定"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "hummingdata"
INPUT_AUDIO_DIR = DATA_DIR / "20"
SONG_DATABASE_FILE = DATA_DIR / "Processed_Output.docx"


@dataclass(frozen=True)
class OnsetSettings:
    high_pass_hz: float = 60.0
    threshold_scale: float = 1.3
    minimum_gap_seconds: float = 0.2
    prominence_scale: float = 0.015


@dataclass(frozen=True)
class PitchSettings:
    frame_milliseconds: float = 28.0
    hop_milliseconds: float = 8.0
    minimum_hz: float = 60.0
    maximum_hz: float = 500.0
    correlation_threshold: float = 0.45
    maximum_semitone_jump: float = 7.0


@dataclass(frozen=True)
class MatchingSettings:
    gap_penalty: float = 4.0
    rhythm_weight: float = 0.05


ONSET = OnsetSettings()
PITCH = PitchSettings()
MATCHING = MatchingSettings()
