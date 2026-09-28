"""搜尋、選擇與讀取 WAV 音訊檔案"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

import settings


def normalize_samples(samples: np.ndarray) -> np.ndarray:
    """將 PCM 或浮點數取樣資料轉成單聲道 float32 波形"""
    if samples.ndim > 1:
        samples = samples[:, 0]

    if np.issubdtype(samples.dtype, np.unsignedinteger):
        limits = np.iinfo(samples.dtype)
        midpoint = (limits.max + 1) / 2.0
        return ((samples.astype(np.float32) - midpoint) / midpoint).astype(np.float32)

    if np.issubdtype(samples.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
        return (samples.astype(np.float32) / scale).astype(np.float32)

    return samples.astype(np.float32, copy=False)


def read_wave_file(file_path: str | Path) -> tuple[np.ndarray, int]:
    """讀取一個 WAV 檔案，回傳正規化波形與取樣率"""
    path = Path(file_path)
    try:
        sample_rate, samples = wavfile.read(path)
    except Exception as exc:
        raise RuntimeError(f"讀取 WAV 檔案失敗: {path}") from exc

    waveform = normalize_samples(samples)
    print("\n--- 讀檔資訊 ---")
    print(f"wave_data.shape = {waveform.shape}")
    print(f"fs = {sample_rate}")
    print(f"總樣本數 = {len(waveform)}")
    print(f"總秒數 = {len(waveform) / sample_rate:.2f}")
    return waveform, int(sample_rate)


def _available_wave_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.name.casefold(),
    )


def select_wave_file(directory: str | Path) -> Path:
    """要求使用者輸入檔名關鍵字，再選擇符合的 WAV 檔案"""
    audio_dir = Path(directory).resolve()
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"音訊資料夾不存在: {audio_dir}")

    all_files = _available_wave_files(audio_dir)
    if not all_files:
        raise FileNotFoundError(f"資料夾中沒有 WAV 檔案: {audio_dir}")

    print(f"使用預設音訊資料夾: {audio_dir}")
    print(f"此資料夾共有 {len(all_files)} 個 WAV 檔案")

    while True:
        keyword = input("請輸入檔名的一部分（可留空顯示全部）: ").strip()
        matches = [path for path in all_files if keyword in path.name]
        if matches:
            break

        print(f"找不到包含「{keyword}」的 WAV 檔案，請重新輸入")
        print("參考檔名:")
        for path in all_files[:5]:
            print(f"  {path.name}")

    print(f"找到 {len(matches)} 個符合的檔案")
    for index, path in enumerate(matches):
        print(f"[{index}] {path.name}")
    return _choose_match(matches)


def _choose_match(matches: list[Path]) -> Path:
    """檢查使用者輸入的編號，並回傳所選檔案"""
    while True:
        choice = input(f"請輸入想要的檔案編號 [0-{len(matches) - 1}]: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(matches):
            return matches[int(choice)]
        print(f"無效編號，請輸入 0 到 {len(matches) - 1}")


def get_audio_input() -> tuple[np.ndarray, int, dict[str, str]]:
    """從設定的預設資料夾載入使用者選擇的 WAV 檔案"""
    print("=" * 60)
    print("哼唱辨識系統")
    print("=" * 60)

    selected_path = select_wave_file(settings.INPUT_AUDIO_DIR)
    waveform, sample_rate = read_wave_file(selected_path)
    source = {
        "mode": "file",
        "path": str(selected_path),
        "filename": selected_path.name,
    }
    return waveform, sample_rate, source
