# Query by Humming System

姓名: 林琮凱  
學號: B12504107

## 程式結構

- `run_qbh.py`: 主程式，依序執行音訊載入、音高分析、查詢序列建立與歌曲比對。
- `wav_input.py`: 搜尋、選擇、讀取及正規化 WAV 音訊。
- `pitch_detector.py`: 執行高通濾波、onset detection、音訊分段及 ACF F0 estimation。
- `melody_database.py`: 建立 MIDI difference/beat query、讀取歌曲資料庫並執行 DP matching。
- `settings.py`: 集中管理資料路徑與 onset、F0、DP 參數。
- `hummingdata/Processed_Output.docx`: 歌名、MIDI sequence 與 beat sequence 資料庫。
- `hummingdata/20/`: 程式預設讀取的 20 秒 WAV 測試資料夾。

## 執行方式

在專案資料夾內執行：

```bash
python run_qbh.py
```

程式會直接使用 `hummingdata/20`，不會詢問輸入模式或資料夾。接著輸入檔名關鍵字，再輸入符合檔案的編號即可開始辨識。若關鍵字找不到檔案，程式會直接要求重新輸入。

若要改用其他預設資料夾，請修改 `settings.py` 的 `INPUT_AUDIO_DIR`。

## 環境需求

```bash
pip install numpy scipy python-docx
```
