"""哼唱歌曲查詢系統的命令列主程式"""

import melody_database
import pitch_detector
import settings
import wav_input


def _print_results(results: list[melody_database.MatchResult], limit: int = 10) -> None:
    print("\n" + "=" * 60)
    print("匹配結果")
    print("=" * 60)
    print(f"\n前 {min(limit, len(results))} 名:")
    print("-" * 60)

    for rank, result in enumerate(results[:limit], start=1):
        print(
            f"{rank:2d}. {result.song.name:<30} | "
            f"距離: {result.distance:7.2f}"
        )

    best = results[0]
    print("\n" + "=" * 60)
    print(f"最佳匹配: {best.song.name}")
    print(f"編輯距離: {best.distance:.2f}")
    print("=" * 60)


def main() -> None:
    """執行從 WAV 音訊到歌曲排名的完整流程"""
    try:
        waveform, sample_rate, _source = wav_input.get_audio_input()
        pitches, segment_times, _onsets, _segments = pitch_detector.analyze_audio(
            waveform,
            sample_rate,
        )
        query_intervals, query_beats = melody_database.pitches_to_query(
            pitches,
            segment_times,
        )
        if not query_intervals:
            print("無法由這段音訊建立有效的旋律查詢序列")
            return

        songs = melody_database.load_song_database(settings.SONG_DATABASE_FILE)
        if not songs:
            print("歌曲資料庫中沒有可供比對的歌曲")
            return

        results = melody_database.rank_songs(query_intervals, query_beats, songs)
        if not results:
            print("沒有產生匹配結果")
            return
        _print_results(results)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"程式無法繼續執行: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程式已由使用者中止")
