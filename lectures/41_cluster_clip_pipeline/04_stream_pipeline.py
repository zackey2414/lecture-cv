"""04 Stream — multiprocessing で「取得」と「推論」を分離した擬似ストリーム処理。

参考実装対応:
  stream/pipeline.py（3 プロセスのオーケストレーション）
  + stream/capture.py（ソース抽象化 + 適応サンプリング + キュー満杯ドロップ）
  + stream/consumer.py（メモリ上画像で推論+クラスタリング）
  + stream/writer.py（DB + FAISS をインクリメンタル更新し常時検索可能に保つ）。

構成:  Capture(CPU) --task_queue--> Consumer(CPU重) --result_queue--> Writer(CPU)

学ぶこと:
  - 取得は軽い／推論(CLIP)は重い。別プロセスにして並行させ、CPU バウンドの推論を詰まらせない。
  - task_queue は maxsize 付き。満杯なら put_nowait で即ドロップ → 実時間に遅れない（リアルタイム保証）。
  - POISON_PILL を流して各段を順に安全終了させる。
  - 適応サンプリング（ヒストグラム差分）で「場面の変わり目」だけを残し、無駄な推論を減らせる。

注意: torch を載せるので各プロセスは spawn で独立起動する（__main__ ガード必須）。

実行:
    uv run python lectures/41_cluster_clip_pipeline/04_stream_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_clip_helpers as H  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

N_FRAMES = 24


def plot_comparison(passthrough: dict, sampled: dict, out_name: str) -> Path:
    """2 シナリオの captured/dropped/sampled_out/processed を棒グラフで比較する。"""
    labels = ["captured", "dropped", "sampled_out", "processed"]
    a = [passthrough[k] for k in labels]
    b = [sampled[k] for k in labels]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, a, w, label="passthrough (small queue)", color="tab:red")
    ax.bar(x + w / 2, b, w, label="histogram sampler", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("frames")
    ax.set_title("Stream: realtime drop vs adaptive sampling (n_frames=24)")
    ax.legend(fontsize=8)
    return H.save_fig(fig, out_name)


def verify_searchable(build_dir: Path) -> int:
    """ストリーム構築済みの index が検索可能か、テキストクエリで確認する。"""
    import faiss

    index = faiss.read_index(str(build_dir / H.INDEX_NAME))
    model, _, tokenizer = H.load_clip(H.pick_device())
    qv = H.encode_text(model, tokenizer, ["a blue region"], H.pick_device())
    hits = H.search_index(build_dir, qv, top_k=3)
    print(f"[stream] streamed index ntotal={index.ntotal}, query='a blue region' top3:")
    for r in hits:
        print(f"   {r['frame_id'][-9:]} cluster={r['cluster_idx']} score={r['score']:.3f}")
    return index.ntotal


def main() -> int:
    H.set_seed(0)
    base = H.outputs_dir("capstone")

    # --- シナリオ A: 全フレーム送出・小さいキュー → リアルタイム保証のためドロップが出る ---
    print("=" * 64)
    print("[stream] シナリオ A: passthrough + 小さいキュー（ドロップで実時間を守る）")
    pa = H.run_stream_pipeline(
        base / "stream_passthrough", n_frames=N_FRAMES, sampler="passthrough",
        queue_size=4, fps_cap=30, batch_size=2, video_name="liveA",
    )
    print(f"   warmup={pa['warmup_sec']:.1f}s stream={pa['elapsed_sec']:.1f}s "
          f"effective_fps={pa['effective_fps']:.2f}")
    print(f"   total={pa['total']} captured={pa['captured']} dropped={pa['dropped']} processed={pa['processed']}")

    # --- シナリオ B: ヒストグラム差分サンプラー → 場面の変わり目だけ推論（無駄を削減）---
    print("\n[stream] シナリオ B: histogram_delta サンプラー（場面転換だけ残す）")
    sb = H.run_stream_pipeline(
        base / "stream_sampled", n_frames=N_FRAMES, sampler="histogram_delta", threshold=0.1,
        queue_size=8, fps_cap=30, batch_size=2, video_name="liveB",
    )
    print(f"   stream={sb['elapsed_sec']:.1f}s effective_fps={sb['effective_fps']:.2f}")
    print(f"   total={sb['total']} sampled_out={sb['sampled_out']} captured={sb['captured']} "
          f"dropped={sb['dropped']} processed={sb['processed']}")

    saved_pct = 100.0 * sb["sampled_out"] / max(1, sb["total"])
    print(f"   → サンプラーが {sb['sampled_out']}/{sb['total']} 枚 ({saved_pct:.0f}%) の推論を省いた")

    # --- 構築された index が検索可能か確認 ---
    print()
    ntotal = verify_searchable(base / "stream_passthrough")

    # --- 比較図 ---
    fig_path = plot_comparison(pa, sb, "04_stream_compare.png")
    print(f"[plot] saved -> {fig_path}")

    # --- 自己検証 ---
    assert pa["processed"] >= 1 and ntotal > 0, "ストリームで index が構築されていない"
    assert pa["dropped"] >= 1, "小さいキューなのにドロップが起きていない（リアルタイム保証の確認失敗）"
    assert sb["sampled_out"] >= 1, "適応サンプラーが 1 枚も間引いていない"
    print("[done] Stream 完了。総仕上げは mini_project.py（Split→Build→Search→Stream 一気通貫）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
