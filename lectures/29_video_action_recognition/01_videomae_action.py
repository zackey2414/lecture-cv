"""01_videomae_action.py — transformers VideoMAE で行動クラスを推定する。

学べること:
  1) VideoMAE は「クリップ(複数フレーム)」を受け取る Transformer 系の行動認識モデル。
  2) VideoMAEImageProcessor が clip_len(=num_frames) 枚を前提に専用正規化を行う。
     入力は「フレーム(HWC uint8)のリスト」。processor が (1, T, C, H, W) に整える。
  3) transformers v5 の落とし穴: VideoMAE の q_bias/v_bias がロード時に取りこぼされ
     バイアスが 0 のままになることがある。helpers で元チェックポイントから手当てする。
  4) pipeline('video-classification') は高レベル API だが PyAV(av) が必要。
     本講座は av を入れない方針なので、未導入なら「概念紹介＋手書き等価コード」に切り替える。

VideoMAE は r3d_18 より大きく CPU では重い（初回 ~330MB DL）。本スクリプトは
モデルが落とせない/重い環境でも必ず exit 0 になるよう try/except でガードする。

実行: uv run python lectures/29_video_action_recognition/01_videomae_action.py
"""

from __future__ import annotations

import numpy as np
import torch

import action_helpers as H

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"


def show_bias_patch_note(model, n_patched: int) -> None:
    """v5 のバイアス取りこぼしと、その手当ての効果を数値で示す。"""
    attn0 = model.videomae.encoder.layer[0].attention.attention
    qsum = float(attn0.query.bias.detach().abs().sum())
    print("[transformers v5 の落とし穴: VideoMAE の q_bias/v_bias]")
    print(f"  手当てしたレイヤ数            : {n_patched}（12 なら全層 OK）")
    print(f"  layer0.query.bias の絶対値和  : {qsum:.2f}（0.00 のままならバイアス未ロード＝壊れている）")
    if n_patched == 0:
        print("  → 手当て不可（オフライン等）。バイアス 0 でも動くが精度は本来より落ちる点に注意。")


def run_videomae(clip_rgb: np.ndarray):
    """VideoMAE 本体を processor → model で動かし、top-5 を返す。"""
    model, processor, id2label, n_patched = H.load_videomae(MODEL_NAME)
    clip_len = model.config.num_frames  # 通常 16
    print(f"モデル: VideoMAE / クラス数: {model.config.num_labels} / clip_len(num_frames)={clip_len}")
    show_bias_patch_note(model, n_patched)

    # processor は「フレーム(HWC uint8)のリスト」を受け取る。clip_len 枚を等間隔で選ぶ。
    idx = H.uniform_indices(clip_rgb.shape[0], clip_len)
    frames = [clip_rgb[i] for i in idx]  # list of (H,W,3) uint8
    inputs = processor(frames, return_tensors="pt")  # -> pixel_values (1, T, C, H, W)
    print(f"\nprocessor 出力 pixel_values 形状: {tuple(inputs['pixel_values'].shape)} "
          f"（(1, T={clip_len}, C=3, H=224, W=224)）")

    with torch.inference_mode():
        logits = model(**inputs).logits
    top5 = H.topk_from_logits(logits, id2label, k=5)
    print("\n[VideoMAE] top-5:")
    for rank, (name, prob) in enumerate(top5, 1):
        print(f"  {rank}. {name:<40s} {prob:6.3f}")
    return top5, logits


def show_pipeline_concept(clip_rgb: np.ndarray) -> None:
    """pipeline('video-classification') の概念紹介（av があれば実行、無ければ案内のみ）。"""
    print("\n" + "-" * 72)
    print("補足: 高レベル API pipeline('video-classification')")
    print("-" * 72)
    try:
        import av  # noqa: F401
    except Exception:
        print("PyAV(av) が未導入のため pipeline('video-classification') は実行しません（概念のみ）。")
        print("  pipeline は内部で av を使って mp4 をデコードする。本講座は av を入れず")
        print("  cv2.VideoCapture + processor + model の手書き経路（上の実行内容）で等価を実現する。")
        print("  使いたい場合: uv add --group video av  （numpy2 系との相性は要確認）")
        return
    # av があるなら mp4 を書き出してパイプラインに渡す
    from transformers import pipeline

    mp4 = H.write_clip_mp4(clip_rgb, H.OUT_DIR / "01_pipeline_clip.mp4", fps=8.0)
    pipe = pipeline("video-classification", model=MODEL_NAME)
    out = pipe(str(mp4), top_k=5)
    print("pipeline 出力(top-5):")
    for o in out:
        print(f"  {o['label']:<40s} {o['score']:6.3f}")


def main() -> None:
    H.limit_cpu_threads(4)
    print("=" * 72)
    print("transformers VideoMAE(kinetics finetuned) で合成クリップを行動認識")
    print("=" * 72)

    # 合成クリップ（拡大する円。ズームイン的な動き）
    clip = H.make_action_clip(kind=2, num_frames=16)
    H.save_clip_grid(clip, H.OUT_DIR / "01_clip_grid.png", title="synthetic clip (circle_expand)")

    try:
        top5, _ = run_videomae(clip)
    except Exception as e:  # DL 失敗・メモリ不足等でも exit 0
        print(f"[案内] VideoMAE をロード/推論できませんでした: {type(e).__name__}: {e}")
        print("       初回はネットワークが必要（~330MB）。重みは ~/.cache/huggingface に保存されます。")
        print("       軽い代替として 02_r3d18_action.py（小さい 3D CNN）を試してください。")
        return

    names = [n.replace(" ", "\n", 1) for n, _ in top5]
    H.save_bar(names, [p for _, p in top5], H.OUT_DIR / "01_top5.png",
               title="VideoMAE top-5 (circle_expand clip)", ylabel="prob", ylim=(0, 1))
    H.save_json(
        {
            "model": MODEL_NAME,
            "pixel_values_shape": "(1, T=16, C=3, H=224, W=224)",
            "top5": top5,
        },
        H.OUT_DIR / "01_videomae.json",
    )

    show_pipeline_concept(clip)
    print("\n保存: 01_clip_grid.png / 01_top5.png / 01_videomae.json")


if __name__ == "__main__":
    main()
