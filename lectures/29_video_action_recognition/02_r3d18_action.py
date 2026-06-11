"""02_r3d18_action.py — torchvision r3d_18(3D CNN) で行動クラスを推定する。

学べること:
  1) 3D CNN は (B, C, T, H, W) という「時間 T を持つ 5 次元テンソル」を入力に取る。
  2) r3d_18 専用の正規化（ImageNet とは別の mean/std・112x112）を正しく適用する。
  3) torchvision の weights.transforms() と「手書き前処理」が一致することを確認する。
  4) cv2.VideoCapture で mp4 からフレームを読み戻して推論する（read_video 廃止の代替）。

r3d_18 は小さく CPU でも速いので、本章の「動かして理解する」主役。
合成クリップに本物の Kinetics ラベルは無いが、モデルは時間方向の動きに反応して
それらしい行動クラスを返す。ここでは「形式が正しく通ること」を主目的にする。

実行: uv run python lectures/29_video_action_recognition/02_r3d18_action.py
"""

from __future__ import annotations

import numpy as np
import torch

import action_helpers as H


def manual_vs_official_preprocess(clip_rgb: np.ndarray) -> None:
    """手書き前処理 と torchvision 公式 transforms() が一致することを確かめる。"""
    from torchvision.models.video import R3D_18_Weights

    weights = R3D_18_Weights.DEFAULT
    # --- 手書き版（helpers）: (T,H,W,3)uint8 -> (1,C,T,H,W) ---
    idx = H.uniform_indices(clip_rgb.shape[0], 16)
    mine = H.preprocess_for_r3d(clip_rgb, idx)

    # --- 公式版: transforms() は (T,C,H,W)uint8 を受け取り (C,T,H,W) を返す ---
    # 公式は内部で resize(128x171)->center-crop(112) を行うので、合成クリップ(160x120)では
    # 手書きの単純 resize と画素は厳密一致しない。ここでは「mean/std と最終形状が同じ」ことを見る。
    frames = clip_rgb[idx]  # (T,H,W,3)
    t_chw = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T,C,H,W) uint8
    official = weights.transforms()(t_chw).unsqueeze(0)  # (1,C,T,H,W)

    print("[前処理の比較]")
    print(f"  手書き  形状={tuple(mine.shape)} mean={mine.mean():+.4f} std={mine.std():.4f}")
    print(f"  公式    形状={tuple(official.shape)} mean={official.mean():+.4f} std={official.std():.4f}")
    print("  → どちらも (1,3,16,112,112)。正規化後の統計が近いことを確認（リサイズ流儀の差で完全一致はしない）")


def predict_clip(model, categories, clip_rgb: np.ndarray, normalize: bool, tag: str):
    """1 クリップを推論して top-5 を表示。normalize を切り替えると壊れ方が見える。"""
    idx = H.uniform_indices(clip_rgb.shape[0], 16)
    x = H.preprocess_for_r3d(clip_rgb, idx, normalize=normalize)
    with torch.inference_mode():
        logits = model(x)
    top5 = H.topk_from_logits(logits, categories, k=5)
    print(f"\n[{tag}] top-5:")
    for rank, (name, prob) in enumerate(top5, 1):
        print(f"  {rank}. {name:<40s} {prob:6.3f}")
    return top5


def main() -> None:
    H.limit_cpu_threads(4)
    print("=" * 72)
    print("torchvision r3d_18(Kinetics-400) で合成クリップを行動認識")
    print("=" * 72)

    # 1) モデルロード（初回のみ ~127MB DL）
    try:
        model, categories = H.load_r3d()
    except Exception as e:  # オフライン等で DL できない場合も exit 0 で案内のみ
        print(f"[案内] r3d_18 をロードできませんでした（オフライン？）: {type(e).__name__}: {e}")
        print("       初回はネットワークが必要です。重みは ~/.cache/torch/hub に保存されます。")
        return
    print(f"モデル: r3d_18 / クラス数: {len(categories)} / device: {H.get_device()}")

    # 2) 合成クリップを生成（円が左→右に動く）
    clip = H.make_action_clip(kind=0, num_frames=16)
    H.save_clip_grid(clip, H.OUT_DIR / "02_clip_grid.png", title="synthetic clip (circle_right)")

    # 3) 手書き前処理 vs 公式 transforms()
    manual_vs_official_preprocess(clip)

    # 4) 正しい正規化で推論（これが基準）
    top5_good = predict_clip(model, categories, clip, normalize=True, tag="正規化あり(正しい)")

    # 5) 正規化を外すと予測が変わる（前処理の重要性）
    top5_bad = predict_clip(model, categories, clip, normalize=False, tag="正規化なし(壊れた前処理)")

    changed = top5_good[0][0] != top5_bad[0][0]
    print(f"\n→ top-1 は正規化の有無で {'変化した' if changed else '変化しなかった'}"
          f"（正規化を誤ると出力が無意味になりうる）")

    # 6) cv2.VideoCapture でのフレーム抽出（torchvision read_video 廃止の代替）
    mp4 = H.write_clip_mp4(clip, H.OUT_DIR / "02_clip.mp4", fps=8.0)
    frames_back = H.read_clip_mp4(mp4)
    top5_io = predict_clip(model, categories, frames_back, normalize=True, tag="mp4読戻し→推論")
    print(f"\nmp4 経由({frames_back.shape[0]}枚)でも top-1 = {top5_io[0][0]}"
          f"（直接クリップと {'一致' if top5_io[0][0]==top5_good[0][0] else '近い'}）")

    # 7) top-5 の確率を棒グラフで保存
    names = [n.replace(" ", "\n", 1) for n, _ in top5_good]
    H.save_bar(names, [p for _, p in top5_good], H.OUT_DIR / "02_top5.png",
               title="r3d_18 top-5 (circle_right clip)", ylabel="prob", ylim=(0, 1))

    H.save_json(
        {
            "model": "r3d_18 / Kinetics-400",
            "input_shape": "(1, C=3, T=16, H=112, W=112)",
            "top5_normalized": top5_good,
            "top5_not_normalized": top5_bad,
            "top1_changed_by_normalization": bool(changed),
            "mp4_roundtrip_top1": top5_io[0][0],
        },
        H.OUT_DIR / "02_r3d18.json",
    )
    print("\n保存: 02_clip_grid.png / 02_top5.png / 02_clip.mp4 / 02_r3d18.json")


if __name__ == "__main__":
    main()
