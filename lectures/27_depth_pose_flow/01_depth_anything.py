"""01: 単眼深度推定 ― Depth Anything V2 (Small) で 1 枚画像から相対深度を得る。

学ぶこと:
  - transformers の pipeline('depth-estimation') で前処理〜推論〜後処理を一気に通す正準フロー。
  - 出力 predicted_depth は **相対(逆)深度**（値が大きい＝カメラに近い）であり、絶対距離ではない。
    そのため可視化前に正規化が必須、評価には GT へのスケール合わせが必要になる、という勘所。
  - 相対深度を絶対深度に近づける中央値スケール合わせと、AbsRel/RMSE/δ<1.25 の算出。

入力は合成シーン（床・壁・箱・球からなる擬似室内）。data/27_depth_pose_flow/ に画像を置けば
それを優先して使う。ネット不通などで重み DL に失敗しても、フォールバック深度で最後まで通す
（必ず exit 0）。

実行: uv run python lectures/27_depth_pose_flow/01_depth_anything.py
出力: outputs/27_depth_pose_flow/01_*.png / 01_depth_metrics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import dpf_helpers as H  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load_input() -> tuple[np.ndarray, np.ndarray | None]:
    """入力 RGB と（あれば）GT 深度を返す。data/ に実画像があれば優先（その場合 GT は無し）。"""
    H.DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        [p for p in H.DATA_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )
    if candidates:
        img = np.array(Image.open(candidates[0]).convert("RGB"))
        print(f"[入力] data/ の実画像を使用: {candidates[0].name}（GT 深度は無し）")
        return img, None
    print("[入力] 合成シーン（床・壁・箱・球）を生成（GT 深度つき）")
    rgb, gt = H.make_depth_scene()
    return rgb, gt


def run_depth_anything(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Depth Anything V2 で相対(逆)深度を推定。失敗時は GT 風フォールバックを返す。

    返り値: (predicted_inverse_depth(H,W) float32, source 文字列)
    """
    try:
        from transformers import pipeline

        # device は指定しなければ CPU。CPU でも Small なら数秒で完了する。
        pipe = pipeline("depth-estimation", model=MODEL_ID)
        out = pipe(Image.fromarray(rgb))
        # predicted_depth は逆深度（近い=大）。numpy float32 に揃える。
        pred = np.asarray(out["predicted_depth"], dtype=np.float32)
        return pred, "Depth-Anything-V2-Small"
    except Exception as e:  # noqa: BLE001  ネット不通/未導入でも止めない
        print(f"[警告] モデル実行に失敗（{type(e).__name__}: {e}）。フォールバック深度で継続。")
        # フォールバック: 行方向グラデを逆深度に見立てた擬似マップ（パイプラインの形だけ保つ）
        h, w = rgb.shape[:2]
        ramp = np.linspace(1.0, 0.1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
        return ramp, "fallback(ramp)"


def main() -> None:
    out = H.ensure_output_dir()
    H.set_seed(0)

    rgb, gt_depth = load_input()
    Image.fromarray(rgb).save(out / "01_input.png")

    print(f"\n=== 01: Depth Anything V2 (Small) ===\nmodel = {MODEL_ID}")
    pred_inv, source = run_depth_anything(rgb)
    print(f"[出力] source={source}, predicted_depth shape={pred_inv.shape}, "
          f"min={pred_inv.min():.3f}, max={pred_inv.max():.3f}（値が大きいほど“近い”=逆深度）")

    # --- 可視化: 逆深度をそのまま正規化して colormap（近い=明るい）---
    depth_vis = H.colorize_depth(pred_inv)
    Image.fromarray(depth_vis).save(out / "01_depth_colored.png")

    # 入力と深度を横並びで保存（見比べやすく）
    side = np.concatenate(
        [rgb, np.array(Image.fromarray(depth_vis).resize((rgb.shape[1], rgb.shape[0])))], axis=1
    )
    Image.fromarray(side).save(out / "01_input_vs_depth.png")
    print(f"  saved: {out/'01_depth_colored.png'}, {out/'01_input_vs_depth.png'}")

    # --- 評価（GT がある合成シーンのときだけ）---
    metrics: dict = {"source": source}
    if gt_depth is not None:
        # 逆深度 → 深度へ: depth ∝ 1/inv。eps で 0 割りを防ぐ。
        pred_depth = 1.0 / (pred_inv + 1e-6)
        # 相対深度はスケール不定 → 中央値で GT にスケール合わせしてから指標化（これが定石）。
        pred_aligned = H.align_scale_median(pred_depth, gt_depth)
        metrics["abs_rel"] = H.abs_rel(pred_aligned, gt_depth)
        metrics["rmse"] = H.rmse(pred_aligned, gt_depth)
        metrics["delta_1.25"] = H.delta_accuracy(pred_aligned, gt_depth, 1.25)
        print("\n[評価] 相対深度をスケール合わせ後に算出（合成 GT 基準）:")
        print(f"  AbsRel    = {metrics['abs_rel']:.4f}（小さいほど良い）")
        print(f"  RMSE      = {metrics['rmse']:.4f}")
        print(f"  δ<1.25    = {metrics['delta_1.25']:.4f}（大きいほど良い, 最大1.0）")
        print("  ※ 合成シーンはモデルの学習分布外なので数値は参考値。定義の確認が目的。")
    else:
        print("\n[評価] 実画像入力（GT 無し）のため指標はスキップ。可視化のみ。")

    (out / "01_depth_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out/'01_input.png'}, {out/'01_depth_metrics.json'}")


if __name__ == "__main__":
    main()
