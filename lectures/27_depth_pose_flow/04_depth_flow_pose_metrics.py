"""04: 評価指標の定義確認 ― 深度/フロー/姿勢の指標を“合成 GT + 既知の予測”で検算する。

ここではモデルを使わない。代わりに「真値(GT)」と、それを少し劣化させた「予測」を自分で作り、
指標がどう動くかを完全に制御された状況で確かめる。指標の **定義そのもの** を血肉にするのが目的。

  - 深度: 相対深度はスケール不定 → 中央値スケール合わせ後に AbsRel / RMSE / δ<1.25。
  - フロー: EPE（端点誤差）= 予測と真のベクトル差の平均ノルム。
  - 姿勢: OKS（物体スケールと per-keypoint 定数で正規化した類似度）と PCK@0.2。

実行: uv run python lectures/27_depth_pose_flow/04_depth_flow_pose_metrics.py
出力: outputs/27_depth_pose_flow/04_metrics.json / 04_metrics_table.png
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")  # headless 描画（imshow を使わない）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import dpf_helpers as H  # noqa: E402


def eval_depth() -> dict:
    """深度指標の検算: GT に対し『定数倍 + ノイズ』の予測を作り、スケール合わせ後に算出。"""
    _, gt = H.make_depth_scene()
    rng = np.random.default_rng(10)
    # 相対深度を模す: 真の深度の 0.4 倍 + 軽いノイズ（スケールがズレた予測）
    pred = gt * 0.4 + rng.normal(0, 0.05, size=gt.shape).astype(np.float32)
    pred = np.clip(pred, 0.05, None)

    before = H.abs_rel(pred, gt)  # スケール合わせ前（大きく出る）
    aligned = H.align_scale_median(pred, gt)  # 中央値で GT に合わせる
    return {
        "abs_rel_before_align": before,
        "abs_rel": H.abs_rel(aligned, gt),
        "rmse": H.rmse(aligned, gt),
        "delta_1.25": H.delta_accuracy(aligned, gt, 1.25),
        "note": "相対深度はスケール合わせ後に評価する。align 前後で AbsRel が大きく変わる点に注目。",
    }


def eval_flow() -> dict:
    """フロー指標の検算: 一定並進 GT に対し『一様ノイズ』を足した予測の EPE。"""
    _, _, gt = H.make_translating_pair(dx=6, dy=3)
    rng = np.random.default_rng(11)
    pred = gt + rng.normal(0, 0.5, size=gt.shape).astype(np.float32)
    return {
        "epe": H.endpoint_error(pred, gt),
        "epe_perfect": H.endpoint_error(gt, gt),
        "note": "EPE は px 単位。ノイズ σ を上げると EPE が比例して増える。",
    }


def eval_pose() -> dict:
    """姿勢指標の検算: GT 関節に画素ノイズを足した予測の OKS と PCK@0.2。"""
    frames = H.make_humanoid_frames(1)
    _, gt = frames[0]
    rng = np.random.default_rng(12)

    # bbox から物体スケール(area)と参照長(対角)を計算
    xy = gt[:, :2]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    area = float((x1 - x0) * (y1 - y0))
    ref_len = float(np.hypot(x1 - x0, y1 - y0))

    # 予測 = GT + 等方ガウスノイズ（数 px）
    noise = rng.normal(0, 4.0, size=(17, 2)).astype(np.float32)
    pred = xy + noise

    return {
        "area": area,
        "ref_len": ref_len,
        "oks": H.object_keypoint_similarity(pred, gt, area=area),
        "oks_perfect": H.object_keypoint_similarity(xy, gt, area=area),
        "pck@0.2": H.pck(pred, gt, ref_len=ref_len, alpha=0.2),
        "note": "OKS は 1 が完全一致。小関節(σ小)ほど誤差に厳しい。PCK は閾値内に入った点の割合。",
    }


def render_table(results: dict, path: pathlib.Path) -> None:
    """主要指標を 1 枚の表 PNG にまとめる（レポート用）。"""
    # 表 PNG のラベルは ASCII に統一（DejaVu フォントに日本語グリフが無く警告が出るため）
    rows = [
        ("Depth AbsRel (aligned)", f"{results['depth']['abs_rel']:.4f}"),
        ("Depth AbsRel (raw)", f"{results['depth']['abs_rel_before_align']:.4f}"),
        ("Depth RMSE", f"{results['depth']['rmse']:.4f}"),
        ("Depth δ<1.25", f"{results['depth']['delta_1.25']:.4f}"),
        ("Flow EPE", f"{results['flow']['epe']:.4f}"),
        ("Pose OKS", f"{results['pose']['oks']:.4f}"),
        ("Pose PCK@0.2", f"{results['pose']['pck@0.2']:.4f}"),
    ]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["metric", "value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("27_depth_pose_flow : metrics on synthetic GT", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    print("=== 04: 評価指標の定義確認（合成 GT, モデル不要）===")

    results = {"depth": eval_depth(), "flow": eval_flow(), "pose": eval_pose()}

    d, f, p = results["depth"], results["flow"], results["pose"]
    print("\n[深度] AbsRel(align前)={:.4f} → (align後)={:.4f}, RMSE={:.4f}, δ<1.25={:.4f}".format(
        d["abs_rel_before_align"], d["abs_rel"], d["rmse"], d["delta_1.25"]))
    print(f"  {d['note']}")
    print(f"\n[フロー] EPE={f['epe']:.4f}px（完全一致なら {f['epe_perfect']:.4f}）")
    print(f"  {f['note']}")
    print("\n[姿勢] OKS={:.4f}（完全={:.4f}）, PCK@0.2={:.4f}".format(
        p["oks"], p["oks_perfect"], p["pck@0.2"]))
    print(f"  {p['note']}")

    render_table(results, out / "04_metrics_table.png")
    (out / "04_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out/'04_metrics_table.png'}, {out/'04_metrics.json'}")


if __name__ == "__main__":
    main()
