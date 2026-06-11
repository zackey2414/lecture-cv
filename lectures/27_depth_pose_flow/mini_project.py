"""章末ミニプロジェクト: 合成クリップに「深度 → 姿勢 → フロー」を一気通貫で適用する。

この回の3要素を1本のパイプラインへ統合し、1枚の総合レポート(PNG モンタージュ + JSON)を出す。

  ステージ1【深度】   代表フレームを Depth Anything V2 で相対深度化し、カラーマップで可視化。
  ステージ2【姿勢】   keypoint R-CNN で人体17点を推定（合成図形は検出されにくいので、
                      検出が無ければ合成 GT 関節でスケルトン描画にフォールバック）。
  ステージ3【フロー】 連続2フレームを RAFT(small) で密フロー化し、矢印で重ね描き。
                      合成の真フロー(GT)に対する EPE も算出。

どのモデルが落ちても（ネット不通など）パイプラインは止めず、できた範囲でレポートを出す
（必ず exit 0）。

実行: uv run python lectures/27_depth_pose_flow/mini_project.py
出力: outputs/27_depth_pose_flow/mini_montage.png / mini_report.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import dpf_helpers as H  # noqa: E402

# 各スクリプトの実装を再利用（重複実装を避ける）
import importlib  # noqa: E402

depth_mod = importlib.import_module("01_depth_anything")
pose_mod = importlib.import_module("02_pose_keypoints")
flow_mod = importlib.import_module("03_raft_optical_flow")


def build_clip() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """奥行き背景の上を人型が右へ歩く 2 フレームのクリップを合成する。

    返り値: (frames[rgb,...], gt_keypoints[...], gt_flow) 。
    人型の水平移動 shift をそのまま全画素フローの真値に使う（背景は固定なので近似 GT）。
    """
    bg, _ = H.make_depth_scene(h=240, w=320, seed=5)
    humans = H.make_humanoid_frames(2, h=180, w=120, seed=6)
    shift = 18  # 2 フレーム目で人型を右へずらす量（= 近似的な真フロー dx）

    y_off = 50
    frames, kps = [], []
    # 背景は固定なので真フローは 0。動くのは人型領域だけ → そこだけ dx=shift にする。
    gt_flow = np.zeros((bg.shape[0], bg.shape[1], 2), dtype=np.float32)
    for i, (fig, kp) in enumerate(humans):
        canvas = bg.copy()
        x_off = 40 + i * shift
        fh, fw = fig.shape[:2]
        # 人型を背景に貼り込み（235 を背景とみなして単純合成）
        region = canvas[y_off:y_off + fh, x_off:x_off + fw]
        mask = np.any(fig < 220, axis=2)
        region[mask] = fig[mask]
        canvas[y_off:y_off + fh, x_off:x_off + fw] = region
        frames.append(canvas)
        kp2 = kp.copy()
        kp2[:, 0] += x_off  # キーポイント座標も貼り付け位置へオフセット
        kp2[:, 1] += y_off
        kps.append(kp2)
        if i == 0:
            # フレーム0 の人型画素にだけ「右へ shift」の真フローを置く（背景は 0 のまま）
            gf = gt_flow[y_off:y_off + fh, x_off:x_off + fw]
            gf[mask, 0] = shift
    return frames, kps, gt_flow


def main() -> None:
    out = H.ensure_output_dir()
    H.set_seed(7)
    print("=== ミニプロジェクト: 深度→姿勢→フロー 統合パイプライン ===")

    frames, gt_kps, gt_flow = build_clip()
    report: dict = {}

    # --- ステージ1: 深度（代表フレーム = 1枚目）---
    pred_inv, src = depth_mod.run_depth_anything(frames[0])
    depth_vis = H.colorize_depth(pred_inv)
    report["depth_source"] = src
    print(f"[深度] source={src}")

    # --- ステージ2: 姿勢（keypoint R-CNN, 失敗/未検出なら GT 関節）---
    model = pose_mod.load_model()
    kp_used = "ground-truth(fallback)"
    kp_draw = gt_kps[0]
    if model is not None:
        pred_kp, score = pose_mod.infer_keypoints(model, frames[0])
        if pred_kp is not None:
            kp_draw = pred_kp
            kp_used = f"model(score={score:.2f})"
    pose_vis = H.draw_skeleton(frames[0], kp_draw)
    report["pose_source"] = kp_used
    # OKS（合成 GT に対する一致度。検出があれば実測、無ければ GT 同士=1.0 の参照）
    xy = gt_kps[0][:, :2]
    area = float((xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min()))
    report["pose_oks_vs_gt"] = H.object_keypoint_similarity(kp_draw[:, :2], gt_kps[0], area=area)
    print(f"[姿勢] source={kp_used}, OKS(vs GT)={report['pose_oks_vs_gt']:.3f}")

    # --- ステージ3: フロー（2フレーム間, RAFT）---
    flow, fsrc = flow_mod.raft_flow(frames[0], frames[1])
    if flow is not None:
        flow_overlay = H.draw_flow_arrows(frames[0], flow, step=20)
        report["flow_source"] = fsrc
        report["flow_epe_vs_gt"] = H.endpoint_error(flow, gt_flow)
        print(f"[フロー] source={fsrc}, EPE(vs 近似GT)={report['flow_epe_vs_gt']:.3f}px")
    else:
        flow_overlay = frames[0].copy()
        report["flow_source"] = "skip"
        print("[フロー] スキップ（重み DL 不可）")

    # --- モンタージュ: 入力 / 深度 / 姿勢 / フロー を 2x2 で1枚に ---
    titles = ["input frame0", f"depth ({src})", f"pose ({kp_used})", f"flow ({report['flow_source']})"]
    imgs = [frames[0], depth_vis, pose_vis, flow_overlay]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, im, ti in zip(axes.ravel(), imgs, titles):
        ax.imshow(im)
        ax.set_title(ti, fontsize=11)
        ax.axis("off")
    fig.suptitle("27_depth_pose_flow : integrated mini-project", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "mini_montage.png", dpi=110)
    plt.close(fig)

    (out / "mini_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out/'mini_montage.png'}, {out/'mini_report.json'}")
    print("完成: 1 本の合成クリップから 深度・姿勢・フロー を取り出し統合レポート化できた。")


if __name__ == "__main__":
    main()
