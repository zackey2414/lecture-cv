"""01: IoU の定義と、confidence 降順による「予測→GT」貪欲マッチング（TP/FP/FN の決定）。

物体検出の評価は、分類のように「正解ラベルと予測ラベルを突き合わせる」だけでは済まない。
予測は "場所（ボックス）" を持つので、まず「その予測が、どの正解(GT)を当てたのか」を
決める対応付けが要る。その対応付けの物差しが IoU（Intersection over Union）であり、
対応付けの手続きが「confidence 降順の貪欲マッチング」である。この2つが mAP の土台になる。

学ぶこと:
  - IoU = 交差面積 / 和集合面積。numpy で一から書き、torchvision.ops.box_iou と一致を確認。
  - bbox 形式: 本講座内部は xyxy、COCO JSON は xywh。取り違えると IoU が壊れる典型バグ。
  - 貪欲マッチング: 予測をスコアの高い順に見て、IoU≥閾値の "未使用" GT のうち IoU 最大へ割当て。
        割り当てられたら TP、見つからなければ FP。割り当てられなかった GT が FN。
  - 1つの GT に複数予測が当たっても TP は1つだけ（残りは FP）。この二重カウント防止が肝。

実行: uv run python lectures/19_detection_map_from_scratch/01_iou_matching.py
結果（マッチングの可視化図・カウントの JSON）は lectures/19_detection_map_from_scratch/outputs/ に保存。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from torchvision.ops import box_iou

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from det_helpers import (  # noqa: E402
    CATEGORIES,
    IMAGE_H,
    IMAGE_W,
    box_iou_numpy,
    make_detection_dataset,
    match_image,
    output_dir,
)


def iou_pair(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """2つの箱(xyxy)の IoU を“定義式そのまま”で1つ返す。行列版の理解のための最小実装。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    # 交差矩形 = 左上は大きい方、右下は小さい方。負なら重なり無し → 0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """貪欲マッチングを“原系列の順序”で書き下す教材版（戻り値は元の予測順に対応）。

    det_helpers.match_image はスコア降順に並べた結果を返すが、ここでは可視化のために
    「元の予測index に対する TP フラグ」と「各予測が当てた GT index(無ければ -1)」を返す。
    アルゴリズムの本質は match_image と同一で、後で assert で一致を確かめる。
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    tp = np.zeros(n_pred, dtype=bool)
    matched_gt = np.full(n_pred, -1, dtype=int)
    if n_pred == 0 or n_gt == 0:
        return tp, matched_gt

    ious = box_iou_numpy(pred_boxes, gt_boxes)  # (n_pred, n_gt)
    gt_used = np.zeros(n_gt, dtype=bool)

    # 「スコアが高い予測ほど優先的に GT を取れる」ので降順に処理する（ここが confidence 依存の核心）
    for di in np.argsort(-pred_scores, kind="stable"):
        best_iou, best_gt = iou_thr, -1
        for gi in range(n_gt):
            if gt_used[gi]:
                continue  # 既に他の（より高スコアの）予測が取った GT は使えない
            if ious[di, gi] >= best_iou:
                best_iou, best_gt = ious[di, gi], gi  # 未使用 GT で IoU 最大を探す
        if best_gt >= 0:
            gt_used[best_gt] = True
            tp[di] = True
            matched_gt[di] = best_gt
    return tp, matched_gt


def demo_double_counting() -> dict:
    """『1つの GT に2つの予測が重なったら、TP は1つだけ』を最小例で確かめる。"""
    gt = np.array([[100, 100, 200, 200]], dtype=float)  # GT 1個
    # 同じ GT にほぼ重なる予測を2つ用意。スコアは pred0 > pred1
    preds = np.array([[102, 98, 198, 202], [95, 105, 205, 195]], dtype=float)
    scores = np.array([0.9, 0.8])
    tp, matched = greedy_match(preds, scores, gt, iou_thr=0.5)
    # 高スコアの pred0 が GT を取り TP、pred1 は GT が既に使われていて FP になる
    print("[二重カウント防止] 同じ GT に2予測 → "
          f"pred0(score0.9)={'TP' if tp[0] else 'FP'}, "
          f"pred1(score0.8)={'TP' if tp[1] else 'FP'}  （TP は1つだけが正しい）")
    return {"tp_flags": tp.tolist(), "matched_gt": matched.tolist()}


def render_scene(d: dict) -> np.ndarray:
    """合成シーンを RGB 画像として描く（GT オブジェクトを薄色の塗り矩形にしただけの簡易絵）。

    実画像が無くてもマッチングの様子を“絵”で確認できるようにするための飾り。
    BGR ではなく最初から RGB で作るので matplotlib にそのまま渡せる。
    """
    img = np.full((IMAGE_H, IMAGE_W, 3), 245, dtype=np.uint8)
    palette = {1: (180, 215, 255), 2: (200, 240, 200), 3: (255, 220, 190)}  # カテゴリ別の淡色
    for box, label in zip(d["gt_boxes"], d["gt_labels"]):
        x1, y1, x2, y2 = box.astype(int)
        img[y1:y2, x1:x2] = palette.get(int(label), (220, 220, 220))
    return img


def main() -> None:
    out = output_dir()

    # === 1. IoU を自作し、torchvision.ops.box_iou と一致することを確認 ============
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 400, (4, 2))
    a = np.concatenate([a, a + rng.uniform(20, 90, (4, 2))], axis=1)  # xyxy
    b = rng.uniform(0, 400, (5, 2))
    b = np.concatenate([b, b + rng.uniform(20, 90, (5, 2))], axis=1)
    iou_mine = box_iou_numpy(a, b)
    iou_tv = box_iou(torch.tensor(a), torch.tensor(b)).numpy()
    assert np.allclose(iou_mine, iou_tv, atol=1e-9), "自作 IoU が torchvision と一致しない"
    # 1ペア版も同じ値になることを確認（定義式の理解チェック）
    assert np.isclose(iou_pair(a[0], b[0]), iou_mine[0, 0])
    print("=== IoU の自作 ===")
    print(f"自作 box_iou と torchvision.ops.box_iou の最大差 = {np.abs(iou_mine - iou_tv).max():.2e}"
          "  → 完全一致")

    # === 2. 二重カウント防止の最小デモ ==========================================
    print("\n=== 貪欲マッチングの肝: 二重カウント防止 ===")
    dc = demo_double_counting()

    # === 3. 1枚の合成画像でマッチングを可視化（GT=緑, TP=青, FP=赤）===============
    dataset = make_detection_dataset(n_images=12, seed=0)
    d = dataset[0]
    iou_thr = 0.5

    # 画像内の予測を「カテゴリごと」にマッチング（カテゴリ跨ぎは決して当たらない）
    tp_all = np.zeros(len(d["pred_boxes"]), dtype=bool)
    for c in [cat["id"] for cat in CATEGORIES]:
        dm = d["pred_labels"] == c
        gm = d["gt_labels"] == c
        if dm.sum() == 0:
            continue
        tp_c, _ = greedy_match(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
        tp_all[np.where(dm)[0]] = tp_c

    # 教材版 greedy_match が正準エンジン match_image と一致することを確認（スコア降順に直して比較）
    for c in [cat["id"] for cat in CATEGORIES]:
        dm, gm = d["pred_labels"] == c, d["gt_labels"] == c
        if dm.sum() == 0:
            continue
        tp_sorted, _, _ = match_image(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
        tp_local, _ = greedy_match(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
        order = np.argsort(-d["pred_scores"][dm], kind="stable")
        assert np.array_equal(tp_sorted.astype(bool), tp_local[order]), "教材版とエンジンが不一致"

    n_tp = int(tp_all.sum())
    n_fp = int((~tp_all).sum())
    n_gt = int(len(d["gt_boxes"]))
    n_fn = n_gt - n_tp  # 当てられなかった GT
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / n_gt if n_gt else 0.0
    print(f"\n=== 画像0 のマッチング結果 (IoU≥{iou_thr}) ===")
    print(f"  GT={n_gt}  TP={n_tp}  FP={n_fp}  FN={n_fn}  "
          f"→ precision={precision:.3f}  recall={recall:.3f}")
    print("  注: ここでの precision/recall は『1閾値・1画像』の値。"
          "スコア閾値を動かすと両者は変わる → 次の 02 で PR 曲線にする。")

    # --- 可視化（GT=緑実線, TP=青, FP=赤）---
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.imshow(render_scene(d))
    for box in d["gt_boxes"]:
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="green", lw=2.2))
    for box, is_tp in zip(d["pred_boxes"], tp_all):
        x1, y1, x2, y2 = box
        color = "blue" if is_tp else "red"
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, lw=1.6, linestyle="--"))
    ax.set_title(f"GT=green / TP=blue(--) / FP=red(--)   IoU>={iou_thr}\n"
                 f"GT={n_gt} TP={n_tp} FP={n_fp} FN={n_fn}")
    ax.set_xlim(0, IMAGE_W)
    ax.set_ylim(IMAGE_H, 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "01_matching.png", dpi=120)
    plt.close(fig)

    # === JSON 保存 =============================================================
    summary = {
        "iou_max_abs_diff_vs_torchvision": float(np.abs(iou_mine - iou_tv).max()),
        "double_counting_demo": dc,
        "image0": {"n_gt": n_gt, "tp": n_tp, "fp": n_fp, "fn": n_fn,
                   "precision": precision, "recall": recall, "iou_thr": iou_thr},
    }
    (out / "01_iou_matching.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n完了。01_matching.png / 01_iou_matching.json を保存しました。")


if __name__ == "__main__":
    main()
