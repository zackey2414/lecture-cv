"""02: GrabCut（矩形指定）で前景を抽出し、IoU / Dice で精度を測る（完成物）。

学ぶこと:
  - cv2.grabCut(GC_INIT_WITH_RECT) は「矩形の外は確実に背景、内側は前景かも」
    という弱い事前知識だけから、色のガウス混合モデル(GMM)とグラフカットで
    前景/背景を反復推定する半自動セグメンテーション。
  - 出力 mask は 4 値: GC_BGD=0 / GC_FGD=1 / GC_PR_BGD=2 / GC_PR_FGD=3。
    前景マスクは「FGD または PR_FGD」を 1 にして作る。
  - 初期化（矩形の置き方）に強く依存することを、3 通りの矩形で IoU/Dice が
    どう変わるかを測って体感する＝古典手法の「敏感さ」の体系化:
      good : 物体をちょうど囲む → ほぼ完璧
      loose: 画面いっぱい → 同色の「おとり(distractor)」まで前景に拾う(FP)
      clip : 物体を切ってしまう → 矩形外は「確実に背景」扱いで欠ける(FN)
  - GC_INIT_WITH_MASK で前景/背景の「なすり書き(scribble)」を足して loose の
    失敗を直す（対話的セグメンテーションの心臓部）。

評価は自作の混同行列から:
  IoU  = TP / (TP + FP + FN)        （交差 ÷ 和）
  Dice = 2*TP / (2*TP + FP + FN)    （F1 と同値。小領域に少し優しい）
合成画像なので「真の前景マスク」が分かる → 教師あり評価ができる。

実行:
  uv run python lectures/08_classical_segmentation/02_grabcut.py
結果は outputs/08_classical_segmentation/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "08_classical_segmentation"
OUT_DIR = os.path.join("outputs", MODULE_ID)


def panel(img: np.ndarray, title: str, size: tuple[int, int] = (340, 270)) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w, h = size
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# =====================================================================
# サンプル生成（前景オブジェクト + 同色のおとり + 真の前景マスク）
# =====================================================================

def make_scene_and_truth() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """背景の上に前景(オレンジの塊)と、紛らわしい同色のおとりを置いた画像を返す。

    戻り値: (画像, 真マスク0/255, 前景のタイト外接矩形 (x,y,w,h))。
    おとりは truth に含めない → loose な矩形がこれを拾うと FP になり IoU が落ちる。
    """
    h, w = 300, 400
    rng = np.random.default_rng(3)
    # 背景: 青緑のグラデーション + ざらつき（前景の色とは離す）
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.zeros((h, w, 3), np.uint8)
    bg[..., 0] = (120 + xx / w * 80).astype(np.uint8)  # B
    bg[..., 1] = (90 + yy / h * 70).astype(np.uint8)   # G
    bg[..., 2] = 40                                     # R 低め（前景の R 高めと分離）
    bg = np.clip(bg.astype(np.float32) + rng.normal(0, 8, bg.shape), 0, 255).astype(np.uint8)

    # 前景: オレンジ(B 低, G 中, R 高)の楕円 + 突起の小円。内部にも軽いテクスチャ。
    truth = np.zeros((h, w), np.uint8)
    cv2.ellipse(truth, (190, 150), (85, 62), 20, 0, 360, 255, -1)
    cv2.circle(truth, (255, 120), 30, 255, -1)

    img = bg.copy()
    fg_color = np.array([40, 120, 235], np.float32)  # BGR のオレンジ
    fg_tex = np.clip(fg_color + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8)
    img[truth == 255] = fg_tex[truth == 255]

    # おとり(distractor): 前景と同じ色だが truth ではない小円（右下）。
    distractor = np.zeros((h, w), np.uint8)
    cv2.circle(distractor, (350, 250), 22, 255, -1)
    img[distractor == 255] = fg_tex[distractor == 255]

    x, y, bw, bh = cv2.boundingRect(truth)  # 前景のタイト外接矩形
    return img, truth, (x, y, bw, bh)


# =====================================================================
# GrabCut と評価
# =====================================================================

def run_grabcut_rect(img: np.ndarray, rect: tuple[int, int, int, int], iters: int = 5) -> np.ndarray:
    """矩形初期化の GrabCut を実行し、前景マスク(0/255)を返す。"""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)  # 背景 GMM の作業領域（中身は触らない）
    fgd = np.zeros((1, 65), np.float64)  # 前景 GMM の作業領域
    cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    # FGD(=1) と PR_FGD(=3) を前景とする
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def confusion(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    """二値マスク同士の混同行列 (TP, FP, FN, TN) を返す。"""
    p = pred > 0
    t = truth > 0
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    tn = int(np.sum(~p & ~t))
    return tp, fp, fn, tn


def iou_dice(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """混同行列から IoU と Dice を計算する。"""
    tp, fp, fn, _ = confusion(pred, truth)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return iou, dice


def draw_rect(img: np.ndarray, rect: tuple[int, int, int, int], color, label: str) -> None:
    """画像に矩形とラベルを描く（可視化用、in-place）。"""
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x + 2, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                cv2.LINE_AA)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    img, truth, (x, y, bw, bh) = make_scene_and_truth()
    H, W = img.shape[:2]

    # --- 初期化への敏感性: 3 通りの矩形 ---------------------------------
    good = (x - 10, y - 10, bw + 20, bh + 20)         # 物体をちょうど囲む
    loose = (6, 6, W - 12, H - 12)                    # 画面いっぱい（おとりも含む）
    clip = (x + 25, y + 10, bw - 50, bh - 15)         # 物体を切ってしまう（小さすぎ）

    fg_good = run_grabcut_rect(img, good)
    fg_loose = run_grabcut_rect(img, loose)
    fg_clip = run_grabcut_rect(img, clip)
    iou_g, dice_g = iou_dice(fg_good, truth)
    iou_l, dice_l = iou_dice(fg_loose, truth)
    iou_c, dice_c = iou_dice(fg_clip, truth)

    # --- loose の失敗をなすり書き(GC_INIT_WITH_MASK)で直す --------------
    #   loose 結果を初期マスクにし、「物体中心=確実に前景」「おとり=確実に背景」を教える。
    refined_mask = np.where(fg_loose > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    cv2.circle(refined_mask, (190, 150), 6, cv2.GC_FGD, -1)   # 物体中心は前景
    cv2.circle(refined_mask, (350, 250), 10, cv2.GC_BGD, -1)  # おとりは背景
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, refined_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    fg_refined = np.where((refined_mask == cv2.GC_FGD) | (refined_mask == cv2.GC_PR_FGD),
                          255, 0).astype(np.uint8)
    iou_r, dice_r = iou_dice(fg_refined, truth)

    # --- 切り出し（good の前景だけ残し背景を黒に）= deliverable ----------
    cutout = cv2.bitwise_and(img, img, mask=fg_good)
    cv2.imwrite(os.path.join(OUT_DIR, "02_grabcut_cutout.png"), cutout)

    # --- 可視化 ---------------------------------------------------------
    viz_rect = img.copy()
    draw_rect(viz_rect, good, (0, 255, 0), "good")
    draw_rect(viz_rect, loose, (255, 0, 255), "loose")
    draw_rect(viz_rect, clip, (255, 255, 0), "clip")

    sheet = grid(
        [
            panel(viz_rect, "1. input + 3 rects"),
            panel(truth, "2. ground-truth mask"),
            panel(cutout, "3. cutout (good rect)"),
            panel(fg_good, f"good  IoU={iou_g:.2f}"),
            panel(fg_loose, f"loose IoU={iou_l:.2f} (grabs decoy)"),
            panel(fg_clip, f"clip  IoU={iou_c:.2f} (cuts object)"),
        ],
        3,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "02_grabcut_pipeline.png"), sheet)

    # --- IoU/Dice の棒グラフ（matplotlib。ラベルは英語=フォント問題回避）--
    labels = ["good", "loose", "clip", "loose+scribble"]
    ious = [iou_g, iou_l, iou_c, iou_r]
    dices = [dice_g, dice_l, dice_c, dice_r]
    pos = np.arange(len(labels))
    plt.figure(figsize=(6.6, 3.8))
    plt.bar(pos - 0.2, ious, width=0.4, label="IoU")
    plt.bar(pos + 0.2, dices, width=0.4, label="Dice")
    plt.xticks(pos, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("score (1.0 = perfect)")
    plt.title("GrabCut: sensitivity to rect init & scribble refinement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_grabcut_scores.png"), dpi=110)
    plt.close()

    # --- ログ -----------------------------------------------------------
    print("=== GrabCut（矩形初期化）と IoU/Dice 評価 ===")
    tp, fp, fn, tn = confusion(fg_good, truth)
    print(f"  混同行列(good): TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  good           : IoU={iou_g:.3f}  Dice={dice_g:.3f}")
    print(f"  loose          : IoU={iou_l:.3f}  Dice={dice_l:.3f}  ← 同色のおとりを拾い FP↑")
    print(f"  clip           : IoU={iou_c:.3f}  Dice={dice_c:.3f}  ← 物体を矩形で切り FN↑")
    print(f"  loose+scribble : IoU={iou_r:.3f}  Dice={dice_r:.3f}  ← なすり書きで対話的に改善")
    print("\n完了。outputs/08_classical_segmentation/ の 02_*.png を確認してください。")
    print("  02_grabcut_pipeline.png に工程、02_grabcut_scores.png に IoU/Dice の比較。")


if __name__ == "__main__":
    main()
