"""mini_project.py — 章末ミニプロジェクト: 「撮影シーンを切り出し・数え・清掃する」統合パイプライン。

この回の 3 手法（Watershed / GrabCut / 古典 inpaint）と 2 系統の評価指標
（IoU・Dice と PSNR・SSIM）を、1 本のワークフローへ統合する「完成形」:

    机の上に「触れ合った硬貨の山」と「不要なシミ(汚れ)」が写った合成シーンを作る。
    （硬貨領域の真マスク・真の枚数・シミのマスク・シミの無いキレイな参照画像も保持）
      → ① 前景抽出: GrabCut(GC_INIT_WITH_RECT) で硬貨の山を切り出し、
                     真マスクとの IoU / Dice で精度を測る。
      → ② 計数:     Otsu 二値化 + distanceTransform + マーカ制御 Watershed で
                     接触した硬貨を 1 枚ずつに分離して数え、単純連結成分と対比する。
      → ③ 清掃:     シミのマスクを使い cv2.inpaint(TELEA / NS) で汚れを消し、
                     キレイな参照画像との PSNR / SSIM で「どこまで直せたか」を測る。
      → ④ 可視化:   各段の工程図・まとめ図・数値ログ(JSON)を outputs/ に保存する。

外部データもネットも GPU も不要（合成画像のみ・CPU 完結）。digit 始まりの 01〜03 は
import できないため、必要な処理（PSNR/SSIM・パネル合成など）はこのファイル内に自己完結で書く。

実行:
    uv run python lectures/08_classical_segmentation/mini_project.py
結果（画像・JSON）は outputs/08_classical_segmentation/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg を指定する（鉄則）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "08_classical_segmentation"
OUT_DIR = os.path.join("outputs", MODULE_ID)


# =====================================================================
# 可視化ヘルパ（このモジュール内で自己完結。digit 始まりスクリプトは import しない）
# =====================================================================

def panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 252)) -> np.ndarray:
    """1枚の画像にタイトル帯を付けて固定サイズへ整える（グレーは BGR 化）。"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w, h = size
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    """パネルを cols 列のグリッドへ連結する（足りない枠は黒で埋める）。"""
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# =====================================================================
# 評価指標（PSNR / SSIM を numpy/cv2 だけで自前実装。03 と同じ定義）
# =====================================================================

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """ピーク信号対雑音比[dB]。大きいほど元画像に近い。"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """構造的類似度（グレースケール、Wang らのガウス窓版）。1.0 で完全一致。"""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64) if a.ndim == 3 else a.astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64) if b.ndim == 3 else b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    var_b = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    cov = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    num = (2 * mu_ab + c1) * (2 * cov + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def confusion(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    """二値マスク同士の混同行列 (TP, FP, FN, TN) を返す。"""
    p, t = pred > 0, truth > 0
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


# =====================================================================
# 課題シーン生成（硬貨の山 + 不要なシミ。真マスク・真枚数・キレイ参照も返す）
# =====================================================================

def make_scene() -> dict:
    """撮影シーンを合成する。

    戻り値の dict:
      damaged    : 実際の入力（硬貨の山 + シミ）BGR
      clean      : シミの無いキレイな参照（inpaint の正解）BGR
      coins_truth: 硬貨領域の真マスク 0/255（GrabCut 評価の正解）
      stain_mask : シミの欠損マスク 0/255（inpaint の対象）
      true_count : 硬貨の真の枚数
      coins_rect : 硬貨群を囲むタイト外接矩形 (x,y,w,h)（GrabCut 初期化用）
    """
    h, w = 420, 600
    rng = np.random.default_rng(8)

    # --- 背景: 冷たい色のほぼ一様な暗い面（前景の硬貨と色を離す） ---
    #   GrabCut は矩形外の画素だけで背景の色モデル(GMM)を作るため、矩形内の背景も
    #   同じ色分布なら「背景」と正しく分類できる。逆に背景が大きく変化（強い
    #   グラデ）すると矩形内の隙間まで前景に拾われ FP が増える。だから背景は
    #   弱いグラデ＋低ノイズの「ほぼ一様」にして、硬貨の隙間が前景化しないようにする。
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.zeros((h, w, 3), np.uint8)
    bg[..., 0] = (62 + xx / w * 16).astype(np.uint8)   # B: 右へ僅かに明るく
    bg[..., 1] = (48 + yy / h * 14).astype(np.uint8)   # G: 下へ僅かに明るく
    bg[..., 2] = 30                                     # R: 低く固定（硬貨の暖色と分離）
    bg = np.clip(bg.astype(np.float32) + rng.normal(0, 4, bg.shape), 0, 255).astype(np.uint8)

    # --- 硬貨: 接触ペア + 孤立。(cx, cy, r)。半径和より少し近づけて「触れ合う」状態に ---
    coins = [
        (140, 175, 52), (232, 192, 50),   # クラスタ1（2枚が接触）
        (392, 168, 54), (486, 198, 50),   # クラスタ2（2枚が接触）
        (172, 332, 52),                   # 孤立
        (452, 338, 54),                   # 孤立
    ]
    img = bg.copy()
    coins_truth = np.zeros((h, w), np.uint8)
    for cx, cy, r in coins:
        color = tuple(int(c) for c in rng.integers(165, 240, size=3))  # 明るい暖色寄り
        cv2.circle(img, (cx, cy), r, color, -1)
        cv2.circle(coins_truth, (cx, cy), r, 255, -1)
    # 硬貨内部に軽いテクスチャ（GrabCut の色モデルを現実的にする）
    tex = np.clip(img.astype(np.float32) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    img[coins_truth == 255] = tex[coins_truth == 255]

    clean = img.copy()  # ← シミを足す前の状態が inpaint の「正解」

    # --- 不要なシミ(汚れ): 硬貨の無い上部の帯に配置（除去で背景が綺麗に戻るように） ---
    stain_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(stain_mask, (110, 60), (46, 26), 25, 0, 360, 255, -1)  # 楕円のシミ
    cv2.line(stain_mask, (300, 40), (470, 80), 255, 7)                 # 細い擦り傷
    damaged = clean.copy()
    stain_color = np.array([95, 95, 95], np.float32)  # 中間グレー（Otsu の硬貨閾値より暗い）
    stain_tex = np.clip(stain_color + rng.normal(0, 10, (h, w, 3)), 0, 255).astype(np.uint8)
    damaged[stain_mask > 0] = stain_tex[stain_mask > 0]

    x, y, bw, bh = cv2.boundingRect(coins_truth)
    return {
        "damaged": damaged,
        "clean": clean,
        "coins_truth": coins_truth,
        "stain_mask": stain_mask,
        "true_count": len(coins),
        "coins_rect": (x - 8, y - 8, bw + 16, bh + 16),
    }


# =====================================================================
# ① 前景抽出（GrabCut）
# =====================================================================

def grabcut_foreground(img: np.ndarray, rect: tuple[int, int, int, int], iters: int = 5) -> np.ndarray:
    """矩形初期化の GrabCut で前景マスク(0/255)を返す。"""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)  # 背景 GMM の作業領域（形・dtype 固定）
    fgd = np.zeros((1, 65), np.float64)  # 前景 GMM の作業領域
    cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


# =====================================================================
# ② 計数（マーカ制御 Watershed）
# =====================================================================

def count_with_watershed(img: np.ndarray) -> dict:
    """Otsu→距離変換→マーカ→Watershed で接触硬貨を分離して数える。

    戻り値: naive_count（単純連結成分）, ws_count（Watershed）, markers, boundary。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                 # 背景を 1 にずらす（0 は unknown 用）
    markers[unknown == 255] = 0
    cv2.watershed(img, markers)           # in-place 更新。境界に -1

    labels = set(np.unique(markers).tolist()) - {-1, 1}
    ws_count = len(labels)
    naive_n, _ = cv2.connectedComponents(opening)
    naive_count = naive_n - 1             # ラベル0=背景を除く

    boundary = img.copy()
    boundary[markers == -1] = (0, 0, 255)
    return {
        "naive_count": naive_count,
        "ws_count": ws_count,
        "markers": markers,
        "boundary": boundary,
        "labels": sorted(labels),
    }


def colorize_regions(markers: np.ndarray, labels: list[int]) -> np.ndarray:
    """Watershed の各領域へランダム色を塗った可視化画像を作る（背景・境界は黒）。"""
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, size=(len(labels) + 1, 3), dtype=np.uint8)
    seg = np.zeros((*markers.shape, 3), np.uint8)
    for i, lab in enumerate(labels):
        seg[markers == lab] = colors[i].tolist()
    return seg


# =====================================================================
# ③ 清掃（古典 inpaint）
# =====================================================================

def clean_stain(damaged: np.ndarray, stain_mask: np.ndarray) -> dict:
    """シミのマスクを使い TELEA / NS で復元し、結果一式を返す。"""
    telea = cv2.inpaint(damaged, stain_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    ns = cv2.inpaint(damaged, stain_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return {"telea": telea, "ns": ns}


# =====================================================================
# メイン
# =====================================================================

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    scene = make_scene()
    damaged = scene["damaged"]
    clean = scene["clean"]
    coins_truth = scene["coins_truth"]
    stain_mask = scene["stain_mask"]
    true_count = scene["true_count"]
    rect = scene["coins_rect"]
    H, W = damaged.shape[:2]

    print("=" * 64)
    print("章末ミニプロジェクト: シーンを切り出し・数え・清掃する統合パイプライン")
    print(f"  シーン={W}x{H}px / 硬貨の真の枚数={true_count}")

    # === ① 前景抽出（GrabCut）＋ IoU/Dice 評価 ===========================
    fg = grabcut_foreground(damaged, rect)
    iou, dice = iou_dice(fg, coins_truth)
    tp, fp, fn, _ = confusion(fg, coins_truth)
    cutout = cv2.bitwise_and(damaged, damaged, mask=fg)
    grabcut_ok = bool(iou > 0.85)
    print("\n  [①前景抽出 GrabCut]")
    print(f"    混同行列: TP={tp} FP={fp} FN={fn}")
    print(f"    IoU={iou:.3f}  Dice={dice:.3f}  判定={'OK' if grabcut_ok else 'NG'}")

    # === ② 計数（Watershed）============================================
    wd = count_with_watershed(damaged)
    count_ok = bool(wd["ws_count"] == true_count)
    print("\n  [②計数 Watershed]")
    print(f"    単純連結成分 = {wd['naive_count']}  ← 接触ペアが融合して少なく出る")
    print(f"    Watershed    = {wd['ws_count']}  (真値 {true_count})  判定={'OK' if count_ok else 'NG'}")

    # === ③ 清掃（inpaint）＋ PSNR/SSIM 評価 =============================
    rest = clean_stain(damaged, stain_mask)
    psnr_broken = psnr(clean, damaged)
    psnr_telea = psnr(clean, rest["telea"])
    psnr_ns = psnr(clean, rest["ns"])
    ssim_telea = ssim(clean, rest["telea"])
    ssim_ns = ssim(clean, rest["ns"])
    inpaint_ok = bool(psnr_telea > psnr_broken + 5.0)
    print("\n  [③清掃 inpaint]")
    print(f"    破損時 PSNR        : {psnr_broken:.2f} dB")
    print(f"    TELEA PSNR / SSIM  : {psnr_telea:.2f} dB / {ssim_telea:.3f}")
    print(f"    NS    PSNR / SSIM  : {psnr_ns:.2f} dB / {ssim_ns:.3f}")
    print(f"    → シミ除去で PSNR が回復  判定={'OK' if inpaint_ok else 'NG'}")

    # === ④ 可視化（工程図 3 枚 + まとめ図 + JSON）=======================
    # ①前景抽出の工程図
    viz_rect = damaged.copy()
    x, y, bw, bh = rect
    cv2.rectangle(viz_rect, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    sheet1 = grid(
        [
            panel(viz_rect, "1. input + grabcut rect"),
            panel(coins_truth, "2. coins ground-truth"),
            panel(fg, f"3. grabcut fg  IoU={iou:.2f}"),
            panel(cutout, "4. cutout (bg removed)"),
        ],
        2,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "mini_project_grabcut.png"), sheet1)

    # ②計数の工程図
    seg = colorize_regions(wd["markers"], wd["labels"])
    sheet2 = grid(
        [
            panel(damaged, "1. input scene"),
            panel(wd["boundary"], "2. watershed boundary (red)"),
            panel(seg, f"3. separated x{wd['ws_count']} (naive={wd['naive_count']})"),
        ],
        3,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "mini_project_watershed.png"), sheet2)

    # ③清掃の工程図
    sheet3 = grid(
        [
            panel(damaged, "1. damaged (stain)"),
            panel(stain_mask, "2. stain mask"),
            panel(rest["telea"], f"3. TELEA PSNR={psnr_telea:.1f}"),
            panel(clean, "4. clean (ground truth)"),
        ],
        2,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "mini_project_inpaint.png"), sheet3)

    # まとめ図（matplotlib, Agg, BGR→RGB に注意）
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    axes[0, 0].imshow(cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("input scene (coins + stain)")
    axes[0, 1].imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"(1) grabcut cutout  IoU={iou:.2f}")
    axes[0, 2].imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"(2) watershed count = {wd['ws_count']}")
    axes[1, 0].imshow(cv2.cvtColor(rest["telea"], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"(3) inpaint TELEA  PSNR={psnr_telea:.1f}")
    axes[1, 1].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("clean reference")
    # スコアの棒グラフ
    axes[1, 2].bar([0, 1], [iou, dice], width=0.6, color=["#4C72B0", "#55A868"])
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_xticklabels(["IoU", "Dice"])
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].set_title("grabcut foreground score")
    for ax in axes.ravel()[:5]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mini_project_summary.png"), dpi=110)
    plt.close(fig)

    # 数値ログ（再現性・自動評価のため JSON 保存）
    metrics = {
        "scene": {"size": [W, H], "true_coin_count": true_count},
        "grabcut": {
            "iou": round(iou, 4), "dice": round(dice, 4),
            "tp": tp, "fp": fp, "fn": fn, "ok": grabcut_ok,
        },
        "watershed": {
            "true_count": true_count,
            "naive_connected_components": wd["naive_count"],
            "watershed_count": wd["ws_count"],
            "ok": count_ok,
        },
        "inpaint": {
            "psnr_broken": round(psnr_broken, 3),
            "psnr_telea": round(psnr_telea, 3), "ssim_telea": round(ssim_telea, 4),
            "psnr_ns": round(psnr_ns, 3), "ssim_ns": round(ssim_ns, 4),
            "ok": inpaint_ok,
        },
        "all_ok": bool(grabcut_ok and count_ok and inpaint_ok),
    }
    json_path = os.path.join(OUT_DIR, "mini_project_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n  保存物:")
    print("    - 工程図 : mini_project_grabcut.png / mini_project_watershed.png / mini_project_inpaint.png")
    print("    - まとめ図: mini_project_summary.png")
    print(f"    - 指標   : {os.path.basename(json_path)}")
    print("\n完成: 前景抽出(IoU/Dice)→計数(Watershed)→清掃(PSNR/SSIM) を外部依存ゼロで通せた。")
    print(f"  総合判定 all_ok = {metrics['all_ok']}")
    print("発展: 硬貨を実物の部品/細胞に、シミを実写の汚れに差し替えれば検査・修復の雛形になる。")


if __name__ == "__main__":
    main()
