"""02: エッジ検出 — Sobel / Laplacian / Canny と「CV_64F + convertScaleAbs」の理由。

学ぶこと:
  - エッジ = 輝度が急に変わる場所。微分（勾配）の大きいところを拾うのが基本。
  - Sobel は1次微分（勾配）。出力 dtype を CV_8U にすると「負の勾配」が 0 に潰れて
    エッジの半分が消える。だから CV_64F で計算 → convertScaleAbs で絶対値→8bit化、が定石。
  - Laplacian は2次微分。convertScaleAbs で見えるようにする。
  - Canny は多段アルゴリズム（平滑化→勾配→非最大抑制→ヒステリシス閾値）。
    2つのしきい値の決め方、前段の GaussianBlur でノイズ由来の偽エッジが減ることを確認する。

実行:
  uv run python lectures/04_filtering_edges_morphology/02_edges_canny.py
結果は outputs/04_filtering_edges_morphology/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

MODULE_ID = "04_filtering_edges_morphology"
OUT_DIR = os.path.join("outputs", MODULE_ID)


def panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 270)) -> np.ndarray:
    """画像をタイトル付きパネルにする（グレーは3ch化）。"""
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


def make_gray_shapes(h: int = 270, w: int = 360) -> np.ndarray:
    """エッジの豊富なグレースケール画像を合成する。"""
    img = np.full((h, w), 90, dtype=np.uint8)  # 中間グレー背景
    cv2.rectangle(img, (30, 40), (150, 180), 230, -1)  # 明るい矩形
    cv2.circle(img, (260, 120), 55, 30, -1)  # 暗い円
    cv2.line(img, (20, 230), (340, 230), 255, 2, cv2.LINE_AA)
    cv2.putText(img, "EDGE", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    return img


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # === 1. CV_8U の落とし穴を「両方向のエッジ」で見せる ==================
    # 左:黒 中:白 右:黒 の縦縞 → 左の境界は黒→白(正の勾配)、右の境界は白→黒(負の勾配)。
    step = np.zeros((120, 300), dtype=np.uint8)
    step[:, 100:200] = 255
    # CV_8U で Sobel すると、負の勾配（白→黒の右境界）が 0 にクリップされ片側しか出ない。
    sob_8u = cv2.Sobel(step, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
    # CV_64F で計算してから絶対値→8bit。両方向のエッジが残る。これが正しい手順。
    sob_64f = cv2.Sobel(step, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sob_64f_abs = cv2.convertScaleAbs(sob_64f)  # |値| を 0..255 にスケールして可視化

    print("[1] Sobel の出力 dtype による違い（縦の白帯=両側にエッジ）")
    print(f"  CV_8U  の非ゼロ画素数: {int(np.count_nonzero(sob_8u)):5d}  ← 片側しか出ない")
    print(f"  CV_64F の非ゼロ画素数: {int(np.count_nonzero(sob_64f_abs)):5d}  ← 両側のエッジが出る")

    pit = [
        panel(step, "input (black/white/black)"),
        panel(sob_8u, "Sobel CV_8U (loses one edge)"),
        panel(sob_64f_abs, "CV_64F + convertScaleAbs"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "02_sobel_dtype_pitfall.png"), grid(pit, 3))

    # === 2. Sobel x / y / 勾配の大きさ・Laplacian ========================
    gray = make_gray_shapes()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 横方向の勾配（縦エッジを拾う）
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 縦方向の勾配（横エッジを拾う）
    mag = cv2.magnitude(gx, gy)  # sqrt(gx^2+gy^2)。勾配の強さ
    mag_u8 = cv2.convertScaleAbs(mag)
    # Laplacian は2次微分。ノイズに敏感なので普通は事前に少しぼかす。
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F)
    lap_u8 = cv2.convertScaleAbs(lap)

    print("\n[2] Sobel(x,y)→magnitude と Laplacian を可視化")

    sob_panels = [
        panel(gray, "gray"),
        panel(cv2.convertScaleAbs(gx), "Sobel x (|gx|)"),
        panel(cv2.convertScaleAbs(gy), "Sobel y (|gy|)"),
        panel(mag_u8, "gradient magnitude"),
        panel(lap_u8, "Laplacian"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "02_sobel_laplacian.png"), grid(sob_panels, 3))

    # === 3. Canny — しきい値と前段ぼかしの効果 ===========================
    # ノイズを乗せた画像でやると「ノイズ由来の偽エッジ」がよく分かる。
    noisy = gray.astype(np.float32) + np.random.default_rng(2).normal(0, 18, gray.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    canny_raw = cv2.Canny(noisy, 50, 150)  # ぼかさずに Canny → ノイズの粒も拾う
    blurred = cv2.GaussianBlur(noisy, (5, 5), 0)
    canny_blur = cv2.Canny(blurred, 50, 150)  # 事前ぼかし → 偽エッジが減る
    canny_low = cv2.Canny(blurred, 20, 60)  # しきい値が低い → エッジが増える（細かい）
    canny_high = cv2.Canny(blurred, 120, 220)  # しきい値が高い → 強いエッジだけ残る

    print("\n[3] Canny: 前段ぼかしと2つのしきい値の効き方")
    print(f"  ぼかし無し(50,150) のエッジ画素: {int(np.count_nonzero(canny_raw)):5d}")
    print(f"  ぼかし有り(50,150) のエッジ画素: {int(np.count_nonzero(canny_blur)):5d}  ← 偽エッジ減")
    print(f"  低しきい値(20,60)              : {int(np.count_nonzero(canny_low)):5d}  ← 増える")
    print(f"  高しきい値(120,220)            : {int(np.count_nonzero(canny_high)):5d}  ← 減る")

    canny_panels = [
        panel(noisy, "noisy input"),
        panel(canny_raw, "Canny no blur (50,150)"),
        panel(canny_blur, "Canny + blur (50,150)"),
        panel(canny_low, "Canny (20,60) more"),
        panel(canny_high, "Canny (120,220) fewer"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "02_canny_thresholds.png"), grid(canny_panels, 3))

    print("\n完了。outputs/04_filtering_edges_morphology/ の 02_*.png を確認してください。")


if __name__ == "__main__":
    main()
