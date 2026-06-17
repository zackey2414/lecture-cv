"""03: 閾値処理（二値化）とモルフォロジー演算。

学ぶこと（前半: 閾値処理）:
  - threshold(固定しきい値): 照明ムラがあると一律のしきい値では破綻する。
  - THRESH_OTSU: ヒストグラムから自動でしきい値を1つ決める（画像全体で1つ＝大域的）。
  - adaptiveThreshold: 画素ごとに近傍から局所的にしきい値を決める。照明ムラに強い。
学ぶこと（後半: モルフォロジー）:
  - getStructuringElement で構造要素（カーネル）を作る（矩形/楕円/十字）。
  - erode(収縮)/dilate(膨張)、その合成である opening(雑音除去)/closing(穴埋め)/gradient/tophat。
  - 二値化 → モルフォロジーで整形、という古典CVの前処理連鎖の中核。

実行:
  uv run python lectures/04_filtering_edges_morphology/03_threshold_morphology.py
結果は lectures/04_filtering_edges_morphology/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np

MODULE_ID = "04_filtering_edges_morphology"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 270)) -> np.ndarray:
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


def make_uneven_doc(h: int = 300, w: int = 420) -> np.ndarray:
    """照明ムラのある「文書」グレースケールを合成する（右へ行くほど暗い）。"""
    img = np.full((h, w), 235, dtype=np.uint8)  # 明るい紙
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "DOCUMENT", (25, 80), f, 1.3, 20, 3, cv2.LINE_AA)  # 濃いインク
    cv2.putText(img, "binarize me", (25, 135), f, 1.0, 20, 2, cv2.LINE_AA)
    cv2.rectangle(img, (25, 170), (395, 265), 20, 2)
    cv2.putText(img, "OpenCV 4.x", (45, 230), f, 1.0, 20, 2, cv2.LINE_AA)
    # 照明ムラ: 左=明るい(1.0倍) 右=暗い(0.35倍) のグラデーションを掛ける
    xx = np.linspace(1.0, 0.35, w, dtype=np.float32)
    illum = np.repeat(xx[None, :], h, axis=0)
    return np.clip(img.astype(np.float32) * illum, 0, 255).astype(np.uint8)


def make_binary_blobs(h: int = 300, w: int = 420) -> np.ndarray:
    """前景の塊＋白い粒ノイズ＋黒い小穴を含む二値画像を合成する。"""
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(img, (100, 150), 55, 255, -1)
    cv2.circle(img, (240, 150), 55, 255, -1)
    cv2.rectangle(img, (320, 90), (400, 210), 255, -1)
    rng = np.random.default_rng(3)
    # 前景の中に黒い小穴をあける（closing で埋める対象）
    holes = rng.random((h, w))
    img[(img == 255) & (holes < 0.04)] = 0
    # 背景に白い粒ノイズをまく（opening で消す対象）
    specks = rng.random((h, w))
    img[(img == 0) & (specks < 0.01)] = 255
    return img


def count_components(binary: np.ndarray) -> int:
    """白の連結成分の個数（背景を除く）。雑音が増えると増える。"""
    n, _ = cv2.connectedComponents(binary)
    return n - 1  # ラベル0は背景


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # === 1. 閾値処理: 固定 vs Otsu vs 適応的 =============================
    doc = make_uneven_doc()
    # 固定しきい値127。INV=暗い画素(インク)を白(255)にする。右側の暗い紙まで白く潰れる。
    _, th_fixed = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY_INV)
    # Otsu: しきい値を自動決定（戻り値の第1要素が選ばれた値）。ただし画像全体で1つ＝大域的。
    otsu_t, th_otsu = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 適応的: 近傍 blockSize から局所しきい値を計算。照明ムラに強い。
    th_adapt = cv2.adaptiveThreshold(
        doc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=31, C=10
    )

    print(f"[1] Otsu が自動で選んだしきい値 = {otsu_t:.1f}")
    print("    固定/Otsu は画像全体で1つのしきい値 → 右の暗い領域で紙が白く潰れる")
    print("    適応的は近傍ごとに判断 → 照明ムラがあっても文字だけ拾える")

    th_panels = [
        panel(doc, "uneven illumination"),
        panel(th_fixed, "fixed 127 (fails right)"),
        panel(th_otsu, f"Otsu t={otsu_t:.0f} (global)"),
        panel(th_adapt, "adaptive (robust)"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "03_threshold_compare.png"), grid(th_panels, 2))

    # === 2. 構造要素（カーネル）の形 =====================================
    # 同じ 5x5 でも矩形/楕円/十字で「どの近傍を見るか」が変わる。
    k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    print("\n[2] 構造要素 5x5（1の位置が「見る近傍」）")
    print(" RECT:\n", k_rect)
    print(" ELLIPSE:\n", k_ellipse)
    print(" CROSS:\n", k_cross)

    # === 3. モルフォロジー演算 ===========================================
    blobs = make_binary_blobs()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(blobs, kernel, iterations=1)  # 白が痩せる（細い橋・粒が消える）
    dilated = cv2.dilate(blobs, kernel, iterations=1)  # 白が太る（穴が埋まる）
    opened = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)  # 収縮→膨張: 白い粒ノイズ除去
    closed = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, kernel)  # 膨張→収縮: 黒い小穴を埋める
    gradient = cv2.morphologyEx(blobs, cv2.MORPH_GRADIENT, kernel)  # dilate-erode: 輪郭線
    tophat = cv2.morphologyEx(blobs, cv2.MORPH_TOPHAT, kernel)  # input-open: 細部（粒）強調

    print("\n[3] モルフォロジーの効果（白の連結成分数で雑音除去を確認）")
    print(f"  元画像 opening前 の白成分数 : {count_components(blobs):3d}  （粒ノイズで多い）")
    print(f"  opening後 の白成分数        : {count_components(opened):3d}  ← 粒が消えて減る")

    morph_panels = [
        panel(blobs, "binary (noise+holes)"),
        panel(eroded, "erode"),
        panel(dilated, "dilate"),
        panel(opened, "open (remove specks)"),
        panel(closed, "close (fill holes)"),
        panel(gradient, "gradient (outline)"),
        panel(tophat, "tophat"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "03_morphology.png"), grid(morph_panels, 3))

    print("\n完了。lectures/04_filtering_edges_morphology/outputs/ の 03_*.png を確認してください。")


if __name__ == "__main__":
    main()
