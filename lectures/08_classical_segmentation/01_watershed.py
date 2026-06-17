"""01: 距離変換 + マーカ制御 Watershed で「接触した物体」を分離する。

学ぶこと:
  - 単純な二値化 + connectedComponents は、触れ合った物体を「1つ」に
    まとめてしまう（境界が繋がっているから）。これが古典セグメンテーションの
    最初の壁。
  - これを解くのが distanceTransform（各前景画素から最も近い背景までの距離）。
    距離のピーク = 各物体の「芯」なので、ピーク付近だけを取り出すと
    触れ合った物体でも別々の「確実な前景(sure_fg)」になる。
  - その sure_fg を connectedComponents でラベル付け → マーカにし、
    背景(sure_bg)と「どちらか分からない領域(unknown)」を区別して
    cv2.watershed に渡すと、地形を水で満たすように境界線を引いて分離してくれる。
  - watershed は markers を in-place で書き換え、境界画素に -1 を入れる。

接触物体の分離は、細胞・硬貨・粒など「くっついた同種物体を数える」古典タスクの
定番。深層の SAM(20回) に渡る前に「距離とマーカで人手の事前知識を与える」感覚を掴む。

実行:
  uv run python lectures/08_classical_segmentation/01_watershed.py
結果は lectures/08_classical_segmentation/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np

# matplotlib は GUI を持たない Agg バックエンドで使う（Docker/CI/SSH でも動く）。
# pyplot を import する前にバックエンドを固定するのが鉄則。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "08_classical_segmentation"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


# =====================================================================
# 可視化ヘルパ（このモジュール内で自己完結。外部 import はしない）
# =====================================================================

def panel(img: np.ndarray, title: str, size: tuple[int, int] = (340, 270)) -> np.ndarray:
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
# サンプル生成（触れ合った「硬貨」を合成。真の個数も返す）
# =====================================================================

def make_touching_coins() -> tuple[np.ndarray, int]:
    """暗い背景に、いくつか触れ合う明るい円（硬貨）を描いた BGR 画像を作る。"""
    h, w = 360, 520
    img = np.full((h, w, 3), 28, dtype=np.uint8)  # 暗い背景
    rng = np.random.default_rng(8)
    # (cx, cy, r): ペアは半径和より少しだけ近く配置し「触れ合う(細い首で繋がる)」状態にする。
    # こうすると二値画像では1塊に繋がる（naive CC は融合）が、距離変換の谷は深いので
    # watershed なら分離できる、という対比をきれいに作れる。
    coins = [
        (120, 150, 50), (206, 166, 48),   # クラスタ1（2枚が接触）
        (330, 150, 52), (418, 178, 48),   # クラスタ2（2枚が接触）
        (150, 295, 50),                   # 孤立
        (385, 300, 52),                   # 孤立
    ]
    for cx, cy, r in coins:
        color = tuple(int(c) for c in rng.integers(150, 235, size=3))  # 明るめのランダム色
        cv2.circle(img, (cx, cy), r, color, -1)
    # わずかな撮影ノイズ（二値化が壊れない程度）
    noise = rng.normal(0, 5, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, len(coins)


# =====================================================================
# Watershed パイプライン
# =====================================================================

def count_labels(markers: np.ndarray) -> int:
    """watershed 後の markers から物体数を数える（背景=1 と 境界=-1 を除く）。"""
    labels = set(np.unique(markers).tolist())
    labels.discard(-1)  # 境界
    labels.discard(1)   # 背景
    return len(labels)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    img, true_count = make_touching_coins()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. 二値化（Otsu）。硬貨=白・背景=黒 ----------------------------
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 2. オープニングで小さなノイズを除去 ----------------------------
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- 3. 「確実な背景(sure_bg)」: 膨張で前景を太らせ、残りを背景とみなす --
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # --- 4. 「確実な前景(sure_fg)」: 距離変換のピーク付近だけを採用 --------
    #   distanceTransform = 各白画素から最寄りの黒(背景)までの距離。
    #   触れ合った2枚でも、各芯にピークが立つので閾値で切ると別々の塊になる。
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # --- 5. 「どちらか不明な領域(unknown)」= sure_bg - sure_fg -----------
    unknown = cv2.subtract(sure_bg, sure_fg)

    # --- 6. マーカ作成 --------------------------------------------------
    #   connectedComponents で sure_fg にラベル(0=背景, 1,2,...=各芯)。
    #   +1 して背景を 1 にずらし、unknown を 0 にして「ここは未確定」と教える。
    n_seeds, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # --- 7. Watershed 実行（markers を in-place 更新。境界に -1 が入る）----
    markers_ws = markers.copy()
    cv2.watershed(img, markers_ws)
    ws_count = count_labels(markers_ws)

    # --- 比較用: 単純 connectedComponents（接触物体を1つにまとめてしまう）--
    naive_n, _ = cv2.connectedComponents(opening)
    naive_count = naive_n - 1  # ラベル0は背景なので引く

    # --- 可視化 ---------------------------------------------------------
    boundary = img.copy()
    boundary[markers_ws == -1] = (0, 0, 255)  # 境界線を赤で描く

    # 各領域に色を付けた分離結果（背景と境界は黒）
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, size=(ws_count + 2, 3), dtype=np.uint8)
    seg = np.zeros_like(img)
    for new_label, lab in enumerate(sorted(set(np.unique(markers_ws).tolist()) - {-1, 1})):
        seg[markers_ws == lab] = colors[new_label % len(colors)].tolist()

    # 距離変換のヒートマップは matplotlib で（連続値の可視化に向く）。
    plt.figure(figsize=(5, 3.5))
    plt.imshow(dist, cmap="jet")  # dist は単チャンネルなので BGR/RGB 変換は不要
    plt.colorbar(label="distance to background")
    plt.title("distanceTransform (peaks = coin centers)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_distance_transform.png"), dpi=110)
    plt.close()

    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sheet = grid(
        [
            panel(img, "1. input (touching coins)"),
            panel(thresh, "2. Otsu binary"),
            panel(dist_vis, "3. distanceTransform"),
            panel(sure_fg, "4. sure_fg (peaks)"),
            panel(boundary, "5. watershed boundary (red)"),
            panel(seg, "6. separated regions"),
        ],
        3,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "01_watershed_pipeline.png"), sheet)
    cv2.imwrite(os.path.join(OUT_DIR, "01_watershed_boundary.png"), boundary)

    # --- ログ -----------------------------------------------------------
    print("=== Watershed による接触物体の分離 ===")
    print(f"  真の硬貨枚数            : {true_count}")
    print(f"  単純 connectedComponents: {naive_count}  ← 接触ペアが融合して少なくなる")
    print(f"  watershed の領域数      : {ws_count}  ← 距離変換のマーカで分離できた")
    print(f"  sure_fg の連結成分(種)  : {n_seeds - 1}")
    print("\n完了。lectures/08_classical_segmentation/outputs/ の 01_*.png を確認してください。")
    print("  01_watershed_pipeline.png に 6 工程、01_distance_transform.png に距離のヒートマップ。")


if __name__ == "__main__":
    main()
