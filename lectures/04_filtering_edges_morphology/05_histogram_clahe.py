"""05: ヒストグラムとコントラスト補正 — equalizeHist（大域）/ CLAHE（局所）。

学ぶこと:
  - ヒストグラム = 各輝度値が何画素あるかの分布。コントラストが低い画像は分布が狭い。
  - equalizeHist: 分布を全体に引き伸ばす「大域的」平坦化。手軽だが場所ごとの濃淡差に弱い。
  - CLAHE(createCLAHE): 画像をタイルに区切って局所的に平坦化し、増幅しすぎを clipLimit で抑える。
    照明ムラのある画像で equalizeHist より自然に細部を出せる。
  - カラー画像は BGR の各チャンネルを別々に平坦化すると色が壊れる。
    正しくは HSV の V（明度）だけに適用して色相を保つ。
  - 比較用に normalize（min-max ストレッチ）も確認する。

実行:
  uv run python lectures/04_filtering_edges_morphology/05_histogram_clahe.py
結果は outputs/04_filtering_edges_morphology/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# matplotlib は「pyplot を import する前に」Agg バックエンドへ固定する（画面不要・savefig可）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "04_filtering_edges_morphology"
OUT_DIR = os.path.join("outputs", MODULE_ID)


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


def make_lowcontrast_scene(h: int = 270, w: int = 360) -> np.ndarray:
    """細部のある情景を作り、コントラストを潰して片側を暗くした低コントラスト画像を返す。"""
    rng = np.random.default_rng(5)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    g = np.linspace(20, 230, w, dtype=np.uint8)  # 横グラデーション
    img[:] = np.repeat(g[None, :], h, axis=0)[:, :, None]
    cv2.rectangle(img, (40, 40), (150, 150), (90, 70, 60), -1)
    cv2.circle(img, (290, 110), 50, (60, 120, 200), -1)
    cv2.putText(img, "CLAHE", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 2)
    # 細かいテクスチャ（CLAHE で浮き上がる細部）
    tex = rng.integers(-22, 22, (h, w, 1), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + tex, 0, 255).astype(np.uint8)
    # コントラストを潰す: 値域を狭い帯（約95〜170）に圧縮
    low = img.astype(np.float32) * 0.30 + 95.0
    # 片側を暗くする照明ムラ → 大域 equalize では取りきれない状況を作る
    xx = np.linspace(1.0, 0.6, w, dtype=np.float32)
    low = low * xx[None, :, None]
    return np.clip(low, 0, 255).astype(np.uint8)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    scene = make_lowcontrast_scene()
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # === 1. グレースケールの平坦化: equalize vs CLAHE vs normalize ========
    eq = cv2.equalizeHist(gray)  # 大域的なヒストグラム平坦化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 局所平坦化（過増幅を抑制）
    cl = clahe.apply(gray)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # 単純な min-max ストレッチ

    print("[1] 輝度の広がり（標準偏差。大きいほどコントラストが高い）")
    print(f"  低コントラスト原画 : {gray.std():6.1f}")
    print(f"  normalize          : {norm.std():6.1f}")
    print(f"  equalizeHist       : {eq.std():6.1f}  ← 大域的に最大限引き伸ばす")
    print(f"  CLAHE              : {cl.std():6.1f}  ← 局所的で自然（過増幅しない）")

    g_panels = [
        panel(gray, "low contrast"),
        panel(norm, "normalize (stretch)"),
        panel(eq, "equalizeHist (global)"),
        panel(cl, "CLAHE (local)"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "05_clahe_gray.png"), grid(g_panels, 2))

    # === 2. ヒストグラムを matplotlib で可視化 ===========================
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, im, name in zip(axes, (gray, eq, cl), ("low contrast", "equalizeHist", "CLAHE")):
        # calcHist: [画像], [チャンネル], マスク, [ビン数], [値域]
        hist = cv2.calcHist([im], [0], None, [256], [0, 256])
        ax.plot(hist, color="black")
        ax.set_title(name)
        ax.set_xlim(0, 255)
        ax.set_xlabel("pixel value")
    fig.suptitle("Histogram: narrow -> spread out")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_histograms.png"), dpi=110)
    plt.close(fig)

    # === 3. カラー: 各BGRチャンネル平坦化(誤) vs V チャンネルのみ(正) =====
    # 誤: B/G/R を別々に equalize → チャンネルごとに伸び方が違い色相がずれる
    b, g, r = cv2.split(scene)
    wrong = cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])
    # 正: HSV に変換し V（明度）だけ平坦化 → 色相 H と彩度 S は保たれる
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    right_eq = cv2.cvtColor(cv2.merge([h_, s_, cv2.equalizeHist(v_)]), cv2.COLOR_HSV2BGR)
    right_clahe = cv2.cvtColor(cv2.merge([h_, s_, clahe.apply(v_)]), cv2.COLOR_HSV2BGR)

    print("\n[2] カラー: BGR各chを個別に平坦化すると色が崩れる → V(明度)だけに適用するのが正")

    c_panels = [
        panel(scene, "original (low)"),
        panel(wrong, "per-BGR eq (WRONG color)"),
        panel(right_eq, "HSV-V equalize (OK)"),
        panel(right_clahe, "HSV-V CLAHE (OK)"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "05_color_value_channel.png"), grid(c_panels, 2))

    print("\n完了。outputs/04_filtering_edges_morphology/ の 05_*.png を確認してください。")


if __name__ == "__main__":
    main()
