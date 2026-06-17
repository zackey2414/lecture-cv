"""01: 平滑化フィルタの比較 — blur / GaussianBlur / medianBlur / bilateralFilter / filter2D。

学ぶこと:
  - 平滑化（スムージング）= 近傍画素の重み付き平均で「画素のばらつき」を均す処理。
  - 4つの定番フィルタの性質の違い:
      blur           : 単純平均（box）。速いがエッジまでボケる。
      GaussianBlur   : 中心ほど重い重み（ガウス）。自然なボケ。ノイズ除去の標準。
      medianBlur     : 近傍の「中央値」。ごま塩（salt&pepper）ノイズに圧倒的に強い。
      bilateralFilter: 距離＋輝度差の両方で重み付け。エッジを残したままノイズだけ消す。
  - filter2D は「自前カーネルでの畳み込み」の一般形。平均化も先鋭化も書ける。
  - 「どのノイズに、どのフィルタが効くか」を MSE（小さいほど元画像に近い）で数値確認する。

実行:
  uv run python lectures/04_filtering_edges_morphology/01_smoothing.py
結果は lectures/04_filtering_edges_morphology/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np

# 画面表示はせず、結果は必ずファイルへ保存する（headless / Docker / CI でも動くように）。
MODULE_ID = "04_filtering_edges_morphology"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# 表示用の小道具（このモジュール内で自己完結。他モジュールからは import しない）
# ---------------------------------------------------------------------------
def panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 270)) -> np.ndarray:
    """画像をタイトル付きの1枚パネルにする（グレーは3chへ昇格してから並べられるように）。"""
    if img.ndim == 2:  # グレースケール (H, W) は 3ch にしないと横に並べられない
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w, h = size
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)  # 黒い帯
    cv2.putText(p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    """パネルの配列を cols 列のグリッドに連結する。余りは黒で埋める。"""
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:  # 端数を黒パネルで埋めて hstack 可能にする
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# サンプル画像の合成生成（サンプルが無くても必ず動く）
# ---------------------------------------------------------------------------
def make_clean_image(h: int = 270, w: int = 360) -> np.ndarray:
    """エッジ・面・細線（高周波）を含む合成画像を作る。BGR uint8。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # ゆるい縦グラデーション背景（平滑化で滑らかさが保たれるか確認用）
    grad = np.linspace(40, 200, w, dtype=np.uint8)
    img[:] = np.repeat(grad[None, :], h, axis=0)[:, :, None]
    # はっきりした色面（エッジ保持の確認用）
    cv2.rectangle(img, (30, 40), (150, 180), (60, 180, 75), -1)  # 緑
    cv2.circle(img, (260, 110), 55, (40, 60, 220), -1)  # 赤
    # 細い線＝高周波成分。ボケると最初に消える
    for x in range(20, w - 20, 16):
        cv2.line(img, (x, 210), (x, 250), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "EDGE", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def add_gaussian_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """ガウスノイズ（センサノイズ相当）を加える。float で計算し clip して uint8 へ戻す。"""
    noise = np.random.default_rng(0).normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, amount: float = 0.06) -> np.ndarray:
    """ごま塩ノイズ（画素が突発的に 0 か 255 に化ける）を加える。"""
    out = img.copy()
    rng = np.random.default_rng(1)
    mask = rng.random(img.shape[:2])
    out[mask < amount / 2] = 0  # 黒い粒（pepper）
    out[mask > 1 - amount / 2] = 255  # 白い粒（salt）
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """平均二乗誤差。小さいほど元画像（クリーン）に近い＝ノイズが取れている。"""
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    clean = make_clean_image()

    # === 1. ガウスノイズに対する各フィルタ ===============================
    g_noisy = add_gaussian_noise(clean)
    # blur: 単純平均。カーネルサイズが大きいほど強くボケる。
    g_box = cv2.blur(g_noisy, (5, 5))
    # GaussianBlur: ksize は奇数、sigmaX=0 なら ksize から自動算出。最も使う平滑化。
    g_gauss = cv2.GaussianBlur(g_noisy, (5, 5), sigmaX=0)
    # medianBlur: ksize は奇数の int。ガウスノイズにはガウスより少し不利。
    g_median = cv2.medianBlur(g_noisy, 5)
    # bilateralFilter: d=近傍直径, sigmaColor=輝度差の許容, sigmaSpace=距離の効き。
    g_bilat = cv2.bilateralFilter(g_noisy, d=9, sigmaColor=75, sigmaSpace=75)

    print("[1] ガウスノイズ画像に対する MSE（小さいほどクリーンに近い）")
    print(f"  noisy(無処理) : {mse(clean, g_noisy):8.1f}")
    print(f"  blur          : {mse(clean, g_box):8.1f}")
    print(f"  GaussianBlur  : {mse(clean, g_gauss):8.1f}  ← ガウスノイズの定番")
    print(f"  medianBlur    : {mse(clean, g_median):8.1f}")
    print(f"  bilateral     : {mse(clean, g_bilat):8.1f}  ← エッジを保ちつつ除去")

    g_panels = [
        panel(clean, "clean"),
        panel(g_noisy, "gaussian noise"),
        panel(g_box, "blur 5x5"),
        panel(g_gauss, "GaussianBlur 5x5"),
        panel(g_median, "medianBlur 5"),
        panel(g_bilat, "bilateral"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "01_smoothing_gaussian_noise.png"), grid(g_panels, 3))

    # === 2. ごま塩ノイズに対する各フィルタ（medianBlur が圧勝する場面）===
    sp_noisy = add_salt_pepper(clean)
    sp_box = cv2.blur(sp_noisy, (5, 5))
    sp_gauss = cv2.GaussianBlur(sp_noisy, (5, 5), sigmaX=0)
    sp_median = cv2.medianBlur(sp_noisy, 5)

    print("\n[2] ごま塩ノイズ画像に対する MSE")
    print(f"  noisy(無処理) : {mse(clean, sp_noisy):8.1f}")
    print(f"  blur          : {mse(clean, sp_box):8.1f}  （粒を薄く広げるだけ）")
    print(f"  GaussianBlur  : {mse(clean, sp_gauss):8.1f}")
    print(f"  medianBlur    : {mse(clean, sp_median):8.1f}  ← 中央値が粒を無視して圧勝")

    sp_panels = [
        panel(clean, "clean"),
        panel(sp_noisy, "salt & pepper"),
        panel(sp_box, "blur (poor)"),
        panel(sp_gauss, "GaussianBlur (poor)"),
        panel(sp_median, "medianBlur (best)"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "01_smoothing_salt_pepper.png"), grid(sp_panels, 3))

    # === 3. filter2D — 自前カーネルでの畳み込みの一般形 ==================
    # 平均化カーネル（blur と等価）。総和=1 にしないと明るさが変わる。
    avg_kernel = np.ones((5, 5), np.float32) / 25.0
    f_avg = cv2.filter2D(clean, ddepth=-1, kernel=avg_kernel)  # ddepth=-1 は入力と同じ型
    # 先鋭化（シャープ）カーネル。中心を強め周囲を引く。総和=1 で明るさ維持。
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    f_sharp = cv2.filter2D(clean, -1, sharp_kernel)

    print("\n[3] filter2D は平均化も先鋭化も同じ枠組みで書ける（カーネル次第）")

    f_panels = [
        panel(clean, "clean"),
        panel(f_avg, "filter2D average"),
        panel(f_sharp, "filter2D sharpen"),
    ]
    cv2.imwrite(os.path.join(OUT_DIR, "01_smoothing_filter2d.png"), grid(f_panels, 3))

    print("\n完了。lectures/04_filtering_edges_morphology/outputs/ の 01_*.png を確認してください。")


if __name__ == "__main__":
    main()
