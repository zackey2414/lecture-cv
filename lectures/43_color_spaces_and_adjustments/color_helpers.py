"""第43回モジュール共通ヘルパ（合成シーン生成・色空間ユーティリティ・可視化）。

このモジュールは「色空間と画像の調整」を学ぶための小さな道具箱です。
01〜05 のスクリプト・mini_project・exercises はこのファイルを import して使います。

ここに置く責務は次の4つに絞っています（単一責務を意識）。
  1. 学習用の合成シーンを作る（ネット不要・data/ が無くても必ず動く・乱数は seed 固定で再現可能）
  2. 日本語・空白を含むパスでも壊れない imwrite ラッパ
  3. 評価に使う定量指標（平均彩度・知覚的色差 ΔE）
  4. headless 環境向けの可視化（matplotlib=Agg でモンタージュ／ヒストグラムをファイル保存）

なぜ「他モジュールから import しない」のか:
  各モジュールを自己完結させるためです。第3回にも色空間の話（03/cv_helpers.py）がありますが、
  あえて重複を許し、このモジュールだけを取り出しても動くようにしています。
  教材は「読んで学ぶ」ものなので、依存を減らして追いやすさを優先します。
"""

from __future__ import annotations

import pathlib

import cv2
import matplotlib

# headless（GUI なし）環境でも確実に図を保存できるよう、pyplot を読む前に Agg を選ぶ。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  バックエンド確定後に import する必要がある
import numpy as np  # noqa: E402

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/43_color_spaces_and_adjustments/ にある。
#   parents[2] = プロジェクトルート（lecture-cv/）
# __file__ 基準なので、どのディレクトリから実行しても場所がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
# 出力はスクリプト隣の outputs/（= lectures/43_color_spaces_and_adjustments/outputs/）。
OUTPUT_ROOT = _HERE.parent / "outputs"
MODULE_ID = "43_color_spaces_and_adjustments"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    その保存先をここで一元管理する。存在しなければ自動で作成する。
    """
    d = OUTPUT_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d


def imwrite_unicode(path: str | pathlib.Path, img: np.ndarray) -> bool:
    """日本語・空白を含むパスでも安全に書く imwrite の代替。

    cv2.imwrite は内部で C のファイル API を使うため、環境によっては非ASCIIパスに書けない。
    「メモリ上にエンコード(cv2.imencode) → tofile でバイト列を書き出す」に分解して回避する。
    拡張子(.png/.jpg)で形式が決まるのは cv2.imwrite と同じ。成功したら True を返す。
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix, img)
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    """matplotlib 表示用に BGR → RGB へ変換する。

    OpenCV だけがチャンネル順 BGR で、matplotlib/PIL は RGB 前提。
    plt.imshow にそのまま BGR を渡すと赤と青が入れ替わるので、必ずここを通す。
    グレースケール（2次元）はそのまま返す。
    """
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =====================================================================
# 合成シーン生成（すべて seed 固定で決定的・BGR uint8 を返す）
# =====================================================================

def make_color_chart_bgr(cell: int = 70, margin: int = 10) -> np.ndarray:
    """色空間の数値を読み解くための「カラーチャート」を BGR uint8 で作る。

    純色〜無彩色のパッチを横一列に並べる。01 でこの各パッチを HSV/Lab/YCrCb に
    変換し、スケールの違い（H=0-179、Lab の L が 0-255 など）を表で対比する土台。
    """
    # (名前, BGR) のパッチ定義。彩度を最大にした純色＋無彩色（白/灰/黒）。
    patches = [
        ("red", (0, 0, 255)),
        ("orange", (0, 128, 255)),
        ("yellow", (0, 255, 255)),
        ("green", (0, 255, 0)),
        ("cyan", (255, 255, 0)),
        ("blue", (255, 0, 0)),
        ("magenta", (255, 0, 255)),
        ("white", (255, 255, 255)),
        ("gray", (128, 128, 128)),
        ("black", (0, 0, 0)),
    ]
    h = cell + 2 * margin
    w = margin + len(patches) * (cell + margin)
    img = np.full((h, w, 3), 235, dtype=np.uint8)  # 薄い背景
    for i, (_name, bgr) in enumerate(patches):
        x0 = margin + i * (cell + margin)
        img[margin:margin + cell, x0:x0 + cell] = bgr
    return img


def color_chart_patches() -> list[tuple[str, tuple[int, int, int]]]:
    """make_color_chart_bgr と同じパッチ定義（名前と BGR）を返す。

    チャート画像から座標計算で色を拾い直すより、定義を直接参照する方が安全で読みやすい。
    """
    return [
        ("red", (0, 0, 255)),
        ("orange", (0, 128, 255)),
        ("yellow", (0, 255, 255)),
        ("green", (0, 255, 0)),
        ("cyan", (255, 255, 0)),
        ("blue", (255, 0, 0)),
        ("magenta", (255, 0, 255)),
        ("white", (255, 255, 255)),
        ("gray", (128, 128, 128)),
        ("black", (0, 0, 0)),
    ]


def make_photo_scene_bgr(height: int = 320, width: int = 480, seed: int = 43) -> np.ndarray:
    """明るさ調整・彩度調整・WB の効果が目で見える合成「写真」を BGR uint8 で作る。

    設計のねらい:
      - 背景に左→右の明るさグラデーション（暗30〜明220）を敷き、トーン全域を使う
        → 明るさ/コントラスト/ガンマの効きと、輝度ヒストグラムの広がりが分かる。
      - 彩度の高い色物体（赤/緑/青/黄）を置く → 彩度・色相の調整が見える。
      - 肌色だ円を1つ置く → 後段の肌色検出やホワイトバランスの題材になる。
      - 無彩色（灰）の背景があるので gray-world ホワイトバランスが安定する。
    乱数は seed 固定で、毎回同じ画像になる（出力の再現性を確保）。
    """
    rng = np.random.default_rng(seed)
    h, w = height, width

    # 横方向の明るさグラデーション（左=暗, 右=明）＋ 縦方向の弱い傾き。
    hramp = np.linspace(30.0, 220.0, w, dtype=np.float32)[None, :]
    vramp = np.linspace(-15.0, 15.0, h, dtype=np.float32)[:, None]
    gray = np.clip(hramp + vramp, 0, 255)
    img = np.repeat(gray[:, :, None], 3, axis=2)  # 無彩色背景 (H, W, 3)
    # 軽いガウスノイズで「写真らしさ」を足す（コントラスト調整の効果を見やすくする）。
    img = np.clip(img + rng.normal(0.0, 4.0, img.shape), 0, 255).astype(np.uint8)

    # 彩度の高い物体（BGR）。座標は (x, y)、色は BGR、thickness=-1 で塗りつぶし。
    cv2.circle(img, (int(w * 0.20), int(h * 0.34)), int(h * 0.16), (40, 40, 210), -1,
               lineType=cv2.LINE_AA)                                            # 赤円
    cv2.rectangle(img, (int(w * 0.54), int(h * 0.10)),
                  (int(w * 0.80), int(h * 0.40)), (60, 175, 60), -1)            # 緑矩形
    cv2.circle(img, (int(w * 0.34), int(h * 0.74)), int(h * 0.15), (205, 90, 35), -1,
               lineType=cv2.LINE_AA)                                            # 青円
    cv2.rectangle(img, (int(w * 0.60), int(h * 0.58)),
                  (int(w * 0.88), int(h * 0.86)), (40, 205, 215), -1)           # 黄矩形
    # 肌色のだ円（B<G<R の暖色）。肌色検出の題材。
    cv2.ellipse(img, (int(w * 0.80), int(h * 0.28)), (int(w * 0.07), int(h * 0.12)),
                0, 0, 360, (150, 180, 225), -1, lineType=cv2.LINE_AA)
    return img


def make_lowlight_scene_bgr(seed: int = 43) -> np.ndarray:
    """暗くてコントラストの低い「夜景・露出不足」風シーンを BGR uint8 で作る。

    ヒストグラム平坦化／CLAHE の効果を見せるため、画素値を低い帯域（おおむね 20〜90）に
    押し込めてある。equalizeHist や CLAHE でトーンを引き伸ばすと細部が浮かび上がる。
    """
    base = make_photo_scene_bgr(seed=seed)
    # 全体を暗く・低コントラストに圧縮する（alpha<1 で傾きを寝かせ、beta で底上げ）。
    dark = cv2.convertScaleAbs(base, alpha=0.32, beta=18)
    return dark


def apply_color_cast(bgr: np.ndarray, gains_bgr: tuple[float, float, float]) -> np.ndarray:
    """各チャンネルにゲインを掛けて「色かぶり（color cast）」を作る。

    照明や WB ずれを再現する。gains=(1.4, 1.0, 0.9) なら青が強く赤が弱い寒色かぶり。
    飽和（255 超え）は np.clip で頭打ちにする。ホワイトバランス章の入力を作るのに使う。
    """
    out = bgr.astype(np.float32) * np.array(gains_bgr, dtype=np.float32)[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_skin_scene_bgr() -> np.ndarray:
    """肌色検出の題材。肌色のだ円群＋紛らわしい非肌色（赤・緑・黄）を配置する。

    肌色は「赤が一番強い暖色」だが、純粋な赤とは別物。YCrCb/HSV の領域で
    肌だけを抜けるか（赤い物体を巻き込まないか）を試すための合成シーン。
    """
    h, w = 280, 420
    img = np.full((h, w, 3), 205, dtype=np.uint8)  # 薄いグレー背景
    # 肌色（明〜暗、いずれも B<G<R）のだ円。
    skins = [(150, 180, 225), (120, 150, 200), (95, 125, 175)]
    for i, c in enumerate(skins):
        cx = int(w * (0.18 + 0.16 * i))
        cv2.ellipse(img, (cx, int(h * 0.35)), (40, 55), 0, 0, 360, c, -1, lineType=cv2.LINE_AA)
    # 紛らわしい非肌色（純赤・緑・黄）。肌マスクが拾ってはいけない。
    cv2.circle(img, (int(w * 0.20), int(h * 0.75)), 36, (0, 0, 235), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (int(w * 0.50), int(h * 0.75)), 36, (50, 200, 50), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (int(w * 0.80), int(h * 0.75)), 36, (40, 220, 230), -1, lineType=cv2.LINE_AA)
    return img


# =====================================================================
# 色空間ユーティリティ（調整の中核）
# =====================================================================

def build_gamma_lut(gamma: float) -> np.ndarray:
    """ガンマ補正用の 256 要素ルックアップテーブル（uint8）を作る。

    出力 = 255 * (入力/255)^(1/gamma)。
      - gamma > 1: 暗部を持ち上げ全体が明るくなる（暗所写真の救済）。
      - gamma < 1: 明部を抑え全体が暗くなる。
    1画素ずつ pow を計算すると重いので、0-255 の対応表を1度だけ作り cv2.LUT で一括適用する。
    """
    inv = 1.0 / float(gamma)
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((x ** inv) * 255.0, 0, 255).astype(np.uint8)
    return lut


def adjust_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """ガンマ補正を適用する（非線形な明るさ調整）。cv2.LUT で高速に行う。"""
    return cv2.LUT(bgr, build_gamma_lut(gamma))


def adjust_brightness_contrast(bgr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """線形な明るさ/コントラスト調整 out = alpha*in + beta（飽和演算）。

    alpha がコントラスト（傾き、1.0 で等倍）、beta が明るさ（オフセット）。
    cv2.convertScaleAbs は内部で |alpha*in+beta| を取り 0-255 にクリップ（飽和）する。
    素の numpy 加算はオーバーフローでラップアラウンドするので、調整にはこちらを使う。
    """
    return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)


def scale_saturation(bgr: np.ndarray, factor: float) -> np.ndarray:
    """彩度だけを factor 倍する（色相・明度は保つ）。HSV の S チャンネルを操作。

    factor>1 で鮮やか、factor<1 でくすむ、0 でグレースケール相当。
    uint8 のまま S を掛けるとオーバーフローするので、float32 で計算して clip→uint8 に戻す。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(factor), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def shift_hue(bgr: np.ndarray, delta: int) -> np.ndarray:
    """色相を delta（0-179 スケール）だけ回す。色相環は一周するので mod 180。

    明るさ・彩度を一切変えずに「色だけ」を移す。delta=90 でおよそ補色側へ回る。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int32) + int(delta)) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def equalize_luminance(bgr: np.ndarray) -> np.ndarray:
    """輝度チャンネル(YCrCb の Y)だけをヒストグラム平坦化する（色を崩さない）。

    BGR の各チャンネルに equalizeHist を掛けると色相がずれて色が崩れる。
    輝度(Y)だけを平坦化し、色差(Cr,Cb)はそのまま戻すのが正しい作法。
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_luminance(bgr: np.ndarray, clip_limit: float = 2.0,
                    tile: int = 8) -> np.ndarray:
    """Lab の L チャンネルにだけ CLAHE（局所コントラスト強調）を掛ける（色を保つ）。

    equalizeHist は画像全体で1本のヒストグラムを使うため局所的な明暗には弱い。
    CLAHE はタイルごとに平坦化し、増幅しすぎを clip_limit で抑える。
    色差 a*,b* は触らないので色が崩れない。
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """gray-world 仮説によるホワイトバランス。各チャンネル平均を共通の灰色へ揃える。

    仮説:「シーン全体を平均すれば無彩色（灰）になるはず」。
    各チャンネル平均 (mB,mG,mR) を全体平均 gray に合わせるゲインを掛けて色かぶりを打ち消す。
    float で計算し clip→uint8 に戻す。
    """
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)            # (mB, mG, mR)
    gray = float(means.mean())
    gains = gray / np.clip(means, 1e-6, None)         # 0 割りを避ける
    return np.clip(f * gains[None, None, :], 0, 255).astype(np.uint8)


# =====================================================================
# 評価指標（定量確認）
# =====================================================================

def mean_saturation(bgr: np.ndarray) -> float:
    """画像全体の平均彩度（HSV の S, 0-255）を返す。彩度調整の効果を数値で測る。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def delta_e_map(bgr_a: np.ndarray, bgr_b: np.ndarray) -> np.ndarray:
    """2画像の画素ごとの知覚的色差 ΔE*76（CIE L*a*b* のユークリッド距離）を返す。

    重要: 必ず float の L*a*b*（L:0-100, a/b:約 -127〜127）で計算する。
    uint8 の Lab（全て 0-255 にスケール）で距離を取るとスケールが歪んで ΔE の意味が崩れる。
    目安: ΔE<1 はほぼ識別不能、2-3 で訓練された目が気づく、>5 で明確に別の色。
    """
    lab_a = cv2.cvtColor(bgr_a.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lab_b = cv2.cvtColor(bgr_b.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return np.sqrt(((lab_a - lab_b) ** 2).sum(axis=2))


def mean_delta_e(bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
    """2画像の平均 ΔE*76。調整がどれだけ色を動かしたかの定量指標。"""
    return float(delta_e_map(bgr_a, bgr_b).mean())


def luminance(bgr: np.ndarray) -> np.ndarray:
    """輝度チャンネル（YCrCb の Y, 0-255 の (H,W) uint8）を返す。ヒストグラム用。"""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]


# =====================================================================
# 可視化（matplotlib=Agg でファイル保存。図タイトルは ASCII）
# =====================================================================

def save_montage(panels: list[tuple[str, np.ndarray]], path: str | pathlib.Path,
                 suptitle: str = "", cols: int = 3) -> None:
    """(タイトル, BGR画像) のリストを格子状に並べて1枚の図に保存する。

    BGR→RGB 変換を内部で行うので、呼び出し側は OpenCV の BGR をそのまま渡せる。
    タイトルは matplotlib のフォント問題（日本語豆腐化）を避けるため ASCII 推奨。
    """
    n = len(panels)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img) in zip(axes, panels):
        rgb = to_rgb(img)
        ax.imshow(rgb, cmap="gray" if rgb.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


def save_luminance_hist(named: list[tuple[str, np.ndarray]], path: str | pathlib.Path,
                        suptitle: str = "Luminance histogram") -> None:
    """複数 BGR 画像の輝度ヒストグラムを重ねてプロットし保存する。

    明るさ/コントラスト/ガンマ/平坦化の効果は、輝度分布の「位置と広がり」に最も素直に出る。
    各画像の Y(0-255) ヒストグラムを 256bin で描く。
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, bgr in named:
        y = luminance(bgr).ravel()
        hist = cv2.calcHist([y], [0], None, [256], [0, 256]).ravel()
        ax.plot(np.arange(256), hist, label=label, linewidth=1.2)
    ax.set_xlim(0, 255)
    ax.set_xlabel("luminance (Y, 0-255)")
    ax.set_ylabel("pixel count")
    ax.set_title(suptitle)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
