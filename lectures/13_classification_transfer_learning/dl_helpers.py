"""第13回 共通の道具箱（device 判定・合成データ・保存ユーティリティ）。

このモジュールは「読み物」ではなく各スクリプトが import して使う再利用部品です。
深層CVトラックの最初の回なので、ここで一度だけ次の3つを固めておくと、
以降の 14〜41 回でもそのまま流用できます。

  1. get_device()      … cpu / mps / cuda を自動判定して torch.device を返す定石
  2. 合成データ生成     … 色×形の幾何図形を描いた小さな分類データセット（ネット不要）
  3. 保存ユーティリティ … outputs/ への図・JSON 保存（matplotlib は Agg、文字は ASCII）

合成データを使う理由: 本講座は GPU もデータセットの大量DLも無しに必ず完走できる必要があります。
そこで「赤い丸・緑の四角・青い三角…」のような、人間にも機械にも区別しやすい図形をその場で
描いて分類タスクの題材にします。実画像で試したい人向けの導線（data/ 配下）も用意します。
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # ヘッドレス環境で固まらないよう、画面ではなくファイルに描く

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# このモジュール（13回）の入出力ディレクトリ。__file__ から絶対パスで解決する。
MODULE_ID = "13_classification_transfer_learning"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = _REPO_ROOT / "outputs" / MODULE_ID
DATA_DIR = _REPO_ROOT / "data" / MODULE_ID


def ensure_out_dir() -> pathlib.Path:
    """outputs/13_.../ を作って返す（既にあっても OK）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def get_device() -> torch.device:
    """cuda → mps → cpu の優先順で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提なので既定では cpu になる。Apple Silicon なら mps、
    NVIDIA GPU があれば cuda を自動で拾う。重要なのは「モデルと入力の device を必ず揃える」こと
    （片方だけ .to(device) すると RuntimeError になる）。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """再現性のため乱数シードを固定する。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================================================================
# 合成データ: 色 × 形 の幾何図形（9クラス）
# =====================================================================
# 転移学習の題材。色(3) × 形(3) = 9 クラスにしておくと、
#   - 事前学習バックボーンの特徴で線形分離できる（= 転移学習が効くことを示せる）
#   - top-5 accuracy や混同行列が「クラス数 > 5」で意味を持つ
# という教材上のバランスが良い。
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 90, 220),
}
SHAPES: list[str] = ["circle", "square", "triangle"]
# クラス名は "red_circle" のように "色_形"。並び順は固定（id2label と対応させるため）。
SHAPE_CLASSES: list[str] = [f"{c}_{s}" for c in COLORS for s in SHAPES]


def make_shape_image(
    class_name: str,
    size: int = 96,
    rng: np.random.Generator | None = None,
    noise_std: float = 10.0,
) -> Image.Image:
    """指定クラス（"色_形"）の図形を1枚描いて PIL 画像で返す。

    位置・大きさ・背景の明るさ・ガウスノイズを毎回ランダムにずらすことで、
    「同じクラスでも見た目が少しずつ違う」現実的なばらつきを作る。
    これにより、ただ画素を丸暗記するのではなく特徴で分類する練習になる。
    """
    if rng is None:
        rng = np.random.default_rng()
    color_name, shape = class_name.split("_")
    col = COLORS[color_name]

    # 背景はうっすら明るいグレー（明度を毎回少し変える）。
    bg = int(rng.integers(150, 210))
    im = Image.new("RGB", (size, size), (bg, bg, bg))
    draw = ImageDraw.Draw(im)

    # 図形の外接矩形をランダムに決める（中央寄りに収める）。
    margin = size // 5
    x0, y0 = rng.integers(margin, size // 3, size=2)
    x1, y1 = rng.integers(2 * size // 3, size - margin // 2, size=2)
    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=col)
    elif shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=col)
    else:  # triangle
        draw.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=col)

    # 撮影ノイズの模擬。clip して uint8 に戻す。
    arr = np.asarray(im).astype(np.float32)
    arr += rng.normal(0.0, noise_std, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_dataset(
    n_per_class: int,
    size: int = 96,
    seed: int = 0,
    noise_std: float = 10.0,
) -> tuple[list[Image.Image], np.ndarray]:
    """合成分類データセットを作る。

    返り値:
      images: PIL 画像のリスト（長さ = n_per_class * 9）
      labels: 各画像のクラスインデックス（0..8）の int64 配列
    """
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    labels: list[int] = []
    for class_idx, class_name in enumerate(SHAPE_CLASSES):
        for _ in range(n_per_class):
            images.append(make_shape_image(class_name, size=size, rng=rng, noise_std=noise_std))
            labels.append(class_idx)
    return images, np.asarray(labels, dtype=np.int64)


def load_demo_images(n: int = 4, seed: int = 0) -> tuple[list[Image.Image], list[str]]:
    """分類デモ用の画像を n 枚返す（説明ラベル付き）。

    data/13_classification_transfer_learning/ に画像ファイル（.jpg/.png）を置いてあれば
    それを優先的に読み込む（実写真で試したい人向けの導線）。無ければ合成図形を生成する。
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if DATA_DIR.is_dir():
        files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
        if files:
            imgs = [Image.open(p).convert("RGB") for p in files[:n]]
            names = [p.name for p in files[:n]]
            return imgs, names

    # 合成: 代表的なクラスを n 種類選んで1枚ずつ描く。
    rng = np.random.default_rng(seed)
    chosen = SHAPE_CLASSES[:n]
    imgs = [make_shape_image(c, size=224, rng=rng) for c in chosen]
    names = [f"synthetic:{c}" for c in chosen]
    return imgs, names


def save_prediction_panel(
    images: list[Image.Image],
    titles: list[str],
    path: pathlib.Path,
    suptitle: str = "",
    ncols: int = 4,
) -> None:
    """画像とタイトル（予測ラベル等）を並べたパネルを PNG 保存する。

    PIL 画像は RGB なので matplotlib にそのまま渡せる（cv2 の BGR ではないので変換不要）。
    図中の文字は CJK フォントが無い環境で豆腐(□)にならないよう ASCII に保つこと。
    """
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
