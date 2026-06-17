"""第12回モジュール共通の道具箱（PyTorch 画像テンソルとデータ拡張）。

01〜03 の各スクリプトはこのファイルを import して使います。ここに置く責務は
「データパイプラインの演習で毎回書くことになる定型処理」だけに絞っています
（過度な抽象化はしない。学習の主眼であるテンソル変換・Dataset・拡張は各スクリプト
本体に“生のまま”書いて、読んで学べるようにしています）。

ここに置くもの:
  1. 出力先 lectures/12_data_pipeline_augmentation/outputs/ の場所を返す
  2. ネット非依存で動かすための「合成画像フォルダ」生成
     （3クラス＝円/四角/三角の小画像。カスタム Dataset の練習台）
  3. ImageNet と CLIP の正規化統計（mean/std）— この2つは別物である点が重要
  4. 可視化のための逆正規化（denormalize）と、CHW float テンソル → HWC uint8 変換
  5. matplotlib(Agg) で画像を格子状に保存する関数

なぜ合成画像なのか:
  Webカメラもデータセットの大量DLも無い環境（CI/Docker）で、全スクリプトが
  そのまま exit 0 で完走できるようにするため。実画像で試したい人は data/ に
  クラスごとのフォルダを置けば、そちらが自動で優先されます（下の find_or_make_dataset）。
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")  # headless 環境向け。画面を出さずファイルに保存する
import matplotlib.pyplot as plt  # noqa: E402

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/12_data_pipeline_augmentation/ にある。
#   parents[0] = モジュール / parents[1] = lectures / parents[2] = プロジェクトルート
# __file__ 基準なので、どのディレクトリから実行しても出力先がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
MODULE_ID = "12_data_pipeline_augmentation"

# --- 正規化統計（mean/std は RGB 順・[0,1] スケール前提）-------------------
# ImageNet で学習した CNN/ViT（torchvision の ResNet など）はこの統計で正規化する。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# OpenAI CLIP は専用の統計を使う。ImageNet 値で代用すると精度が落ちるので必ず使い分ける。
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# 合成データセットのクラス（フォルダ名＝クラス名にする）。
CLASS_NAMES = ("circle", "square", "triangle")


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    存在しなければ自動で作成する。
    """
    d = _HERE.parent / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick_device() -> torch.device:
    """cuda → mps → cpu の順で使える device を選ぶ全回共通の定石。

    本章はデータパイプラインが主役なので CPU で完結するが、学習回（第13回〜）で
    「バッチを model と同じ device に載せる」ときにこの関数を使い回す。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- 合成画像フォルダの生成 ----------------------------------------------

def _draw_shape(shape: str, rng: np.random.Generator, size: int = 96) -> Image.Image:
    """1枚の合成画像（背景＋図形）を PIL.Image(RGB) で作る。

    クラスごとに図形（円/四角/三角）を描く。色・位置・背景は乱数で少し散らして、
    『同じクラスでも見た目に幅がある』教材的に意味のあるばらつきを作る。
    """
    # 背景はクラスに依らないランダムな淡色（拡張の効果が見やすいよう中間調にする）。
    bg = tuple(int(v) for v in rng.integers(40, 110, size=3))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    # 図形の色は明るめにして背景と分離する。
    color = tuple(int(v) for v in rng.integers(150, 255, size=3))
    # 図形の大きさ・位置を散らす。
    r = int(rng.integers(size // 4, size // 3))
    cx = int(rng.integers(r + 4, size - r - 4))
    cy = int(rng.integers(r + 4, size - r - 4))

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:  # triangle
        draw.polygon(
            [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color
        )
    return img


def make_synthetic_dataset(root: pathlib.Path, per_class: int = 8, seed: int = 0) -> pathlib.Path:
    """root/<class>/*.png に合成画像を書き出す（既に十分あれば作り直さない）。

    クラス名のサブフォルダを作り、その中に画像を置く——これは torchvision の
    ImageFolder と同じ『フォルダ＝ラベル』の定番レイアウト。自作 Dataset の練習台になる。
    """
    root.mkdir(parents=True, exist_ok=True)
    for label, name in enumerate(CLASS_NAMES):
        cls_dir = root / name
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing = list(cls_dir.glob("*.png"))
        for i in range(len(existing), per_class):
            # seed をクラス・枚数で変えて毎回同じ絵を再現できるようにする。
            local_rng = np.random.default_rng(seed * 1000 + label * 100 + i)
            img = _draw_shape(name, local_rng)
            img.save(cls_dir / f"{name}_{i:02d}.png")
    return root


def find_or_make_dataset(per_class: int = 8) -> pathlib.Path:
    """学習用の画像フォルダを返す。data/ に実画像があればそれを優先する。

    優先順位:
      1. data/synth_shapes/ がクラスフォルダを持っていればそれを使う（自分の画像で試せる）
      2. 無ければ lectures/<module>/outputs/synthetic_dataset/ に合成画像を作って使う
    """
    user_dir = PROJECT_ROOT / "data" / "synth_shapes"
    if user_dir.exists() and any(p.is_dir() for p in user_dir.iterdir()):
        return user_dir
    return make_synthetic_dataset(output_dir() / "synthetic_dataset", per_class=per_class)


# --- 可視化のためのテンソル変換 ------------------------------------------

def denormalize(t: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """Normalize の逆操作: x*std + mean で [0,1] スケールに戻す（表示用）。

    Normalize 後のテンソルは負値を含み、そのまま imshow すると色が壊れる。
    可視化するときは必ず『使った mean/std で戻す』のが鉄則。
    """
    mean_t = torch.tensor(mean, dtype=t.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype).view(-1, 1, 1)
    return t * std_t + mean_t


def chw_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """CHW float テンソル（[0,1] 想定）を HWC uint8 の numpy 配列にする（表示用）。

    PyTorch のテンソルは (C, H, W)、画像表示ライブラリは (H, W, C) を期待する。
    この並び替え（permute）が CHW↔HWC 変換の正体。範囲外は 0..1 にクランプする。
    """
    arr = t.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return arr.permute(1, 2, 0).cpu().numpy()


def save_image_grid(
    images: list[np.ndarray],
    path: pathlib.Path,
    titles: list[str] | None = None,
    ncols: int = 4,
    suptitle: str = "",
) -> None:
    """HWC uint8(RGB) 画像のリストを格子状に並べて1枚の図として保存する。

    拡張は『毎回ランダムに変わる』ので、同じ画像に何度も拡張をかけた結果を並べると
    効果が一目で分かる。図中の文字は日本語フォントが無い環境で豆腐(□)にならないよう
    ASCII にしている。
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
