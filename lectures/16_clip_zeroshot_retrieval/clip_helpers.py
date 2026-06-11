"""第16回 共通の道具箱（CLIP/SigLIP ゼロショット & 画像テキスト検索）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務は4つだけに絞ってあります（単一責務）:

  1. device の自動判定（cuda > mps > cpu。既定は cpu）
  2. 合成画像コレクションの生成（色 × 形。ネット不要・決定的）
  3. CLIP / SigLIP モデルとプロセッサのロード（eval + device + inference 前提）
  4. 出力ディレクトリの用意と matplotlib(Agg) の共通設定

なぜ合成画像か: CLIP は「赤い円」「青い四角」のような色＋形の概念を
ゼロショットで強く捉えるので、データセットを大量ＤＬしなくても
"a photo of a red circle" → 赤い円 の検索が妥当に決まり、教材として意味が出ます。
実画像で試したい人向けの導線は load_user_or_synthetic() を参照。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

# --- transformers / huggingface_hub のログを静かにする（教材の出力を読みやすく） ---
# 重み DL の進捗バーや未認証警告は学習の本質ではないので抑制する。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 環境（GUI なし）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# プロジェクトの規約に合わせ、結果は outputs/<module_id>/ に集約する。
MODULE_ID = "16_clip_zeroshot_retrieval"
OUTPUT_DIR = pathlib.Path("outputs") / MODULE_ID

# CPU でも現実的な「最速の定番」。patch16 や large は CPU では重い。
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# SigLIP2 の base。sigmoid 損失で学習されており、確率解釈が CLIP と異なる（後述）。
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"

# 合成コレクションの語彙。色は (R, G, B) で持つ（PIL はRGB、後述のBGR注意を回避）。
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 180, 60),
    "blue": (30, 60, 220),
    "yellow": (240, 210, 30),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle")


def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提だが、手元に GPU/Apple Silicon があれば自動で使えるようにする。
    重要: モデルと入力テンソルの device は必ず揃える（.to(device) を両方に）。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir() -> pathlib.Path:
    """outputs/<module_id>/ を作って返す（既にあってもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def make_shape_image(
    color_name: str, shape_name: str, size: int = 224, bg: int = 245
) -> Image.Image:
    """指定した色・形の合成画像（PIL.Image, RGB）を1枚作る。

    cv2 は描画ツールとして使うだけで、配列は最初から RGB として解釈する。
    （色タプルを (R,G,B) で渡し、その並びのまま channel に書くので BGR 混乱は起きない。
      cv2.imread/imwrite を経由しないので「BGR→RGB変換」は不要。）
    """
    img = np.full((size, size, 3), bg, dtype=np.uint8)  # 明るい灰色の背景
    color = COLORS[color_name]
    c = size // 2
    if shape_name == "circle":
        cv2.circle(img, (c, c), int(size * 0.32), color, thickness=-1)
    elif shape_name == "square":
        r = int(size * 0.30)
        cv2.rectangle(img, (c - r, c - r), (c + r, c + r), color, thickness=-1)
    elif shape_name == "triangle":
        pts = np.array(
            [[c, int(size * 0.16)], [int(size * 0.18), int(size * 0.83)], [int(size * 0.82), int(size * 0.83)]],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], color)
    else:
        raise ValueError(f"未知の形: {shape_name}")
    return Image.fromarray(img)  # RGB のまま PIL へ


def build_collection() -> tuple[list[Image.Image], list[tuple[str, str]]]:
    """色 × 形 の全組み合わせ（4×3=12枚）の画像とメタ情報を作る。

    返り値:
      images: PIL.Image のリスト
      meta:   (color_name, shape_name) のリスト（images と同じ並び）
    検索の「正解」はこの meta を使って判定する（ID とメタの対応表の最小版）。
    """
    images: list[Image.Image] = []
    meta: list[tuple[str, str]] = []
    for color_name in COLORS:
        for shape_name in SHAPES:
            images.append(make_shape_image(color_name, shape_name))
            meta.append((color_name, shape_name))
    return images, meta


def load_user_or_synthetic() -> tuple[list[Image.Image], list[str]]:
    """実画像があればそれを、無ければ合成コレクションを返す（実画像への導線）。

    data/16_clip_zeroshot_retrieval/ に .png/.jpg を置くと、それを読み込んで使う。
    無ければ合成（色×形）にフォールバックするので、引数なしでも必ず動く。
    返り値の caption は「赤い円」等の人間可読ラベル（合成時）／ファイル名（実画像時）。
    """
    data_dir = pathlib.Path("data") / MODULE_ID
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if data_dir.is_dir():
        files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in exts)
        if files:
            images = [Image.open(p).convert("RGB") for p in files]
            return images, [p.stem for p in files]
    images, meta = build_collection()
    return images, [f"{c} {s}" for c, s in meta]


def load_clip(device: torch.device):
    """CLIP モデルとプロセッサを返す（eval + device 済み）。

    transformers v5 の正準: AutoModel* と AutoImageProcessor 系。CLIP は
    専用クラス CLIPModel / CLIPProcessor が分かりやすいので本講座はこれを使う。
    （AutoModel.from_pretrained(CLIP_MODEL_ID) でも同じものが得られる。）
    """
    from transformers import CLIPModel, CLIPProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # 警告類を抑制
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def load_siglip(device: torch.device):
    """SigLIP2 モデルとプロセッサを返す（eval + device 済み）。

    SigLIP は AutoModel/AutoProcessor で読む。トークナイザに sentencepiece が要る。
    """
    from transformers import AutoModel, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
    return model, processor


def clip_image_embeds(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """CLIP の画像埋め込み（射影後・未正規化）を取り出す。

    ★transformers v5 の重要な変化:
      get_image_features(...) は『テンソル』ではなく BaseModelOutputWithPooling を返す。
      射影後の埋め込み (B, proj_dim) は .pooler_output に入っている。
      しかも未正規化（ノルムは 1 ではない）なので、コサイン類似度の前に
      torch.nn.functional.normalize(..., p=2, dim=-1) が必須（02/03 で体験する）。
    """
    return model.get_image_features(pixel_values=pixel_values).pooler_output


def clip_text_embeds(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """CLIP のテキスト埋め込み（射影後・未正規化）を取り出す（v5 は .pooler_output）。"""
    return model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).pooler_output


def save_image_grid(images: list[Image.Image], titles: list[str], path: pathlib.Path, ncol: int = 4) -> None:
    """画像コレクションをタイトル付きグリッドで保存する（中身の確認用）。"""
    n = len(images)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (im, title) in enumerate(zip(images, titles)):
        axes[i].imshow(im)  # PIL は RGB なので matplotlib にそのまま渡せる
        axes[i].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認する。
    out = ensure_output_dir()
    dev = pick_device()
    images, meta = build_collection()
    titles = [f"{c} {s}" for c, s in meta]
    grid_path = out / "00_collection_grid.png"
    save_image_grid(images, titles, grid_path)
    print(f"device={dev}  images={len(images)}  saved={grid_path}")
