"""第21回 共通の道具箱（セマンティックセグメンテーション 入門）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務はおおむね4つに絞ってあります（単一責務）:

  1. device の自動判定（cuda > mps > cpu。既定は cpu）
  2. 入力画像の用意（実画像があれば優先、無ければ合成シーンを生成）
  3. クラスID → 色 の可視化道具（VOC カラーマップ・乱数パレット・重ね描き）
  4. モデルのロード（torchvision セグ / HF SegFormer）と図の保存

なぜ合成画像か:
  本講座はネット接続やデータセットDLに依存せず完走させたい。検出/セグメは
  合成画像だと「実在クラス（人・車など）」が乏しく予測が薄くなることがあるが、
  スクリプトは必ず exit 0 にし、`data/21_segmentation_intro/` に実写を置けば
  自動でそちらを使う導線にしてある（load_user_or_synthetic_image を参照）。

評価指標（mIoU/Dice/pixel acc）の「定義そのもの」を確かめる 03 では、
モデルに依存しない決定的な GT/予測ペア（make_toy_gt_pred）を使う。
こうすると数値が手計算でき、torchmetrics との一致確認が再現可能になる。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

# --- transformers / hub のログを静かにする（教材の出力を読みやすく） ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 環境（GUI なし）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# 結果は規約どおり lectures/<module_id>/outputs/ に集約する。
MODULE_ID = "21_segmentation_intro"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = pathlib.Path("data") / MODULE_ID

# torchvision のセグメモデルは Pascal VOC（21クラス, index0 = __background__）で学習。
# HF SegFormer-b0 は ADE20K（150クラス）。クラス体系が違う点に注意。
TORCHVISION_SEG = "lraspp"  # 既定は最軽量（CPU 向き, 約12MB）
SEGFORMER_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


# =====================================================================
# device / 出力ディレクトリ
# =====================================================================
def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir() -> pathlib.Path:
    """lectures/<module_id>/outputs/ を作って返す（既にあってもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# =====================================================================
# 入力画像（実画像優先 / 無ければ合成シーン）
# =====================================================================
def make_photo_like_scene(size: int = 384, seed: int = 0) -> Image.Image:
    """実在クラスを“それっぽく”含む合成シーン（PIL.Image, RGB）を1枚作る。

    空（上部のグラデ）＋地面（緑）＋中央に人物シルエット（胴＋頭）＋小物（瓶）を描く。
    VOC/ADE のモデルが「person」等を拾えることもあるが、合成なので保証はしない。
    実写で試したいときは data/21_segmentation_intro/ に画像を置く（README 参照）。
    cv2 は描画にだけ使い、配列は最初から RGB として扱う（imread/imwrite を経由しない）。
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # 空: 上から下へ青→白のグラデーション
    for y in range(size):
        t = y / size
        img[y, :, 0] = int(120 + 110 * t)  # R
        img[y, :, 1] = int(160 + 80 * t)  # G
        img[y, :, 2] = int(210 + 30 * (1 - t))  # B
    # 地面: 下 35% を緑に
    gy = int(size * 0.65)
    img[gy:, :, :] = (70, 140, 60)
    # 人物シルエット: 頭・胴・腕・脚を描いて“人らしさ”を上げる（実写なら良く当たる作り）。
    cx = size // 2
    cv2.circle(img, (cx, gy - 150), 26, (210, 180, 150), thickness=-1)  # 頭
    cv2.rectangle(img, (cx - 30, gy - 122), (cx + 30, gy - 30), (70, 80, 130), thickness=-1)  # 胴
    cv2.rectangle(img, (cx - 48, gy - 118), (cx - 30, gy - 50), (70, 80, 130), thickness=-1)  # 左腕
    cv2.rectangle(img, (cx + 30, gy - 118), (cx + 48, gy - 50), (70, 80, 130), thickness=-1)  # 右腕
    cv2.rectangle(img, (cx - 26, gy - 30), (cx - 4, gy + 18), (40, 45, 70), thickness=-1)  # 左脚
    cv2.rectangle(img, (cx + 4, gy - 30), (cx + 26, gy + 18), (40, 45, 70), thickness=-1)  # 右脚
    # 小物: 右下に瓶っぽい縦長
    cv2.rectangle(img, (int(size * 0.72), gy - 60), (int(size * 0.78), gy + 5), (40, 120, 160), thickness=-1)
    # わずかなノイズで“写真っぽさ”を足す（決定的）
    noise = rng.randint(-8, 9, size=(size, size, 3))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def load_user_or_synthetic_image() -> tuple[Image.Image, str]:
    """実画像があればそれを、無ければ合成シーンを返す（実画像への導線）。

    data/21_segmentation_intro/ に .png/.jpg を置くと、最初の1枚を読み込んで使う。
    無ければ合成にフォールバックするので、引数なしでも必ず動く。
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if DATA_DIR.is_dir():
        files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
        if files:
            return Image.open(files[0]).convert("RGB"), files[0].name
    return make_photo_like_scene(), "synthetic_scene"


# =====================================================================
# クラスID → 色 の可視化道具
# =====================================================================
def voc_colormap(n: int = 256) -> np.ndarray:
    """Pascal VOC の標準カラーマップ (n,3) uint8 を生成する。

    各クラスIDのビットを R/G/B に振り分ける古典的な手法。index0 は黒（背景）。
    torchvision のセグメ出力（VOC 21クラス）の可視化に使う。
    """
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap


def random_palette(num: int, seed: int = 0) -> np.ndarray:
    """num 色の決定的なランダムパレット (num,3) uint8 を返す（ADE20K 150 クラス用）。"""
    rng = np.random.RandomState(seed)
    pal = rng.randint(0, 256, size=(num, 3)).astype(np.uint8)
    pal[0] = (0, 0, 0)  # 0 番は見やすさのため黒に寄せる
    return pal


def colorize(class_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """クラスIDマップ (H,W) を palette でRGB画像 (H,W,3) uint8 に変換する。

    palette[class_map] の高速な索引で色付けする。class_map の最大値 < len(palette) が前提。
    """
    return palette[class_map]


def overlay(rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """元画像 (H,W,3) と色マスク (H,W,3) を alpha で重ねた uint8 画像を返す。"""
    blended = (1.0 - alpha) * rgb.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# =====================================================================
# モデルのロード
# =====================================================================
def load_torchvision_seg(name: str, device: torch.device):
    """torchvision のセマンティックセグメンテーションモデルと weights を返す。

    name: 'lraspp'（軽量・CPU向き） / 'deeplabv3'（高精度・resnet50） / 'fcn'。
    返り値の weights には .transforms()（前処理）と .meta['categories']（クラス名）が入る。
    """
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        FCN_ResNet50_Weights,
        LRASPP_MobileNet_V3_Large_Weights,
        deeplabv3_resnet50,
        fcn_resnet50,
        lraspp_mobilenet_v3_large,
    )

    table = {
        "lraspp": (lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights),
        "deeplabv3": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights),
        "fcn": (fcn_resnet50, FCN_ResNet50_Weights),
    }
    if name not in table:
        raise ValueError(f"未知のモデル名: {name}（lraspp/deeplabv3/fcn）")
    ctor, wenum = table[name]
    weights = wenum.DEFAULT
    model = ctor(weights=weights).to(device).eval()
    return model, weights


def load_segformer_pipeline(device: torch.device):
    """HF pipeline('image-segmentation') で SegFormer をロードする。"""
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    return pipeline(
        "image-segmentation",
        model=SEGFORMER_ID,
        device=0 if device.type == "cuda" else -1,
    )


# =====================================================================
# 図の保存
# =====================================================================
def save_panel(images: list[np.ndarray], titles: list[str], path: pathlib.Path, ncol: int = 3) -> None:
    """画像（RGB uint8）をタイトル付きで横並びに保存する。"""
    n = len(images)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.4))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (im, title) in enumerate(zip(images, titles)):
        axes[i].imshow(im)
        axes[i].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# 評価用の決定的な GT / 予測ペア（03 で使用）
# =====================================================================
# 小さな「おもちゃの世界」のクラス。person(=5) は GT にも予測にも出さず、
# 「両方に未出現のクラス」が mIoU からどう扱われるか（除外）を見せるために置く。
TOY_CLASSES = ["background", "sky", "tree", "road", "car", "person"]


def make_toy_gt_pred(size: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """決定的な GT クラスマップと、わざと誤らせた予測クラスマップを返す。

    返り値はどちらも (size,size) の int 配列（値は 0..len(TOY_CLASSES)-1）。
    予測は GT を土台に「境界のズレ・領域の取り違え・小領域の消失」を入れてあり、
    per-class IoU/Dice が 0<x<1 の手頃な値になるよう調整してある（手計算で検算可能）。
    """
    h = w = size
    # cv2 の描画関数は uint8 配列を要求するので、ラベルマップは uint8 で作って後で int64 化する。
    gt = np.zeros((h, w), dtype=np.uint8)  # 0 = background
    sky_y = int(h * 0.35)
    road_y = int(h * 0.72)
    gt[:sky_y, :] = 1  # sky: 上部
    gt[road_y:, :] = 3  # road: 下部
    cv2.circle(gt, (int(w * 0.30), int(h * 0.52)), int(size * 0.16), 2, thickness=-1)  # tree
    cv2.rectangle(gt, (int(w * 0.60), road_y - 10), (int(w * 0.85), road_y + 14), 4, thickness=-1)  # car

    # 予測: GT をコピーして誤りを混ぜる
    pred = gt.copy()
    # (a) tree を右下へ数px ずらす（境界誤差 → IoU < 1）
    tree_mask = gt == 2
    shifted = np.zeros_like(tree_mask)
    shifted[6:, 6:] = tree_mask[:-6, :-6]
    pred[tree_mask] = 1  # 元の tree 位置を sky(1) に誤る部分が出る
    pred[shifted] = 2  # ずれた位置を tree とする
    # (b) road の一部を background に取り違える（FN/FP）
    pred[int(h * 0.85):, : int(w * 0.25)] = 0
    # (c) car を少し小さく予測する（過小検出）
    pred[gt == 4] = 3  # いったん road に戻し
    cv2.rectangle(pred, (int(w * 0.62), road_y - 6), (int(w * 0.80), road_y + 10), 4, thickness=-1)
    return gt.astype(np.int64), pred.astype(np.int64)


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認する。
    out = ensure_output_dir()
    dev = pick_device()
    img, src = load_user_or_synthetic_image()
    gt, pred = make_toy_gt_pred()
    vp = voc_colormap()
    panel = [
        np.asarray(img),
        colorize(gt, vp),
        colorize(pred, vp),
    ]
    save_panel(panel, ["input", "toy GT", "toy pred"], out / "00_helpers_smoke.png")
    print(f"device={dev}  input={src}  toy classes={len(TOY_CLASSES)}  saved={out / '00_helpers_smoke.png'}")
