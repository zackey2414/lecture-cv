"""第22回 共通の道具箱（インスタンス/パノプティック/SAM）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務を絞ってあります（単一責務の集合）:

  1. device の自動判定（cuda > mps > cpu。本講座は CPU 前提）
  2. 合成シーン画像と「正解（GT）インスタンスマスク」の生成（ネット不要・決定的）
  3. 実画像があればそれを優先するローダ（data/<module_id>/ への導線）
  4. 可視化のための小道具（ラベルマップ→カラー化、uint8 CHW テンソル化、図の保存）

なぜ合成画像か:
  物体検出・セグメンテーションの学習済みモデル（Mask R-CNN / Mask2Former）は
  COCO の「人・車・犬…」を覚えているので、抽象的な合成図形だと自信を持って
  検出できないことが多い。それでも本講座が合成画像を既定にするのは、
    - ネットへ出るのは「モデル重みのDL」だけに限定したい（入力はローカル生成）
    - 評価指標（mask AP / PQ）は『予測とGTの形が分かっていれば』計算でき、
      GT を完全に制御できる合成データの方がむしろ学びやすい
  ためです。実写で試したい人は data/22_instance_panoptic_sam/ に画像を置けば、
  各スクリプトが自動でそちらを使います（README の「実写で試す」を参照）。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

# --- HuggingFace 系のログ・進捗バーを静かにして、教材の出力を読みやすくする ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless（GUIなし）でも図を保存できるバックエンド。imshow は使わない
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "22_instance_panoptic_sam"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = pathlib.Path("data") / MODULE_ID

# 可視化用のはっきり区別できるパレット（RGB, 0-255）。インデックス0は背景=黒に近い灰。
PALETTE: np.ndarray = np.array(
    [
        (40, 40, 40),     # 0 背景
        (220, 50, 50),    # 1 赤
        (50, 110, 230),   # 2 青
        (240, 200, 40),   # 3 黄
        (60, 180, 75),    # 4 緑
        (145, 30, 180),   # 5 紫
        (70, 200, 200),   # 6 シアン
        (245, 130, 48),   # 7 橙
        (170, 110, 40),   # 8 茶
        (128, 128, 128),  # 9 灰
    ],
    dtype=np.uint8,
)


def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提だが、手元に GPU/Apple Silicon があれば自動で使う。
    重要: モデルと入力テンソルの device は必ず揃える（両方に .to(device)）。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir() -> pathlib.Path:
    """lectures/<module_id>/outputs/ を作って返す（既にあってもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_synthetic_scene(
    height: int = 480, width: int = 640
) -> tuple[Image.Image, list[dict]]:
    """合成シーン画像と、その「正解インスタンス（things）」のマスクを返す。

    背景は上半分=空(stuff)、下半分=地面(stuff)。前景(things)として
    赤い円・青い四角・黄色い円を置く。すべて自分で描くので各マスクの
    画素位置は厳密に分かっており、これを GT として SAM の品質確認や
    インスタンスのオーバーレイ表示に使う。

    返り値:
      image: PIL.Image（RGB）
      instances: list of {"mask": (H,W) bool, "label": str, "category_id": int}
                 category_id は本講座内の便宜的な ID（COCO とは無関係）。
    """
    img = np.empty((height, width, 3), dtype=np.uint8)
    # 空（上）と地面（下）を stuff として塗り分ける（グラデーションでそれっぽく）。
    sky = np.linspace(180, 230, height // 2, dtype=np.uint8)[:, None]
    img[: height // 2] = np.stack(
        [np.full_like(sky, 120), np.full_like(sky, 170), sky.clip(190, 235)], axis=-1
    ).repeat(width, axis=1).reshape(height // 2, width, 3)
    img[height // 2 :] = (90, 140, 90)  # 地面（くすんだ緑）

    instances: list[dict] = []

    def add_circle(cx: int, cy: int, r: int, color: tuple[int, int, int], label: str, cid: int) -> None:
        cv2.circle(img, (cx, cy), r, color, thickness=-1)
        mask = np.zeros((height, width), dtype=bool)
        yy, xx = np.ogrid[:height, :width]
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True
        instances.append({"mask": mask, "label": label, "category_id": cid})

    def add_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], label: str, cid: int) -> None:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=-1)
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        instances.append({"mask": mask, "label": label, "category_id": cid})

    # 前景 things（赤円・青四角・黄円）。位置と大きさは決定的。
    add_circle(150, 250, 70, (220, 50, 50), "red_ball", 1)
    add_rect(360, 180, 520, 330, (50, 110, 230), "blue_box", 2)
    add_circle(470, 130, 45, (240, 200, 40), "yellow_ball", 1)

    return Image.fromarray(img), instances


def load_user_or_synthetic() -> tuple[Image.Image, list[dict] | None]:
    """実画像があればそれを、無ければ合成シーン＋GT を返す。

    data/22_instance_panoptic_sam/ に .png/.jpg 等を置くと先頭の1枚を使う。
    その場合 GT マスクは無い（None）ので、SAM の IoU などは「点/箱の位置だけ」で
    動かす。無ければ合成シーン（GT 付き）にフォールバックするので必ず動く。
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if DATA_DIR.is_dir():
        files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
        if files:
            return Image.open(files[0]).convert("RGB"), None
    return build_synthetic_scene()


def to_uint8_chw(image: Image.Image) -> torch.Tensor:
    """PIL(RGB) を draw_* 系が要求する uint8 の (3, H, W) テンソルへ変換する。

    torchvision.utils.draw_bounding_boxes / draw_segmentation_masks は
    『uint8 の画像テンソル』を要求する（float のままだと例外や真っ黒になる）。
    """
    arr = np.array(image.convert("RGB"), dtype=np.uint8)  # (H, W, 3) RGB（書き込み可能なコピー）
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """整数ラベルマップ (H, W) を PALETTE で色付けした RGB 画像 (H, W, 3) にする。"""
    idx = np.asarray(label_map).astype(int) % len(PALETTE)
    return PALETTE[idx]


def save_side_by_side(
    panels: list[tuple[str, np.ndarray]], path: pathlib.Path, suptitle: str = ""
) -> None:
    """(タイトル, RGB画像) の並びを横に並べて1枚に保存する（結果確認用）。

    画像は RGB の uint8 を前提にする。matplotlib は RGB をそのまま表示するので、
    cv2.imread 由来の BGR を渡さないよう注意（本講座は基本 RGB で統一）。
    注: 図中の文字は ASCII にする（既定フォント DejaVu Sans は日本語グリフを持たず
    豆腐□になり警告も出るため、日本語の解説は README とコンソール出力側に置く）。
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.2, 4.4))
    if n == 1:
        axes = [axes]
    for ax, (title, rgb) in zip(axes, panels):
        ax.imshow(rgb)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """2つの bool マスクの IoU = 交差画素 / 和集合画素 を返す（和集合0なら0）。"""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認する。
    out = ensure_output_dir()
    dev = pick_device()
    image, instances = build_synthetic_scene()
    # GT インスタンスを色分けしたラベルマップとして可視化する。
    label_map = np.zeros(image.size[::-1], dtype=int)
    for i, inst in enumerate(instances, start=1):
        label_map[inst["mask"]] = i
    panels = [
        ("synthetic scene", np.asarray(image)),
        ("GT instances", colorize_label_map(label_map)),
    ]
    path = out / "00_scene_and_gt.png"
    save_side_by_side(panels, path, suptitle="seg_helpers smoke test")
    print(f"device={dev}  instances={len(instances)}  saved={path}")
