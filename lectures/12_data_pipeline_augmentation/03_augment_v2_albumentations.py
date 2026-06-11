"""03: データ拡張 — torchvision transforms v2 と albumentations を対比する。

データ拡張(augmentation)は、学習画像にランダムな変形（反転・切り抜き・色ゆらぎ等）を
かけて“水増し”し、過学習を抑えてモデルの汎化を上げる技術です。代表的な2つのライブラリ:

  - torchvision transforms v2: PyTorch 純正。Tensor/PIL をそのまま扱え、分類で定番。
  - albumentations: OpenCV ベースで高速・拡張の種類が豊富。検出/セグメ向けに
    「画像と一緒に bbox / mask も同じ幾何変換で動かす」のが得意（ここが本命の差）。

このスクリプトは、(1) v2 と albumentations で同じ系統の拡張を並べて可視化し、
(2) albumentations で画像・バウンディングボックス・マスクを“同時に”変換できることを
実演します（分類だけなら v2 で十分、検出/セグメなら albumentations が効く、の使い分け）。

実行:
  uv run python lectures/12_data_pipeline_augmentation/03_augment_v2_albumentations.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import albumentations as A
import cv2
import numpy as np
import torch
from torchvision.transforms import v2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline_helpers import (  # noqa: E402
    chw_to_hwc_uint8,
    find_or_make_dataset,
    output_dir,
    save_image_grid,
)
from PIL import Image  # noqa: E402


def augment_grid_v2(pil_img: Image.Image, n: int = 8) -> list[np.ndarray]:
    """transforms v2 の拡張を同じ画像に n 回かけ、HWC uint8 のリストで返す。

    v2 はランダム変換を含むので、同じ入力でも毎回違う結果になる（＝水増し）。
    可視化のため Normalize は付けず、ToDtype(scale=True) までで [0,1] に留める。
    """
    aug = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(96, 96), scale=(0.6, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return [chw_to_hwc_uint8(aug(pil_img)) for _ in range(n)]


def augment_grid_albumentations(np_img: np.ndarray, n: int = 8) -> list[np.ndarray]:
    """albumentations の拡張を同じ画像に n 回かけ、HWC uint8 のリストで返す。

    albumentations は numpy(HWC, uint8) を入出力にする（PIL/Tensor ではない点に注意）。
    ToTensorV2 や Normalize はあえて付けず、見た目で比較できるよう uint8 のまま返す。
    """
    aug = A.Compose([
        A.RandomResizedCrop(size=(96, 96), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.9),
    ])
    return [aug(image=np_img)["image"] for _ in range(n)]


def make_object_scene(size: int = 128) -> tuple[np.ndarray, list[float], np.ndarray]:
    """検出/セグメ拡張のデモ用に、1物体のシーンを作る。

    返り値:
      image: HWC uint8(RGB) の画像（暗い背景に明るい四角の物体）
      bbox : [x_min, y_min, x_max, y_max]（pascal_voc 形式）
      mask : 物体=1 / 背景=0 の 2値マスク（H×W uint8）
    """
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    x0, y0, x1, y1 = 24, 36, 84, 96  # 物体の矩形（左寄り＝反転で位置が変わるのが分かる）
    img[y0:y1, x0:x1] = (60, 200, 120)  # 物体の色（RGB）
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return img, [float(x0), float(y0), float(x1), float(y1)], mask


def draw_box(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    """RGB画像に bbox(pascal_voc) を赤枠で描いて返す（可視化用、非破壊）。"""
    out = img.copy()
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)  # RGBなので赤=(255,0,0)
    return out


def overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB画像にマスク領域を半透明で重ねて返す（可視化用）。"""
    out = img.copy().astype(np.float32)
    color = np.array([255, 255, 0], dtype=np.float32)  # 黄色で重ねる
    sel = mask.astype(bool)
    out[sel] = 0.5 * out[sel] + 0.5 * color
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    out = output_dir()
    np.random.seed(0)
    torch.manual_seed(0)

    # --- 0. 入力画像を1枚用意 -----------------------------------------------
    ds_root = find_or_make_dataset()
    sample_path = sorted((ds_root / "triangle").glob("*.png"))[0]
    pil_img = Image.open(sample_path).convert("RGB")
    np_img = np.asarray(pil_img)  # albumentations 用（HWC uint8）

    # --- 1. v2 と albumentations の拡張を並べて可視化 -----------------------
    print("[拡張の可視化] 同じ画像に拡張を8回ずつ適用")
    v2_imgs = augment_grid_v2(pil_img, n=8)
    save_image_grid(v2_imgs, out / "03_aug_v2.png", ncols=4,
                    suptitle="torchvision transforms v2 (8 random augmentations)")
    alb_imgs = augment_grid_albumentations(np_img, n=8)
    save_image_grid(alb_imgs, out / "03_aug_albumentations.png", ncols=4,
                    suptitle="albumentations (8 random augmentations)")
    print(f"  保存: {out / '03_aug_v2.png'}")
    print(f"  保存: {out / '03_aug_albumentations.png'}")

    # --- 2. albumentations で画像 + bbox + mask を“同時に”変換 --------------
    # ここが検出/セグメで albumentations を使う最大の理由。画像を反転したら、
    # bbox も mask も同じ幾何変換で動かないと『教師データの対応』が崩れる。
    # bbox_params に format='pascal_voc'(=xyxy) を指定し、ラベルは label_fields で渡す。
    print("\n[bbox/mask 同時変換] 画像を反転すると bbox も mask も追従する")
    scene, bbox, mask = make_object_scene()
    joint = A.Compose(
        [
            A.HorizontalFlip(p=1.0),   # 必ず反転（デモなので決定論にする）
            A.Resize(128, 128),        # サイズ変更でも座標は自動追従
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    res = joint(image=scene, mask=mask, bboxes=[bbox], labels=[0])
    new_img, new_mask = res["image"], res["mask"]
    new_bbox = list(res["bboxes"][0])
    print(f"  反転前 bbox(xyxy) = {[round(v, 1) for v in bbox]}")
    print(f"  反転後 bbox(xyxy) = {[round(float(v), 1) for v in new_bbox]}  ← x座標が左右反転")

    # before/after を bbox+mask 付きで並べて保存。
    panel = [
        overlay_mask(draw_box(scene, bbox), mask),
        overlay_mask(draw_box(new_img, new_bbox), new_mask),
    ]
    save_image_grid(panel, out / "03_bbox_mask_joint.png",
                    titles=["before (bbox + mask)", "after HFlip (bbox + mask follow)"],
                    ncols=2, suptitle="albumentations: image + bbox + mask transformed together")
    print(f"  保存: {out / '03_bbox_mask_joint.png'}")

    # --- 3. 使い分けの要点を JSON にも残す ----------------------------------
    summary = {
        "input": sample_path.name,
        "bbox_before": [round(v, 2) for v in bbox],
        "bbox_after_hflip_resize": [round(float(v), 2) for v in new_bbox],
        "mask_area_before": int(mask.sum()),
        "mask_area_after": int(new_mask.sum()),
        "note": "分類なら v2 で十分。検出/セグメは bbox/mask 同時変換ができる albumentations。",
    }
    (out / "03_augment.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n数値を保存: {out / '03_augment.json'}")
    print("要点: 拡張で“水増し”して汎化を上げる。v2=分類で手軽、"
          "albumentations=bbox/mask を同時に動かせて検出/セグメに強い。")


if __name__ == "__main__":
    main()
