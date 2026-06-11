"""01: 画像テンソルの並び（HWC↔CHW）とスケーリング・正規化を“自力で”書く。

深層CVの最初の関門は、見慣れた画像（H×W×C, 0〜255 の uint8）を、モデルが食べる形
（C×H×W, [0,1] の float, さらに mean/std で正規化）へ正しく変換することです。ここで
つまずく典型は2つ:
  - 並び替え(HWC↔CHW)を忘れる/間違える → 形が合わずエラー、または色が壊れる
  - スケーリングを二重にかける/忘れる → 値域が 0..255 のまま Normalize して破綻

このスクリプトは torchvision transforms v2 の正準フロー
    ToImage() -> ToDtype(float32, scale=True) -> Normalize(mean, std)
を1段ずつ分解し、各段で「dtype・shape・値域」がどう変わるかを表示します。さらに
ImageNet 統計と CLIP 専用統計の違い、よくある二重スケーリングの罠も実演します。

実行:
  uv run python lectures/12_data_pipeline_augmentation/01_tensor_layout_normalize.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from torchvision.transforms import v2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline_helpers import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    chw_to_hwc_uint8,
    denormalize,
    find_or_make_dataset,
    output_dir,
    save_image_grid,
)
from PIL import Image  # noqa: E402


def describe(name: str, t: torch.Tensor) -> dict:
    """テンソルの dtype・shape・値域を1行で要約して表示し、dict でも返す。"""
    info = {
        "step": name,
        "dtype": str(t.dtype),
        "shape": tuple(t.shape),
        "min": round(float(t.min()), 4),
        "max": round(float(t.max()), 4),
    }
    print(f"  {name:28s} dtype={info['dtype']:14s} shape={info['shape']!s:18s} "
          f"range=[{info['min']:.3f}, {info['max']:.3f}]")
    return info


def main() -> None:
    out = output_dir()
    torch.manual_seed(0)

    # --- 0. 入力画像を1枚用意（合成データセットの先頭画像）-------------------
    ds_root = find_or_make_dataset()
    sample_path = sorted((ds_root / "circle").glob("*.png"))[0]
    pil_img = Image.open(sample_path).convert("RGB")
    print(f"入力画像: {sample_path.name}  size(W,H)={pil_img.size}")

    # PIL→numpy にすると (H, W, C) の uint8。これが『人間が見慣れた並び』。
    np_hwc = np.asarray(pil_img)
    print(f"\nnumpy(HWC): shape={np_hwc.shape} dtype={np_hwc.dtype} "
          f"range=[{np_hwc.min()}, {np_hwc.max()}]  ← H×W×C, 0..255")

    # --- 1. transforms v2 の正準フローを“1段ずつ”適用して観察 ---------------
    # v2 は変換を小さな部品に分解する設計。各部品が何をするかを順に確認する。
    print("\n[transforms v2 を1段ずつ適用]")
    records: list[dict] = []

    # ToImage: PIL/ndarray を tv_tensors.Image(=Tensor) にする。dtype は uint8 のまま、
    #          並びは CHW になる（ここで HWC→CHW 変換が起きる）。スケーリングはしない。
    to_image = v2.ToImage()
    t_img = to_image(pil_img)
    records.append(describe("ToImage()", t_img))  # uint8, (3,H,W), 0..255

    # ToDtype(scale=True): uint8(0..255) → float32(0..1)。scale=True が『/255』の本体。
    #   ここを scale=False にすると 0..255 のまま float になり、後段 Normalize が破綻する。
    to_float = v2.ToDtype(torch.float32, scale=True)
    t_float = to_float(t_img)
    records.append(describe("ToDtype(float32, scale=True)", t_float))  # float32, 0..1

    # Normalize: チャンネルごとに (x-mean)/std。ImageNet 統計を使う（[0,1] が前提）。
    normalize = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    t_norm = normalize(t_float)
    records.append(describe("Normalize(ImageNet)", t_norm))  # float32, 負値を含む

    # 実務ではこの3段を Compose でまとめる。結果は1段ずつと同じ。
    canonical = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    t_compose = canonical(pil_img)
    assert torch.allclose(t_compose, t_norm), "Compose と1段ずつは一致するはず"
    print("  → Compose([...]) は上の3段と完全一致（assert OK）")

    # --- 2. ImageNet 統計 vs CLIP 専用統計 ----------------------------------
    # 同じ画像でも『どの統計で正規化したか』で値が変わる。混同すると精度が落ちる。
    print("\n[ImageNet 統計 と CLIP 専用統計は別物]")
    t_imagenet = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t_float)
    t_clip = v2.Normalize(CLIP_MEAN, CLIP_STD)(t_float)
    print(f"  ImageNet mean={IMAGENET_MEAN} std={IMAGENET_STD}")
    print(f"  CLIP     mean={CLIP_MEAN} std={CLIP_STD}")
    print(f"  正規化後の平均値:  ImageNet={float(t_imagenet.mean()):+.4f}  "
          f"CLIP={float(t_clip.mean()):+.4f}（統計が違えば結果も違う）")

    # --- 3. よくある罠: 二重スケーリング / スケール忘れ ----------------------
    # (A) 正しい: scale=True で /255 → Normalize は妥当な範囲（おおむね -2.6〜+2.7）
    # (B) 罠: scale=False（/255 を忘れる）と 0..255 のまま Normalize → 値が桁外れに膨らむ
    print("\n[罠: スケーリングを忘れると Normalize が破綻する]")
    wrong_no_scale = v2.ToDtype(torch.float32, scale=False)(t_img)  # 0..255 のまま float
    wrong_norm = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)(wrong_no_scale)
    print(f"  (A) 正しい  scale=True : Normalize 後 range="
          f"[{float(t_norm.min()):.2f}, {float(t_norm.max()):.2f}]  ← 妥当")
    print(f"  (B) 罠 scale=False     : Normalize 後 range="
          f"[{float(wrong_norm.min()):.2f}, {float(wrong_norm.max()):.2f}]  ← 桁外れ＝バグ")
    # v1 ToTensor との関係: v1 の ToTensor() は『CHW化＋/255』を一気にやる（=ToImage+ToDtype相当）。
    # v2 ではこれを2部品に分けた。両方を重ねがけ（ToTensor の後に ToDtype(scale=True)）すると
    # float→float の scale は何もしないので無害だが、設計としてはどちらか一方に統一する。

    # --- 4. 可視化: 正規化テンソルを“戻して”並べて保存 ----------------------
    # Normalize 後をそのまま表示すると色が壊れる。denormalize で [0,1] に戻して確認する。
    panel = [
        np.asarray(pil_img),                                   # 元画像(HWC uint8)
        chw_to_hwc_uint8(t_float),                             # スケール後[0,1]を戻した絵
        chw_to_hwc_uint8(denormalize(t_imagenet, IMAGENET_MEAN, IMAGENET_STD)),
        chw_to_hwc_uint8(denormalize(t_clip, CLIP_MEAN, CLIP_STD)),
    ]
    titles = ["original (HWC uint8)", "scaled [0,1]",
              "ImageNet norm -> denorm", "CLIP norm -> denorm"]
    grid_path = out / "01_layout_normalize.png"
    save_image_grid(panel, grid_path, titles=titles, ncols=4,
                    suptitle="HWC->CHW / scale / normalize (denormalized for display)")
    print(f"\n図を保存: {grid_path}")

    # --- 5. 数値を JSON でも保存（後から検算できるように）-------------------
    summary = {
        "input": sample_path.name,
        "steps": records,
        "imagenet_mean_after_norm": round(float(t_imagenet.mean()), 6),
        "clip_mean_after_norm": round(float(t_clip.mean()), 6),
        "wrong_no_scale_max": round(float(wrong_norm.max()), 4),
    }
    json_path = out / "01_layout_normalize.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"数値を保存: {json_path}")
    print("\n要点: ToImage(並び替え) → ToDtype(scale=True で/255) → Normalize(mean/std)。"
          "ImageNet と CLIP の統計は別物。scale を忘れると Normalize が壊れる。")


if __name__ == "__main__":
    main()
