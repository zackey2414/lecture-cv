"""03: SAM（Segment Anything）でプロンプト型セグメンテーション（HF SamModel）。

SAM はこれまでのモデルと発想が違う。クラスを当てるのではなく、『プロンプト
（点 / 箱）で指した“もの”の輪郭マスク』を返す、クラス非依存のセグメンタ。
事前に「赤い円とは何か」を知らなくても、点を置けばそこにある領域を切り出す。
だから合成図形でも実写でも、指した対象をきれいにマスクできる（このモジュールで
唯一、合成画像でも“絵になる”デモ）。

HF での正準フロー（transformers v5）:
  inputs = processor(image, input_points=[[[x, y]]], input_labels=[[1]], return_tensors="pt")
  outputs = model(**inputs)                              # pred_masks:(B,1,3,256,256) iou_scores:(B,1,3)
  masks = processor.post_process_masks(                   # ★低解像マスクを原寸へ戻す（必須）
      outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])
ポイント:
  - 点ラベルは 1=前景 / 0=背景（取り違え注意）。箱は input_boxes=[[[x0,y0,x1,y1]]]。
  - SAM は1プロンプトに対し曖昧性を考慮した3枚のマスクを返す。iou_scores（予測品質,
    1を超えることもある“スコア”であって真の IoU ではない）が最大の1枚を採る。
  - pred_masks は 256x256 の低解像。post_process_masks を通さないと原寸に合わない。

`segment-anything` の pip ではなく HF SAM（facebook/sam-vit-base）を使う。CPU で
さらに軽くするなら SlimSAM（Zigeng/SlimSAM-uniform-77）に MODEL_ID を変えるだけ。

実行: uv run python lectures/22_instance_panoptic_sam/03_sam_prompt_seg.py
出力: lectures/22_instance_panoptic_sam/outputs/03_sam_prompt_seg.png / 03_sam.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import seg_helpers as H  # noqa: E402

MODEL_ID = "facebook/sam-vit-base"  # CPU 既定。さらに軽くするなら Zigeng/SlimSAM-uniform-77


def load_model(device: torch.device):
    """SAM 本体とプロセッサをロードして返す。"""
    from transformers import SamModel, SamProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    processor = SamProcessor.from_pretrained(MODEL_ID)
    model = SamModel.from_pretrained(MODEL_ID).to(device).eval()
    return model, processor


def _best_mask(model, processor, image, device, **prompt) -> tuple[np.ndarray, float]:
    """1プロンプトで推論し、iou_scores 最大のマスク(bool H,W)とそのスコアを返す。"""
    inputs = processor(image, return_tensors="pt", **prompt).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    # 低解像マスク→原寸へ。返りは list（画像ごと）。masks[0]:(1, 3, H, W)。
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]  # (3, H, W) bool
    iou_scores = outputs.iou_scores[0, 0].cpu()  # (3,) 予測品質スコア
    best = int(torch.argmax(iou_scores))
    return masks[best].numpy().astype(bool), float(iou_scores[best])


def run_sam() -> dict:
    """点プロンプトと箱プロンプトで SAM を動かし、可視化・IoU・JSON を保存する。"""
    device = H.pick_device()
    out_dir = H.ensure_output_dir()
    image, instances = H.load_user_or_synthetic()
    W, Hgt = image.size  # PIL は (W, H)

    model, processor = load_model(device)

    # --- プロンプトを決める ---
    # 合成シーンなら GT の重心/外接矩形から作る。実写なら画像中央付近を指す。
    if instances:
        p_inst = instances[0]  # 先頭インスタンス（赤い円）
        ys, xs = np.where(p_inst["mask"])
        point = [float(xs.mean()), float(ys.mean())]  # 重心（前景点）
        b_inst = instances[1]  # 2番目（青い四角）
        ys2, xs2 = np.where(b_inst["mask"])
        box = [float(xs2.min()), float(ys2.min()), float(xs2.max()), float(ys2.max())]
        gt_point_mask = p_inst["mask"]
        gt_box_mask = b_inst["mask"]
    else:
        point = [W / 2.0, Hgt / 2.0]
        box = [W * 0.3, Hgt * 0.3, W * 0.7, Hgt * 0.7]
        gt_point_mask = gt_box_mask = None

    # --- 点プロンプト（label=1 で前景）---
    point_mask, point_score = _best_mask(
        model, processor, image, device,
        input_points=[[[point[0], point[1]]]], input_labels=[[1]],
    )
    # --- 箱プロンプト ---
    box_mask, box_score = _best_mask(
        model, processor, image, device, input_boxes=[[box]],
    )

    img_uint8 = H.to_uint8_chw(image)
    point_overlay = _overlay(img_uint8, point_mask, "lime")
    box_overlay = _overlay(img_uint8, box_mask, "deepskyblue")

    panels = [
        ("input", np.asarray(image.convert("RGB"))),
        (f"point prompt (q={point_score:.2f})", point_overlay),
        (f"box prompt (q={box_score:.2f})", box_overlay),
    ]
    png_path = out_dir / "03_sam_prompt_seg.png"
    H.save_side_by_side(panels, png_path, suptitle="SAM: point / box prompts")

    # --- GT があれば IoU を測って“品質スコア”との違いを見せる ---
    point_iou = H.mask_iou(point_mask, gt_point_mask) if gt_point_mask is not None else None
    box_iou = H.mask_iou(box_mask, gt_box_mask) if gt_box_mask is not None else None

    payload = {
        "model": MODEL_ID,
        "point_prompt": {"point_xy": [round(v, 1) for v in point],
                         "pred_quality": round(point_score, 4),
                         "iou_vs_gt": None if point_iou is None else round(point_iou, 4),
                         "mask_area_px": int(point_mask.sum())},
        "box_prompt": {"box_xyxy": [round(v, 1) for v in box],
                       "pred_quality": round(box_score, 4),
                       "iou_vs_gt": None if box_iou is None else round(box_iou, 4),
                       "mask_area_px": int(box_mask.sum())},
    }
    json_path = out_dir / "03_sam.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[03] device={device}  SAM={MODEL_ID}")
    print(f"  点プロンプト {point}: quality={point_score:.3f}"
          + (f"  IoU(GT)={point_iou:.3f}" if point_iou is not None else "  (実写: GT無し)"))
    print(f"  箱プロンプト {[round(v) for v in box]}: quality={box_score:.3f}"
          + (f"  IoU(GT)={box_iou:.3f}" if box_iou is not None else "  (実写: GT無し)"))
    print("  ※ iou_scores は『予測した品質』であって真の IoU ではない（GT との IoU と区別）。")
    print(f"  saved: {png_path.name}, {json_path.name}")
    return payload


def _overlay(img_uint8, mask_bool: np.ndarray, color: str) -> np.ndarray:
    """1枚の bool マスクを画像に重ねた RGB(H,W,3) を返す。"""
    from torchvision.utils import draw_segmentation_masks

    m = torch.from_numpy(mask_bool)[None]  # (1, H, W)
    canvas = draw_segmentation_masks(img_uint8, m, alpha=0.55, colors=[color])
    return canvas.permute(1, 2, 0).numpy()


if __name__ == "__main__":
    run_sam()
