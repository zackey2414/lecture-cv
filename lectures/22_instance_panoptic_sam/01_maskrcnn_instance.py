"""01: Mask R-CNN でインスタンスセグメンテーション（torchvision・Weights API）。

インスタンスセグメンテーションとは「物体検出（どこに何が）＋画素マスク（その物体の
正確な形）」を同時に出すタスク。セマンティックセグメンテーション（第21回）が
『画素ごとのクラス』だけを返し同じクラスの2匹の犬を区別しないのに対し、
インスタンスは『犬1 / 犬2』を別物として、それぞれにマスクを付ける。

torchvision の maskrcnn_resnet50_fpn_v2 は 1 枚あたり次の dict を返す:
  boxes  : (N, 4) xyxy 絶対座標
  labels : (N,)   COCO のクラス index（0 は __background__）
  scores : (N,)   信頼度
  masks  : (N, 1, H, W) ★確率マップ（0〜1）。>0.5 で二値化して bool にする
この「masks は確率・(N,1,H,W)」という形が最初の関門。squeeze して閾値で bool 化し、
draw_segmentation_masks（uint8画像 + bool マスクを要求）に渡す。

合成図形は COCO のクラスではないので検出は乏しい（0 件のこともある）。それでも
スクリプトは必ず exit 0 で完了する。実写を data/22_instance_panoptic_sam/ に置くと
人や車などが検出され、実用的なオーバーレイになる。

実行: uv run python lectures/22_instance_panoptic_sam/01_maskrcnn_instance.py
出力: lectures/22_instance_panoptic_sam/outputs/01_maskrcnn_instance.png / 01_maskrcnn.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import seg_helpers as H  # noqa: E402


def load_model(device: torch.device):
    """Mask R-CNN（ResNet50-FPN v2）と前処理 transforms をロードして返す。

    Weights enum から重み・前処理・クラス名（meta['categories']）を一括取得するのが
    torchvision の正準パターン。検出モデルの前処理は『[0,1] float の CHW テンソル』化が
    主で、ImageNet 正規化はモデル内部が行う（分類のように二重に正規化しない）。
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    return model, weights


def run_instance_segmentation(score_thr: float = 0.5) -> dict:
    """合成（or 実写）画像に Mask R-CNN をかけ、結果の可視化と JSON を保存する。"""
    device = H.pick_device()
    out_dir = H.ensure_output_dir()
    image, _ = H.load_user_or_synthetic()

    model, weights = load_model(device)
    categories = weights.meta["categories"]  # index→クラス名（0=__background__）
    preprocess = weights.transforms()  # 推論用の前処理（テンソル化）

    # 検出モデルは「[0,1] float CHW テンソルのリスト」を受け取る。1枚でもリストにする。
    x = preprocess(image).to(device)
    with torch.inference_mode():  # eval() + inference_mode() は推論の必須作法
        pred = model([x])[0]

    # スコア閾値でフィルタ。masks(N,1,H,W) は確率なので >0.5 で bool 化し squeeze。
    keep = pred["scores"] >= score_thr
    boxes = pred["boxes"][keep].cpu()
    labels = pred["labels"][keep].cpu()
    scores = pred["scores"][keep].cpu()
    masks_prob = pred["masks"][keep].cpu()  # (M, 1, H, W)
    masks_bool = (masks_prob.squeeze(1) > 0.5)  # (M, H, W) bool

    img_uint8 = H.to_uint8_chw(image)  # (3, H, W) uint8

    # --- 可視化（マスクのオーバーレイ＋ボックスとラベル） ---
    overlay_rgb = _draw_overlay(img_uint8, boxes, labels, scores, masks_bool, categories)

    panels = [("input", np.asarray(image.convert("RGB"))), ("Mask R-CNN", overlay_rgb)]
    png_path = out_dir / "01_maskrcnn_instance.png"
    n = int(keep.sum())
    H.save_side_by_side(panels, png_path, suptitle=f"instance segmentation: {n} objs (score>={score_thr})")

    # --- JSON 保存（boxes / クラス名 / score / マスク面積） ---
    detections = []
    for b, lab, sc, m in zip(boxes.tolist(), labels.tolist(), scores.tolist(), masks_bool):
        detections.append(
            {
                "box_xyxy": [round(v, 1) for v in b],
                "label": categories[lab],
                "score": round(float(sc), 4),
                "mask_area_px": int(m.sum()),
            }
        )
    result = {"num_detections": n, "score_threshold": score_thr, "detections": detections}
    json_path = out_dir / "01_maskrcnn.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- ログ ---
    print(f"[01] device={device}  検出 {n} 件 (score>={score_thr})")
    if n == 0:
        print("  合成図形は COCO クラスではないため検出 0 件は想定内です。")
        print("  → data/22_instance_panoptic_sam/ に実写（人・車など）を置くと検出されます。")
    for d in detections:
        print(f"  - {d['label']:14s} score={d['score']:.3f}  mask={d['mask_area_px']}px")
    print(f"  saved: {png_path.name}, {json_path.name}")
    return result


def _draw_overlay(img_uint8, boxes, labels, scores, masks_bool, categories) -> np.ndarray:
    """マスク・ボックス・ラベルを重ねた RGB 画像（H,W,3 uint8）を返す。

    draw_segmentation_masks は uint8 画像 + bool マスクを要求する。検出 0 件のときは
    入力画像をそのまま返す（描画関数に空テンソルを渡さないための早期分岐）。
    """
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    if masks_bool.shape[0] == 0:
        return img_uint8.permute(1, 2, 0).numpy()

    canvas = draw_segmentation_masks(img_uint8, masks_bool, alpha=0.6)
    label_texts = [f"{categories[int(lab)]} {float(s):.2f}" for lab, s in zip(labels, scores)]
    canvas = draw_bounding_boxes(canvas, boxes, labels=label_texts, colors="yellow", width=2)
    return canvas.permute(1, 2, 0).numpy()


if __name__ == "__main__":
    run_instance_segmentation()
