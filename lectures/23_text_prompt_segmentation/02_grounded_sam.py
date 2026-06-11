"""02: Grounded-SAM の2段構成（Grounding DINO で box 検出 → SAM で box をマスク化）。

CLIPSeg は1モデルで「文 → マスク」を直接出した（01）。本スクリプトは別アプローチで、
役割の違う2モデルを連結する『検出 → セグメ』のパイプライン（Grounded-SAM）を組む:

  段1: Grounding DINO（オープン語彙検出）に "a red circle. a blue square. ..." を渡し、
        文が指す物体の bounding box を得る（テキストは小文字＋ピリオド区切りが作法）。
  段2: その box を SAM の input_boxes に渡し、box 内の物体を画素マスクに切り出す。

CLIPSeg との対比:
  - CLIPSeg = 軽量・1段・確率ヒートマップ。曖昧な領域（"sky" など）も塗れるが境界はソフト。
  - Grounded-SAM = 2段・重いが、SAM のマスク品質が高く境界がシャープ。検出器を差し替えれば
    語彙も精度も伸ばせる。実務では「検出は得意なモデル、切り出しは SAM」と分業させる発想。

合成画像では Grounding DINO の検出が振るわないことがある（テクスチャが乏しいため）。
その場合でも、検出ゼロを検知して『GT の box を SAM に与えるフォールバック』に切り替え、
SAM 単体のマスク品質は必ず確認できるようにしてある（スクリプトは常に exit 0）。

実行: uv run python lectures/23_text_prompt_segmentation/02_grounded_sam.py
出力: outputs/23_text_prompt_segmentation/02_*.png / 02_grounded_sam.json
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import seg_helpers as H  # noqa: E402


def mask_to_box(mask: np.ndarray) -> list[int] | None:
    """bool マスクから外接 box [x0,y0,x1,y1] を作る（空なら None）。GT box 化に使う。"""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box_iou(a: list[int], b: list[int]) -> float:
    """2つの box の IoU（検出 box を GT に対応付けるために使う）。"""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_gdino(model, processor, image, prompt_text: str, device, threshold: float = 0.25):
    """Grounding DINO で box を検出して [{box, score, label}, ...] を返す。

    テキストは小文字＋各物体をピリオド区切りにするのが作法（"a red circle. a blue square."）。
    post_process の target_sizes は (height, width) 順（PIL の size は (W,H) なので反転）。
    """
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    w, h = image.size
    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=threshold, text_threshold=0.25, target_sizes=[(h, w)]
    )[0]
    dets = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        dets.append({"box": [int(v) for v in box.tolist()], "score": float(score), "label": str(label)})
    return dets


def sam_mask_for_box(model, processor, image, box: list[int], device) -> np.ndarray:
    """SAM に1つの box を渡し、最良の(IoU 最大)マスクを bool 配列で返す。

    SAM は box ごとに候補マスクを3枚返す（multimask）。outputs.iou_scores が
    各候補の自己推定品質なので、その argmax を採用する。post_process_masks で原寸化。
    """
    inputs = processor(image, input_boxes=[[box]], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]  # (n_boxes=1, n_masks=3, H, W)
    scores = outputs.iou_scores[0, 0]  # (3,)
    best = int(torch.argmax(scores))
    return masks[0, best].numpy().astype(bool)


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    image, gt, scene = H.load_user_or_scene()
    # オープン語彙検出のテキスト: 小文字＋ピリオド区切り（Grounding DINO の作法）。
    prompt_text = ". ".join(o["prompt"] for o in scene) + "."

    gdino, gdino_proc = H.load_gdino(device)
    sam, sam_proc = H.load_sam(device)

    # --- 段1: Grounding DINO で検出 ---
    box_threshold = 0.25
    dets = detect_gdino(gdino, gdino_proc, image, prompt_text, device, threshold=box_threshold)
    used_fallback = False

    # 合成画像で検出ゼロなら GT の外接 box をフォールバック入力にする（SAM 単体の品質確認用）。
    if not dets and gt is not None:
        used_fallback = True
        for obj in scene:
            b = mask_to_box(gt[obj["name"]])
            if b is not None:
                dets.append({"box": b, "score": 1.0, "label": obj["name"]})

    # --- 段2: 各 box を SAM でマスク化 ---
    records = []
    sam_masks = []
    gt_boxes = {o["name"]: mask_to_box(gt[o["name"]]) for o in scene} if gt is not None else {}
    for d in dets:
        mask = sam_mask_for_box(sam, sam_proc, image, d["box"], device)
        sam_masks.append(mask)
        rec = {"label": d["label"], "score": round(d["score"], 4), "box": d["box"]}
        # 検出 box を GT に対応付け（box-IoU 最大の GT オブジェクトへ）、最終マスク IoU を測る。
        if gt is not None and gt_boxes:
            best_name, best_biou = None, 0.0
            for name, gb in gt_boxes.items():
                if gb is None:
                    continue
                bi = box_iou(d["box"], gb)
                if bi > best_biou:
                    best_biou, best_name = bi, name
            if best_name is not None:
                rec["matched_gt"] = best_name
                rec["mask_IoU"] = round(H.mask_iou(mask, gt[best_name]), 4)
                rec["mask_Dice"] = round(H.mask_dice(mask, gt[best_name]), 4)
        records.append(rec)

    # --- 可視化: 元画像＋検出 box＋SAM マスクのオーバレイ ---
    _save_grounded_panel(image, dets, sam_masks, out / "02_grounded_sam_panel.png")

    # --- pipeline('mask-generation') との対比（既定は概念のみ。RUN_MASKGEN=1 で軽量実行）---
    maskgen = _maybe_mask_generation(image, device)

    # --- コンソール出力 ---
    src = "GT box フォールバック" if used_fallback else f"Grounding DINO (threshold={box_threshold})"
    print(f"device = {device}  検出元 = {src}  検出数 = {len(dets)}")
    for rec in records:
        line = f"  [{rec['label']}] score={rec['score']:.2f} box={rec['box']}"
        if "mask_IoU" in rec:
            line += f"  → SAM mask IoU={rec['mask_IoU']:.3f} (GT={rec['matched_gt']})"
        print(line)
    print(f"  mask-generation: {maskgen['note']}")

    payload = {
        "prompt_text": prompt_text,
        "box_threshold": box_threshold,
        "used_gt_fallback": used_fallback,
        "detections": records,
        "mask_generation": maskgen,
    }
    (out / "02_grounded_sam.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / '02_grounded_sam_panel.png'}, {out / '02_grounded_sam.json'}")


def _save_grounded_panel(image, dets, masks, path) -> None:
    """[左] 検出 box / [右] SAM マスクの合成オーバレイ を並べて保存。"""
    base = np.asarray(image.convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(base)
    axes[0].set_title("Grounding DINO boxes", fontsize=10)
    colors = [(255, 80, 80), (80, 120, 255), (80, 200, 120), (240, 200, 60)]
    for i, d in enumerate(dets):
        x0, y0, x1, y1 = d["box"]
        col = np.array(colors[i % len(colors)]) / 255.0
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=col, linewidth=2))
        axes[0].text(x0, max(0, y0 - 4), f"{d['label']} {d['score']:.2f}", color=col, fontsize=7)
    # 右: 全 SAM マスクを1枚に色分け合成。
    overlay = base.astype(np.float32)
    for i, m in enumerate(masks):
        col = np.array(colors[i % len(colors)], dtype=np.float32)
        for c in range(3):
            overlay[..., c][m] = 0.45 * base[..., c][m] + 0.55 * col[c]
    axes[1].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[1].set_title("SAM masks (from boxes)", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _maybe_mask_generation(image, device) -> dict:
    """SAM の自動マスク生成（プロンプト無し・点グリッド）。CPU では重いので既定は概念のみ。

    RUN_MASKGEN=1 のときだけ points_per_side を絞って軽量実行する。
    Grounded-SAM（文で指定）との違い: こちらは『画像中の全部位を網羅的に』切り出す。
    """
    if os.environ.get("RUN_MASKGEN") != "1":
        return {
            "ran": False,
            "note": "概念のみ（RUN_MASKGEN=1 で軽量実行）。点グリッドで全領域を網羅的にマスク化する prompt-free 方式。",
        }
    from transformers import pipeline

    gen = pipeline("mask-generation", model=H.SAM_MODEL_ID, device=device)
    outputs = gen(image, points_per_side=8, points_per_batch=16)  # 点を絞って CPU でも現実的に
    n = len(outputs["masks"])
    return {"ran": True, "num_masks": int(n), "note": f"prompt-free で {n} 個のマスクを自動生成"}


if __name__ == "__main__":
    main()
