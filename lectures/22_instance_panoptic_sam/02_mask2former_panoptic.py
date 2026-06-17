"""02: Mask2Former でパノプティックセグメンテーション（HF・post_process_*）。

パノプティック（panoptic）= 「things（数えられる前景: 人・車…）」のインスタンスと
「stuff（数えられない背景: 空・道路・芝…）」の領域を、画像の全画素にもれなく・
重なりなく1枚に統合したセグメンテーション。インスタンス（things のみ・重なり可）と
セマンティック（クラスのみ・個体を区別しない）の“良いとこ取り”で、各画素に
（カテゴリ, インスタンスID）の組を1つだけ割り当てる。

Mask2Former は「N 個の固定クエリが、それぞれ1枚のマスク＋1個のクラスを予測する」
Transformer 型の統一アーキテクチャ。同じ重みで instance / semantic / panoptic を
後処理だけ変えて出し分けられるのが特徴で、HF では:
  post_process_panoptic_segmentation(outputs, target_sizes=[(H,W)])[0]
    -> {"segmentation": (H,W) long, "segments_info": [{id,label_id,score,was_fused}]}
target_sizes は (height, width) 順（PIL の image.size は (W,H) なので [::-1]）。

合成図形は COCO のクラスに一致しないので、既定しきい値では「自信のある領域なし」
＝ segments_info が空になることが多い（=モデルは壊れていない。抽象図形に確信が
持てないだけ）。その場合でも、本スクリプトは『各クエリが提案するマスク』を
そのまま可視化して、Mask2Former の仕組み（クエリ→マスク＋クラス）を見せる。
実写を data/22_instance_panoptic_sam/ に置けば本物のパノプティック結果になる。

実行: uv run python lectures/22_instance_panoptic_sam/02_mask2former_panoptic.py
出力: lectures/22_instance_panoptic_sam/outputs/02_mask2former_panoptic.png / 02_panoptic.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import seg_helpers as H  # noqa: E402

# swin-tiny の COCO パノプティック版。CPU でも 1 枚 1 秒未満で推論できる軽量モデル。
# swin-large や ViT-Huge は CPU には重すぎるので避ける（README 参照）。
MODEL_ID = "facebook/mask2former-swin-tiny-coco-panoptic"


def load_model(device: torch.device):
    """Mask2Former（panoptic）と画像プロセッサをロードして返す。"""
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID).to(device).eval()
    return model, processor


def run_panoptic(threshold: float = 0.5) -> dict:
    """画像に Mask2Former をかけ、パノプティック結果（or クエリ可視化）を保存する。"""
    device = H.pick_device()
    out_dir = H.ensure_output_dir()
    image, _ = H.load_user_or_synthetic()

    model, processor = load_model(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    # target_sizes は (height, width)。image.size=(W,H) なので [::-1] で渡す。
    target_size = image.size[::-1]
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[target_size], threshold=threshold
    )[0]
    segmentation = result["segmentation"].cpu().numpy()  # (H, W) 各画素=segment id（-1=未割当）
    segments_info = result["segments_info"]  # [{id, label_id, score, was_fused}]

    id2label = model.config.id2label
    segments = [
        {
            "id": s["id"],
            "label": id2label.get(s["label_id"], str(s["label_id"])),
            "score": round(float(s.get("score", 0.0)), 4),
        }
        for s in segments_info
    ]

    # --- 可視化 ---
    if segments:
        panel_title = "panoptic segmentation"
        panel_img = H.colorize_label_map(np.where(segmentation < 0, 0, segmentation + 1))
    else:
        # 確信のある領域が無いとき: 各クエリの提案マスク上位を重ねて『仕組み』を見せる。
        panel_title = "top query masks (illustrative)"
        panel_img = _visualize_query_masks(outputs, model, target_size)

    panels = [("input", np.asarray(image.convert("RGB"))), (panel_title, panel_img)]
    png_path = out_dir / "02_mask2former_panoptic.png"
    H.save_side_by_side(
        panels, png_path, suptitle=f"Mask2Former panoptic: {len(segments)} segments (thr={threshold})"
    )

    payload = {
        "model": MODEL_ID,
        "threshold": threshold,
        "num_segments": len(segments),
        "segments": segments,
    }
    json_path = out_dir / "02_panoptic.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- ログ ---
    print(f"[02] device={device}  panoptic segments={len(segments)} (thr={threshold})")
    if not segments:
        print("  合成図形では確信のある領域が無く segments=0 は想定内です。")
        print("  → クエリの提案マスクを可視化しました。data/ に実写を置くと本物の panoptic に。")
    for s in segments[:10]:
        print(f"  - id={s['id']:>3}  {s['label']:24s} score={s['score']:.3f}")
    print(f"  saved: {png_path.name}, {json_path.name}")
    print("  ※ PQ=SQ×RQ の数値計算は 04_maskap_pq_eval.py で（制御された合成GTを使って）行う。")
    return payload


def _visualize_query_masks(outputs, model, target_size, topk: int = 5) -> np.ndarray:
    """クラス確信度が高い上位クエリの予測マスクを色分けして重ねた RGB を返す。

    Mask2Former は 100 個のクエリそれぞれが (マスク, クラス) を出す。後処理しきい値を
    満たさなくても、ここでは『どんなマスクを提案しているか』を可視化して仕組みを示す。
    """
    h, w = target_size
    class_prob = outputs.class_queries_logits.softmax(-1)[0, :, :-1]  # (Q, C)  最後の列=no-object
    conf, _ = class_prob.max(-1)
    top_idx = conf.topk(topk).indices.tolist()

    mask_logits = outputs.masks_queries_logits[0]  # (Q, h0, w0)
    up = torch.nn.functional.interpolate(
        mask_logits[top_idx][None], size=(h, w), mode="bilinear", align_corners=False
    )[0]
    label_map = np.zeros((h, w), dtype=int)
    for rank, m in enumerate(up.sigmoid() > 0.5, start=1):
        label_map[m.cpu().numpy()] = rank  # 後のクエリほど手前に上書き（簡易）
    return H.colorize_label_map(label_map)


if __name__ == "__main__":
    run_panoptic()
