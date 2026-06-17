"""02: HuggingFace DETR で検出し、post_process と target_sizes=(H,W) の作法を体得する。

DETR（DEtection TRansformer）は、アンカーや NMS を使わず「集合予測（set prediction）」
で検出する Transformer ベースの検出器。N 個（既定100）の物体クエリがそれぞれ
「1物体 or 背景(no-object)」を予測し、二部マッチングで重複を避けるので、原理上は
後段 NMS が不要という点が torchvision の R-CNN / SSD / RetinaNet と大きく異なる。

transformers v5 の正準フロー:
    proc   = AutoImageProcessor.from_pretrained(model_id)
    model  = AutoModelForObjectDetection.from_pretrained(model_id).eval()
    inputs = proc(images=pil, return_tensors="pt")
    outputs = model(**inputs)                              # 生出力（cxcywh 正規化座標）
    result  = proc.post_process_object_detection(          # ← ここで xyxy 絶対座標へ
        outputs, threshold=0.5, target_sizes=[(H, W)])[0]  # {'scores','labels','boxes'}

★最頻バグ: target_sizes は (height, width) 順。PIL の image.size は (width, height) なので
  image.size[::-1] で渡す。逆にすると box が縦横に歪む。本スクリプトは正しい (H,W) と
  誤った (W,H) を両方描いて、歪みを目で確認できるようにする。

実行: uv run python lectures/18_object_detection_intro/02_detr_huggingface.py
出力: lectures/18_object_detection_intro/outputs/02_*.png / 02_detr_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import detection_helpers as H  # noqa: E402

# facebook/detr-resnet-50 は CPU でも現実的な定番。バックボーン読み込みに timm が要る。
# さらに軽い RT-DETR v2（PekingU/rtdetr_v2_r18vd）も同じ AutoModelForObjectDetection で
# 同じ post_process が使える（README の「使い分け」参照）。
DETR_MODEL_ID = "facebook/detr-resnet-50"
SCORE_THRESH = 0.30  # 合成画像は弱いので低め。実写なら 0.5〜0.7。


def load_detr(device: torch.device):
    """DETR のプロセッサとモデルを返す（eval + device 済み）。"""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # 進捗バー以外の情報ログも抑制
    proc = AutoImageProcessor.from_pretrained(DETR_MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(DETR_MODEL_ID).to(device).eval()
    return proc, model


def run_detr(proc, model, image, device: torch.device, target_sizes) -> dict:
    """与えた target_sizes で後処理した検出結果を返す（(H,W) 検証に流用するため引数化）。"""
    inputs = proc(images=image, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.inference_mode():
        outputs = model(**inputs)
    elapsed = time.time() - t0

    # post_process_object_detection が cxcywh 正規化 → xyxy 絶対座標へ変換してくれる。
    result = proc.post_process_object_detection(
        outputs, threshold=SCORE_THRESH, target_sizes=target_sizes
    )[0]
    boxes = result["boxes"].cpu()
    scores = result["scores"].cpu()
    labels = result["labels"].cpu()
    names = [model.config.id2label[int(i)] for i in labels]  # DETR は id2label でクラス名
    return {"boxes": boxes, "scores": scores, "labels": labels, "names": names, "elapsed": elapsed}


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    image, source = H.load_input_image()
    image.save(out / "02_input.png")
    W, H_ = image.size  # PIL は (width, height)

    proc, model = load_detr(device)
    print(f"device = {device}   model = {DETR_MODEL_ID}   input = {source}  size=(W={W},H={H_})")

    # --- 正しい (H, W) で後処理 ---
    correct = run_detr(proc, model, image, device, target_sizes=[(H_, W)])
    # --- わざと (W, H) で後処理（典型バグ）---
    wrong = run_detr(proc, model, image, device, target_sizes=[(W, H_)])

    # 可視化: 入力 / 正しい(H,W) / 誤った(W,H)。
    vis_correct = H.draw_detections(image, correct["boxes"], correct["names"], correct["scores"], color="lime")
    vis_wrong = H.draw_detections(image, wrong["boxes"], wrong["names"], wrong["scores"], color="red")
    H.save_panel(
        [
            (image, f"input ({source})"),
            (vis_correct, f"target_sizes=(H,W)  OK\n{len(correct['names'])} det"),
            (vis_wrong, f"target_sizes=(W,H)  BUG\n{len(wrong['names'])} det (boxes skewed)"),
        ],
        out / "02_detr_target_sizes.png",
        "HF DETR: post_process_object_detection needs target_sizes=(H,W)",
    )

    # NMS 不要の確認: DETR は set prediction なので、同一物体への重複箱が出にくい。
    detections = [
        {"name": n, "label": int(lb), "score": round(float(s), 3),
         "box_xyxy": [round(float(v), 1) for v in b]}
        for n, lb, s, b in zip(correct["names"], correct["labels"], correct["scores"], correct["boxes"])
    ]
    summary = {
        "model": DETR_MODEL_ID,
        "source": source,
        "image_size_WH": [W, H_],
        "score_thresh": SCORE_THRESH,
        "elapsed_sec": round(correct["elapsed"], 3),
        "note": "DETRはset predictionで後段NMS不要。target_sizesは(H,W)順。",
        "detections": detections,
    }
    (out / "02_detr_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    det_names = ", ".join(f"{d['name']}({d['score']})" for d in detections) or "(none)"
    print(f"検出(正しいH,W): {len(detections)}件  time={summary['elapsed_sec']}s  -> {det_names}")
    print("ポイント:")
    print(" - 生出力 pred_boxes は cxcywh 正規化座標。post_process を通して初めて xyxy 絶対座標。")
    print(" - target_sizes は (H,W)。image.size は (W,H) なので [::-1] で渡す（パネル右が歪んだ例）。")
    print(" - DETR は二部マッチングで重複を避けるため、後段 NMS は基本不要。")
    print(f"saved: {out / '02_detr_target_sizes.png'}, {out / '02_detr_results.json'}")


if __name__ == "__main__":
    main()
