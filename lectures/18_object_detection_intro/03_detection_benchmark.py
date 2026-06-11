"""03: torchvision と HF DETR を「統一API」で束ね、torchmetrics で mAP を出すベンチ。

第18回の総仕上げ。狙いは2つ:

  (A) 別フレームワークの検出器を、同じ戻り値 {boxes(xyxy), scores, labels, names} に
      正規化する薄いラッパ（Detector）を書くこと。こうしておくと、閾値・NMS・可視化・
      評価をモデル非依存の1経路で回せる（実務のベンチマーク基盤の最小形）。
  (B) torchmetrics.detection.MeanAveragePrecision で mAP@0.5 / mAP@[.5:.95] を出すこと。
      これは「予測 vs 正解(GT)」を IoU マッチングして PR 曲線→AP→平均する指標で、
      検出の定量評価の標準。GT/予測を box 辞書のリストで渡すだけで算出できる。

★mAP の数値を意味あるものにするには「正解ラベル付きデータ」が要る。合成シーンには
  正解アノテーションが無いので、本スクリプトの mAP デモは *決め打ちの GT と予測* で
  計算手順を確認する（下の demo_map）。実モデルの mAP を測るときは、各画像の予測と
  データセットの GT を同じ形式で渡せばよい。numpy で一から組む実装は第19回で扱う。

注: Ultralytics YOLO は概念のみ（後述）。理由は実行経路に入れない方針のため。

実行: uv run python lectures/18_object_detection_intro/03_detection_benchmark.py
出力: outputs/18_object_detection_intro/03_*.png / 03_benchmark.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import detection_helpers as H  # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))

SCORE_THRESH = 0.30
NMS_IOU = 0.50


# =====================================================================
# (A) 統一API: 2フレームワークを同じ戻り値に正規化する薄いラッパ
# =====================================================================

class TorchvisionDetector:
    """torchvision の検出器を統一インターフェースに包む（ssdlite を既定に）。"""

    def __init__(self, device: torch.device):
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        self.name = "torchvision/ssdlite320"
        self.device = device
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.preproc = self.weights.transforms()
        self.categories = self.weights.meta["categories"]  # index0 = __background__
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights).to(device).eval()

    def detect(self, image) -> dict:
        x = self.preproc(image).to(self.device)
        t0 = time.time()
        with torch.inference_mode():
            raw = self.model([x])[0]
        elapsed = time.time() - t0
        boxes, scores, labels = H.apply_threshold_nms(
            raw["boxes"].cpu(), raw["scores"].cpu(), raw["labels"].cpu(),
            score_thresh=SCORE_THRESH, iou_thresh=NMS_IOU,
        )
        names = [self.categories[int(i)] for i in labels]
        return {"boxes": boxes, "scores": scores, "labels": labels, "names": names, "elapsed": elapsed}


class DetrDetector:
    """HF DETR を同じインターフェースに包む（post_process で xyxy 絶対座標に）。"""

    def __init__(self, device: torch.device):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.name = "hf/detr-resnet-50"
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()

    def detect(self, image) -> dict:
        W, Hgt = image.size  # (width, height)
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        elapsed = time.time() - t0
        # ★target_sizes は (H, W)。DETR は NMS 不要なので閾値だけ掛ける。
        res = self.proc.post_process_object_detection(
            outputs, threshold=SCORE_THRESH, target_sizes=[(Hgt, W)]
        )[0]
        labels = res["labels"].cpu()
        names = [self.model.config.id2label[int(i)] for i in labels]
        return {"boxes": res["boxes"].cpu(), "scores": res["scores"].cpu(),
                "labels": labels, "names": names, "elapsed": elapsed}


# =====================================================================
# (B) torchmetrics で mAP を出す（決め打ち GT/予測で計算手順を確認）
# =====================================================================

def demo_map() -> dict:
    """小さな決め打ちの「GT・予測」で mAP@0.5 / mAP@[.5:.95] を算出する。

    2枚分のミニデータセットを模擬:
      画像0: GT=person@[10,10,60,110]、car@[120,40,210,100]
             予測=person(0.95, ほぼ一致)・car(0.80, 少しズレ)・誤検出person(0.40)
      画像1: GT=person@[30,20,90,140]
             予測=person(0.88, 一致)
    これで TP/FP と IoU の効き方が mAP に効く様子を、再現可能な数値で確認できる。
    """
    from torchmetrics.detection import MeanAveragePrecision

    # label 規約はここでは 1=person, 3=car（COCO の id に合わせる必要はなく、
    # 予測と GT で同じ整数を使えばよい。torchmetrics はクラスごとに評価する）。
    preds = [
        dict(
            boxes=torch.tensor([[10., 10., 60., 110.], [122., 42., 205., 98.], [200., 5., 240., 60.]]),
            scores=torch.tensor([0.95, 0.80, 0.40]),
            labels=torch.tensor([1, 3, 1]),
        ),
        dict(
            boxes=torch.tensor([[31., 22., 88., 138.]]),
            scores=torch.tensor([0.88]),
            labels=torch.tensor([1]),
        ),
    ]
    targets = [
        dict(
            boxes=torch.tensor([[10., 10., 60., 110.], [120., 40., 210., 100.]]),
            labels=torch.tensor([1, 3]),
        ),
        dict(
            boxes=torch.tensor([[30., 20., 90., 140.]]),
            labels=torch.tensor([1]),
        ),
    ]
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    out = metric.compute()
    return {
        "map_50_95": round(float(out["map"]), 4),       # mAP@[.5:.95]（COCO 主指標）
        "map_50": round(float(out["map_50"]), 4),       # mAP@0.5（PASCAL 流）
        "map_75": round(float(out["map_75"]), 4),       # mAP@0.75（厳しめ）
        "note": "決め打ちGT/予測。実評価はデータセットGTに予測を合わせて渡す。自作実装は第19回。",
    }


def yolo_concept_note() -> str:
    """Ultralytics YOLO を『概念のみ』にする理由と、参考の1行推論コードを返す。"""
    return (
        "Ultralytics YOLO は `YOLO('yolo11n.pt')` → `model(img)` の1行推論が魅力で、\n"
        "    results[0].boxes.xyxy / .conf / .cls、results[0].plot() で結果を扱える。\n"
        "    ただし本講座では実行経路に入れない: ultralytics は opencv-python(フル版)を\n"
        "    引き込み、本プロジェクトの opencv-python-headless と衝突するため。試す場合は\n"
        "    別環境で `uv add ultralytics`（依存衝突に注意）。YOLO11はNMS内蔵、YOLO26は\n"
        "    NMS-free（後段で再NMSを掛けて二重抑制しないこと）。"
    )


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    image, source = H.load_input_image()
    print(f"device = {device}   input = {source}   size = {image.size}\n")

    # --- (A) 統一API で 2 フレームワークを同条件比較 ---
    detectors = [TorchvisionDetector(device), DetrDetector(device)]
    panel_items = [(image, f"input ({source})")]
    bench = {"source": source, "score_thresh": SCORE_THRESH, "nms_iou": NMS_IOU, "detectors": {}}

    for det in detectors:
        r = det.detect(image)
        vis = H.draw_detections(image, r["boxes"], r["names"], r["scores"])
        panel_items.append((vis, f"{det.name}\n{len(r['names'])} det / {r['elapsed']:.2f}s"))
        bench["detectors"][det.name] = {
            "n_det": len(r["names"]),
            "elapsed_sec": round(r["elapsed"], 3),
            "detections": [
                {"name": n, "score": round(float(s), 3)}
                for n, s in zip(r["names"], r["scores"])
            ],
        }
        names = ", ".join(f"{n}({float(s):.2f})" for n, s in zip(r["names"], r["scores"])) or "(none)"
        print(f"[{det.name}] {len(r['names'])} det  {r['elapsed']:.2f}s  -> {names}")

    H.save_panel(panel_items, out / "03_benchmark_compare.png",
                 "Unified API: torchvision vs HF DETR (same threshold & draw path)")

    # --- (B) torchmetrics で mAP（計算手順の確認）---
    map_res = demo_map()
    bench["map_demo"] = map_res
    print(f"\n[mAP demo] mAP@[.5:.95]={map_res['map_50_95']}  "
          f"mAP@0.5={map_res['map_50']}  mAP@0.75={map_res['map_75']}")
    print("  ↑ 決め打ちGT/予測での計算確認。実モデルのmAPはデータセットGTと突き合わせる。")

    # --- YOLO は概念のみ ---
    note = yolo_concept_note()
    bench["yolo_note"] = note.replace("\n", " ")
    print("\n[YOLO 概念]")
    print("  " + note.replace("\n", "\n  "))

    (out / "03_benchmark.json").write_text(
        json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / '03_benchmark_compare.png'}, {out / '03_benchmark.json'}")


if __name__ == "__main__":
    main()
