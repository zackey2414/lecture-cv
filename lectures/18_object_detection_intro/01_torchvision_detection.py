"""01: torchvision の weights API で物体検出の最小推論を書く（3モデル横断）。

この回の主役は「どの検出モデルでも同じ書き方で動く」という torchvision の統一作法。
モデルごとに変わるのは weights enum だけで、

    weights = SomeWeights.DEFAULT          # 学習済み重み（COCO）
    model   = build_model(weights=weights) # ネットワーク本体
    preproc = weights.transforms()         # そのモデル専用の前処理
    names   = weights.meta["categories"]   # クラス名（index0 = __background__）

という4点セットは全モデル共通。本スクリプトでは 1-stage 軽量(ssdlite320)・
1-stage(retinanet)・2-stage 高精度(fasterrcnn) の3つを同じループで回し、
出力 {boxes(xyxy), labels, scores} の読み方と、score 閾値 → NMS → 可視化までを通す。

★検出モデル特有の入力ルール（分類との最大の違い）:
  検出モデルは「[0,1] の float CHW テンソルの *リスト*」を受け取り、ImageNet 正規化は
  モデル内部で行う。だから weights.transforms()（ObjectDetection プリセット）は
  PIL→[0,1] float テンソルへ変換するだけで、自分で mean/std 正規化を重ねてはいけない。

実行: uv run python lectures/18_object_detection_intro/01_torchvision_detection.py
出力: outputs/18_object_detection_intro/01_*.png / 01_torchvision_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

# 同フォルダの helper を import できるよう、最初にパスを通す（スクリプト単体実行のため）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import detection_helpers as H  # noqa: E402  同じフォルダの道具箱

# CPU スレッドを物理コア数程度まで使うと推論が安定して速い（任意・効果は環境依存）。
torch.set_num_threads(min(8, torch.get_num_threads()))

# 可視化・JSON 出力で使う共通の閾値（README と揃える）。
SCORE_THRESH = 0.30  # 合成画像は検出が弱いので低めに。実写なら 0.5 前後が定番。
NMS_IOU = 0.50


def build_models():
    """3つの torchvision 検出モデルを (名前, weights, model) で返す。

    どれも同じ weights API。min_size/max_size を小さめに渡して CPU 推論を軽くする
    （重い resnet50 系でも入力解像度を下げれば1枚あたり 0.1〜0.5 秒に収まる）。
    """
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        RetinaNet_ResNet50_FPN_Weights,
        FasterRCNN_ResNet50_FPN_Weights,
        ssdlite320_mobilenet_v3_large,
        retinanet_resnet50_fpn,
        fasterrcnn_resnet50_fpn,
    )

    specs = [
        # (表示名, weights enum, builder, 種別の説明)
        ("ssdlite320_mobilenet_v3_large", SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
         ssdlite320_mobilenet_v3_large, "1-stage / mobilenet（最軽量・CPU向き）"),
        ("retinanet_resnet50_fpn", RetinaNet_ResNet50_FPN_Weights.DEFAULT,
         retinanet_resnet50_fpn, "1-stage / focal loss（密なアンカー）"),
        ("fasterrcnn_resnet50_fpn", FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
         fasterrcnn_resnet50_fpn, "2-stage / RPN→ROI（高精度・低速）"),
    ]
    models = []
    for name, weights, builder, kind in specs:
        # ssdlite は固定 320 入力なので min/max は渡さない。resnet 系だけ縮めて軽量化。
        if name.startswith("ssdlite"):
            model = builder(weights=weights)
        else:
            model = builder(weights=weights, min_size=320, max_size=512)
        model.eval()  # ★BatchNorm/Dropout を推論モードに（忘れると結果が壊れる）
        models.append((name, weights, model, kind))
    return models


def detect_one(model, weights, image, device: torch.device) -> dict:
    """1モデルで1枚を推論し、閾値・NMS 後の検出を辞書で返す。"""
    preproc = weights.transforms()          # そのモデル専用の前処理（PIL→[0,1] float CHW）
    names = weights.meta["categories"]      # COCO 91クラス（index0 = __background__）

    x = preproc(image).to(device)           # (3,H,W) float [0,1]
    model.to(device)

    t0 = time.time()
    # ★検出モデルは「テンソルのリスト」を受け取る。torch.inference_mode で勾配・メモリを抑制。
    with torch.inference_mode():
        raw = model([x])[0]                 # {'boxes','labels','scores'} の辞書
    elapsed = time.time() - t0

    # 共通後処理: score 閾値 → クラス別 NMS。
    boxes, scores, labels = H.apply_threshold_nms(
        raw["boxes"].cpu(), raw["scores"].cpu(), raw["labels"].cpu(),
        score_thresh=SCORE_THRESH, iou_thresh=NMS_IOU,
    )
    # label(int) → クラス名。index0=__background__ なので生の整数を名前と誤解しない。
    name_list = [names[int(i)] for i in labels]

    detections = [
        {"name": n, "label": int(lb), "score": round(float(s), 3),
         "box_xyxy": [round(float(v), 1) for v in b]}
        for n, lb, s, b in zip(name_list, labels, scores, boxes)
    ]
    return {
        "n_raw": int(raw["scores"].numel()),        # 後処理前の生検出数
        "n_kept": len(detections),                  # 閾値+NMS後
        "elapsed_sec": round(elapsed, 3),
        "detections": detections,
        "_boxes": boxes, "_scores": scores, "_names": name_list,  # 可視化用（json には出さない）
    }


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    image, source = H.load_input_image()
    image.save(out / "01_input.png")

    models = build_models()

    panel_items = [(image, f"input ({source})")]
    summary = {"source": source, "score_thresh": SCORE_THRESH, "nms_iou": NMS_IOU, "models": {}}

    print(f"device = {device}   input source = {source}   size = {image.size}")
    print(f"score_thresh = {SCORE_THRESH}  nms_iou = {NMS_IOU}\n")

    for name, weights, model, kind in models:
        res = detect_one(model, weights, image, device)
        # 可視化（uint8 画像に箱と名前を描く）。
        vis = H.draw_detections(image, res["_boxes"], res["_names"], res["_scores"])
        panel_items.append((vis, f"{name}\n{res['n_kept']} det / {res['elapsed_sec']}s"))

        # json には描画用テンソルを含めない。
        summary["models"][name] = {
            "kind": kind,
            "n_raw": res["n_raw"], "n_kept": res["n_kept"],
            "elapsed_sec": res["elapsed_sec"], "detections": res["detections"],
        }
        det_names = ", ".join(f"{d['name']}({d['score']})" for d in res["detections"]) or "(none)"
        print(f"[{name}] {kind}")
        print(f"   raw={res['n_raw']}  kept={res['n_kept']}  time={res['elapsed_sec']}s  -> {det_names}")

    H.save_panel(panel_items, out / "01_torchvision_compare.png",
                 "torchvision detection (weights API): same code, only the enum changes")
    (out / "01_torchvision_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nポイント:")
    print(" - 4点セット(weights/build/transforms/categories)は全モデル共通。enumを差し替えるだけ。")
    print(" - 出力は {boxes(xyxy,絶対), labels(int), scores}。label の index0 は __background__。")
    print(" - 合成画像は検出が乏しいのが正常。data/18_object_detection_intro/ に実写を置くと実用的。")
    print(f"saved: {out / '01_torchvision_compare.png'}, {out / '01_torchvision_results.json'}")


if __name__ == "__main__":
    main()
