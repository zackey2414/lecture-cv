"""01: OWL-ViT / OWLv2 で「任意のテキストラベル」による検出を体験する。

オープン語彙検出（open-vocabulary detection）とは: 学習時に決めた固定クラス
（COCO の 80 種など）ではなく、推論時に渡した任意の語で物体を検出すること。
第16回の CLIP ゼロショット『分類』を『検出（=どこに何があるか）』へ拡張したものと捉えると分かりやすい。

このスクリプトの流れ:
  1) まず pipeline('zero-shot-object-detection') で最短体験（成功体験を得る）。
  2) 次に AutoProcessor + AutoModelForZeroShotObjectDetection に分解して手書き。
     post_process_grounded_object_detection / target_sizes=(H,W) を自分で扱う。
  3) OWL-ViT と OWLv2 を同じ候補ラベルで比較し、スコアの出方の違いを見る。
  4) シーンに『無いラベル』(a dog / a traffic light) がほぼ検出されない＝幻覚しにくいことを確認。

実行: uv run python lectures/20_open_vocabulary_detection/01_owlvit_owlv2.py
出力: lectures/20_open_vocabulary_detection/outputs/01_*.png / 01_owl_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


import ovd_helpers as H  # noqa: E402


def quickstart_pipeline(image, labels: list[str]) -> list[dict]:
    """高レベル API で最短検出。返り値は [{'score','label','box':{xmin..}}] のリスト。"""
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    device = H.pick_device()
    detector = pipeline(
        "zero-shot-object-detection",
        model=H.OWLVIT_ID,
        device=0 if device.type == "cuda" else -1,
    )
    # threshold は低めにして取りこぼしを減らす（後段で必要なら絞る）。
    return detector(image, candidate_labels=labels, threshold=0.1)


def detect_and_summarize(model_id: str, image, labels: list[str], gt_labels, gt_boxes, out) -> dict:
    """1つの OWL モデルでシーンを検出し、図を保存して要約 dict を返す。"""
    processor, model = H.load_zero_shot_detector(model_id)
    # OWLv2 は自己学習で高めのスコアを出すので閾値を少し上げる。OWL-ViT は低めに。
    threshold = 0.2 if "owlv2" in model_id else 0.1
    boxes, scores, text_labels = H.detect_owl(processor, model, image, labels, threshold=threshold)

    tag = "owlv2" if "owlv2" in model_id else "owlvit"
    H.draw_detections(
        image,
        boxes,
        text_labels,
        scores,
        f"{tag}: open-vocabulary detection (threshold={threshold})",
        out / f"01_{tag}_detections.png",
        gt_boxes=gt_boxes,
    )

    # 検出を『シーンに有るラベル』『無いラベル』に分けて、幻覚の少なさを見る。
    present = {o for o in gt_labels} if gt_labels else set()
    detections = [
        {
            "label": lab,
            "score": round(float(sc), 3),
            "box": [round(float(v), 1) for v in b],
            "in_scene": (lab in present),
        }
        for b, sc, lab in zip(boxes, scores, text_labels)
    ]
    detections.sort(key=lambda d: -d["score"])
    n_absent_hits = sum(1 for d in detections if not d["in_scene"])
    return {
        "model_id": model_id,
        "threshold": threshold,
        "n_detections": len(detections),
        "n_absent_label_hits": n_absent_hits,  # 無いラベルの誤検出数（小さいほど良い）
        "detections": detections,
    }


def main() -> None:
    out = H.ensure_output_dir()
    image, gt_boxes, gt_labels = H.load_user_or_synthetic()
    labels = H.scene_candidate_labels()  # 有り 4 + 無し 2 の候補ラベル

    # --- (1) pipeline で最短体験 -----------------------------------------
    quick = quickstart_pipeline(image, labels)
    print(f"device = {H.pick_device()}")
    print("[pipeline] zero-shot-object-detection (owlvit):")
    for d in sorted(quick, key=lambda x: -x["score"])[:6]:
        bx = d["box"]
        print(
            f"  {d['score']:.3f}  {d['label']:18s}  "
            f"[{bx['xmin']},{bx['ymin']},{bx['xmax']},{bx['ymax']}]"
        )

    # --- (2)(3) AutoModel で OWL-ViT と OWLv2 を比較 ----------------------
    summaries = []
    for model_id in (H.OWLVIT_ID, H.OWLV2_ID):
        s = detect_and_summarize(model_id, image, labels, gt_labels, gt_boxes, out)
        summaries.append(s)
        print(
            f"\n[{s['model_id']}] threshold={s['threshold']}  "
            f"detections={s['n_detections']}  absent-label hits={s['n_absent_label_hits']}"
        )
        for d in s["detections"][:6]:
            mark = "（シーンに有）" if d["in_scene"] else "（シーンに無→幻覚）"
            print(f"  {d['score']:.3f}  {d['label']:18s} {mark}")

    payload = {
        "candidate_labels": labels,
        "gt_labels": gt_labels,
        "used_synthetic_scene": gt_boxes is not None,
        "pipeline_quickstart": [
            {"label": d["label"], "score": round(float(d["score"]), 3)} for d in quick
        ],
        "models": summaries,
    }
    (out / "01_owl_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nポイント: 候補ラベルは推論時に自由に決められる（=オープン語彙）。")
    print("OWLv2 は OWL-ViT よりスコアが高く出やすい（自己学習で強化されているため）。")
    print("『無いラベル』はほとんど検出されない＝固定クラス分類器と違い幻覚しにくい。")
    print(
        f"saved: {out / '01_owlvit_detections.png'}, {out / '01_owlv2_detections.png'}, "
        f"{out / '01_owl_results.json'}"
    )


if __name__ == "__main__":
    main()
