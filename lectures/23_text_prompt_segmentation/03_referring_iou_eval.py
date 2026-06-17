"""03: 参照セグメの評価（IoU/Dice）— CLIPSeg のしきい値スイープと Grounded-SAM の box 閾値比較。

参照セグメは「文が指す領域」を出すので、評価は『予測マスク vs GT マスク』の重なり具合
（IoU = |∩|/|∪|、Dice = 2|∩|/(|P|+|G|)）で測る。本スクリプトの狙いは2つ:

  A) CLIPSeg は確率マップなので『どこで2値化するか（しきい値）』が IoU を左右する。
     しきい値を 0.05〜0.95 でスイープして IoU 曲線を描き、最良点が必ずしも 0.5 でない
     ことを確かめる（プロンプトや対象でピークがずれる）。

  B) Grounded-SAM は段1（Grounding DINO）の box 閾値が最終マスク IoU を支配する。
     閾値を上げると誤検出は減るが取りこぼし（recall 低下）が増え、下げると過検出になる。
     閾値を変えて『検出数』と『平均マスク IoU』のトレードオフを表で見る。

最後に、同じシーンを CLIPSeg(最良しきい値) と Grounded-SAM で解き、両手法の IoU を並べる。

実行: uv run python lectures/23_text_prompt_segmentation/03_referring_iou_eval.py
出力: lectures/23_text_prompt_segmentation/outputs/03_*.png / 03_eval.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import seg_helpers as H  # noqa: E402

# 02 のパイプライン関数を再利用（単一責務: 検出・SAM の中身は 02 に置いたまま）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib  # noqa: E402

gsam = importlib.import_module("02_grounded_sam")


def sweep_clipseg(probs, scene, gt, thresholds):
    """各プロンプトの IoU をしきい値でスイープし、曲線データと最良しきい値を返す。"""
    curves = {}
    best = {}
    for i, obj in enumerate(scene):
        name = obj["name"]
        if gt is None or name not in gt:
            continue
        ious = [H.mask_iou(probs[i] >= t, gt[name]) for t in thresholds]
        curves[name] = ious
        bt, bv = H.best_threshold(probs[i], gt[name], thresholds)
        best[name] = {"best_threshold": round(bt, 3), "best_IoU": round(bv, 4),
                      "IoU@0.5": round(H.mask_iou(probs[i] >= 0.5, gt[name]), 4)}
    return curves, best


def sweep_box_threshold(image, scene, gt, device, thresholds):
    """Grounding DINO の box 閾値を変え、検出数と平均マスク IoU のトレードオフを測る。

    GDINO の forward は1回だけ実行し、post_process を閾値違いで呼び直す（forward は高コスト、
    post_process は安いため）。SAM は box ごとに必要だが、同一 box はキャッシュで再利用する。
    """
    gdino, gdino_proc = H.load_gdino(device)
    sam, sam_proc = H.load_sam(device)
    prompt_text = ". ".join(o["prompt"] for o in scene) + "."

    inputs = gdino_proc(images=image, text=prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = gdino(**inputs)  # ここが重い。1回だけ。
    w, h = image.size
    gt_boxes = {o["name"]: gsam.mask_to_box(gt[o["name"]]) for o in scene}

    mask_cache: dict[tuple, np.ndarray] = {}
    rows = []
    for thr in thresholds:
        res = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=thr, text_threshold=0.25, target_sizes=[(h, w)]
        )[0]
        boxes = [[int(v) for v in b.tolist()] for b in res["boxes"]]
        ious = []
        for box in boxes:
            key = tuple(box)
            if key not in mask_cache:
                mask_cache[key] = gsam.sam_mask_for_box(sam, sam_proc, image, box, device)
            mask = mask_cache[key]
            # box を GT に対応付けて最終マスク IoU。
            best_name, best_biou = None, 0.0
            for name, gb in gt_boxes.items():
                bi = gsam.box_iou(box, gb) if gb else 0.0
                if bi > best_biou:
                    best_biou, best_name = bi, name
            if best_name is not None:
                ious.append(H.mask_iou(mask, gt[best_name]))
        rows.append({
            "box_threshold": round(float(thr), 2),
            "num_detections": len(boxes),
            "mean_mask_IoU": round(float(np.mean(ious)), 4) if ious else 0.0,
        })
    return rows


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    image, gt, scene = H.load_user_or_scene()
    if gt is None:
        print("実画像モード（GT なし）: 評価は IoU を出せないため、可視化のみの 01/02 を参照。")
        return

    # === A) CLIPSeg しきい値スイープ ===
    model, processor = H.load_clipseg(device)
    prompts = [o["prompt"] for o in scene]
    probs = H.clipseg_probs(model, processor, image, prompts, device)
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    curves, clipseg_best = sweep_clipseg(probs, scene, gt, thresholds)

    # IoU vs threshold 曲線を保存。
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ious in curves.items():
        ax.plot(thresholds, ious, marker="o", ms=3, label=name)
        bt = clipseg_best[name]["best_threshold"]
        ax.axvline(bt, color="gray", ls=":", lw=0.8)
    ax.axvline(0.5, color="red", ls="--", lw=1.0, label="default 0.5")
    ax.set_xlabel("sigmoid threshold")
    ax.set_ylabel("IoU vs GT")
    ax.set_title("CLIPSeg: IoU vs threshold (peak is not always 0.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_clipseg_threshold_sweep.png", dpi=110)
    plt.close(fig)

    # === B) Grounded-SAM の box 閾値スイープ ===
    # 低〜中域に加え高域（0.80〜0.95）も入れる。合成シーンは易しく GDINO の確信度が高いので、
    # 高閾値まで上げて初めて取りこぼし（検出数の減少＝recall 低下）が現れる。
    box_thresholds = [0.25, 0.50, 0.80, 0.85, 0.90, 0.95]
    gsam_rows = sweep_box_threshold(image, scene, gt, device, box_thresholds)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    bx = [r["box_threshold"] for r in gsam_rows]
    ax1.plot(bx, [r["num_detections"] for r in gsam_rows], "s-", color="tab:blue", label="num detections")
    ax1.set_xlabel("Grounding DINO box threshold")
    ax1.set_ylabel("num detections", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(bx, [r["mean_mask_IoU"] for r in gsam_rows], "o-", color="tab:red", label="mean mask IoU")
    ax2.set_ylabel("mean mask IoU", color="tab:red")
    ax1.set_title("Grounded-SAM: box threshold vs (detections / mask IoU)")
    fig.tight_layout()
    fig.savefig(out / "03_gsam_box_threshold.png", dpi=110)
    plt.close(fig)

    # === 仕上げ: 両手法の per-object IoU 比較（CLIPSeg は最良しきい値で）===
    print(f"device = {device}")
    print("[A] CLIPSeg しきい値スイープ（IoU 最大のしきい値）:")
    for name, b in clipseg_best.items():
        print(f"  {name:16s} IoU@0.5={b['IoU@0.5']:.3f}  best_t={b['best_threshold']:.2f}  best_IoU={b['best_IoU']:.3f}")
    print("[B] Grounded-SAM box 閾値スイープ:")
    for r in gsam_rows:
        print(f"  box_thr={r['box_threshold']:.2f}  detections={r['num_detections']}  mean_mask_IoU={r['mean_mask_IoU']:.3f}")

    payload = {
        "clipseg_threshold_sweep": {"thresholds": thresholds.tolist(), "iou_curves": curves, "best": clipseg_best},
        "grounded_sam_box_threshold": gsam_rows,
    }
    (out / "03_eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out / '03_clipseg_threshold_sweep.png'}, {out / '03_gsam_box_threshold.png'}, {out / '03_eval.json'}")


if __name__ == "__main__":
    main()
