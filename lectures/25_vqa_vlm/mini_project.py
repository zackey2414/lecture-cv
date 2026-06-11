"""章末ミニプロジェクト: VLM「レポートカード」を作る。

この回の要素を1本に統合する総合課題。小さな VQA ベンチマークに対して、

  (A) VQA: モデルに質問応答させ、VQAv2 accuracy / exact-match で採点する
  (B) グラウンディング: 「名前で指定した物体」の位置を当て、
      点-in-箱（point の正しさ）と IoU（box の正しさ）で採点する

の2軸でモデルを評価し、図と JSON の「レポートカード」を出力する。

グラウンディングについて: 本来は moondream2 の model.point / model.detect が
「赤い円はどこ？」に対して点や箱を返す（README 参照）。ただし ~2B でCPUだと重く、
trust_remote_code も要るため、ここでは色マスクの簡易ローカライザで「予測の箱・点」を
作って評価フローを最後まで体験する。実務ではこの予測部分を VLM の出力に差し替えればよい。

このスクリプトはモデルが無い環境でも採点ロジックが動くよう設計してあり、常に exit 0。

実行: uv run python lectures/25_vqa_vlm/mini_project.py
出力: outputs/25_vqa_vlm/mini_*.png / mini_report_card.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import vqa_helpers as H  # noqa: E402


# ----------------------------------------------------------------------
# (A) VQA 採点
# ----------------------------------------------------------------------
def run_vqa(bench: list[dict], device: torch.device) -> tuple[list[dict], str]:
    """ViLT でベンチを採点。モデルが無ければ固定予測で代用し、流れは止めない。"""
    try:
        processor, model = H.load_vilt(device)

        def predict(item):
            enc = processor(item["image"], item["question"], return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**enc).logits
            return model.config.id2label[int(logits.argmax(-1))]

        source = "ViLT"
    except Exception as e:  # noqa: BLE001
        print(f"  VQA モデルをスキップ: {H.hint_for_model_error(e)}")
        fixed = {
            "What color is the circle?": "red",
            "How many shapes are in the image?": "2",
            "Is there a square in the image?": "yes",
            "What color is the triangle?": "green",
        }

        def predict(item):
            return fixed.get(item["question"], "yes")

        source = "固定予測（モデル代替）"

    rows = []
    for item in bench:
        pred = predict(item)
        rows.append(
            {
                "scene": item["scene_id"],
                "question": item["question"],
                "gt": item["gt_answer"],
                "pred": pred,
                "vqa_v2": round(H.vqa_accuracy_vqav2(pred, item["human_answers"]), 3),
                "exact_match": float(H.normalize_answer(pred) == H.normalize_answer(item["gt_answer"])),
            }
        )
    return rows, source


# ----------------------------------------------------------------------
# (B) グラウンディング採点（色マスクで「予測の箱・点」を作る簡易ローカライザ）
# ----------------------------------------------------------------------
def localize_by_color(image, color_name: str) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    """指定色の領域を見つけ、(box=(x1,y1,x2,y2), center=(cx,cy)) を返す。無ければ None。

    本番ではここを moondream.detect/point など VLM の出力に置き換える。教材では
    合成シーンの色が既知なので、色距離のしきい値で確実にローカライズできる。
    """
    rgb = np.asarray(image).astype(np.int16)
    target = np.array(H.COLORS[color_name], dtype=np.int16)
    dist = np.linalg.norm(rgb - target, axis=2)  # 各画素と目標色の距離
    mask = (dist < 60).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    cx, cy = int(xs.mean()), int(ys.mean())
    return (x1, y1, x2, y2), (cx, cy)


def run_grounding(bench: list[dict]) -> list[dict]:
    """target/gt_box を持つ設問だけグラウンディング評価する。"""
    rows = []
    for item in bench:
        if item["target"] is None or item["gt_box"] is None:
            continue
        color_name, shape_name = item["target"]
        pred = localize_by_color(item["image"], color_name)
        gt_box = item["gt_box"]
        if pred is None:
            rows.append({"scene": item["scene_id"], "target": f"{color_name} {shape_name}",
                         "iou": 0.0, "point_hit": False, "pred_box": None})
            continue
        pred_box, center = pred
        rows.append(
            {
                "scene": item["scene_id"],
                "target": f"{color_name} {shape_name}",
                "iou": round(H.box_iou(pred_box, gt_box), 3),
                "point_hit": H.point_in_box(center, gt_box),
                "pred_box": pred_box,
                "center": center,
                "gt_box": gt_box,
                "image": item["image"],
            }
        )
    return rows


# ----------------------------------------------------------------------
# レポート出力
# ----------------------------------------------------------------------
def plot_report(vqa_rows: list[dict], gnd_rows: list[dict], path: pathlib.Path) -> None:
    """左: VQA スコア棒グラフ / 右: グラウンディング箱の重なりを1枚に。"""
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    labels = [f"{r['scene']}:{r['question'][:16]}…" for r in vqa_rows]
    scores = [r["vqa_v2"] for r in vqa_rows]
    y = np.arange(len(vqa_rows))
    ax1.barh(y, scores, color=["#2ca02c" if s >= 0.99 else "#ff7f0e" if s > 0 else "#d62728" for s in scores])
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)
    ax1.set_title("(A) VQA accuracy (VQAv2)")
    ax1.set_xlabel("min(#agree/3, 1)")

    ax2 = fig.add_subplot(1, 2, 2)
    # グラウンディングは代表1件（最初の画像つき行）を描く。
    drawable = next((r for r in gnd_rows if "image" in r), None)
    if drawable is not None:
        ax2.imshow(drawable["image"])
        gx1, gy1, gx2, gy2 = drawable["gt_box"]
        px1, py1, px2, py2 = drawable["pred_box"]
        ax2.add_patch(patches.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                                        fill=False, edgecolor="lime", lw=2, label="GT"))
        ax2.add_patch(patches.Rectangle((px1, py1), px2 - px1, py2 - py1,
                                        fill=False, edgecolor="red", lw=2, ls="--", label="pred"))
        cx, cy = drawable["center"]
        ax2.plot([cx], [cy], "rx", markersize=12, mew=3)
        ax2.legend(loc="upper right")
        ax2.set_title(f"(B) grounding: {drawable['target']}  IoU={drawable['iou']:.2f}")
    ax2.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    bench = H.build_benchmark()
    print(f"device = {device}  benchmark = {len(bench)} questions\n")

    print("=== (A) VQA 採点 ===")
    vqa_rows, source = run_vqa(bench, device)
    for r in vqa_rows:
        print(f"  [{r['scene']}] {r['question'][:30]:30s} gt={r['gt']:5s} pred={r['pred']:8s}"
              f" v2={r['vqa_v2']:.2f} EM={r['exact_match']:.0f}")
    mean_v2 = float(np.mean([r["vqa_v2"] for r in vqa_rows]))
    mean_em = float(np.mean([r["exact_match"] for r in vqa_rows]))
    print(f"  -> mean VQAv2={mean_v2:.3f}  mean exact-match={mean_em:.3f}  (source={source})")

    print("\n=== (B) グラウンディング採点 ===")
    gnd_rows = run_grounding(bench)
    for r in gnd_rows:
        print(f"  [{r['scene']}] {r['target']:16s} IoU={r['iou']:.2f}  point_hit={r['point_hit']}")
    mean_iou = float(np.mean([r["iou"] for r in gnd_rows])) if gnd_rows else 0.0
    point_acc = float(np.mean([r["point_hit"] for r in gnd_rows])) if gnd_rows else 0.0
    print(f"  -> mean IoU={mean_iou:.3f}  point accuracy={point_acc:.3f}")

    plot_report(vqa_rows, gnd_rows, out / "mini_report_card.png")
    report = {
        "vqa": {"source": source, "rows": vqa_rows, "mean_vqa_v2": round(mean_v2, 3),
                "mean_exact_match": round(mean_em, 3)},
        "grounding": {
            "rows": [{k: v for k, v in r.items() if k != "image"} for r in gnd_rows],
            "mean_iou": round(mean_iou, 3),
            "point_accuracy": round(point_acc, 3),
        },
    }
    (out / "mini_report_card.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n完成: VQA(質問応答)とグラウンディング(位置当て)を1つのレポートに統合した。")
    print("  本番では localize_by_color を moondream.detect/point の出力に置き換えればよい。")
    print(f"saved: {out/'mini_report_card.png'}, {out/'mini_report_card.json'}")


if __name__ == "__main__":
    main()
