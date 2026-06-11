"""03: 閾値スイープで precision / recall / F1 を測り、閾値選択を定量化する。

オープン語彙検出の実務では『どの閾値で切るか』が品質を左右する（02 で見たとおり、
緩いと過検出、厳しいと取りこぼし）。本スクリプトは GT 付きの合成シーンを使い、

  1) モデルを十分低い閾値で1回だけ走らせて『全候補（スコア付き）』を得る。
  2) スコア閾値 t を 0.05〜0.95 で掃引し、t 以上の検出だけ残して
     IoU>=0.5 のクラス込み貪欲マッチング（第19回と同じ）で TP/FP/FN を数える。
  3) 各 t の precision / recall / F1 を出し、F1 が最大になる t を『推奨閾値』とする。
  4) OWLv2 と Grounding DINO で同じことをやって比較し、PR 曲線も描く。

合成シーンは検出が容易なので F1 はきれいに 1.0 近くまで上がる。実写を data/ に置くと
もっと現実的な（precision と recall がトレードオフする）曲線になる。

実行: uv run python lectures/20_open_vocabulary_detection/03_threshold_sweep_eval.py
出力: outputs/20_open_vocabulary_detection/03_*.png / 03_sweep_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ovd_helpers as H  # noqa: E402

# 掃引するスコア閾値（0.05 刻み）。
_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def sweep_one(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    gt_boxes: np.ndarray,
    gt_labels: list[str],
) -> dict:
    """1モデルの全候補に対し、スコア閾値を掃引して P/R/F1 の系列を作る。"""
    rows = []
    for t in _THRESHOLDS:
        keep = scores >= t  # t 以上の検出だけ採用
        kb, ks = boxes[keep], scores[keep]
        kl = [labels[i] for i in range(len(labels)) if keep[i]]
        tp, fp, fn = H.greedy_match(kb, ks, kl, gt_boxes, gt_labels, iou_thr=0.5)
        p, r, f1 = H.prf(tp, fp, fn)
        rows.append(
            {
                "t": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f1, 3),
            }
        )
    # F1 最大の点（同点なら閾値が高い＝precision 寄りの方を選ぶ）。
    best = max(rows, key=lambda d: (d["f1"], d["t"]))
    return {"rows": rows, "best": best}


def collect_detections(model_kind: str, image, labels: list[str]):
    """モデル種別に応じて『低閾値で全候補』を集める。(boxes, scores, labels) を返す。"""
    if model_kind == "owlv2":
        processor, model = H.load_zero_shot_detector(H.OWLV2_ID)
        return H.detect_owl(processor, model, image, labels, threshold=0.01)
    if model_kind == "gdino":
        processor, model = H.load_zero_shot_detector(H.GDINO_ID)
        caption = H.labels_to_caption(labels)
        return H.detect_gdino(
            processor, model, image, caption, box_threshold=0.05, text_threshold=0.05
        )
    raise ValueError(model_kind)


def plot_sweep(name: str, sweep: dict, path: pathlib.Path) -> None:
    """P/R/F1 の閾値スイープ曲線を描き、F1 最大点に縦線を引く。"""
    rows = sweep["rows"]
    ts = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ts, [r["precision"] for r in rows], "-o", ms=3, label="precision")
    ax.plot(ts, [r["recall"] for r in rows], "-s", ms=3, label="recall")
    ax.plot(ts, [r["f1"] for r in rows], "-^", ms=3, label="F1")
    bt = sweep["best"]["t"]
    ax.axvline(bt, color="0.5", ls="--", lw=1)
    ax.text(bt, 0.05, f" best F1 @ t={bt:.2f}", fontsize=8, color="0.3")
    ax.set_xlabel("score threshold")
    ax.set_ylabel("metric")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{name}: precision / recall / F1 vs threshold")
    ax.legend(loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_pr(sweeps: dict[str, dict], path: pathlib.Path) -> None:
    """両モデルの PR 曲線を1枚に重ねて描く。"""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for name, sweep in sweeps.items():
        rows = sweep["rows"]
        rec = [r["recall"] for r in rows]
        prec = [r["precision"] for r in rows]
        ax.plot(rec, prec, "-o", ms=3, label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("PR curve (threshold sweep, IoU>=0.5)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    image, gt_boxes, gt_labels = H.load_user_or_synthetic()
    if gt_boxes is None:
        # 実画像（GT 無し）のときは評価できないので合成シーンに切り替える。
        print("[注意] data/ の実画像には GT が無いため、評価は合成シーンで行います。")
        image, gt_boxes, gt_labels = H.build_scene()

    labels = H.scene_candidate_labels()
    print(f"device = {H.pick_device()}  GT objects = {len(gt_boxes)}")

    sweeps: dict[str, dict] = {}
    for kind, name in [("owlv2", "OWLv2"), ("gdino", "Grounding DINO")]:
        try:
            boxes, scores, det_labels = collect_detections(kind, image, labels)
            sw = sweep_one(boxes, scores, det_labels, gt_boxes, gt_labels)
            sweeps[name] = sw
            plot_sweep(name, sw, out / f"03_sweep_{kind}.png")
            b = sw["best"]
            print(
                f"\n[{name}] 全候補 {len(boxes)} 件。F1 最大: "
                f"t={b['t']:.2f}  P={b['precision']:.2f} R={b['recall']:.2f} F1={b['f1']:.2f} "
                f"(TP={b['tp']} FP={b['fp']} FN={b['fn']})"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[フォールバック] {name} をスキップ: {type(e).__name__}: {e}")

    if sweeps:
        plot_pr(sweeps, out / "03_pr_curve.png")

    payload = {
        "iou_threshold": 0.5,
        "thresholds": [float(t) for t in _THRESHOLDS],
        "gt_labels": gt_labels,
        "models": {name: {"best": sw["best"], "rows": sw["rows"]} for name, sw in sweeps.items()},
    }
    (out / "03_sweep_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nポイント: 低すぎる閾値は過検出で precision が落ち、高すぎると recall が落ちる。")
    print("F1 最大点が『その画像・その候補ラベルに対する妥当な閾値』の一案になる。")
    print(f"saved: {out / '03_pr_curve.png'}, {out / '03_sweep_results.json'} ほか")


if __name__ == "__main__":
    main()
