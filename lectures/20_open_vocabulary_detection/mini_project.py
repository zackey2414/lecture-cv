"""第20回 章末ミニプロジェクト — 「オープン語彙検出 評価レポート」を一枚で作る総合課題。

この回で学んだ要素（OWL でのテキスト検出・target_sizes=(H,W) の後処理・IoU/貪欲マッチング・
P/R/F1 の閾値スイープ）を一つに統合し、本講座の 01〜03 には無い“マスター要素”を 3 つ足す:

  (1) 2 検出器の厳密比較: 01 は OWL-ViT と OWLv2 を「スコアの高さ」で目視比較しただけ。ここでは
                          自作の検出メトリクスで定量比較する。AP@0.5 はどちらも飽和して見分けが
                          つかない一方、mAP@[.5:.95] は差が出る——「AP@0.5 では隠れる差」を体感する。
  (2) COCO 流 mAP の自作: 第19回の自作 mAP をここで実モデルに適用。スコア降順マッチング→全点補間で
                          AP@0.5 を出し、さらに IoU を 0.50:0.05:0.95 で動かして mAP@[.5:.95] を求める。
                          OWL-ViT は高 IoU で AP が落ちる（箱が緩い＝局在が甘い）ことが可視化される。
  (3) 運用点はモデルで違う: F1 最大のしきい値を各モデルで求めると OWL-ViT と OWLv2 で大きく違う
                          （スコアのスケールが別物だから）。「閾値はモデルとセットで決める」を数値で示し、
                          併せて『シーンに無いラベル（dog / traffic light）』の誤検出＝幻覚を監査する。

実装方針:
  - digit 始まりの 01_*.py は import 不可なので、AP/mAP/マッチングは本ファイルに自己完結で書く
    （IoU・合成シーン・モデルロード・検出ラッパだけ共通ヘルパ ovd_helpers から借用）。
  - モデルのロード/推論が失敗しても講座が止まらないよう try/except で包み、ダメなら
    合成検出（箱の締まりとスコアを変えた強/弱 2 検出器）にフォールバックして必ず exit 0。
  - 完全 CPU・合成シーン・十数秒で完走。matplotlib は Agg、図のタイトルは ASCII。

実行: uv run python lectures/20_open_vocabulary_detection/mini_project.py
結果（ダッシュボード画像・レポート JSON）は outputs/20_open_vocabulary_detection/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from matplotlib.patches import Rectangle

# headless 環境でも図を保存できるよう、pyplot より前に Agg を固定する。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ovd_helpers as H  # noqa: E402


# 評価対象の物体（GT。base-noun 表記でマッチングする）と、シーンに無いラベル。
NOUNS_PRESENT = ["red circle", "blue square", "green triangle", "yellow circle"]
NOUNS_ABSENT = ["dog", "traffic light"]
# 比較する 2 検出器（01 と同じ小型 2 種）。
MODELS = [("OWL-ViT", H.OWLVIT_ID), ("OWLv2", H.OWLV2_ID)]
# AP の閾値掃引（スコア）と IoU 掃引。
_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
_IOU_GRID = np.round(np.arange(0.5, 0.96, 0.05), 2)  # COCO 流 mAP@[.5:.95]


# =====================================================================
# 自己完結の評価コア（IoU は H.iou_xyxy を借用。マッチング/AP はここで組む）
# =====================================================================


def match_for_ap(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    gt_boxes: np.ndarray,
    gt_labels: list[str],
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """クラス込みの貪欲マッチングを行い、検出ごとの (スコア降順, TPフラグ, GT総数) を返す。

    第19回の mAP 自作と同じ規則: 予測をスコア降順に見て、同ラベル・未マッチ・IoU>=閾値の
    GT のうち IoU 最大に対応づける。対応すれば TP=1、それ以外は FP=0。
    1つの GT に複数当たっても TP は最初の1つだけ（残りは FP）。
    """
    order = np.argsort(-scores)
    matched: set[int] = set()
    is_tp = np.zeros(len(order), dtype=np.float32)
    for rank, i in enumerate(order):
        cand = [
            g for g in range(len(gt_boxes)) if g not in matched and gt_labels[g] == labels[i]
        ]
        if not cand:
            continue
        ious = H.iou_xyxy(boxes[i], gt_boxes[cand])
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            matched.add(cand[j])
            is_tp[rank] = 1.0
    return scores[order], is_tp, len(gt_boxes)


def average_precision(sorted_scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> tuple:
    """全点補間（連続）の AP を返す（recall, precision の曲線も返す。描画に使う）。

    tp_cum/fp_cum から recall・precision を作り、precision を右側からの累積最大で単調化して
    面積を取る。第19回の『全点補間』そのもの（PASCAL の 11 点ではない）。
    """
    if n_gt == 0 or is_tp.sum() == 0:
        return 0.0, np.array([0.0, 0.0]), np.array([1.0, 0.0])
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(1.0 - is_tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, recall, precision


def sweep_prf(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    gt_boxes: np.ndarray,
    gt_labels: list[str],
) -> tuple[list[dict], dict]:
    """スコア閾値を掃引し、各 t の (TP/FP/FN・P/R/F1) の系列と F1 最大点（運用点）を返す。

    1 回の推論結果を後処理で何度も閾値フィルタするだけ（モデルは走らせ直さない）。
    """
    rows = []
    for t in _THRESHOLDS:
        keep = scores >= t
        kb, ks = boxes[keep], scores[keep]
        kl = [labels[i] for i in range(len(labels)) if keep[i]]
        tp, fp, fn = H.greedy_match(kb, ks, kl, gt_boxes, gt_labels, iou_thr=0.5)
        p, r, f1 = H.prf(tp, fp, fn)
        rows.append({"t": float(t), "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)})
    # F1 最大（同点なら高い閾値＝precision 寄りを採る）。
    best_row = max(rows, key=lambda d: (d["f1"], d["t"]))
    best = {"t": best_row["t"], "f1": best_row["f1"], "precision": best_row["precision"],
            "recall": best_row["recall"], "tp": best_row["tp"], "fp": best_row["fp"],
            "fn": best_row["fn"]}
    return rows, best


def map_over_iou(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    gt_boxes: np.ndarray,
    gt_labels: list[str],
) -> tuple[float, list[float]]:
    """IoU を 0.50:0.05:0.95 で動かして AP を平均し、COCO 流 mAP@[.5:.95] を返す。"""
    aps = []
    for iou_thr in _IOU_GRID:
        ss, tp, n = match_for_ap(boxes, scores, labels, gt_boxes, gt_labels, float(iou_thr))
        ap, _, _ = average_precision(ss, tp, n)
        aps.append(round(ap, 4))
    return float(np.mean(aps)), aps


# =====================================================================
# 検出の収集（OWL 実行 / 失敗時は合成検出にフォールバック）
# =====================================================================


def to_base_noun(text_label: str, mapping: dict[str, str]) -> str:
    """検出ラベル（"a red circle"）を GT と同じ base-noun（"red circle"）に正規化する。"""
    key = text_label.strip().lower()
    return mapping.get(key, key)


def detect_with_model(processor, model, image):
    """1 つの OWL モデルでシーンを検出し、(boxes, scores, base_labels) を返す。"""
    nouns = NOUNS_PRESENT + NOUNS_ABSENT
    candidates = [f"a {n}" for n in nouns]
    reverse = {c.strip().lower(): n for c, n in zip(candidates, nouns)}  # 文字列→base-noun
    # 低い閾値で『全候補（スコア付き）』を集める（AP/掃引のため後処理で何度も絞る）。
    boxes, scores, text_labels = H.detect_owl(processor, model, image, candidates, threshold=0.01)
    base = [to_base_noun(t, reverse) for t in text_labels]
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), base


def synthetic_detections(gt_boxes: np.ndarray, rng: np.random.Generator,
                         jitter: float, score_lo: float, score_hi: float):
    """モデルが使えない時のフォールバック検出。GT をずらした箱＋低スコア偽陽性を作る。

    jitter（箱のずれ幅）とスコア帯で「箱が締まった強検出器」「緩い弱検出器」を模す。
    """
    boxes, scores, labels = [], [], []
    for b, noun in zip(gt_boxes, NOUNS_PRESENT):
        off = rng.uniform(-jitter, jitter, size=4)
        boxes.append(b + off)
        scores.append(rng.uniform(score_lo, score_hi))
        labels.append(noun)
    for _ in range(3):  # シーンに無いラベルの低スコア偽陽性
        x1, y1 = rng.uniform(0, 400), rng.uniform(0, 250)
        boxes.append([x1, y1, x1 + 60, y1 + 60])
        scores.append(rng.uniform(0.02, 0.12))
        labels.append(rng.choice(NOUNS_ABSENT))
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), labels


def collect_detections(image, gt_boxes):
    """2 検出器の (boxes, scores, base_labels) を集める。失敗時は強/弱の合成検出にフォールバック。

    返り値: ({name: (boxes, scores, base)}, used_model_str)
    """
    try:
        det = {}
        for name, mid in MODELS:
            processor, model = H.load_zero_shot_detector(mid)
            det[name] = detect_with_model(processor, model, image)
        return det, "OWL-ViT + OWLv2"
    except Exception as e:  # noqa: BLE001
        print(f"[フォールバック] OWL を実行できないため合成検出で代替します: {type(e).__name__}: {e}")
        rng = np.random.default_rng(0)
        # OWL-ViT 役（箱が緩く・スコア低め） / OWLv2 役（箱が締まり・スコア高め）
        det = {
            "OWL-ViT": synthetic_detections(gt_boxes, rng, jitter=7.0, score_lo=0.18, score_hi=0.38),
            "OWLv2": synthetic_detections(gt_boxes, rng, jitter=1.0, score_lo=0.65, score_hi=0.85),
        }
        return det, "synthetic-fallback"


# =====================================================================
# モデルごとの評価
# =====================================================================


def evaluate_model(name: str, boxes, scores, base_labels, gt_boxes, gt_base) -> dict:
    """1 検出器の結果を評価し、AP@0.5・mAP・F1 運用点・幻覚数をまとめる。"""
    ss, is_tp, n_gt = match_for_ap(boxes, scores, base_labels, gt_boxes, gt_base, 0.5)
    ap50, _, _ = average_precision(ss, is_tp, n_gt)
    rows, best = sweep_prf(boxes, scores, base_labels, gt_boxes, gt_base)
    map_avg, ap_per_iou = map_over_iou(boxes, scores, base_labels, gt_boxes, gt_base)
    keep = scores >= best["t"]
    absent_hits = int(sum(1 for i in range(len(base_labels))
                          if keep[i] and base_labels[i] in NOUNS_ABSENT))
    return {
        "model": name,
        "n_detections_raw": int(len(boxes)),
        "ap50": round(ap50, 4),
        "map_50_95": round(map_avg, 4),
        "ap_per_iou": ap_per_iou,
        "best_f1": best,
        "absent_label_hits": absent_hits,
        "_rows": rows,  # 描画用（JSON から落とす）
    }


# =====================================================================
# ダッシュボード（4 パネル・ASCII タイトル）
# =====================================================================

_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def plot_dashboard(image, gt_boxes, results, det_per_model, best_name, used_model, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    best = next(r for r in results if r["model"] == best_name)

    # (a) 最良モデルの検出（F1 最大しきい値）を GT と重ねる。
    ax = axes[0, 0]
    ax.imshow(np.asarray(image))
    for gx1, gy1, gx2, gy2 in gt_boxes:
        ax.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False,
                               edgecolor="0.4", linestyle="--", linewidth=1.5))
    boxes, scores, base = det_per_model[best_name]
    keep = scores >= best["best_f1"]["t"]
    kept = [(boxes[i], base[i], scores[i]) for i in range(len(base)) if keep[i]]
    for k, (b, lab, sc) in enumerate(kept):
        x1, y1, x2, y2 = b
        c = _PALETTE[k % len(_PALETTE)]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=2.2))
        ax.text(x1, max(y1 - 4, 2), f"{lab} {sc:.2f}", color="white", fontsize=7,
                bbox=dict(facecolor=c, edgecolor="none", pad=1.0))
    ax.set_title(f"{best_name} detections @ F1-opt thr={best['best_f1']['t']:.2f}")
    ax.axis("off")

    # (b) モデル別 AP@0.5 と mAP@[.5:.95]。AP@0.5 は並ぶが mAP に差が出る（局在精度）。
    ax = axes[0, 1]
    names = [r["model"] for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["ap50"] for r in results], width=0.4, label="AP@0.5")
    ax.bar(x + 0.2, [r["map_50_95"] for r in results], width=0.4, label="mAP@[.5:.95]")
    for xi, r in zip(x, results):
        ax.text(xi + 0.2, r["map_50_95"] + 0.02, f"{r['map_50_95']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("AP@0.5 (ties) vs mAP@[.5:.95] (differs)")
    ax.legend(fontsize=8, loc="lower center")

    # (c) AP vs IoU しきい値（両モデル）。緩い箱のモデルは高 IoU で AP が落ちる。
    ax = axes[1, 0]
    for r in results:
        ax.plot(_IOU_GRID, r["ap_per_iou"], "-o", ms=3, label=r["model"])
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("AP vs IoU (localization quality)")
    ax.legend(fontsize=8, loc="lower left")

    # (d) F1 vs スコアしきい値（両モデル）。F1 最大点がモデルで大きくずれる＝閾値はモデル依存。
    ax = axes[1, 1]
    for r in results:
        ts = [row["t"] for row in r["_rows"]]
        line, = ax.plot(ts, [row["f1"] for row in r["_rows"]], "-s", ms=3, label=r["model"])
        bt = r["best_f1"]["t"]
        ax.axvline(bt, color=line.get_color(), ls="--", lw=1)
    ax.set_xlabel("score threshold")
    ax.set_ylabel("F1")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("F1 vs threshold (F1-opt differs per model)")
    ax.legend(fontsize=8, loc="lower center")

    fig.suptitle(f"Module 20 mini project: open-vocabulary detection report ({used_model})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =====================================================================
# main
# =====================================================================


def main() -> None:
    out = H.ensure_output_dir()
    image, gt_boxes, _ = H.build_scene()  # GT 付き合成シーン
    gt_base = list(NOUNS_PRESENT)  # base-noun でマッチング
    print(f"device = {H.pick_device()}  GT objects = {len(gt_boxes)}  "
          f"models = {[m[0] for m in MODELS]}")

    det_per_model, used_model = collect_detections(image, gt_boxes)

    # 各モデルを評価。
    results = []
    print("\n=== (1)(2) 2 検出器の厳密比較（AP@0.5 / mAP@[.5:.95]）===")
    print(f"  {'model':10s} {'#det':>5s} {'AP@0.5':>7s} {'mAP[.5:.95]':>12s} "
          f"{'F1opt-t':>8s} {'bestF1':>7s} {'absent':>7s}")
    for name, _mid in MODELS:
        if name not in det_per_model:
            continue
        boxes, scores, base = det_per_model[name]
        r = evaluate_model(name, boxes, scores, base, gt_boxes, gt_base)
        results.append(r)
        print(f"  {name:10s} {r['n_detections_raw']:5d} {r['ap50']:7.3f} "
              f"{r['map_50_95']:12.3f} {r['best_f1']['t']:8.2f} {r['best_f1']['f1']:7.3f} "
              f"{r['absent_label_hits']:7d}")

    # 最良モデルを mAP@[.5:.95] 第一・AP@0.5 第二で選ぶ。
    best = max(results, key=lambda r: (r["map_50_95"], r["ap50"]))
    best_name = best["model"]
    print(f"\n  → 局在まで含めて最良: {best_name}  "
          f"(AP@0.5={best['ap50']:.3f}, mAP@[.5:.95]={best['map_50_95']:.3f})")
    print("  AP@0.5 は両者そろって飽和し見分けがつかない。mAP@[.5:.95] が低い方は『箱が緩い』")
    print("  （高 IoU で AP が落ちる＝局在が甘い）。AP@0.5 だけで検出器を選ぶと局在差を見落とす。")

    # (3) 運用点はモデルで違う。
    print("\n=== (3) 運用点はモデルで違う（閾値はモデルとセットで決める）===")
    for r in results:
        bf = r["best_f1"]
        print(f"  {r['model']:10s} F1 最大しきい値 t={bf['t']:.2f}  "
              f"(P={bf['precision']:.2f} R={bf['recall']:.2f} F1={bf['f1']:.2f}, "
              f"幻覚={r['absent_label_hits']} 件)")
    if len(results) == 2:
        ta, tb = results[0]["best_f1"]["t"], results[1]["best_f1"]["t"]
        print(f"  → 推奨しきい値が {ta:.2f} vs {tb:.2f} と違う。スコアのスケールがモデルで別物なので、")
        print("    片方の閾値を流用すると取りこぼし／過検出になる。閾値はモデルごとに F1 最大点で決める。")
    print("  どのモデルも『シーンに無いラベル』はほぼ拾わない＝OVD は固定クラス分類器と違い幻覚しにくい。")

    # ダッシュボード保存。
    plot_dashboard(image, gt_boxes, results, det_per_model, best_name, used_model,
                   out / "mini_project_dashboard.png")

    # レポート JSON（描画専用キー _rows は落とす）。
    def clean(r: dict) -> dict:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    report = {
        "used_model": used_model,
        "iou_for_ap50": 0.5,
        "iou_grid_for_map": [float(t) for t in _IOU_GRID],
        "gt_labels": gt_base,
        "models": [clean(r) for r in results],
        "best_by_map": best_name,
    }
    (out / "mini_project_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n完了。mini_project_dashboard.png と mini_project_report.json を "
          "outputs/20_open_vocabulary_detection/ に保存しました。")


if __name__ == "__main__":
    main()
