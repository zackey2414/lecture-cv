"""mini_project: 第18回 総合課題 — 「検出パイプライン × 自作 mAP」を1本に統合する。

本章の学びを統合する capstone。やることは2つ:

  Part A 検出パイプライン:
    torchvision の 4点セット（weights/build/transforms/categories）で検出器をロードし、
    [0,1] float CHW リスト入力 → eval+inference_mode 推論 → score 閾値 + クラス別 NMS
    （detection_helpers）→ draw_bounding_boxes 可視化、という定番経路をひと続きで回す。
    合成画像のときは make_scene_image が描く図形の既知座標を pseudo-GT とし、検出器の
    mAP@0.5 を測る → COCO 学習器は抽象図形に弱く mAP がほぼ 0 になる「ドメインギャップ」を
    定量的に観察する（exit 0。低い値＝故障ではない）。

  Part B 自作 mAP の検算（第19回の予告）:
    IoU マッチング → PR 曲線 → 全点補間 AP → クラス平均、という mAP@0.5 の全手順を
    numpy/torch で自作し、torchmetrics.detection.MeanAveragePrecision の map_50 と
    突き合わせる（小さな差は補間方式: 自作=全点補間 vs torchmetrics=COCO 101点）。
    PR 曲線を1枚図に保存し、AP が「PR 曲線の下面積」であることを目で確認する。

★digit 始まりの 01〜03 スクリプトは import できないため、必要な処理はこのファイル内で
  自己完結させ、共通部品だけ detection_helpers から借りる。

実行: uv run python lectures/18_object_detection_intro/mini_project.py
出力: outputs/18_object_detection_intro/
  - mini_detection_overlay.png  予測（と合成時は pseudo-GT）のオーバレイ
  - mini_pr_curve.png           controlled シナリオの PR 曲線（全点補間 AP を陰影）
  - mini_project.json           検出・自作 mAP・torchmetrics・両者の比較
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import detection_helpers as H  # noqa: E402  共通の道具箱（画像生成/後処理/可視化/保存）

torch.set_num_threads(min(8, torch.get_num_threads()))

SCORE_THRESH = 0.30
NMS_IOU = 0.50
IOU_MATCH = 0.50  # mAP のマッチング IoU 閾値（mAP@0.5）

# make_scene_image が描く図形に対応する pseudo-GT（合成画像のときのみ意味を持つ）。
# label は COCO id（person=1, car=3）。座標は draw_person / draw_car の描画範囲から概算。
SYNTH_GT: list[tuple[int, list[float]]] = [
    (1, [75.0, 40.0, 145.0, 308.0]),   # 大きい人物（左）
    (1, [222.0, 70.0, 278.0, 272.0]),  # 小さい人物（中央）
    (3, [340.0, 204.0, 480.0, 282.0]),  # 車（右）
]


# =====================================================================
# 自作 mAP@0.5（IoU マッチング → PR 曲線 → 全点補間 AP → クラス平均）
# ＝ exercises の ex7〜ex9 を「複数画像・複数クラス」へ拡張したもの
# =====================================================================

def iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """xyxy の2集合 (N,4),(M,4) から IoU 行列 (N,M) を作る（自前実装）。"""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]))
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # 交差の左上
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # 交差の右下
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-12)


def ap_all_points(precision: np.ndarray, recall: np.ndarray) -> float:
    """全点補間（PASCAL VOC2010+/COCO の考え方）で PR 曲線の下面積=AP を返す。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):  # precision を右から累積 max（envelope 化）
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]  # recall が変化する位置だけ積分
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision_one_class(
    preds_per_image: list[tuple[torch.Tensor, torch.Tensor]],
    gts_per_image: list[torch.Tensor],
    iou_thresh: float = IOU_MATCH,
) -> dict:
    """1クラスの AP@iou_thresh を、複数画像にまたがって計算する。

    - preds_per_image[i] = (scores(K,), boxes(K,4))  … 画像 i のこのクラスの予測
    - gts_per_image[i]   = boxes(G,4)                 … 画像 i のこのクラスの GT
    返り値: {"ap", "precision"(list), "recall"(list), "n_gt"}
    """
    n_gt = int(sum(g.shape[0] for g in gts_per_image))
    # 全画像の予測を (score, image_idx, pred_local_idx) で集約し、score 降順に並べる。
    entries: list[tuple[float, int, int]] = []
    for img_idx, (scores, _boxes) in enumerate(preds_per_image):
        for k in range(scores.shape[0]):
            entries.append((float(scores[k]), img_idx, k))
    entries.sort(key=lambda e: e[0], reverse=True)

    if n_gt == 0:
        # GT が無いクラスは AP を定義できない（COCO も無視）。
        return {"ap": None, "precision": [], "recall": [], "n_gt": 0}

    matched: dict[int, set[int]] = {i: set() for i in range(len(gts_per_image))}
    tp = np.zeros(len(entries), dtype="float64")
    fp = np.zeros(len(entries), dtype="float64")
    for rank, (_score, img_idx, k) in enumerate(entries):
        gt_boxes = gts_per_image[img_idx]
        pred_box = preds_per_image[img_idx][1][k][None]  # (1,4)
        if gt_boxes.shape[0] == 0:
            fp[rank] = 1.0
            continue
        ious = iou_matrix(pred_box, gt_boxes)[0]  # (G,)
        # 未マッチGTのうち IoU 最大を選ぶ（既に使われた GT は除外）。
        best_iou, best_g = 0.0, -1
        for g in range(gt_boxes.shape[0]):
            if g in matched[img_idx]:
                continue
            if float(ious[g]) > best_iou:
                best_iou, best_g = float(ious[g]), g
        if best_g >= 0 and best_iou >= iou_thresh:
            tp[rank] = 1.0
            matched[img_idx].add(best_g)
        else:
            fp[rank] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = ap_all_points(precision, recall) if len(entries) else 0.0
    return {"ap": ap, "precision": precision.tolist(), "recall": recall.tolist(), "n_gt": n_gt}


def mean_average_precision_50(preds: list[dict], targets: list[dict]) -> dict:
    """自作 mAP@0.5。各画像の {boxes,scores,labels} 予測と {boxes,labels} GT を受け取り、
    クラスごとに AP を出して（GT のあるクラスで）平均する。

    返り値: {"map_50", "per_class": {label: {ap, n_gt, ...}}}
    """
    classes: set[int] = set()
    for t in targets:
        classes.update(int(c) for c in t["labels"].tolist())
    for p in preds:
        classes.update(int(c) for c in p["labels"].tolist())

    per_class: dict[int, dict] = {}
    aps: list[float] = []
    for cls in sorted(classes):
        preds_pi: list[tuple[torch.Tensor, torch.Tensor]] = []
        gts_pi: list[torch.Tensor] = []
        for p in preds:
            m = p["labels"] == cls
            preds_pi.append((p["scores"][m], p["boxes"][m]))
        for t in targets:
            m = t["labels"] == cls
            gts_pi.append(t["boxes"][m])
        res = average_precision_one_class(preds_pi, gts_pi)
        per_class[cls] = res
        if res["ap"] is not None:
            aps.append(res["ap"])
    map_50 = float(np.mean(aps)) if aps else 0.0
    return {"map_50": map_50, "per_class": per_class}


# =====================================================================
# Part A: 検出パイプライン（torchvision ssdlite）＋ pseudo-GT で mAP
# =====================================================================

def run_detector(image, device: torch.device) -> dict:
    """ssdlite320 で1枚検出し、閾値+クラス別NMS後の {boxes,scores,labels,names} を返す。"""
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT  # ① 重み(COCO)
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()  # ② 本体+eval
    preproc = weights.transforms()                      # ③ 前処理（PIL→[0,1] float CHW）
    categories = weights.meta["categories"]             # ④ クラス名（index0=__background__）

    x = preproc(image).to(device)
    with torch.inference_mode():                        # ★勾配・メモリを止める
        raw = model([x])[0]                             # ★入力は「テンソルのリスト」
    boxes, scores, labels = H.apply_threshold_nms(
        raw["boxes"].cpu(), raw["scores"].cpu(), raw["labels"].cpu(),
        score_thresh=SCORE_THRESH, iou_thresh=NMS_IOU,
    )
    names = [categories[int(i)] for i in labels]        # index0=背景 なので生整数を名前と誤解しない
    return {"boxes": boxes, "scores": scores, "labels": labels, "names": names}


def gt_to_tensors(gt: list[tuple[int, list[float]]]) -> dict:
    """pseudo-GT のリストを {boxes,labels} テンソル辞書に変換する。"""
    boxes = torch.tensor([b for _lbl, b in gt], dtype=torch.float32) if gt else torch.zeros((0, 4))
    labels = torch.tensor([lbl for lbl, _b in gt], dtype=torch.int64) if gt else torch.zeros((0,), dtype=torch.int64)
    return {"boxes": boxes, "labels": labels}


# =====================================================================
# Part B: PR 曲線の可視化
# =====================================================================

def save_pr_curve(precision: list[float], recall: list[float], ap: float,
                  path: pathlib.Path, title: str) -> None:
    """PR 曲線（生の点）と全点補間 envelope、AP の陰影を1枚に保存する。"""
    import matplotlib.pyplot as plt  # helper 側で Agg 設定済み

    rec = np.asarray(recall, dtype="float64")
    pre = np.asarray(precision, dtype="float64")
    # envelope（ap_all_points と同じ単調化）を描画用に再構成。
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], pre, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # ★step="pre": 区間 (mrec[i], mrec[i+1]] の高さを mpre[i+1] にする。これが全点補間 AP の
    #   面積式 Σ(Δrecall)*mpre[i+1] と一致する（"post" にすると陰影面積が AP とズレる）。
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.fill_between(mrec, mpre, step="pre", alpha=0.20, color="tab:blue", label=f"AP area = {ap:.3f}")
    ax.step(mrec, mpre, where="pre", color="tab:blue", lw=1.6, label="interp (all-point)")
    ax.plot(rec, pre, "o--", color="tab:red", lw=1.0, ms=5, label="raw PR points")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# controlled シナリオ（検算用の決め打ち GT/予測）
# =====================================================================

def controlled_scenario() -> tuple[list[dict], list[dict]]:
    """2画像・2クラス（person=1, car=3）の決め打ちデータ。

    person は GT3個に対し TP2・FP2・FN1（recall が 1 に届かない）になるよう作り、
    AP が 1.0 未満になる「非自明な」ケースにしてある。car は素直に TP1 で AP=1.0。
    """
    preds = [
        dict(  # 画像0
            boxes=torch.tensor([
                [10., 10., 60., 110.],     # person 0.95: G0 に一致 -> TP
                [122., 42., 205., 98.],    # car    0.80: 車GT に一致 -> TP
                [12., 12., 58., 108.],     # person 0.55: G0 は使用済み -> FP（重複）
                [400., 10., 440., 60.],    # person 0.30: どの GT とも無関係 -> FP
            ]),
            scores=torch.tensor([0.95, 0.80, 0.55, 0.30]),
            labels=torch.tensor([1, 3, 1, 1]),
        ),
        dict(  # 画像1
            boxes=torch.tensor([[31., 22., 88., 138.]]),  # person 0.88: 一致 -> TP
            scores=torch.tensor([0.88]),
            labels=torch.tensor([1]),
        ),
    ]
    targets = [
        dict(  # 画像0: person2個 + car1個（うち person1個は検出されず FN）
            boxes=torch.tensor([
                [10., 10., 60., 110.],     # person G0
                [300., 200., 360., 320.],  # person G1（対応する予測なし -> FN）
                [120., 40., 210., 100.],   # car
            ]),
            labels=torch.tensor([1, 1, 3]),
        ),
        dict(  # 画像1: person1個
            boxes=torch.tensor([[30., 20., 90., 140.]]),
            labels=torch.tensor([1]),
        ),
    ]
    return preds, targets


def torchmetrics_map(preds: list[dict], targets: list[dict]) -> dict:
    """torchmetrics の MeanAveragePrecision で map_50 / map を出す（検算の基準）。"""
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    out = metric.compute()
    return {"map_50": round(float(out["map_50"]), 4), "map_50_95": round(float(out["map"]), 4)}


# =====================================================================
# main
# =====================================================================

def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    image, source = H.load_input_image()
    is_synthetic = source == "synthetic"
    print(f"device = {device}   input source = {source}   size = {image.size}\n")

    result: dict = {
        "source": source, "score_thresh": SCORE_THRESH, "nms_iou": NMS_IOU,
        "iou_match": IOU_MATCH,
    }

    # ---------- Part A: 検出パイプライン ----------
    pred_dict = None
    try:
        det = run_detector(image, device)
        pred_dict = {"boxes": det["boxes"], "scores": det["scores"], "labels": det["labels"]}
        det_str = ", ".join(f"{n}({float(s):.2f})" for n, s in zip(det["names"], det["scores"])) or "(none)"
        print(f"[detector] ssdlite320: {len(det['names'])} det  -> {det_str}")
        result["detector"] = {
            "model": "torchvision/ssdlite320_mobilenet_v3_large",
            "n_det": len(det["names"]),
            "detections": [
                {"name": n, "score": round(float(s), 3),
                 "box_xyxy": [round(float(v), 1) for v in b]}
                for n, s, b in zip(det["names"], det["scores"], det["boxes"])
            ],
        }
        # 可視化: 予測（赤）。合成画像なら pseudo-GT（緑）も並べる。
        vis_pred = H.draw_detections(image, det["boxes"], det["names"], det["scores"], color="red")
        panel = [(image, f"input ({source})")]
        if is_synthetic:
            gt = gt_to_tensors(SYNTH_GT)
            gt_names = ["person", "person", "car"]
            vis_gt = H.draw_detections(image, gt["boxes"], gt_names, torch.ones(len(gt_names)), color="lime")
            panel.append((vis_gt, "pseudo-GT (synthetic shapes)"))
        panel.append((vis_pred, f"ssdlite preds\n{len(det['names'])} det"))
        H.save_panel(panel, out / "mini_detection_overlay.png",
                     "Mini project: detection pipeline (preds vs pseudo-GT)")

        # 合成画像のときだけ、検出器の mAP@0.5 を pseudo-GT で測る（ドメインギャップの定量化）。
        if is_synthetic:
            gt = gt_to_tensors(SYNTH_GT)
            det_map = mean_average_precision_50([pred_dict], [gt])
            tm = torchmetrics_map([pred_dict], [gt])
            print(f"[detector mAP@0.5 vs pseudo-GT] self={det_map['map_50']:.4f}  "
                  f"torchmetrics={tm['map_50']:.4f}  (低いのはドメインギャップ＝正常)")
            result["detector_map_vs_pseudo_gt"] = {
                "self_map_50": round(det_map["map_50"], 4),
                "torchmetrics_map_50": tm["map_50"],
                "note": "COCO学習器は合成図形に弱く mAP がほぼ0。低い値＝故障ではない。",
            }
    except Exception as e:  # noqa: BLE001  重みDL不可などでも capstone 全体は落とさない
        print(f"[detector] スキップ（{type(e).__name__}: {e})。Part B は続行します。")
        result["detector"] = {"skipped": f"{type(e).__name__}: {e}"}

    # ---------- Part B: 自作 mAP を torchmetrics で検算 ----------
    preds, targets = controlled_scenario()
    self_map = mean_average_precision_50(preds, targets)
    tm = torchmetrics_map(preds, targets)

    per_class_print = {
        cls: (round(v["ap"], 4) if v["ap"] is not None else None)
        for cls, v in self_map["per_class"].items()
    }
    print("\n[controlled scenario] 自作 mAP@0.5 の検算:")
    print(f"   per-class AP@0.5 = {per_class_print}  (1=person, 3=car)")
    print(f"   self  mAP@0.5      = {self_map['map_50']:.4f}  (全点補間)")
    print(f"   torchmetrics map_50 = {tm['map_50']:.4f}  (COCO 101点補間)")
    print(f"   torchmetrics map    = {tm['map_50_95']:.4f}  (mAP@[.5:.95])")
    print("   ↑ 小さな差は補間方式の違い（全点 vs 101点）。値はほぼ一致するはず。")

    result["self_map_demo"] = {
        "per_class_ap_50": per_class_print,
        "self_map_50_all_point": round(self_map["map_50"], 4),
        "torchmetrics_map_50_coco101": tm["map_50"],
        "torchmetrics_map_50_95": tm["map_50_95"],
        "abs_diff_map_50": round(abs(self_map["map_50"] - tm["map_50"]), 4),
        "note": "自作=全点補間 / torchmetrics=COCO 101点補間。差は補間方式由来。",
    }

    # PR 曲線（person クラス）を保存。
    person = self_map["per_class"].get(1)
    if person and person["ap"] is not None and person["recall"]:
        save_pr_curve(
            person["precision"], person["recall"], person["ap"],
            out / "mini_pr_curve.png",
            f"PR curve (person): AP@0.5(all-point) = {person['ap']:.3f}",
        )
        print(f"saved: {out / 'mini_pr_curve.png'}")

    (out / "mini_project.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {out / 'mini_project.json'}")
    print("\nまとめ: 検出→閾値→NMS→可視化→評価(mAP) を1本で通し、自作mAPを"
          "torchmetricsで検算した。numpyでの完全実装と COCOeval 一致確認は第19回。")


if __name__ == "__main__":
    main()
