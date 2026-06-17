"""第19回 共通の道具箱 — 合成検出データの生成・COCO形式への変換・IoU・貪欲マッチング。

ここには「読み物」ではなく「各スクリプトが import して使い回す道具」を置く。
教材の核心（IoU の定義式・confidence 降順マッチング・PR/AP の組み立て）は各スクリプトの
本体で改めて手で書き下す。ここではそれらを支える次の4つだけを提供する:

  - output_dir()        : 出力先 lectures/19_detection_map_from_scratch/outputs/ を作って返す
  - box_iou_numpy()     : xyxy 同士の IoU 行列を numpy だけで計算する（検出評価の心臓部）
  - match_image()       : 1画像・1閾値で予測を GT に貪欲対応付けし TP/FP を決める正準エンジン
  - make_detection_dataset() / to_coco_gt() / to_coco_dt() : 合成データと COCO 形式変換

なぜ合成データか:
  本講座はネットへ出てよいのは「モデル重みのDL」だけ。検出評価の学習には実画像は不要で、
  GT ボックスと予測ボックス（スコア付き）さえあれば mAP は完全に計算できる。そこで GT を
  ランダム生成し、その一部を「少しずらした予測」「検出漏れ(FN)」「無関係な誤検出(FP)」へ
  変換することで、PR 曲線に起伏のある“評価に向いた”データを再現性つきで作る。
  実画像で試したい場合は data/ に COCO 形式の GT/予測を置けば同じ関数にそのまま流せる。
"""

from __future__ import annotations

import pathlib

import numpy as np

# COCO のカテゴリ定義（id は 1 始まりが COCO 流儀。id=0 は背景予約のため使わない）。
# 形状名は 01 の可視化でオブジェクトを描き分けるために使うだけで、評価計算には関与しない。
CATEGORIES: list[dict] = [
    {"id": 1, "name": "circle"},
    {"id": 2, "name": "square"},
    {"id": 3, "name": "triangle"},
]
IMAGE_H, IMAGE_W = 480, 640  # 合成シーンの大きさ（ピクセル）


def output_dir() -> pathlib.Path:
    """このモジュールの出力先を作って返す（headless 環境でも結果を後から確認できるように）。"""
    out = pathlib.Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def box_iou_numpy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU 行列を numpy だけで計算する。boxes_* は (?,4) の xyxy（左上x,左上y,右下x,右下y）。

    IoU = 交差面積 / 和集合面積。和集合 = 面積A + 面積B − 交差。
    交差矩形の左上 = それぞれの左上の "大きい方"、右下 = それぞれの右下の "小さい方"。
    幅・高さが負（重なりなし）の場合は 0 にクリップする。これが検出評価の全ての土台になる。
    """
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)  # (M,)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)  # (N,)

    # ブロードキャストで全 M×N ペアの交差矩形を一度に求める
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # 交差の左上 (M,N,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # 交差の右下 (M,N,2)
    wh = (rb - lt).clip(0)  # 負（重なりなし）は 0 に
    inter = wh[..., 0] * wh[..., 1]  # 交差面積 (M,N)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-12, None)  # 0除算回避


def match_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """1画像・1つの IoU 閾値で「予測→GT」の貪欲マッチングを行う正準エンジン（COCO 互換）。

    手順（これが mAP の中核アルゴリズム）:
      1. 予測を confidence(score) の降順に並べる（安定ソート＝同点は元の順を保つ）。
      2. 上から1件ずつ、まだ未マッチの GT の中で IoU が最大かつ閾値以上のものへ割り当てる。
         割り当てられたら TP、見つからなければ FP。1つの GT は1度しか使わない（二重カウント防止）。
      3. 最後まで割り当てられなかった GT が FN（戻り値の n_gt と TP 数の差で分かる）。

    戻り値:
      tp_sorted : スコア降順に並べた各予測の TP=1.0 / FP=0.0 フラグ (N,)
      score_sorted : それに対応するスコア (N,)（後段で画像横断に再ソートして PR を積む）
      n_gt : この画像のGT数（recall の分母に使う）
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    n_gt = len(gt_boxes)

    # スコア降順（安定ソート）。kind="stable" にしないと同点時の TP/FP がライブラリとズレる
    order = np.argsort(-pred_scores, kind="stable")
    score_sorted = pred_scores[order]
    tp_sorted = np.zeros(len(order), dtype=np.float64)

    if n_gt == 0:
        return tp_sorted, score_sorted, 0  # GT が無ければ全予測が FP

    ious = box_iou_numpy(pred_boxes[order], gt_boxes)  # (N, n_gt)
    gt_used = np.zeros(n_gt, dtype=bool)  # GT ごとに「もう使ったか」を持つ

    for di in range(len(order)):
        best_iou, best_gt = iou_thr, -1
        for gi in range(n_gt):
            if gt_used[gi]:
                continue  # 既にマッチ済みの GT は飛ばす（1GT=1TP）
            if ious[di, gi] < best_iou:
                continue  # 閾値未満、または現在の最良より低いなら無視
            best_iou, best_gt = ious[di, gi], gi  # より良い未使用 GT を更新
        if best_gt >= 0:
            gt_used[best_gt] = True  # その GT を消費
            tp_sorted[di] = 1.0  # この予測は TP
        # best_gt == -1 のままなら FP（tp_sorted は 0 のまま）
    return tp_sorted, score_sorted, n_gt


def make_detection_dataset(
    n_images: int = 12,
    seed: int = 0,
) -> list[dict]:
    """合成の検出データセットを作る。各画像につき GT と予測（スコア付き）を返す。

    1画像分の dict:
      gt_boxes (M,4 xyxy), gt_labels (M,)            … 正解
      pred_boxes (N,4 xyxy), pred_scores (N,), pred_labels (N,)  … 予測

    予測の作り方（PR 曲線に起伏を作るための設計）:
      - GT の多くは「中心と大きさを少しずらした予測」に変換 → IoU は 0.5〜0.9 に分布（TP 候補）。
      - 一部の GT は意図的にずらしを大きくし IoU≈0.3〜0.55 にする → 閾値 0.5/0.75 で結果が変わる。
      - 一部の GT はあえて予測を作らない → 検出漏れ(FN)。
      - 各画像に無関係な箱を数個ばらまく → 誤検出(FP)。スコアは低めだが TP と値域が重なる。
    スコアは連続値（一様乱数）で全て相異なるため、同点による曖昧さが起きず pycocotools と一致する。
    """
    rng = np.random.default_rng(seed)
    cat_ids = [c["id"] for c in CATEGORIES]
    dataset: list[dict] = []

    for _ in range(n_images):
        n_obj = int(rng.integers(4, 9))  # この画像のGT数
        gt_boxes, gt_labels = [], []
        pred_boxes, pred_scores, pred_labels = [], [], []

        for _ in range(n_obj):
            # --- GT を1つ生成（画面内に収まるようサイズと位置を決める） ---
            bw = float(rng.uniform(50, 130))
            bh = float(rng.uniform(50, 130))
            x1 = float(rng.uniform(0, IMAGE_W - bw))
            y1 = float(rng.uniform(0, IMAGE_H - bh))
            box = [x1, y1, x1 + bw, y1 + bh]
            label = int(rng.choice(cat_ids))
            gt_boxes.append(box)
            gt_labels.append(label)

            # --- その GT を予測へ変換するか（=検出できたか）を確率で決める ---
            roll = rng.random()
            if roll < 0.15:
                continue  # 15%: 検出漏れ（FN）。予測を作らない

            # ずらし量。10% は「大きくずれた予測」にして閾値依存を作る
            if roll < 0.25:
                shift_frac, scale_lo, scale_hi, score_lo = 0.22, 0.70, 1.30, 0.30
            else:
                shift_frac, scale_lo, scale_hi, score_lo = 0.08, 0.88, 1.12, 0.45

            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            dx = rng.uniform(-shift_frac, shift_frac) * bw
            dy = rng.uniform(-shift_frac, shift_frac) * bh
            nw = bw * rng.uniform(scale_lo, scale_hi)
            nh = bh * rng.uniform(scale_lo, scale_hi)
            px1 = np.clip(cx + dx - nw / 2, 0, IMAGE_W - 1)
            py1 = np.clip(cy + dy - nh / 2, 0, IMAGE_H - 1)
            px2 = np.clip(cx + dx + nw / 2, 1, IMAGE_W)
            py2 = np.clip(cy + dy + nh / 2, 1, IMAGE_H)
            pred_boxes.append([px1, py1, px2, py2])
            pred_scores.append(float(rng.uniform(score_lo, 0.99)))  # 当たり予測は高めのスコア
            pred_labels.append(label)

        # --- 無関係な誤検出(FP)を数個ばらまく。スコアは低めだが TP と重なる値域 ---
        for _ in range(int(rng.integers(2, 5))):
            bw = float(rng.uniform(40, 110))
            bh = float(rng.uniform(40, 110))
            x1 = float(rng.uniform(0, IMAGE_W - bw))
            y1 = float(rng.uniform(0, IMAGE_H - bh))
            pred_boxes.append([x1, y1, x1 + bw, y1 + bh])
            pred_scores.append(float(rng.uniform(0.05, 0.6)))
            pred_labels.append(int(rng.choice(cat_ids)))

        dataset.append(
            {
                "gt_boxes": np.array(gt_boxes, dtype=np.float64).reshape(-1, 4),
                "gt_labels": np.array(gt_labels, dtype=np.int64).reshape(-1),
                "pred_boxes": np.array(pred_boxes, dtype=np.float64).reshape(-1, 4),
                "pred_scores": np.array(pred_scores, dtype=np.float64).reshape(-1),
                "pred_labels": np.array(pred_labels, dtype=np.int64).reshape(-1),
            }
        )
    return dataset


def to_coco_gt(dataset: list[dict]) -> dict:
    """データセットを pycocotools が読む COCO GT 辞書に変換する。

    重要: COCO の bbox は [x, y, w, h]（xywh）で、本講座が内部で使う xyxy とは別物。
    annotation id は 1 始まりの一意な整数、area は w*h、iscrowd=0（群衆扱いしない）にする。
    """
    images = [{"id": i, "width": IMAGE_W, "height": IMAGE_H} for i in range(len(dataset))]
    annotations = []
    ann_id = 1
    for img_id, d in enumerate(dataset):
        for box, label in zip(d["gt_boxes"], d["gt_labels"]):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(w), float(h)],  # xywh!
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def to_coco_dt(dataset: list[dict]) -> list[dict]:
    """予測を pycocotools の loadRes が読む検出リスト（xywh + score）に変換する。"""
    results = []
    for img_id, d in enumerate(dataset):
        for box, score, label in zip(d["pred_boxes"], d["pred_scores"], d["pred_labels"]):
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # xywh
                    "score": float(score),
                }
            )
    return results
