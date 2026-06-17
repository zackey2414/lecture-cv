"""第20回 オープン語彙物体検出の道具箱（合成シーン・device・モデルロード・評価・描画）。

このモジュールは「読み物」ではなく 01〜03 の各スクリプトが import して使う共通部品。
責務を1つずつ小さく切ってあるので、まずここを一読してから各スクリプトへ進むと、
何を import しているかが腑に落ちる。

ここに入れている共通処理:
  - device 判定（本講座は CPU 前提だが CUDA があれば拾う）
  - 出力先 / データ置き場のパス管理
  - 合成シーン生成（GT ボックス付き＝評価に使える）/ 実画像の自動読み込み
  - OWL-ViT / OWLv2 / Grounding DINO のロードと「テキストで検出」する薄いラッパ
  - IoU・貪欲マッチング・precision/recall/F1（第19回の自作 mAP と地続き）
  - matplotlib による検出結果の保存（headless・Agg。imshow は呼ばない）

注意（transformers v5）:
  OWL-ViT/OWLv2 も Grounding DINO も後処理は
  `processor.post_process_grounded_object_detection(...)` に統一されている。
  古い記事の `post_process_object_detection` は OWL 系では使わない（v5 の変更点）。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

# matplotlib は必ず Agg（画面なし環境でも図を保存できる）。pyplot より前に backend を固定する。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# --- パス（プロジェクトルート基準で固定）-------------------------------------
MODULE_ID = "20_open_vocabulary_detection"
_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # lectures/20_.../ovd_helpers.py → リポジトリルート
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = _ROOT / "data" / MODULE_ID

# --- モデル ID（CPU で現実的な小型を既定にする）------------------------------
OWLVIT_ID = "google/owlvit-base-patch32"  # 最軽量・最速の OWL-ViT
OWLV2_ID = "google/owlv2-base-patch16-ensemble"  # 自己学習で高精度化した後継（やや重い）
GDINO_ID = "IDEA-Research/grounding-dino-tiny"  # Grounding DINO の最小版（timm 依存）


def pick_device() -> torch.device:
    """CUDA があれば cuda、無ければ cpu。本講座は CPU 前提だが移植性のため判定する。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir() -> pathlib.Path:
    """出力先 lectures/20_open_vocabulary_detection/outputs/ を作って返す（無ければ作成、あってもエラーにしない）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


# =====================================================================
# 合成シーン（GT 付き）と実画像読み込み
# =====================================================================
# オープン語彙検出は「色」「形」という概念を強く捉えるので、合成画像でも
# "a red circle" のような任意ラベルがちゃんと当たる。GT ボックスを手で決めて
# 持っておけば、第19回流の precision/recall/F1 をそのまま測れる（03 で使用）。

# シーンに置く 4 物体。box は [x1, y1, x2, y2]（絶対画素・xyxy）、label は検出に使う英語フレーズ。
_SCENE_H, _SCENE_W = 400, 560
_SCENE_OBJECTS: list[dict] = [
    {"label": "a red circle", "kind": "circle", "box": (52, 72, 168, 188), "bgr": (0, 0, 255)},
    {"label": "a blue square", "kind": "rect", "box": (320, 70, 470, 210), "bgr": (235, 90, 30)},
    {
        "label": "a green triangle",
        "kind": "triangle",
        "box": (110, 250, 250, 360),
        "bgr": (40, 170, 60),
    },
    {
        "label": "a yellow circle",
        "kind": "circle",
        "box": (375, 265, 485, 375),
        "bgr": (40, 215, 230),
    },
]


def build_scene() -> tuple[Image.Image, np.ndarray, list[str]]:
    """合成シーンを描き、(PIL画像(RGB), GTボックス(N,4 xyxy), GTラベル(list[str])) を返す。

    cv2 は BGR で描くので、最後に必ず RGB へ変換してから PIL.Image にする（色反転バグ防止）。
    各図形は _SCENE_OBJECTS の box にぴったり収まるように描き、その box をそのまま GT とする。
    """
    img = np.full((_SCENE_H, _SCENE_W, 3), 235, dtype=np.uint8)  # 明るい灰色の背景
    gt_boxes: list[tuple[int, int, int, int]] = []
    gt_labels: list[str] = []
    for obj in _SCENE_OBJECTS:
        x1, y1, x2, y2 = obj["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = obj["bgr"]
        if obj["kind"] == "circle":
            r = (x2 - x1) // 2
            cv2.circle(img, (cx, cy), r, color, -1)
        elif obj["kind"] == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif obj["kind"] == "triangle":
            pts = np.array([[x1, y2], [x2, y2], [cx, y1]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
        gt_boxes.append((x1, y1, x2, y2))
        gt_labels.append(obj["label"])
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil, np.array(gt_boxes, dtype=np.float32), gt_labels


def scene_candidate_labels() -> list[str]:
    """検出に投げる候補ラベル（=オープン語彙のクエリ）。

    GT の 4 ラベル＋『シーンに無いラベル』2つを混ぜる。無いラベルがほとんど検出されない
    （＝幻覚しにくい）ことを観察できるようにするのが狙い。
    """
    present = [o["label"] for o in _SCENE_OBJECTS]
    absent = ["a dog", "a traffic light"]
    return present + absent


def load_user_or_synthetic() -> tuple[Image.Image, np.ndarray | None, list[str] | None]:
    """data/20_.../ に画像があれば最初の1枚を使い、無ければ合成シーンを返す。

    実画像のときは GT が無いので (画像, None, None)。合成のときは GT 付き。
    実写を置けば実用的な検出になる旨は README で案内している。
    """
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                return Image.open(p).convert("RGB"), None, None
    pil, gt_boxes, gt_labels = build_scene()
    return pil, gt_boxes, gt_labels


# =====================================================================
# モデルのロードと「テキストで検出」する薄いラッパ
# =====================================================================
# どのモデルも AutoProcessor + AutoModelForZeroShotObjectDetection でロードでき、
# 後処理は post_process_grounded_object_detection に統一されている（v5）。
# 違いはクエリの渡し方（OWL=候補ラベルのリスト / GDINO=ピリオド区切りキャプション）と
# 閾値の引数（GDINO だけ text_threshold を持つ）だけ。


def load_zero_shot_detector(model_id: str):
    """AutoProcessor と AutoModelForZeroShotObjectDetection をロードして eval() で返す。"""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # ロード時の冗長なログを抑制
    device = pick_device()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


def detect_owl(processor, model, image: Image.Image, labels: list[str], threshold: float = 0.1):
    """OWL-ViT / OWLv2 で候補ラベル検出。(boxes(N,4 xyxy), scores(N,), text_labels(list)) を返す。

    - labels は『候補ラベルのリスト』。processor には [labels]（バッチ1件ぶんの二重リスト）で渡す。
    - target_sizes は必ず (height, width) 順。PIL の image.size は (W,H) なので [::-1] で渡す。
    - post_process_grounded_object_detection は xyxy 絶対座標の boxes を返す（生出力の
      cxcywh 正規化座標を絶対 xyxy に直してくれる）。可視化前に必ずこの後処理を通すこと。
    """
    device = pick_device()
    nested = [labels]  # OWL は「画像ごとの候補ラベル集合」を期待するので一段ネストする
    inputs = processor(text=nested, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():  # 推論は勾配を切る（CPU で効く）
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (H, W)
    result = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=target_sizes, text_labels=nested
    )[0]
    boxes = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    text_labels = list(result["text_labels"])  # 候補ラベルそのものの文字列
    return boxes, scores, text_labels


def detect_gdino(
    processor,
    model,
    image: Image.Image,
    caption: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
):
    """Grounding DINO でキャプション検出。(boxes(N,4 xyxy), scores(N,), text_labels(list)) を返す。

    - caption は『小文字＋各物体をピリオドで区切る』形式が必須（例: "a cat. a remote control."）。
    - 閾値は2種類: box_threshold（=引数 threshold。検出の確信度）と text_threshold（どの語に
      対応づけるかの確信度）。デフォルトのままだと過検出/未検出になりやすく、調整が要る。
    - input_ids を後処理に渡すことで、検出ボックスをキャプション中の語に紐づける。
    """
    device = pick_device()
    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (H, W)
    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]
    boxes = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    text_labels = [str(t) for t in result["text_labels"]]
    return boxes, scores, text_labels


def labels_to_caption(labels: list[str]) -> str:
    """候補ラベルのリストを Grounding DINO 用キャプションへ。小文字化＋ピリオド区切り。

    例: ["a Red circle", "a Blue square"] -> "a red circle. a blue square."
    """
    parts = [lab.strip().rstrip(".").lower() for lab in labels if lab.strip()]
    return " ".join(f"{p}." for p in parts)


# =====================================================================
# 評価（IoU・貪欲マッチング・P/R/F1）— 第19回の自作 mAP と同じ筋
# =====================================================================


def iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """1つの box [x1,y1,x2,y2] と複数 boxes (M,4) の IoU を返す（M,）。numpy で素朴に。"""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_s = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_b + area_s - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: list[str],
    gt_boxes: np.ndarray,
    gt_labels: list[str],
    iou_thr: float = 0.5,
) -> tuple[int, int, int]:
    """クラス込みの貪欲マッチングで (TP, FP, FN) を数える。

    手順（第19回と同じ）:
      1. 予測を score 降順に並べる。
      2. 各予測について『同じラベルかつ未マッチで IoU>=閾値』の GT のうち IoU 最大に対応づける。
      3. 対応すれば TP、余れば FP。最後まで対応されなかった GT は FN。
    1つの GT に複数予測が当たっても TP は最初の1つだけ（残りは FP）。
    """
    order = np.argsort(-pred_scores)  # confidence 降順
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for i in order:
        # 同じラベルの未マッチ GT に限定して IoU を測る
        cand = [
            g
            for g in range(len(gt_boxes))
            if g not in matched_gt and gt_labels[g] == pred_labels[i]
        ]
        if not cand:
            fp += 1
            continue
        ious = iou_xyxy(pred_boxes[i], gt_boxes[cand])
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            matched_gt.add(cand[j])
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """TP/FP/FN から precision・recall・F1 を計算（ゼロ割は 0.0）。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# =====================================================================
# 描画（matplotlib・Agg）
# =====================================================================

_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]


def draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    labels: list[str],
    scores: np.ndarray,
    title: str,
    path: pathlib.Path,
    gt_boxes: np.ndarray | None = None,
) -> None:
    """画像に検出ボックス（任意で GT ボックスを点線で）を重ねて保存する。

    GT を渡すと灰色の点線で重ね描きし、予測（実線・色つき）との重なりを目視できる。
    日本語フォントの豆腐を避けるため、ラベルは英語のまま描く。
    """
    fig, ax = plt.subplots(figsize=(image.width / 90, image.height / 90))
    ax.imshow(np.asarray(image))
    if gt_boxes is not None:
        for gx1, gy1, gx2, gy2 in gt_boxes:
            ax.add_patch(
                Rectangle(
                    (gx1, gy1),
                    gx2 - gx1,
                    gy2 - gy1,
                    fill=False,
                    edgecolor="0.4",
                    linestyle="--",
                    linewidth=1.5,
                )
            )
    for i, (b, lab, sc) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = b
        color = _PALETTE[i % len(_PALETTE)]
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.2)
        )
        ax.text(
            x1,
            max(y1 - 4, 2),
            f"{lab} {sc:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, edgecolor="none", pad=1.0),
        )
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def smoke_test() -> None:
    """道具箱の自己診断: 合成シーンを描いて GT を重ね、IoU/PRF が動くことを確認する。"""
    out = ensure_output_dir()
    pil, gt_boxes, gt_labels = build_scene()
    # GT だけを描いた図（ラベルは GT 名・スコアは 1.0 のダミー）
    draw_detections(
        pil,
        gt_boxes,
        gt_labels,
        np.ones(len(gt_boxes)),
        "synthetic scene with ground-truth boxes",
        out / "00_scene_ground_truth.png",
    )
    # IoU と PRF の最小確認（同じ箱どうしは IoU=1, ずらすと下がる）
    iou_same = float(iou_xyxy(gt_boxes[0], gt_boxes[:1])[0])
    shifted = gt_boxes[:1] + np.array([20, 20, 20, 20], dtype=np.float32)
    iou_shift = float(iou_xyxy(gt_boxes[0], shifted)[0])
    tp, fp, fn = greedy_match(gt_boxes, np.ones(len(gt_boxes)), gt_labels, gt_boxes, gt_labels)
    p, r, f1 = prf(tp, fp, fn)
    print(f"device = {pick_device()}")
    print(f"scene: {len(gt_boxes)} objects, labels = {gt_labels}")
    print(f"IoU(box, box) = {iou_same:.3f} / IoU(box, +20px) = {iou_shift:.3f}")
    print(f"perfect match -> TP={tp} FP={fp} FN={fn} / P={p:.2f} R={r:.2f} F1={f1:.2f}")
    print(f"saved: {out / '00_scene_ground_truth.png'}")


if __name__ == "__main__":
    # 単体実行すると道具箱のスモークテストになる（モデル DL 不要・純計算＋描画）。
    smoke_test()
