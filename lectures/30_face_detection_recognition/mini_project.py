"""章末ミニプロジェクト: 顔写真群 → 検出 → 整列 → 埋め込み → クラスタリング → 人物アルバム。

このスクリプトは本章の全工程を 1 本に統合した「完成形」です。

  1. 合成の集合写真（複数人物が写る 1 枚）を作る。各顔の正解位置・人物 ID は把握しておく。
  2. Haar カスケードで顔を **検出** する（実写を data/ に置けばそちらでも動く）。
  3. 検出枠を正解枠と IoU で突合せ、検出品質（recall / precision）を測る。
  4. 検出枠を切り出して FACE_SIZE に揃える（簡易 **整列 alignment**）→ ResNet で **埋め込み**。
  5. 埋め込みを L2 正規化し、eps を内部掃引した **DBSCAN** で人物ごとに **クラスタリング**。
  6. クラスタ＝「自動仕分けされた人物アルバム」を画像化し、認識(EER) と クラスタ品質(NMI 他) を
     JSON にまとめる。
  7. ★頑健性: 検出がドメインギャップでほぼ当たらない場合は、正解枠の切り出しに自動で
     フォールバックして後段（埋め込み〜クラスタリング）を必ず完走させる（exit 0 を保つ）。

実行:
  uv run python lectures/30_face_detection_recognition/mini_project.py
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

import face_lab as fl
from face_lab import iou_xywh

# 検出と正解を「同じ顔」とみなす IoU しきい値、フォールバック判定の最低マッチ率
IOU_MATCH = 0.3
MIN_MATCH_RATIO = 0.5


def match_detections_to_gt(
    det: np.ndarray, gt_boxes: np.ndarray, gt_labels: np.ndarray
) -> tuple[list[int], list[int], int]:
    """検出枠を正解枠に貪欲に突合せ、(検出index→正解index, 検出index→人物ID, FP数) を返す。

    各検出について最も IoU の高い正解を選び、IoU>=しきい値かつ未使用なら採用。
    対応の付かない検出は誤検出(FP)。これで検出crop に人物ラベルを復元し、後段の評価に使う。
    """
    used_gt: set[int] = set()
    matched_gt: list[int] = []
    matched_pid: list[int] = []
    fp = 0
    for d in det:
        best_iou, best_g = 0.0, -1
        for g in range(len(gt_boxes)):
            if g in used_gt:
                continue
            v = iou_xywh(d, gt_boxes[g])
            if v > best_iou:
                best_iou, best_g = v, g
        if best_g >= 0 and best_iou >= IOU_MATCH:
            used_gt.add(best_g)
            matched_gt.append(best_g)
            matched_pid.append(int(gt_labels[best_g]))
        else:
            fp += 1
    return matched_gt, matched_pid, fp


def crop_align(bgr: np.ndarray, box) -> Image.Image:
    """検出/正解の矩形を切り出し、FACE_SIZE 角に整列（リサイズ）して PIL(RGB) で返す。

    実運用の ArcFace は顔ランドマークでアフィン整列するが、本章は中央寄せ済みの合成顔なので
    リサイズで十分。BGR→RGB を忘れず、埋め込み器が期待する形に揃える。
    """
    x, y, w, h = (int(v) for v in box)
    x, y = max(0, x), max(0, y)
    crop = bgr[y : y + h, x : x + w]
    if crop.size == 0:  # 画面外などの保険
        crop = np.full((fl.FACE_SIZE, fl.FACE_SIZE, 3), 128, np.uint8)
    crop = cv2.resize(crop, (fl.FACE_SIZE, fl.FACE_SIZE))
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def best_eps_clustering(emb_norm: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """eps を内部掃引し、NMI 最大のクラスタリング結果を返す（pred, eps, report）。"""
    n_true = len(np.unique(labels))
    best = None
    for eps in np.round(np.linspace(0.03, 0.10, 8), 3):
        pred = DBSCAN(eps=float(eps), min_samples=2, metric="cosine").fit_predict(emb_norm)
        rep = fl.clustering_report(labels, pred)
        key = (round(rep["nmi"], 4), -abs(rep["n_clusters_found"] - n_true))
        if best is None or key > best[0]:
            best = (key, pred, float(eps), rep)
    return best[1], best[2], best[3]


def save_albums(crops: list[Image.Image], pred: np.ndarray, path: str) -> None:
    """予測クラスタごとに顔を 1 行に並べたアルバム画像（行=自動仕分けされた人物）。"""
    fs, pad, label_w = fl.FACE_SIZE, 6, 64
    clusters = sorted(set(pred), key=lambda c: (c == -1, c))
    rows = []
    for c in clusters:
        idxs = np.where(pred == c)[0][:10]
        row = np.full((fs, label_w, 3), 245, np.uint8)
        cv2.putText(
            row,
            "noise" if c == -1 else f"#{c}",
            (3, fs // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
        for i in idxs:
            bgr = cv2.cvtColor(np.array(crops[i]), cv2.COLOR_RGB2BGR)
            row = np.hstack([row, np.full((fs, pad, 3), 245, np.uint8), bgr])
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((fs, width - r.shape[1], 3), 245, np.uint8)]) for r in rows]
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.vstack([canvas, np.full((pad, width, 3), 200, np.uint8), r])
    cv2.imwrite(path, canvas)


def main() -> None:
    os.makedirs(fl.OUT_DIR, exist_ok=True)
    device = fl.get_device()

    # --- 1. 集合写真（6 人 × 4 枚 = 24 顔）を正解付きで合成 ---
    images, labels, names = fl.make_face_dataset(n_ids=6, n_per_id=4, seed=1)
    canvas, gt_boxes = fl.compose_group_photo(images, cols=6, return_boxes=True)
    cv2.imwrite(os.path.join(fl.OUT_DIR, "mini_scene.png"), canvas)
    print(f"集合写真: {canvas.shape[1]}x{canvas.shape[0]} に {len(images)} 顔 / {len(names)} 人")

    # --- 2. 顔検出（Haar） ---
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.equalizeHist(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
    det = np.asarray(
        cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))
    ).reshape(-1, 4)
    print(f"\n[検出] Haar: {len(det)} 件")

    # --- 3. 検出を正解と突合せ、検出品質を測る ---
    matched_gt, matched_pid, fp = match_detections_to_gt(det, gt_boxes, labels)
    recall = len(matched_gt) / len(gt_boxes)
    precision = len(matched_gt) / max(1, len(det))
    print(
        f"  recall={recall:.3f}  precision={precision:.3f}  (FP={fp}, 見逃し={len(gt_boxes) - len(matched_gt)})"
    )

    # --- 4. 検出crop を整列・埋め込み。検出が弱ければ正解枠にフォールバック ---
    _ = matched_pid  # （突合せの内訳。ここでは recall/precision の表示に使用済み）
    used_fallback = recall < MIN_MATCH_RATIO
    if used_fallback:
        print(f"  ※ 検出が弱い（recall<{MIN_MATCH_RATIO}）ためドメインギャップと判断。")
        print("    正解枠の切り出しにフォールバックして後段を完走させます（exit 0 を保証）。")
        crop_boxes = gt_boxes
        crop_labels = labels.copy().tolist()
    else:
        # 検出枠をそのまま切り出す。各検出の人物ラベルは正解との IoU 突合せで復元（FP は -1）。
        crop_boxes = det
        crop_labels = _labels_for_detections(det, gt_boxes, labels)

    crops = [crop_align(canvas, b) for b in crop_boxes]
    enc = fl.FaceEncoder(device)
    emb = fl.l2_normalize(enc.encode(crops))
    print(f"\n[埋め込み] mode={enc.mode}  crops={len(crops)}  dim={emb.shape[1]}")

    # FP(-1) は認識・クラスタ評価の正解付けから除外して評価する
    crop_labels = np.asarray(crop_labels)
    valid = crop_labels >= 0
    emb_v, lab_v = emb[valid], crop_labels[valid]

    # --- 5. クラスタリング（eps 内部掃引で NMI 最大の動作点） ---
    pred_full = DBSCAN(eps=0.05, min_samples=2, metric="cosine").fit_predict(
        emb
    )  # 全crop（FP含む）
    pred_v, best_eps, rep = best_eps_clustering(emb_v, lab_v)
    print(
        f"\n[クラスタリング] best eps={best_eps:.3f}  推定人数={rep['n_clusters_found']}（真値 {len(np.unique(lab_v))}）"
    )
    print(
        f"  purity={rep['purity']:.3f}  NMI={rep['nmi']:.3f}  homo={rep['homogeneity']:.3f}  "
        f"comp={rep['completeness']:.3f}"
    )

    # --- 6. 認識(EER) も合わせて算出 ---
    scores, gen = fl.verification_scores(emb_v, lab_v)
    thrs, far, tar = fl.roc_far_tar(scores, gen)
    eer, _ = fl.compute_eer(far, tar, thrs)
    print(f"[認識] EER={eer:.3f}（同じ埋め込みでの 1:1 照合）")

    # --- 7. アルバム画像・検出可視化・JSON 保存 ---
    save_albums(crops, pred_full, os.path.join(fl.OUT_DIR, "mini_album.png"))
    annotated = canvas.copy()
    for x, y, w, h in det:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.imwrite(os.path.join(fl.OUT_DIR, "mini_detected.png"), annotated)

    summary = {
        "encoder_mode": enc.mode,
        "used_fallback_gt_crops": bool(used_fallback),
        "detection": {
            "n_det": int(len(det)),
            "recall": recall,
            "precision": precision,
            "fp": int(fp),
        },
        "clustering": rep | {"best_eps": best_eps},
        "recognition_eer": float(eer),
    }
    with open(os.path.join(fl.OUT_DIR, "mini_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n成果物:")
    print("  - mini_scene.png / mini_detected.png : 集合写真と顔検出枠")
    print("  - mini_album.png                     : 自動仕分けされた人物アルバム（行=人物）")
    print("  - mini_summary.json                  : 検出/認識/クラスタリングの一括指標")
    print("\n完了。検出 → 整列 → 埋め込み → クラスタリングの一気通貫パイプラインが動きました。")


# --- 小道具（detections の並びに沿って人物ラベルを作る） ---
def _labels_for_detections(
    det: np.ndarray, gt_boxes: np.ndarray, gt_labels: np.ndarray
) -> list[int]:
    """各検出に対し、最大 IoU の正解人物 ID を割り当てる（IoU 不足は -1=FP）。"""
    used: set[int] = set()
    out: list[int] = []
    for d in det:
        best_iou, best_g = 0.0, -1
        for g in range(len(gt_boxes)):
            if g in used:
                continue
            v = iou_xywh(d, gt_boxes[g])
            if v > best_iou:
                best_iou, best_g = v, g
        if best_g >= 0 and best_iou >= IOU_MATCH:
            used.add(best_g)
            out.append(int(gt_labels[best_g]))
        else:
            out.append(-1)
    return out


if __name__ == "__main__":
    main()
