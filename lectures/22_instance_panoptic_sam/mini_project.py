"""第22回 章末ミニプロジェクト: SAM の対話セグメンテーションを mask AP と PQ で採点する。

この回の学び（インスタンス／パノプティック／SAM／評価）を1本に統合した総合課題。
ストーリーはこうだ——

  「クラスを知らない SAM に GT の重心を“点プロンプト”として教え、各物体を切り出す
   （= 対話的インスタンスセグメンテーション）。さらに背景にも1点打って、わざと
   余計な検出（FP）を作る。こうして得た予測マスク群を、第22回で学んだ2つの指標
   ——mask AP@0.5（自作・全点補間）と PQ=SQ×RQ（自作 ＝ torchmetrics）——で採点し、
   『同じ予測でも AP と PQ で FP の効き方が違う』ことを数値で体感する。」

統合している要素:
  - SAM の正準フロー: 画像エンコードを1回だけ行い（get_image_embeddings）、
    各プロンプトはその埋め込みを使い回す（interactive segmentation の肝）。
  - post_process_masks で原寸化、3マスクから iou_scores 最大を選択。
  - mask IoU（交差/和集合）、score 降順の貪欲マッチ、PR→全点補間 AP。
  - パノプティック合成（stuff+things を全画素に1つ）と PQ=SQ×RQ の自作＝torchmetrics 照合。

実行: uv run python lectures/22_instance_panoptic_sam/mini_project.py
出力: outputs/22_instance_panoptic_sam/mini_project.png / mini_project_report.json

注: SAM 重みの DL（初回のみ・数百MB）以外はネット不要。ネット不通やロード失敗時は
    GT を少し揺らした疑似予測へ自動フォールバックし、評価パイプラインは必ず完走する
    （合成データなので“検出が弱い/モデルが無い”でも exit 0）。
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import seg_helpers as H  # noqa: E402

SAM_MODEL_ID = "facebook/sam-vit-base"  # CPU 既定。さらに軽くするなら Zigeng/SlimSAM-uniform-77
# stuff（背景）カテゴリ ID（本講座内の便宜値。things は GT の category_id を使う）。
SKY_CAT, GROUND_CAT = 10, 11


# =====================================================================
# 1) SAM で「点プロンプト → ベストマスク」を作る（画像埋め込みは1回だけ）
# =====================================================================

def load_sam(device: torch.device):
    """SAM をロードして (model, processor) を返す。失敗時は None（→フォールバック）。"""
    try:
        from transformers import SamModel, SamProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        model = SamModel.from_pretrained(SAM_MODEL_ID).to(device).eval()
        return model, processor
    except Exception as e:  # noqa: BLE001  ネット不通・環境差はフォールバックで吸収
        print(f"  [warn] SAM をロードできませんでした（{type(e).__name__}）。疑似予測へフォールバックします。")
        return None


def sam_predict_points(model, processor, image, points, device) -> list[tuple[np.ndarray, float]]:
    """各点プロンプトについて (best_mask(bool,H,W), quality) を返す。

    SAM の interactive な使い方の肝: 重い画像エンコーダ（get_image_embeddings）は
    1回だけ走らせ、軽いプロンプトデコーダ側（点ごとの forward）でその埋め込みを
    使い回す。CPU でも複数プロンプトを高速に処理できる。
    """
    base = processor(image, return_tensors="pt").to(device)
    with torch.inference_mode():
        image_embeddings = model.get_image_embeddings(base["pixel_values"])  # (1,256,64,64)

    results: list[tuple[np.ndarray, float]] = []
    for x, y in points:
        pin = processor(
            image, input_points=[[[float(x), float(y)]]], input_labels=[[1]], return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            out = model(
                image_embeddings=image_embeddings,          # ★埋め込みを使い回す（再エンコードしない）
                input_points=pin["input_points"],
                input_labels=pin["input_labels"],
                multimask_output=True,                      # 曖昧性ぶん3マスク返す
            )
        masks = processor.post_process_masks(               # 256x256→原寸（必須）
            out.pred_masks.cpu(), pin["original_sizes"].cpu(), pin["reshaped_input_sizes"].cpu()
        )[0][0]  # (3, H, W) bool
        scores = out.iou_scores[0, 0].cpu()                 # (3,) 予測品質（真の IoU ではない）
        best = int(torch.argmax(scores))
        results.append((masks[best].numpy().astype(bool), float(scores[best])))
    return results


def fallback_predict_points(instances, points, hw) -> list[tuple[np.ndarray, float]]:
    """SAM が使えない時の疑似予測: GT を少し揺らした“ほぼ正解”マスクを作る。

    こうしておけば SAM が無くても mask AP / PQ の採点パイプラインはそのまま動く。
    GT 物体の点なら対応 GT を少し膨張、背景の点なら小さな円（→ FP）を返す。
    """
    height, width = hw
    out: list[tuple[np.ndarray, float]] = []
    for x, y in points:
        owner = _instance_at(instances, x, y)
        if owner is not None:
            jittered = _dilate(owner["mask"], iters=1)  # 形を少しだけ崩した近似予測
            out.append((jittered, 0.95))
        else:
            yy, xx = np.ogrid[:height, :width]
            disk = (yy - int(y)) ** 2 + (xx - int(x)) ** 2 <= 30 ** 2
            out.append((disk, 0.30))
    return out


def _instance_at(instances, x, y):
    """点 (x,y) を含む GT インスタンスを返す（無ければ None=背景）。"""
    xi, yi = int(round(x)), int(round(y))
    for inst in instances:
        m = inst["mask"]
        if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]:
            return inst
    return None


def _dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """簡易膨張（4近傍）。cv2 を使わず numpy だけで GT を少し崩す。"""
    m = mask.copy()
    for _ in range(iters):
        s = m.copy()
        s[1:, :] |= m[:-1, :]
        s[:-1, :] |= m[1:, :]
        s[:, 1:] |= m[:, :-1]
        s[:, :-1] |= m[:, 1:]
        m = s
    return m


# =====================================================================
# 2) 予測を組み立てる（things は GT 重心 + 背景1点で FP を仕込む）
# =====================================================================

def build_predictions(image, instances, device) -> tuple[list[dict], bool]:
    """SAM（or フォールバック）で予測インスタンス群を作る。

    返り値の各 dict: {"mask", "category_id", "score", "prompt_xy", "matched_gt"}
      - GT インスタンスごとに重心へ点を打ち、その GT の category_id を引き継ぐ
        （SAM はクラス非依存なので、プロンプト元のクラスを“ラベル付け”として使う）。
      - 背景（空）にも1点打ち、わざと余計な予測（FP, category_id=最初の things クラス）を作る。
    """
    height, width = np.asarray(image).shape[:2]
    points: list[tuple[float, float]] = []
    owners: list[dict | None] = []
    for inst in instances:
        ys, xs = np.where(inst["mask"])
        points.append((float(xs.mean()), float(ys.mean())))
        owners.append(inst)
    # 背景（左上の空あたり）に1点 → FP を意図的に作る
    points.append((40.0, 40.0))
    owners.append(None)

    sam = load_sam(device)
    if sam is not None:
        used_sam = True
        masks_scores = sam_predict_points(sam[0], sam[1], image, points, device)
    else:
        used_sam = False
        masks_scores = fallback_predict_points(instances, points, (height, width))

    things_cat = instances[0]["category_id"] if instances else 1
    preds: list[dict] = []
    for (mask, score), owner, (px, py) in zip(masks_scores, owners, points):
        if mask.sum() == 0:
            continue  # 空マスクは捨てる
        is_fp = owner is None
        # 背景点は「わざと入れた低確信の誤検出」として score を低く固定する。
        # こうすると AP（スコア降順・末尾に来るので無害）と PQ（RQ を必ず下げる）の
        # 効き方の違いが、SAM の品質スコアのブレに左右されず再現的に観察できる。
        preds.append({
            "mask": mask,
            "category_id": things_cat if is_fp else owner["category_id"],
            "score": 0.10 if is_fp else float(score),
            "prompt_xy": [round(px, 1), round(py, 1)],
            "matched_gt": None if is_fp else owner["label"],
        })
    return preds, used_sam


# =====================================================================
# 3) 評価: mask AP@0.5（自作・全点補間）と PQ=SQ×RQ（自作 ＝ torchmetrics）
# =====================================================================

def mask_ap50(gt: list[dict], preds: list[dict], iou_thr: float = 0.5) -> dict:
    """クラスを跨いだ mask AP@0.5（mAP）を numpy で自作する（全点補間）。"""
    cats = sorted({g["category_id"] for g in gt})
    per_class: dict[str, float] = {}
    for cat in cats:
        gmasks = [g["mask"] for g in gt if g["category_id"] == cat]
        cpreds = sorted([p for p in preds if p["category_id"] == cat],
                        key=lambda p: -p["score"])
        matched = [False] * len(gmasks)
        tp = np.zeros(len(cpreds))
        fp = np.zeros(len(cpreds))
        for rank, p in enumerate(cpreds):
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gmasks):
                if matched[j]:
                    continue
                iou = H.mask_iou(p["mask"], g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                tp[rank] = 1.0
                matched[best_j] = True
            else:
                fp[rank] = 1.0
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / max(len(gmasks), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        per_class[str(cat)] = round(_ap_all_point(recall, precision), 4)
    mean_ap = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return {"mAP@0.5": round(mean_ap, 4), "per_class_ap": per_class}


def _ap_all_point(recall: np.ndarray, precision: np.ndarray) -> float:
    """PR 曲線の全点補間 AP（exercises ex6 と同じ式）。"""
    mrec = np.concatenate(([0.0], np.asarray(recall, float), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precision, float), [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compose_panoptic(stuff: list[tuple[np.ndarray, int]],
                     things: list[tuple[np.ndarray, int, int]],
                     hw: tuple[int, int]) -> np.ndarray:
    """(H, W, 2) の panoptic ラベルを作る。最後の次元は (category_id, instance_id)。

    先に stuff を敷き、上から things を載せる（後勝ちで重なりを一意化）。
    各画素はちょうど1つの (category, instance) を持つ＝パノプティックの定義。
    """
    height, width = hw
    cat = np.zeros((height, width), dtype=np.int64)
    inst = np.zeros((height, width), dtype=np.int64)
    for m, c in stuff:
        cat[m] = c
        inst[m] = 0  # stuff は instance_id=0
    for m, c, iid in things:
        cat[m] = c
        inst[m] = iid
    return np.stack([cat, inst], axis=-1)


def evaluate_pq(gt_pan: np.ndarray, pred_pan: np.ndarray,
                things: set[int], stuffs: set[int]) -> dict:
    """PQ=SQ×RQ を自作 numpy で計算し、torchmetrics と一致するか確認する。"""
    pq_manual = _manual_pq(pred_pan, gt_pan, things | stuffs)

    tm = None
    try:
        from torchmetrics.detection import PanopticQuality

        metric = PanopticQuality(things=things, stuffs=stuffs, return_sq_and_rq=True)
        out = metric(torch.tensor(pred_pan)[None], torch.tensor(gt_pan)[None])
        tm = {"PQ": round(float(out[0]), 4), "SQ": round(float(out[1]), 4), "RQ": round(float(out[2]), 4)}
        agree = bool(np.isclose(pq_manual["PQ"], tm["PQ"], atol=1e-3))
    except Exception as e:  # noqa: BLE001
        agree = None
        print(f"  [warn] torchmetrics の照合をスキップ（{type(e).__name__}）。自作 PQ のみ報告します。")
    return {"manual": pq_manual, "torchmetrics": tm, "agree": agree}


def _segments_of(panoptic: np.ndarray, category: int) -> list[np.ndarray]:
    cat, inst = panoptic[..., 0], panoptic[..., 1]
    return [(cat == category) & (inst == iid) for iid in np.unique(inst[cat == category])]


def _manual_pq(pred: np.ndarray, gt: np.ndarray, valid_cats: set[int]) -> dict:
    """カテゴリ別に SQ/RQ/PQ を出して最後に平均する（panopticapi 流の numpy 実装）。"""
    per_pq, per_sq, per_rq = [], [], []
    for c in sorted(valid_cats):
        gt_segs = _segments_of(gt, c)
        pr_segs = _segments_of(pred, c)
        if not gt_segs and not pr_segs:
            continue
        matched: set[int] = set()
        iou_sum, tp = 0.0, 0
        for pr in pr_segs:
            best_iou, best_k = 0.0, -1
            for k, g in enumerate(gt_segs):
                if k in matched:
                    continue
                iou = H.mask_iou(pr, g)
                if iou > best_iou:
                    best_iou, best_k = iou, k
            if best_iou > 0.5 and best_k >= 0:  # >0.5 なのでマッチ相手は一意
                matched.add(best_k)
                iou_sum += best_iou
                tp += 1
        fp = len(pr_segs) - tp
        fn = len(gt_segs) - tp
        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        per_sq.append(sq)
        per_rq.append(rq)
        per_pq.append(sq * rq)
    f = lambda xs: round(float(np.mean(xs)), 4) if xs else 0.0  # noqa: E731
    return {"PQ": f(per_pq), "SQ": f(per_sq), "RQ": f(per_rq)}


# =====================================================================
# 4) 可視化
# =====================================================================

def _label_map(masks_with_id: list[tuple[np.ndarray, int]], hw) -> np.ndarray:
    label = np.zeros(hw, dtype=int)
    for m, idx in masks_with_id:
        label[m] = idx
    return label


def make_figure(image, instances, preds, out_dir, ap, pq):
    """input / GT instances / SAM predictions / overlay の4枚を1図に保存する。"""
    rgb = np.asarray(image.convert("RGB"))
    hw = rgb.shape[:2]

    gt_label = _label_map([(inst["mask"], i) for i, inst in enumerate(instances, 1)], hw)
    pred_label = _label_map([(p["mask"], i) for i, p in enumerate(preds, 1)], hw)

    # overlay: 予測マスクを画像に半透明合成
    overlay = rgb.copy().astype(np.float32)
    colors = H.PALETTE[1:1 + len(preds)] if len(preds) else H.PALETTE[1:2]
    for p, col in zip(preds, colors):
        overlay[p["mask"]] = 0.45 * overlay[p["mask"]] + 0.55 * col
    overlay = overlay.clip(0, 255).astype(np.uint8)

    panels = [
        ("input", rgb),
        (f"GT instances (n={len(instances)})", H.colorize_label_map(gt_label)),
        (f"SAM preds (n={len(preds)})", H.colorize_label_map(pred_label)),
        ("pred overlay", overlay),
    ]
    path = out_dir / "mini_project.png"
    H.save_side_by_side(
        panels, path,
        suptitle=f"capstone: SAM instances  mAP@0.5={ap['mAP@0.5']:.3f}  PQ={pq['manual']['PQ']:.3f}",
    )
    return path


# =====================================================================
# main
# =====================================================================

def main() -> None:
    device = H.pick_device()
    out_dir = H.ensure_output_dir()

    # --- 入力: 合成シーン（GT インスタンス付き）。実写があればそれを使う ---
    image, instances = H.load_user_or_synthetic()
    if not instances:
        # 実写は GT が無いので、この総合課題は合成シーンに固定して採点を成立させる。
        print("  [info] data/ の実写には GT が無いため、採点用に合成シーンを使います。")
        image, instances = H.build_synthetic_scene()
    rgb = np.asarray(image.convert("RGB"))
    hw = rgb.shape[:2]

    print(f"[mini] device={device}  GT instances={len(instances)}")

    # --- 予測（SAM 点プロンプト ＋ 背景FP）---
    preds, used_sam = build_predictions(image, instances, device)
    print(f"  予測 {len(preds)} 件（{'SAM' if used_sam else 'fallback'}）。背景点で FP を1つ仕込み済み。")

    # --- (A) mask AP@0.5 ---
    gt_for_ap = [{"mask": inst["mask"], "category_id": inst["category_id"]} for inst in instances]
    ap = mask_ap50(gt_for_ap, preds)
    print(f"  (A) mask AP@0.5 = {ap['mAP@0.5']:.4f}  per_class={ap['per_class_ap']}")

    # --- (B) PQ=SQ×RQ（panoptic 合成 → 自作 ＝ torchmetrics）---
    sky = np.zeros(hw, bool)
    sky[: hw[0] // 2] = True
    ground = np.zeros(hw, bool)
    ground[hw[0] // 2:] = True
    things_ids = {inst["category_id"] for inst in instances}
    stuffs_ids = {SKY_CAT, GROUND_CAT}

    gt_things = [(inst["mask"], inst["category_id"], i) for i, inst in enumerate(instances, 1)]
    gt_pan = compose_panoptic([(sky, SKY_CAT), (ground, GROUND_CAT)], gt_things, hw)
    pred_things = [(p["mask"], p["category_id"], i) for i, p in enumerate(preds, 1)]
    pred_pan = compose_panoptic([(sky, SKY_CAT), (ground, GROUND_CAT)], pred_things, hw)

    pq = evaluate_pq(gt_pan, pred_pan, things_ids, stuffs_ids)
    print(f"  (B) PQ={pq['manual']['PQ']:.4f}  SQ={pq['manual']['SQ']:.4f}  RQ={pq['manual']['RQ']:.4f}"
          + (f"  (torchmetrics 一致={pq['agree']})" if pq["torchmetrics"] else ""))
    print("  → 低スコアの背景FPは AP（末尾に来るので mAP は下がらない）に効かないが、")
    print("     PQ では未マッチ予測=FP として RQ を必ず下げる。同じ予測でも指標で効き方が違う。")

    # --- 図・JSON 保存 ---
    fig_path = make_figure(image, instances, preds, out_dir, ap, pq)
    report = {
        "device": str(device),
        "predictor": "SAM" if used_sam else "fallback(jittered GT)",
        "sam_model": SAM_MODEL_ID,
        "num_gt_instances": len(instances),
        "predictions": [
            {"category_id": p["category_id"], "score": round(p["score"], 4),
             "prompt_xy": p["prompt_xy"], "matched_gt": p["matched_gt"],
             "mask_area_px": int(p["mask"].sum()),
             "iou_vs_prompt_gt": (
                 round(H.mask_iou(p["mask"], next(g["mask"] for g in instances if g["label"] == p["matched_gt"])), 4)
                 if p["matched_gt"] is not None else None)}
            for p in preds
        ],
        "mask_ap": ap,
        "panoptic_quality": pq,
    }
    json_path = out_dir / "mini_project_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved: {fig_path.name}, {json_path.name}")


if __name__ == "__main__":
    main()
