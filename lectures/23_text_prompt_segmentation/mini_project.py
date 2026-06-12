"""ミニプロジェクト: テキストプロンプト・セグメンテーションの手法比較ベンチ。

本章で学んだ部品（CLIPSeg の logits→sigmoid→閾値、IoU/Dice、しきい値スイープ、
Grounding DINO→SAM の2段構成）を1つに統合し、『同じシーンを3つの設定で解いて
表・図・JSON で勝敗を出す』総合課題の完成形。

比較する3設定:
  (1) CLIPSeg @0.5        … 既定しきい値で2値化（素朴な使い方）
  (2) CLIPSeg @best       … 各プロンプトでしきい値をスイープして IoU 最大点を採用
  (3) Grounded-SAM        … Grounding DINO の box を SAM でマスク化（2段・高精度）

成果物（outputs/23_text_prompt_segmentation/ に保存）:
  - mini_project_compare.png   … オブジェクト×手法 の IoU 棒グラフ
  - mini_project_panel.png     … GT / CLIPSeg@best / Grounded-SAM の重ね合わせ
  - mini_project_report.json   … 全数値・オブジェクト別の勝者・手法別平均 IoU

合成画像では Grounding DINO の検出が弱いことがある（domain gap）。検出ゼロなら
GT の外接 box を SAM に渡すフォールバックに切り替え、必ず exit 0 で完走する。
digit 始まりの 02_grounded_sam.py は import できないため、検出/SAM の処理は本ファイル
内に自己完結させ、モデルロード・評価指標・シーン生成は seg_helpers から借用する。

実行: uv run python lectures/23_text_prompt_segmentation/mini_project.py
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


# ---------------------------------------------------------------------------
# Grounded-SAM の部品（02 と同等。digit 始まりは import 不可なので自己完結させる）
# ---------------------------------------------------------------------------
def mask_to_box(mask: np.ndarray) -> list[int] | None:
    """bool マスクから外接 box [x0,y0,x1,y1] を作る（空なら None）。"""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def detect_gdino(model, processor, image, prompt_text: str, device, threshold: float = 0.25):
    """Grounding DINO で box を検出して [{box, score, label}, ...] を返す。

    テキストは小文字＋ピリオド区切りが作法。target_sizes は (height, width) 順。
    """
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    w, h = image.size
    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=threshold, text_threshold=0.25, target_sizes=[(h, w)]
    )[0]
    dets = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        dets.append({"box": [int(v) for v in box.tolist()], "score": float(score), "label": str(label)})
    return dets


def sam_mask_for_box(model, processor, image, box: list[int], device) -> np.ndarray:
    """SAM に1つの box を渡し、候補3枚から iou_scores 最大のマスクを bool で返す。"""
    inputs = processor(image, input_boxes=[[box]], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]  # (n_boxes=1, n_masks=3, H, W)
    best = int(torch.argmax(outputs.iou_scores[0, 0]))
    return masks[0, best].numpy().astype(bool)


# ---------------------------------------------------------------------------
# 評価ロジック
# ---------------------------------------------------------------------------
def eval_clipseg(probs: np.ndarray, scene, gt, thresholds: np.ndarray) -> list[dict]:
    """各プロンプトについて CLIPSeg@0.5 と CLIPSeg@best を評価して返す。"""
    rows = []
    for i, obj in enumerate(scene):
        name = obj["name"]
        g = gt[name]
        m05 = probs[i] >= 0.5
        bt, _ = H.best_threshold(probs[i], g, thresholds)
        mbest = probs[i] >= bt
        rows.append({
            "object": name,
            "clipseg@0.5": {"IoU": round(H.mask_iou(m05, g), 4), "Dice": round(H.mask_dice(m05, g), 4)},
            "clipseg@best": {
                "threshold": round(float(bt), 3),
                "IoU": round(H.mask_iou(mbest, g), 4),
                "Dice": round(H.mask_dice(mbest, g), 4),
            },
            "_mask_best": mbest,  # 図用（JSON には出さない）
        })
    return rows


def best_sam_mask_for_gt(sam_masks: list[np.ndarray], g: np.ndarray) -> tuple[np.ndarray | None, float]:
    """GT マスク g に対し IoU 最大の SAM マスクを返す（無ければ None, 0.0）。"""
    best_mask, best_iou = None, 0.0
    for m in sam_masks:
        v = H.mask_iou(m, g)
        if v >= best_iou:
            best_iou, best_mask = v, m
    return best_mask, best_iou


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------
def save_compare_bar(rows: list[dict], path: pathlib.Path) -> None:
    """オブジェクト×手法 の IoU を棒グラフにする（タイトルは ASCII）。"""
    names = [r["object"] for r in rows]
    methods = ["clipseg@0.5", "clipseg@best", "grounded_sam"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, method in enumerate(methods):
        vals = [r[method]["IoU"] for r in rows]
        ax.bar(x + (k - 1) * width, vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("IoU vs GT")
    ax.set_ylim(0, 1.05)
    ax.set_title("Referring segmentation: per-object IoU by method")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_panel(image, rows: list[dict], gt, grounded_masks: dict, path: pathlib.Path) -> None:
    """各オブジェクトについて [GT / CLIPSeg@best / Grounded-SAM] を並べる。"""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.0 * n))
    axes = np.array(axes).reshape(n, 3)
    for r, row in enumerate(rows):
        name = row["object"]
        axes[r, 0].imshow(gt[name], cmap="gray")
        axes[r, 0].set_title(f"GT: {name}", fontsize=9)
        ov_c = H.overlay_mask(image, row["_mask_best"], color=(255, 60, 60))
        axes[r, 1].imshow(ov_c)
        axes[r, 1].set_title(f"CLIPSeg@best IoU={row['clipseg@best']['IoU']:.3f}", fontsize=9)
        gm = grounded_masks.get(name)
        if gm is not None:
            ov_g = H.overlay_mask(image, gm, color=(60, 120, 255))
            axes[r, 2].imshow(ov_g)
        else:
            axes[r, 2].imshow(image)
        axes[r, 2].set_title(f"Grounded-SAM IoU={row['grounded_sam']['IoU']:.3f}", fontsize=9)
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # --- 1) 合成シーン＋GT（評価のため GT を厳密に持つ build_scene を使う）---
    image, gt, scene = H.build_scene()
    prompts = [o["prompt"] for o in scene]

    # --- 2) CLIPSeg: 確率マップ → @0.5 と @best を評価 ---
    clipseg, clipseg_proc = H.load_clipseg(device)
    probs = H.clipseg_probs(clipseg, clipseg_proc, image, prompts, device)
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    rows = eval_clipseg(probs, scene, gt, thresholds)

    # --- 3) Grounded-SAM: GDINO で検出 → SAM でマスク化（検出ゼロは GT box フォールバック）---
    gdino, gdino_proc = H.load_gdino(device)
    sam, sam_proc = H.load_sam(device)
    prompt_text = ". ".join(prompts) + "."
    dets = detect_gdino(gdino, gdino_proc, image, prompt_text, device, threshold=0.25)
    used_fallback = False
    if not dets:
        used_fallback = True
        for obj in scene:
            b = mask_to_box(gt[obj["name"]])
            if b is not None:
                dets.append({"box": b, "score": 1.0, "label": obj["name"]})
    sam_masks = [sam_mask_for_box(sam, sam_proc, image, d["box"], device) for d in dets]

    # --- 4) オブジェクト別に Grounded-SAM の最良マスクを対応付け、3手法を比較 ---
    grounded_masks: dict = {}
    method_iou_sum = {"clipseg@0.5": 0.0, "clipseg@best": 0.0, "grounded_sam": 0.0}
    winner_count = {"clipseg@0.5": 0, "clipseg@best": 0, "grounded_sam": 0}
    for row in rows:
        name = row["object"]
        gm, _ = best_sam_mask_for_gt(sam_masks, gt[name])
        grounded_masks[name] = gm
        g_iou = round(H.mask_iou(gm, gt[name]), 4) if gm is not None else 0.0
        g_dice = round(H.mask_dice(gm, gt[name]), 4) if gm is not None else 0.0
        row["grounded_sam"] = {"IoU": g_iou, "Dice": g_dice}
        # 勝者判定（IoU 最大の手法）
        cand = {
            "clipseg@0.5": row["clipseg@0.5"]["IoU"],
            "clipseg@best": row["clipseg@best"]["IoU"],
            "grounded_sam": g_iou,
        }
        winner = max(cand, key=cand.get)
        row["winner"] = winner
        winner_count[winner] += 1
        for m in method_iou_sum:
            method_iou_sum[m] += cand[m]

    n = len(rows)
    mean_iou = {m: round(s / n, 4) for m, s in method_iou_sum.items()}

    # --- 5) 図を保存 ---
    save_compare_bar(rows, out / "mini_project_compare.png")
    save_panel(image, rows, gt, grounded_masks, out / "mini_project_panel.png")

    # --- 6) JSON レポート（図用の生マスクは落としてから書く）---
    json_rows = []
    for row in rows:
        jr = {k: v for k, v in row.items() if not k.startswith("_")}
        json_rows.append(jr)
    report = {
        "prompt_text": prompt_text,
        "used_gt_fallback": used_fallback,
        "num_detections": len(dets),
        "per_object": json_rows,
        "summary": {"mean_IoU": mean_iou, "winner_count": winner_count},
    }
    (out / "mini_project_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 7) コンソール要約 ---
    src = "GT box フォールバック" if used_fallback else "Grounding DINO"
    print(f"device = {device}  検出元 = {src}  検出数 = {len(dets)}")
    print(f"{'object':16s} {'CLIPSeg@0.5':>12s} {'CLIPSeg@best':>14s} {'Grounded-SAM':>14s}  winner")
    for row in rows:
        print(f"{row['object']:16s} "
              f"{row['clipseg@0.5']['IoU']:>12.3f} "
              f"{row['clipseg@best']['IoU']:>14.3f} "
              f"{row['grounded_sam']['IoU']:>14.3f}  {row['winner']}")
    print(f"\n手法別 平均 IoU: {mean_iou}")
    print(f"オブジェクト別 勝者数: {winner_count}")
    print(f"\nsaved: {out / 'mini_project_compare.png'}, "
          f"{out / 'mini_project_panel.png'}, {out / 'mini_project_report.json'}")


if __name__ == "__main__":
    main()
