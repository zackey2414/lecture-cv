"""01: CLIPSeg でテキスト条件付きセグメンテーション（logits → sigmoid → 閾値）。

参照セグメ（referring segmentation）の最小形を体験する。検出のように「箱」を出すのでは
なく、与えた文（プロンプト）が指す領域の『画素ごとの確率マップ』を直接出すのが CLIPSeg。

手順（このスクリプトの核）:
  1. 合成シーン（赤い円・青い四角・緑の三角）と、各オブジェクトの GT マスクを用意。
  2. CLIPSeg に画像＋複数プロンプトを渡し、(P, 352, 352) のロジットを得る。
  3. 原寸へ補間 → torch.sigmoid → 確率マップ (0〜1)。閾値 0.5 で2値マスク化。
  4. GT があるので IoU/Dice を測り、ヒートマップ＋オーバレイを保存。
  5. 「シーンに無い概念（a yellow star）」も投げ、確率が低く保たれる＝閾値で消える様子を見る。

実行: uv run python lectures/23_text_prompt_segmentation/01_clipseg.py
出力: lectures/23_text_prompt_segmentation/outputs/01_*.png / 01_clipseg_metrics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import seg_helpers as H  # noqa: E402


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # --- 1) 入力（実画像があれば優先、無ければ合成シーン）---
    image, gt, scene = H.load_user_or_scene()
    prompts = [o["prompt"] for o in scene]

    # --- 2) CLIPSeg ロード → 確率マップ (P, H, W) ---
    model, processor = H.load_clipseg(device)
    probs = H.clipseg_probs(model, processor, image, prompts, device)  # 0〜1, 原寸

    # --- 3) 閾値 0.5 で2値マスク化（参照セグメの既定。最適値は 03 でスイープ）---
    threshold = 0.5
    masks = [probs[i] >= threshold for i in range(len(prompts))]

    # --- 4) GT があれば IoU/Dice を測る ---
    metrics = []
    for i, obj in enumerate(scene):
        rec = {
            "prompt": obj["prompt"],
            "prob_min": round(float(probs[i].min()), 4),
            "prob_max": round(float(probs[i].max()), 4),
            "mask_pixels": int(masks[i].sum()),
        }
        if gt is not None and obj["name"] in gt:
            rec["IoU@0.5"] = round(H.mask_iou(masks[i], gt[obj["name"]]), 4)
            rec["Dice@0.5"] = round(H.mask_dice(masks[i], gt[obj["name"]]), 4)
        metrics.append(rec)

    # 可視化パネル（input / prob heatmap / mask overlay）を保存。
    H.save_panel(image, probs, prompts, masks, out / "01_clipseg_panel.png", gt=gt)

    # --- 5) 「シーンに無い概念」を投げて、確率が低いまま＝閾値で消えることを確認 ---
    absent_prompt = "a yellow star"
    absent_prob = H.clipseg_probs(model, processor, image, [absent_prompt], device)[0]
    absent = {
        "prompt": absent_prompt,
        "prob_max": round(float(absent_prob.max()), 4),
        "pixels_over_0.5": int((absent_prob >= 0.5).sum()),
    }

    # --- コンソール出力 ---
    print(f"device = {device}  prompts = {prompts}")
    for rec in metrics:
        line = f"  '{rec['prompt']}'  prob[max]={rec['prob_max']:.2f}  pixels={rec['mask_pixels']}"
        if "IoU@0.5" in rec:
            line += f"  IoU@0.5={rec['IoU@0.5']:.3f}  Dice@0.5={rec['Dice@0.5']:.3f}"
        print(line)
    print(f"  [不在の概念] '{absent['prompt']}'  prob[max]={absent['prob_max']:.2f}  "
          f"pixels>0.5={absent['pixels_over_0.5']}  ← 低ければ閾値で正しく『無い』と判定")

    # --- json 保存 ---
    payload = {"threshold": threshold, "prompts": metrics, "absent_concept": absent}
    (out / "01_clipseg_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / '01_clipseg_panel.png'}, {out / '01_clipseg_metrics.json'}")


if __name__ == "__main__":
    main()
