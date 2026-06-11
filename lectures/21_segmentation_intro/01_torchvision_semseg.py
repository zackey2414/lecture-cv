"""01: torchvision でセマンティックセグメンテーションを最短で動かす。

セマンティックセグメンテーションは「画像の“すべての画素”にクラスを割り当てる」タスク。
分類が画像1枚に1ラベルを返すのに対し、ここでは出力が (クラス数, H, W) のロジットになり、
画素ごとに argmax して “クラスマップ” を得る。物体検出（bbox）よりも輪郭まで細かく取れる。

torchvision の正準フロー（分類とほぼ同じだが、出力が dict なのが要注意）:
  1. weights = LRASPP_..._Weights.DEFAULT     # 学習済み重み＋前処理＋クラス名がセット
  2. model  = lraspp_...(weights=weights).eval()
  3. x = weights.transforms()(pil_image)      # リサイズ＋ImageNet正規化を自動で行う
  4. out = model(x.unsqueeze(0))              # ★ out は OrderedDict。out['out'] にロジット
  5. pred = out['out'].argmax(1)              # (N, H, W) のクラスマップ

落とし穴: 返り値の dict をそのまま argmax しない。必ず out['out'] を取り出す。
また weights.transforms() は内部で 520px へリサイズするので、可視化時は
nn.functional.interpolate で元の解像度に戻す（次の節）。

実行: uv run python lectures/21_segmentation_intro/01_torchvision_semseg.py
出力: outputs/21_segmentation_intro/01_*.png / 01_torchvision_semseg.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import seg_helpers as H  # noqa: E402  同じフォルダの道具箱

# CPU でも現実的なモデルだけを既定にする。
#   lraspp    : MobileNetV3 バックボーン。最軽量（約12MB）・CPU向き。
#   deeplabv3 : ResNet50 バックボーン。高精度だが重い（約160MB DL）。
# fcn も同系列だが、ここでは lraspp と deeplabv3 の2本で「軽量 vs 高精度」を対比する。
MODEL_NAMES = ["lraspp", "deeplabv3"]


def segment_one(model, weights, image, device: torch.device) -> np.ndarray:
    """1枚の PIL 画像をセグメンテーションし、元解像度のクラスマップ (H,W) を返す。

    手順のキモは2つ:
      - out['out'] でロジットを取り出す（dict をそのまま argmax しない）。
      - logits を元画像サイズへ interpolate してから argmax する。
        （ラベルを最近傍リサイズするより、ロジットを bilinear で戻す方が境界が素直。）
    """
    w0, h0 = image.size  # PIL は (W, H) 順
    x = weights.transforms()(image).unsqueeze(0).to(device)  # (1,3,520,520) 正規化済み
    with torch.inference_mode():  # 推論は勾配を切る（CPU では特に効く）
        out = model(x)  # ★ OrderedDict。out['out'] が (1, num_classes, 520, 520)
    logits = out["out"]
    # ロジットを元解像度へ戻してから argmax（“解像度を戻す”の正準手順）。
    logits_full = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)
    pred = logits_full.argmax(dim=1)[0]  # (H, W) クラスID
    return pred.cpu().numpy().astype(np.int64)


def class_histogram(pred: np.ndarray, categories: list[str], top: int = 6) -> list[dict]:
    """予測クラスマップの画素割合の多い順に上位 top クラスを返す（読み解き用）。"""
    ids, counts = np.unique(pred, return_counts=True)
    total = pred.size
    rows = sorted(
        ({"class_id": int(i), "name": categories[int(i)], "pixel_ratio": round(c / total, 4)}
         for i, c in zip(ids, counts)),
        key=lambda r: r["pixel_ratio"],
        reverse=True,
    )
    return rows[:top]


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    image, src = H.load_user_or_synthetic_image()
    rgb = np.asarray(image)
    voc_palette = H.voc_colormap()  # VOC 21クラスの可視化に十分

    panel_imgs = [rgb]
    panel_titles = [f"input ({src})"]
    summary: dict = {"input": src, "device": str(device), "models": {}}

    for name in MODEL_NAMES:
        model, weights = H.load_torchvision_seg(name, device)
        categories = weights.meta["categories"]  # ['__background__', 'aeroplane', ...]
        pred = segment_one(model, weights, image, device)

        color = H.colorize(pred, voc_palette)  # (H,W,3) クラス色
        over = H.overlay(rgb, color, alpha=0.55)  # 元画像に重ねる
        panel_imgs.append(over)
        panel_titles.append(f"{name}  ({weights.meta.get('num_params', '?')} params)")

        hist = class_histogram(pred, categories)
        summary["models"][name] = {
            "num_classes": len(categories),
            "top_classes": hist,
        }
        print(f"[{name}] 検出された主なクラス（画素割合）:")
        for r in hist:
            print(f"    {r['name']:14s} {r['pixel_ratio']:.3f}")

    H.save_panel(panel_imgs, panel_titles, out / "01_torchvision_semseg.png", ncol=len(panel_imgs))
    (out / "01_torchvision_semseg.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nポイント: 出力は dict。out['out'] を argmax する（dict をそのまま argmax しない）。")
    print("合成画像では実在クラス（person 等）が薄いことがある。data/ に実写を置くと実用的。")
    print(f"saved: {out / '01_torchvision_semseg.png'}, {out / '01_torchvision_semseg.json'}")


if __name__ == "__main__":
    main()
