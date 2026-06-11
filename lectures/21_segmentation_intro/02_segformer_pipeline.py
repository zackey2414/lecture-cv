"""02: HuggingFace の SegFormer でセマンティックセグメンテーション（pipeline と手動）。

torchvision（01）は Pascal VOC の 21 クラスだった。ここでは HF Transformers の
SegFormer-b0（ADE20K の 150 クラスで学習）を使う。クラス体系が違うので、
同じ画像でも「空・木・道・建物…」のような別の語彙でラベル付けされる。

2通りの呼び方を体験する:
  (A) pipeline('image-segmentation') … 前処理〜後処理を全部任せる最短路。
       返り値は [{'score', 'label', 'mask'(PIL L 画像)}, ...] のリスト。
       セマンティックでは score は None（インスタンス/パノプティックだとスコアが付く）。
  (B) 手動 … AutoImageProcessor + AutoModelForSemanticSegmentation で読み、
       image_processor.post_process_semantic_segmentation(target_sizes=[(H,W)]) で
       原寸のクラスマップにする。target_sizes が (高さ, 幅) 順なのが最重要の落とし穴
       （PIL の image.size は (幅, 高さ) なので image.size[::-1] で渡す）。

(A) と (B) は同じモデル・同じ後処理なので、得られるクラスマップはほぼ一致する。
それを画素一致率で確認する。

実行: uv run python lectures/21_segmentation_intro/02_segformer_pipeline.py
出力: outputs/21_segmentation_intro/02_*.png / 02_segformer_pipeline.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import seg_helpers as H  # noqa: E402


def classmap_from_pipeline(seg_outputs: list[dict], label2id: dict, hw: tuple[int, int]) -> np.ndarray:
    """pipeline の [{'label','mask'}, ...] を1枚のクラスマップ (H,W) に畳み込む。

    各セグメントの mask（PIL の L 画像, 前景 255）の前景画素に、そのラベルの id を書く。
    セマンティックではマスク同士は重ならない（各画素は1クラス）ので順序は問題にならない。
    """
    h, w = hw
    class_map = np.zeros((h, w), dtype=np.int64)
    for seg in seg_outputs:
        cid = label2id[seg["label"]]
        m = np.asarray(seg["mask"]) > 127  # L 画像 → bool 前景
        class_map[m] = cid
    return class_map


def run_pipeline(image, device: torch.device) -> tuple[np.ndarray, list[str], dict]:
    """pipeline でセグメンテーションし、クラスマップ・出現ラベル・モデル設定を返す。"""
    seg = H.load_segformer_pipeline(device)
    label2id = seg.model.config.label2id  # {'wall':0, 'building':1, ...}
    num_classes = len(seg.model.config.id2label)
    outputs = seg(image)  # [{'score'(None), 'label', 'mask'(PIL L)}, ...]
    w, h = image.size
    class_map = classmap_from_pipeline(outputs, label2id, (h, w))
    labels_present = [o["label"] for o in outputs]
    return class_map, labels_present, {"num_classes": num_classes}


def run_manual(image, device: torch.device) -> np.ndarray:
    """手動（AutoImageProcessor + AutoModelForSemanticSegmentation）でクラスマップを返す。

    SegFormer のロジットは入力の 1/4 解像度で出る。post_process_semantic_segmentation が
    target_sizes に合わせて interpolate＋argmax まで行い、原寸のクラスマップを返す。
    """
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    proc = AutoImageProcessor.from_pretrained(H.SEGFORMER_ID)
    model = AutoModelForSemanticSegmentation.from_pretrained(H.SEGFORMER_ID).to(device).eval()
    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)  # out.logits: (1, 150, H/4, W/4)
    # ★ target_sizes は (高さ, 幅)。image.size=(幅,高さ) なので [::-1] で渡す。
    seg_map = proc.post_process_semantic_segmentation(out, target_sizes=[image.size[::-1]])[0]
    return seg_map.cpu().numpy().astype(np.int64)


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    image, src = H.load_user_or_synthetic_image()
    rgb = np.asarray(image)

    cm_pipe, labels_present, meta = run_pipeline(image, device)
    cm_manual = run_manual(image, device)

    # ADE20K は 150 クラス。決定的な乱数パレットで色付けする。
    palette = H.random_palette(meta["num_classes"], seed=7)
    color_pipe = H.colorize(cm_pipe, palette)
    over = H.overlay(rgb, color_pipe, alpha=0.55)

    # (A) pipeline と (B) 手動 の画素一致率（同じ後処理なのでほぼ一致するはず）。
    agree = float((cm_pipe == cm_manual).mean())

    H.save_panel(
        [rgb, color_pipe, over],
        [f"input ({src})", "SegFormer (ADE20K classes)", "overlay"],
        out / "02_segformer_pipeline.png",
        ncol=3,
    )
    summary = {
        "input": src,
        "device": str(device),
        "num_classes": meta["num_classes"],
        "labels_present_pipeline": labels_present,
        "pixel_agreement_pipeline_vs_manual": round(agree, 4),
    }
    (out / "02_segformer_pipeline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"device = {device}  model = {H.SEGFORMER_ID}")
    print(f"pipeline が付けたラベル: {labels_present}")
    print(f"(A)pipeline と (B)手動 の画素一致率 = {agree:.4f}（同じ後処理なので ≈1.0）")
    print("落とし穴: post_process の target_sizes は (高さ,幅)。image.size[::-1] で渡す。")
    print(f"saved: {out / '02_segformer_pipeline.png'}, {out / '02_segformer_pipeline.json'}")


if __name__ == "__main__":
    main()
