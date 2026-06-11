"""章末ミニプロジェクト: 生成 → 劣化 → 復元 → 編集 → 評価 の一気通貫パイプライン。

本章の全工程を 1 本に統合した「完成形」です。CPU で数分以内に完走します。

  1. **生成**: SD-Turbo でプロンプトから画像を生成（無ければ合成シーンで代替）。
  2. **整合評価**: 生成画像とプロンプトの CLIPScore を測る（参照なし評価）。
  3. **劣化**: 生成画像を ×2 ダウンスケール＋ノイズで「壊す」（参照あり評価の入力を作る）。
  4. **復元(超解像)**: 古典バイキュービック と Swin2SR で復元し、元画像との PSNR/SSIM を比較。
  5. **編集(インペイント)**: 一部を矩形マスクで隠し、cv2.inpaint で物体除去する。
  6. **集計**: モンタージュ画像と、全指標をまとめた JSON を outputs に保存。
  7. ★頑健性: 各モデルはガード済み。オフラインでも合成画像＋古典手法で必ず完走（exit 0）。

実行:
  uv run python lectures/31_generation_editing/mini_project.py
結果: outputs/31_generation_editing/mini_project_montage.png と mini_project_report.json
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

import gen_lab as gl

SIZE = 256
PROMPT = "a photo of a red apple on a wooden table"


def step_generate() -> tuple[np.ndarray, str]:
    """SD-Turbo で生成（1 step / guidance 0）。失敗時は合成シーンで代替。返り値 (RGB, 由来)。"""
    pipe = gl.load_t2i_pipeline()
    if pipe is None:
        scene, origin = gl.load_or_make_scene(SIZE, seed=0)
        return scene, origin if origin != "synthetic" else "fallback:synthetic"
    import torch

    with torch.inference_mode():
        img = pipe(
            prompt=PROMPT,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=SIZE,
            width=SIZE,
            generator=torch.manual_seed(0),
        ).images[0]
    return gl.to_np(img), "sd-turbo"


def step_clip_score(rgb: np.ndarray) -> float | None:
    """生成画像 × プロンプトの CLIPScore（取得できなければ None）。"""
    scorer = gl.load_clip_scorer()
    return None if scorer is None else round(scorer(rgb, PROMPT), 4)


def step_restore(hr: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """HR→劣化→復元（bicubic / swin2sr）。復元画像と指標を返す。

    劣化は ×2 ダウンスケール（Swin2SR classical-sr が想定する『ノイズなしのバイキュービック
    縮小』に合わせる）。ノイズを足すと Swin2SR は学習想定外となり古典に負けることがある——
    その劣化ミスマッチは 04 の評価回で別途扱う。
    """
    degraded = gl.downscale(hr, scale=2)  # 低解像（超解像の正準的な劣化）
    target = (hr.shape[1], hr.shape[0])

    restored = {"bicubic": gl.upscale_bicubic(degraded, target)}
    swin = gl.load_swin2sr()
    if swin is not None:
        proc, model = swin
        sr = gl.swin2sr_upscale(proc, model, degraded)
        restored["swin2sr"] = cv2.resize(sr, target)

    metrics = {
        name: {"psnr": round(gl.psnr(hr, img), 3), "ssim": round(gl.ssim(hr, img), 4)}
        for name, img in restored.items()
    }
    # 劣化直後（バイキュービック前）も参考に
    base = gl.upscale_bicubic(degraded, target)
    restored["_degraded_input"] = base
    return restored, metrics


def step_inpaint(hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """中央付近の矩形を「除去対象」としてインペイント。(マスク重ね, 除去結果) を返す。"""
    rect = (int(SIZE * 0.40), int(SIZE * 0.40), int(SIZE * 0.25), int(SIZE * 0.25))
    mask = gl.make_box_mask(SIZE, rect)
    removed = cv2.inpaint(hr, mask, 3, cv2.INPAINT_TELEA)
    overlay = hr.copy()
    overlay[mask > 0] = (255, 0, 0)
    return overlay, removed


def build_montage(generated, restored, overlay, removed) -> np.ndarray:
    """主要な成果物を 1 枚に並べる（生成 / 劣化入力 / 最良復元 / 除去対象 / 除去後）。"""
    best = restored.get("swin2sr", restored["bicubic"])
    panels = [generated, restored["_degraded_input"], best, overlay, removed]
    return gl.hstack(panels)


def main() -> None:
    gl.ensure_dirs()
    print("=" * 70)
    print("ミニプロジェクト: 生成 → 評価 → 劣化 → 復元 → 編集")

    # 1) 生成
    generated, origin = step_generate()
    print(f"[1] 生成: 由来={origin}")
    gl.save_rgb(generated, "mini_project_generated.png")

    # 2) プロンプト整合（CLIPScore）
    clip = step_clip_score(generated)
    print(f"[2] CLIPScore(プロンプト整合): {clip}")

    # 3-4) 劣化 → 復元 → 参照あり評価
    restored, restore_metrics = step_restore(generated)
    print("[3-4] 復元品質（PSNR↑ / SSIM↑）:")
    for name, m in restore_metrics.items():
        print(f"      {name:9}: PSNR={m['psnr']:6.2f} dB  SSIM={m['ssim']:.4f}")

    # 5) インペイント編集
    overlay, removed = step_inpaint(generated)
    gl.save_rgb(removed, "mini_project_inpainted.png")
    print("[5] インペイント: 矩形領域を cv2.inpaint で除去")

    # 6) モンタージュ＋レポート JSON
    montage = build_montage(generated, restored, overlay, removed)
    gl.save_rgb(montage, "mini_project_montage.png")
    report = {
        "prompt": PROMPT,
        "generation_origin": origin,
        "clip_score": clip,
        "restore_metrics": restore_metrics,
        "metric_directions": {
            "psnr": "higher_better",
            "ssim": "higher_better",
            "clip_score": "higher_better",
        },
        "notes": "CPU・合成データ完結。Swin2SR/SD-Turbo/CLIP は未取得でも古典手法で完走。",
    }
    path = os.path.join(gl.OUT_DIR, "mini_project_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[6] 保存: mini_project_montage.png / mini_project_report.json")
    print("\n完了。生成・復元・編集・評価を 1 本で通せました。")


if __name__ == "__main__":
    main()
