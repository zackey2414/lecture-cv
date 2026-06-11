"""02: img2img と インペイント（部分編集・物体除去）。

学ぶこと:
  - `AutoPipelineForImage2Image`（SD-Turbo）で「入力画像 × プロンプト」から変換する。
    ★最重要: SD-Turbo の img2img は **effective_steps = round(num_inference_steps * strength)
    が 1 以上** でなければエラーになる。strength（どれだけ元画像を壊すか）と steps の関係を体感する。
  - インペイント（マスク領域だけを描き直す）の正準的な考え方。本講座では **古典 cv2.inpaint
    (Telea / Navier-Stokes)** を主役にして「傷消し・物体除去」を実装する。大型の拡散インペイント
    モデルは概念＋任意（重い DL を避けるため既定ではスキップ。環境変数で有効化できる）。

実行:
  uv run python lectures/31_generation_editing/02_img2img_inpaint.py
モデルが無くても古典インペイントは必ず動き、**exit 0** で終わります。
結果は outputs/31_generation_editing/ に保存（headless）。
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

import gen_lab as gl

SIZE = 256


# ----------------------------------------------------------------------------
# (A) img2img: SD-Turbo で入力画像を変換
# ----------------------------------------------------------------------------
def run_img2img(pipe, init_rgb: np.ndarray, prompt: str, steps: int, strength: float) -> np.ndarray:
    """img2img を 1 回実行して RGB を返す。guidance_scale=0.0 は SD-Turbo の必須。"""
    import torch

    generator = torch.manual_seed(0)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=gl.to_pil(init_rgb),
            num_inference_steps=steps,
            strength=strength,  # 0=元画像のまま, 1=ほぼ作り直し
            guidance_scale=0.0,
            generator=generator,
        )
    return gl.to_np(result.images[0])


def demo_img2img(pipe, init_rgb: np.ndarray) -> None:
    """strength を変えて「元画像をどれだけ残すか」を観察する。"""
    prompt = "an oil painting of a house by a lake, vivid colors"
    print(
        "\n[img2img] strength を変えて変換（effective steps = round(steps*strength) ≥ 1 が必要）:"
    )
    panels = [init_rgb]
    # steps=4 固定。strength=0.5,0.75,1.0 で effective steps = 2,3,4
    for strength in (0.5, 0.75, 1.0):
        eff = round(4 * strength)
        if eff < 1:  # 念のためのガード（SD-Turbo はここでエラーを出す）
            print(f"  strength={strength}: effective steps={eff} <1 のためスキップ")
            continue
        t0 = time.time()
        out = run_img2img(pipe, init_rgb, prompt, steps=4, strength=strength)
        print(f"  strength={strength} (eff steps={eff}): {time.time() - t0:5.1f}s")
        panels.append(out)
    gl.save_rgb(gl.hstack(panels), "02_img2img_strength.png")
    print("  → strength が大きいほど元画像から離れ、プロンプト寄りになる（左端が入力）。")


# ----------------------------------------------------------------------------
# (B) 古典インペイント: cv2.inpaint で傷消し・物体除去
# ----------------------------------------------------------------------------
def classic_inpaint(damaged_rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Telea 法と Navier-Stokes 法で復元し、両方を返す。

    cv2.inpaint はマスク領域の境界から色を内側へ伝播させて埋める。小さな傷・細い欠損に強く、
    大きな穴や複雑な構造の復元は苦手（そこは深層 LaMa や拡散インペイントの出番）。
    """
    radius = 3  # 近傍をどれだけ参照するか
    telea = cv2.inpaint(damaged_rgb, mask, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(damaged_rgb, mask, radius, cv2.INPAINT_NS)
    return {"telea": telea, "ns": ns}


def demo_scratch_removal(clean_rgb: np.ndarray) -> None:
    """きれいな画像にわざと傷を付け→マスクで復元→PSNR/SSIM で元に戻った度合いを測る。"""
    print("\n[インペイント①] 傷消し（合成した傷をマスク指定で復元）:")
    damaged, mask = gl.add_scratches(clean_rgb, n=7, seed=3)
    results = classic_inpaint(damaged, mask)
    gl.save_rgb(
        gl.hstack([clean_rgb, damaged, results["telea"], results["ns"]]), "02_inpaint_scratch.png"
    )
    print("  保存: 02_inpaint_scratch.png（元 / 傷あり / Telea / NS）")
    # 参照あり評価: 復元結果が元画像にどれだけ近いか
    for name, img in results.items():
        print(
            f"    {name.upper():5}: PSNR={gl.psnr(clean_rgb, img):5.2f} dB  SSIM={gl.ssim(clean_rgb, img):.4f}"
        )
    print(f"    （参考）傷あり vs 元: PSNR={gl.psnr(clean_rgb, damaged):5.2f} dB → 復元で改善する")


def demo_object_removal(clean_rgb: np.ndarray) -> None:
    """画像中の一部（矩形領域）を「消したい物体」とみなしてインペイントで除去する。"""
    print("\n[インペイント②] 物体除去（矩形マスクの中身を周囲から埋める）:")
    rect = (int(SIZE * 0.62), int(SIZE * 0.12), int(SIZE * 0.22), int(SIZE * 0.22))  # 太陽あたり
    mask = gl.make_box_mask(SIZE, rect)
    removed = cv2.inpaint(clean_rgb, mask, 3, cv2.INPAINT_TELEA)
    # 可視化: マスクを赤で重ねた図も作る
    overlay = clean_rgb.copy()
    overlay[mask > 0] = (255, 0, 0)
    gl.save_rgb(gl.hstack([clean_rgb, overlay, removed]), "02_inpaint_object.png")
    print("  保存: 02_inpaint_object.png（元 / 除去対象=赤 / 除去後）")
    print("  → 古典法は『周囲の色で塗りつぶす』ため、消した跡が平坦になりやすい。")


def try_diffusion_inpaint(clean_rgb: np.ndarray) -> None:
    """任意: 拡散インペイント（AutoPipelineForInpainting）。重い DL を避けるため既定はスキップ。

    GEN_INPAINT_MODEL を設定したときだけ実行する（例: stabilityai/stable-diffusion-2-inpainting）。
    モデルは数 GB あり CPU でも遅いので、教材本体は古典 cv2.inpaint で完結させている。
    """
    model_id = os.environ.get("GEN_INPAINT_MODEL")
    if not model_id:
        print("\n[任意] 拡散インペイントはスキップ（既定）。")
        print("  使う場合: GEN_INPAINT_MODEL=stabilityai/stable-diffusion-2-inpainting で再実行。")
        print("  拡散インペイントはマスク内を『周囲＋プロンプトに整合する新しい内容』で生成でき、")
        print("  大穴や物体の置換は古典 cv2.inpaint より自然になる（ただし重い）。")
        return
    try:
        import torch
        from diffusers import AutoPipelineForInpainting

        pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float32).to(
            "cpu"
        )
        gl._quiet_pipe(pipe)
        rect = (int(SIZE * 0.62), int(SIZE * 0.12), int(SIZE * 0.22), int(SIZE * 0.22))
        mask = gl.make_box_mask(SIZE, rect)
        with torch.inference_mode():
            out = pipe(
                prompt="clear blue sky",
                image=gl.to_pil(clean_rgb),
                mask_image=gl.to_pil(np.stack([mask] * 3, -1)),
                num_inference_steps=15,
                guidance_scale=7.0,
            ).images[0]
        gl.save_rgb(gl.to_np(out), "02_inpaint_diffusion.png")
        print("\n[任意] 拡散インペイント完了 → 02_inpaint_diffusion.png")
    except Exception as e:  # noqa: BLE001
        print(f"\n[任意] 拡散インペイントに失敗（{type(e).__name__}）。スキップします。")


def main() -> None:
    gl.ensure_dirs()
    print("=" * 70)
    print("02: img2img と インペイント")
    clean, origin = gl.load_or_make_scene(SIZE, seed=0)
    print(f"入力画像: {origin}（data/ に画像を置けば実画像でも動きます）")

    # (A) img2img（SD-Turbo が無ければスキップして古典インペイントへ）
    pipe = gl.load_img2img_pipeline()
    if pipe is not None:
        demo_img2img(pipe, clean)
    else:
        print("\n[img2img] モデルが無いためスキップ（古典インペイントは続行します）。")

    # (B) 古典インペイント（常に動く・主役）
    demo_scratch_removal(clean)
    demo_object_removal(clean)

    # (C) 任意: 拡散インペイント（既定スキップ）
    try_diffusion_inpaint(clean)

    print("\n完了。outputs/31_generation_editing/02_*.png を確認してください。")


if __name__ == "__main__":
    main()
