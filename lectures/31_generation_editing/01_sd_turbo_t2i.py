"""01: text-to-image — SD-Turbo を CPU で動かす（steps=1〜2 / guidance=0.0）。

学ぶこと:
  - 拡散モデルの直感（ノイズ → 反復デノイズ → 画像）と、なぜ SD-Turbo は 1〜2 ステップで
    生成できるのか（蒸留：少ステップで多ステップ生成を再現するよう学習されている）。
  - `AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")` の正準フロー。
    **SD-Turbo は num_inference_steps=1〜2・guidance_scale=0.0 が必須**（>0 にすると崩れる）。
  - `torch.manual_seed` による再現性（同じ seed → 同じ画像、ビット一致を確認）。
  - CPU 設定: float32・`enable_attention_slicing()`・小サイズ(256〜384px)・少プロンプト。
  - safety checker の NSFW 判定で黒画像が返ることがある、という運用上の注意。

実行:
  uv run python lectures/31_generation_editing/01_sd_turbo_t2i.py
初回はモデル重み(~2.5GB)をダウンロードします。以後はキャッシュから高速に動きます。
モデルをロードできない（オフライン等）場合も、合成画像で代替して **必ず exit 0** で終わります。
結果は lectures/31_generation_editing/outputs/ に保存（画面表示はしない・headless）。
"""

from __future__ import annotations

import time

import numpy as np

import gen_lab as gl

# CPU でも数秒で終わるよう、プロンプトは 2 個・サイズは小さめに絞る（重要）
PROMPTS = [
    "a photo of a red apple on a wooden table",
    "a watercolor painting of a mountain lake at sunset",
]
SIZE = 256  # 256〜384 推奨。大きくするほど CPU 時間が増える


def generate_one(pipe, prompt: str, steps: int, seed: int, size: int) -> np.ndarray:
    """1 枚生成して RGB 配列で返す。SD-Turbo の必須パラメータをここで固定する。"""
    import torch

    generator = torch.manual_seed(seed)  # 再現性: 同じ seed → 同じ初期ノイズ
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,  # SD-Turbo は 1〜2
            guidance_scale=0.0,  # SD-Turbo は 0.0 必須（CFG を使わない蒸留モデル）
            height=size,
            width=size,
            generator=generator,
        )
    return gl.to_np(result.images[0])


def demo_text_to_image(pipe) -> list[np.ndarray]:
    """各プロンプトを 1 ステップで生成し、所要時間と共に保存する。"""
    outputs = []
    print("\n[生成] 各プロンプトを num_inference_steps=1 / guidance_scale=0.0 で生成:")
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        img = generate_one(pipe, prompt, steps=1, seed=i, size=SIZE)
        dt = time.time() - t0
        gl.save_rgb(img, f"01_t2i_prompt{i}.png")
        outputs.append(img)
        print(f"  [{i}] {dt:5.1f}s  '{prompt}' → 01_t2i_prompt{i}.png")
    gl.save_rgb(gl.hstack(outputs), "01_t2i_grid.png")
    return outputs


def demo_reproducibility(pipe) -> None:
    """同じ seed なら出力がビット一致、別 seed なら変わることを確認する。"""
    print("\n[再現性] torch.manual_seed の効果を検証:")
    a = generate_one(pipe, PROMPTS[0], steps=1, seed=123, size=SIZE)
    b = generate_one(pipe, PROMPTS[0], steps=1, seed=123, size=SIZE)
    c = generate_one(pipe, PROMPTS[0], steps=1, seed=999, size=SIZE)
    same = bool(np.array_equal(a, b))
    diff = int(np.abs(a.astype(int) - c.astype(int)).mean())
    print(f"  seed=123 を 2 回 → 完全一致: {same}（再現できる）")
    print(f"  seed=123 vs seed=999 → 平均画素差: {diff}（seed が変われば絵も変わる）")


def demo_steps(pipe) -> None:
    """steps=1 と steps=2 を比べる。SD-Turbo では 1 でも成立、2 で少し精細になる。"""
    print("\n[ステップ数] steps=1 と steps=2 を比較（CPU 時間も計測）:")
    imgs = []
    for steps in (1, 2):
        t0 = time.time()
        img = generate_one(pipe, PROMPTS[1], steps=steps, seed=7, size=SIZE)
        print(f"  steps={steps}: {time.time() - t0:5.1f}s")
        imgs.append(img)
    gl.save_rgb(gl.hstack(imgs), "01_t2i_steps_1_vs_2.png")
    print("  → 通常の SD は 20〜50 ステップ必要だが、SD-Turbo は蒸留により 1〜2 で足りる。")
    print("    guidance_scale は 0.0 のまま（>0 にすると Turbo は破綻する）。")


def fallback_without_model() -> None:
    """モデルが無いとき: 合成画像を保存し、何が起きるはずだったかを説明して exit 0 を保つ。"""
    print("\n[フォールバック] 拡散モデル無しのため合成シーンを保存します（学習用の代替）。")
    scene = gl.make_scene(SIZE, seed=0)
    gl.save_rgb(scene, "01_t2i_fallback_synthetic.png")
    print("  本来は SD-Turbo がプロンプトから画像を生成します。ネット接続時に再実行してください。")


def main() -> None:
    gl.ensure_dirs()
    print("=" * 70)
    print("01: text-to-image (SD-Turbo, CPU)")
    print("拡散モデル＝ランダムノイズから少しずつノイズを除いて画像にする生成器。")
    print("SD-Turbo は『1〜2 ステップで多ステップ生成を再現する』よう蒸留されている。")

    pipe = gl.load_t2i_pipeline()  # 失敗時は None
    if pipe is None:
        fallback_without_model()
        print("\n完了（フォールバック）。lectures/31_generation_editing/outputs/ を確認してください。")
        return

    demo_text_to_image(pipe)
    demo_reproducibility(pipe)
    demo_steps(pipe)

    print("\n[注意] safety checker: 通常の SD は NSFW 判定で黒画像を返すことがある。")
    print("  SD-Turbo の既定構成は safety_checker=None（黒画像にはならないが、")
    print("  公開サービスでは内容フィルタを別途用意するのが実務上の作法）。")
    print("\n完了。lectures/31_generation_editing/outputs/01_*.png を確認してください。")


if __name__ == "__main__":
    main()
