"""04: 生成・復元の評価 — PSNR / SSIM（自前）・CLIPScore（CLIP）・FID（概念＋任意）。

学ぶこと:
  - 参照あり（復元・超解像・インペイント）: **PSNR と SSIM**。本講座では numpy/cv2 で
    自前実装し（gen_lab）、torchmetrics の実装と突き合わせて正しさを確認する。
  - 生成（text-to-image）のプロンプト整合: **CLIPScore**。CLIP の画像/テキスト埋め込みの
    コサイン類似度（×2.5）で「絵が指示に合っているか」を測る（参照画像が要らない）。
  - 分布の近さ（生成のリアルさ）: **FID**（InceptionV3 特徴の Fréchet 距離）。概念を説明し、
    torchmetrics があれば小デモ（★小サンプルでは極端に不安定、数百〜数万枚が前提）。
  - 指標ごとの「良い方向」を必ず明示する（混同が最頻ミス）。

実行:
  uv run python lectures/31_generation_editing/04_eval_psnr_ssim_clipscore.py
torchmetrics / CLIP が無くても自前指標は動き、**exit 0** で終わります。
"""

from __future__ import annotations

import numpy as np

import gen_lab as gl

SIZE = 256


# ----------------------------------------------------------------------------
# (A) PSNR / SSIM の自前実装を torchmetrics と突き合わせる
# ----------------------------------------------------------------------------
def demo_reference_metrics() -> None:
    """きれいな画像と劣化画像で PSNR/SSIM を計算し、自前 vs torchmetrics を比較。"""
    print("\n[参照あり] PSNR / SSIM（大きい/1 に近いほど良い）:")
    hr = gl.make_scene(SIZE, seed=0)
    noisy = gl.add_gaussian_noise(hr, sigma=18, seed=1)
    blur = gl.upscale_bicubic(gl.downscale(hr, 2), (SIZE, SIZE))  # ぼけ（ダウン→アップ）

    for name, img in [("noisy", noisy), ("blurred", blur)]:
        p, s = gl.psnr(hr, img), gl.ssim(hr, img)
        print(f"  {name:8}: PSNR={p:5.2f} dB  SSIM={s:.4f}（自前実装）")

    # torchmetrics と突き合わせ（あれば）。値がほぼ一致すれば自前実装が正しい根拠になる。
    _crosscheck_with_torchmetrics(hr, noisy)


def _crosscheck_with_torchmetrics(hr: np.ndarray, img: np.ndarray) -> None:
    """torchmetrics の PSNR/SSIM と自前実装を比較（無ければスキップ）。"""
    try:
        import torch
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )

        def to_t(x: np.ndarray) -> "torch.Tensor":
            return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)

        a, b = to_t(hr), to_t(img)
        psnr_m = PeakSignalNoiseRatio(data_range=255.0)
        ssim_m = StructuralSimilarityIndexMeasure(data_range=255.0)
        tm_psnr = float(psnr_m(b, a))
        tm_ssim = float(ssim_m(b, a))
        print("  [突き合わせ] torchmetrics vs 自前:")
        print(f"    PSNR: torchmetrics={tm_psnr:5.2f} / 自前={gl.psnr(hr, img):5.2f}（ほぼ一致）")
        print(
            f"    SSIM: torchmetrics={tm_ssim:.4f} / 自前={gl.ssim(hr, img):.4f}（境界処理差で僅差）"
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [突き合わせ] torchmetrics が使えないためスキップ（{type(e).__name__}）。")


# ----------------------------------------------------------------------------
# (B) CLIPScore: プロンプトと画像の整合
# ----------------------------------------------------------------------------
def demo_clip_score() -> None:
    """合成画像に対し、合う説明文／合わない説明文の CLIPScore を比較する。"""
    print("\n[生成整合] CLIPScore = 2.5 * max(cos(img, text), 0)（大きいほど整合が良い）:")
    scorer = gl.load_clip_scorer()
    if scorer is None:
        print("  CLIP が無いためスキップ。")
        return
    scene = gl.make_scene(SIZE, seed=0)  # 空・太陽・丘・家
    texts = [
        "a house and a sun on a sunny hill",  # 合う
        "a landscape with a house under the sky",  # ほどほど合う
        "a close-up photo of a kitten",  # 合わない
    ]
    for t in texts:
        print(f"  CLIPScore={scorer(scene, t):.3f}  ← '{t}'")
    print("  → 内容に合う文ほど高い。参照画像なしで『プロンプト忠実度』を測れるのが利点。")


# ----------------------------------------------------------------------------
# (C) FID: 分布間距離（概念＋任意の小デモ）
# ----------------------------------------------------------------------------
def demo_fid() -> None:
    """FID の概念説明と、torchmetrics があれば小サンプルデモ（不安定さも体感）。"""
    print("\n[分布距離] FID（InceptionV3 特徴の Fréchet 距離。小さいほど実画像分布に近い）:")
    print("  FID = ||μr-μg||^2 + Tr(Σr+Σg-2(ΣrΣg)^1/2)。実画像群と生成群の特徴分布を比べる。")
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance

        # 「実画像群」= 多様なシーン、「生成群A」= 似た分布、「生成群B」= ノイズで崩した分布
        rng = np.random.RandomState(0)

        def batch(kind: str, n: int = 24) -> "torch.Tensor":
            imgs = []
            for i in range(n):
                base = gl.make_scene(128, seed=int(rng.randint(0, 1_000_000)))
                if kind == "far":
                    base = gl.add_gaussian_noise(base, sigma=60, seed=i)  # 大きく劣化
                imgs.append(torch.from_numpy(base).permute(2, 0, 1))
            return torch.stack(imgs).to(torch.uint8)  # FID は uint8 (N,3,H,W)

        real, near, far = batch("real"), batch("near"), batch("far")
        fid = FrechetInceptionDistance(feature=64, normalize=False)  # feature=64 は軽量版
        fid.update(real, real=True)
        fid.update(near, real=False)
        fid_near = float(fid.compute())
        fid.reset()
        fid.update(real, real=True)
        fid.update(far, real=False)
        fid_far = float(fid.compute())
        print(f"  実 vs 似た分布: FID={fid_near:7.2f}")
        print(f"  実 vs 崩れた分布: FID={fid_far:7.2f}（崩れた方が大きい＝離れている）")
        print("  ★ここは各 24 枚の極小サンプル。FID は本来 数百〜数万枚で安定する点に注意。")
    except Exception as e:  # noqa: BLE001
        print(f"  torchmetrics(FID) が使えないため概念のみ（{type(e).__name__}）。")
        print(
            "  有効化（任意）: uv add --group metrics torch-fidelity（InceptionV3 特徴抽出に必要）。"
        )
        print("  実運用: 実画像と生成画像を各数千枚そろえ、同条件で特徴抽出して算出する。")


def print_metric_directions() -> None:
    """指標の良し悪し方向の早見表（混同が最頻ミスなので最後に固定する）。"""
    print("\n[早見表] 指標と良い方向:")
    print("  PSNR   ↑ 大きいほど良い（参照あり・忠実度。dB）")
    print("  SSIM   ↑ 1 に近いほど良い（参照あり・構造類似）")
    print("  LPIPS  ↓ 小さいほど良い（参照あり・知覚距離。要 net 重み＝任意）")
    print("  CLIPScore ↑ 大きいほど良い（生成・プロンプト整合）")
    print("  FID    ↓ 小さいほど良い（生成・分布の近さ）")
    print("  KID/IS は FID 同系統（KID↓ / IS↑）。十分なサンプル数が前提。")


def main() -> None:
    gl.ensure_dirs()
    print("=" * 70)
    print("04: 生成・復元の評価指標")
    demo_reference_metrics()
    demo_clip_score()
    demo_fid()
    print_metric_directions()
    print("\n完了。指標は『何を・どの向きで』測るかを取り違えないこと。")


if __name__ == "__main__":
    main()
