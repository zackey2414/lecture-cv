"""02: dense CLIP 特徴の取り出し方 — ViT visual encoder を手で展開する。

学ぶこと:
  - CLIP の通常 API（encode_image）は最後にトークンを 1 本へ pool してしまう。
    領域検索にはパッチトークンが要るので、transformer を自前で forward して
    最終トークン列を取り出し、**CLS を捨てて** 空間 (gh, gw) に並べ替える。
  - 各ステップの shape を追う: conv1(パッチ埋め込み) -> +CLS -> +位置埋め込み
    -> ln_pre -> transformer -> ln_post -> proj(共通空間) -> CLS除去 -> L2正規化。
  - 前処理は CenterCrop を排し正方形へ強制 Resize。端を切らないので、画面端の
    小物体も dense 特徴に残る（領域分割の品質に直結する勘所）。

実行:
  uv run python lectures/40_cluster_clip_dense_cluster/02_dense_features_extraction.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cc_common as C  # noqa: E402


def trace_vit(visual, images) -> None:
    """dense_vit_tokens と同じ手順を step ごとに print（中身を理解するための実況）。"""
    if visual is None:  # フォールバック時はスキップ
        print("[trace] CLIP 非搭載のためフォールバック。dense() の出力 shape のみ確認します。")
        return
    with torch.inference_mode():
        x = visual.conv1(images)
        b, width, gh, gw = x.shape
        print(f"  conv1        -> {tuple(x.shape)}  (パッチ埋め込み: {gh}x{gw} グリッド, 幅={width})")
        x = x.reshape(b, width, gh * gw).permute(0, 2, 1).contiguous()
        print(f"  flatten      -> {tuple(x.shape)}  (パッチを系列に)")
        cls = visual.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(b, 1, -1)
        x = torch.cat([cls, x], dim=1)
        print(f"  +CLS         -> {tuple(x.shape)}  (先頭に CLS トークン)")
        x = x + visual.positional_embedding.to(x.dtype).unsqueeze(0)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        print(f"  transformer  -> {tuple(x.shape)}  (自己注意で文脈を混ぜる)")
        proj = getattr(visual, "proj", None)
        if proj is not None:
            x = x @ proj
            print(f"  proj         -> {tuple(x.shape)}  (テキストと同じ共通潜在空間へ)")
        x = x[:, 1:, :]
        print(f"  drop CLS     -> {tuple(x.shape)}  (★ CLS を捨てパッチだけ残す)")


def main() -> None:
    out = C.output_dir()
    enc = C.load_encoder()

    img_bgr, _ = C.make_scene(seed=1)
    x = torch.stack([enc.preprocess_bgr(img_bgr)])
    print(f"[input] 前処理後テンソル {tuple(x.shape)} (CenterCrop なし・正方形 Resize)")

    print("\n[trace] ViT を手で forward して dense 特徴を作る:")
    trace_vit(enc._model.visual if enc.backend == "clip" else None, x)

    fmap = enc.dense(x)[0]  # [C, gh, gw]
    c, gh, gw = fmap.shape
    print(f"\n[result] dense 特徴マップ [C,H,W]={fmap.shape}  ({gh * gw} パッチ x {c} 次元)")

    # 各パッチが単位ベクトルか（L2 正規化の確認）
    norms = np.linalg.norm(fmap.reshape(c, -1), axis=0)
    assert np.allclose(norms, 1.0, atol=1e-3), "パッチが L2 正規化されていない"
    print(f"[check] 全パッチが単位ベクトル (norm in [{norms.min():.3f}, {norms.max():.3f}]) -> OK")

    # 隣接パッチほど特徴が似ているか（空間的な滑らかさの簡易確認）
    horiz = float(np.mean(np.sum(fmap[:, :, :-1] * fmap[:, :, 1:], axis=0)))
    print(f"[check] 水平方向の隣接パッチ 平均コサイン={horiz:.3f}（高いほど空間的に滑らか）")

    # 可視化: 原画像 + パッチ格子 / dense 特徴 PCA-RGB
    img_rgb = img_bgr[:, :, ::-1]
    H, W = img_rgb.shape[:2]
    grid = img_rgb.copy()
    for gi in range(1, gh):
        grid[gi * H // gh, :] = (255, 255, 255)
    for gj in range(1, gw):
        grid[:, gj * W // gw] = (255, 255, 255)
    feat_rgb = C.features_to_rgb(fmap)
    feat_vis = np.stack(
        [C.upscale_nearest(feat_rgb[..., k], (H, W)) for k in range(3)], axis=-1
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    axes[0].imshow(grid)
    axes[0].set_title(f"patches {gh}x{gw}")
    axes[1].imshow(feat_vis)
    axes[1].set_title("dense features (PCA->RGB)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Dense ViT patch features", fontsize=13)
    fig.tight_layout()
    path = out / "02_dense_features.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\n[save] {path}")


if __name__ == "__main__":
    main()
