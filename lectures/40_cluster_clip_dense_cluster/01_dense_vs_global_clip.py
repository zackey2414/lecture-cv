"""01: なぜ「画像全体 1 ベクトル」では足りないのか — global CLIP vs dense CLIP。

学ぶこと:
  - 従来の CLIP 画像検索は 1 枚 = 1 本のベクトル（CLS 由来）。画像全体の「雰囲気」は
    つかめるが、「画面のどこに何があるか」は表現できない。
  - dense CLIP は visual encoder を手で展開して **パッチ単位** の特徴 [C, gh, gw] を取り出す。
    7x7=49 本のベクトルが空間に並ぶので、領域ごとの検索・ローカライズが可能になる。
  - 小物体は 1 本に潰すと希釈される（49 パッチ中 数パッチ）。dense なら自分の領域を持てる。

実行:
  uv run python lectures/40_cluster_clip_dense_cluster/01_dense_vs_global_clip.py
結果は lectures/40_cluster_clip_dense_cluster/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless 環境（imshow を呼ばない）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cc_common as C  # noqa: E402


def main() -> None:
    out = C.output_dir()
    enc = C.load_encoder()

    # === 1. 合成シーン（複数物体 + 小物体の黄色いボール） ==================
    img_bgr, regions = C.make_scene(seed=0)
    x = torch.stack([enc.preprocess_bgr(img_bgr)])  # [1,3,224,224]

    # === 2. 従来の global ベクトル（1 枚 = 1 本） =========================
    g = enc.global_embedding(x)  # [1, C]
    print(f"[global] 画像全体の埋め込み shape={g.shape}  ノルム={np.linalg.norm(g[0]):.3f}")
    print("         -> 1 本のベクトル。画像全体の検索はできるが『どこに』は分からない。")

    # === 3. dense 特徴（パッチを空間に並べた [C, gh, gw]） ================
    fmap = enc.dense(x)[0]  # [C, gh, gw]
    c, gh, gw = fmap.shape
    print(f"\n[dense]  パッチ特徴 shape={fmap.shape}  = {gh}x{gw}={gh * gw} 本のベクトル")
    # 各パッチが単位ベクトルであることを確認（L2 正規化済み）
    norms = np.linalg.norm(fmap.reshape(c, -1), axis=0)
    print(f"         各パッチのノルム: min={norms.min():.3f} max={norms.max():.3f} (=1 なら正規化OK)")

    # === 4. 小物体の希釈を定量化 ========================================
    # 黄色いボールの bbox が 7x7 グリッドの何パッチを占めるか。
    ball = next(r for r in regions if r["label"] == "a yellow ball")
    x1, y1, x2, y2 = ball["bbox"]
    H, W = img_bgr.shape[:2]
    gx1, gy1 = x1 * gw // W, y1 * gh // H
    gx2, gy2 = max(gx1, (x2 - 1) * gw // W), max(gy1, (y2 - 1) * gh // H)
    n_ball = (gy2 - gy1 + 1) * (gx2 - gx1 + 1)
    print(
        f"\n[希釈]   小物体 'a yellow ball' はグリッド上 約{n_ball}/{gh * gw} パッチ。"
        f"\n         global 1 本では信号の約 {100 * n_ball / (gh * gw):.0f}% に薄まる"
        f" -> dense なら専用の領域ベクトルを持てる。"
    )

    # === 5. 可視化: 原画像 | パッチ格子 | dense 特徴の PCA-RGB ============
    img_rgb = img_bgr[:, :, ::-1]  # BGR -> RGB
    feat_rgb = C.features_to_rgb(fmap)  # [gh, gw, 3]
    feat_vis = np.stack(
        [C.upscale_nearest(feat_rgb[..., k], (H, W)) for k in range(3)], axis=-1
    ).astype(np.uint8)

    grid_img = img_rgb.copy()
    for gi in range(1, gh):
        grid_img[gi * H // gh, :] = (255, 255, 255)
    for gj in range(1, gw):
        grid_img[:, gj * W // gw] = (255, 255, 255)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    axes[0].imshow(img_rgb)
    axes[0].set_title("scene (global = 1 vector)")
    axes[1].imshow(grid_img)
    axes[1].set_title(f"patch grid {gh}x{gw} (dense = {gh * gw} vectors)")
    axes[2].imshow(feat_vis)
    axes[2].set_title("dense features (PCA->RGB)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Global CLIP vs Dense CLIP", fontsize=13)
    fig.tight_layout()
    path = out / "01_dense_vs_global.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\n[save] {path}")


if __name__ == "__main__":
    main()
