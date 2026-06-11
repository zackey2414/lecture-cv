"""03: 画像品質評価(IQA) — 参照あり(PSNR/SSIM) と 無参照(分散ラプラシアン等) を自前計算。

学ぶこと:
  - **参照あり (full-reference)**: 劣化前のクリーン画像と比べて品質を測る。
      * PSNR … 画素誤差(MSE)の対数。高いほど良い。値は分かりやすいが知覚と乖離しやすい。
      * SSIM … 局所の明るさ・コントラスト・構造の一致。高いほど良い。知覚に PSNR より近い。
    復元(デノイズ/インペイント)・超解像・生成の「どれだけ元に近いか」を測るのに使う。
  - **無参照 (no-reference / blind)**: 参照が無いときに 1 枚だけで品質を推定。
      * variance of Laplacian / Tenengrad … 鮮鋭度（ボケ検出）。高いほど鮮鋭。
    実運用（撮影品質ゲート・ボケ画像の自動弾き）で「参照が無い」状況に対応する。
  - **lower_better の向き**: 指標ごとに「高い方が良い/低い方が良い」が違う。
    PSNR/SSIM/鮮鋭度は高い方が良いが、LPIPS/BRISQUE/NIQE は低い方が良い。混同しないこと。
  - 31(生成)・29(超解像) の評価との接続: これらの指標がそのまま生成/復元の良し悪しの物差し。

実行:
  uv run python lectures/32_anomaly_iqa/03_iqa_metrics.py
出力:
  outputs/32_anomaly_iqa/03_degradations.png … 劣化画像と各指標
  outputs/32_anomaly_iqa/03_iqa_table.json    … 指標表（向き付き）
"""

from __future__ import annotations

import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import anomaly_iqa_lab as lab


def make_degradations(clean: np.ndarray) -> dict[str, np.ndarray]:
    """クリーン画像から代表的な劣化版を作る（参照=clean を後で使う）。"""
    rng = np.random.RandomState(0)
    noise = np.clip(clean.astype(np.float32) + rng.normal(0, 25, clean.shape), 0, 255).astype(
        np.uint8
    )
    blur = cv2.GaussianBlur(clean, (0, 0), sigmaX=2.0)
    # JPEG 風の劣化: 縮小→拡大でディテールを落とす（コーデック非依存の簡易版）。
    small = cv2.resize(clean, (lab.IMG_SIZE // 4, lab.IMG_SIZE // 4), interpolation=cv2.INTER_AREA)
    blocky = cv2.resize(small, (lab.IMG_SIZE, lab.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return {"clean": clean, "gaussian_noise": noise, "blur": blur, "downscale": blocky}


def compute_table(clean: np.ndarray, variants: dict[str, np.ndarray]) -> list[dict]:
    """各劣化版について 参照あり/無参照 指標をまとめた行リストを作る。"""
    rows = []
    for name, img in variants.items():
        rows.append(
            {
                "variant": name,
                "psnr": round(lab.psnr(clean, img), 2),  # 参照あり（高いほど良い）
                "ssim": round(lab.ssim(clean, img), 3),  # 参照あり（高いほど良い）
                "var_laplacian": round(lab.variance_of_laplacian(img), 4),  # 無参照（鮮鋭度）
                "tenengrad": round(lab.tenengrad(img), 4),  # 無参照（鮮鋭度）
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    """表をコンソールに整形表示。指標名の隣に良し悪しの向きを添える。"""
    head = ("variant", "PSNR↑", "SSIM↑", "varLap↑", "Teneng↑")
    print(f"  {head[0]:14s}{head[1]:>9s}{head[2]:>8s}{head[3]:>10s}{head[4]:>10s}")
    for r in rows:
        print(
            f"  {r['variant']:14s}{r['psnr']:>9.2f}{r['ssim']:>8.3f}"
            f"{r['var_laplacian']:>10.4f}{r['tenengrad']:>10.4f}"
        )


def plot_variants(clean: np.ndarray, variants: dict, rows: list[dict], path: str) -> None:
    """劣化画像を並べ、各画像の主要指標を題に書く。"""
    fig, axes = plt.subplots(1, len(variants), figsize=(3.2 * len(variants), 3.6))
    for ax, (name, img), r in zip(axes, variants.items(), rows):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\nPSNR={r['psnr']} SSIM={r['ssim']}\nvarLap={r['var_laplacian']}")
    fig.suptitle("Image quality under degradations (PSNR/SSIM need a reference; varLap does not)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = lab.ensure_out_dir()
    print("=" * 64)
    print("03 画像品質評価(IQA): 参照あり PSNR/SSIM と 無参照 鮮鋭度")

    # クリーン画像はデータセットの正常テクスチャを 1 枚流用（合成・再現性あり）。
    ds = lab.make_dataset(n_train=1, n_test_normal=1, n_test_anom=0, seed=7)
    clean = ds["test"][0]

    variants = make_degradations(clean)
    rows = compute_table(clean, variants)
    print_table(rows)

    # lower_better の向き表（pyiqa の metric.lower_better に相当）。
    print("\n  指標の向き（True=小さいほど良い / False=大きいほど良い）:")
    for m in ["psnr", "ssim", "variance_of_laplacian", "lpips", "brisque", "niqe", "musiq"]:
        print(f"    {m:22s} lower_better = {lab.METRIC_DIRECTION[m]}")

    # 参照あり/無参照の「使い分け」を実演:
    #   - blur は PSNR/SSIM だけでなく varLaplacian の低下でも検出できる（参照不要）。
    #   - noise は鮮鋭度(varLaplacian)を“上げて”しまう → 鮮鋭度指標だけでは品質を測れない。
    print("\n  使い分けの要点:")
    print(
        f"    blur:  varLaplacian {rows[0]['var_laplacian']} → {rows[2]['var_laplacian']}（低下＝ボケを無参照で検出）"
    )
    print(
        f"    noise: varLaplacian {rows[0]['var_laplacian']} → {rows[1]['var_laplacian']}（ノイズで上昇＝鮮鋭度だけでは品質を測れない）"
    )

    img_path = os.path.join(out, "03_degradations.png")
    json_path = os.path.join(out, "03_iqa_table.json")
    plot_variants(clean, variants, rows, img_path)
    with open(json_path, "w") as f:
        json.dump(
            {"rows": rows, "metric_direction": lab.METRIC_DIRECTION},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  保存: {img_path}")
    print(f"  保存: {json_path}")
    print("  → 31(生成)・29(超解像)の評価でも、参照があれば PSNR/SSIM、無ければ無参照指標を選ぶ。")


if __name__ == "__main__":
    main()
