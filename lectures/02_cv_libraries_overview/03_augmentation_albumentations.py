"""03: データ拡張（augmentation）の地図 — 概念を自作で掴み、albumentations 等と対比する。

学ぶこと:
  - データ拡張とは「1枚の画像から、ラベルを変えずに見た目の違う複数枚を作る」こと。
    学習データの多様性を水増しし、過学習を抑えて汎化性能を上げるのが目的。
  - まず main 依存（cv2/numpy）だけで小さな拡張パイプラインを自作し、仕組みを体感する。
  - 拡張が「画素の分布」に与える影響を、明るさのヒストグラムで可視化する（評価ポイント）。
  - 実務の定番 albumentations / torchvision transforms v2 / kornia は、入っていれば実演、
    無ければ「役割の違い」を説明して案内のみ（exit 0 を必ず守る）。

ライブラリの住み分け（この回の核心）:
  - albumentations  : CPU・ndarray・bbox/mask/keypoint も一緒に変換。検出/セグメ向けの定番。
  - torchvision v2  : Tensor・PyTorch 学習と一体。GPU・微分可能。
  - kornia          : Tensor・GPU バッチ・勾配が流れる（学習ループ内で拡張したい時）。

実行:
  uv run python lectures/02_cv_libraries_overview/03_augmentation_albumentations.py
結果は lectures/02_cv_libraries_overview/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")  # pyplot より前に Agg 固定（headless 対応）
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir, probe, to_rgb  # noqa: E402


# === 1. main 依存だけで作る自作の拡張パイプライン ==========================
# 「拡張＝確率的に画像へ手を加える関数の連なり」という骨格を、cv2/numpy だけで体感する。
# どのライブラリを使っても、内部でやっているのは結局こういう処理の集まり。

def aug_hflip(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """確率 0.5 で左右反転。numpy スライス [:, ::-1] が最も速い反転。"""
    if rng.random() < 0.5:
        return rgb[:, ::-1].copy()
    return rgb


def aug_brightness_contrast(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """明るさ・コントラストをランダムに変える。

    new = clip(alpha * x + beta, 0, 255)。alpha=コントラスト, beta=明るさ。
    uint8 のまま掛け算するとオーバーフローするので float で計算してから clip → uint8。
    """
    alpha = rng.uniform(0.7, 1.3)            # コントラスト倍率
    beta = rng.uniform(-40, 40)              # 明るさオフセット
    out = rgb.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_rotate(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """-15〜+15 度のランダム回転（中心まわり・等倍）。境界は反射で埋める。"""
    h, w = rgb.shape[:2]
    angle = rng.uniform(-15, 15)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(rgb, mat, (w, h), borderMode=cv2.BORDER_REFLECT)


def aug_hsv_jitter(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """色相(H)・彩度(S)をわずかに揺らす。OpenCV の HSV は H が 0-179 である点に注意。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.uniform(-10, 10))) % 180        # 色相は循環
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(rng.uniform(-30, 30)), 0, 255)  # 彩度
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def aug_gaussian_noise(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ガウスノイズを加える。撮影ノイズへの頑健性を上げる狙い。"""
    noise = rng.normal(0, 12, rgb.shape).astype(np.float32)
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def manual_pipeline(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """上の関数を順に適用する自作 Compose。各処理は確率的に効く。"""
    out = aug_hflip(rgb, rng)
    out = aug_brightness_contrast(out, rng)
    out = aug_rotate(out, rng)
    out = aug_hsv_jitter(out, rng)
    out = aug_gaussian_noise(out, rng)
    return out


def save_grid(out: pathlib.Path, original: np.ndarray, variants: list[np.ndarray],
              filename: str, title: str) -> None:
    """原画像＋拡張サンプルを 3x3 グリッドで保存する（すべて RGB 前提）。"""
    tiles = [original, *variants][:9]
    fig, axes = plt.subplots(3, 3, figsize=(9, 7))
    for i, ax in enumerate(axes.ravel()):
        if i < len(tiles):
            ax.imshow(tiles[i])
            ax.set_title("original" if i == 0 else f"aug #{i}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  保存:", path.name)


def save_distribution(out: pathlib.Path, original: np.ndarray,
                      rng: np.random.Generator) -> None:
    """拡張が「明るさの分布」に与える影響をヒストグラムで可視化する。

    評価ポイント: 拡張を多数回かけると、元画像 1 点だった平均明るさが
    ばらつきを持った分布に広がる。これが「データの多様性が増えた」ことの数値的な証拠。
    """
    means = []
    for _ in range(300):
        aug = manual_pipeline(original, rng)
        means.append(float(aug.mean()))     # 画像全体の平均明るさ（0-255）

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(means, bins=30, color="#ef6c00", alpha=0.8, label="augmented (300 samples)")
    ax.axvline(original.mean(), color="#1565c0", linewidth=2.5, label="original")
    ax.set_xlabel("mean brightness (0-255)")
    ax.set_ylabel("count")
    ax.set_title("Augmentation spreads the data distribution")
    ax.legend()
    fig.tight_layout()
    path = out / "03_aug_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("  保存:", path.name)
    print(f"  原画像の平均明るさ={original.mean():.1f} / "
          f"拡張後の範囲=[{min(means):.1f}, {max(means):.1f}]")


# === 2. albumentations（任意・定番）======================================
def demo_albumentations(out: pathlib.Path, rgb: np.ndarray) -> None:
    """albumentations が入っていれば本物のパイプラインで拡張グリッドを作る。

    正準形: A.Compose([...]) を作り、transform(image=rgb)[\"image\"] で適用。
    入力は RGB の ndarray（OpenCV の BGR から渡すなら先に BGR→RGB 変換しておく）。
    """
    p = probe("albumentations", "albumentations", "uv add --group aug albumentations")
    if not p.available:
        print("\n[albumentations] 未導入のためスキップ。")
        print("  役割: CPU・ndarray・bbox/mask/keypoint も同時変換できる検出/セグメ向け定番。")
        print("  使う場合:", p.hint)
        return

    a = p.module
    print(f"\n[albumentations] 検出 (v{p.version}) → 実演します。")
    # バージョン差で一部 Transform 名が変わることがあるため、構築自体も try で守る。
    try:
        transform = a.Compose([
            a.HorizontalFlip(p=0.5),
            a.RandomBrightnessContrast(p=0.8),
            a.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.8),
            a.HueSaturationValue(p=0.5),
            a.GaussNoise(p=0.4),
        ])
        variants = [transform(image=rgb)["image"] for _ in range(8)]
    except Exception as e:  # noqa: BLE001  バージョン差・引数差は案内に変えて握る
        print("  パイプライン構築に失敗（バージョン差の可能性）→ スキップ:", e)
        return
    save_grid(out, rgb, variants, "03_albumentations_grid.png",
              "albumentations Compose (8 random augmentations)")


# === 3. torchvision transforms v2（任意・学習一体型）=====================
def demo_torchvision_v2(out: pathlib.Path, rgb: np.ndarray) -> None:
    """torchvision が入っていれば transforms.v2 で拡張する（Tensor ベース）。"""
    p = probe("torchvision", "torchvision", "uv add --group dl torchvision")
    if not p.available:
        print("\n[torchvision v2] 未導入のためスキップ。")
        print("  役割: Tensor・PyTorch 学習と一体・GPU/微分可能。学習前処理の標準。")
        print("  使う場合:", p.hint)
        return

    print(f"\n[torchvision v2] 検出 (v{p.version}) → 実演します。")
    try:
        import torch  # torchvision があれば torch も入っている
        from torchvision.transforms import v2

        # (H,W,3)uint8 RGB → Tensor (3,H,W)。v2 は uint8 Tensor をそのまま扱える。
        chw = torch.from_numpy(rgb).permute(2, 0, 1)
        tf = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            v2.RandomRotation(degrees=15),
        ])
        variants = []
        for _ in range(8):
            t = tf(chw)                       # (3,H,W) uint8
            variants.append(t.permute(1, 2, 0).numpy())  # 表示用に (H,W,3) へ戻す
    except Exception as e:  # noqa: BLE001
        print("  実行に失敗 → スキップ:", e)
        return
    save_grid(out, rgb, variants, "03_torchvision_v2_grid.png",
              "torchvision transforms v2 (8 random augmentations)")


# === 4. kornia（任意・微分可能 GPU 拡張）の案内のみ ======================
def note_kornia() -> None:
    """kornia は導入有無に関わらず役割を一言案内する（重い実演はしない）。"""
    p = probe("kornia", "kornia", "uv add --group aug kornia")
    state = f"検出 (v{p.version})" if p.available else "未導入"
    print(f"\n[kornia] {state}。")
    print("  役割: Tensor・GPU バッチ拡張で勾配が流れる（拡張を学習ループ内で微分可能に）。")
    if not p.available:
        print("  使う場合:", p.hint)


def main() -> None:
    out = output_dir()
    # 拡張は乱数を使うので、再現できるよう固定シードの Generator を用意する。
    rng = np.random.default_rng(42)

    bgr = get_sample_bgr()
    rgb = to_rgb(bgr)                          # 以降は RGB で統一（拡張系は RGB 前提が多い）

    print("=== 自作パイプラインで拡張（main 依存だけ）===")
    variants = [manual_pipeline(rgb, rng) for _ in range(8)]
    save_grid(out, rgb, variants, "03_manual_aug_grid.png",
              "Hand-made augmentation pipeline (cv2 + numpy)")

    print("\n=== 拡張が分布に与える影響を可視化 ===")
    save_distribution(out, rgb, rng)

    # ここから先は任意ライブラリ。入っていれば実演、無ければ案内のみで必ず exit 0。
    demo_albumentations(out, rgb)
    demo_torchvision_v2(out, rgb)
    note_kornia()

    print("\n完了。03_manual_aug_grid.png と 03_aug_distribution.png を確認してください。")


if __name__ == "__main__":
    main()
