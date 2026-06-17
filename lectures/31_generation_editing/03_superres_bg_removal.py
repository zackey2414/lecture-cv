"""03: 超解像（古典補間 → Swin2SR）と 背景除去（GrabCut）。

学ぶこと:
  - 超解像（low-res → high-res）の評価の正しいやり方: きれいな高解像 HR を **ダウンスケール
    して LR を作り**、それを各手法で復元 → 元の HR と PSNR/SSIM で比較する（参照あり評価）。
  - 古典ベースライン: `cv2.resize`（バイキュービック / Lanczos）。
    深層: `Swin2SRForImageSuperResolution`（×2、Transformer ベース）。差を定量比較する。
  - `cv2.dnn_superres`（ESPCN/FSRCNN/EDSR）は opencv-contrib 限定で headless 版には無い。
    存在チェックして無ければ案内のみ（概念として紹介）。
  - 背景除去: 古典 **GrabCut**（矩形で前景を初期化 → 反復でセグメント）で前景を切り出し、
    別の背景に合成する。深層マッティング（rembg 等）は概念・任意（実行経路に入れない）。

実行:
  uv run python lectures/31_generation_editing/03_superres_bg_removal.py
Swin2SR が無くても古典補間と GrabCut は動き、**exit 0** で終わります。
結果は lectures/31_generation_editing/outputs/ に保存（headless）。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import gen_lab as gl

SIZE = 256  # HR のサイズ。LR は SIZE//2、復元先は再び SIZE


# ----------------------------------------------------------------------------
# (A) 超解像
# ----------------------------------------------------------------------------
def demo_super_resolution(hr_rgb: np.ndarray) -> None:
    """HR→LR→各手法で復元→PSNR/SSIM 比較。Swin2SR があれば深層も比較。"""
    lr = gl.downscale(hr_rgb, scale=2)  # 128x128 の低解像
    target = (hr_rgb.shape[1], hr_rgb.shape[0])  # (w, h)=(256,256)
    print(f"\n[超解像] HR {hr_rgb.shape[:2]} → LR {lr.shape[:2]} → 復元 {target[::-1]}")

    results: dict[str, np.ndarray] = {}
    # 最近傍（最弱のベースライン。ブロックノイズが出る）
    results["nearest"] = cv2.resize(lr, target, interpolation=cv2.INTER_NEAREST)
    # バイキュービック / Lanczos（古典の定番）
    results["bicubic"] = gl.upscale_bicubic(lr, target)
    results["lanczos"] = gl.upscale_lanczos(lr, target)

    # 深層: Swin2SR（×2）。ロードできれば追加
    swin = gl.load_swin2sr()
    if swin is not None:
        proc, model = swin
        t0 = time.time()
        sr = gl.swin2sr_upscale(proc, model, lr)
        sr = cv2.resize(sr, target)  # 念のため目標サイズへ厳密化
        results["swin2sr"] = sr
        print(f"  Swin2SR 推論: {time.time() - t0:5.1f}s")

    # 評価（元 HR が ground truth）
    print("  手法ごとの復元品質（PSNR は大きいほど良い / SSIM は 1 に近いほど良い）:")
    order = ["nearest", "bicubic", "lanczos", "swin2sr"]
    panels = [hr_rgb]
    for name in order:
        if name not in results:
            continue
        img = results[name]
        print(f"    {name:9}: PSNR={gl.psnr(hr_rgb, img):5.2f} dB  SSIM={gl.ssim(hr_rgb, img):.4f}")
        panels.append(img)
    gl.save_rgb(gl.hstack(panels), "03_superres_compare.png")
    print("  保存: 03_superres_compare.png（左端=元 HR、以降が各復元）")

    # dnn_superres は contrib 限定 → 有無を案内
    if not hasattr(cv2, "dnn_superres"):
        print(
            "  [補足] cv2.dnn_superres は opencv-contrib 限定で headless 版には無いためスキップ。"
        )
        print("        ESPCN/FSRCNN/EDSR を使うなら opencv-contrib-python が必要（概念のみ紹介）。")


# ----------------------------------------------------------------------------
# (B) 背景除去（GrabCut）
# ----------------------------------------------------------------------------
def make_foreground_scene() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """単純な背景の上に前景オブジェクト（色付きの円＋四角）を置いた画像と、その矩形を返す。"""
    img = np.full((SIZE, SIZE, 3), 220, np.uint8)  # 明るい無地背景
    # 背景に薄い縞（GrabCut が前景/背景を区別する必要がある状況にする）
    img[:, ::16] = (200, 205, 215)
    fg_rect = (int(SIZE * 0.30), int(SIZE * 0.30), int(SIZE * 0.40), int(SIZE * 0.45))
    x, y, w, h = fg_rect
    cv2.circle(img, (x + w // 2, y + h // 3), w // 3, (40, 80, 200), -1)
    cv2.rectangle(img, (x + w // 4, y + h // 2), (x + 3 * w // 4, y + h), (60, 170, 90), -1)
    return img, fg_rect


def grabcut_foreground(rgb: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """GrabCut で前景マスク（0/1 の uint8）を返す。矩形の外は確実な背景として初期化。"""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    # 確定/推定の前景(1,3)を 1、背景(0,2)を 0 にする
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def demo_background_removal() -> None:
    """GrabCut で前景を抜き、新しい背景（グラデーション）に合成する。"""
    print("\n[背景除去] GrabCut で前景を抽出 → 別背景に合成:")
    img, rect = make_foreground_scene()
    fg_mask = grabcut_foreground(img, rect)
    coverage = float(fg_mask.mean())
    print(f"  前景マスクの面積比: {coverage:.2%}")

    # 新しい背景（縦グラデ）に α 合成（前景=元画素, 背景=新背景）
    new_bg = np.zeros_like(img)
    for yy in range(SIZE):
        t = yy / SIZE
        new_bg[yy, :] = (int(60 + 150 * t), int(200 - 80 * t), int(255 - 100 * t))
    alpha = fg_mask[..., None].astype(np.float32)
    composite = (img * alpha + new_bg * (1 - alpha)).astype(np.uint8)

    # 切り抜き（背景を白に）も作る
    cutout = np.where(fg_mask[..., None] > 0, img, 255).astype(np.uint8)
    gl.save_rgb(gl.hstack([img, cutout, composite]), "03_bg_removal.png")
    print("  保存: 03_bg_removal.png（元 / 切り抜き / 別背景に合成）")
    print("  → GrabCut は矩形ヒント＋反復で前景を推定。境界の毛羽立ちが苦手なところは")
    print("    深層マッティング（rembg / MODNet 等）が得意（本講座では概念のみ・任意）。")


def main() -> None:
    gl.ensure_dirs()
    print("=" * 70)
    print("03: 超解像（古典→Swin2SR）と 背景除去（GrabCut）")
    hr, origin = gl.load_or_make_scene(SIZE, seed=0)
    print(f"超解像の入力(HR): {origin}")

    demo_super_resolution(hr)
    demo_background_removal()

    print("\n完了。lectures/31_generation_editing/outputs/03_*.png を確認してください。")


if __name__ == "__main__":
    main()
