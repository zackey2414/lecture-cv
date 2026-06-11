"""02: 同じ処理（リサイズ・ぼかし・回転）を OpenCV / Pillow / scikit-image で書き比べる。

学ぶこと:
  - 「同じことをするAPI」がライブラリごとにどう違うか（引数の順序・データ型・既定の補間）。
  - OpenCV: ndarray(BGR, uint8)。dsize は (幅, 高さ) で shape の (高さ, 幅) と逆。
  - Pillow : PIL.Image(RGB)。size も (幅, 高さ)。resize の既定は LANCZOS を明示する流儀。
  - skimage: ndarray(RGB, float[0,1])。形は (高さ, 幅) 順。値域が 0〜1 になる点が最大の罠。
  - 結果を 1 枚の比較図にして「補間/平滑化の差」を目で見る。

scikit-image は任意ライブラリ。未導入でも OpenCV と Pillow の 2 列で必ず動く（exit 0）。
未導入のときは導入コマンドを案内してスキップする。

実行:
  uv run python lectures/02_cv_libraries_overview/02_same_op_across_libs.py
結果は outputs/02_cv_libraries_overview/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter

import matplotlib

matplotlib.use("Agg")  # pyplot より前に Agg 固定（headless 対応）
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir, probe, to_rgb  # noqa: E402


# === OpenCV 版（基準）======================================================
def ops_opencv(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """OpenCV で resize/blur/rotate を行い、表示用に RGB(uint8) で返す。

    覚えどころ:
      - resize の第2引数 dsize は (幅 W, 高さ H)。shape の (H, W) とは逆順！
      - 縮小は INTER_AREA がモアレに強い定石（拡大なら INTER_CUBIC/LINEAR）。
      - 回転は getRotationMatrix2D で行列を作り warpAffine で適用する。
    """
    h, w = bgr.shape[:2]

    # リサイズ（半分に縮小）。dsize=(w//2, h//2) の順に注意。
    resized = cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    # ガウシアンぼかし。ksize=(0,0) にすると sigma から自動でカーネルサイズが決まる。
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3.0)

    # 30 度回転（中心まわり・等倍）。warpAffine は元サイズのまま回す。
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
    rotated = cv2.warpAffine(bgr, mat, (w, h), flags=cv2.INTER_LINEAR)

    return {"resize": to_rgb(resized), "blur": to_rgb(blurred), "rotate": to_rgb(rotated)}


# === Pillow 版 =============================================================
def ops_pillow(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Pillow で同じ 3 処理を行い、表示用に RGB(uint8) で返す。

    覚えどころ:
      - Pillow は RGB が前提。cv2(BGR) から渡すときは必ず BGR→RGB 変換してから fromarray。
      - size も (幅 W, 高さ H)。resize((w,h)) の順は numpy の shape と逆。
      - Pillow 12 では Image.ANTIALIAS は廃止。Image.Resampling.LANCZOS を使う。
    """
    rgb = to_rgb(bgr)
    pil = Image.fromarray(rgb)               # RGB ndarray → PIL.Image
    w, h = pil.size                          # PIL の size は (幅, 高さ)

    resized = pil.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=3))
    rotated = pil.rotate(30, resample=Image.Resampling.BICUBIC, expand=False)

    return {
        "resize": np.asarray(resized),
        "blur": np.asarray(blurred),
        "rotate": np.asarray(rotated),
    }


# === scikit-image 版（任意）===============================================
def ops_skimage(bgr: np.ndarray, skimage_mod) -> dict[str, np.ndarray]:
    """scikit-image で同じ 3 処理。値域が float[0,1] になる点が OpenCV/Pillow と最大の違い。

    覚えどころ:
      - skimage は内部で float64[0,1] を好む。img_as_float で 0〜1 に正規化してから処理。
      - transform.resize の出力サイズは (高さ, 幅) 順（cv2/PIL の (幅,高さ) と逆！）。
      - 表示・保存時は img_as_ubyte で 0〜255 uint8 に戻す。
    """
    transform = skimage_mod.transform
    filters = skimage_mod.filters
    util = skimage_mod.util

    rgb = to_rgb(bgr)
    f = util.img_as_float(rgb)               # uint8[0,255] → float[0,1]
    h, w = f.shape[:2]

    # resize の出力 shape は (H, W) 順。anti_aliasing で縮小時のモアレを抑える。
    resized = transform.resize(f, (h // 2, w // 2), anti_aliasing=True)
    # gaussian は channel_axis でカラー軸を教える（新しめの skimage の作法）。
    blurred = filters.gaussian(f, sigma=3, channel_axis=-1)
    rotated = transform.rotate(f, 30, resize=False)

    return {
        "resize": util.img_as_ubyte(np.clip(resized, 0, 1)),
        "blur": util.img_as_ubyte(np.clip(blurred, 0, 1)),
        "rotate": util.img_as_ubyte(np.clip(rotated, 0, 1)),
    }


def save_comparison(out: pathlib.Path, results: dict[str, dict[str, np.ndarray]]) -> None:
    """ライブラリ×処理 のグリッド比較図を保存する。行=ライブラリ、列=処理。"""
    ops = ["resize", "blur", "rotate"]
    libs = list(results.keys())
    fig, axes = plt.subplots(len(libs), len(ops), figsize=(3.3 * len(ops), 3.0 * len(libs)))
    # axes が 1 行だと 1 次元になるので 2 次元へ正規化。
    axes = np.atleast_2d(axes)

    for r, lib in enumerate(libs):
        for c, op in enumerate(ops):
            ax = axes[r, c]
            ax.imshow(results[lib][op])      # すべて RGB(uint8) に揃えてある
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(op, fontsize=12)
            if c == 0:
                ax.set_ylabel(lib, fontsize=12)
    fig.suptitle("Same op across libraries (rows) x operations (cols)", y=1.0)
    fig.tight_layout()
    path = out / "02_same_op_compare.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  保存:", path.name)


def report_numeric_difference(cv_res: dict, pil_res: dict) -> None:
    """OpenCV と Pillow の結果がどれだけ数値的に違うかを表示する。

    同じ「リサイズ」でも補間アルゴリズムが違えば 1 画素ずつの値はズレる。
    「見た目はほぼ同じでも完全一致はしない」ことを数値で体感する。
    """
    print("\n=== OpenCV vs Pillow の数値差（平均絶対差 / 0-255スケール） ===")
    for op in ("resize", "blur", "rotate"):
        a = cv_res[op].astype(np.int16)
        b = pil_res[op].astype(np.int16)
        if a.shape != b.shape:
            print(f"  {op:7s}: shape 不一致 {a.shape} vs {b.shape}（サイズの丸めの差）")
            continue
        mad = float(np.mean(np.abs(a - b)))
        print(f"  {op:7s}: 平均絶対差 = {mad:5.2f}  ← 0 ではない=実装/補間が違う証拠")


def main() -> None:
    out = output_dir()
    bgr = get_sample_bgr()
    print("入力画像 shape:", bgr.shape, "(H, W, C) ※ OpenCV は BGR")

    results: dict[str, dict[str, np.ndarray]] = {}
    results["OpenCV"] = ops_opencv(bgr)
    results["Pillow"] = ops_pillow(bgr)

    # scikit-image は任意。入っていれば 3 列目として加える。
    sk = probe("scikit-image", "skimage", "uv add --group aug scikit-image")
    if sk.available:
        print(f"scikit-image 検出 (v{sk.version}) → 比較に追加します。")
        results["scikit-image"] = ops_skimage(bgr, sk.module)
    else:
        print("scikit-image は未導入のためスキップ（OpenCV と Pillow で比較）。")
        print("  使う場合:", sk.hint)

    report_numeric_difference(results["OpenCV"], results["Pillow"])

    print("\n=== 図を保存 ===")
    save_comparison(out, results)

    print("\n--- 要点（引数の順序とデータ型のクセ）---")
    print("  OpenCV : ndarray(BGR,uint8) / resize dsize=(W,H) / 縮小は INTER_AREA")
    print("  Pillow : PIL.Image(RGB)     / resize (W,H)        / LANCZOS を明示")
    print("  skimage: ndarray(RGB,float) / resize (H,W) 逆順!  / 出力 0〜1 を ubyte へ戻す")
    print("\n完了。02_same_op_compare.png を開いて補間/平滑化の差を見比べてください。")


if __name__ == "__main__":
    main()
