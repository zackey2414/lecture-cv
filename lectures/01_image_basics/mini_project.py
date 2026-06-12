"""章末ミニプロジェクト: 画像 I/O ＆ 色順サニティ・ツールキット。

この回で学んだ要素を1本に統合する総合課題。1枚の画像を入口に、
  - 画像の素性（shape / dtype / 値域 / カラーかグレーか）を JSON 化し、
  - cv2 / Pillow / matplotlib の3経路で「正しい色」で保存し（headless 前提）、
  - BGR を変換し忘れた「失敗例」も並べて、色順の食い違いを目で見せ、
  - cv2(BGR)→PIL(RGB)→numpy→cv2 のラウンドトリップが完全一致するか検証し、
  - 日本語(非ASCII)パスでの保存/読み戻しが往復できるか検証し、
  - uint8 オーバーフロー（numpy +）と飽和（cv2.add）の違いを数値と画像で示し、
  - 以上を1枚の比較パネル（mini_panel.png）＋ JSON ＋ テキストにまとめる。

ねらいは「画像入出力と色順を、自分一人で・画面が無くても・確実に検証できる
小さな健康診断ツールを完成させる」こと。実データを使いたい場合は
data/sample.jpg を置けば get_sample_bgr() が自動でそちらを使う。

実行:
  uv run python lectures/01_image_basics/mini_project.py
出力:
  outputs/01_image_basics/mini_panel.png / mini_report.json / mini_report.txt
  （加えて検証で使った中間ファイル mini_* も同じフォルダに残す）
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

# matplotlib は import の前に Agg（画面不要）へ固定する。これで DISPLAY 無しでも savefig が動く。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from PIL import Image  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    get_sample_bgr,
    imread_unicode,
    imwrite_unicode,
    output_dir,
)

# 明るさ調整で +する量。下端グラデーション帯の明るい画素で 255 を超えさせるため大きめにする。
BRIGHTNESS_DELTA = 120


def inspect_image(bgr: np.ndarray) -> dict:
    """画像の素性を JSON に載せられる素朴な型でまとめて返す（単一責務: 観察）。

    デバッグの第一歩は「いま手元の配列の素性を見る」こと。shape / dtype / 値域に加え、
    カラー(3次元)かグレー(2次元)かも記録しておく。
    """
    is_color = bgr.ndim == 3 and bgr.shape[2] == 3
    return {
        "shape": list(bgr.shape),                  # tuple は JSON にできないので list 化
        "dtype": str(bgr.dtype),                   # 期待は "uint8"
        "min": int(bgr.min()),
        "max": int(bgr.max()),
        "ndim": int(bgr.ndim),
        "is_color": bool(is_color),
        "channels": int(bgr.shape[2]) if is_color else 1,
    }


def save_three_libraries(bgr: np.ndarray, out: pathlib.Path) -> dict:
    """同じ画像を cv2 / Pillow / matplotlib の3経路で「正しい色」で保存する。

    要点は「OpenCV の外（PIL / matplotlib）へ出す瞬間に BGR→RGB へ変換する」こと。
    比較のため、変換を忘れた失敗例（BGR をそのまま RGB 扱い）も1枚保存する。
    戻り値は保存したファイル名と、PIL/matplotlib で読み戻した色が一致したかの検証結果。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # 外の世界へ渡す前に必ず変換

    # 方法A: cv2.imwrite は BGR 前提。そのまま渡すのが正しい。
    cv2.imwrite(str(out / "mini_cv2_correct.png"), bgr)
    # 方法B: Pillow は RGB 前提。変換済み rgb を渡すのが正しい。
    Image.fromarray(rgb).save(out / "mini_pil_correct.png")
    # 失敗例: BGR を RGB と誤解して渡すと赤と青が入れ替わる。
    Image.fromarray(bgr).save(out / "mini_pil_wrong.png")
    # 方法C: matplotlib も RGB 前提。plt.imsave に rgb を渡す。
    plt.imsave(str(out / "mini_mpl_correct.png"), rgb)

    # 検証: PIL で書いた correct を読み戻し、元の RGB と完全一致すれば色が崩れていない証拠。
    pil_back = np.asarray(Image.open(out / "mini_pil_correct.png").convert("RGB"))
    pil_ok = np.array_equal(rgb, pil_back)

    return {
        "files": [
            "mini_cv2_correct.png",
            "mini_pil_correct.png",
            "mini_pil_wrong.png",
            "mini_mpl_correct.png",
        ],
        "pil_color_preserved": bool(pil_ok),
    }


def verify_roundtrip(bgr: np.ndarray) -> bool:
    """cv2(BGR) → PIL(RGB) → numpy → cv2(BGR) と1周し、元と完全一致するか返す。

    一致すれば「3者の橋渡し（境界ごとに cvtColor）」が正しく閉じている証拠。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    back = cv2.cvtColor(np.asarray(Image.fromarray(rgb)), cv2.COLOR_RGB2BGR)
    return bool(np.array_equal(bgr, back))


def verify_unicode_io(bgr: np.ndarray, out: pathlib.Path) -> bool:
    """日本語(非ASCII)パスでの保存→読み戻しが、欠損なく往復できるか検証する。

    imencode+tofile / fromfile+imdecode の定石（imwrite_unicode / imread_unicode）を使う。
    """
    jp_path = out / "ミニ_日本語パス.png"
    if not imwrite_unicode(jp_path, bgr):
        return False
    reread = imread_unicode(jp_path, cv2.IMREAD_COLOR)
    if reread is None:               # None ガード（読み込み失敗を握りつぶさない）
        return False
    # PNG は可逆圧縮なので、往復しても画素は完全一致するはず。
    return bool(np.array_equal(bgr, reread))


def overflow_demo(bgr: np.ndarray, out: pathlib.Path) -> dict:
    """明るさ +delta を「numpy の +」と「cv2.add」で作り、画像と数値で違いを示す。

    numpy の + は 256 で巻き戻る（オーバーフロー）、cv2.add は 255 で頭打ち（飽和）。
    """
    add = np.full_like(bgr, BRIGHTNESS_DELTA)
    naive = bgr + add                 # 255 を超えると巻き戻る
    saturated = cv2.add(bgr, add)     # 255 で頭打ち

    cv2.imwrite(str(out / "mini_overflow_numpy.png"), naive)
    cv2.imwrite(str(out / "mini_overflow_cv2add.png"), saturated)

    # 明るい画素（右下端のグラデーション）を1点サンプリングして数値で比較する。
    sample = bgr[-1, -1]
    naive_px = (sample + np.uint8(BRIGHTNESS_DELTA)).tolist()
    sat_px = cv2.add(
        sample.reshape(1, 1, 3),
        np.full((1, 1, 3), BRIGHTNESS_DELTA, np.uint8),
    ).reshape(3).tolist()
    return {
        "delta": BRIGHTNESS_DELTA,
        "sample_pixel_bgr": sample.tolist(),
        "numpy_plus_pixel": [int(v) for v in naive_px],   # 巻き戻る例
        "cv2_add_pixel": [int(v) for v in sat_px],         # 飽和する例
    }


def build_panel(bgr: np.ndarray, out: pathlib.Path) -> pathlib.Path:
    """この回の要点を1枚に並べた比較パネルを保存する（headless でも見られる確認用）。

    2行×3列:
      [正しいRGB] [失敗: BGRをRGB扱い] [グレースケール (H,W)]
      [明るさ+ numpy(オーバーフロー)] [明るさ+ cv2.add(飽和)] [Rチャンネルのみ]
    図タイトルは ASCII で統一する（日本語フォント非依存にするため）。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)            # (H, W) 2次元になる
    add = np.full_like(bgr, BRIGHTNESS_DELTA)
    naive_rgb = cv2.cvtColor(bgr + add, cv2.COLOR_BGR2RGB)
    sat_rgb = cv2.cvtColor(cv2.add(bgr, add), cv2.COLOR_BGR2RGB)
    red_plane = bgr[:, :, 2]                                # R チャンネル（BGR の index 2）

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    panels = [
        (rgb, "OK: BGR2RGB (correct color)", None),
        (bgr, "WRONG: BGR shown as RGB", None),
        (gray, "grayscale  shape=(H,W)", "gray"),
        (naive_rgb, f"numpy + {BRIGHTNESS_DELTA} (overflow)", None),
        (sat_rgb, f"cv2.add {BRIGHTNESS_DELTA} (saturate)", None),
        (red_plane, "R channel only (BGR idx2)", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Module 01 sanity panel: color order / dims / uint8 overflow", fontsize=12)
    fig.tight_layout()
    panel_path = out / "mini_panel.png"
    fig.savefig(panel_path, dpi=110)
    plt.close(fig)
    return panel_path


def write_text_report(report: dict, path: pathlib.Path) -> None:
    """人が読むテキストの要約を書き出す（数値の答え合わせ用）。"""
    info = report["image_info"]
    ov = report["overflow"]
    checks = report["checks"]
    lines = [
        "# Module 01 Image-Basics Sanity Report",
        "",
        "## image info",
        f"  shape    = {info['shape']}",
        f"  dtype    = {info['dtype']}",
        f"  min/max  = {info['min']} / {info['max']}",
        f"  is_color = {info['is_color']} (channels={info['channels']})",
        "",
        "## checks (True = OK)",
        f"  roundtrip cv2->PIL->numpy->cv2  = {checks['roundtrip_equal']}",
        f"  PIL color preserved             = {checks['pil_color_preserved']}",
        f"  unicode-path save/reload equal  = {checks['unicode_io_equal']}",
        "",
        "## uint8 overflow vs saturate (bright pixel)",
        f"  sample pixel (BGR) = {ov['sample_pixel_bgr']}",
        f"  numpy + {ov['delta']:<3d}       = {ov['numpy_plus_pixel']}  (wraps around)",
        f"  cv2.add {ov['delta']:<3d}       = {ov['cv2_add_pixel']}  (clamped at 255)",
        "",
        f"all_checks_passed = {report['all_checks_passed']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = output_dir()
    bgr = get_sample_bgr()  # data/sample.jpg があればそれ、無ければ合成画像（BGR uint8）

    # --- 各検証を順に実行（どれも単一責務の関数に分けてある）------------------
    info = inspect_image(bgr)
    save_info = save_three_libraries(bgr, out)
    roundtrip_ok = verify_roundtrip(bgr)
    unicode_ok = verify_unicode_io(bgr, out)
    overflow = overflow_demo(bgr, out)
    panel_path = build_panel(bgr, out)

    checks = {
        "roundtrip_equal": roundtrip_ok,
        "pil_color_preserved": save_info["pil_color_preserved"],
        "unicode_io_equal": unicode_ok,
    }
    report = {
        "image_info": info,
        "saved_files": save_info["files"] + [panel_path.name],
        "overflow": overflow,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
    }

    # --- レポート出力（JSON＋テキスト）----------------------------------------
    (out / "mini_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_text_report(report, out / "mini_report.txt")

    # --- コンソール要約 -------------------------------------------------------
    print("=== 画像 I/O ＆ 色順サニティ・ツールキット ===")
    print(f"  画像: shape={info['shape']} dtype={info['dtype']} "
          f"min/max={info['min']}/{info['max']} color={info['is_color']}")
    print("  検証:")
    for name, ok in checks.items():
        print(f"    [{'OK' if ok else 'NG'}] {name}")
    print(f"  uint8 +{overflow['delta']}: numpy+={overflow['numpy_plus_pixel']} "
          f"(巻き戻り) / cv2.add={overflow['cv2_add_pixel']} (飽和)")
    print(f"  全検証パス: {report['all_checks_passed']}")
    print(f"\nsaved: {panel_path}")
    print(f"       {out / 'mini_report.json'}")
    print(f"       {out / 'mini_report.txt'}")
    print("\n完了。outputs/01_image_basics/mini_panel.png を開いて目で確認してください。")


if __name__ == "__main__":
    main()
