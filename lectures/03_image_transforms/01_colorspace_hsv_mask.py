"""01: 色空間変換（Gray / HSV）と inRange による色マスク。

学ぶこと:
  - cv2.cvtColor で BGR → Gray / HSV へ変換する。Gray は (H, W) で次元が1つ減る。
  - OpenCV の HSV は H=0-179（0-360 ではない！）、S/V=0-255 という独自スケール。
  - cv2.inRange で「特定の色域だけ」を白(255)に抜き出す2値マスクを作る。
  - 赤は色相環の 0/179 をまたぐので、2区間のマスクを bitwise_or で合成する。
  - cv2.split / cv2.merge でチャンネルを分解・再合成する。

実行:
  uv run python lectures/03_image_transforms/01_colorspace_hsv_mask.py
結果は outputs/03_image_transforms/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# 同じディレクトリの cv_helpers / preprocess を import できるようにする。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_color_scene_bgr, output_dir  # noqa: E402
from preprocess import extract_color  # noqa: E402


def main() -> None:
    out = output_dir()

    # 合成シーン（赤円・緑矩形・青円・黄矩形のある薄グレー背景）を BGR で用意する。
    bgr = make_color_scene_bgr()
    cv2.imwrite(str(out / "01_scene.png"), bgr)

    # === 1. グレースケール化は次元が1つ減る ================================
    # BGR(3ch) → Gray(1ch)。戻り値の shape は (H, W) で、チャンネル軸が消える。
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print("[1] グレースケール化")
    print("  元 shape :", bgr.shape, "→ gray shape:", gray.shape, "（次元が1つ減る）")
    cv2.imwrite(str(out / "01_gray.png"), gray)

    # === 2. HSV のスケールを数値で確認する ================================
    # 純色の画素を作って HSV に変換し、H が 0-179 スケールであることを目で確かめる。
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    print("\n[2] OpenCV の HSV スケール（H=0-179, S/V=0-255）")
    for name, sample_bgr in [("赤", (0, 0, 255)), ("黄", (0, 255, 255)),
                             ("緑", (0, 255, 0)), ("青", (255, 0, 0))]:
        px = np.uint8([[list(sample_bgr)]])
        h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
        print(f"  {name}: BGR={sample_bgr} → H={h:3d} S={s:3d} V={v:3d}")
    print("  ※ 一般的な色相環の角度(0-360)を半分にした値が H。赤は 0 付近＝179 もまたぐ。")

    # === 3. inRange で「青」だけを抜き出す（単一区間） ====================
    # 青の色相中心は H=120。その周辺と、彩度・明度の下限を指定して2値マスクを作る。
    # 下限の S/V を上げることで、背景の低彩度グレーを確実に除外できる。
    lower_blue = np.array([100, 80, 60], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)  # (H, W) の 0/255 マスク
    print("\n[3] 青マスク: mask shape =", mask_blue.shape,
          " 白画素数 =", int((mask_blue == 255).sum()))
    cv2.imwrite(str(out / "01_mask_blue.png"), mask_blue)

    # === 4. 赤は 0/179 をまたぐので2区間を合成する ========================
    # 赤の色相は 0 付近にあり、色相環が一周する 179 側にもはみ出す。
    # 片側だけの区間では取りこぼすため、[0,10] と [170,179] の2マスクを OR で合成する。
    mask_red1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 80, 60), (179, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    print("\n[4] 赤マスク（2区間を bitwise_or で合成）: 白画素数 =",
          int((mask_red == 255).sum()))
    cv2.imwrite(str(out / "01_mask_red.png"), mask_red)

    # === 5. マスクで元画像から色領域だけを残す ============================
    # bitwise_and の mask 引数で、白(255)の画素だけ元の色を残し、他は黒にする。
    blue_only = cv2.bitwise_and(bgr, bgr, mask=mask_blue)
    cv2.imwrite(str(out / "01_blue_only.png"), blue_only)

    # === 6. 完成物 extract_color を使う（赤の2区間処理込み） ==============
    # preprocess.py の再利用関数。色名を渡すだけで、赤の両端またぎや軽いノイズ除去まで面倒を見る。
    print("\n[6] preprocess.extract_color で各色を抽出")
    for color in ("red", "yellow", "green", "blue"):
        m = extract_color(bgr, color)
        cv2.imwrite(str(out / f"01_extract_{color}.png"), m)
        print(f"  {color:6s}: 白画素数 = {int((m == 255).sum()):6d}")

    # === 7. split / merge でチャンネルを分解・再合成 ======================
    # BGR を3枚の (H, W) に分解。順序は b, g, r。merge で元に戻せる。
    b, g, r = cv2.split(bgr)
    merged = cv2.merge([b, g, r])
    print("\n[7] split/merge: b/g/r 各 shape =", b.shape,
          " merge 復元一致 =", bool(np.array_equal(bgr, merged)))

    print("\n完了。outputs/03_image_transforms/ を確認してください。")


if __name__ == "__main__":
    main()
