"""02: 最重要前提 BGR vs RGB と uint8 オーバーフロー。

学ぶこと:
  - OpenCV は BGR、Pillow / matplotlib / PyTorch は RGB。この食い違いが最頻出バグ。
  - cv2.cvtColor(img, cv2.COLOR_BGR2RGB) で並びを入れ替える正しい作法。
  - 「BGR配列をそのままRGBとして表示/保存」すると赤と青が入れ替わる様子を実際に見る。
  - uint8 の足し算: numpy の `+` は 256 で巻き戻る（オーバーフロー）。
    一方 cv2.add は 255 で頭打ちになる「飽和演算」。明るさ調整では飽和が正しい。

実行:
  uv run python lectures/01_image_basics/02_bgr_rgb_pitfall.py
結果は lectures/01_image_basics/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# headless 環境で確実に動くよう、matplotlib は描画バックエンドを Agg（画面不要）に固定する。
# import matplotlib より前に use() を呼ぶ必要がある点に注意。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir  # noqa: E402

# Pillow も並べて比較に使う。
from PIL import Image  # noqa: E402


def main() -> None:
    out = output_dir()
    bgr = get_sample_bgr()  # OpenCV 流儀の BGR 画像（左上が赤になるよう作ってある）

    # === 1. cv2.imwrite は BGR 前提なので「そのまま」で正しい色になる =======
    # imread↔imwrite は両方 BGR なので、OpenCV内で完結する限り色は崩れない。
    cv2.imwrite(str(out / "02_cv2_correct.png"), bgr)

    # === 2. BGR 配列を「RGBのつもり」で Pillow に渡すと赤青が入れ替わる ======
    # PIL.Image.fromarray は配列を RGB と解釈する。BGR をそのまま渡すのは典型的な誤り。
    wrong = Image.fromarray(bgr)              # ← BGR を RGB と誤解 → 色が入れ替わる
    wrong.save(out / "02_pil_wrong_swapped.png")

    # 正しくは BGR→RGB へ変換してから渡す。
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    correct = Image.fromarray(rgb)            # ← これで本来の色
    correct.save(out / "02_pil_correct.png")

    # 並びが実際に入れ替わっていることを数値でも確認する。
    # 左上(赤)の画素は BGR で [0,0,255]。RGB変換後は [255,0,0] になっているはず。
    print("[2] 左上画素の値で並びを確認")
    print("  BGR(OpenCV) [B,G,R]:", bgr[10, 20].tolist())
    print("  RGB変換後   [R,G,B]:", rgb[10, 20].tolist())

    # === 3. matplotlib で「BGRのまま」vs「RGB変換後」を並べて保存 ===========
    # plt.imshow / plt.imsave は RGB を期待する。BGRを渡すと色が化ける。
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].imshow(bgr)   # ← BGR を RGB と誤解させて表示（赤青が入れ替わる）
    axes[0].set_title("WRONG: imshow(bgr)")
    axes[0].axis("off")
    axes[1].imshow(rgb)   # ← 正しく変換してから表示
    axes[1].set_title("OK: imshow(cvtColor BGR2RGB)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "02_matplotlib_compare.png", dpi=120)
    plt.close(fig)  # メモリ解放。ループで大量に作るときは特に重要。
    print("[3] 比較図を保存: 02_matplotlib_compare.png")

    # === 4. uint8 オーバーフロー: numpy の + vs cv2.add ===================
    # 明るさを +100 したい。uint8 は 0〜255 しか表せないので、255 を超えた分の扱いが問題になる。
    add_value = np.full_like(bgr, 100)  # 全画素に 100 を足すための配列

    # (a) numpy の `+`: 256 を法とした剰余演算。例えば 200+100=300 → 300-256=44 と巻き戻る。
    #     明るくしたはずの所が逆に暗くなり、ノイズのような模様が出る。
    naive = bgr + add_value

    # (b) cv2.add: 飽和演算。255 を超えたら 255 で止まる。明るさ調整の意図に合う。
    saturated = cv2.add(bgr, add_value)

    cv2.imwrite(str(out / "02_overflow_numpy_plus.png"), naive)
    cv2.imwrite(str(out / "02_overflow_cv2_add.png"), saturated)

    # 数値で挙動を確認する。下端グラデーション帯の明るい画素を見る。
    sample = bgr[-1, -1]  # 右下端（グラデーションで明るい）。値は大きめ。
    print("\n[4] uint8 オーバーフロー（明るさ +100）")
    print("  元の画素        :", sample.tolist())
    print("  numpy +  結果   :", (sample + np.uint8(100)).tolist(), "← 255超で巻き戻る")
    print("  cv2.add 結果    :", cv2.add(sample.reshape(1, 1, 3),
                                         np.full((1, 1, 3), 100, np.uint8)).reshape(3).tolist(),
          "← 255で頭打ち（飽和）")

    print("\n完了。lectures/01_image_basics/outputs/ で色と明るさの違いを見比べてください。")


if __name__ == "__main__":
    main()
