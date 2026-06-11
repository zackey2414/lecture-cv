"""04: headless 環境での「表示」— cv2.imshow の罠と安全な代替。

学ぶこと:
  - cv2.imshow + cv2.waitKey は GUI バックエンド(GTK/Qt 等)が必要。
    Docker / SSH / CI などディスプレイの無い環境では「未実装」エラーや固まりが起きる。
  - 既定の表示手段は「ファイルに保存して後で見る」。cv2.imwrite と
    matplotlib(Agg バックエンド)の savefig を使えば画面が一切要らない。
  - どうしてもローカルGUIで見たい時だけ、環境変数 CV_SHOW=1 で cv2.imshow を任意に有効化する。
  - おまけ: 複数画像を1枚に並べるコンタクトシート（一覧画像）の作り方。

実行（保存のみ・画面なし）:
  uv run python lectures/01_image_basics/04_display_headless.py
ローカルにGUIがある場合のみ、ウィンドウ表示を試す:
  CV_SHOW=1 uv run python lectures/01_image_basics/04_display_headless.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

# 重要: matplotlib は import の前に Agg（画面不要の画像生成バックエンド）へ固定する。
# これで「DISPLAY が無い」環境でも savefig が確実に動く。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir  # noqa: E402


def save_with_cv2(bgr: np.ndarray, path: pathlib.Path) -> None:
    """方法A: cv2.imwrite で保存（BGRのまま渡す）。最も手軽で画面不要。"""
    cv2.imwrite(str(path), bgr)


def save_with_matplotlib(bgr: np.ndarray, path: pathlib.Path) -> None:
    """方法B: matplotlib で保存。imshow は RGB 前提なので必ず BGR→RGB に変換する。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(rgb)                 # 画面には出さず、図オブジェクトに描くだけ
    ax.set_title("saved by matplotlib (Agg)")
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)                 # 図を閉じてメモリ解放


def has_display() -> bool:
    """GUI ウィンドウを開ける見込みがあるか（X11 / Wayland のどちらかがあるか）。

    重要: ディスプレイが無い環境で cv2.imshow を呼ぶと、Qt バックエンドは
    例外ではなく「プロセスごと強制終了(abort)」することがある。これは Python の
    try/except では捕まえられない。だから「呼ぶ前に」開ける環境か確認するのが唯一安全な道。
    """
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def maybe_imshow(bgr: np.ndarray) -> None:
    """方法C: ローカルGUIがある時だけ cv2.imshow。CV_SHOW=1 かつ DISPLAY 有りのときのみ実行。

    二重のガードをかける:
      1. 明示的に CV_SHOW=1 を立てたか（既定では表示しない）。
      2. 実際にディスプレイがあるか（無ければ abort を避けるため呼ばない）。
    """
    if os.environ.get("CV_SHOW") != "1":
        print("[C] CV_SHOW=1 が無いので imshow はスキップ（headless 既定）。")
        return
    if not has_display():
        # ここで止めないと Qt の abort でプロセスごと落ちる。捕捉不能なので事前回避する。
        print("[C] CV_SHOW=1 だが DISPLAY が無いためスキップ（headless で abort を回避）。")
        return
    try:
        cv2.imshow("preview (press any key)", bgr)
        cv2.waitKey(0)            # キー入力まで待機。
        cv2.destroyAllWindows()
    except cv2.error as e:
        # GUI バックエンドのビルド状況によっては cv2.error が出ることもある。
        print("[C] imshow 失敗（GUI非対応の環境）:", e)


def make_contact_sheet(images_bgr: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """複数のBGR画像を格子状に並べた1枚の一覧画像を作る（headlessでの確認に便利）。

    すべて同じ高さに揃えてから np.hstack/np.vstack で連結する素朴な実装。
    過度に汎用化せず「同サイズ前提」で短く書く（必要十分）。
    """
    h = min(im.shape[0] for im in images_bgr)
    # 高さを h に揃える（アスペクト比は保つ）。
    resized = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images_bgr]
    # 幅も揃えないと hstack できないので、最小幅にクロップして簡単に揃える。
    w = min(im.shape[1] for im in resized)
    resized = [im[:, :w] for im in resized]

    rows = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i:i + cols]
        while len(row_imgs) < cols:                  # 端数は黒画像で埋める
            row_imgs.append(np.zeros_like(resized[0]))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)


def main() -> None:
    out = output_dir()
    bgr = get_sample_bgr()

    print("[A] cv2.imwrite で保存（画面不要・BGRのまま）")
    save_with_cv2(bgr, out / "04_saved_cv2.png")

    print("[B] matplotlib(Agg) で保存（BGR→RGB 変換を忘れない）")
    save_with_matplotlib(bgr, out / "04_saved_matplotlib.png")

    # おまけ: 色違いを並べたコンタクトシート。
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 並べるため3chに戻す
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)         # 参考: 別色空間も1枚に
    sheet = make_contact_sheet([bgr, gray_bgr, hsv, cv2.bitwise_not(bgr)], cols=2)
    cv2.imwrite(str(out / "04_contact_sheet.png"), sheet)
    print("    コンタクトシート保存: 04_contact_sheet.png")

    # 最後に、任意のGUI表示（既定ではスキップ）。
    maybe_imshow(bgr)

    print("\n完了。headless でも outputs/01_image_basics/ に結果が残ります。")


if __name__ == "__main__":
    main()
