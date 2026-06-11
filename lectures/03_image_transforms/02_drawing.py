"""02: 図形・テキストの描画（検出結果の可視化の基礎）。

学ぶこと:
  - cv2.line / rectangle / circle / polylines / putText の使い方。
  - 座標は (x, y)、色は BGR タプル、thickness=-1 で塗りつぶし。
  - lineType=cv2.LINE_AA でアンチエイリアス（縁が滑らかになる）。
  - putText のフォント・スケール・基準点（左下原点）と、textSize での文字寸法取得。
  - 検出結果の定番表現「バウンディングボックス＋ラベル背景＋文字」を自分で組み立てる。

実行:
  uv run python lectures/03_image_transforms/02_drawing.py
結果は outputs/03_image_transforms/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import output_dir  # noqa: E402


def draw_label_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    text: str,
    color: tuple[int, int, int] = (0, 200, 0),
) -> None:
    """検出結果の定番可視化: 矩形＋「文字が読める」ラベル背景＋文字を描く。

    1つのことだけ（1つの枠＋ラベルを描く）を行う関数。img はその場で書き換える。
    ラベルは枠の上辺に「塗りつぶし帯」を置き、その上に白文字を載せると視認性が高い。
    """
    # 1) バウンディングボックス本体。座標は (x, y) の2点、color は BGR。
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

    # 2) 文字の寸法を測る。putText だけでは背景が描けないので textSize で幅・高さを得る。
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    # 3) 枠の左上に、文字がちょうど収まる塗りつぶし帯を描く（thickness=-1）。
    cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, thickness=-1)

    # 4) 帯の上に白文字を載せる。putText の座標は「文字の左下(ベースライン)」が原点。
    cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), font, scale,
                (255, 255, 255), thick, lineType=cv2.LINE_AA)


def main() -> None:
    out = output_dir()

    # 描画用キャンバス。黒地だと色が映えるので (H, W, 3) のゼロ配列から始める。
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)

    # === 1. 基本図形 ======================================================
    # すべて座標は (x, y)、色は BGR。numpy の img[y, x] とは引数順が逆な点に注意。
    cv2.line(canvas, (10, 10), (390, 10), (255, 0, 0), thickness=2)           # 青い水平線
    cv2.rectangle(canvas, (20, 30), (120, 110), (0, 255, 0), thickness=3)     # 緑の枠（中空）
    cv2.rectangle(canvas, (140, 30), (240, 110), (0, 255, 0), thickness=-1)   # 緑の塗りつぶし
    cv2.circle(canvas, (320, 70), 40, (0, 0, 255), thickness=-1)             # 赤い塗り円

    # 多角形。点列は (N, 1, 2) の int32 配列にして polylines に渡す。
    pts = np.array([[60, 160], [110, 200], [90, 260], [30, 260], [10, 200]], dtype=np.int32)
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    print("[1] 基本図形を描画（line/rectangle/circle/polylines）")

    # === 2. アンチエイリアスの有無を比べる ================================
    # 同じ斜線でも lineType=cv2.LINE_AA だと縁がギザギザせず滑らかになる。
    cv2.line(canvas, (160, 150), (260, 250), (255, 255, 255), thickness=1)                 # 既定（8連結）
    cv2.line(canvas, (180, 150), (280, 250), (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    print("[2] LINE_AA ありなしの斜線を描画（拡大すると縁の滑らかさが違う）")

    # === 3. テキスト描画 ==================================================
    # putText の座標は「文字の左下」。color は BGR、fontScale で大きさを決める。
    cv2.putText(canvas, "OpenCV draw", (150, 285), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (200, 200, 0), 2, lineType=cv2.LINE_AA)
    print("[3] putText でテキスト描画（基準点は文字の左下）")

    cv2.imwrite(str(out / "02_shapes.png"), canvas)

    # === 4. 検出結果の可視化（バウンディングボックス＋ラベル） ============
    # 実務で最頻出の図。ダミーの「検出結果」を draw_label_box で重ねる。
    scene = np.full((300, 400, 3), 40, dtype=np.uint8)
    detections = [
        (30, 40, 170, 180, "cat 0.92", (0, 200, 0)),
        (210, 90, 360, 250, "dog 0.78", (0, 140, 255)),
    ]
    for x1, y1, x2, y2, label, col in detections:
        draw_label_box(scene, x1, y1, x2, y2, label, col)
    cv2.imwrite(str(out / "02_detections.png"), scene)
    print("[4] バウンディングボックス＋ラベルの可視化を保存（検出可視化の定番）")

    print("\n完了。outputs/03_image_transforms/ を確認してください。")


if __name__ == "__main__":
    main()
