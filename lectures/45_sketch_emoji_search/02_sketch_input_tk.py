"""02: 手書きスケッチ入力（Tkinter キャンバス、display 無しは合成へフォールバック）。

学ぶこと:
  - 本講座の OpenCV は headless ビルドなので cv2.imshow / setMouseCallback は使えない。
    手書き入力は標準ライブラリ tkinter.Canvas のマウスドラッグで実装する（白背景に黒線）。
  - 描いた軌跡は Tk の画面だけでなく、PIL.Image にも同時に描いて保存する
    （Canvas のピクセルを直接取り出す移植性の高い方法が無いため、PIL に二重描きするのが定石）。
  - display が無い環境（Docker / CI / SSH 無 X11）では Tk のウィンドウ生成が TclError 等で
    失敗する。これを try/except で捕捉し、合成スケッチ（線画）に自動フォールバックして
    必ず exit 0 で終わる。tkinter の import 自体も try/except でガードする。

操作（ローカルに display があるとき）:
  - マウス左ドラッグで描画 / 「c」キーまたは Clear ボタンで消去
  - 「s」キー / Enter / Save ボタンで保存して終了
  - ウィンドウを閉じても保存される

出力:
  lectures/45_sketch_emoji_search/outputs/02_sketch.png
  data/45_sketch_emoji_search/sketch.png（03 が既定で読みに行く場所）

実行:
  uv run python lectures/45_sketch_emoji_search/02_sketch_input_tk.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from emoji_lab import (  # noqa: E402
    CANVAS,
    data_dir,
    make_synthetic_sketch,
    output_dir,
    preprocess_sketch,
)

# 描画キャンバスの一辺（Tk 上の見た目サイズ。保存時に CANVAS へ正規化する）。
DRAW_SIZE = 320
PEN_WIDTH = 8  # 黒線の太さ（指で描くくらいの太め）


def run_tk_canvas():
    """Tkinter のキャンバスで手書き入力を受け取り、PIL.Image(RGB) を返す。

    display が無い等でウィンドウを作れない場合は例外を送出する（呼び出し側で捕捉）。
    Canvas 上の線と PIL.ImageDraw への線を「同時に」引くのがポイント
    （Canvas のピクセル吸い出しは環境依存なので、保存用の真実は PIL 側に持つ）。
    """
    import tkinter as tk  # ← import 自体が失敗する環境もあるので呼び出し側で捕捉

    from PIL import Image, ImageDraw

    root = tk.Tk()  # display 無しならここで TclError
    root.title("Sketch an emoji - draw with mouse, press 's' to save")

    # 保存用の PIL 画像（白背景）。Canvas と同じ座標系で同時に描く。
    pil_img = Image.new("RGB", (DRAW_SIZE, DRAW_SIZE), (255, 255, 255))
    pil_draw = ImageDraw.Draw(pil_img)

    canvas = tk.Canvas(root, width=DRAW_SIZE, height=DRAW_SIZE, bg="white", cursor="pencil")
    canvas.pack()
    state = {"last": None, "saved": False}

    def on_press(event):
        state["last"] = (event.x, event.y)

    def on_drag(event):
        # 直前の点から現在の点へ線分を引く（Canvas と PIL の両方に）。
        x, y = event.x, event.y
        last = state["last"]
        if last is not None:
            canvas.create_line(last[0], last[1], x, y, fill="black",
                               width=PEN_WIDTH, capstyle=tk.ROUND, smooth=True)
            pil_draw.line([last[0], last[1], x, y], fill=(0, 0, 0), width=PEN_WIDTH)
        state["last"] = (x, y)

    def on_release(_event):
        state["last"] = None

    def clear(_event=None):
        canvas.delete("all")
        pil_draw.rectangle([0, 0, DRAW_SIZE, DRAW_SIZE], fill=(255, 255, 255))

    def save_and_quit(_event=None):
        state["saved"] = True
        root.quit()

    canvas.bind("<Button-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("c", clear)
    root.bind("s", save_and_quit)
    root.bind("<Return>", save_and_quit)
    root.protocol("WM_DELETE_WINDOW", save_and_quit)

    # 操作ボタン（キー操作が分からない人向け）。
    bar = tk.Frame(root)
    bar.pack(fill="x")
    tk.Button(bar, text="Clear (c)", command=clear).pack(side="left")
    tk.Button(bar, text="Save (s)", command=save_and_quit).pack(side="right")

    root.mainloop()
    root.destroy()
    return pil_img


def main() -> None:
    out = output_dir()
    sketch_path = out / "02_sketch.png"
    data_sketch = data_dir() / "sketch.png"
    data_sketch.parent.mkdir(parents=True, exist_ok=True)

    raw = None
    source = ""
    try:
        raw = run_tk_canvas()
        source = "Tkinter キャンバスでの手書き入力"
    except Exception as e:  # noqa: BLE001
        # display 無し / tkinter import 不可 / TclError などをすべて捕捉。
        print(f"[fallback] 手書きウィンドウを開けませんでした（{type(e).__name__}）。")
        print("           → 合成スケッチ（線画）を代わりに保存します（headless 完走）。")
        raw = make_synthetic_sketch()  # happy の線画
        source = "合成スケッチ（display 無しのフォールバック）"

    # CLIP に入れる前の標準形（白背景・黒線・正方化・グレースケール）に整えて保存。
    sketch = preprocess_sketch(raw)
    sketch.save(sketch_path)
    sketch.save(data_sketch)

    print(f"[done] 入力ソース: {source}")
    print(f"       保存: {sketch_path}")
    print(f"       保存: {data_sketch}（03 が既定で読み込む）")
    print(f"       サイズ: {sketch.size}（{CANVAS}x{CANVAS} 正方・グレースケール RGB）")
    print("\n次は 03_search_topn.py でこのスケッチから絵文字を検索します。")


if __name__ == "__main__":
    main()
