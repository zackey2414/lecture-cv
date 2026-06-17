"""章末ミニプロジェクト: 手書き絵文字検索アプリ（エンドツーエンドの完成形）。

この回で学んだ要素を1本に統合する総合課題の「完成形」。

統合する流れ（SBIR の全部入り）:
  1. 絵文字コレクション（data→フォント→合成）→ グレースケール化 → CLIP 埋め込み
  2. L2 正規化 → IndexFlatIP を IndexIDMap2 で包んで add → .faiss/.json をセット永続化
  3. 手書きスケッチ入力（display があれば Tkinter キャンバス、無ければ合成スケッチ）
  4. スケッチ前処理（白背景・黒線・正方化）→ 同じ CLIP で埋め込み → 検索
  5. 類似 上位 N 件の絵文字とスコアをパネル画像 + JSON で出力

設計メモ:
  - digit 始まりの 01〜04 は import 不可なので、共通処理は emoji_lab から借用し、
    手書き入力だけ本ファイルに最小実装する（フォールバック込み）。
  - すべて CPU・数十秒以内に完走。CLIP を取得できない環境では画素記述子へ自動
    フォールバックし、display 無しでは合成スケッチに切り替えて exit 0 で完走する。
  - 出力は lectures/45_sketch_emoji_search/outputs/ に保存（matplotlib=Agg, cv2.imshow は呼ばない）。

実行:
  uv run python lectures/45_sketch_emoji_search/mini_project.py
ローカルに display があれば手書きウィンドウが開く。Docker/CI では合成スケッチで完走。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from emoji_lab import (  # noqa: E402
    EmojiEmbedder,
    build_emoji_collection,
    build_index,
    data_dir,
    make_synthetic_sketch,
    output_dir,
    preprocess_sketch,
    save_index,
    save_result_panel,
    search_topn,
    to_grayscale_rgb,
)

ID_BASE = 100  # 01 と同じ ID 起点
TOP_N = 5
DRAW_SIZE = 320
PEN_WIDTH = 8


# =====================================================================
# 手書き入力（Tkinter、display 無しは合成スケッチへフォールバック）
# =====================================================================


def _capture_tk_sketch():
    """Tkinter キャンバスで手書き入力を受け取り PIL.Image を返す（不可なら例外）。

    02_sketch_input_tk.py の最小版。Canvas と PIL に同時に線を引くのがポイント。
    """
    import tkinter as tk

    from PIL import Image, ImageDraw

    root = tk.Tk()  # display 無しならここで TclError
    root.title("Mini project - sketch an emoji, press 's' to search")
    pil = Image.new("RGB", (DRAW_SIZE, DRAW_SIZE), (255, 255, 255))
    pdraw = ImageDraw.Draw(pil)
    canvas = tk.Canvas(root, width=DRAW_SIZE, height=DRAW_SIZE, bg="white", cursor="pencil")
    canvas.pack()
    state = {"last": None}

    def on_drag(e):
        last = state["last"]
        if last is not None:
            canvas.create_line(*last, e.x, e.y, fill="black", width=PEN_WIDTH,
                               capstyle=tk.ROUND, smooth=True)
            pdraw.line([last[0], last[1], e.x, e.y], fill=(0, 0, 0), width=PEN_WIDTH)
        state["last"] = (e.x, e.y)

    def on_release(_):
        state["last"] = None

    def finish(_=None):
        root.quit()

    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("s", finish)
    root.bind("<Return>", finish)
    root.protocol("WM_DELETE_WINDOW", finish)
    root.mainloop()
    root.destroy()
    return pil


def capture_sketch():
    """手書きスケッチを取得（Tk→失敗時は data/sketch.png→合成 の順）。返り値: (img, source)。"""
    from PIL import Image

    try:
        return _capture_tk_sketch(), "Tkinter キャンバスでの手書き入力"
    except Exception as e:  # noqa: BLE001
        print(f"[fallback] 手書きウィンドウを開けません（{type(e).__name__}）→ 代替入力を使用。")

    saved = data_dir() / "sketch.png"
    if saved.exists():  # 02 を実行済みならその保存物を使う
        return Image.open(saved).convert("RGB"), f"{saved}（02 の保存物）"
    return make_synthetic_sketch(), "合成スケッチ（display 無しのフォールバック）"


# =====================================================================
# アプリ本体
# =====================================================================


def build_and_save_index(out: pathlib.Path):
    """絵文字を索引化して永続化し、(index, labels, grays, ids, meta) を返す。"""
    emojis_color, labels, source = build_emoji_collection()
    grays = [to_grayscale_rgb(im) for im in emojis_color]
    embedder = EmojiEmbedder()
    feats = embedder.encode_images(grays)
    ids = np.arange(ID_BASE, ID_BASE + len(grays)).astype(np.int64)
    index = build_index(feats, ids)

    meta = {
        "module": "45_sketch_emoji_search",
        "model": embedder.backend,
        "dim": int(index.d),
        "source": source,
        "style": "gray",
        "items": [{"id": int(i), "label": lab} for i, lab in zip(ids, labels)],
    }
    save_index(index, meta, out / "mini_emoji_index.faiss", out / "mini_emoji_meta.json")
    print(f"[1] 索引構築: {source} / {len(grays)} 種  エンコーダ: {embedder.backend}  d={index.d}")
    return index, embedder, labels, grays, ids, meta


def main() -> None:
    import faiss

    out = output_dir()
    print(f"=== ミニプロジェクト: 手書き絵文字検索 (faiss {faiss.__version__}) ===")

    # --- 1. 索引（絵文字→グレースケール→CLIP→FAISS）を構築・永続化 ---
    index, embedder, labels, grays, ids, meta = build_and_save_index(out)
    id2label = {int(i): lab for i, lab in zip(ids, labels)}
    id2pos = {int(i): pos for pos, i in enumerate(ids)}

    # --- 2. スケッチ入力（手書き or 合成） ---
    raw_sketch, sk_source = capture_sketch()
    sketch = preprocess_sketch(raw_sketch)
    sketch.save(out / "mini_sketch.png")
    print(f"[2] スケッチ入力: {sk_source}")

    # --- 3. スケッチを埋め込み → 検索（上位 N 件） ---
    q_feat = embedder.encode_images([sketch])
    n = min(TOP_N, index.ntotal)
    scores, ret_ids = search_topn(index, q_feat, n)

    # --- 4. 結果の解決（ID→ラベル/サムネ、-1 ガード）+ 可視化 + JSON ---
    result_imgs, result_rows = [], []
    print(f"[3] 類似 上位 {n} 件:")
    for rank, (i, sc) in enumerate(zip(ret_ids, scores), start=1):
        i = int(i)
        if i < 0:  # 近傍が足りないと -1。参照前にガード。
            continue
        lab = id2label.get(i, "(missing)")
        pos = id2pos.get(i)
        result_imgs.append(grays[pos] if pos is not None else sketch)
        result_rows.append({"rank": rank, "id": i, "label": lab, "cosine": round(float(sc), 4)})
        print(f"    {rank}. id={i:3d}  cos={sc:+.3f}  {lab}")

    save_result_panel(
        sketch, result_imgs,
        [r["cosine"] for r in result_rows], [r["label"] for r in result_rows],
        out / "mini_project_result.png",
        title="Mini project: sketch -> emoji (top-N, left: query)",
    )

    report = {
        "faiss": faiss.__version__,
        "backend": meta["model"],
        "n_emoji": int(index.ntotal),
        "dim": int(index.d),
        "emoji_source": meta["source"],
        "sketch_source": sk_source,
        "top_n": result_rows,
    }
    report_path = out / "mini_project_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 結果パネル: mini_project_result.png / レポート: {report_path.name}")
    print("完了。lectures/45_sketch_emoji_search/outputs/ を確認してください。")


if __name__ == "__main__":
    main()
