"""03: スケッチを前処理・埋め込みし、FAISS で類似 絵文字 上位N件を検索する。

学ぶこと（SBIR の「検索フェーズ」）:
  - 01 で保存した索引（.faiss）とメタ（.json）を読み戻す。
  - クエリのスケッチを取得する（優先順: 02 が保存した data/.../sketch.png →
    無ければ合成スケッチ）。
  - スケッチを「絵文字と同じ手順」で前処理（白背景・黒線・正方化）→ 同じ CLIP で埋め込み
    → L2 正規化 → index.search で上位 N 件（スコア＝コサイン類似度）。
  - 結果（クエリ＋上位 N 件の絵文字とスコア）をパネル画像に可視化する。

ポイント:
  - クエリ側も DB 側と「同じエンコーダ・同じ正規化」を通すこと。片方だけ正規化すると崩れる。
  - 埋め込み次元が索引と食い違うと search が落ちる。次元を assert で先に確認する
    （CLIP が使えずフォールバックに替わった等で起こりうるので、その場合は索引を組み直す）。

出力:
  outputs/45_sketch_emoji_search/03_search_topn.png

実行:
  uv run python lectures/45_sketch_emoji_search/03_search_topn.py
事前に 01 を実行して索引を作っておくこと（無ければ自動で 01 相当を組み直す）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from emoji_lab import (  # noqa: E402
    EmojiEmbedder,
    build_emoji_collection,
    build_index,
    data_dir,
    load_index,
    make_synthetic_sketch,
    output_dir,
    preprocess_sketch,
    save_index,
    save_result_panel,
    search_topn,
    to_grayscale_rgb,
)

ID_BASE = 100  # 01 と同じ ID 起点（id = ID_BASE + 並び位置）
TOP_N = 5


def load_query_sketch():
    """検索クエリのスケッチを取得して返す（前処理前の生画像, ソース説明）。

    優先順: 02 が保存した data/45_sketch_emoji_search/sketch.png → 合成スケッチ。
    """
    from PIL import Image

    saved = data_dir() / "sketch.png"
    if saved.exists():
        return Image.open(saved).convert("RGB"), f"{saved}（02 の保存物）"
    return make_synthetic_sketch(), "合成スケッチ（02 未実行のフォールバック）"


def load_or_rebuild_index(embedder, emojis_gray, labels):
    """保存済み索引を読み、次元が合わなければ現在のエンコーダで組み直す。

    返り値: (index, id->label dict, id->position dict)。
    """
    out = output_dir()
    index_path, meta_path = out / "emoji_index.faiss", out / "emoji_meta.json"
    ids = np.arange(ID_BASE, ID_BASE + len(emojis_gray)).astype(np.int64)

    if index_path.exists() and meta_path.exists():
        index, meta = load_index(index_path, meta_path)
        if index.d == embedder.dim and index.ntotal == len(emojis_gray):
            id2label = {int(it["id"]): it["label"] for it in meta["items"]}
            id2pos = {int(it["id"]): pos for pos, it in enumerate(meta["items"])}
            print(f"[idx] 保存済み索引を読み込み: ntotal={index.ntotal} d={index.d}")
            return index, id2label, id2pos
        print(f"[idx] 索引の次元/件数が不一致（d={index.d} vs {embedder.dim}）→ 組み直します。")

    # 索引が無い / 不一致なら、その場で 01 相当を再構築（exit 0 を保つ）。
    feats = embedder.encode_images(emojis_gray)
    index = build_index(feats, ids)
    meta = {"items": [{"id": int(i), "label": lab} for i, lab in zip(ids, labels)]}
    save_index(index, meta, index_path, meta_path)
    id2label = {int(i): lab for i, lab in zip(ids, labels)}
    id2pos = {int(i): pos for pos, i in enumerate(ids)}
    print(f"[idx] 索引を再構築・保存: ntotal={index.ntotal} d={index.d}")
    return index, id2label, id2pos


def main() -> None:
    out = output_dir()

    # === 1. 検索対象（サムネ用）とエンコーダを用意 =======================
    emojis_color, labels, source = build_emoji_collection()
    emojis_gray = [to_grayscale_rgb(im) for im in emojis_color]
    embedder = EmojiEmbedder()
    print(f"[1] コレクション: {source} / {len(emojis_gray)} 種  エンコーダ: {embedder.backend}")

    index, id2label, id2pos = load_or_rebuild_index(embedder, emojis_gray, labels)

    # === 2. クエリのスケッチを取得して前処理 ============================
    raw_sketch, sk_source = load_query_sketch()
    sketch = preprocess_sketch(raw_sketch)
    print(f"[2] クエリ: {sk_source}")

    # === 3. 埋め込み → 正規化 → 検索（上位 N 件） =======================
    q_feat = embedder.encode_images([sketch])
    assert q_feat.shape[1] == index.d, "クエリと索引の次元が不一致（同じエンコーダを使うこと）"
    n = min(TOP_N, index.ntotal)
    scores, ids = search_topn(index, q_feat, n)
    print(f"[3] 上位 {n} 件（コサイン類似度）:")

    # === 4. 結果を解決（ID → ラベル / サムネイル） + 可視化 =============
    result_imgs, result_labels, result_scores = [], [], []
    for rank, (i, sc) in enumerate(zip(ids, scores), start=1):
        i = int(i)
        if i < 0:  # 近傍が足りないと -1。そのまま参照するとクラッシュするのでガード。
            continue
        lab = id2label.get(i, "(missing)")
        pos = id2pos.get(i)
        thumb = emojis_gray[pos] if (pos is not None and pos < len(emojis_gray)) else sketch
        result_imgs.append(thumb)
        result_labels.append(lab)
        result_scores.append(float(sc))
        print(f"    {rank}. id={i:3d}  cos={sc:+.3f}  {lab}")

    panel_path = out / "03_search_topn.png"
    save_result_panel(sketch, result_imgs, result_scores, result_labels, panel_path,
                      title="Sketch -> emoji search (top-N, left: query)")
    print(f"\n[done] 結果パネルを保存: {panel_path}")
    print("outputs/45_sketch_emoji_search/ を確認してください。")


if __name__ == "__main__":
    main()
