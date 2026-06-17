"""02: end-to-end 画像検索 — IndexIDMap・SQLiteメタデータ・永続化・テキスト検索。

学ぶこと:
  - 埋め込み(CLIP) → L2正規化 → IndexIDMap で任意IDを付けて add_with_ids → search、
    という画像検索の一気通貫パイプライン。
  - FAISS は「ベクトルと連番ID」しか持たない。画像パス等のメタデータは別管理(SQLite)し、
    検索結果の ID から引く、という実運用の定石。
  - write_index / read_index で index を永続化。メタデータDBと「セットで」保存・復元する。
  - search の戻り値 I に紛れる -1（近傍が足りないとき）をガードする。
  - CLIP は画像とテキストを同じ空間に置くので「テキスト→画像」検索もできる（マルチモーダル）。

実行:
  uv run python lectures/17_faiss_image_search/02_idmap_persist_sqlite.py
初回は CLIP の重みダウンロードが走る（以降はキャッシュ）。ネット不通時は色記述子に
自動フォールバックして完走する（その場合テキスト検索のみスキップ）。
"""

from __future__ import annotations

import pathlib
import sqlite3
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_helpers import (  # noqa: E402
    Embedder,
    as_faiss_array,
    images_from_data_or_synthetic,
    l2_normalize,
    output_dir,
)


def build_metadata_db(db_path: pathlib.Path, ids: np.ndarray, labels: list[str]) -> None:
    """ID → ラベル(画像パスの代わり) の対応表を SQLite に保存する。

    実務では label の列に画像の絶対パスやサムネイル、撮影日時などを入れる。
    .faiss ファイル(ベクトル)と この .db ファイル(メタデータ)は「セットで1つの検索DB」。
    """
    if db_path.exists():
        db_path.unlink()  # デモのたびに作り直す
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)")
    con.executemany(
        "INSERT INTO items (id, label) VALUES (?, ?)",
        [(int(i), lab) for i, lab in zip(ids, labels)],
    )
    con.commit()
    con.close()


def lookup_labels(db_path: pathlib.Path, ids) -> list[str]:
    """検索で得た ID 列 → ラベル列。-1（無効ID）は '(none)' に変換してガードする。"""
    con = sqlite3.connect(db_path)
    out = []
    for i in ids:
        i = int(i)
        if i < 0:  # ← -1 ガード。そのままDB参照するとクラッシュ/誤参照する
            out.append("(none)")
            continue
        row = con.execute("SELECT label FROM items WHERE id = ?", (i,)).fetchone()
        out.append(row[0] if row else "(missing)")
    con.close()
    return out


def main() -> None:
    out = output_dir()
    index_path = out / "image_index.faiss"
    db_path = out / "image_meta.db"

    # === 1. 画像を用意してエンコード =====================================
    images, labels, source = images_from_data_or_synthetic(repeat=2)
    print(f"[1] 検索対象: {source} / {len(images)} 枚")
    embedder = Embedder()
    print(f"    エンコーダ: {embedder.backend}  (dim={embedder.dim}, text検索={embedder.supports_text})")

    feats = embedder.encode_images(images)  # (N, dim) 未正規化
    print("    生の埋め込み:", feats.shape, "ノルム例:", round(float(np.linalg.norm(feats[0])), 2))

    # === 2. 正規化 → IndexIDMap で任意IDを付与して add ===================
    # コサイン類似度にするため L2 正規化してから IP インデックスに入れる。
    db_vecs = l2_normalize(feats)
    d = db_vecs.shape[1]

    # IndexFlatIP を IndexIDMap2 で包むと、連番ではなく自分で決めた int64 ID を割り当てられる。
    # （IndexIDMap2 は ID からベクトルを復元する reconstruct も可能な上位版）
    base = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap2(base)

    # ここでは 1000 番台の ID を振る（連番 0..N-1 と区別できることを示すため）。
    ids = np.arange(1000, 1000 + len(images)).astype(np.int64)
    index.add_with_ids(db_vecs, ids)
    print(f"\n[2] IndexIDMap2 に add_with_ids 完了  ntotal={index.ntotal}  次元 d={d}")

    # メタデータ(ID→ラベル)を SQLite に保存。
    build_metadata_db(db_path, ids, labels)
    print(f"    メタDBを保存: {db_path.name}")

    # === 3. 永続化（write_index）→ 別オブジェクトで read_index ===========
    faiss.write_index(index, str(index_path))
    size_kb = index_path.stat().st_size / 1024
    print(f"\n[3] write_index: {index_path.name} ({size_kb:.1f} KB)")
    # まっさらな状態から読み戻す（=アプリ再起動を模す）。メタDBも同じパスから読む。
    index = faiss.read_index(str(index_path))
    print(f"    read_index 後 ntotal={index.ntotal} ← 保存前と一致すれば永続化OK")

    # === 4. 画像クエリで検索（画像→画像） ===============================
    # 検索対象の中から1枚をクエリに選び、似た画像を引く（同じラベルが上位に来るはず）。
    query_idx = labels.index("赤い円") if "赤い円" in labels else 0
    q_img_feat = embedder.encode_images([images[query_idx]])
    q_img = l2_normalize(q_img_feat)
    k = 4
    D, I = index.search(as_faiss_array(q_img), k)  # noqa: E741
    names = lookup_labels(db_path, I[0])
    print(f"\n[4] 画像クエリ = 「{labels[query_idx]}」 で類似画像検索 (top-{k})")
    for rank, (i, score, name) in enumerate(zip(I[0], D[0], names), start=1):
        print(f"    {rank}. id={int(i)} cos={score:+.3f}  {name}")

    # === 5. テキストクエリで検索（テキスト→画像、CLIPのときのみ） =========
    text_results = None
    if embedder.supports_text:
        queries = ["a photo of a red circle", "a blue square", "a green triangle"]
        t_feats = embedder.encode_texts(queries)
        t_norm = l2_normalize(t_feats)
        Dt, It = index.search(as_faiss_array(t_norm), 3)
        print("\n[5] テキストクエリ → 画像検索（CLIP 共有空間）")
        text_results = {}
        for q, row, drow in zip(queries, It, Dt):
            hit_labels = lookup_labels(db_path, row)
            print(f"    '{q}'")
            for rank, (i, s, name) in enumerate(zip(row, drow, hit_labels), start=1):
                print(f"        {rank}. id={int(i)} cos={s:+.3f}  {name}")
            text_results[q] = hit_labels
    else:
        print("\n[5] テキスト検索はスキップ（色記述子フォールバックのため）。")

    # === 6. -1 ガードの実演 ==============================================
    # k が登録件数より大きいと、足りない分の ID は -1 で埋められる。
    big_k = index.ntotal + 3
    _, I_over = index.search(as_faiss_array(q_img), big_k)
    n_invalid = int((I_over[0] == -1).sum())
    print(f"\n[6] k={big_k} > ntotal={index.ntotal} で検索 → 末尾に -1 が {n_invalid} 個。")
    print("    -1 をそのままメタDB参照に使うとクラッシュ。lookup_labels で (none) に変換済み。")

    _save_query_figure(out, images, labels, query_idx, I[0], names)

    print("\n完了。lectures/17_faiss_image_search/outputs/ を確認してください。")


# 図のタイトルは matplotlib 既定フォントに日本語が無く文字化け警告が出るため、
# 合成ラベル(赤い円 等)は ASCII に置換してから描く（コンソール出力は日本語のまま）。
_ROMAJI = {
    "赤": "red",
    "緑": "green",
    "青": "blue",
    "黄": "yellow",
    "円": "circle",
    "四角": "square",
    "三角": "triangle",
    "い": " ",
}


def _ascii_label(text: str) -> str:
    for jp, en in _ROMAJI.items():
        text = text.replace(jp, en)
    # 置換し切れない非ASCII（実画像のファイル名など）はそのまま英数字部分を返す
    return text.encode("ascii", "ignore").decode().strip() or text


def _save_query_figure(out, images, labels, query_idx, result_ids, result_names) -> None:
    """クエリ画像と検索結果のサムネイルを1枚に並べて保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result_ids)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.0 * (n + 1), 2.4))
    axes[0].imshow(images[query_idx])
    axes[0].set_title(f"query\n{_ascii_label(labels[query_idx])}", fontsize=9)
    axes[0].axis("off")
    for ax, rid, name in zip(axes[1:], result_ids, result_names):
        local = int(rid) - 1000  # ID は 1000 始まりで振った
        if 0 <= local < len(images):
            ax.imshow(images[local])
        ax.set_title(_ascii_label(name), fontsize=9)
        ax.axis("off")
    fig.suptitle("Image-to-image search (left: query)", fontsize=11)
    fig.tight_layout()
    path = out / "02_query_results.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("    図を保存:", path.name)


if __name__ == "__main__":
    main()
