"""実践ユースケース: 逆画像検索（reverse image search）の動く出発点。

これは何のツール？
  「手元の1枚（クエリ画像）を投げると、フォルダの中から“似ている／そっくり（重複）”
   な画像を FAISS で引いてくる」小さな逆画像検索ツールです。写真の重複整理、素材の
   再利用チェック、「この画像もう持ってたっけ？」の判定などに使えます。
  章末の mini_project.py（検索エンジンを組んで Recall@k / QPS まで“評価”する統合課題）
  とは別物で、こちらは「現実の小ツールとして1本で完結し、すぐ拡張して遊べる」ことを狙います。

実データ優先・合成フォールバック:
  - 自分の画像フォルダを  data/17_faiss_image_search/  に置けば、そのまま“実運用”になります
    （*.png / *.jpg などを置くだけ。ファイル名がラベルとして使われます）。
  - 画像が無ければ合成画像（色×形）で必ず完走（exit 0）します。まず動かして雰囲気を掴み、
    あとから自分の画像に差し替えてください。

仕組み（逆画像検索の最小構成）:
  1) フォルダの画像を CLIP で埋め込み → L2正規化 → IndexIDMap2(IndexFlatIP) に登録（=ギャラリー索引）
  2) 索引を write_index で保存し、read_index で読み直す（実ツールは「一度作って何度も引く」）
  3) クエリ画像を同じ手順でベクトル化 → search で上位 k 件（コサイン類似度）を取得
  4) コサインが閾値以上のヒットは「near-duplicate（ほぼ重複）」とフラグ付けして報告

拡張アイデア（このファイルをいじって練習）:
  - data/17_faiss_image_search/ に自分の写真を入れて差し替える（最初の一歩）。
  - 環境変数 QUERY_IMAGE=/path/to/query.jpg で“外部の”クエリ画像を指定できるようにしてある。
  - DUP_THRESHOLD を上げ下げして「重複」とみなす厳しさを調整する。
  - 索引を IndexIDMap2(IndexIVFFlat(...)) に替えて大量画像でも高速化（03_ivf_hnsw_pq.py 参照）。
  - メタデータを SQLite に移して撮影日時・パス・サムネを持たせる（02_idmap_persist_sqlite.py 参照）。
  - CLIP の共有空間を活かし、テキスト（"a red car"）でも引けるようにする（16章の応用）。

実行:
  uv run python lectures/17_faiss_image_search/use_case.py
  # 外部画像をクエリにする例:
  # QUERY_IMAGE=/path/to/photo.jpg uv run python lectures/17_faiss_image_search/use_case.py

CPU・headless で動き、結果は lectures/17_faiss_image_search/outputs/ に
use_case_reverse_search.png / use_case_report.json として保存されます。
初回は CLIP の重みダウンロードが走ります（ネット不通時は色記述子へ自動フォールバック）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import faiss
import numpy as np

# 同じフォルダの共通ヘルパ（道具箱）を借用する。番号付きスクリプト(01〜)は import 不可だが、
# search_helpers.py は普通のモジュールなので再利用してよい。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_helpers import (  # noqa: E402
    Embedder,
    as_faiss_array,
    images_from_data_or_synthetic,
    l2_normalize,
    output_dir,
)

# ===== ユーザが触って遊ぶ設定（ここを変えるのが最初の練習） =====================
TOP_K = 5  # 上位何件を「似ている候補」として出すか
DUP_THRESHOLD = 0.90  # コサイン類似度がこれ以上なら「near-duplicate（ほぼ重複）」とみなす
GALLERY_REPEAT = 2  # 合成ギャラリーの水増し回数（実画像を置けば無視される）


def _make_query_variant(img):
    """ギャラリー画像から「ほぼ重複だが少しだけ違う」クエリ画像を作る（自己完結）。

    実運用での想定: 「再保存・再圧縮・明るさ調整された“同じ画像”が、すでにフォルダに
    あるか？」を当てる場面。明度を少し変え、縮小→拡大の往復で軽く劣化させることで、
    ピクセルは完全一致しないが意味的にはほぼ同じ＝高コサインになる画像を作る。
    （QUERY_IMAGE 環境変数で外部画像を渡せば、この合成は使われない。）
    """
    from PIL import Image, ImageEnhance

    out = img.convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(1.12)  # 少し明るくする
    w, h = out.size
    # 縮小→拡大の往復で軽い再圧縮ノイズ相当の劣化を与える
    out = out.resize((max(8, w // 2), max(8, h // 2)), Image.BILINEAR)
    out = out.resize((w, h), Image.BILINEAR)
    return out


def build_gallery_index():
    """フォルダの画像を埋め込み → 正規化 → IndexIDMap2 に登録してギャラリー索引を作る。

    返り値: (index, images, labels, id_to_label, embedder, source)
    ID は 0..N-1 を振り、images[id] でそのままサムネイルを引けるようにしておく
    （任意IDやメタデータ別管理の発展は 02_idmap_persist_sqlite.py を参照）。
    """
    images, labels, source = images_from_data_or_synthetic(repeat=GALLERY_REPEAT)
    embedder = Embedder()
    feats = embedder.encode_images(images)  # (N, dim) 未正規化
    db_vecs = l2_normalize(feats)  # コサイン類似度にするため正規化（DB側）
    d = db_vecs.shape[1]

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))  # 内積=コサイン（正規化済みのため）
    ids = np.arange(len(images)).astype(np.int64)
    index.add_with_ids(db_vecs, ids)
    id_to_label = {int(i): lab for i, lab in zip(ids, labels)}
    return index, images, labels, id_to_label, embedder, source


def load_query(images, labels, embedder):
    """クエリ画像を1枚用意する。外部指定が無ければギャラリーから near-duplicate を合成。

    返り値: (query_image, query_origin[str])
    """
    env_path = os.environ.get("QUERY_IMAGE", "").strip()
    if env_path:
        p = pathlib.Path(env_path)
        if p.exists():
            from PIL import Image

            return Image.open(p).convert("RGB"), f"外部画像 {p.name}"
        print(f"  [warn] QUERY_IMAGE={env_path} が見つからないので合成クエリを使います")

    # 既定: 代表的な1枚（合成なら「赤い円」）の“ほぼ重複”を作ってクエリにする。
    query_idx = labels.index("赤い円") if "赤い円" in labels else 0
    variant = _make_query_variant(images[query_idx])
    return variant, f"合成した near-duplicate（元: 「{labels[query_idx]}」）"


def search_gallery(index, embedder, query_image, k):
    """クエリ画像でギャラリー索引を検索し、(ids, cosines) を返す（-1 はガード）。"""
    q_feat = embedder.encode_images([query_image])  # (1, dim) 未正規化
    q_norm = l2_normalize(q_feat)  # クエリ側も正規化（片側だけだと崩れる）
    k = min(k, index.ntotal)  # k > ntotal だと -1 が混じるので件数で頭打ち
    D, I = index.search(as_faiss_array(q_norm), k)  # noqa: E741 D=コサイン, I=ID
    ids, cosines = [], []
    for i, s in zip(I[0], D[0]):
        if int(i) < 0:  # -1 ガード（近傍不足の埋め草。そのまま使うと誤参照）
            continue
        ids.append(int(i))
        cosines.append(float(s))
    return ids, cosines


def main() -> None:
    out = output_dir()
    print(f"faiss {faiss.__version__} / 逆画像検索ツール")

    # === 1. ギャラリー索引を構築 ========================================
    index, images, labels, id_to_label, embedder, source = build_gallery_index()
    print(f"[1] ギャラリー: {source} / {len(images)} 枚")
    print(f"    エンコーダ: {embedder.backend} (dim={embedder.dim})")
    print(f"    index.ntotal={index.ntotal}（登録済み画像数）")

    # === 2. 永続化（実ツールは「一度作って何度も引く」） =================
    index_path = out / "use_case_index.faiss"
    meta_path = out / "use_case_meta.json"
    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps({str(k): v for k, v in id_to_label.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # まっさらな状態から読み戻す（=アプリ再起動を模す）。.faiss とメタ json はセットで保存。
    index = faiss.read_index(str(index_path))
    id_to_label = {int(k): v for k, v in json.loads(meta_path.read_text(encoding="utf-8")).items()}
    print(f"[2] 索引を保存→再読込: {index_path.name} + {meta_path.name} (ntotal={index.ntotal})")

    # === 3. クエリ画像で逆画像検索 ======================================
    query_image, query_origin = load_query(images, labels, embedder)
    ids, cosines = search_gallery(index, embedder, query_image, TOP_K)
    print(f"[3] クエリ: {query_origin}")

    # === 4. 結果を「重複/類似」に仕分けして報告 =========================
    results = []
    for rank, (i, cos) in enumerate(zip(ids, cosines), start=1):
        is_dup = cos >= DUP_THRESHOLD
        label = id_to_label.get(i, "(missing)")
        results.append({"rank": rank, "id": i, "label": label, "cosine": cos, "is_duplicate": is_dup})
        tag = "  <= near-duplicate!" if is_dup else ""
        print(f"    {rank}. id={i} cos={cos:+.3f}  {label}{tag}")

    n_dup = sum(r["is_duplicate"] for r in results)
    verdict = (
        f"フォルダ内に near-duplicate を {n_dup} 件発見（コサイン>= {DUP_THRESHOLD}）"
        if n_dup
        else "near-duplicate は見つからず（似た候補のみ）"
    )
    print(f"[4] 判定: {verdict}")

    # === 5. 図と JSON レポートを保存 ====================================
    _save_figure(out, query_image, query_origin, images, results)
    report = {
        "tool": "reverse_image_search",
        "source": source,
        "backend": embedder.backend,
        "dim": int(embedder.dim),
        "gallery_size": len(images),
        "query_origin": query_origin,
        "top_k": TOP_K,
        "duplicate_threshold": DUP_THRESHOLD,
        "num_duplicates_found": int(n_dup),
        "verdict": verdict,
        "results": results,
        "persisted": {"index_file": index_path.name, "meta_file": meta_path.name},
    }
    report_path = out / "use_case_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5] 保存: {report_path.name}, use_case_reverse_search.png")
    print("\n完了。lectures/17_faiss_image_search/outputs/ を確認してください。")


# --- 図のラベルは ASCII 化（matplotlib 既定フォントに日本語が無く文字化けするため） ----
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
    return text.encode("ascii", "ignore").decode().strip() or "item"


def _save_figure(out, query_image, query_origin, images, results) -> None:
    """クエリ画像と検索結果サムネを横一列に並べて保存（重複は [DUP] と印を付ける）。"""
    import matplotlib

    matplotlib.use("Agg")  # headless 用バックエンド（imshow は使わず savefig のみ）
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.0 * (n + 1), 2.6))
    if n + 1 == 1:  # 念のため（結果0件）軸を配列化
        axes = [axes]

    axes[0].imshow(query_image)
    axes[0].set_title("query", fontsize=10)
    axes[0].axis("off")

    for ax, r in zip(axes[1:], results):
        gid = r["id"]
        if 0 <= gid < len(images):  # ID は 0..N-1 で画像と対応
            ax.imshow(images[gid])
        mark = "[DUP] " if r["is_duplicate"] else ""
        ax.set_title(f"{mark}{_ascii_label(r['label'])}\ncos={r['cosine']:.2f}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Reverse image search (left: query, [DUP]=near-duplicate)", fontsize=11)
    fig.tight_layout()
    path = out / "use_case_reverse_search.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("    図を保存:", path.name)


if __name__ == "__main__":
    main()
