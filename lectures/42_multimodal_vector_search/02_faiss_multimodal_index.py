"""02: 画像とテキストを1つの FAISS index に載せる — id↔メタデータ付き永続化。

学ぶこと:
  - 画像とテキストの埋め込みを「同じ」index に add する（CLIP で同一空間だから可能）。
  - faiss.normalize_L2（in-place）→ IndexFlatIP で コサイン類似度検索。
  - IndexIDMap + add_with_ids で「任意の int64 ID」を割り当て、別管理のメタdata
    （modality / color / shape / text）と対応づける実運用パターン。
  - write_index / read_index で index を保存・復元し、メタは JSON で必ずセット保存。
  - 規模が出たときの近似 index（IVFFlat / HNSWFlat）を Flat と比べて体感する。

このスクリプトは「エンジンの中身」を生 FAISS API で手を動かして理解する回。
03 以降は mm_helpers の MultimodalSearchEngine（=ここを packaging したもの）を使う。

実行:
  uv run python lectures/42_multimodal_vector_search/02_faiss_multimodal_index.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_helpers import (  # noqa: E402
    as_faiss_array,
    build_caption_set,
    build_collection,
    embed_images,
    embed_texts,
    ensure_output_dir,
    load_clip,
    pick_device,
)


def main() -> None:
    out = ensure_output_dir()
    device = pick_device()
    print(f"faiss {faiss.__version__} / device {device}")

    # === 1. 画像＋テキストを埋め込み、1つの行列にまとめる ================
    images, img_metas = build_collection(variants=4)
    captions, cap_metas = build_caption_set()
    model, processor = load_clip(device)
    img_emb = embed_images(model, processor, images, device)
    txt_emb = embed_texts(model, processor, captions, device)

    # 画像とテキストを縦に積む。メタも同じ並びで連結（id とメタの対応表になる）。
    embeddings = np.concatenate([img_emb, txt_emb], axis=0)
    metas = list(img_metas) + list(cap_metas)
    d = embeddings.shape[1]
    print(f"\n[1] 統合行列 {embeddings.shape}（画像{len(img_emb)} + テキスト{len(txt_emb)}）次元 d={d}")

    # === 2. 正規化（in-place の faiss.normalize_L2 を体験） ==============
    # ★normalize_L2 は元配列をその場で書き換える。後で生ベクトルが要るならコピーしてから。
    xb = as_faiss_array(embeddings.copy())  # 破壊されてもいいようにコピー
    faiss.normalize_L2(xb)  # これで各行 ‖x‖=1 → 内積=コサイン
    print(f"[2] normalize_L2 後のノルム（先頭3件）= {np.round(np.linalg.norm(xb[:3], axis=1), 3).tolist()}")

    # === 3. IndexIDMap + add_with_ids で「任意ID」を付ける ==============
    # Flat だけだと ID は 0..N-1 に固定。IndexIDMap で包むと好きな int64 を割り当てられる。
    # ここでは 1000 番台を画像、2000 番台をテキストに割り当ててみる（運用での採番例）。
    ids = np.array(
        [1000 + i for i in range(len(img_emb))] + [2000 + i for i in range(len(txt_emb))],
        dtype=np.int64,
    )
    index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    index.add_with_ids(xb, ids)
    print(f"[3] IndexIDMap(IndexFlatIP) に add_with_ids 完了 ntotal={index.ntotal}")

    # ID→メタの対応表（実運用ではこれを SQLite/JSON で別管理する）。
    id_to_meta = {int(i): m for i, m in zip(ids, metas)}

    # === 4. テキストクエリで画像を引く（クロスモーダル検索の生API版） ===
    query_text = "a photo of a blue square"
    q = embed_texts(model, processor, [query_text], device)
    q = as_faiss_array(q.copy())
    faiss.normalize_L2(q)  # クエリ側も必ず正規化（片方だけだと崩れる）
    scores, found = index.search(q, 5)
    print(f"\n[4] テキストクエリ \"{query_text}\" の上位5:")
    for s, i in zip(scores[0].tolist(), found[0].tolist()):
        if i == -1:  # 近傍不足のとき -1。ID参照前に必ずガード。
            continue
        m = id_to_meta[int(i)]
        print(f"    id={i} cos={s:.3f} modality={m['modality']:5s} {m['color']} {m['shape']}")

    # === 5. 永続化: index は .faiss、メタは .json でセット保存 ===========
    # ★write_index は index 本体のみ。ID→メタの対応表は別ファイルに必ず一緒に保存する。
    idx_path = out / "02_mm_index.faiss"
    meta_path = out / "02_mm_index.meta.json"
    faiss.write_index(index, str(idx_path))
    meta_path.write_text(
        json.dumps({str(k): v for k, v in id_to_meta.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[5] 保存: {idx_path.name} + {meta_path.name}")

    # 読み戻して件数が一致するか確認（再現性チェック）。
    reloaded = faiss.read_index(str(idx_path))
    reloaded_meta = {int(k): v for k, v in json.loads(meta_path.read_text(encoding="utf-8")).items()}
    print(f"    読込後 ntotal={reloaded.ntotal} / meta件数={len(reloaded_meta)}（保存前と一致）")

    # === 6. 規模が出たら近似 index（IVF / HNSW）も視野に ================
    # 今回は 60 本と小さいので Flat で十分。だが N が大きくなると Flat は遅くなる。
    # ここでは「同じデータに対し IVF / HNSW を作り、Flat と同じ結果が出るか」を確認する。
    _compare_indexes(d, xb, q, k=5)

    print("\n完了。lectures/42_multimodal_vector_search/outputs/ を確認してください。")


def _compare_indexes(d: int, xb: np.ndarray, q: np.ndarray, k: int) -> None:
    """同一データに Flat / IVFFlat / HNSWFlat を作り、上位kの一致を比べる（近似の入口）。"""
    metric = faiss.METRIC_INNER_PRODUCT

    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    _, gt = flat.search(q, k)  # 厳密 top-k（近似の正解）

    # IVF: 空間を nlist セルに分け nprobe 個だけ探索（train 必須）。小データなので nlist は控えめ。
    nlist = 8
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, metric)
    ivf.train(xb)  # ★train を忘れると add で例外
    ivf.add(xb)
    ivf.nprobe = 4  # nprobe=1 のままだと recall が落ちやすい
    _, ivf_ids = ivf.search(q, k)

    # HNSW: グラフを辿る ANN。train 不要。efSearch が速度-精度の主ダイヤル。
    hnsw = faiss.IndexHNSWFlat(d, 16, metric)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    hnsw.hnsw.efSearch = 32
    _, hnsw_ids = hnsw.search(q, k)

    same_ivf = set(gt[0].tolist()) == set(ivf_ids[0].tolist())
    same_hnsw = set(gt[0].tolist()) == set(hnsw_ids[0].tolist())
    print("\n[6] 近似 index の上位kが Flat(厳密)と一致するか:")
    print(f"    Flat top-k ID = {gt[0].tolist()}")
    print(f"    IVFFlat (nprobe=4) 一致={same_ivf}  ID={ivf_ids[0].tolist()}")
    print(f"    HNSWFlat (efSearch=32) 一致={same_hnsw}  ID={hnsw_ids[0].tolist()}")
    print("    → 小データでは一致しがち。大規模では nprobe/efSearch で精度を上げる（04 で定量化）。")


if __name__ == "__main__":
    main()
