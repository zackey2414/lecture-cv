"""03: 統一インターフェースで4方向の検索 — 画像→画像 / テキスト→画像 / テキスト→テキスト / クロスモーダル。

学ぶこと:
  - 画像とテキストを1つの MultimodalSearchEngine に載せると、クエリのモダリティと
    結果のモダリティを自由に組み合わせて引ける（CLIP の共有空間のおかげ）。
  - 「同じ index・同じ search 呼び出し」で4方向すべてを実現する統一設計。
  - modality="image"/"text" 引数で結果側のモダリティを絞る（後フィルタ）。
  - クエリ自身が DB に入っている場合、先頭は自分自身になる点（先頭を捨てる扱い）。

ポイント: index には画像とテキストの両方が入っている。よって
  - テキストクエリを素で投げると「テキスト→テキスト」が上位に来やすい（同モダリティが近い）。
  - 「テキスト→画像」が欲しければ modality="image" で結果を画像に絞る。

実行:
  uv run python lectures/42_multimodal_vector_search/03_crossmodal_search.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_helpers import (  # noqa: E402
    MultimodalSearchEngine,
    build_caption_set,
    build_collection,
    embed_images,
    embed_texts,
    ensure_output_dir,
    load_clip,
    pick_device,
    save_query_results_figure,
)


def main() -> None:
    out = ensure_output_dir()
    device = pick_device()
    print(f"device {device}")

    # === 1. 画像＋テキストを1つのエンジンに載せる ========================
    images, img_metas = build_collection(variants=4)
    captions, cap_metas = build_caption_set()
    model, processor = load_clip(device)
    img_emb = embed_images(model, processor, images, device)
    txt_emb = embed_texts(model, processor, captions, device)

    engine = MultimodalSearchEngine()
    img_ids = engine.add(img_emb, img_metas)  # 画像を追加（id を採番して返る）
    txt_ids = engine.add(txt_emb, cap_metas)  # 続けてテキストを追加
    print(f"\n[1] エンジン構築: 画像{len(img_ids)} + テキスト{len(txt_ids)} = ntotal {engine.index.ntotal}")
    # id→画像 の対応（画像→画像検索の結果可視化に使う）
    id2image = {i: images[k] for k, i in enumerate(img_ids)}

    # === 2. テキスト→画像（クロスモーダル） =============================
    # テキストを埋め込み、modality="image" で結果を画像だけに絞る。
    query_text = "a photo of a yellow triangle"
    q_txt = embed_texts(model, processor, [query_text], device)
    res = engine.search(q_txt, k=5, modality="image")[0]
    print(f"\n[2] テキスト→画像  query=\"{query_text}\"")
    _print_hits(res)
    save_query_results_figure(
        query_text, None,
        [id2image[h["id"]] for h in res],
        [f"{h['meta']['color']} {h['meta']['shape']}\ncos={h['score']:.2f}" for h in res],
        out / "03_text_to_image.png",
    )

    # === 3. 画像→画像 ===================================================
    # ある画像をクエリにし、似た画像（同じ色・形）を引く。先頭は自分自身になる。
    q_idx = img_metas.index(next(m for m in img_metas if m["color"] == "red" and m["shape"] == "circle"))
    q_img_emb = img_emb[q_idx : q_idx + 1]
    res_ii = engine.search(q_img_emb, k=6, modality="image")[0]
    # 先頭(=自分自身)を1件落として「他の似た画像」を見る。
    res_ii = res_ii[1:]
    print(f"\n[3] 画像→画像  query=画像#{q_idx} ({img_metas[q_idx]['color']} {img_metas[q_idx]['shape']})")
    _print_hits(res_ii)
    save_query_results_figure(
        f"{img_metas[q_idx]['color']} {img_metas[q_idx]['shape']}", images[q_idx],
        [id2image[h["id"]] for h in res_ii],
        [f"{h['meta']['color']} {h['meta']['shape']}\ncos={h['score']:.2f}" for h in res_ii],
        out / "03_image_to_image.png",
    )

    # === 4. テキスト→テキスト ==========================================
    # 同じ index・同じ search を modality="text" で使うだけ。意味の近い説明文が並ぶ。
    res_tt = engine.search(q_txt, k=4, modality="text")[0]
    print(f"\n[4] テキスト→テキスト  query=\"{query_text}\"")
    _print_hits(res_tt)

    # === 5. クロスモーダル混在（絞り込みなし） ==========================
    # modality を指定しないと、画像もテキストも混ざって近い順に出る。
    # テキストクエリでは「同モダリティのテキスト」が上位を占めやすいことを観察する。
    res_mix = engine.search(q_txt, k=6, modality=None)[0]
    print(f"\n[5] 絞り込みなし（混在）  query=\"{query_text}\"")
    _print_hits(res_mix)
    print("    → テキスト同士が近いので text が上位に来やすい。"
          "用途に応じ modality で絞るのが実務の定石。")

    # === 6. 統一インターフェースまとめ =================================
    print("\n[6] 4方向すべて『同じ engine.search()』で実現:")
    print("    画像→画像     : search(画像emb, modality='image')")
    print("    テキスト→画像 : search(テキストemb, modality='image')")
    print("    テキスト→テキスト: search(テキストemb, modality='text')")
    print("    クロスモーダル: search(任意emb, modality=None)  ← 混在のまま近い順")

    print("\n完了。lectures/42_multimodal_vector_search/outputs/ を確認してください。")


def _print_hits(hits: list[dict]) -> None:
    """検索結果（id / score / メタ）を1行ずつ表示する。"""
    for h in hits:
        m = h["meta"]
        print(f"    id={h['id']:>3} cos={h['score']:.3f} [{m['modality']:5s}] {m['color']} {m['shape']}")


if __name__ == "__main__":
    main()
