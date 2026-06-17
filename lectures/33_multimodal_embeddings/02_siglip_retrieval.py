"""02: SigLIP 埋め込みと画像-テキスト検索（FAISS への橋渡し）。

学ぶこと:
  - get_image_features / get_text_features で「同じ768次元」のベクトルを取り出す。
    → 画像とテキストが同じ空間にあるから、テキストで画像を、画像でテキストを引ける。
  - get_*_features の出力は『未正規化』。コサイン類似度にするには L2 正規化が必須
    （forward 内部の logits は正規化済みだが、get_*_features は生ベクトル）。
  - 正規化したベクトルを faiss.IndexFlatIP に載せると、内積 = コサインで最近傍検索できる。
    これが第17回/第42回の大規模ベクトル検索（FAISS）へそのまま繋がる。

実行:
  uv run python lectures/33_multimodal_embeddings/02_siglip_retrieval.py
結果は lectures/33_multimodal_embeddings/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from siglip_lab import (  # noqa: E402
    MMEncoder,
    build_collection,
    build_concept_captions,
    build_ip_index,
    ensure_output_dir,
    l2_normalize,
    load_user_images_or_none,
    save_retrieval_figure,
    search_index,
)


def main() -> None:
    out = ensure_output_dir()

    # === 1. コレクション（画像）と概念テキストを用意 =====================
    # data/ に実画像があればそちら優先。無ければ合成 36 枚（4色×3形×3枚）。
    user = load_user_images_or_none()
    if user is not None:
        images, img_metas = user
        print(f"[1] data/ の実画像 {len(images)} 枚を使用")
    else:
        images, img_metas = build_collection(variants=3)
        print(f"[1] 合成画像 {len(images)} 枚（4色 × 3形 × 3枚）を生成")
    captions, cap_metas = build_concept_captions("en")  # 12本の英語キャプション

    # === 2. SigLIP で埋め込み（画像・テキストとも同じ 768 次元） =========
    enc = MMEncoder("siglip")
    img_emb = enc.embed_images(images)  # (N, 768) 未正規化
    txt_emb = enc.embed_texts(captions)  # (12, 768) 未正規化
    print(f"[2] 画像埋め込み {img_emb.shape} / テキスト埋め込み {txt_emb.shape}")

    # === 3. 「未正規化」を体感 → L2 正規化の必要性 =======================
    raw_norm = float(np.linalg.norm(img_emb[0]))
    img_n = l2_normalize(img_emb)  # L2 正規化すると各行のノルムが 1 になる
    txt_n = l2_normalize(txt_emb)  # テキスト側も同様（コサイン用に揃える）
    print(f"[3] get_image_features のノルム例 = {raw_norm:.3f}（≠1）→ 正規化後 = "
          f"{float(np.linalg.norm(img_n[0])):.3f}（テキスト側ノルム = {float(np.linalg.norm(txt_n[0])):.3f}）")

    # === 4. テキスト → 画像 検索（FAISS IndexFlatIP） ===================
    # 画像を index 化し、各テキストで上位3件を引く。正規化済みなので内積=コサイン。
    image_index = build_ip_index(img_emb)
    scores, ids = search_index(image_index, txt_emb, k=3)
    print("\n[4] テキスト → 画像 検索（上位3件の概念が一致すれば成功）:")
    hit = 0
    for qi, cap in enumerate(captions):
        top_concepts = [img_metas[int(j)]["concept"] for j in ids[qi]]
        ok = top_concepts[0] == cap_metas[qi]["concept"]
        hit += ok
        mark = "OK" if ok else "  "
        print(f'  [{mark}] "{cap}" -> {top_concepts}  (cos top1={scores[qi][0]:.3f})')
    print(f"    テキスト→画像 top-1 一致: {hit}/{len(captions)}")

    # === 5. 画像 → テキスト 検索（逆向きも同じ空間で可能） ===============
    text_index = build_ip_index(txt_emb)
    qi = 0  # 先頭画像をクエリにする
    s2, i2 = search_index(text_index, img_emb[qi : qi + 1], k=3)
    print(f"\n[5] 画像 → テキスト 検索（クエリ画像 = {img_metas[qi]['concept']}）:")
    for rank, (j, sc) in enumerate(zip(i2[0], s2[0]), start=1):
        print(f'    {rank}. "{captions[int(j)]}"  (cos={sc:.3f})')

    # === 6. 図に保存（テキストクエリ1本の上位ヒット画像） ===============
    q = 0
    _, top_ids = search_index(image_index, txt_emb[q : q + 1], k=4)
    result_imgs = [images[int(j)] for j in top_ids[0]]
    titles = [f"{img_metas[int(j)]['concept']}" for j in top_ids[0]]
    save_retrieval_figure(captions[q], None, result_imgs, titles, out / "02_text_to_image.png")
    print("\n[6] 図を保存:", (out / "02_text_to_image.png").name)
    print("\n[まとめ] get_*_features→L2正規化→IndexFlatIP で text↔image 検索が一直線。")
    print("         この index をディスクに保存すれば第42回の大規模マルチモーダル検索になる。")


if __name__ == "__main__":
    main()
