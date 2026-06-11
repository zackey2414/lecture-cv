"""章末ミニプロジェクト: 多言語マルチモーダル検索エンジン（SigLIP2 + FAISS）。

統合課題:
  これまでの部品を1つにまとめ、「多言語クエリで画像を検索し、その精度を Recall@k と
  mAP@k で評価し、さらに音→画像のクロスモーダル検索（概念）まで体験できる」小さな
  検索エンジンを完成させる。これがこのモジュールの deliverable。

完成形がやること:
  1. 合成画像コレクションを SigLIP2 で埋め込み、FAISS(IndexFlatIP) に載せる。
  2. 英・日・仏・西の多言語テキストクエリで画像検索を行い、上位ヒットを図に保存。
  3. 言語別に Recall@k / mAP@k を測り、metrics.json に保存。
  4. （拡張）音→画像のクロスモーダル検索をトイ共有空間で実演（ImageBind の発想）。

実行:
  uv run python lectures/33_multimodal_embeddings/mini_project.py
結果は outputs/33_multimodal_embeddings/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from siglip_lab import (  # noqa: E402
    MMEncoder,
    average_precision_at_k,
    build_collection,
    build_concept_captions,
    build_ip_index,
    ensure_output_dir,
    load_user_images_or_none,
    make_trimodal_toy,
    mean,
    recall_at_k,
    save_retrieval_figure,
    search_index,
)

LANGS = ["en", "ja", "fr", "es"]
K_VALUES = [1, 3, 5]


class MultilingualImageSearch:
    """SigLIP2 + FAISS による多言語画像検索エンジン（統合の中核）。

    設計:
      - 画像を SigLIP2 で埋め込み、L2 正規化して IndexFlatIP に載せる（内積=コサイン）。
      - クエリ文（任意言語）も同じ SigLIP2 で埋め込めば、同一空間なので画像を引ける。
      - 画像メタ（concept）を Python 側に保持し、Recall@k / mAP@k の正解判定に使う。
    """

    def __init__(self, images: list, img_metas: list[dict]) -> None:
        self.images = images
        self.img_metas = img_metas
        self.enc = MMEncoder("siglip2")
        self.img_emb = self.enc.embed_images(images)  # (N, 768) 未正規化
        self.index = build_ip_index(self.img_emb)  # 内部で L2 正規化して add

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """テキストクエリ（任意言語）で上位 k 件の画像を引く。(scores, ids) を返す。"""
        q_emb = self.enc.embed_texts(queries)
        return search_index(self.index, q_emb, k=k)

    def evaluate(self, queries: list[str], query_metas: list[dict], k: int) -> dict:
        """クエリ群に対する Recall@k / mAP@k（正解 = 同じ concept の画像）。"""
        q_emb = self.enc.embed_texts(queries)
        _s, ids = search_index(self.index, q_emb, k=self.index.ntotal)
        recalls, aps = [], []
        for qi, qm in enumerate(query_metas):
            relevant = {j for j, m in enumerate(self.img_metas) if m["concept"] == qm["concept"]}
            ranked = [int(i) for i in ids[qi].tolist()]
            recalls.append(recall_at_k(ranked, relevant, k))
            aps.append(average_precision_at_k(ranked, relevant, k))
        return {"recall_at_k": mean(recalls), "map_at_k": mean(aps)}


def main() -> None:
    out = ensure_output_dir()

    # === 1. コレクション構築 → SigLIP2 でインデックス化 =================
    user = load_user_images_or_none()
    if user is not None:
        images, img_metas = user
        print(f"[1] data/ の実画像 {len(images)} 枚でエンジンを構築")
    else:
        images, img_metas = build_collection(variants=3)
        print(f"[1] 合成画像 {len(images)} 枚（4色×3形×3枚）でエンジンを構築")
    engine = MultilingualImageSearch(images, img_metas)
    print(f"    SigLIP2 で埋め込み → FAISS IndexFlatIP に {engine.index.ntotal} 件登録\n")

    # === 2. 多言語クエリで実検索（代表クエリの上位ヒットを図に保存） =====
    demo_queries = {
        "en": "a photo of a blue triangle",
        "ja": "青色の三角形の写真",
        "fr": "une photo d'un triangle bleu",
        "es": "una foto de un triángulo azul",
    }
    print("[2] 多言語クエリで『青い三角』を検索（言語が違っても同じ画像が出れば成功）:")
    for lang, q in demo_queries.items():
        _s, ids = engine.search([q], k=3)
        concepts = [img_metas[int(j)]["concept"] for j in ids[0]]
        print(f'    [{lang}] "{q}" -> {concepts}')
    # 図は日本語クエリの上位4件を保存（図のタイトルは ASCII にしてフォント警告を回避）。
    _s, ids = engine.search([demo_queries["ja"]], k=4)
    result_imgs = [images[int(j)] for j in ids[0]]
    titles = [img_metas[int(j)]["concept"] for j in ids[0]]
    save_retrieval_figure(
        "ja query -> blue triangle", None, result_imgs, titles, out / "mini_multilingual_query.png"
    )
    print("    図を保存:", (out / "mini_multilingual_query.png").name)

    # === 3. 言語別に Recall@k / mAP@k を評価 ============================
    print("\n[3] 言語別の テキスト→画像 検索精度（12概念クエリ）")
    metrics: dict[str, dict] = {}
    for lang in LANGS:
        caps, cap_metas = build_concept_captions(lang)
        metrics[lang] = {}
        for k in K_VALUES:
            ev = engine.evaluate(caps, cap_metas, k=k)
            metrics[lang][f"recall@{k}"] = round(ev["recall_at_k"], 3)
            metrics[lang][f"map@{k}"] = round(ev["map_at_k"], 3)
    header = "    lang  " + "".join(f"{('R@' + str(k)):>8}" for k in K_VALUES) + "".join(
        f"{('mAP@' + str(k)):>9}" for k in K_VALUES
    )
    print(header)
    for lang in LANGS:
        row = f"    {lang:<5}" + "".join(f"{metrics[lang][f'recall@{k}']:>8.3f}" for k in K_VALUES)
        row += "".join(f"{metrics[lang][f'map@{k}']:>9.3f}" for k in K_VALUES)
        print(row)

    # === 4. 拡張: 音 → 画像 クロスモーダル検索（ImageBind の発想） ======
    # SigLIP2 は画像×テキストだが、共有空間の発想は音声にも一般化できる。
    # ここではトイ共有空間で audio→image を実演（05 と同じ原理）。
    sound_concepts = ["dog", "bird", "bell", "drum", "rain", "engine"]
    toy = make_trimodal_toy(sound_concepts, dim=64, noise=0.18, seed=1)
    timg_index = build_ip_index(toy["image"])
    _s, aids = search_index(timg_index, toy["audio"], k=1)
    audio_recall = mean(
        [recall_at_k([int(j) for j in aids[qi]], {qi}, k=1) for qi in range(len(sound_concepts))]
    )
    print(f"\n[4] （拡張）音→画像 クロスモーダル検索 Recall@1 = {audio_recall:.3f}")
    print("    共有空間さえあれば、画像×テキストと同じ仕組みで音声クエリも扱える。")

    # === 5. 成果物を JSON にまとめて保存 ================================
    payload = {
        "engine": "SigLIP2 + FAISS IndexFlatIP",
        "n_images": engine.index.ntotal,
        "languages": LANGS,
        "k_values": K_VALUES,
        "text_to_image_metrics": metrics,
        "audio_to_image_recall@1_toy": round(audio_recall, 3),
    }
    (out / "mini_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[5] 成果物を保存:", (out / "mini_metrics.json").name)
    print("\n完了。多言語画像検索 + 音→画像クロスモーダルの統合デモが動きました。")


if __name__ == "__main__":
    main()
