"""章末ミニプロジェクト: 統一マルチモーダル検索エンジンの完成形。

このスクリプト1本で、本章の学びを end-to-end に統合する:

  STEP1  画像＋テキストを CLIP で埋め込み、1つの FAISS エンジンに載せる（構築）
  STEP2  エンジンを .faiss + .meta.json で永続化し、別オブジェクトとして読み戻す（永続化）
  STEP3  読み戻したエンジンで 4方向検索（画像→画像 / テキスト→画像 / テキスト→テキスト
         / クロスモーダル）を1つの API で実行（統一インターフェース）
  STEP4  テキスト→画像の Recall@k / mAP を測り、結果を JSON と図に保存（評価）

「テキストでも画像でも引ける検索エンジンを、作って・保存して・読み直して・評価する」
という実運用の最小一周を体験するのが目的。完成形なのでそのまま動く。

実行:
  uv run python lectures/42_multimodal_vector_search/mini_project.py
成果物は outputs/42_multimodal_vector_search/ に出力される。
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_helpers import (  # noqa: E402
    MultimodalSearchEngine,
    average_precision_at_k,
    build_caption_set,
    build_collection,
    embed_images,
    embed_texts,
    ensure_output_dir,
    load_clip,
    load_user_images_or_none,
    mean_at_k,
    pick_device,
    recall_at_k,
    save_query_results_figure,
)


def build_engine(model, processor, device) -> tuple[MultimodalSearchEngine, list, list, dict]:
    """画像＋テキストを埋め込み、エンジンを構築して返す（STEP1）。

    実画像が data/ にあればそれを使い、無ければ合成コレクション（色×形）を使う。
    返り値: (engine, images, img_metas, id2image)
    """
    user = load_user_images_or_none()
    if user is not None:
        images, img_metas = user
        print(f"  実画像を使用: {len(images)} 枚（data/ から）")
    else:
        images, img_metas = build_collection(variants=4)
        print(f"  合成コレクションを使用: {len(images)} 枚（色×形）")
    captions, cap_metas = build_caption_set()

    img_emb = embed_images(model, processor, images, device)
    txt_emb = embed_texts(model, processor, captions, device)

    engine = MultimodalSearchEngine()
    img_ids = engine.add(img_emb, img_metas)
    engine.add(txt_emb, cap_metas)
    id2image = {i: images[k] for k, i in enumerate(img_ids)}
    return engine, images, img_metas, id2image


def run_searches(engine: MultimodalSearchEngine, img_metas, img_emb_lookup, out, id2image) -> dict:
    """読み戻したエンジンで4方向検索を実行し、図に保存する（STEP3）。"""
    summary: dict = {}

    # --- テキスト→画像 ---
    q = "a photo of a blue circle"
    qv = img_emb_lookup["embed_texts"]([q])
    res = engine.search(qv, k=4, modality="image")[0]
    summary["text_to_image"] = {"query": q, "hits": [_brief(h) for h in res]}
    save_query_results_figure(
        q, None, [id2image[h["id"]] for h in res],
        [f"{h['meta']['color']} {h['meta']['shape']}\ncos={h['score']:.2f}" for h in res],
        out / "mini_text_to_image.png",
    )

    # --- 画像→画像（green square をクエリに）---
    qi = next(k for k, m in enumerate(img_metas) if m["color"] == "green" and m["shape"] == "square")
    qv_img = img_emb_lookup["img_emb"][qi : qi + 1]
    res_ii = engine.search(qv_img, k=5, modality="image")[0][1:]  # 先頭(自分)を除く
    summary["image_to_image"] = {"query_index": qi, "hits": [_brief(h) for h in res_ii]}

    # --- テキスト→テキスト ---
    res_tt = engine.search(qv, k=3, modality="text")[0]
    summary["text_to_text"] = {"query": q, "hits": [_brief(h) for h in res_tt]}

    # --- クロスモーダル（絞り込みなし）---
    res_mix = engine.search(qv, k=5, modality=None)[0]
    summary["crossmodal_mixed"] = {"query": q, "hits": [_brief(h) for h in res_mix]}
    return summary


def _brief(h: dict) -> dict:
    """検索ヒットを JSON 用の簡潔な dict にする。"""
    m = h["meta"]
    return {"id": h["id"], "score": round(h["score"], 3), "modality": m["modality"],
            "label": f"{m['color']} {m['shape']}"}


def main() -> None:
    out = ensure_output_dir()
    device = pick_device()
    print(f"=== ミニプロジェクト: 統一マルチモーダル検索エンジン (device={device}) ===")

    model, processor = load_clip(device)

    # ---- STEP1: 構築 ----
    print("\n[STEP1] エンジン構築（画像＋テキストを同一空間へ）")
    engine, images, img_metas, id2image = build_engine(model, processor, device)
    print(f"  ntotal = {engine.index.ntotal}")

    # 評価・画像クエリ用に、画像埋め込みとテキスト埋め込み関数を手元に保持。
    img_emb = embed_images(model, processor, images, device)
    captions, cap_metas = build_caption_set()
    txt_emb = embed_texts(model, processor, captions, device)
    img_emb_lookup = {
        "img_emb": img_emb,
        "embed_texts": lambda ts: embed_texts(model, processor, ts, device),
    }

    # ---- STEP2: 永続化して読み戻す ----
    print("\n[STEP2] 永続化 → 読み戻し（別オブジェクトで再現）")
    stem = out / "mini_engine"
    idx_path, meta_path = engine.save(stem)
    engine2 = MultimodalSearchEngine.load(stem)
    print(f"  保存: {idx_path.name} + {meta_path.name}")
    print(f"  読込後 ntotal = {engine2.index.ntotal} / meta {len(engine2.meta)} 件（一致を確認）")
    assert engine2.index.ntotal == engine.index.ntotal

    # ---- STEP3: 4方向検索（読み戻したエンジンで） ----
    print("\n[STEP3] 4方向検索（同じ engine.search で実現）")
    searches = run_searches(engine2, img_metas, img_emb_lookup, out, id2image)
    for mode, payload in searches.items():
        head = payload["hits"][0] if payload["hits"] else {}
        print(f"  {mode:18s} top1 = {head}")

    # ---- STEP4: 評価（テキスト→画像 Recall@k / mAP） ----
    print("\n[STEP4] 評価（テキスト→画像）")
    # 画像だけを対象にした検索結果から Recall を計算する。
    results = engine2.search(txt_emb, k=8, modality="image")
    retrieved = [[h["id"] for h in row] for row in results]
    relevant = []
    for cm in cap_metas:
        rel = {i for i, m in engine2.meta.items()
               if m["modality"] == "image" and m["color"] == cm["color"] and m["shape"] == cm["shape"]}
        relevant.append(rel)
    ks = [1, 2, 4, 8]
    recall = {k: mean_at_k([recall_at_k(r, rel, k) for r, rel in zip(retrieved, relevant)]) for k in ks}
    mapk = mean_at_k([average_precision_at_k(r, rel, 8) for r, rel in zip(retrieved, relevant)])
    for k in ks:
        print(f"  mean Recall@{k} = {recall[k]:.3f}")
    print(f"  mAP@8 = {mapk:.3f}")

    # ---- 成果物をまとめて保存 ----
    report = {
        "device": str(device),
        "model_id": engine.model_id,
        "ntotal": engine.index.ntotal,
        "n_images": len([m for m in engine.meta.values() if m["modality"] == "image"]),
        "n_texts": len([m for m in engine.meta.values() if m["modality"] == "text"]),
        "searches": searches,
        "evaluation": {"mean_recall_at_k": recall, "mAP@8": mapk},
    }
    report_path = out / "mini_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n成果物: {report_path.name} / mini_text_to_image.png / mini_engine.faiss(+meta.json)")
    print("完了。outputs/42_multimodal_vector_search/ を確認してください。")


if __name__ == "__main__":
    main()
