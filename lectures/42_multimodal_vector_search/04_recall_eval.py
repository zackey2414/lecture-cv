"""04: マルチモーダル検索の定量評価 — Recall@k と mAP（テキスト→画像）。

学ぶこと:
  - 「クエリ→正解集合」を定義して Recall@k / mAP を測る正攻法。
    ここでは正解集合を「テキストの (色,形) と一致する画像」とする。
  - Recall@k = 上位k件に入った正解数 / 正解総数。
    mAP     = 順位を考慮した AP をクエリ平均（上位に正解を集められたか）。
  - k を振って Recall@k 曲線を描き、検索品質を1枚の図にする。
  - 結果は JSON と PNG に保存し、後から数値で再現・比較できるようにする。

評価設計の注意:
  - 正解は「人手のラベル（color/shape）」で決める。検索器の出力で正解を作らない。
  - 合成データなので数値は高めに出る。実データほど難しくなる点は README で補足。

実行:
  uv run python lectures/42_multimodal_vector_search/04_recall_eval.py
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
    mean_at_k,
    pick_device,
    recall_at_k,
)


def main() -> None:
    out = ensure_output_dir()
    device = pick_device()
    print(f"device {device}")

    # === 1. 画像をDBに、テキストをクエリにする ==========================
    images, img_metas = build_collection(variants=4)
    captions, cap_metas = build_caption_set()
    model, processor = load_clip(device)
    img_emb = embed_images(model, processor, images, device)
    txt_emb = embed_texts(model, processor, captions, device)

    # 画像だけをエンジンに入れる（テキスト→画像の評価なので、検索対象は画像のみ）。
    engine = MultimodalSearchEngine()
    img_ids = engine.add(img_emb, img_metas)
    print(f"\n[1] 画像DB {len(img_ids)} 件 / テキストクエリ {len(captions)} 本")

    # === 2. 各クエリの「正解集合」を作る（色×形 が一致する画像ID） ======
    # ★ground truth は人手ラベル（color/shape）で定義する。検索結果から作らない。
    relevant: list[set[int]] = []
    for cm in cap_metas:
        rel = {
            img_ids[k]
            for k, im in enumerate(img_metas)
            if im["color"] == cm["color"] and im["shape"] == cm["shape"]
        }
        relevant.append(rel)
    n_rel = [len(r) for r in relevant]
    print(f"[2] 各クエリの正解数（色×形 一致の画像枚数）= {n_rel[0]} 枚（全クエリ共通）")

    # === 3. 検索して Recall@k / mAP を計算 ==============================
    ks = [1, 2, 4, 8]
    max_k = max(ks)
    # 全クエリを一括検索（modality="image" は不要：DBは画像のみ）。
    results = engine.search(txt_emb, k=max_k)
    retrieved_ids: list[list[int]] = [[h["id"] for h in row] for row in results]

    report: dict = {"k_values": ks, "n_db_images": len(img_ids), "n_queries": len(captions)}
    recall_curve = {}
    for k in ks:
        per_q = [recall_at_k(rid, rel, k) for rid, rel in zip(retrieved_ids, relevant)]
        recall_curve[k] = mean_at_k(per_q)
        print(f"[3] mean Recall@{k} = {recall_curve[k]:.3f}")
    report["mean_recall_at_k"] = recall_curve

    # mAP は k を固定（ここでは max_k）して順位品質を見る。
    aps = [average_precision_at_k(rid, rel, max_k) for rid, rel in zip(retrieved_ids, relevant)]
    report["mAP@%d" % max_k] = mean_at_k(aps)
    print(f"[4] mAP@{max_k} = {report['mAP@%d' % max_k]:.3f}（順位を考慮した平均適合率）")

    # === 4. クエリ別の内訳も出す（どの色×形が難しいか） =================
    print("\n[5] クエリ別 Recall@%d:" % max_k)
    per_query_detail = []
    for cap, rid, rel in zip(captions, retrieved_ids, relevant):
        r = recall_at_k(rid, rel, max_k)
        per_query_detail.append({"query": cap, "recall": r})
        cs = cap.replace("a photo of a ", "")
        print(f"    Recall@{max_k}={r:.2f}  \"{cs}\"")
    report["per_query"] = per_query_detail

    # === 5. 保存（JSON + 曲線PNG） =====================================
    json_path = out / "04_recall_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[6] 数値を保存: {json_path.name}")
    _save_recall_curve(out, ks, [recall_curve[k] for k in ks], report["mAP@%d" % max_k], max_k)

    print("\n完了。outputs/42_multimodal_vector_search/ を確認してください。")


def _save_recall_curve(out: pathlib.Path, ks, recalls, mapk, max_k) -> None:
    """Recall@k 曲線を保存する（横:k, 縦:mean Recall）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, recalls, "o-", color="#2a8", label="mean Recall@k (text→image)")
    for k, r in zip(ks, recalls):
        ax.annotate(f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("k")
    ax.set_ylabel("mean Recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.set_title(f"Text→Image retrieval quality  (mAP@{max_k} = {mapk:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "04_recall_curve.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("    図を保存:", path.name)


if __name__ == "__main__":
    main()
