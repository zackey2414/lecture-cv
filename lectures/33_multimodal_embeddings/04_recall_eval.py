"""04: クロスモーダル検索の評価 — Recall@k と retrieval mAP（CLIP vs SigLIP）。

学ぶこと:
  - 共有埋め込み空間の「質」は、検索タスクの指標で定量化する:
      * Recall@k … クエリの正解のうち、上位 k 件に入った割合（“取りこぼし”の少なさ）。
      * mAP@k    … 順位まで考慮した適合率の平均（“上位に正解を集められたか”）。
  - 同じデータ・同じ正解定義（concept 一致）で CLIP と SigLIP を測り、横並びで比較する。
  - 評価は「テキスト → 画像」検索で行う。正解は『同じ概念(color shape)の画像すべて』。

実行:
  uv run python lectures/33_multimodal_embeddings/04_recall_eval.py
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
    build_collection,
    build_concept_captions,
    ensure_output_dir,
    evaluate_text_to_image,
)

K_VALUES = [1, 3, 5]


def main() -> None:
    out = ensure_output_dir()

    # === 1. データと正解定義 ============================================
    # 画像 36枚（4色×3形×3枚）。テキストクエリ 12本（各概念1本）。
    # 正解 = クエリ概念 "red circle" に一致する画像（= 3枚ずつ）。
    images, img_metas = build_collection(variants=3)
    captions, cap_metas = build_concept_captions("en")
    print(f"[1] 画像 {len(images)} 枚 / テキストクエリ {len(captions)} 本")
    print("    正解定義: クエリと同じ concept（color shape）を持つ画像すべて\n")

    # === 2. 各モデルで埋め込み → Recall@k / mAP@k ======================
    results: dict[str, dict[int, dict]] = {}
    for kind in ("clip", "siglip"):
        enc = MMEncoder(kind)
        img_emb = enc.embed_images(images)
        txt_emb = enc.embed_texts(captions)
        results[kind] = {}
        for k in K_VALUES:
            ev = evaluate_text_to_image(txt_emb, img_emb, cap_metas, img_metas, k=k)
            results[kind][k] = {"recall": ev["recall_at_k"], "map": ev["map_at_k"]}

    # === 3. 表で比較 ====================================================
    print("[2] テキスト→画像 検索の評価（数値が高いほど良い）\n")
    print("    Recall@k:")
    _print_table(results, "recall")
    print("\n    mAP@k:")
    _print_table(results, "map")

    # === 4. 図と JSON に保存 ============================================
    _save_bars(out, results)
    payload = {
        "task": "text->image retrieval",
        "n_images": len(images),
        "n_queries": len(captions),
        "k_values": K_VALUES,
        "metrics": results,
    }
    (out / "04_recall_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[3] 図を保存:", (out / "04_recall_eval.png").name)
    print("    指標を保存:", (out / "04_recall_metrics.json").name)

    print("\n[まとめ]")
    print("  - Recall@k は『取りこぼしの少なさ』、mAP@k は『上位に正解を集める力』を測る。")
    print("  - k を上げれば Recall は単調に上がる。実務では用途に応じた k を決めて比較する。")
    print("  - この評価関数は、音→画像など他のクロスモーダル検索にもそのまま使える（05/mini）。")


def _print_table(results: dict[str, dict[int, dict]], metric: str) -> None:
    """モデル×k の指標表を表示する。"""
    header = "      model   " + "".join(f"{('@' + str(k)):>9}" for k in K_VALUES)
    print(header)
    for kind in results:
        row = f"      {kind:<8}" + "".join(f"{results[kind][k][metric]:>9.3f}" for k in K_VALUES)
        print(row)


def _save_bars(out: pathlib.Path, results: dict[str, dict[int, dict]]) -> None:
    """Recall@k と mAP@k を CLIP vs SigLIP の並びで棒グラフ化して保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(K_VALUES))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ("recall", "map"), ("Recall@k", "mAP@k")):
        ax.bar(x - w / 2, [results["clip"][k][metric] for k in K_VALUES], w, label="CLIP", color="tab:blue")
        ax.bar(x + w / 2, [results["siglip"][k][metric] for k in K_VALUES], w, label="SigLIP", color="tab:green")
        ax.set_xticks(x)
        ax.set_xticklabels([f"@{k}" for k in K_VALUES])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} (text -> image)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_recall_eval.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
