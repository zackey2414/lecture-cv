"""04: ドメインギャップを測る — 絵文字の様式を変えると Recall@N がどう変わるか。

問題意識:
  スケッチは「白地に黒線の線画」、絵文字は「色つきの塗り」。この見た目の差
  （ドメインギャップ）が埋め込みの距離を開かせ、検索を当たりにくくする…というのが
  古典 SBIR の出発点。そこで絵文字側をクエリの様式に寄せる（グレースケール化 /
  エッジ抽出 / 反転）と Recall がどう変わるかを、正解ラベルつきで定量比較する。

実験設定（正解がはっきりする合成ペア）:
  - 表情ごとに「合成絵文字（塗り）」と「合成スケッチ（線画）」を独立に描く。
    両者は同じ表情だが別レンダリングなので、本物のドメインギャップがある。
  - 絵文字に style ∈ {color, gray, edge, invert} を施して索引を作り、スケッチを
    クエリに上位 N 件を引いて「正しい表情の絵文字が上位N件に入るか」で Recall@N
    （= top-N ヒット率）を計算する。各表情は1個なので Recall@N＝ヒット率。

2 つのエンコーダで比べるのが肝:
  - 画素記述子（低レベル）: 文字どおり画素の濃淡を照合するので、様式に敏感で脆い。
    特に invert（白黒の極性反転）で Recall がほぼ 0 まで崩れる。グレースケールは
    「色という無関係な手がかりを落とすだけ」で安全だが、エッジ化は平面的な絵文字では
    二重輪郭などのノイズを生み、必ずしも当たりが良くならない。
  - CLIP（意味ベース）   : 「笑った顔」という高レベルの意味で照合するため、様式の
    影響が小さく、塗り・反転でも崩れにくい。事前学習済みエンコーダはドメインギャップを
    既に大きく橋渡ししている、という現代的な知見が数字で見える。
  ＝「ドメイン差を手作業で寄せる（古典 SBIR の発想）」よりも「強い共有埋め込みを使う」
    方が効く、という対比が本実験の主眼。グレースケール既定は色の混入を防ぐ安全策。

出力:
  lectures/45_sketch_emoji_search/outputs/04_domaingap_recall.png（エンコーダ×style の Recall）
  lectures/45_sketch_emoji_search/outputs/04_styles_example.png（同じ絵文字の4様式）

実行:
  uv run python lectures/45_sketch_emoji_search/04_eval_domaingap.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from emoji_lab import (  # noqa: E402
    EmojiEmbedder,
    build_index,
    build_synthetic_pairs,
    output_dir,
    preprocess_sketch,
    search_topn,
    style_transform,
)

STYLES = ["color", "gray", "edge", "invert"]
NS = [1, 3, 5]  # Recall@N を測る N


def evaluate_style(embedder, emojis_color, sketch_feats, labels, style: str) -> dict:
    """1つの style について Recall@N（=top-N ヒット率）を計算して返す。"""
    # 絵文字に style を施してから埋め込み → 索引化。
    styled = [style_transform(im, style) for im in emojis_color]
    e_feats = embedder.encode_images(styled)
    ids = np.arange(len(styled)).astype(np.int64)
    index = build_index(e_feats, ids)

    max_n = max(NS)
    hits = {n: 0 for n in NS}
    for qi in range(len(labels)):
        _, ret_ids = search_topn(index, sketch_feats[qi], max_n)
        ret_labels = [labels[int(r)] for r in ret_ids if int(r) >= 0]
        for n in NS:
            if labels[qi] in ret_labels[:n]:  # 正しい表情が上位 n 件に入ったか
                hits[n] += 1
    q = len(labels)
    return {f"recall@{n}": hits[n] / q for n in NS}


def evaluate_backend(name: str, embedder, emojis_color, sketches, labels) -> dict:
    """1つのエンコーダで全 style の Recall@N を測る。"""
    # クエリ（スケッチ）は様式に依らず固定。先に前処理・埋め込みしておく。
    sketch_inputs = [preprocess_sketch(s) for s in sketches]
    feats = embedder.encode_images(sketch_inputs)
    sketch_feats = [feats[i] for i in range(len(labels))]

    print(f"[{name}] style 別 Recall@N:")
    res = {}
    for style in STYLES:
        res[style] = evaluate_style(embedder, emojis_color, sketch_feats, labels, style)
        row = "  ".join(f"R@{n}={res[style][f'recall@{n}']:.2f}" for n in NS)
        print(f"    {style:7s} : {row}")
    best = max(STYLES, key=lambda s: res[s]["recall@1"])
    print(f"    → Recall@1 最良の様式: '{best}' ({res[best]['recall@1']:.2f})")
    return res


def main() -> None:
    out = output_dir()

    # === 1. 正解つきの合成ペア（絵文字 / スケッチ）を用意 ================
    emojis_color, sketches, labels = build_synthetic_pairs()
    print(f"[1] 合成ペア: {len(labels)} 表情で評価\n")

    # === 2. 2 つのエンコーダで評価（CLIP と 画素記述子ベースライン） =====
    main_emb = EmojiEmbedder()
    results: dict[str, dict] = {}
    results[main_emb.backend] = evaluate_backend(
        main_emb.backend, main_emb, emojis_color, sketches, labels)

    # CLIP が使えたときだけ、対比として低レベルの画素記述子でも測る。
    if main_emb.backend.startswith("CLIP"):
        print()
        px = EmojiEmbedder(force_fallback=True)
        results["pixel-descriptor"] = evaluate_backend(
            "pixel-descriptor", px, emojis_color, sketches, labels)

    # === 3. 結論の言語化（データ駆動で述べる） =========================
    # 「様式による Recall の振れ幅」が小さいほど、ドメイン差に頑健と言える。
    print("\n[3] 読み取り方（様式による振れ幅が小さいほどドメイン差に強い）:")
    spreads = {}
    for name in results:
        r1 = {s: results[name][s]["recall@1"] for s in STYLES}
        best, worst = max(r1, key=r1.get), min(r1, key=r1.get)
        spreads[name] = r1[best] - r1[worst]
        print(f"    - {name}: 最良 '{best}'({r1[best]:.2f}) / 最悪 '{worst}'({r1[worst]:.2f}) "
              f"/ 振れ幅={spreads[name]:.2f}")
    if "pixel-descriptor" in results:
        robust = min(spreads, key=spreads.get)
        print(f"    → 振れ幅が最小なのは '{robust}'。CLIP は意味で照合するため様式に頑健で、")
        print("      グレースケールも安全（無関係な色を落とすだけ）。一方 invert は画素記述子を壊す")
        print("      （白黒の極性は低レベル照合では致命的）。エッジ化も平面的絵文字では必ずしも得でない。")

    # === 4. 可視化 =====================================================
    _save_recall_bars(out, results)
    _save_styles_example(out, emojis_color[0])
    print("\n完了。lectures/45_sketch_emoji_search/outputs/ を確認してください。")


# =====================================================================
# 図の保存（タイトル・凡例は ASCII）
# =====================================================================


def _save_recall_bars(out: pathlib.Path, results: dict) -> None:
    """エンコーダごとに subplot を分け、style × Recall@N のグループ棒グラフを描く。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backends = list(results.keys())
    fig, axes = plt.subplots(1, len(backends), figsize=(6.0 * len(backends), 4.5), squeeze=False)
    x = np.arange(len(STYLES))
    width = 0.25
    for ax, name in zip(axes[0], backends):
        for j, n in enumerate(NS):
            vals = [results[name][s][f"recall@{n}"] for s in STYLES]
            ax.bar(x + (j - 1) * width, vals, width, label=f"Recall@{n}")
        ax.set_xticks(x)
        ax.set_xticklabels(STYLES)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Recall (top-N hit rate)")
        ax.set_xlabel("emoji style before indexing")
        # タイトルは ASCII 化（CLIP(openai/...) はそのまま ASCII）。
        ax.set_title(name if name.isascii() else "encoder")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Domain-gap mitigation: sketch -> emoji recall by emoji style")
    fig.tight_layout()
    fig.savefig(out / "04_domaingap_recall.png", dpi=110)
    plt.close(fig)


def _save_styles_example(out: pathlib.Path, emoji_color) -> None:
    """同じ絵文字に4様式を施した見本（color/gray/edge/invert）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(STYLES), figsize=(2.0 * len(STYLES), 2.4))
    for ax, style in zip(axes, STYLES):
        ax.imshow(style_transform(emoji_color, style), cmap="gray")
        ax.set_title(style, fontsize=10)
        ax.axis("off")
    fig.suptitle("Same emoji in 4 styles (query is a line sketch)")
    fig.tight_layout()
    fig.savefig(out / "04_styles_example.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
