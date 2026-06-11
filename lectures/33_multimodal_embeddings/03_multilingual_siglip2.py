"""03: SigLIP2 の多言語埋め込みで検索を強化する（vs 英語中心の SigLIP）。

学ぶこと:
  - SigLIP2 は多言語コーパスと Gemma 系トークナイザで訓練されており、日本語・仏語・
    西語など非英語のクエリでも「同じ概念の画像」をきちんと引ける。
  - 旧 SigLIP(base) は英語中心。非英語クエリでは検索精度が落ちる。これを「同じ画像
    コレクション・同じ概念」で Recall@1 を比較して定量的に体感する。
  - 使う API は CLIP/SigLIP と全く同じ（get_text_features / get_image_features）。
    モデルを差し替えるだけで多言語対応が手に入る、という移植性を理解する。

実行:
  uv run python lectures/33_multimodal_embeddings/03_multilingual_siglip2.py
結果は outputs/33_multimodal_embeddings/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from siglip_lab import (  # noqa: E402
    CAPTION_VOCAB,
    MMEncoder,
    build_collection,
    build_concept_captions,
    build_ip_index,
    ensure_output_dir,
    make_shape_image,
    search_index,
)

LANGS = ["en", "ja", "fr", "es"]
LANG_NAMES = {"en": "English", "ja": "日本語", "fr": "Français", "es": "Español"}


def main() -> None:
    out = ensure_output_dir()

    # === 1. 多言語ゼロショット分類: 各言語で「赤い円 vs 青い四角」の2択 ==
    # 画像=赤い円。各言語で正しいラベル(red circle)と紛らわしいラベル(blue square)を
    # 提示し、正しく赤い円を選べるか（sigmoid の大小）を見る。
    # ★注意: sigmoid の『絶対値』はモデル間で直接比較できない（学習時のバイアスが違う）。
    #   なので part1 は各モデル内の2択正解だけ見て、モデル比較は part2 の Recall で行う。
    image = make_shape_image("red", "circle")
    print("[1] 多言語ゼロショット分類（画像=赤い円, 各言語で red circle vs blue square の2択）\n")
    for kind in ("siglip", "siglip2"):
        enc = MMEncoder(kind)
        print(f"  --- {kind} ---")
        correct = 0
        for lang in LANGS:
            red_label = _caption(lang, "red", "circle")
            blue_label = _caption(lang, "blue", "square")
            logits = enc.logits_per_image([image], [red_label, blue_label])[0]
            p = 1.0 / (1.0 + np.exp(-logits))  # sigmoid（各ラベル独立）
            ok = p[0] > p[1]
            correct += ok
            mark = "OK" if ok else "  "
            print(f"    [{mark}] {LANG_NAMES[lang]:<9} sigmoid(red circle)={p[0]:.3f}  "
                  f"sigmoid(blue square)={p[1]:.3f}")
        print(f"    → 2択正解 {correct}/{len(LANGS)} 言語\n")

    # === 2. 多言語テキスト → 画像 検索の Recall@1 比較 ==================
    # 同じ画像コレクション（36枚）に対し、各言語の12概念クエリで Recall@1 を測る。
    images, img_metas = build_collection(variants=3)
    print("[2] 多言語クエリでの テキスト→画像 Recall@1（高いほど多言語に強い）")
    print(f"    画像コレクション = {len(images)}枚 / 各言語 12 クエリ\n")
    table: dict[str, dict[str, float]] = {}
    for kind in ("siglip", "siglip2"):
        enc = MMEncoder(kind)
        img_emb = enc.embed_images(images)
        index = build_ip_index(img_emb)
        table[kind] = {}
        for lang in LANGS:
            caps, cap_metas = build_concept_captions(lang)
            txt_emb = enc.embed_texts(caps)
            _s, ids = search_index(index, txt_emb, k=1)
            ok = sum(
                img_metas[int(ids[qi][0])]["concept"] == cap_metas[qi]["concept"]
                for qi in range(len(caps))
            )
            table[kind][lang] = ok / len(caps)

    # 表示（行 = モデル, 列 = 言語）
    header = "    model   " + "".join(f"{LANG_NAMES[lg]:>10}" for lg in LANGS)
    print(header)
    for kind in ("siglip", "siglip2"):
        row = f"    {kind:<8}" + "".join(f"{table[kind][lg]:>10.2f}" for lg in LANGS)
        print(row)

    # === 3. 図に保存（言語別 Recall@1 の棒グラフ） ======================
    _save_bars(out, table)
    print("\n[3] 図を保存:", (out / "03_multilingual_recall.png").name)

    print("\n[まとめ]")
    print("  - 英語では SigLIP も SigLIP2 も強い。差が出るのは日本語・仏語・西語などの非英語。")
    print("  - SigLIP2 はモデルを差し替えるだけ（API は同一）で多言語検索を強化できる。")


def _caption(lang: str, color: str, shape: str) -> str:
    voc = CAPTION_VOCAB[lang]
    return voc["template"].format(color=voc["color"][color], shape=voc["shape"][shape])


def _save_bars(out: pathlib.Path, table: dict[str, dict[str, float]]) -> None:
    """言語別 Recall@1 を SigLIP vs SigLIP2 の並びで棒グラフ化して保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(LANGS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [table["siglip"][lg] for lg in LANGS], w, label="SigLIP (en-centric)", color="tab:gray")
    ax.bar(x + w / 2, [table["siglip2"][lg] for lg in LANGS], w, label="SigLIP2 (multilingual)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(["en", "ja", "fr", "es"])  # ASCII で統一（フォント警告回避）
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall@1 (text -> image)")
    ax.set_title("Multilingual retrieval: SigLIP vs SigLIP2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_multilingual_recall.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
