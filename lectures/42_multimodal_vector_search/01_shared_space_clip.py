"""01: 共有埋め込み空間 — CLIP で画像とテキストを「同じ512次元」へ写す。

学ぶこと:
  - CLIP は画像エンコーダとテキストエンコーダを対照学習で訓練し、
    「対応する画像と文」が近く、「無関係な対」が遠くなる共通空間を作る。
  - get_image_features / get_text_features で同じ次元(512)のベクトルが得られる。
    → だからテキスト埋め込みで画像を引ける（クロスモーダル検索の土台）。
  - 類似度は「L2 正規化してから内積」= コサイン類似度で測る。
  - transformers v5 では get_*_features の戻り値は .pooler_output に入る（重要）。

実行:
  uv run python lectures/42_multimodal_vector_search/01_shared_space_clip.py
結果は outputs/42_multimodal_vector_search/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_helpers import (  # noqa: E402
    build_caption_set,
    build_collection,
    embed_images,
    embed_texts,
    ensure_output_dir,
    l2_normalize,
    load_clip,
    pick_device,
)


def main() -> None:
    out = ensure_output_dir()
    device = pick_device()
    print(f"device = {device}")

    # === 1. 合成コレクション（色 × 形）とテキストを用意 ===================
    # 画像は 4色 × 3形 × 4枚 = 48枚。テキストは 12本（各 色×形 の説明文）。
    images, img_metas = build_collection(variants=4)
    captions, _cap_metas = build_caption_set()
    print(f"\n[1] 画像 {len(images)} 枚 / テキスト {len(captions)} 本を生成")

    # === 2. CLIP で埋め込み（画像とテキストを同じ空間へ） ================
    # 初回のみ CLIP の重みを DL（以後はキャッシュ）。eval + inference_mode は helper 側。
    model, processor = load_clip(device)
    img_emb = embed_images(model, processor, images, device)  # (48, 512)
    txt_emb = embed_texts(model, processor, captions, device)  # (12, 512)
    print(f"[2] 画像埋め込み {img_emb.shape} / テキスト埋め込み {txt_emb.shape}")
    # ★ここが本質: 画像もテキストも同じ 512 次元。だから直接 内積を取れる。
    assert img_emb.shape[1] == txt_emb.shape[1], "画像とテキストの次元が違うと比較できない"
    print(f"    どちらも同じ次元 = {img_emb.shape[1]}（→ クロスモーダルに比較可能）")

    # === 3. 正規化前後でノルムを確認 ====================================
    # get_*_features の出力は未正規化（ノルム≠1）。コサインにするには正規化が要る。
    raw_norm = np.linalg.norm(img_emb[0])
    img_n = l2_normalize(img_emb)
    txt_n = l2_normalize(txt_emb)
    print(f"\n[3] 正規化前のノルム例 = {raw_norm:.3f} → 正規化後 = {np.linalg.norm(img_n[0]):.3f}")

    # === 4. テキスト×画像のコサイン類似度（クロスモーダル一致を確認） ===
    # 正規化済みベクトルの内積 = コサイン類似度。txt_n @ img_n.T で全組み合わせ。
    sim = txt_n @ img_n.T  # (12テキスト, 48画像)
    print("\n[4] 各テキストに最も近い画像（クロスモーダル最近傍）:")
    correct = 0
    for ti, cap in enumerate(captions):
        best = int(np.argmax(sim[ti]))
        m = img_metas[best]
        # 説明文の (色 形) と、引かれた画像の (色 形) が一致していれば正しく結びついている。
        cap_cs = cap.replace("a photo of a ", "")
        img_cs = f"{m['color']} {m['shape']}"
        ok = cap_cs == img_cs
        correct += ok
        mark = "OK" if ok else "  "
        print(f"  [{mark}] \"{cap}\" -> 画像 #{best} ({img_cs}) cos={sim[ti, best]:.3f}")
    print(f"\n    クロスモーダル top-1 一致: {correct}/{len(captions)}")

    # === 5. 同モダリティ内のコサインも見ておく（テキスト→テキスト） ======
    # 同じ空間なのでテキスト同士・画像同士のコサインも自然に測れる。
    tt = txt_n @ txt_n.T
    print("\n[5] テキスト→テキストのコサイン例（対角=自分自身=1.0）:")
    print(f"    \"{captions[0]}\" vs \"{captions[1]}\" = {tt[0, 1]:.3f}")
    print(f"    \"{captions[0]}\" vs \"{captions[0]}\" = {tt[0, 0]:.3f}（自分自身）")

    # === 6. 類似度行列をヒートマップに保存 ==============================
    _save_heatmap(out, sim, captions, img_metas)
    print("\n完了。outputs/42_multimodal_vector_search/ を確認してください。")


def _save_heatmap(out: pathlib.Path, sim: np.ndarray, captions, img_metas) -> None:
    """テキスト×画像 のコサイン類似度行列をヒートマップに保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(sim, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(captions)))
    ax.set_yticklabels([c.replace("a photo of a ", "") for c in captions], fontsize=8)
    ax.set_xlabel("image index (48 images = 4 colors x 3 shapes x 4 variants)")
    ax.set_ylabel("text query")
    ax.set_title("CLIP cosine similarity: text (rows) vs image (cols) — same 512-d space")
    fig.colorbar(im, ax=ax, fraction=0.025, label="cosine")
    fig.tight_layout()
    path = out / "01_text_image_similarity.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("    図を保存:", path.name)


if __name__ == "__main__":
    main()
