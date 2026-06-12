"""01: 絵文字コレクションを CLIP で埋め込み、FAISS 索引を構築・保存する。

学ぶこと（SBIR の「準備フェーズ」）:
  - 顔絵文字の集合を用意する（data/ → 絵文字フォント → 合成 の優先順で自動選択）。
  - 絵文字を「グレースケール化」してから埋め込む（なぜ？ スケッチ＝白地に黒線の線画と
    ドメインを近づけ、色という余計な手がかりを落とすため。03/04 で効果を体感する）。
  - CLIP の画像エンコーダ get_image_features で 1 枚ずつベクトル化（v5 は .pooler_output、
    しかも未正規化）。
  - L2 正規化 → IndexFlatIP（= コサイン類似度）を IndexIDMap2 で包み、任意 int64 ID を付与。
  - .faiss（ベクトル＋ID）と .json（ID→ラベル等のメタ）をセットで永続化する。

出力:
  outputs/45_sketch_emoji_search/emoji_index.faiss / emoji_meta.json
  outputs/45_sketch_emoji_search/01_emoji_gallery.png（色 vs グレースケールのギャラリー）

実行:
  uv run python lectures/45_sketch_emoji_search/01_build_emoji_index.py
初回は CLIP の重みダウンロードが走る（以降キャッシュ）。取得不可なら画素記述子に
自動フォールバックして完走する。
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from emoji_lab import (  # noqa: E402
    EmojiEmbedder,
    build_emoji_collection,
    build_index,
    output_dir,
    save_gallery,
    save_index,
    to_grayscale_rgb,
)


def main() -> None:
    out = output_dir()
    index_path = out / "emoji_index.faiss"
    meta_path = out / "emoji_meta.json"

    # === 1. 絵文字コレクションを用意 =====================================
    emojis_color, labels, source = build_emoji_collection()
    print(f"[1] 絵文字コレクション: {source} / {len(emojis_color)} 種")

    # === 2. グレースケール化（スケッチ＝線画とドメインを寄せる前処理） =====
    emojis_gray = [to_grayscale_rgb(im) for im in emojis_color]
    print("[2] 全絵文字をグレースケール化（色の手がかりを落として線・形へ寄せる）")

    # 色 vs グレースケールを見比べられるギャラリーを保存（学習者の直感づくり）。
    save_gallery(emojis_color, labels, out / "01_emoji_gallery_color.png",
                 suptitle="Emoji collection (color)")
    save_gallery(emojis_gray, labels, out / "01_emoji_gallery.png",
                 suptitle="Emoji collection (grayscale = indexed form)")

    # === 3. CLIP で埋め込み =============================================
    embedder = EmojiEmbedder()
    print(f"[3] エンコーダ: {embedder.backend}  (dim={embedder.dim})")
    feats = embedder.encode_images(emojis_gray)  # (N, dim) 未正規化
    print("    生の埋め込み:", feats.shape, "ノルム例:", round(float(np.linalg.norm(feats[0])), 2),
          "(未正規化なので 1 ではない)")

    # === 4. 正規化 → IndexFlatIP を IndexIDMap2 で包んで add ============
    # ID は 0..N-1 でもよいが、連番と区別できることを示すため 100 番台にする。
    ids = np.arange(100, 100 + len(emojis_gray)).astype(np.int64)
    index = build_index(feats, ids)  # 内部で L2 正規化してから add_with_ids
    print(f"[4] FAISS 索引を構築: ntotal={index.ntotal}  d={index.d}（コサイン= 正規化+IP）")

    # === 5. 永続化（.faiss と .json をセットで） =========================
    meta = {
        "module": "45_sketch_emoji_search",
        "model": embedder.backend,
        "dim": int(index.d),
        "source": source,
        "style": "gray",  # 既定の索引はグレースケール絵文字
        "items": [{"id": int(i), "label": lab} for i, lab in zip(ids, labels)],
    }
    save_index(index, meta, index_path, meta_path)
    size_kb = index_path.stat().st_size / 1024
    print(f"[5] 永続化: {index_path.name} ({size_kb:.1f} KB) + {meta_path.name}")
    print("    .faiss（ベクトル＋ID）と .json（ID→ラベル）は『セットで1つの検索DB』。")

    print("\n完了。次は 02 で手書きスケッチ、03 で検索を試してください。")
    print("outputs/45_sketch_emoji_search/ を確認してください。")


if __name__ == "__main__":
    main()
