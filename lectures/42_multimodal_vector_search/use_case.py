"""use_case.py — 自分の写真ライブラリを『自然文』でも『見本画像』でも引くセマンティック検索ツール。

【これは何？ mini_project.py との違い】
  mini_project.py は「作る → 保存 → 読み戻す → Recall/mAP で評価」を一周する“ベンチ寄り”の
  統合課題。こちらの use_case.py は、より現実の小ツールに寄せた『動く出発点』です。
  目的は一つ —— 手元の写真フォルダを **一度だけ** CLIP+FAISS でインデックス化してディスクに
  キャッシュしておき、あとは『a red car』のような **自然文** でも、**見本画像1枚** でも、
  似た写真を **何度でも素早く** 引ける “パーソナル画像検索（semantic photo search）” を作ること。
  （= Google Photos の「犬」「海」検索のローカル版。ベクトル検索の実アプリの最小形）

【実データ優先・合成フォールバック】
  data/42_multimodal_vector_search/ に .png/.jpg などを置けば、それが本物の写真ライブラリとして
  そのままインデックス化されます（＝そのまま実運用）。1枚も無ければ、合成の「色×形」写真ライブラリで
  必ず最後まで動きます（exit 0）。つまり **自分の画像フォルダを data/<id>/ に置くだけで実運用ツールに化ける**。

【なぜインデックスをキャッシュするのか（このツールの肝）】
  写真をCLIPで埋め込む処理はCPUでは重い。だが index をディスク（.faiss + .meta.json）に保存して
  おけば、2回目以降は **埋め込みをやり直さず読み込むだけ** で即検索できる。これがベクトルDBを
  「永続化」する実運用上の最大の価値で、本ツールはそれを体験できる形にしてあります。

【拡張アイデア（ここから自分で育てる練習）】
  - クエリを argparse でコマンドライン引数から受け取る対話CLIにする（`--text "a dog"` / `--image path`）。
  - 写真が数万枚に増えたら IndexFlatIP を index_factory(d, "HNSW32", METRIC_INNER_PRODUCT) に差し替える
    （add/search のコードは同じまま速くなる）。
  - サムネ＋ファイルパスの一覧を HTML ギャラリーとして書き出す。
  - EXIF の撮影日時・GPS をメタデータに足し、日付や場所で絞り込む（id↔メタの別管理を活かす）。
  - image→image を使って「重複/そっくり写真」を炙り出す near-duplicate ファインダーへ発展させる。
  - 動画なら数フレーム間引きして各フレームを1枚として登録し、「シーン検索」にする。
  - 検索ログ（クエリ→クリックした結果）をためて、しきい値や並べ替えをチューニングする。

実行:
  uv run python lectures/42_multimodal_vector_search/use_case.py
成果物（図・JSON・index キャッシュ）は outputs/42_multimodal_vector_search/ に保存されます。
"""

from __future__ import annotations

import json
import pathlib
import sys

# 同じフォルダの共通ヘルパ（mm_helpers.py）を import する。
# ※ 01_*.py のような「数字始まり」のスクリプトは import 名にできないが、mm_helpers は普通の名前なので借用できる。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_helpers import (  # noqa: E402
    MultimodalSearchEngine,
    build_collection,
    embed_images,
    embed_texts,
    ensure_output_dir,
    load_clip,
    load_user_images_or_none,
    pick_device,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless でも savefig できるバックエンド（imshow は使わない）
import matplotlib.pyplot as plt  # noqa: E402

MODULE_LABEL = "42_multimodal_vector_search"


# ---------------------------------------------------------------------------
# 1. ライブラリの読み込み（実データ優先・合成フォールバック）
# ---------------------------------------------------------------------------
def load_library() -> tuple[list, list[dict], bool]:
    """検索対象の写真ライブラリを返す。

    返り値: (images, metas, is_synthetic)
      - data/<id>/ に実画像があればそれを使う（is_synthetic=False）。
      - 無ければ合成の「色×形」写真ライブラリで完走する（is_synthetic=True）。
    画像の読み込み（PIL）は軽い処理。重いのは後段の CLIP 埋め込みの方。
    """
    user = load_user_images_or_none()
    if user is not None:
        images, metas = user
        print(f"  実画像ライブラリを使用: {len(images)} 枚（data/{MODULE_LABEL}/ から）")
        return images, metas, False
    # フォールバック: 36枚（色4 × 形3 × variants3）の合成写真ライブラリ。
    images, metas = build_collection(variants=3)
    print(f"  合成ライブラリを使用: {len(images)} 枚（色×形）。data/{MODULE_LABEL}/ に画像を置けば実運用に切替。")
    return images, metas, True


# ---------------------------------------------------------------------------
# 2. インデックスの構築 or キャッシュ読み込み（build once, query many）
# ---------------------------------------------------------------------------
def get_or_build_engine(model, processor, device, images, metas, stem: pathlib.Path) -> MultimodalSearchEngine:
    """ディスクのキャッシュがあれば読み込み、無ければ埋め込んで構築・保存する。

    キャッシュ有効性は「保存済み index の件数 == 今のライブラリ枚数」で簡易判定する。
    （写真を足し引きすると件数が変わるので作り直す。実運用では mtime ハッシュ等で厳密化する → 拡張案）
    """
    idx_file = stem.with_suffix(".faiss")
    meta_file = stem.with_suffix(".meta.json")
    if idx_file.exists() and meta_file.exists():
        engine = MultimodalSearchEngine.load(stem)
        if engine.index.ntotal == len(images):
            print(f"  キャッシュを再利用: {idx_file.name}（再埋め込みをスキップ。これが永続化の価値）")
            return engine
        print("  キャッシュとライブラリの件数が不一致 → 作り直します。")

    # --- 構築（重いのはここ。CLIP で全画像を埋め込む）---
    print("  画像を CLIP で埋め込み中…（初回のみ重い。次回はキャッシュから即起動）")
    img_emb = embed_images(model, processor, images, device)
    engine = MultimodalSearchEngine()
    engine.add(img_emb, metas)  # 内部で L2 正規化 → IndexFlatIP（=コサイン検索）
    engine.save(stem)  # .faiss + .meta.json をセットで永続化
    print(f"  インデックスを保存: {idx_file.name} + {meta_file.name}")
    return engine


# ---------------------------------------------------------------------------
# 3. クエリ語彙（合成 / 実写で出し分け）
# ---------------------------------------------------------------------------
def pick_text_queries(is_synthetic: bool) -> list[str]:
    """テキスト検索の問い合わせ文を返す。

    合成ライブラリは「色×形」しか写っていないので、それに効くクエリを使う。
    実写ライブラリは中身が分からないので汎用の英文を置く（自分の写真に合わせて自由に書き換えて練習）。
    CLIP は英語で学習されているため、クエリは英語が無難。
    """
    if is_synthetic:
        return ["a red circle", "a blue square", "a green triangle"]
    # ↓ 実写真向け。あなたのライブラリの中身に合わせて編集してください（ここが練習ポイント）。
    return ["a person", "a cat or a dog", "food on a plate", "a car on the street"]


# ---------------------------------------------------------------------------
# 4. 可視化（コンタクトシート）— 図タイトルは ASCII のみ
# ---------------------------------------------------------------------------
def save_contact_sheet(rows: list[tuple[str, object, list]], path: pathlib.Path, suptitle: str) -> None:
    """各クエリ（行）ごとに [クエリ] + [上位ヒット画像] を横一列に並べた一覧図を保存する。

    rows の各要素: (query_label, query_image_or_None, hits)
      - query_label: 図に書く ASCII 文字列（テキストクエリの本文 or "QUERY IMAGE"）
      - query_image_or_None: 画像クエリなら PIL.Image、テキストクエリなら None
      - hits: [{"image": PIL.Image, "caption": str(ASCII)}] のリスト（上位ヒット）
    """
    nrows = len(rows)
    max_hits = max((len(h) for _, _, h in rows), default=1)
    ncols = 1 + max(max_hits, 1)  # 左端1列をクエリ表示に使う
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.1, nrows * 2.3), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")
    for r, (q_label, q_img, hits) in enumerate(rows):
        # 左端: クエリ（画像クエリなら画像、テキストクエリなら文字を描く）
        if q_img is not None:
            axes[r][0].imshow(q_img)
            axes[r][0].set_title("QUERY IMAGE", fontsize=8, color="darkred")
        else:
            axes[r][0].text(0.5, 0.5, f"QUERY:\n{q_label}", ha="center", va="center",
                            fontsize=9, color="darkred", wrap=True)
        # 右側: 上位ヒット
        for c, hit in enumerate(hits, start=1):
            axes[r][c].imshow(hit["image"])
            axes[r][c].set_title(hit["caption"], fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. メイン
# ---------------------------------------------------------------------------
def main() -> None:
    out = ensure_output_dir()
    device = pick_device()
    print(f"=== use_case: パーソナル画像検索（自然文・見本画像でライブラリを引く） device={device} ===")

    # --- ライブラリ準備 ---
    print("\n[1] 写真ライブラリの読み込み")
    images, metas, is_synthetic = load_library()
    id2image = {i: images[i] for i in range(len(images))}  # 採番は 0..N-1（add 順）なので位置で対応づく

    # --- CLIP ロード（重み DL は初回だけ） ---
    model, processor = load_clip(device)

    # --- インデックス構築 or キャッシュ読み込み ---
    print("\n[2] インデックス（FAISS）の準備")
    stem = out / "use_case_library"
    engine = get_or_build_engine(model, processor, device, images, metas, stem)
    assert engine.index.ntotal == len(images), "index 件数とライブラリ枚数が不一致"
    topk = min(4, max(engine.index.ntotal - 1, 1))  # 表示する上位件数（小さいライブラリでも安全に）

    report: dict = {
        "mode": "real_images" if not is_synthetic else "synthetic_fallback",
        "device": str(device),
        "model_id": engine.model_id,
        "library_size": len(images),
        "topk": topk,
        "text_search": [],
        "image_search": {},
    }

    # --- テキスト → 画像（自然文で写真を引く）---
    print("\n[3] テキスト検索（自然文 → 写真）")
    queries = pick_text_queries(is_synthetic)
    text_rows: list[tuple[str, object, list]] = []
    for q in queries:
        qv = embed_texts(model, processor, [q], device)
        hits_raw = engine.search(qv, k=topk, modality="image")[0]
        hits_fig = [{"image": id2image[h["id"]], "caption": f"id={h['id']}\ncos={h['score']:.2f}"} for h in hits_raw]
        text_rows.append((q, None, hits_fig))
        report["text_search"].append({
            "query": q,
            "hits": [{"id": h["id"], "score": round(h["score"], 3),
                      "label": h["meta"].get("text", "")} for h in hits_raw],
        })
        top = hits_raw[0] if hits_raw else {}
        print(f"  '{q}' -> top1 id={top.get('id')} cos={top.get('score', 0):.2f} ({top.get('meta', {}).get('text', '')})")
    save_contact_sheet(text_rows, out / "use_case_text_search.png", "use_case: text-to-photo search")

    # --- 画像 → 画像（見本写真1枚で似た写真を引く）---
    print("\n[4] 画像検索（見本写真 → 似た写真）")
    q_id = 0  # 先頭の写真を見本クエリにする（実運用ではユーザが選んだ1枚）
    q_image = id2image[q_id]
    q_vec = embed_images(model, processor, [q_image], device)
    # 自分自身（cos≈1.0）が必ず1位に来るので、それを除いて似た“別の”写真を見せる。
    ii_raw = [h for h in engine.search(q_vec, k=topk + 1, modality="image")[0] if h["id"] != q_id][:topk]
    ii_fig = [{"image": id2image[h["id"]], "caption": f"id={h['id']}\ncos={h['score']:.2f}"} for h in ii_raw]
    save_contact_sheet([("QUERY IMAGE", q_image, ii_fig)], out / "use_case_image_search.png",
                       "use_case: image-to-photo search")
    report["image_search"] = {
        "query_id": q_id,
        "query_label": metas[q_id].get("text", ""),
        "hits": [{"id": h["id"], "score": round(h["score"], 3),
                  "label": h["meta"].get("text", "")} for h in ii_raw],
    }
    top_ii = ii_raw[0] if ii_raw else {}
    print(f"  query id={q_id} ({metas[q_id].get('text', '')}) -> top1 id={top_ii.get('id')} cos={top_ii.get('score', 0):.2f}")

    # --- 成果物の保存 ---
    report_path = out / "use_case_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[done] 成果物:")
    print(f"  - {report_path.name}")
    print("  - use_case_text_search.png / use_case_image_search.png")
    print(f"  - use_case_library.faiss (+ .meta.json)  ← 次回はこのキャッシュから即起動")
    if is_synthetic:
        print(f"\nヒント: data/{MODULE_LABEL}/ に自分の写真(.png/.jpg)を置いて再実行すると、実写ライブラリ検索になります。")


if __name__ == "__main__":
    main()
