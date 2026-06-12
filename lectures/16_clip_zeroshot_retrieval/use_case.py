"""use_case.py — 自然文ローカル画像検索ツール（CLIP text->image の実用最小版）。

【何をするツールか】
  自分の画像フォルダに対して、『a red car』『a person smiling』のような
  自然文で検索し、似ている画像を上位から並べて返す小さな検索ツールです。
  Google フォトや Apple 写真の「ことばで写真を探す」機能の、最小自作版にあたります。
  ゼロショット『分類』が候補=ラベル文だったのに対し、検索は候補=画像フォルダになるだけで、
  「画像とテキストを同じ空間に埋め込み → L2 正規化 → コサインで近い順に並べる」という
  CLIP の骨格はまったく同じです（本章 §6 の実用化）。

【実データで使うには（ここが本題）】
  data/16_clip_zeroshot_retrieval/ に自分の .png/.jpg を置くだけで、
  そのフォルダが検索対象になります（合成データは使われなくなります）。
  画像が1枚も無いときは、色×形の合成コレクション（12枚）に自動フォールバックして
  必ず完走します（exit 0）。まず合成で挙動を掴み、次に自分の写真で実運用に切り替える流れです。

【mini_project.py との違い】
  mini_project.py は章の総合課題（評価・アブレーション・棄却を1本に通すベンチ寄りの完成形）。
  こちらは『現実の小ツール』として完結し、コマンドラインから自由文クエリを渡して使えます。
  コピーして自分の写真検索アプリの出発点にしてください。

【使い方】
  # 既定のデモクエリで動かす（引数なし）
  uv run python lectures/16_clip_zeroshot_retrieval/use_case.py
  # 自分の検索語を渡す（複数可・自然文OK）
  uv run python lectures/16_clip_zeroshot_retrieval/use_case.py "a red car" "a blue square"

【出力（すべて outputs/16_clip_zeroshot_retrieval/ に保存）】
  use_case_results.png    クエリごとの上位ヒットを並べた検索結果パネル
  use_case_results.json   ランキング（caption・cos・該当なし判定）の数値
  use_case_index.npz      画像埋め込みのキャッシュ（2回目以降の検索を高速化）

【拡張アイデア（練習）】
  - image->image（似た画像で探す）を足す: クエリを画像にして同じ正規化＋コサイン（§6）。
  - 件数が増えたら FAISS（第17回）へ: IndexFlatIP + normalize_L2 が、ここのコサインそのもの。
  - 棄却（該当なし）を厳密化: SigLIP の sigmoid（§5）や、検証用データで閾値を較正する。
  - メタデータ（撮影日・タグ）を JSON で併せ持ち、検索結果に添えて表示する。
  - クエリ・拡張: "a photo of a {x}" 等のテンプレ平均（mini_project Stage A）で頑健化。
  - サムネイル一覧を Web UI（Flask / Streamlit）に載せて検索ボックスを付ける。
"""

from __future__ import annotations

import json
import pathlib
import sys

# digit 始まりの番号付きスクリプトは import 不可だが、clip_helpers は import できる。
# 同じディレクトリを import パスへ追加して道具箱を借りる（device 判定・モデルロード等）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import clip_helpers as H  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 上位何件を表示・保存するか（実フォルダが小さい時は枚数に合わせて自動で縮める）。
TOP_K = 4
# コサインがこの値未満なら「弱いヒット（該当なしの疑い）」と注記する簡易しきい値。
# ※本当は検証データで較正すべき値。ここでは合成セットで体感できる目安として固定値を使う。
WEAK_COS_THRESHOLD = 0.22

# 引数なしで動かしたときのデモクエリ。自然文で、合成セット（色×形）でも意味が出るもの＋
# 語彙に無い "a red car"（=該当なしの体験用）を1つ混ぜてある。
DEMO_QUERIES = [
    "a red circle",
    "a blue square",
    "a green triangle",
    "a yellow shape",
    "a red car",  # 合成セットには車が無い → 弱いヒットになり「該当なし」を体験できる
]

# 画像埋め込みキャッシュの保存先（フォルダ内容が同じなら再エンコードを省ける）。
INDEX_FILENAME = "use_case_index.npz"


def find_image_files() -> list[pathlib.Path]:
    """data/<module_id>/ にある画像ファイル一覧を返す（無ければ空リスト）。

    実データを使っているか合成にフォールバックしたかを、利用者に明示するために自前で確認する。
    """
    data_dir = pathlib.Path("data") / H.MODULE_ID
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not data_dir.is_dir():
        return []
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in exts)


def embed_images(model, processor, images, device, batch_size: int = 16) -> torch.Tensor:
    """画像フォルダを CLIP で埋め込む（射影後・未正規化、(N, D)）。

    実フォルダは枚数が多くなりうるので、メモリに優しいようにバッチ分割して回す
    （CPU でも一気に全画像を processor へ渡すと重い）。正規化は呼び出し側で行う。
    """
    embs: list[torch.Tensor] = []
    with torch.inference_mode():  # 推論は勾配を切る（CPU では特に効く）
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            embs.append(H.clip_image_embeds(model, inputs["pixel_values"]))
    return torch.cat(embs, dim=0)


def embed_texts(model, processor, texts, device) -> torch.Tensor:
    """検索クエリ群を CLIP で埋め込む（射影後・未正規化、(Q, D)）。

    複数文をまとめて通すので padding=True を忘れない（長さ不一致でエラーになる）。
    """
    with torch.inference_mode():
        inputs = processor(text=list(texts), return_tensors="pt", padding=True).to(device)
        return H.clip_text_embeds(model, inputs["input_ids"], inputs["attention_mask"])


def build_or_load_index(
    model, processor, images, captions, device, cache_path: pathlib.Path
) -> tuple[torch.Tensor, bool]:
    """画像埋め込み（L2正規化済み）を作る。キャッシュが使えれば再エンコードを省く。

    実運用の定石「インデックスは一度作って保存、検索のたびに再利用」を小さく再現する。
    キャッシュは (caption 一覧, モデルID) が一致するときだけ採用し、ズレていれば作り直す。
    返り値: (正規化済み埋め込み (N, D), キャッシュから読んだか)
    """
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            cached_caps = json.loads(data["captions"].item())
            cached_model = data["model_id"].item()
            # フォルダ内容（=caption列）とモデルが一致して初めて再利用できる。
            if cached_caps == list(captions) and cached_model == H.CLIP_MODEL_ID:
                return torch.from_numpy(data["emb"]).to(device), True
        except Exception:
            pass  # 壊れた/古い形式のキャッシュは黙って無視して作り直す

    raw = embed_images(model, processor, images, device)
    emb = F.normalize(raw, p=2, dim=-1)  # コサイン類似度のための L2 正規化（必須・§4/§8）
    # キャッシュ保存: 埋め込み本体＋（整合性チェック用の）caption一覧とモデルID。
    # caption は JSON 文字列を 0次元配列に入れて保存すると allow_pickle 不要で安全。
    np.savez(
        cache_path,
        emb=emb.detach().cpu().numpy().astype(np.float32),
        captions=np.array(json.dumps(list(captions))),
        model_id=np.array(H.CLIP_MODEL_ID),
    )
    return emb, False


def search(txt_emb: torch.Tensor, img_emb: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """正規化済みクエリ・画像埋め込みからコサイン類似度で上位 topk を引く。

    正規化済みベクトル同士の内積 = コサイン類似度（-1〜1）。topk が大きいほど近い。
    返り値: (上位スコア (Q, topk), 上位インデックス (Q, topk))
    """
    sims = txt_emb @ img_emb.t()  # (Q, N)
    top_scores, top_idx = torch.topk(sims, k=topk, dim=1)
    return top_scores, top_idx


def plot_results(
    images, captions, queries, top_scores, top_idx, out_path: pathlib.Path
) -> None:
    """クエリごとの上位ヒットを並べた検索結果パネルを保存する。

    図タイトルは CJK 豆腐（□）回避のため ASCII 固定。クエリ本文（日本語可）は
    コンソールと JSON 側に出すので、ここでは行ラベルを Q1, Q2... のアンカーにする。
    """
    nrow = len(queries)
    ncol = top_idx.shape[1]
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.7))
    axes = np.array(axes).reshape(nrow, ncol)
    for r in range(nrow):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            idx = int(top_idx[r, c])
            cos = float(top_scores[r, c])
            ax.imshow(images[idx])  # PIL は RGB なので matplotlib にそのまま渡せる
            weak = "  (weak)" if cos < WEAK_COS_THRESHOLD else ""
            ax.set_title(f"#{c+1} {captions[idx]}\ncos={cos:.3f}{weak}", fontsize=7)
        # 行の左に Q1/Q2... のアンカー（クエリ本文は console/JSON 参照）。
        axes[r, 0].set_ylabel(f"Q{r+1}", fontsize=10, rotation=0, ha="right", va="center")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle("natural-language image search (text -> image, cosine top-k)", fontsize=11)
    fig.tight_layout(rect=(0.04, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv: list[str]) -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # --- 1) 検索対象フォルダの読み込み（実データ優先・合成フォールバック）---
    files = find_image_files()
    images, captions = H.load_user_or_synthetic()
    source = f"data/{H.MODULE_ID}/（実画像 {len(files)} 枚）" if files else "合成コレクション（色×形）"
    topk = min(TOP_K, len(images))

    # --- 2) クエリの決定（引数があればそれ、無ければデモ）---
    queries = argv[1:] if len(argv) > 1 else DEMO_QUERIES

    # --- 3) モデルとインデックス（画像埋め込み）---
    model, processor = H.load_clip(device)
    cache_path = out / INDEX_FILENAME
    img_emb, from_cache = build_or_load_index(model, processor, images, captions, device, cache_path)

    # --- 4) クエリ埋め込み → 正規化 → コサインで上位 topk ---
    txt_emb = F.normalize(embed_texts(model, processor, queries, device), p=2, dim=-1)
    top_scores, top_idx = search(txt_emb, img_emb, topk)

    # --- 5) コンソール出力 + JSON 整形 ---
    print(f"device={device}  source={source}  images={len(images)}  index_cache={'hit' if from_cache else 'built'}")
    results = []
    for r, q in enumerate(queries):
        hits = []
        for c in range(topk):
            idx = int(top_idx[r, c])
            cos = round(float(top_scores[r, c]), 4)
            hits.append({"rank": c + 1, "caption": captions[idx], "cos": cos})
        top_cos = hits[0]["cos"] if hits else 0.0
        weak = top_cos < WEAK_COS_THRESHOLD  # 最上位でも弱い → 「該当なし」の疑い
        results.append({"query": q, "weak_match": weak, "hits": hits})
        flag = "  [該当なしの疑い]" if weak else ""
        head = hits[0] if hits else {"caption": "-", "cos": 0.0}
        print(f"  Q{r+1} {q!r:28s} -> #1 {head['caption']:16s} cos={head['cos']:.3f}{flag}")

    # --- 6) 図 + JSON 保存 ---
    plot_results(images, captions, queries, top_scores, top_idx, out / "use_case_results.png")
    payload = {
        "source": source,
        "n_images": len(images),
        "model_id": H.CLIP_MODEL_ID,
        "top_k": topk,
        "weak_cos_threshold": WEAK_COS_THRESHOLD,
        "index_cache": "hit" if from_cache else "built",
        "queries": results,
    }
    (out / "use_case_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {out/'use_case_results.png'}, {out/'use_case_results.json'}, {cache_path}")


if __name__ == "__main__":
    main(sys.argv)
