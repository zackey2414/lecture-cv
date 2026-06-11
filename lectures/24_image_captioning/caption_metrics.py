"""第24回 キャプション評価メトリクスの道具箱。

ここには「参照あり」「参照なし」両系統の指標を、教材として中身が見える形で
実装してあります（単一責務の純関数の集まり）。

  参照あり（生成 vs 人手キャプション）:
    - BLEU      : torchmetrics.text.BLEUScore（n-gram 適合率 × 簡潔さ罰）
    - ROUGE-L   : torchmetrics.text.ROUGEScore（最長共通部分列ベース）
    - CIDEr     : 自前実装（TF-IDF 重み付き n-gram のコサイン類似度＝主指標）
  参照なし（画像とキャプションの整合度）:
    - CLIPScore : 自前実装（CLIP 埋め込みのコサイン × 100）

★なぜ CLIPScore を自前実装するのか（落とし穴）:
  torchmetrics.multimodal.CLIPScore は内部で CLIP.get_image_features() の戻り値を
  「テンソル」として扱うが、transformers v5 では BaseModelOutputWithPooling（.pooler_output に
  本体が入るオブジェクト）が返るため、現状そのままでは動かない。原理はシンプル
  （正規化したコサイン × 100）なので、ここでは手で書いて中身を理解する。

SPICE / METEOR は Java 実行環境が要るため本講座では扱わない（概念のみ README で紹介）。
"""

from __future__ import annotations

import functools
import math
import string
from collections import Counter

import torch
import torch.nn.functional as F
from PIL import Image

# 句読点を空白に潰す変換表（トークナイズを決定的にする）。
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


# =====================================================================
# 基本部品（BLEU / CIDEr の土台）
# =====================================================================


def tokenize(text: str) -> list[str]:
    """小文字化 → 句読点を除去 → 空白で分割（決定的な簡易トークナイザ）。

    例: "A red, CAR." -> ["a", "red", "car"]
    """
    return text.lower().translate(_PUNCT_TABLE).split()


def ngram_counts(tokens: list[str], n: int) -> Counter:
    """トークン列から n-gram の出現回数を数える（キーは tuple）。

    例: (["a","b","c"], 2) -> {("a","b"):1, ("b","c"):1}
    """
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(candidate: list[str], references: list[list[str]], n: int) -> float:
    """BLEU の核「修正 n-gram 適合率」。

    候補の各 n-gram の数を、参照側での最大出現回数で「クリップ」して数える
    （同じ語を水増ししても稼げないようにする）。返り値は clip 合計 / 候補 n-gram 総数。
    候補に n-gram が無ければ 0.0。
    """
    cand = ngram_counts(candidate, n)
    total = sum(cand.values())
    if total == 0:
        return 0.0
    # 参照側での各 n-gram の最大出現回数
    max_ref: Counter = Counter()
    for ref in references:
        rc = ngram_counts(ref, n)
        for g, c in rc.items():
            if c > max_ref[g]:
                max_ref[g] = c
    clipped = sum(min(c, max_ref[g]) for g, c in cand.items())
    return clipped / total


def brevity_penalty(cand_len: int, ref_lens: list[int]) -> float:
    """簡潔さ罰（Brevity Penalty）。短すぎる候補にペナルティを与える。

    候補長 c に最も近い参照長 r を選び、c>r なら 1.0、c<=r なら exp(1 - r/c)。
    c==0 なら 0.0。
    """
    if cand_len == 0:
        return 0.0
    # |r - c| 最小（同点なら短い方）の参照長を選ぶ
    r = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))
    if cand_len > r:
        return 1.0
    return math.exp(1.0 - r / cand_len)


def sentence_bleu(candidate: str, references: list[str], max_n: int = 4) -> float:
    """1文ぶんの BLEU（教材用の素朴実装、加算スムージング付き）。

    手順: n=1..max_n の修正適合率 p_n を求め、0 のものは微小値で底上げ（短文で
    p_n=0 になり全体が 0 に潰れるのを防ぐ）→ 幾何平均 → 簡潔さ罰を掛ける。
    """
    cand = tokenize(candidate)
    refs = [tokenize(r) for r in references]
    if not cand:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p = modified_precision(cand, refs, n)
        if p == 0.0:
            p = 1e-9  # 加算スムージング（log(0) 回避）
        log_sum += math.log(p)
    geo_mean = math.exp(log_sum / max_n)
    bp = brevity_penalty(len(cand), [len(r) for r in refs])
    return bp * geo_mean


# =====================================================================
# CIDEr（自前・簡易版）: TF-IDF 重み付き n-gram のコサイン類似度
# =====================================================================
# 直感: 「誰でも言う一般的な語（a, photo, of）」は重みを下げ、
#       「その画像特有の珍しい n-gram」が一致したときに高く評価する。
#       珍しさは IDF（参照コーパス全体での出現の少なさ）で測る。


def _tfidf_vector(counts: Counter, idf: dict[tuple, float]) -> dict[tuple, float]:
    """正規化 TF（count / 総数）に IDF を掛けた疎ベクトル（dict）を作る。"""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {g: (c / total) * idf.get(g, 0.0) for g, c in counts.items()}


def tfidf_cosine(vec_a: dict[tuple, float], vec_b: dict[tuple, float]) -> float:
    """2つの疎ベクトル（dict）のコサイン類似度。空ベクトルなら 0.0。"""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(v * vec_b.get(g, 0.0) for g, v in vec_a.items())
    na = math.sqrt(sum(v * v for v in vec_a.values()))
    nb = math.sqrt(sum(v * v for v in vec_b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _compute_idf(references_corpus: list[list[str]], n: int) -> dict[tuple, float]:
    """参照コーパスから n-gram の IDF を計算する。

    references_corpus: 画像ごとの参照キャプション（生文字列）のリストのリスト。
    document = 1画像ぶんの全参照を連結した n-gram 集合とみなす。
    idf(g) = log( 画像数 / (g を含む画像数) )。全画像に出る語は idf=0（＝無視）。
    """
    num_docs = len(references_corpus)
    df: Counter = Counter()
    for refs in references_corpus:
        present: set[tuple] = set()
        for r in refs:
            present |= set(ngram_counts(tokenize(r), n).keys())
        for g in present:
            df[g] += 1
    return {g: math.log(num_docs / d) for g, d in df.items() if d > 0}


def compute_cider(
    candidates: list[str],
    references_corpus: list[list[str]],
    max_n: int = 4,
    idf_corpus: list[list[str]] | None = None,
) -> tuple[float, list[float]]:
    """簡易 CIDEr。各画像のスコアと、その平均を返す。

    各 n（1..max_n）について、候補と各参照の TF-IDF コサインを取り、参照平均→n平均。
    本来の CIDEr-D にある「×10」「長さガウス罰」は省いた教材版（値域はおよそ 0〜1）。

    重要: IDF は本来「データセット全体の参照コーパス」から求める量。評価対象が
    1枚（または参照がほぼ同一）だと IDF が全て 0 に潰れて CIDEr=0 になる。実運用に
    倶い、IDF を別コーパス（idf_corpus）から計算できるようにしてある（None なら
    references_corpus を流用）。
    """
    # n ごとの IDF をコーパスから一度だけ作る（既定は評価集合そのもの）
    corpus_for_idf = idf_corpus if idf_corpus is not None else references_corpus
    idf_by_n = {n: _compute_idf(corpus_for_idf, n) for n in range(1, max_n + 1)}
    per_image: list[float] = []
    for cand, refs in zip(candidates, references_corpus):
        cand_tok = tokenize(cand)
        ref_toks = [tokenize(r) for r in refs]
        n_scores: list[float] = []
        for n in range(1, max_n + 1):
            cand_vec = _tfidf_vector(ngram_counts(cand_tok, n), idf_by_n[n])
            sims = [
                tfidf_cosine(cand_vec, _tfidf_vector(ngram_counts(rt, n), idf_by_n[n]))
                for rt in ref_toks
            ]
            n_scores.append(sum(sims) / len(sims) if sims else 0.0)
        per_image.append(sum(n_scores) / max_n)
    mean = sum(per_image) / len(per_image) if per_image else 0.0
    return mean, per_image


# =====================================================================
# torchmetrics ラッパ（BLEU / ROUGE-L）
# =====================================================================


def corpus_bleu(
    candidates: list[str], references_corpus: list[list[str]], n_gram: int = 4
) -> float:
    """torchmetrics の BLEUScore でコーパス BLEU を計算する（スムージング有効）。"""
    from torchmetrics.text import BLEUScore

    metric = BLEUScore(n_gram=n_gram, smooth=True)
    return float(metric(candidates, references_corpus))


def rouge_l(candidates: list[str], references_corpus: list[list[str]]) -> float:
    """ROUGE-L の F値（複数参照は各候補につき最良の参照で評価）。"""
    from torchmetrics.text import ROUGEScore

    metric = ROUGEScore(rouge_keys=("rougeL",), accumulate="best")
    out = metric(candidates, references_corpus)
    return float(out["rougeL_fmeasure"])


# =====================================================================
# CLIPScore（参照不要・自前実装）
# =====================================================================


@functools.lru_cache(maxsize=2)
def _load_clip(model_id: str, device_str: str):
    """CLIP モデルとプロセッサをロードしてキャッシュする（再ロード防止）。"""
    from transformers import CLIPModel, CLIPProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model = CLIPModel.from_pretrained(model_id).to(device_str).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def clip_score_pairs(
    images: list[Image.Image],
    texts: list[str],
    device: torch.device | None = None,
    model_id: str = "openai/clip-vit-base-patch32",
) -> list[float]:
    """画像とキャプションを1対1で対応づけ、各ペアの CLIPScore を返す。

    CLIPScore = max(0, 100 · cos(image_embed, text_embed))。参照キャプションが
    要らない（＝人手アノテーション無しで品質の当たりが付けられる）のが利点。

    ★v5 の注意: get_image_features / get_text_features は
      BaseModelOutputWithPooling を返すので .pooler_output を取り出す。さらに
      未正規化なので F.normalize でコサインにしてから内積を取る。
    """
    if device is None:
        device = torch.device("cpu")
    model, processor = _load_clip(model_id, str(device))

    with torch.inference_mode():
        img_inputs = processor(images=images, return_tensors="pt").to(device)
        txt_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(
            device
        )
        img_emb = model.get_image_features(**img_inputs).pooler_output
        txt_emb = model.get_text_features(**txt_inputs).pooler_output
        img_emb = F.normalize(img_emb, p=2, dim=-1)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)
        # 1対1ペアのコサイン（行ごとの内積）→ ×100 → 0 でクリップ
        cos = (img_emb * txt_emb).sum(dim=-1)
        scores = (cos * 100.0).clamp(min=0.0)
    return [round(float(s), 3) for s in scores]


if __name__ == "__main__":
    # スモークテスト（モデル DL 無し・純計算のみ）。
    cand = "a red car on the road"
    refs = ["a red car on a road", "a red vehicle on the street"]
    print("sentence_bleu :", round(sentence_bleu(cand, refs), 4))
    print(
        "modified_precision n=1 :",
        round(modified_precision(tokenize(cand), [tokenize(r) for r in refs], 1), 4),
    )
    mean, per = compute_cider([cand], [refs])
    print("cider(mean)   :", round(mean, 4))
    print("rouge-L       :", round(rouge_l([cand], [refs]), 4))
    print("corpus_bleu   :", round(corpus_bleu([cand], [refs]), 4))
