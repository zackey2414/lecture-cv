"""第24回 演習問題（画像キャプションの評価メトリクスを自分の手で書く）。

なぜ「評価」を演習にするのか:
  キャプション生成そのものは model.generate を呼ぶだけで、難しさは少ない。
  本当に理解が問われるのは「出てきた文をどう採点するか」。BLEU / CIDEr / CLIPScore の
  中身を一度自分で実装すると、スコアの大小やパラメータ依存を腹落ちさせられる。

使い方:
  1. 各 exN_*() の TODO を実装する（最初は NotImplementedError だが、採点ランナーが
     拾うのでプロセスは落ちず FAIL と表示されるだけ＝exit 0）。
  2. 自己採点:  uv run python lectures/24_image_captioning/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る:
        SHOW_SOLUTION=1 uv run python lectures/24_image_captioning/exercises.py
     もしくは完全な模範解答ファイル exercises_solutions.py を実行する。

8問（易→難）:
  ex1: トークナイズ（小文字化＋記号除去＋分割）             … 全指標の前処理
  ex2: n-gram の出現回数                                    … BLEU/CIDEr の土台
  ex3: 修正 n-gram 適合率（クリップ）                        … BLEU の核
  ex4: 簡潔さ罰 Brevity Penalty                            … BLEU の核
  ex5: 文 BLEU（適合率の幾何平均 × BP）                     … BLEU 完成
  ex6: TF-IDF コサイン                                      … CIDEr の核
  ex7: CLIPScore（埋め込みのコサイン × 100, 0 でクリップ）   … 参照なし指標
  ex8: ベスト設定の選択（指標の argmax）                     … 比較実験のまとめ
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
from collections import Counter

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 演習（ここを実装する）
# =====================================================================


def ex1_tokenize(text: str) -> list[str]:
    """演習1: 文字列をトークン列にする。

    手順: 小文字化 → 記号（. , ! ? など string.punctuation）を空白に置換 → 空白で分割。
    例: "A red, CAR." -> ["a", "red", "car"]
    ヒント: str.lower(), str.maketrans, str.translate, str.split を使う。
    """
    # TODO: 小文字化→記号除去→split したリストを返す
    raise NotImplementedError


def ex2_ngram_counts(tokens: list[str], n: int) -> dict[tuple, int]:
    """演習2: トークン列から n-gram の出現回数を数える（キーは tuple）。

    例: (["a","b","c","a","b"], 2) -> {("a","b"):2, ("b","c"):1, ("c","a"):1}
    トークン数が n 未満なら空 dict。collections.Counter を使ってよい。
    """
    # TODO: i..i+n の tuple を数えて dict（または Counter）で返す
    raise NotImplementedError


def ex3_modified_precision(candidate: list[str], references: list[list[str]], n: int) -> float:
    """演習3: BLEU の「修正 n-gram 適合率」。

    候補の各 n-gram 数を、参照側での最大出現回数でクリップして合計し、候補の
    n-gram 総数で割る。候補に n-gram が無ければ 0.0。
    例: candidate=["a","a","b"], references=[["a","b","c"]], n=1
        → a は min(2,1)=1、b は min(1,1)=1、合計 2 / 候補3 = 0.667
    ex2 を使ってよい。
    """
    # TODO: clip した一致数 / 候補 n-gram 総数 を返す
    raise NotImplementedError


def ex4_brevity_penalty(cand_len: int, ref_lens: list[int]) -> float:
    """演習4: 簡潔さ罰（Brevity Penalty）。

    候補長 c に最も近い参照長 r（同点なら短い方）を選び、c>r なら 1.0、
    c<=r なら exp(1 - r/c)。c==0 なら 0.0。
    例: (3,[5]) -> exp(1-5/3)≈0.5134 / (6,[5]) -> 1.0
    """
    # TODO: 最も近い r を選び、式に従って返す
    raise NotImplementedError


def ex5_sentence_bleu(candidate: str, references: list[str], max_n: int = 4) -> float:
    """演習5: 1文ぶんの BLEU（加算スムージング付き）。

    手順:
      1. candidate と各 reference を ex1 でトークナイズ。
      2. n=1..max_n の修正適合率 p_n（ex3）を求め、0 のものは 1e-9 に置き換える
         （短文で p_n=0 になり全体が log(0) で潰れるのを防ぐ）。
      3. 幾何平均 exp( (Σ log p_n) / max_n ) を取り、ex4 の BP を掛ける。
    完全一致なら 1.0、全く重ならなければ約 0 になること。
    """
    # TODO: ex1/ex3/ex4 を組み合わせて文 BLEU を返す
    raise NotImplementedError


def ex6_tfidf_cosine(
    counts_a: dict[tuple, int], counts_b: dict[tuple, int], idf: dict[tuple, float]
) -> float:
    """演習6: TF-IDF 重み付き n-gram ベクトルのコサイン類似度（CIDEr の核）。

    各 n-gram g の重みを weight(g) = counts[g] * idf[g] とし、a, b それぞれの
    重みベクトルのコサイン類似度を返す（length 正規化はしない素朴版）。
    どちらかが空、またはノルム 0 なら 0.0。
    例: a={("cat",):2,("dog",):1}, b={("cat",):1,("bird",):1},
        idf={("cat",):0.5,("dog",):2.0,("bird",):2.0} → 約 0.1085
    """
    # TODO: weight=count*idf のベクトルを作り、内積/(ノルム積) を返す
    raise NotImplementedError


def ex7_clip_score(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> list[float]:
    """演習7: CLIPScore（参照不要）。1対1ペアの整合度を返す。

    - image_embeds, text_embeds: ともに形 (N, D)。未正規化で渡される。
    - 各行を L2 正規化し、対応する行どうしのコサインを取り、×100 して 0 でクリップ。
    - 返り値は長さ N の float リスト。
    例: img=[[1,0],[0,3]], txt=[[2,0],[0,-1]] → [100.0, 0.0]
    ヒント: torch.nn.functional.normalize, 行ごとの内積は (a*b).sum(dim=-1)。
    """
    # TODO: 正規化→行ごとの内積→×100→clamp(min=0) を list で返す
    raise NotImplementedError


def ex8_best_setting(scores: dict[str, float]) -> str:
    """演習8: 指標が最大の設定キーを返す（比較実験の締め）。

    - scores: {"blip/beam4": 0.27, "git/greedy": 0.24, ...} のような辞書。
    - 値が最大のキーを返す。同点なら「先に現れたキー」を返す（dict は挿入順）。
    - 空 dict なら空文字列 "" を返す。
    """
    # TODO: 値が最大のキーを（同点は先勝ちで）返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================


def _oracle_bleu(candidate: str, references: list[str], max_n: int = 4) -> float:
    """採点用の独立リファレンス実装（ex5 の答え合わせに使う）。"""
    import string

    table = str.maketrans({c: " " for c in string.punctuation})

    def tok(s: str) -> list[str]:
        return s.lower().translate(table).split()

    def ngrams(ts: list[str], n: int) -> Counter:
        return Counter(tuple(ts[i : i + n]) for i in range(len(ts) - n + 1))

    cand = tok(candidate)
    refs = [tok(r) for r in references]
    if not cand:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        cc = ngrams(cand, n)
        total = sum(cc.values())
        if total == 0:
            p = 1e-9
        else:
            mx: Counter = Counter()
            for r in refs:
                for g, c in ngrams(r, n).items():
                    mx[g] = max(mx[g], c)
            p = sum(min(c, mx[g]) for g, c in cc.items()) / total
            if p == 0.0:
                p = 1e-9
        log_sum += math.log(p)
    geo = math.exp(log_sum / max_n)
    r = min((len(x) for x in refs), key=lambda rl: (abs(rl - len(cand)), rl))
    bp = 1.0 if len(cand) > r else math.exp(1.0 - r / len(cand))
    return bp * geo


def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        got = ex1_tokenize("A red, CAR.")
        got2 = ex1_tokenize("Hello   World!!")
        return got == ["a", "red", "car"] and got2 == ["hello", "world"], f"{got}"

    check("ex1_tokenize", _c1)

    def _c2():
        got = dict(ex2_ngram_counts(["a", "b", "c", "a", "b"], 2))
        exp = {("a", "b"): 2, ("b", "c"): 1, ("c", "a"): 1}
        empty = dict(ex2_ngram_counts(["a"], 2))
        return got == exp and empty == {}, f"{got}"

    check("ex2_ngram_counts", _c2)

    def _c3():
        p1 = ex3_modified_precision(["a", "a", "b"], [["a", "b", "c"]], 1)
        p2 = ex3_modified_precision(["the", "the", "the"], [["the", "cat"]], 1)  # 1/3
        z = ex3_modified_precision(["a"], [["a", "b"]], 2)  # 候補に2-gram無し→0
        return abs(p1 - 2 / 3) < 1e-9 and abs(p2 - 1 / 3) < 1e-9 and z == 0.0, f"p1={p1:.3f}"

    check("ex3_modified_precision", _c3)

    def _c4():
        a = ex4_brevity_penalty(3, [5])
        b = ex4_brevity_penalty(6, [5])
        c = ex4_brevity_penalty(0, [3])
        d = ex4_brevity_penalty(5, [5])
        ok = abs(a - math.exp(1 - 5 / 3)) < 1e-9 and b == 1.0 and c == 0.0 and d == 1.0
        return ok, f"bp(3,[5])={a:.4f}"

    check("ex4_brevity_penalty", _c4)

    def _c5():
        perfect = ex5_sentence_bleu("a red car on the road", ["a red car on the road"])
        disjoint = ex5_sentence_bleu("x y z", ["a b c"])
        mixed_c, mixed_r = "a red car is on the road", ["a red car on the road"]
        mixed = ex5_sentence_bleu(mixed_c, mixed_r)
        ref = _oracle_bleu(mixed_c, mixed_r)
        ok = abs(perfect - 1.0) < 1e-6 and disjoint < 1e-3 and abs(mixed - ref) < 1e-6
        return ok, f"perfect={perfect:.3f} mixed={mixed:.4f}"

    check("ex5_sentence_bleu", _c5)

    def _c6():
        a = {("cat",): 2, ("dog",): 1}
        b = {("cat",): 1, ("bird",): 1}
        idf = {("cat",): 0.5, ("dog",): 2.0, ("bird",): 2.0}
        got = ex6_tfidf_cosine(a, b, idf)
        same = ex6_tfidf_cosine(a, a, idf)
        none = ex6_tfidf_cosine(a, {("x",): 1}, {("x",): 1.0, **idf})
        ok = abs(got - 0.108465) < 1e-3 and abs(same - 1.0) < 1e-6 and none == 0.0
        return ok, f"cos={got:.4f}, self={same:.3f}"

    check("ex6_tfidf_cosine", _c6)

    def _c7():
        img = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
        txt = torch.tensor([[2.0, 0.0], [0.0, -1.0]])
        got = ex7_clip_score(img, txt)
        ok = isinstance(got, list) and abs(got[0] - 100.0) < 1e-3 and abs(got[1] - 0.0) < 1e-3
        return ok, f"{[round(g, 2) for g in got]}"

    check("ex7_clip_score", _c7)

    def _c8():
        s = {"blip/beam4": 0.27, "git/greedy": 0.24, "git/beam4": 0.27}  # 同点は先勝ち
        best = ex8_best_setting(s)
        empty = ex8_best_setting({})
        return best == "blip/beam4" and empty == "", f"best={best}"

    check("ex8_best_setting", _c8)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替え）。まず自力で解くこと。
# =====================================================================


def _sol_ex1(text: str) -> list[str]:
    import string

    return text.lower().translate(str.maketrans({c: " " for c in string.punctuation})).split()


def _sol_ex2(tokens: list[str], n: int) -> dict:
    return dict(Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)))


def _sol_ex3(candidate: list[str], references: list[list[str]], n: int) -> float:
    cand = Counter(_sol_ex2(candidate, n))
    total = sum(cand.values())
    if total == 0:
        return 0.0
    mx: Counter = Counter()
    for r in references:
        for g, c in _sol_ex2(r, n).items():
            mx[g] = max(mx[g], c)
    return sum(min(c, mx[g]) for g, c in cand.items()) / total


def _sol_ex4(cand_len: int, ref_lens: list[int]) -> float:
    if cand_len == 0:
        return 0.0
    r = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))
    return 1.0 if cand_len > r else math.exp(1.0 - r / cand_len)


def _sol_ex5(candidate: str, references: list[str], max_n: int = 4) -> float:
    cand = _sol_ex1(candidate)
    refs = [_sol_ex1(r) for r in references]
    if not cand:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p = _sol_ex3(cand, refs, n)
        log_sum += math.log(p if p > 0 else 1e-9)
    geo = math.exp(log_sum / max_n)
    return _sol_ex4(len(cand), [len(r) for r in refs]) * geo


def _sol_ex6(counts_a: dict, counts_b: dict, idf: dict) -> float:
    va = {g: c * idf.get(g, 0.0) for g, c in counts_a.items()}
    vb = {g: c * idf.get(g, 0.0) for g, c in counts_b.items()}
    if not va or not vb:
        return 0.0
    dot = sum(v * vb.get(g, 0.0) for g, v in va.items())
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _sol_ex7(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> list[float]:
    import torch.nn.functional as F

    ie = F.normalize(image_embeds, p=2, dim=-1)
    te = F.normalize(text_embeds, p=2, dim=-1)
    cos = (ie * te).sum(dim=-1)
    return [float(s) for s in (cos * 100.0).clamp(min=0.0)]


def _sol_ex8(scores: dict[str, float]) -> str:
    if not scores:
        return ""
    return max(scores, key=lambda k: scores[k])


def _install_solutions() -> None:
    g = globals()
    g["ex1_tokenize"] = _sol_ex1
    g["ex2_ngram_counts"] = _sol_ex2
    g["ex3_modified_precision"] = _sol_ex3
    g["ex4_brevity_penalty"] = _sol_ex4
    g["ex5_sentence_bleu"] = _sol_ex5
    g["ex6_tfidf_cosine"] = _sol_ex6
    g["ex7_clip_score"] = _sol_ex7
    g["ex8_best_setting"] = _sol_ex8


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
