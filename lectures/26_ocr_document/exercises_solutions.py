"""第26回 演習の模範解答（このファイルを実行すると全問 PASS する）。

exercises.py の各 TODO を埋めた参照実装。まずは自力で exercises.py を解き、
詰まったらここで答え合わせをすること。

実行: uv run python lectures/26_ocr_document/exercises_solutions.py
"""

from __future__ import annotations

import re
import unicodedata


# =====================================================================
# 模範解答
# =====================================================================

def ex1_normalize_text(s: str) -> str:
    """NFKC → 小文字化 → strip → 連続空白を1つに。"""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s.strip())


def ex2_levenshtein(a, b) -> int:
    """編集距離（S+D+I）。1 行 DP（メモリ O(n)）。文字列でもリストでも動く。"""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def ex3_cer(reference: str, hypothesis: str) -> float:
    """CER = 編集距離(文字) / 参照文字数。"""
    r = ex1_normalize_text(reference)
    h = ex1_normalize_text(hypothesis)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return ex2_levenshtein(r, h) / len(r)


def ex4_wer(reference: str, hypothesis: str) -> float:
    """WER = 編集距離(単語) / 参照単語数。"""
    r = ex1_normalize_text(reference).split()
    h = ex1_normalize_text(hypothesis).split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return ex2_levenshtein(r, h) / len(r)


def ex5_corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    """コーパス CER（マイクロ平均 = 総編集距離 / 総参照文字数）。"""
    tot_err = tot_chars = 0
    for r, h in zip(references, hypotheses):
        rn = ex1_normalize_text(r)
        hn = ex1_normalize_text(h)
        tot_err += ex2_levenshtein(rn, hn)
        tot_chars += len(rn)
    return 0.0 if tot_chars == 0 else tot_err / tot_chars


def ex6_anls(reference: str, hypothesis: str, threshold: float = 0.5) -> float:
    """ANLS = 1 - 編集距離/max(len)。閾値未満は 0 に丸める。"""
    r = ex1_normalize_text(reference)
    h = ex1_normalize_text(hypothesis)
    if len(r) == 0 and len(h) == 0:
        return 1.0
    denom = max(len(r), len(h))
    sim = 1.0 - ex2_levenshtein(r, h) / denom if denom > 0 else 0.0
    return sim if sim >= threshold else 0.0


def ex7_extract_donut_answer(sequence: str) -> str:
    """<s_answer>...</s_answer> の中身、無ければタグ除去後の文字列。"""
    m = re.search(r"<s_answer>(.*?)</s_answer>", sequence, re.S)
    if m:
        return m.group(1).strip()
    return re.sub(r"<.*?>", "", sequence).strip()


def ex8_qa_exact_match_accuracy(references: list[str], hypotheses: list[str]) -> float:
    """正規化後の exact-match 正答率。"""
    if not references:
        return 0.0
    hits = sum(
        ex1_normalize_text(r) == ex1_normalize_text(h)
        for r, h in zip(references, hypotheses)
    )
    return hits / len(references)


# =====================================================================
# 自己採点ランナー（exercises.py と同一のテスト）
# =====================================================================

def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        a = ex1_normalize_text("  Ｈｅｌｌｏ　 World  ")
        b = ex1_normalize_text("１２３")
        return a == "hello world" and b == "123", f"{a!r}, {b!r}"
    check("ex1_normalize_text", _c1)

    def _c2():
        ok = (ex2_levenshtein("kitten", "sitting") == 3
              and ex2_levenshtein("", "abc") == 3
              and ex2_levenshtein("abc", "abc") == 0
              and ex2_levenshtein(list("flaw"), list("lawn")) == 2)
        return ok, f"kitten/sitting={ex2_levenshtein('kitten', 'sitting')}"
    check("ex2_levenshtein", _c2)

    def _c3():
        v = ex3_cer("abc", "abd")
        z = ex3_cer("", "")
        return abs(v - 1 / 3) < 1e-9 and z == 0.0, f"CER={v:.4f}, empty={z}"
    check("ex3_cer", _c3)

    def _c4():
        v = ex4_wer("the cat sat", "the dog sat")
        z = ex4_wer("a b c", "a b c")
        return abs(v - 1 / 3) < 1e-9 and z == 0.0, f"WER={v:.4f}, same={z}"
    check("ex4_wer", _c4)

    def _c5():
        v = ex5_corpus_cer(["abc", "de"], ["abd", "de"])
        return abs(v - 0.2) < 1e-9, f"corpus_CER={v:.4f}"
    check("ex5_corpus_cer", _c5)

    def _c6():
        a = ex6_anls("12345", "12345")
        b = ex6_anls("paris", "pariss")
        c = ex6_anls("yes", "no")
        return a == 1.0 and abs(b - 5 / 6) < 1e-3 and c == 0.0, f"a={a}, b={b:.3f}, c={c}"
    check("ex6_anls", _c6)

    def _c7():
        a = ex7_extract_donut_answer(
            "<s_docvqa><s_question>q</s_question><s_answer> 980 usd </s_answer></s>")
        b = ex7_extract_donut_answer("<s_answer>12345</s_answer>")
        c = ex7_extract_donut_answer("hello </s>")
        return a == "980 usd" and b == "12345" and c == "hello", f"{a!r}, {b!r}, {c!r}"
    check("ex7_extract_donut_answer", _c7)

    def _c8():
        v = ex8_qa_exact_match_accuracy(
            ["12345", "Acme Corp", "980 USD"], ["12345", "acme corp", " 980  usd "])
        w = ex8_qa_exact_match_accuracy(["a", "b"], ["a", "c"])
        return v == 1.0 and abs(w - 0.5) < 1e-9, f"all={v}, half={w}"
    check("ex8_qa_exact_match_accuracy", _c8)

    print("=== 採点結果（模範解答）===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\n（想定外）模範解答が FAIL しました。")


if __name__ == "__main__":
    _grade()
