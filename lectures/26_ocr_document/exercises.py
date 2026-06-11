"""第26回 演習問題（OCRと文書理解 ― CER/WER/ANLS と Donut 出力の整形）。

使い方:
  1. 各関数の TODO を自分で実装する（最初は NotImplementedError だが、採点ランナーが
     拾うのでプロセスは落ちず、FAIL と表示されるだけ＝必ず exit 0）。
  2. 自己採点:  uv run python lectures/26_ocr_document/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 模範解答（実行すると全 PASS）:
        uv run python lectures/26_ocr_document/exercises_solutions.py

この8問は本モジュールの評価コア（モデル DL 不要・純計算）を易→難で1つずつ抜き出したもの:
  ex1: テキスト正規化 NFKC + 小文字化 + 空白圧縮（CER/WER の前処理）         … 04
  ex2: レーベンシュタイン編集距離 S+D+I（CER/WER の心臓部・DP）             … 04
  ex3: CER = 編集距離(文字) / 参照文字数                                     … 02/04
  ex4: WER = 編集距離(単語) / 参照単語数                                     … 04
  ex5: コーパス CER（マイクロ平均 = 総誤り / 総文字）                         … 02/04
  ex6: ANLS（正規化レーベンシュタイン類似度・DocVQA 標準指標）              … 03
  ex7: Donut 出力から答えを取り出す（<s_answer>...</s_answer> の整形）       … 03
  ex8: 正規化後の exact-match 正答率（QA 採点）                              … 03
"""

from __future__ import annotations

import re
import unicodedata


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_normalize_text(s: str) -> str:
    """演習1: OCR 評価のための正規化。NFKC → 小文字化 → 前後空白除去 → 連続空白を1つに。

    - unicodedata.normalize("NFKC", s) で全角英数・全角空白・互換文字を半角の正準形へ畳む
      （"Ｈｅｌｌｏ"→"Hello", "１２３"→"123", 全角空白→半角空白）。
    - その後 .lower() で小文字化し、strip() で前後空白除去。
    - 連続する空白（スペース/タブ/改行）は re.sub(r"\\s+", " ", ...) で1つにまとめる。
    例: "  Ｈｅｌｌｏ　 World  " -> "hello world"
    """
    # TODO: NFKC 正規化 → 小文字化 → 空白圧縮した文字列を返す
    raise NotImplementedError


def ex2_levenshtein(a, b) -> int:
    """演習2: レーベンシュタイン距離（置換 S + 削除 D + 挿入 I の最小回数）を返す。

    - a, b は任意のシーケンス（文字列でも単語リストでも可。要素の == だけ使う）。
    - DP: dp[i][j] = a[:i] と b[:j] を一致させる最小編集回数。
        dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (a[i-1]!=b[j-1]))
    - ヒント: 1 行ぶんの配列を使い回すとメモリ O(n) で書ける。
    例: ex2_levenshtein("kitten","sitting") == 3
    """
    # TODO: 編集距離を DP で計算して int で返す
    raise NotImplementedError


def ex3_cer(reference: str, hypothesis: str) -> float:
    """演習3: CER = 編集距離(文字) / 参照文字数 を返す（前処理は ex1 で揃える）。

    - 参照・予測の両方に ex1_normalize_text を適用してから ex2_levenshtein で距離を取る。
    - 参照が空文字なら、予測も空なら 0.0、そうでなければ 1.0 を返す（ゼロ割回避）。
    例: ex3_cer("abc","abd") == 1/3
    """
    # TODO: 正規化 → 編集距離 / 参照文字数 を返す
    raise NotImplementedError


def ex4_wer(reference: str, hypothesis: str) -> float:
    """演習4: WER = 編集距離(単語) / 参照単語数 を返す。

    - ex1 で正規化したのち .split() で空白トークン化し、語の列で ex2_levenshtein を取る。
    - 参照の語数が 0 なら、予測も空なら 0.0、そうでなければ 1.0。
    例: ex4_wer("the cat sat","the dog sat") == 1/3
    """
    # TODO: 単語列にして編集距離 / 参照単語数 を返す
    raise NotImplementedError


def ex5_corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    """演習5: コーパス全体の CER（マイクロ平均 = 総編集距離 / 総参照文字数）を返す。

    - サンプルごとの CER を平均（マクロ）するのではなく、全サンプルの編集距離の合計を
      全サンプルの参照文字数の合計で割る（短文の誤りが過大評価されない標準的な集計）。
    - 各サンプルは ex1 で正規化してから数える。総文字数が 0 なら 0.0。
    例: ex5_corpus_cer(["abc","de"],["abd","de"]) == 0.2
    """
    # TODO: 総編集距離 / 総参照文字数 を返す
    raise NotImplementedError


def ex6_anls(reference: str, hypothesis: str, threshold: float = 0.5) -> float:
    """演習6: ANLS（DocVQA 標準指標）の1ペア分を返す。

    - 正規化レーベンシュタイン類似度 sim = 1 - 編集距離 / max(len(ref), len(hyp))。
      （ex1 で正規化した文字列で測る）
    - sim が threshold(=0.5) 以上ならその値を、未満なら 0.0 を返す（別物には 0 点）。
    - 両方とも空文字なら 1.0。max が 0 なら 0.0。
    例: ex6_anls("12345","12345") == 1.0,  ex6_anls("yes","no") == 0.0
    """
    # TODO: 1 - 編集距離/max(len) を計算し、閾値で 0 に丸めて返す
    raise NotImplementedError


def ex7_extract_donut_answer(sequence: str) -> str:
    """演習7: Donut/DocVQA の生成列から答えを取り出す。

    - "<s_answer>...</s_answer>" があれば、その中身を strip して返す。
    - 無ければ、すべての "<...>" タグ（特殊トークン含む）を除去して strip した文字列を返す。
    - ヒント: re.search(r"<s_answer>(.*?)</s_answer>", sequence, re.S) と
             re.sub(r"<.*?>", "", sequence)。
    例: ex7_extract_donut_answer("<s_answer> 980 usd </s_answer></s>") == "980 usd"
    """
    # TODO: <s_answer> の中身、無ければタグ除去後の文字列を返す
    raise NotImplementedError


def ex8_qa_exact_match_accuracy(references: list[str], hypotheses: list[str]) -> float:
    """演習8: 正規化後の exact-match 正答率（一致した割合）を返す。

    - 各ペアを ex1_normalize_text で揃えてから完全一致を判定し、一致数 / 総数 を返す。
    - 件数が 0 なら 0.0。
    例: refs=["12345","Acme Corp"], hyps=["12345"," acme  corp "] -> 1.0
    """
    # TODO: 正規化して一致した割合を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

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
        b = ex6_anls("paris", "pariss")  # lev=1, max=6 -> 1-1/6=0.8333
        c = ex6_anls("yes", "no")        # sim 0 -> 0.0
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

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok
          else "\nまだ未達の演習があります。TODO を埋めましょう（解答: exercises_solutions.py）。")


# re / unicodedata は ex1/ex7 の実装で使う（あらかじめ import 済み）。
_ = (re, unicodedata)

if __name__ == "__main__":
    _grade()
