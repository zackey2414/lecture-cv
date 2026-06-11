"""第25回 演習問題（VQA / 軽量VLM / 評価 / グラウンディング）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ＝常に exit 0）。
  2. 自己採点:  uv run python lectures/25_vqa_vlm/exercises.py
     全問 pass すれば "ALL PASS 🎉" と表示される。
  3. 詰まったら模範解答を実行して挙動を確認:
        uv run python lectures/25_vqa_vlm/exercises_solutions.py

この9問はモデルDL不要の「純計算」だけで、本モジュールの核を1つずつ抜き出したもの:
  ex1 answer 正規化（VQAv2の前処理）          … 03 / mini
  ex2 正規化つき一致判定                       … 03
  ex3 簡易 VQA accuracy = min(#agree/3,1)       … 03
  ex4 公式 VQAv2 accuracy（leave-one-out 平均）  … 03
  ex5 データセット平均スコア                   … 03 / mini
  ex6 チャットメッセージ構築（画像をcontentに）… 02
  ex7 logits → ラベル（ViLT の decode）         … 01
  ex8 box の IoU                               … mini
  ex9 グラウンディングの point accuracy         … mini
"""

from __future__ import annotations

import re  # noqa: F401  ex1 の正規化（re.sub）で使う想定。未実装段階では未使用に見える。

import torch

# 正規化で使う小さな辞書（ex1 で使う）。
ARTICLES = {"a", "an", "the"}
NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}


# =====================================================================
# 演習（ここを実装する）
# =====================================================================
def ex1_normalize_answer(ans: str) -> str:
    """演習1: VQAv2 流に回答文字列を正規化する（簡略版）。

    手順:
      1) 小文字化して前後の空白を除去
      2) 句読点・記号を空白に置き換える（re.sub(r"[^\\w\\s]", " ", ...)）
      3) 空白で分割し、各語について:
           - NUMBER_MAP にあれば数字へ（"two" -> "2"）
           - ARTICLES（a/an/the）なら捨てる
      4) 残った語を半角スペース1個で連結して返す
    例: "A Red." -> "red",  "two" -> "2",  "YES!" -> "yes"
    """
    # TODO: 上の手順どおりに実装する
    raise NotImplementedError


def ex2_answers_match(a: str, b: str) -> bool:
    """演習2: 2つの回答が「正規化後に」一致するかを返す。

    ex1_normalize_answer を使い、正規化してから == で比較する。
    例: ("Red.", "red") -> True,  ("blue", "red") -> False
    """
    # TODO: ex1 を使って正規化し、等しいか返す
    raise NotImplementedError


def ex3_vqa_accuracy_simple(pred: str, human_answers: list[str]) -> float:
    """演習3: 簡易 VQA accuracy = min(予測と一致した人間の数 / 3, 1.0)。

    - pred と各 human を正規化してから一致を数える（ex2 を使ってよい）。
    - 一致数 / 3 を 1.0 で頭打ちにする（min(..., 1.0)）。
    例: pred="red", humans=["red"]*7+["crimson","maroon","red"] -> 1.0（8人一致）
        pred="red", humans=["blue","red","green",...]（1人一致） -> 1/3
    """
    # TODO: 一致数を数え、min(agree/3, 1.0) を返す
    raise NotImplementedError


def ex4_vqa_accuracy_vqav2(pred: str, human_answers: list[str]) -> float:
    """演習4: 公式 VQAv2 accuracy（10人の leave-one-out 平均）。

    - 各 i について「i 番目を抜いた残り」に対し min(一致数/3, 1) を計算し、
      全 i の平均を返す（人数 n で割る）。
    - human_answers が空なら 0.0。
    ヒント: others = humans[:i] + humans[i+1:]
    """
    # TODO: leave-one-out で各人を抜いた accuracy を平均する
    raise NotImplementedError


def ex5_dataset_mean_vqa(preds: list[str], dataset: list[dict]) -> float:
    """演習5: データセット全体の平均 VQAv2 accuracy を返す。

    - dataset[i] は {"human_answers": [...]} を持つ dict。
    - 各設問で ex4_vqa_accuracy_vqav2(preds[i], dataset[i]["human_answers"]) を計算し、
      その平均（単純平均）を返す。
    - dataset が空なら 0.0。
    """
    # TODO: 各設問の VQAv2 を平均する
    raise NotImplementedError


def ex6_build_chat_messages(image: object, question: str) -> list[dict]:
    """演習6: チャット VLM 用メッセージ（画像を content に埋め込む正準形）を作る。

    返り値は次の構造（apply_chat_template が受け取れる形）:
      [{"role": "user",
        "content": [{"type": "image", "image": image},
                    {"type": "text",  "text": question}]}]
    ★画像は別引数ではなく content の中に入れるのが v5 の正準。
    """
    # TODO: 上の構造の list を返す
    raise NotImplementedError


def ex7_answer_from_logits(logits: torch.Tensor, id2label: dict[int, str]) -> str:
    """演習7: 分類型 VQA(ViLT) の出力を答えに変換する。

    - logits: 形 (1, num_answers) または (num_answers,) のテンソル。
    - 最大スコアのインデックスを取り（argmax）、id2label でラベル文字列に変換して返す。
    ヒント: int(logits.reshape(-1).argmax())
    """
    # TODO: argmax のインデックスを id2label で引いて返す
    raise NotImplementedError


def ex8_box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """演習8: 2つの箱 (x1,y1,x2,y2) の IoU を返す。

    - 交差領域の幅・高さは負にならないよう max(0, ...) でクランプする。
    - union = areaA + areaB - inter。union が 0 なら 0.0 を返す（ゼロ割回避）。
    例: ((0,0,10,10),(0,0,10,10)) -> 1.0,  ((0,0,10,10),(20,20,30,30)) -> 0.0
    """
    # TODO: 交差面積と和集合面積から IoU を計算する
    raise NotImplementedError


def ex9_point_accuracy(points: list[tuple[float, float]], boxes: list[tuple[float, float, float, float]]) -> float:
    """演習9: グラウンディングの point accuracy を返す。

    - points[i] が boxes[i] の内側（境界含む）にある割合を返す。
    - points が空なら 0.0。
    内側判定: x1 <= x <= x2 かつ y1 <= y <= y2。
    """
    # TODO: 各点が対応する箱の中にあるかを数え、割合を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（常に exit 0）
# =====================================================================
def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        cases = {"A Red.": "red", "two": "2", "YES!": "yes", "  the  cat ": "cat", "an  Apple": "apple"}
        got = {k: ex1_normalize_answer(k) for k in cases}
        ok = all(got[k] == v for k, v in cases.items())
        return ok, f"{got}"
    check("ex1_normalize_answer", _c1)

    def _c2():
        ok = ex2_answers_match("Red.", "red") and ex2_answers_match("two", "2") and not ex2_answers_match("blue", "red")
        return ok, "Red.==red / two==2 / blue!=red"
    check("ex2_answers_match", _c2)

    def _c3():
        humans = ["red"] * 7 + ["crimson", "maroon", "red"]
        full = ex3_vqa_accuracy_simple("red", humans)          # 8人一致 -> 1.0
        one = ex3_vqa_accuracy_simple("red", ["red"] + ["blue"] * 9)  # 1人 -> 1/3
        zero = ex3_vqa_accuracy_simple("green", ["blue"] * 10)         # 0
        ok = abs(full - 1.0) < 1e-9 and abs(one - 1 / 3) < 1e-9 and zero == 0.0
        return ok, f"full={full}, one={one:.3f}, zero={zero}"
    check("ex3_vqa_accuracy_simple", _c3)

    def _c4():
        # 全員 red: どの人を抜いても残り9人中9人一致 -> 1.0
        allred = ex4_vqa_accuracy_vqav2("red", ["red"] * 10)
        # 2人だけ red: 抜く人が red(2回) -> 残り1人, それ以外(8回) -> 残り2人。min(2/3 or 1/3)
        two = ex4_vqa_accuracy_vqav2("red", ["red", "red"] + ["blue"] * 8)
        expected_two = (2 * (1 / 3) + 8 * (2 / 3)) / 10
        empty = ex4_vqa_accuracy_vqav2("red", [])
        ok = abs(allred - 1.0) < 1e-9 and abs(two - expected_two) < 1e-9 and empty == 0.0
        return ok, f"allred={allred}, two={two:.3f}(exp {expected_two:.3f})"
    check("ex4_vqa_accuracy_vqav2", _c4)

    def _c5():
        ds = [{"human_answers": ["red"] * 10}, {"human_answers": ["2"] * 10}]
        good = ex5_dataset_mean_vqa(["red", "2"], ds)   # 両方満点 -> 1.0
        half = ex5_dataset_mean_vqa(["red", "9"], ds)   # 片方0 -> 0.5
        empty = ex5_dataset_mean_vqa([], [])
        ok = abs(good - 1.0) < 1e-9 and abs(half - 0.5) < 1e-9 and empty == 0.0
        return ok, f"good={good}, half={half}"
    check("ex5_dataset_mean_vqa", _c5)

    def _c6():
        sentinel = object()
        msgs = ex6_build_chat_messages(sentinel, "What color?")
        ok = (
            isinstance(msgs, list) and len(msgs) == 1 and msgs[0]["role"] == "user"
            and isinstance(msgs[0]["content"], list)
            and any(c.get("type") == "image" and c.get("image") is sentinel for c in msgs[0]["content"])
            and any(c.get("type") == "text" and c.get("text") == "What color?" for c in msgs[0]["content"])
        )
        return ok, "user / content に image(本体) と text(question)"
    check("ex6_build_chat_messages", _c6)

    def _c7():
        id2label = {0: "red", 1: "blue", 2: "green"}
        a = ex7_answer_from_logits(torch.tensor([[0.1, 2.0, 0.3]]), id2label)  # -> blue
        b = ex7_answer_from_logits(torch.tensor([5.0, 0.0, 1.0]), id2label)    # 1次元 -> red
        return a == "blue" and b == "red", f"a={a}, b={b}"
    check("ex7_answer_from_logits", _c7)

    def _c8():
        same = ex8_box_iou((0, 0, 10, 10), (0, 0, 10, 10))
        none = ex8_box_iou((0, 0, 10, 10), (20, 20, 30, 30))
        half = ex8_box_iou((0, 0, 2, 2), (1, 0, 3, 2))   # inter=2, union=6 -> 1/3
        ok = abs(same - 1.0) < 1e-9 and none == 0.0 and abs(half - 1 / 3) < 1e-9
        return ok, f"same={same}, none={none}, partial={half:.3f}"
    check("ex8_box_iou", _c8)

    def _c9():
        pts = [(5, 5), (15, 15), (1, 1)]
        boxes = [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)]  # 1番目in, 2番目out, 3番目in
        acc = ex9_point_accuracy(pts, boxes)
        empty = ex9_point_accuracy([], [])
        ok = abs(acc - 2 / 3) < 1e-9 and empty == 0.0
        return ok, f"acc={acc:.3f}"
    check("ex9_point_accuracy", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


if __name__ == "__main__":
    _grade()
