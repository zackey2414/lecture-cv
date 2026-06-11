"""第25回 演習の模範解答（実行すると全 PASS になる）。

  uv run python lectures/25_vqa_vlm/exercises_solutions.py

まずは exercises.py を自力で解き、詰まったらここを読むこと。採点ロジックは
exercises.py と同一で、関数の中身だけを埋めてある。すべて純計算（モデルDL不要）。
"""

from __future__ import annotations

import re

import torch

ARTICLES = {"a", "an", "the"}
NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}


# =====================================================================
# 模範解答
# =====================================================================
def ex1_normalize_answer(ans: str) -> str:
    text = ans.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)  # 句読点・記号を空白へ
    words = []
    for w in text.split():
        w = NUMBER_MAP.get(w, w)  # 数詞 -> 数字
        if w in ARTICLES:  # 冠詞は捨てる
            continue
        words.append(w)
    return " ".join(words)


def ex2_answers_match(a: str, b: str) -> bool:
    return ex1_normalize_answer(a) == ex1_normalize_answer(b)


def ex3_vqa_accuracy_simple(pred: str, human_answers: list[str]) -> float:
    agree = sum(1 for h in human_answers if ex2_answers_match(pred, h))
    return min(agree / 3.0, 1.0)


def ex4_vqa_accuracy_vqav2(pred: str, human_answers: list[str]) -> float:
    p = ex1_normalize_answer(pred)
    hn = [ex1_normalize_answer(h) for h in human_answers]
    n = len(hn)
    if n == 0:
        return 0.0
    accs = []
    for i in range(n):
        others = hn[:i] + hn[i + 1 :]
        agree = sum(1 for h in others if h == p)
        accs.append(min(agree / 3.0, 1.0))
    return sum(accs) / n


def ex5_dataset_mean_vqa(preds: list[str], dataset: list[dict]) -> float:
    if not dataset:
        return 0.0
    scores = [ex4_vqa_accuracy_vqav2(p, d["human_answers"]) for p, d in zip(preds, dataset)]
    return sum(scores) / len(scores)


def ex6_build_chat_messages(image: object, question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


def ex7_answer_from_logits(logits: torch.Tensor, id2label: dict[int, str]) -> str:
    idx = int(logits.reshape(-1).argmax())
    return id2label[idx]


def ex8_box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def ex9_point_accuracy(points: list[tuple[float, float]], boxes: list[tuple[float, float, float, float]]) -> float:
    if not points:
        return 0.0
    hits = 0
    for (x, y), (x1, y1, x2, y2) in zip(points, boxes):
        if x1 <= x <= x2 and y1 <= y <= y2:
            hits += 1
    return hits / len(points)


# =====================================================================
# 自己採点ランナー（exercises.py と同一・常に exit 0）
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
        full = ex3_vqa_accuracy_simple("red", humans)
        one = ex3_vqa_accuracy_simple("red", ["red"] + ["blue"] * 9)
        zero = ex3_vqa_accuracy_simple("green", ["blue"] * 10)
        ok = abs(full - 1.0) < 1e-9 and abs(one - 1 / 3) < 1e-9 and zero == 0.0
        return ok, f"full={full}, one={one:.3f}, zero={zero}"
    check("ex3_vqa_accuracy_simple", _c3)

    def _c4():
        allred = ex4_vqa_accuracy_vqav2("red", ["red"] * 10)
        two = ex4_vqa_accuracy_vqav2("red", ["red", "red"] + ["blue"] * 8)
        expected_two = (2 * (1 / 3) + 8 * (2 / 3)) / 10
        empty = ex4_vqa_accuracy_vqav2("red", [])
        ok = abs(allred - 1.0) < 1e-9 and abs(two - expected_two) < 1e-9 and empty == 0.0
        return ok, f"allred={allred}, two={two:.3f}(exp {expected_two:.3f})"
    check("ex4_vqa_accuracy_vqav2", _c4)

    def _c5():
        ds = [{"human_answers": ["red"] * 10}, {"human_answers": ["2"] * 10}]
        good = ex5_dataset_mean_vqa(["red", "2"], ds)
        half = ex5_dataset_mean_vqa(["red", "9"], ds)
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
        a = ex7_answer_from_logits(torch.tensor([[0.1, 2.0, 0.3]]), id2label)
        b = ex7_answer_from_logits(torch.tensor([5.0, 0.0, 1.0]), id2label)
        return a == "blue" and b == "red", f"a={a}, b={b}"
    check("ex7_answer_from_logits", _c7)

    def _c8():
        same = ex8_box_iou((0, 0, 10, 10), (0, 0, 10, 10))
        none = ex8_box_iou((0, 0, 10, 10), (20, 20, 30, 30))
        half = ex8_box_iou((0, 0, 2, 2), (1, 0, 3, 2))
        ok = abs(same - 1.0) < 1e-9 and none == 0.0 and abs(half - 1 / 3) < 1e-9
        return ok, f"same={same}, none={none}, partial={half:.3f}"
    check("ex8_box_iou", _c8)

    def _c9():
        pts = [(5, 5), (15, 15), (1, 1)]
        boxes = [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)]
        acc = ex9_point_accuracy(pts, boxes)
        empty = ex9_point_accuracy([], [])
        ok = abs(acc - 2 / 3) < 1e-9 and empty == 0.0
        return ok, f"acc={acc:.3f}"
    check("ex9_point_accuracy", _c9)

    print("=== 採点結果（模範解答） ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nどこかの模範解答が壊れています。")


if __name__ == "__main__":
    _grade()
