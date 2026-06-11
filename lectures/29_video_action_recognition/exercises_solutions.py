"""第29回 演習問題の模範解答（実行すると全 PASS）。

実行: uv run python lectures/29_video_action_recognition/exercises_solutions.py

exercises.py の TODO をすべて埋めた版。単一責務・早期 return・日本語コメントで、
「なぜそう書くか」が分かるようにしている。採点ランナーは exercises.py と同じ。
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# 模範解答
# =====================================================================

def ex1_uniform_indices(total: int, num_frames: int) -> np.ndarray:
    """等間隔サンプリング: 端から端まで均等に num_frames 個。"""
    if total <= 0:
        raise ValueError("total は 1 以上が必要")
    # linspace は端点(0 と total-1)を必ず含むので「広く薄く」見るのに向く
    idx = np.linspace(0, total - 1, num=num_frames)
    return np.round(idx).astype(int)


def ex2_strided_indices(
    total: int, num_frames: int, frame_rate: int = 1, start: int | None = None
) -> np.ndarray:
    """clip_len + frame_rate(stride) サンプリング（クランプ付き）。"""
    span = (num_frames - 1) * frame_rate + 1  # 必要な区間長
    if start is None:
        start = max(0, (total - span) // 2)  # 中央寄せ
    idx = start + np.arange(num_frames) * frame_rate
    return np.clip(idx, 0, total - 1).astype(int)  # はみ出しは末尾にクランプ


def ex3_normalize_clip(clip01: np.ndarray, mean, std) -> np.ndarray:
    """チャンネル別正規化 (x - mean) / std。最後の軸が C。"""
    mean = np.asarray(mean, dtype=clip01.dtype)
    std = np.asarray(std, dtype=clip01.dtype)
    # 最後の軸(C=3)に沿って自動ブロードキャストされる
    return (clip01 - mean) / std


def ex4_clip_to_ncthw(clip_thwc: np.ndarray) -> np.ndarray:
    """(T,H,W,C) -> (1,C,T,H,W)。3D CNN は時間 T を持つ 5 次元入力を取る。"""
    cthw = np.transpose(clip_thwc, (3, 0, 1, 2))  # (C,T,H,W)
    return np.expand_dims(cthw, axis=0)  # バッチ次元を先頭に


def ex5_softmax_topk(logits: np.ndarray, k: int = 5):
    """logits -> softmax -> 上位 k の (indices, probs)。"""
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max())  # 数値安定化のため max を引く
    probs = e / e.sum()
    order = np.argsort(-probs)[:k]  # 確率の降順に上位 k
    return order, probs[order]


def ex6_topk_accuracy(logits: np.ndarray, gt_indices: np.ndarray, k: int = 1) -> float:
    """top-k accuracy = gt が各行の上位 k 予測に入る割合。"""
    logits = np.asarray(logits)
    gt_indices = np.asarray(gt_indices)
    if logits.shape[0] == 0:
        return 0.0
    topk = np.argsort(-logits, axis=1)[:, :k]  # 各行の上位 k クラス
    hits = [gt in row for gt, row in zip(gt_indices, topk)]
    return float(np.mean(hits))


def ex7_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """混同行列。行=正解, 列=予測。cm[g,p] に件数を足す。"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(np.asarray(gt), np.asarray(pred)):
        cm[int(g), int(p)] += 1
    return cm


def ex8_topk_agreement_vs_pseudo_gt(
    correct_logits: np.ndarray, variant_logits: np.ndarray, k: int = 1
) -> float:
    """正しい前処理の top-1 を基準に、別前処理の top-k 一致率を返す。"""
    pseudo_gt = np.asarray(correct_logits).argmax(axis=1)  # 基準ラベル
    return ex6_topk_accuracy(variant_logits, pseudo_gt, k=k)


# =====================================================================
# 自己採点ランナー（exercises.py と同一）
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
        a = ex1_uniform_indices(32, 8)
        b = ex1_uniform_indices(4, 8)
        ok = (list(a) == [0, 4, 9, 13, 18, 22, 27, 31]
              and list(b) == [0, 0, 1, 1, 2, 2, 3, 3]
              and np.asarray(a).dtype.kind == "i")
        return ok, f"{a.tolist()}"
    check("ex1_uniform_indices", _c1)

    def _c2():
        a = ex2_strided_indices(32, 8, 4)
        b = ex2_strided_indices(10, 8, 3, start=0)
        ok = (list(a) == [1, 5, 9, 13, 17, 21, 25, 29]
              and list(b) == [0, 3, 6, 9, 9, 9, 9, 9])
        return ok, f"center={a.tolist()}, clamp={b.tolist()}"
    check("ex2_strided_indices", _c2)

    def _c3():
        clip = np.full((2, 2, 3), 0.5, dtype=np.float64)
        out = ex3_normalize_clip(clip, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ok = out.shape == clip.shape and np.allclose(out, 0.0)
        out2 = ex3_normalize_clip(np.ones((1, 1, 3)), (0.0, 1.0, 2.0), (1.0, 1.0, 1.0))
        ok = ok and np.allclose(out2.ravel(), [1.0, 0.0, -1.0])
        return ok, f"zeros_ok, per-ch={out2.ravel().tolist()}"
    check("ex3_normalize_clip", _c3)

    def _c4():
        clip = np.zeros((4, 8, 10, 3))
        clip[0, 0, 0, 1] = 7.0
        out = ex4_clip_to_ncthw(clip)
        ok = out.shape == (1, 3, 4, 8, 10) and out[0, 1, 0, 0, 0] == 7.0
        return ok, f"shape={out.shape}"
    check("ex4_clip_to_ncthw", _c4)

    def _c5():
        idx, probs = ex5_softmax_topk(np.array([1.0, 3.0, 2.0, 0.0]), k=2)
        ok = list(idx) == [1, 2] and abs(float(probs[0]) - 0.6439) < 1e-3 and probs[0] >= probs[1]
        return ok, f"idx={idx.tolist()}, probs={np.round(probs,3).tolist()}"
    check("ex5_softmax_topk", _c5)

    def _c6():
        logits = np.array([[0.1, 0.9, 0.0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        a = ex6_topk_accuracy(logits, np.array([1, 0, 1]), k=1)
        b = ex6_topk_accuracy(logits, np.array([1, 0, 1]), k=2)
        ok = abs(a - 2 / 3) < 1e-9 and abs(b - 1.0) < 1e-9
        return ok, f"top1={a:.3f}, top2={b:.3f}"
    check("ex6_topk_accuracy", _c6)

    def _c7():
        cm = ex7_confusion_matrix(np.array([0, 1, 1]), np.array([0, 1, 0]), 2)
        ok = cm.tolist() == [[1, 0], [1, 1]]
        return ok, f"{cm.tolist()}"
    check("ex7_confusion_matrix", _c7)

    def _c8():
        correct = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        variant = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        v = ex8_topk_agreement_vs_pseudo_gt(correct, variant, k=1)
        ok = abs(v - 2 / 3) < 1e-9
        return ok, f"agreement={v:.3f}"
    check("ex8_topk_agreement_vs_pseudo_gt", _c8)

    print("=== 採点結果（模範解答） ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:32s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\n（模範解答が FAIL する場合はバグ報告を）")


if __name__ == "__main__":
    _grade()
