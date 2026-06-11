"""exercises_solutions.py — 30_face_detection_recognition 演習の模範解答（全 PASS）

    uv run python lectures/30_face_detection_recognition/exercises_solutions.py

exercises.py と同じ採点ハーネスを使い、全問が PASS することを確認する。
各解答は単一責務・早期 return を意識し、顔認識/クラスタリング評価の定義式を素直に写している。
"""

from __future__ import annotations

import numpy as np


# 問1: ベクトルの L2 正規化（ゼロは 0 除算を避けてそのまま返す）
def l2_normalize_vec(v):
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.sqrt(np.sum(v * v)))
    if norm < 1e-12:
        return v
    return v / norm


# 問2: コサイン類似度（L2 正規化してから内積）
def cosine_similarity(a, b) -> float:
    return float(np.dot(l2_normalize_vec(a), l2_normalize_vec(b)))


# 問3: コサイン距離
def cosine_distance(a, b) -> float:
    return float(1.0 - cosine_similarity(a, b))


# 問4: 照合ペアの genuine/impostor ラベル付け
def make_pair_labels(labels):
    labels = list(labels)
    out = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            out.append((i, j, bool(labels[i] == labels[j])))
    return out


# 問5: しきい値での FAR / TAR
def far_tar_at_threshold(scores, is_genuine, thr):
    scores = np.asarray(scores, dtype=np.float64)
    gen = np.asarray(is_genuine, dtype=bool)
    accept = scores >= thr
    tar = float(accept[gen].mean()) if gen.any() else 0.0  # 本人を通せた割合
    imp = ~gen
    far = float(accept[imp].mean()) if imp.any() else 0.0  # 別人を誤って通した割合
    return far, tar


# 問6: ROC 曲線から EER
def eer_from_curves(far, tar) -> float:
    far = np.asarray(far, dtype=np.float64)
    frr = 1.0 - np.asarray(tar, dtype=np.float64)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0)


# 問7: TAR@FAR
def tar_at_far(far, tar, target) -> float:
    far = np.asarray(far, dtype=np.float64)
    tar = np.asarray(tar, dtype=np.float64)
    mask = far <= target
    if not mask.any():
        return 0.0
    return float(tar[mask].max())


# 問8: rank-1 識別精度
def rank1_accuracy(sim, gallery_labels, probe_labels) -> float:
    sim = np.asarray(sim, dtype=np.float64)
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    nearest = sim.argmax(axis=1)  # 各プローブで最も似たギャラリー
    pred = gallery_labels[nearest]
    return float((pred == probe_labels).mean())


# 問9: purity
def purity(labels_true, labels_pred) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    total = len(labels_true)
    if total == 0:
        return 0.0
    correct = 0
    for c in np.unique(labels_pred):
        members = labels_true[labels_pred == c]
        if members.size:
            correct += np.bincount(members).max()
    return float(correct / total)


# 問10: 検出と正解の IoU（[x,y,w,h]）
def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 採点ハーネス（exercises.py と同一）
# ---------------------------------------------------------------------------
def _approx(got, exp, tol=1e-3):
    try:
        return bool(np.all(np.abs(np.asarray(got, float) - np.asarray(exp, float)) <= tol))
    except Exception:
        return False


def _run(name, fn):
    try:
        ok = fn()
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: 例外 {type(e).__name__}: {e}")
        return False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return bool(ok)


def _pairs_ok():
    out = make_pair_labels([0, 0, 1])
    norm = sorted((int(i), int(j), bool(g)) for i, j, g in out)
    return norm == [(0, 1, True), (0, 2, False), (1, 2, False)]


def _far_tar_ok():
    s = [0.9, 0.8, 0.4, 0.3]
    g = [True, True, False, False]
    return _approx(far_tar_at_threshold(s, g, 0.5), [0.0, 1.0]) and _approx(
        far_tar_at_threshold(s, g, 0.35), [0.5, 1.0]
    )


def _checks():
    return [
        (
            "問1 l2_normalize_vec",
            lambda: (
                _approx(l2_normalize_vec([3.0, 4.0]), [0.6, 0.8])
                and _approx(l2_normalize_vec([0.0, 0.0]), [0.0, 0.0])
            ),
        ),
        (
            "問2 cosine_similarity",
            lambda: (
                _approx(cosine_similarity([1, 0], [1, 1]), 0.70710678)
                and _approx(cosine_similarity([1, 0], [0, 1]), 0.0)
                and _approx(cosine_similarity([1, 0], [-1, 0]), -1.0)
            ),
        ),
        ("問3 cosine_distance", lambda: _approx(cosine_distance([1, 0], [1, 1]), 0.29289322)),
        ("問4 make_pair_labels", lambda: _pairs_ok()),
        ("問5 far_tar_at_threshold", lambda: _far_tar_ok()),
        (
            "問6 eer_from_curves",
            lambda: _approx(eer_from_curves([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]), 0.5),
        ),
        (
            "問7 tar_at_far",
            lambda: (
                _approx(tar_at_far([0.0, 0.01, 0.1, 1.0], [0.5, 0.8, 0.95, 1.0], 0.01), 0.8)
                and _approx(tar_at_far([0.5, 1.0], [0.9, 1.0], 0.01), 0.0)
            ),
        ),
        (
            "問8 rank1_accuracy",
            lambda: (
                _approx(rank1_accuracy([[0.9, 0.1], [0.2, 0.8]], [0, 1], [0, 1]), 1.0)
                and _approx(rank1_accuracy([[0.1, 0.9]], [0, 1], [0]), 0.0)
            ),
        ),
        ("問9 purity", lambda: _approx(purity([0, 0, 1, 1, 2], [0, 0, 0, 1, 1]), 0.6)),
        (
            "問10 iou_xywh",
            lambda: (
                _approx(iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]), 0.142857)
                and _approx(iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
            ),
        ),
    ]


def main():
    print("=" * 60)
    print("30_face_detection_recognition 演習 模範解答 自己採点")
    checks = _checks()
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    assert passed == len(checks), "模範解答が全 PASS しません（バグ）"
    print("全問 PASS。")


if __name__ == "__main__":
    main()
