"""exercises_solutions.py — 32_anomaly_iqa 演習の模範解答（実行すると全 PASS）。

    uv run python lectures/32_anomaly_iqa/exercises_solutions.py

exercises.py と同じ自己採点ハーネスを使い、各 TODO を埋めた版です。使うのは numpy のみ。
"""

from __future__ import annotations

import numpy as np


# 問1: MSE → PSNR
def mse_to_psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((max_val**2) / mse))


# 問2: min-max 正規化
def min_max_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# 問3: 単点のマハラノビス距離
def mahalanobis(x, mean, inv_cov) -> float:
    d = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    m2 = float(d @ np.asarray(inv_cov, dtype=np.float64) @ d)
    return float(np.sqrt(max(m2, 0.0)))


# 問4: ガウシアン推定（PaDiM の 1 位置分）
def fit_gaussian(X, eps: float = 0.01):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mean = X.mean(axis=0)
    centered = X - mean
    cov = centered.T @ centered / max(1, n - 1)
    cov = cov + eps * np.eye(d)
    return mean, cov


# 問5: 位置別マハラノビス距離（一括）
def batched_mahalanobis(emb, mean, inv_cov):
    emb = np.asarray(emb, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    inv_cov = np.asarray(inv_cov, dtype=np.float64)
    diff = emb - mean  # (P, D)
    left = diff @ inv_cov  # (P, D)
    m2 = np.einsum("pd,pd->p", left, diff)  # (P,)
    return np.sqrt(np.clip(m2, 0.0, None))


# 問6: 異常マップ → image スコア
def image_level_score(anomaly_map) -> float:
    return float(np.asarray(anomaly_map, dtype=np.float64).max())


# 問7: AUROC（Mann-Whitney U / 順位ベース）
def auroc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # 同点は平均順位（1始まり）。
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 同点グループを平均順位に置換
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0  # 1始まりの平均順位
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# 問8: 統計量から SSIM
def ssim_from_stats(mu_x, mu_y, var_x, var_y, cov_xy, c1=1e-4, c2=9e-4) -> float:
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / den)


# 問9: 最遠点サンプリング（PatchCore の coreset）
def farthest_point_sampling(points, n_keep: int, start: int = 0):
    pts = np.asarray(points, dtype=np.float64)
    m = len(pts)
    n_keep = min(n_keep, m)
    selected = [int(start)]
    min_d = np.linalg.norm(pts - pts[start], axis=1)
    for _ in range(1, n_keep):
        idx = int(np.argmax(min_d))
        selected.append(idx)
        d = np.linalg.norm(pts - pts[idx], axis=1)
        min_d = np.minimum(min_d, d)
    return selected


# ============================================================================
# 自己採点ハーネス（exercises.py と同一）。
# ============================================================================
def _approx(got, exp, tol=1e-3):
    try:
        return bool(np.all(np.abs(np.asarray(got, float) - np.asarray(exp, float)) <= tol))
    except Exception:
        return False


def _run(name, fn):
    try:
        ok = fn()
    except NotImplementedError:
        print(f"  [TODO] {name}: 未実装")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: 例外 {type(e).__name__}: {e}")
        return False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return bool(ok)


def _gauss_ok():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    mean, cov = fit_gaussian(X, eps=0.0)
    return _approx(mean, [1.0, 1.0]) and _approx(cov, [[4 / 3, 0.0], [0.0, 4 / 3]])


def _maha_batched_ok():
    emb = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 2.0]])
    mean = np.array([0.0, 0.0])
    inv_cov = np.eye(2)
    return _approx(batched_mahalanobis(emb, mean, inv_cov), [1.0, 0.0, np.sqrt(8.0)])


def _auroc_ok():
    a = auroc([0.1, 0.2, 0.35, 0.8], [0, 0, 1, 1])
    b = auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
    c = auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1])
    return _approx(a, 1.0) and _approx(b, 0.0) and _approx(c, 0.5)


def _fps_ok():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.0, 10.0]])
    sel = list(farthest_point_sampling(pts, 3, start=0))
    return sel == [0, 3, 2]


def _checks():
    return [
        (
            "問1 mse_to_psnr",
            lambda: _approx(mse_to_psnr(0.01, 1.0), 20.0) and _approx(mse_to_psnr(0.0), 99.0),
        ),
        (
            "問2 min_max_normalize",
            lambda: (
                _approx(min_max_normalize([0.0, 5.0, 10.0]), [0.0, 0.5, 1.0])
                and _approx(min_max_normalize([3.0, 3.0]), [0.0, 0.0])
            ),
        ),
        (
            "問3 mahalanobis",
            lambda: (
                _approx(mahalanobis([3.0, 4.0], [0.0, 0.0], np.eye(2)), 5.0)
                and _approx(mahalanobis([1.0, 0.0], [0.0, 0.0], np.diag([4.0, 1.0])), 2.0)
            ),
        ),
        ("問4 fit_gaussian", _gauss_ok),
        ("問5 batched_mahalanobis", _maha_batched_ok),
        (
            "問6 image_level_score",
            lambda: _approx(image_level_score([[0.1, 0.9], [0.3, 0.2]]), 0.9),
        ),
        ("問7 auroc", _auroc_ok),
        (
            "問8 ssim_from_stats",
            lambda: (
                _approx(ssim_from_stats(0.5, 0.5, 0.04, 0.04, 0.04), 1.0)
                and _approx(ssim_from_stats(0.5, 0.5, 0.04, 0.04, 0.0), 0.011125, tol=1e-4)
            ),
        ),
        ("問9 farthest_point_sampling", _fps_ok),
    ]


def main():
    print("=" * 60)
    print("32_anomaly_iqa 演習 模範解答 自己採点")
    checks = _checks()
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    if passed == len(checks):
        print("全問正解！ 異常検知(PaDiM/PatchCore)と IQA の核を自力で書けています。")


if __name__ == "__main__":
    main()
