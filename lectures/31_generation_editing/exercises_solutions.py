"""exercises_solutions.py — 31_generation_editing 演習の模範解答（全 PASS）。

    uv run python lectures/31_generation_editing/exercises_solutions.py

exercises.py と同じ採点ハーネスを使い、全問が PASS することを確認する。
各解答は単一責務・早期 return を意識し、生成・復元・評価の核となる式を numpy だけで素直に写す。
"""

from __future__ import annotations

import numpy as np


# 問1: MSE（平均二乗誤差）
def mse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


# 問2: PSNR（完全一致は inf）
def psnr(a, b, max_val=255.0) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_val**2) / m))


# 問3: float 配列を [0,255] にクリップして uint8 化（四捨五入）
def to_uint8(x):
    x = np.asarray(x, dtype=np.float64)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


# 問4: min-max 正規化（[0,1] へ。定数配列は 0 を返す＝0 除算回避）
def minmax_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# 問5: コサイン類似度
def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# 問6: CLIPScore = w * max(cos, 0)
def clip_score(img_emb, txt_emb, w=2.5) -> float:
    return float(w * max(cosine_similarity(img_emb, txt_emb), 0.0))


# 問7: 大域 SSIM（単一窓・共分散形）。輝度×コントラスト×構造の積。
def ssim_global(a, b, max_val=255.0) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()  # 母分散（ddof=0）
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den)


# 問8: 1 次元ガウシアン間の Fréchet 距離（FID の 1 次元版）
#   d^2 = (m1-m2)^2 + (sqrt(v1)-sqrt(v2))^2
def frechet_distance_1d(m1, v1, m2, v2) -> float:
    return float((m1 - m2) ** 2 + (np.sqrt(v1) - np.sqrt(v2)) ** 2)


# 問9: 最近傍法の整数倍アップスケール（numpy の repeat）
def nearest_upscale(img, factor: int):
    img = np.asarray(img)
    out = np.repeat(img, factor, axis=0)
    out = np.repeat(out, factor, axis=1)
    return out


# 問10: α 合成（前景 fg を alpha で背景 bg に重ねる）。alpha は [0,1]。
def alpha_composite(fg, bg, alpha):
    fg = np.asarray(fg, dtype=np.float64)
    bg = np.asarray(bg, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == fg.ndim - 1:  # (H,W) のマスクを (H,W,1) に拡張
        alpha = alpha[..., None]
    out = fg * alpha + bg * (1.0 - alpha)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 採点ハーネス（exercises.py と同一）
# ---------------------------------------------------------------------------
def _approx(got, exp, tol=1e-3):
    try:
        g = np.asarray(got, dtype=float)
        e = np.asarray(exp, dtype=float)
        if g.shape != e.shape:
            return False
        both_inf = np.isinf(g) & np.isinf(e) & (np.sign(g) == np.sign(e))  # inf==inf を許容
        close = np.abs(g - e) <= tol
        return bool(np.all(close | both_inf))
    except Exception:  # noqa: BLE001
        return False


def _run(name, fn):
    try:
        ok = fn()
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: 例外 {type(e).__name__}: {e}")
        return False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return bool(ok)


def _ssim_checks():
    a = np.array([[0.0, 64.0], [128.0, 255.0]])
    b = a * 0.5 + 10.0
    return (
        _approx(ssim_global(a, a), 1.0)
        and _approx(ssim_global(a, b), ssim_global(b, a))  # 対称
        and _approx(ssim_global(a, b), 0.70090, tol=1e-3)  # 模範解答で算出した値
    )


def _upscale_checks():
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    up = nearest_upscale(img, 2)
    exp = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8)
    return up.shape == (4, 4) and bool(np.array_equal(up, exp))


def _alpha_checks():
    fg = np.full((2, 2, 3), 200.0)
    bg = np.zeros((2, 2, 3))
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    out = alpha_composite(fg, bg, a)
    return out[0, 0, 0] == 200 and out[0, 1, 0] == 0 and out[1, 0, 0] == 100


def _checks():
    return [
        (
            "問1 mse",
            lambda: _approx(mse([0, 0], [3, 4]), 12.5) and _approx(mse([1, 1], [1, 1]), 0.0),
        ),
        (
            "問2 psnr",
            lambda: (
                _approx(psnr(np.zeros(4), np.zeros(4)), float("inf"))
                and _approx(psnr([0.0], [255.0]), 0.0)
            ),
        ),
        (
            "問3 to_uint8",
            lambda: bool(
                np.array_equal(
                    to_uint8([-5.0, 0.4, 0.6, 300.0]), np.array([0, 0, 1, 255], np.uint8)
                )
            ),
        ),
        (
            "問4 minmax_normalize",
            lambda: (
                _approx(minmax_normalize([0.0, 5.0, 10.0]), [0.0, 0.5, 1.0])
                and _approx(minmax_normalize([7.0, 7.0]), [0.0, 0.0])
            ),
        ),
        (
            "問5 cosine_similarity",
            lambda: (
                _approx(cosine_similarity([1, 0], [1, 1]), 0.70710678)
                and _approx(cosine_similarity([1, 0], [-1, 0]), -1.0)
            ),
        ),
        (
            "問6 clip_score",
            lambda: (
                _approx(clip_score([1, 0], [1, 1]), 1.76776695)
                and _approx(clip_score([1, 0], [-1, 0]), 0.0)  # 負の cos は 0 にクリップ
            ),
        ),
        ("問7 ssim_global", _ssim_checks),
        (
            "問8 frechet_distance_1d",
            lambda: (
                _approx(frechet_distance_1d(0, 1, 3, 4), 10.0)
                and _approx(frechet_distance_1d(2, 9, 2, 9), 0.0)
            ),
        ),
        ("問9 nearest_upscale", _upscale_checks),
        ("問10 alpha_composite", _alpha_checks),
    ]


def main():
    print("=" * 60)
    print("31_generation_editing 演習 模範解答 自己採点")
    checks = _checks()
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    assert passed == len(checks), "模範解答が全 PASS しません（バグ）"
    print("全問 PASS。")


if __name__ == "__main__":
    main()
