"""exercises.py — 31_generation_editing 演習（10 問・易→難・自己採点つき）。

各 TODO 関数を実装して保存し、実行すると自動採点されます:

    uv run python lectures/31_generation_editing/exercises.py

- 未実装でも **exit 0** で終了します（PASS/FAIL の一覧が出るだけ）。
- 模範解答は exercises_solutions.py（実行すると全 PASS）。
- ねらい: 生成・復元・編集・評価の核を numpy だけで書けるようにする。
    MSE → PSNR → uint8 化 → 正規化 → コサイン類似 → CLIPScore → 大域 SSIM →
    Fréchet 距離(FID の 1 次元版) → 最近傍アップスケール → α 合成。
- ここで身につく式は、torchmetrics / diffusers の中で実際に使われているものと同じ。
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 問1（易）: MSE（平均二乗誤差）
#   a, b（同形状）の平均二乗誤差を返す。復元・超解像の基本誤差。
# ============================================================================
def mse(a, b) -> float:
    # TODO: float 化して mean((a-b)^2) を返す。
    raise NotImplementedError


# ============================================================================
# 問2（易）: PSNR
#   PSNR = 10*log10(max_val^2 / MSE)。完全一致（MSE≈0）は float("inf") を返す。
# ============================================================================
def psnr(a, b, max_val=255.0) -> float:
    # TODO: 問1 を使う。MSE が極小なら inf。そうでなければ 10*log10(max_val^2/MSE)。
    raise NotImplementedError


# ============================================================================
# 問3（易）: float → uint8
#   実数配列を [0,255] にクリップし、四捨五入して uint8 配列で返す（画像保存の定番処理）。
# ============================================================================
def to_uint8(x):
    # TODO: np.clip(np.round(x), 0, 255).astype(np.uint8)
    raise NotImplementedError


# ============================================================================
# 問4（易）: min-max 正規化
#   x を [0,1] に線形正規化して返す。定数配列（max==min）は 0 除算を避けて全 0 を返す。
#   深度マップや特徴の可視化前処理で頻出。
# ============================================================================
def minmax_normalize(x):
    # TODO: lo,hi を取り、hi-lo がほぼ 0 なら zeros、そうでなければ (x-lo)/(hi-lo)。
    raise NotImplementedError


# ============================================================================
# 問5（中）: コサイン類似度
#   ベクトル a, b の内積 / (|a||b|)。どちらかがゼロベクトルなら 0.0。
# ============================================================================
def cosine_similarity(a, b) -> float:
    # TODO: ノルムを計算し、0 除算を避けつつ dot/(na*nb) を返す。
    raise NotImplementedError


# ============================================================================
# 問6（中）: CLIPScore
#   画像埋め込みとテキスト埋め込みのコサイン類似度を w 倍し、負なら 0 にクリップ。
#   CLIPScore = w * max(cos, 0)（慣例 w=2.5）。生成画像のプロンプト整合度。
# ============================================================================
def clip_score(img_emb, txt_emb, w=2.5) -> float:
    # TODO: 問5 を使って cos を求め、w*max(cos,0) を返す。
    raise NotImplementedError


# ============================================================================
# 問7（中）: 大域 SSIM（単一窓・共分散形）
#   a, b 全体の平均 mu・分散 var(母分散 ddof=0)・共分散 cov から
#   SSIM = ((2*mu_a*mu_b + C1)(2*cov + C2)) / ((mu_a^2+mu_b^2 + C1)(var_a+var_b + C2))。
#   C1=(0.01*max_val)^2, C2=(0.03*max_val)^2。a==b なら 1.0 になるはず。
# ============================================================================
def ssim_global(a, b, max_val=255.0) -> float:
    # TODO: ravel して mean/var/cov を計算し、上式を組み立てる。
    raise NotImplementedError


# ============================================================================
# 問8（中）: 1 次元ガウシアン間の Fréchet 距離（FID の 1 次元版）
#   平均 m・分散 v をもつ 2 つの 1 次元ガウシアンの距離:
#     d^2 = (m1-m2)^2 + (sqrt(v1)-sqrt(v2))^2
#   多次元 FID の Tr(...) 項が 1 次元ではこの形に簡約される。
# ============================================================================
def frechet_distance_1d(m1, v1, m2, v2) -> float:
    # TODO: 上式をそのまま返す。
    raise NotImplementedError


# ============================================================================
# 問9（難）: 最近傍法の整数倍アップスケール
#   (H,W) または (H,W,C) 画像を縦横それぞれ factor 倍に拡大して返す（最近傍＝画素複製）。
#   超解像の最弱ベースライン。np.repeat を 2 回使う。
# ============================================================================
def nearest_upscale(img, factor: int):
    # TODO: axis=0 と axis=1 に np.repeat を適用する。
    raise NotImplementedError


# ============================================================================
# 問10（難）: α 合成
#   前景 fg を alpha（[0,1]）で背景 bg に重ねる: out = fg*alpha + bg*(1-alpha)。
#   alpha が (H,W) なら (H,W,1) に拡張してチャネルへブロードキャストする。
#   結果は [0,255] にクリップして uint8。背景除去後の合成で使う。
# ============================================================================
def alpha_composite(fg, bg, alpha):
    # TODO: float 化 → 必要なら alpha を [...,None] に拡張 → 合成 → uint8 化。
    raise NotImplementedError


# ============================================================================
# 自己採点ハーネス（編集不要）。未実装は FAIL だが exit 0 で終了する。
# ============================================================================
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
    except NotImplementedError:
        print(f"  [TODO] {name}: 未実装")
        return False
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
    print("31_generation_editing 演習 自己採点")
    checks = _checks()
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    if passed == len(checks):
        print("全問正解！ 生成・復元・評価の核を numpy だけで書けています。")


if __name__ == "__main__":
    main()
