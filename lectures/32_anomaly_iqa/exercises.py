"""exercises.py — 32_anomaly_iqa 演習（9 問・易→難・自己採点つき）

各 TODO 関数を実装して保存し、実行すると自動採点されます:

    uv run python lectures/32_anomaly_iqa/exercises.py

- 未実装でも **exit 0** で終了します（PASS/FAIL の一覧が出るだけ）。
- 模範解答は exercises_solutions.py（実行すると全 PASS）。
- ねらい: PSNR → 正規化 → ガウシアン推定 → マハラノビス距離（単点/一括）→ image スコア →
  AUROC（順位ベース自前実装）→ SSIM（統計量から）→ coreset(最遠点サンプリング) まで、
  異常検知と IQA の核を **ライブラリの AUROC/SSIM に頼らず**自分の手で書けるようにする。
  使ってよいのは numpy のみ。
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 問1（易）: MSE から PSNR
#   PSNR = 10 * log10(max_val^2 / mse)。mse が 0 に極めて近いときは 99.0 を返す。
# ============================================================================
def mse_to_psnr(mse: float, max_val: float = 1.0) -> float:
    # TODO: mse<=1e-12 なら 99.0、そうでなければ 10*log10(max_val**2/mse)。
    raise NotImplementedError


# ============================================================================
# 問2（易）: min-max 正規化
#   x を [0,1] に線形正規化して返す。最大と最小が等しい（定数）ときは全 0 を返す。
# ============================================================================
def min_max_normalize(x):
    # TODO: lo,hi=min,max。hi-lo がほぼ 0 なら zeros_like、そうでなければ (x-lo)/(hi-lo)。
    raise NotImplementedError


# ============================================================================
# 問3（易）: 単点のマハラノビス距離
#   x, mean (いずれも (D,))、inv_cov (D,D) から sqrt((x-m)^T inv_cov (x-m)) を返す。
# ============================================================================
def mahalanobis(x, mean, inv_cov) -> float:
    # TODO: d = x-mean。sqrt(d @ inv_cov @ d) を返す（負になり得ないが clip して安全に）。
    raise NotImplementedError


# ============================================================================
# 問4（中）: ガウシアン推定（PaDiM の 1 位置分）
#   X (N,D) から 平均 (D,) と 正則化済み共分散 (D,D) を返す。
#   共分散は不偏推定（1/(N-1)）に eps*I を足す。返り値は (mean, cov)。
# ============================================================================
def fit_gaussian(X, eps: float = 0.01):
    # TODO: mean=X.mean(0)。centered=X-mean。cov=centered.T@centered/(N-1)+eps*I。
    raise NotImplementedError


# ============================================================================
# 問5（中）: 位置別マハラノビス距離（一括）
#   emb (P,D)（P 個の位置ベクトル）, mean (D,), inv_cov (D,D) から、
#   各位置のマハラノビス距離 (P,) を返す。einsum を使うと簡潔。
# ============================================================================
def batched_mahalanobis(emb, mean, inv_cov):
    # TODO: diff=emb-mean (P,D)。left=diff@inv_cov (P,D)。m2=sum(left*diff,axis=1)。sqrt(clip)。
    raise NotImplementedError


# ============================================================================
# 問6（中）: 異常マップ → image スコア
#   anomaly_map (H,W) の最大値を image-level スコアとして返す（最も異常な点を採用）。
# ============================================================================
def image_level_score(anomaly_map) -> float:
    # TODO: map 全体の最大値を float で返す。
    raise NotImplementedError


# ============================================================================
# 問7（中）: AUROC を順位ベースで自前実装（Mann-Whitney U）
#   scores (N,), labels (N,) 0/1。AUROC = 正例が負例より高くスコアされる確率。
#   公式: AUROC = (sum(rank(正例)) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
#   同点は平均順位で扱う。正例か負例が 0 件なら 0.5 を返す。
# ============================================================================
def auroc(scores, labels) -> float:
    # TODO: rankdata 相当（同点は平均順位）を自前で作り、上の公式を適用。
    raise NotImplementedError


# ============================================================================
# 問8（難）: 統計量から SSIM（単一窓）
#   2 つの窓の統計量 mu_x, mu_y, var_x, var_y, cov_xy から SSIM を返す。
#   SSIM = ((2 mu_x mu_y + C1)(2 cov_xy + C2)) / ((mu_x^2+mu_y^2+C1)(var_x+var_y+C2))
#   既定 C1=(0.01)^2, C2=(0.03)^2。
# ============================================================================
def ssim_from_stats(mu_x, mu_y, var_x, var_y, cov_xy, c1=1e-4, c2=9e-4) -> float:
    # TODO: 上の式をそのまま実装して float を返す。
    raise NotImplementedError


# ============================================================================
# 問9（難）: 最遠点サンプリング（PatchCore の coreset）
#   points (M,D) から n_keep 個を貪欲に選ぶ。最初は start を選び、以降は
#   「既に選んだ点集合への最小ユークリッド距離が最大の点」を順に追加する。
#   返り値は選んだ行インデックスの list（長さ n_keep、選んだ順）。
# ============================================================================
def farthest_point_sampling(points, n_keep: int, start: int = 0):
    # TODO: selected=[start]。min_d=各点→startの距離。
    #       n_keep-1 回: idx=argmax(min_d) を選び、min_d=minimum(min_d, 各点→idxの距離)。
    raise NotImplementedError


# ============================================================================
# 自己採点ハーネス（編集不要）。未実装は FAIL だが exit 0 で終了する。
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
    # 平均=(1,1)。各次元の不偏分散=4/3、相関0 → 対角 [[4/3,0],[0,4/3]]。
    return _approx(mean, [1.0, 1.0]) and _approx(cov, [[4 / 3, 0.0], [0.0, 4 / 3]])


def _maha_batched_ok():
    emb = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 2.0]])
    mean = np.array([0.0, 0.0])
    inv_cov = np.eye(2)
    # 単位共分散ならユークリッド距離: [1, 0, sqrt(8)]
    return _approx(batched_mahalanobis(emb, mean, inv_cov), [1.0, 0.0, np.sqrt(8.0)])


def _auroc_ok():
    # 完全分離（正例が全て高い）→ 1.0
    a = auroc([0.1, 0.2, 0.35, 0.8], [0, 0, 1, 1])
    # 反転 → 0.0、同点ばかり → 0.5
    b = auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
    c = auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1])
    return _approx(a, 1.0) and _approx(b, 0.0) and _approx(c, 0.5)


def _fps_ok():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.0, 10.0]])
    sel = list(farthest_point_sampling(pts, 3, start=0))
    # start=0 → 最遠は index3(=(10,10)) → 次の最遠は index2(=(10,0))
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
    print("32_anomaly_iqa 演習 自己採点")
    checks = _checks()
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    if passed == len(checks):
        print("全問正解！ 異常検知(PaDiM/PatchCore)と IQA の核を自力で書けています。")


if __name__ == "__main__":
    main()
