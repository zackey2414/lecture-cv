"""exercises.py — 44_embedding_clustering 演習（8 問・易→難・自己採点つき）

各 TODO 関数を実装して保存し、実行すると自動採点されます:

    uv run python lectures/44_embedding_clustering/exercises.py

- 未実装でも **exit 0** で終了します（PASS/FAIL の一覧が出るだけ）。
- 模範解答は exercises_solutions.py（実行すると全 PASS）。
- ねらい: L2 正規化 → コサイン距離行列 → k-means の E/M ステップ → inertia →
  purity → silhouette による k 選択 → 分割表（NMI の土台）まで、クラスタリングの核を
  ライブラリのブラックボックスに頼らず numpy だけで書けるようにする。
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 問1（易）: 行ベクトルの L2 正規化
#   X (n, d) の各行を L2 ノルム 1 にする。ノルム 0 の行はそのまま（0 除算回避）。
#   クラスタリングの大前提（正規化後は内積＝コサイン類似度）。
# ============================================================================
def l2_normalize_rows(X):
    # TODO: 各行のノルムを求め、np.maximum(norm, eps) で割る（eps=1e-12 程度）。
    raise NotImplementedError


# ============================================================================
# 問2（易）: コサイン距離行列
#   X (n, d) の全ペアのコサイン距離 (n, n) を返す。距離 = 1 - コサイン類似度。
#   対角は 0（自分自身）。
# ============================================================================
def cosine_distance_matrix(X):
    # TODO: 問1 で正規化 → S = Xn @ Xn.T（コサイン類似度）→ D = 1 - S。
    raise NotImplementedError


# ============================================================================
# 問3（易）: k-means の割り当てステップ（E-step）
#   X (n, d) と centroids (k, d) を受け取り、各点を最も近い中心へ割り当てた
#   ラベル配列 (n,) を返す（ユークリッド距離の argmin）。
# ============================================================================
def assign_to_nearest(X, centroids):
    # TODO: 各点と各中心の距離 (n, k) を作り argmin(axis=1)。
    #   ヒント: ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1) が (n, k)。
    raise NotImplementedError


# ============================================================================
# 問4（中）: k-means の更新ステップ（M-step）
#   X, labels, k から、各クラスタの平均ベクトル (k, d) を返す。
#   空クラスタ（割り当て 0 件）は 0 ベクトルにする（堅牢化）。
# ============================================================================
def update_centroids(X, labels, k):
    # TODO: c ごとに mask=labels==c。点があれば X[mask].mean(0)、無ければ零ベクトル。
    raise NotImplementedError


# ============================================================================
# 問5（中）: inertia（クラスタ内二乗距離和）
#   X, labels, centroids から、各点と『所属クラスタ中心』の二乗ユークリッド距離の総和を返す。
#   エルボー法で使う量（小さいほどクラスタが締まっている）。
# ============================================================================
def compute_inertia(X, labels, centroids):
    # TODO: 各点 i について sum((X[i]-centroids[labels[i]])**2) を全点で合計。
    raise NotImplementedError


# ============================================================================
# 問6（中）: purity（クラスタの純度）
#   labels_true, labels_pred から purity を返す。
#   各予測クラスタを「中の多数派の真ラベル」に割り当てたときの全体正解率。
# ============================================================================
def purity(labels_true, labels_pred) -> float:
    # TODO: 予測クラスタごとに、含まれる真ラベルの最頻値の個数を足し、全体数で割る。
    #   真ラベルは np.unique(..., return_inverse=True) で 0..C-1 に詰めると bincount が安全。
    raise NotImplementedError


# ============================================================================
# 問7（中）: silhouette によるベスト k 選択
#   {k: silhouette_score} の dict を受け取り、スコア最大の k を返す。
#   タイのときは小さい k（よりシンプルな分割）を選ぶ。
# ============================================================================
def best_k_by_silhouette(scores: dict) -> int:
    # TODO: items を (-score, k) でソートし先頭の k。または max(..., key=...) で。
    raise NotImplementedError


# ============================================================================
# 問8（難）: 分割表（contingency matrix）
#   labels_true, labels_pred から、行=真クラス・列=予測クラスタ の件数行列を返す。
#   真クラスは 0..R-1、予測クラスタは 0..C-1 に詰め直してから数える（飛び番でも安全に）。
#   これは NMI / homogeneity を自作する際の土台になる。
# ============================================================================
def contingency_matrix(labels_true, labels_pred):
    # TODO: np.unique(return_inverse=True) で両方を 0 始まりに。
    #   M = zeros((R, C)); 各点で M[ti, pi]+=1（np.add.at が便利）。
    raise NotImplementedError


# ============================================================================
# 自己採点ハーネス（編集不要）。未実装は FAIL だが exit 0 で終了する。
# ============================================================================
def _approx(got, exp, tol=1e-6):
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


def _check_norm():
    out = l2_normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    return _approx(out, [[0.6, 0.8], [0.0, 0.0]])


def _check_cosdist():
    D = cosine_distance_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    return _approx(np.diag(D), [0, 0, 0]) and _approx(D[0, 1], 1.0) and _approx(D[0, 2], 0.0)


def _check_assign():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0]])
    C = np.array([[0.0, 0.0], [1.0, 1.0]])
    return _approx(assign_to_nearest(X, C), [0, 1, 0])


def _check_update():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    out = update_centroids(X, np.array([0, 0, 1]), 3)
    return _approx(out[0], [1.0, 0.0]) and _approx(out[1], [0.0, 4.0]) and _approx(out[2], [0.0, 0.0])


def _check_inertia():
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    C = np.array([[1.0, 0.0]])
    # 各点と中心(1,0)の二乗距離: 1 + 1 = 2
    return _approx(compute_inertia(X, np.array([0, 0]), C), 2.0)


def _check_purity():
    return _approx(purity([0, 0, 1, 1, 2], [0, 0, 0, 1, 1]), 0.6)


def _check_bestk():
    return best_k_by_silhouette({2: 0.3, 3: 0.5, 6: 0.5, 4: 0.4}) == 3


def _check_contingency():
    M = contingency_matrix([0, 0, 1, 1, 1], [1, 1, 0, 0, 2])
    M = np.asarray(M)
    # 真0 →予測1 が2件、真1→予測0が2件・予測2が1件
    return M.shape == (2, 3) and _approx(M, [[0, 2, 0], [2, 0, 1]])


def _checks():
    return [
        ("問1 l2_normalize_rows", _check_norm),
        ("問2 cosine_distance_matrix", _check_cosdist),
        ("問3 assign_to_nearest", _check_assign),
        ("問4 update_centroids", _check_update),
        ("問5 compute_inertia", _check_inertia),
        ("問6 purity", _check_purity),
        ("問7 best_k_by_silhouette", _check_bestk),
        ("問8 contingency_matrix", _check_contingency),
    ]


def main():
    print("=" * 60)
    print("44_embedding_clustering 演習 自己採点")
    checks = _checks()
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    if passed == len(checks):
        print("全問正解！ クラスタリングの核（正規化・k-means・評価）を自力で書けています。")


if __name__ == "__main__":
    main()
