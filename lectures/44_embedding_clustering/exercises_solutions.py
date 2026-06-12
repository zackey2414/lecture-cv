"""exercises_solutions.py — 44_embedding_clustering 演習の模範解答（全 PASS）

    uv run python lectures/44_embedding_clustering/exercises_solutions.py

設計: 採点ロジックは exercises.py を **そのまま再利用** し（重複を作らない）、
本ファイルは解答実装だけを持つ。exercises モジュールの関数名を解答で差し替えてから
同じ採点ハーネス（_checks / _run）を回し、全問 PASS することを確認する。
各解答は単一責務・早期 return を意識し、定義式を素直に numpy で写している。
"""

from __future__ import annotations

import numpy as np

import exercises as ex


# 問1: 行ベクトルの L2 正規化（ノルム 0 はそのまま）
def l2_normalize_rows(X):
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, 1e-12)


# 問2: コサイン距離行列（1 - 正規化ベクトルの内積）
def cosine_distance_matrix(X):
    Xn = l2_normalize_rows(X)
    return 1.0 - Xn @ Xn.T


# 問3: k-means の割り当てステップ（最近傍中心の argmin）
def assign_to_nearest(X, centroids):
    X = np.asarray(X, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)  # (n, k)
    return d2.argmin(axis=1)


# 問4: k-means の更新ステップ（クラスタ平均・空クラスタは零ベクトル）
def update_centroids(X, labels, k):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    d = X.shape[1]
    centroids = np.zeros((k, d), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = X[mask].mean(axis=0)
    return centroids


# 問5: inertia（各点と所属中心の二乗距離の総和）
def compute_inertia(X, labels, centroids):
    X = np.asarray(X, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    diff = X - centroids[np.asarray(labels)]
    return float((diff * diff).sum())


# 問6: purity（多数派ラベルへの割り当て正解率）
def purity(labels_true, labels_pred) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    total = len(labels_true)
    if total == 0:
        return 0.0
    _, true_idx = np.unique(labels_true, return_inverse=True)  # 0..C-1 に詰める
    correct = 0
    for c in np.unique(labels_pred):
        correct += np.bincount(true_idx[labels_pred == c]).max()
    return float(correct / total)


# 問7: silhouette 最大の k（タイは小さい k）
def best_k_by_silhouette(scores: dict) -> int:
    # (-score, k) でソートすれば、スコア降順 → タイは k 昇順で先頭が答え。
    return min(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]


# 問8: 分割表（行=真クラス・列=予測クラスタ の件数）
def contingency_matrix(labels_true, labels_pred):
    _, ti = np.unique(labels_true, return_inverse=True)
    _, pi = np.unique(labels_pred, return_inverse=True)
    M = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(M, (ti, pi), 1)
    return M


def main():
    # exercises の関数名を解答で差し替え、同じ採点ハーネスを再利用する。
    ex.l2_normalize_rows = l2_normalize_rows
    ex.cosine_distance_matrix = cosine_distance_matrix
    ex.assign_to_nearest = assign_to_nearest
    ex.update_centroids = update_centroids
    ex.compute_inertia = compute_inertia
    ex.purity = purity
    ex.best_k_by_silhouette = best_k_by_silhouette
    ex.contingency_matrix = contingency_matrix

    print("=" * 60)
    print("44_embedding_clustering 演習 模範解答 自己採点")
    checks = ex._checks()
    passed = sum(ex._run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    assert passed == len(checks), "模範解答が全 PASS しません（バグ）"
    print("全問 PASS。")


if __name__ == "__main__":
    main()
