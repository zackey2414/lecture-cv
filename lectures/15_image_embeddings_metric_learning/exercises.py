"""第15回 演習問題（画像埋め込みとメトリック学習）。

この回の演習は「埋め込みベクトルそのものの扱い」を、合成ベクトル上で手を動かして固める。
モデルのダウンロードは不要で、すべて numpy の小さな配列で完結する（高速・再現可能）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は未実装で FAIL になる）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/15_image_embeddings_metric_learning/exercises.py
     全問 pass すれば "ALL PASS" と表示される（未実装でも例外で落ちず exit 0）。
  3. どうしても分からない時だけ、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/15_image_embeddings_metric_learning/exercises.py
     （まずは自力で！）

テーマ: L2 正規化 / コサイン類似度 / kNN 分類 / Recall@k / Triplet 損失 / ViT の CLS と mean pooling。
"""

from __future__ import annotations

import os

import numpy as np


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_l2_normalize(x: np.ndarray) -> np.ndarray:
    """演習1: 各行ベクトルを L2 ノルム 1 に正規化する。

    正規化後は「内積 = コサイン類似度」になり、検索/分類がベクトルの長さに左右されなくなる。
    ゼロ割を避けるため、ノルムには小さな eps(1e-12) の下限を入れること。
    入力 x: shape (N, D)。返り値も同 shape。
    """
    # TODO: norm = np.linalg.norm(x, axis=1, keepdims=True) を計算し、x / max(norm, eps) を返す
    raise NotImplementedError


def ex2_cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """演習2: a(各行) と b(各行) のコサイン類似度行列 (len(a), len(b)) を返す。

    手順: a と b をそれぞれ L2 正規化してから内積（a_norm @ b_norm.T）を取る。
    ex1_l2_normalize を再利用してよい。
    """
    # TODO: ex1_l2_normalize(a) @ ex1_l2_normalize(b).T を返す
    raise NotImplementedError


def ex3_knn_predict(
    gallery: np.ndarray, gallery_labels: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """演習3: コサイン最近傍 k 件の多数決で、各クエリの予測ラベルを返す。

    手順:
      1. クエリ×ギャラリーのコサイン類似度行列を作る（大きいほど近い）。
      2. 各クエリ行で類似度上位 k 件のギャラリー添字を取る（np.argsort(-sim) の先頭 k）。
      3. その k 件のラベルを多数決（np.bincount(...).argmax()）して予測ラベルにする。
    返り値: shape (len(query),) の int 配列。
    """
    # TODO: ex2_cosine_sim_matrix を使い、上位kの多数決ラベルを返す
    raise NotImplementedError


def ex4_recall_at_k(
    gallery: np.ndarray, gallery_labels: np.ndarray,
    query: np.ndarray, query_labels: np.ndarray, k: int
) -> float:
    """演習4: Recall@k = 各クエリの上位 k 件に「同クラス」が 1 件でも入る割合。

    kNN 分類（多数決で 1 ラベルに当てる）とは別物。『欲しい仲間が上位 k に出るか』だけを問う。
    手順: コサイン類似度上位 k のラベルに query_labels と一致するものが 1 つでもあれば hit、
    その平均を返す。
    """
    # TODO: 上位kのラベルに同クラスが含まれるかを各クエリで判定し、その割合(float)を返す
    raise NotImplementedError


def ex5_triplet_margin_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float
) -> float:
    """演習5: Triplet マージン損失の平均を返す（torch.nn.TripletMarginLoss の numpy 版）。

    定義: 各三つ組について L2 距離 d_ap=||a-p||, d_an=||a-n|| を計算し、
          hinge = max(0, d_ap - d_an + margin)。その平均（float）を返す。
    狙い: 同クラス(p)を近く・異クラス(n)を遠くに、少なくとも margin の余裕をつけて配置する。
    入力はいずれも shape (N, D)。
    """
    # TODO: d_ap, d_an を行ごとに計算し、relu(d_ap - d_an + margin) の平均を返す
    raise NotImplementedError


def ex6_vit_pool(last_hidden_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """演習6: ViT の last_hidden_state から CLS 埋め込みと mean pooling 埋め込みを取り出す。

    入力 last_hidden_state: shape (B, 1+P, D)。先頭トークン[:,0]が CLS、残り[:,1:]がパッチ。
    返り値: (cls, mean) の 2 つ。いずれも shape (B, D)。
      - cls  = last_hidden_state[:, 0]
      - mean = パッチトークン last_hidden_state[:, 1:] を系列方向に平均したもの
    （pooler_output を使わない理由は本文参照: チェックポイントによっては未学習のため。）
    """
    # TODO: CLS とパッチ平均を計算して (cls, mean) を返す
    raise NotImplementedError


# =====================================================================
# サンプル生成（採点用。シード固定で毎回同じ入力）
# =====================================================================

def _sample_labeled(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """3 クラスがそれぞれ別方向にまとまった 8 次元ベクトル群（ノイズ付き）。"""
    rng = np.random.default_rng(seed)
    centers = np.eye(3, 8) * 5.0  # 3 クラスの中心（直交方向）
    X, y = [], []
    for c in range(3):
        for _ in range(10):
            X.append(centers[c] + rng.normal(0, 0.5, 8))
            y.append(c)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _sample_triplets(seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (5, 4)).astype(np.float32)
    p = a + rng.normal(0, 0.1, (5, 4)).astype(np.float32)  # 近い
    n = a + rng.normal(0, 2.0, (5, 4)).astype(np.float32)  # 遠い
    return a, p, n


def _sample_lhs(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (4, 1 + 9, 6)).astype(np.float32)  # (B, 1+patches, D)


# =====================================================================
# 模範解答
# =====================================================================

def _sol_ex1(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _sol_ex2(a, b):
    return _sol_ex1(a) @ _sol_ex1(b).T


def _sol_ex3(gallery, gallery_labels, query, k):
    sim = _sol_ex2(query, gallery)
    topk = np.argsort(-sim, axis=1)[:, :k]
    neigh = gallery_labels[topk]
    return np.array([np.bincount(row).argmax() for row in neigh], dtype=np.int64)


def _sol_ex4(gallery, gallery_labels, query, query_labels, k):
    sim = _sol_ex2(query, gallery)
    topk = np.argsort(-sim, axis=1)[:, :k]
    neigh = gallery_labels[topk]
    hit = (neigh == query_labels[:, None]).any(axis=1)
    return float(hit.mean())


def _sol_ex5(anchor, positive, negative, margin):
    d_ap = np.linalg.norm(anchor - positive, axis=1)
    d_an = np.linalg.norm(anchor - negative, axis=1)
    return float(np.maximum(0.0, d_ap - d_an + margin).mean())


def _sol_ex6(lhs):
    return lhs[:, 0], lhs[:, 1:].mean(axis=1)


def _install_solutions() -> None:
    g = globals()
    g["ex1_l2_normalize"] = _sol_ex1
    g["ex2_cosine_sim_matrix"] = _sol_ex2
    g["ex3_knn_predict"] = _sol_ex3
    g["ex4_recall_at_k"] = _sol_ex4
    g["ex5_triplet_margin_loss"] = _sol_ex5
    g["ex6_vit_pool"] = _sol_ex6


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    X, y = _sample_labeled()  # ギャラリー（既知）
    qX, qy = _sample_labeled(seed=5)  # 別ノイズのクエリ（評価用）
    a, p, n = _sample_triplets()
    lhs = _sample_lhs()
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
        out = ex1_l2_normalize(X)
        norms = np.linalg.norm(out, axis=1)
        return (np.allclose(out, _sol_ex1(X), atol=1e-6) and np.allclose(norms, 1, atol=1e-5),
                "各行の L2 ノルムが 1")
    check("ex1_l2_normalize", _c1)

    check("ex2_cosine_sim_matrix", lambda: (
        np.allclose(ex2_cosine_sim_matrix(qX, X), _sol_ex2(qX, X), atol=1e-6),
        "コサイン類似度行列 (nq, ng)",
    ))
    check("ex3_knn_predict", lambda: (
        np.array_equal(ex3_knn_predict(X, y, qX, 5), _sol_ex3(X, y, qX, 5)),
        "kNN 多数決の予測ラベル",
    ))
    check("ex4_recall_at_k", lambda: (
        abs(ex4_recall_at_k(X, y, qX, qy, 5) - _sol_ex4(X, y, qX, qy, 5)) < 1e-9,
        "Recall@5",
    ))
    check("ex5_triplet_margin_loss", lambda: (
        abs(ex5_triplet_margin_loss(a, p, n, 0.5) - _sol_ex5(a, p, n, 0.5)) < 1e-5,
        "Triplet マージン損失の平均",
    ))

    def _c6():
        cls, mean = ex6_vit_pool(lhs)
        scls, smean = _sol_ex6(lhs)
        return (np.allclose(cls, scls) and np.allclose(mean, smean)
                and cls.shape == (4, 6) and mean.shape == (4, 6),
                "CLS と mean pooling の取り出し")
    check("ex6_vit_pool", _c6)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
