"""第15回 演習問題（画像埋め込みとメトリック学習）。

この回の演習は「埋め込みベクトルそのものの扱い」を、合成ベクトル上で手を動かして固める。
モデルのダウンロードは不要で、すべて numpy の小さな配列で完結する（高速・再現可能）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は未実装で FAIL になる）。
  2. 実装できたら自己採点を実行（未実装でも例外で落とさず、必ず exit 0）:
         uv run python lectures/15_image_embeddings_metric_learning/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時だけ、模範解答（全 PASS）の挙動を見る:
         uv run python lectures/15_image_embeddings_metric_learning/exercises_solutions.py
     （まずは自力で！）

テーマ（易 → 難）:
  1 L2正規化            2 コサイン類似度行列      3 kNN多数決予測
  4 Recall@k            5 Triplet マージン損失     6 ViT の CLS / mean pooling
  7 バッチハード Triplet 8 教師あり InfoNCE         9 検索 mAP（mean Average Precision）

採点ランナー grade() は exercises_solutions.py からも import して使う（採点ロジックの重複なし）。
"""

from __future__ import annotations

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


def ex7_batch_hard_triplet_loss(z: np.ndarray, labels: np.ndarray, margin: float) -> float:
    """演習7: バッチハード Triplet 損失（ハードネガティブ・マイニング）の平均を返す。

    各アンカー i について:
      - 最も遠い正例 d_pos[i] = 同クラス j への L2 距離の最大（自分自身 d=0 は max に影響しない）
      - 最も近い負例 d_neg[i] = 異クラス j への L2 距離の最小
    損失 = mean( relu(d_pos - d_neg + margin) )。
    難しい三つ組ほど勾配が大きく、ランダム負例より少ない反復で先に立ち上がる（サンプル効率）。
    入力 z: shape (N, D)、labels: shape (N,)。返り値は float。
    ヒント: 全ペア距離 d = sqrt(((z[:,None,:]-z[None,:,:])**2).sum(-1))、
            same = labels[:,None]==labels[None,:] を作り、
            d_pos = np.where(same, d, -inf).max(1)、d_neg = np.where(~same, d, +inf).min(1)。
    """
    # TODO: 全ペア距離→同/異クラスマスク→最遠正例 d_pos と最近負例 d_neg→hinge の平均
    raise NotImplementedError


def ex8_supervised_infonce(z: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """演習8: 教師あり InfoNCE（同クラスを正例集合とした温度付き softmax）の平均損失を返す。

    手順:
      1. z を L2 正規化（ex1 を再利用してよい）。
      2. sim = (zn @ zn.T) / temperature。対角（自分自身）は除外するため -1e9 を入れる。
      3. same = 同クラスなら 1、それ以外 0 の行列。対角は 0 にする（自分は正例にしない）。
      4. 行ごとに log_softmax(sim) を計算（オーバーフロー回避に行max を引いてから exp）。
      5. 各行で「正例位置の log 確率の平均」 = (same*logp).sum(1)/same.sum(1) を取り、
         その符号を反転して全行平均したものを損失とする。
    CLIP の画像-テキスト対照学習はこれの 2 モーダル版。返り値は float。
    """
    # TODO: 正規化→sim/温度→対角除外→same マスク→行 log_softmax→正例 log 確率の平均→ -mean
    raise NotImplementedError


def ex9_retrieval_map(
    gallery: np.ndarray, gallery_labels: np.ndarray,
    query: np.ndarray, query_labels: np.ndarray
) -> float:
    """演習9: 検索 mAP（mean Average Precision）を返す。Recall@k より厳密な順位指標。

    各クエリについて、ギャラリー全件をコサイン類似度の降順に並べ:
      - 正解 = 同クラスのギャラリー（rel[r]=1）。
      - 順位 r まで見たときの precision = (それまでの正解数) / r。
      - AP = Σ_r (precision@r * rel[r]) / 正解総数。
    全クエリの AP を平均したものが mAP。正解総数が 0 のクエリは AP=0 とする。返り値は float。
    ヒント: order=np.argsort(-sim, axis=1)、rel=(gallery_labels[order]==query_labels[:,None])、
            cum=np.cumsum(rel,axis=1)、precision=cum/np.arange(1,ng+1)。
    """
    # TODO: 各クエリで類似度降順に並べ、AP を計算して平均（mAP）を返す
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


def _sample_metric(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """わざとクラスが「重なる」4 クラス × 8 点の 6 次元ベクトル群。

    完全分離だと損失や mAP が 0 や 1 に張り付き、誤実装でも偶然 PASS してしまう。
    重なりを持たせることで、batch-hard triplet / InfoNCE / mAP の採点を識別力のある値にする。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(4):
        center = np.zeros(6)
        center[c % 6] = 2.0  # クラス間の距離 ≈ ノイズ幅 と同程度 → 適度に重なる
        for _ in range(8):
            X.append(center + rng.normal(0, 1.2, 6))
            y.append(c)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


# =====================================================================
# 採点ランナー用の参照実装（= 採点ロジック。exercises_solutions からは再実装しない）
# =====================================================================

def _ref_l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _ref_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _ref_l2(a) @ _ref_l2(b).T


def _ref_knn(g, gl, q, k):
    sim = _ref_cos(q, g)
    topk = np.argsort(-sim, axis=1)[:, :k]
    return np.array([np.bincount(gl[r]).argmax() for r in topk], dtype=np.int64)


def _ref_recall(g, gl, q, ql, k):
    sim = _ref_cos(q, g)
    topk = np.argsort(-sim, axis=1)[:, :k]
    hit = (gl[topk] == ql[:, None]).any(axis=1)
    return float(hit.mean())


def _ref_triplet(a, p, n, margin):
    d_ap = np.linalg.norm(a - p, axis=1)
    d_an = np.linalg.norm(a - n, axis=1)
    return float(np.maximum(0.0, d_ap - d_an + margin).mean())


def _ref_batch_hard(z, labels, margin):
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    same = labels[:, None] == labels[None, :]
    d_pos = np.where(same, d, -np.inf).max(1)
    d_neg = np.where(~same, d, np.inf).min(1)
    return float(np.maximum(0.0, d_pos - d_neg + margin).mean())


def _ref_infonce(z, labels, temperature):
    zn = _ref_l2(z)
    sim = (zn @ zn.T) / temperature
    np.fill_diagonal(sim, -1e9)
    same = (labels[:, None] == labels[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    m = sim.max(axis=1, keepdims=True)
    logp = sim - (m + np.log(np.exp(sim - m).sum(axis=1, keepdims=True)))
    loss = -(same * logp).sum(1) / np.maximum(same.sum(1), 1.0)
    return float(loss.mean())


def _ref_map(g, gl, q, ql):
    sim = _ref_cos(q, g)
    order = np.argsort(-sim, axis=1)
    rel = (gl[order] == ql[:, None]).astype(np.float64)
    cum = np.cumsum(rel, axis=1)
    ranks = np.arange(1, g.shape[0] + 1)
    precision = cum / ranks
    nrel = rel.sum(axis=1)
    ap = np.where(nrel > 0, (precision * rel).sum(axis=1) / np.maximum(nrel, 1.0), 0.0)
    return float(ap.mean())


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

def grade(funcs: dict) -> bool:
    """funcs（"ex1".."ex9" → 関数）を採点して PASS/FAIL を表示。全 PASS なら True。"""
    X, y = _sample_labeled()  # ギャラリー（既知）
    qX, qy = _sample_labeled(seed=5)  # 別ノイズのクエリ（評価用）
    a, p, n = _sample_triplets()
    lhs = _sample_lhs()
    zX, zy = _sample_metric(7)  # 重なりのある損失用サンプル（ex7/ex8）
    mgX, mgy = _sample_metric(7)  # mAP のギャラリー
    mqX, mqy = _sample_metric(11)  # mAP のクエリ（別シード）
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def c1():
        out = funcs["ex1"](X)
        norms = np.linalg.norm(out, axis=1)
        return (np.allclose(out, _ref_l2(X), atol=1e-6) and np.allclose(norms, 1, atol=1e-5),
                "各行の L2 ノルムが 1")

    def c2():
        return (np.allclose(funcs["ex2"](qX, X), _ref_cos(qX, X), atol=1e-6),
                "コサイン類似度行列 (nq, ng)")

    def c3():
        return (np.array_equal(funcs["ex3"](X, y, qX, 5), _ref_knn(X, y, qX, 5)),
                "kNN 多数決の予測ラベル")

    def c4():
        return (abs(funcs["ex4"](X, y, qX, qy, 5) - _ref_recall(X, y, qX, qy, 5)) < 1e-9,
                f"Recall@5={_ref_recall(X, y, qX, qy, 5):.3f}")

    def c5():
        return (abs(funcs["ex5"](a, p, n, 0.5) - _ref_triplet(a, p, n, 0.5)) < 1e-5,
                f"Triplet 損失={_ref_triplet(a, p, n, 0.5):.3f}")

    def c6():
        cls, mean = funcs["ex6"](lhs)
        scls, smean = lhs[:, 0], lhs[:, 1:].mean(axis=1)
        return (np.allclose(cls, scls) and np.allclose(mean, smean)
                and cls.shape == (4, 6) and mean.shape == (4, 6),
                "CLS と mean pooling の取り出し")

    def c7():
        val = funcs["ex7"](zX, zy, 0.5)
        return (abs(val - _ref_batch_hard(zX, zy, 0.5)) < 1e-5,
                f"batch-hard triplet={_ref_batch_hard(zX, zy, 0.5):.3f}")

    def c8():
        val = funcs["ex8"](zX, zy, 0.1)
        return (abs(val - _ref_infonce(zX, zy, 0.1)) < 1e-4,
                f"supervised InfoNCE={_ref_infonce(zX, zy, 0.1):.3f}")

    def c9():
        val = funcs["ex9"](mgX, mgy, mqX, mqy)
        return (abs(val - _ref_map(mgX, mgy, mqX, mqy)) < 1e-6,
                f"retrieval mAP={_ref_map(mgX, mgy, mqX, mqy):.3f}")

    check("ex1_l2_normalize", c1)
    check("ex2_cosine_sim_matrix", c2)
    check("ex3_knn_predict", c3)
    check("ex4_recall_at_k", c4)
    check("ex5_triplet_margin_loss", c5)
    check("ex6_vit_pool", c6)
    check("ex7_batch_hard_triplet_loss", c7)
    check("ex8_supervised_infonce", c8)
    check("ex9_retrieval_map", c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
    return {
        "ex1": ex1_l2_normalize,
        "ex2": ex2_cosine_sim_matrix,
        "ex3": ex3_knn_predict,
        "ex4": ex4_recall_at_k,
        "ex5": ex5_triplet_margin_loss,
        "ex6": ex6_vit_pool,
        "ex7": ex7_batch_hard_triplet_loss,
        "ex8": ex8_supervised_infonce,
        "ex9": ex9_retrieval_map,
    }


if __name__ == "__main__":
    grade(current_funcs())
