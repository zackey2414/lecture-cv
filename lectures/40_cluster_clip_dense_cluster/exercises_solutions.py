"""40_cluster_clip_dense_cluster 演習 — 模範解答（すべて PASS）。

exercises.py の TODO を埋めた完成版。Cluster-CLIP の全要素
（dense CLIP 整形 / 空間連結クラスタリング / FAISS / SQLite / ストリーム / 評価）を
自力で再構築できるかを 10 問（易->難）で確認する。

実行:
  uv run python lectures/40_cluster_clip_dense_cluster/exercises_solutions.py
"""

from __future__ import annotations

import pathlib
import queue
import sqlite3
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cc_common as H  # noqa: E402


# ===========================================================================
# 解答（学習者は exercises.py でこれらを実装する）
# ===========================================================================
def ex1_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Q1(易): 各行を L2 正規化する。ノルム 0 の行でも 0 除算で落ちないこと。"""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def ex2_tokens_to_feature_map(tokens: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Q2(易): ViT トークン列 [1+gh*gw, C] から CLS を捨て [C, gh, gw] に並べ替える。"""
    patches = tokens[1:]  # 先頭の CLS を除去
    c = patches.shape[1]
    return patches.reshape(gh, gw, c).transpose(2, 0, 1)


def ex3_mean_pool_clusters(feat_map: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """Q3(易): ラベルマップに従い各クラスタの平均->L2 正規化で代表ベクトルを作る。

    戻り値 reps[k, C]（k = ラベルの種類数、cluster id 0..k-1 の順）。
    """
    c, h, w = feat_map.shape
    y = feat_map.transpose(1, 2, 0).reshape(h * w, c)
    flat = label_map.reshape(-1)
    k = int(label_map.max()) + 1
    reps = np.empty((k, c), dtype="float32")
    for cid in range(k):
        v = y[flat == cid].mean(axis=0)
        reps[cid] = v / (np.linalg.norm(v) + 1e-8)
    return reps


def ex4_cluster_regions_connected(feat_map: np.ndarray, k: int):
    """Q4(中): 空間連結クラスタリングで feat_map[C,H,W] を k 領域に分け、(reps, label_map)。"""
    return H.cluster_regions(feat_map, k, connectivity=True, linkage="ward")


def ex5_connectivity_components(feat_map: np.ndarray, k: int) -> tuple[int, int]:
    """Q5(中): connectivity あり/なしでクラスタリングし、総連結成分数 (with, without)。"""
    _, lab_on = H.cluster_regions(feat_map, k, connectivity=True)
    _, lab_off = H.cluster_regions(feat_map, k, connectivity=False)
    return H.total_components(lab_on), H.total_components(lab_off)


def ex6_self_search_top1(reps: np.ndarray) -> np.ndarray:
    """Q6(中): reps を IndexFlatIP+IDMap に入れ、各ベクトルで検索した top-1 id 列を返す。

    正規化済みベクトルなら自分自身が rank-1 になるはず（=arange）。
    """
    ids = np.arange(len(reps), dtype=np.int64)
    index = H.build_ip_idmap(reps, ids)
    _, idx = index.search(H.as_faiss(reps), 1)
    return idx[:, 0]


def ex7_sqlite_roundtrip(rows: list[tuple], ids: list[int]) -> dict:
    """Q7(難): (faiss_id, frame_id, cluster_idx) を SQLite に入れ、ids で引き戻す。

    戻り値 {faiss_id: (frame_id, cluster_idx)}。faiss_id <-> メタの対応表の最小版。
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE VectorMapping (faiss_id INTEGER PRIMARY KEY, frame_id TEXT, cluster_idx INTEGER)")
    cur.executemany("INSERT INTO VectorMapping VALUES (?,?,?)", rows)
    conn.commit()
    placeholders = ",".join("?" * len(ids))
    cur.execute(
        f"SELECT faiss_id, frame_id, cluster_idx FROM VectorMapping WHERE faiss_id IN ({placeholders})",
        ids,
    )
    out = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    conn.close()
    return out


def ex8_search_regions(index, db_path: str, query_vec: np.ndarray, k: int) -> list[tuple]:
    """Q8(難): FAISS 検索 -> SQLite join で (frame_id, cluster_idx) の top-k を返す。

    -1 の faiss_id（近傍不足）は必ずスキップする。
    """
    _, idx = index.search(H.as_faiss(query_vec), k)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    out = []
    for fid in idx[0]:
        if fid == -1:
            continue
        cur.execute("SELECT frame_id, cluster_idx FROM VectorMapping WHERE faiss_id=?", (int(fid),))
        row = cur.fetchone()
        if row is not None:
            out.append((row[0], row[1]))
    conn.close()
    return out


def ex9_bounded_put(items: list, maxsize: int) -> int:
    """Q9(難): 消費者がいない満杯キューへ put_nowait し、溢れた分をドロップした数を返す。

    リアルタイムストリームでフレームを捨てる方針（参照リポ stream/capture.py）。
    """
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    dropped = 0
    for it in items:
        try:
            q.put_nowait(it)
        except queue.Full:
            dropped += 1
    return dropped


def ex10_coverage_ratio(mask_bool: np.ndarray, bbox: tuple) -> float:
    """Q10(難): クラスタマスクが bbox を覆う割合 = (マスク∩bbox)/bbox面積。"""
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    inter = int(mask_bool[y1:y2, x1:x2].sum())
    return inter / area


# ===========================================================================
# 自己採点ハーネス
# ===========================================================================
def _checker_feature_map(grid=6, dim=8, seed=0) -> np.ndarray:
    """市松模様の特徴マップ。特徴だけで分けると飛び地になり、連結制約の効果が出る。"""
    rng = np.random.default_rng(seed)
    a = H.l2n(rng.standard_normal((dim,)).astype("float32")[None])[0]
    b = H.l2n(rng.standard_normal((dim,)).astype("float32")[None])[0]
    fmap = np.empty((dim, grid, grid), dtype="float32")
    for r in range(grid):
        for c in range(grid):
            base = a if (r + c) % 2 == 0 else b
            v = base + 0.02 * rng.standard_normal(dim).astype("float32")
            fmap[:, r, c] = v / (np.linalg.norm(v) + 1e-8)
    return fmap


def _build_context() -> dict:
    rng = np.random.default_rng(0)
    feat_map = H.make_block_feature_map(grid=7, dim=16, n_blocks=3, seed=1)
    reps = H.l2n(rng.standard_normal((6, 16)).astype("float32"))
    return {"feat_map": feat_map, "reps": reps, "checker": _checker_feature_map()}


def _check1(ctx):
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype="float32")
    y = ex1_l2_normalize_rows(x)
    assert abs(np.linalg.norm(y[0]) - 1.0) < 1e-5, "正規化後ノルムが1でない"
    assert np.isfinite(y[1]).all(), "ゼロ行で 0 除算が起きている"


def _check2(ctx):
    gh, gw, c = 7, 7, 16
    tokens = np.arange((1 + gh * gw) * c, dtype="float32").reshape(1 + gh * gw, c)
    fmap = ex2_tokens_to_feature_map(tokens, gh, gw)
    assert fmap.shape == (c, gh, gw), f"shape が違う: {fmap.shape}"
    # パッチ (r=2,c=3) の特徴は tokens[1 + 2*gw + 3] と一致するはず
    assert np.allclose(fmap[:, 2, 3], tokens[1 + 2 * gw + 3]), "CLS 除去/並べ替えが誤り"


def _check3(ctx):
    fmap = ctx["feat_map"]
    _, label_map = H.cluster_regions(fmap, 4, connectivity=True)
    reps = ex3_mean_pool_clusters(fmap, label_map)
    assert reps.shape == (4, fmap.shape[0]), f"reps shape が違う: {reps.shape}"
    assert np.allclose(np.linalg.norm(reps, axis=1), 1.0, atol=1e-3), "代表ベクトルが未正規化"


def _check4(ctx):
    reps, label_map = ex4_cluster_regions_connected(ctx["feat_map"], 5)
    assert reps.shape[0] == 5 and label_map.shape == (7, 7)
    assert np.allclose(np.linalg.norm(reps, axis=1), 1.0, atol=1e-3), "代表ベクトル未正規化"
    assert H.total_components(label_map) == 5, "連結制約があるのに領域が飛び地になっている"


def _check5(ctx):
    k = 4
    comp_on, comp_off = ex5_connectivity_components(ctx["checker"], k)
    assert comp_on == k, f"connectivity ありなら成分数==k のはず: {comp_on}"
    assert comp_off >= comp_on, "connectivity なしの方が成分数が少ないのはおかしい"


def _check6(ctx):
    top1 = ex6_self_search_top1(ctx["reps"])
    assert np.array_equal(top1, np.arange(len(ctx["reps"]))), "自分自身が rank-1 でない"


def _check7(ctx):
    rows = [(10, "f0", 2), (11, "f0", 3), (20, "f1", 0)]
    got = ex7_sqlite_roundtrip(rows, [10, 20])
    assert got.get(10) == ("f0", 2) and got.get(20) == ("f1", 0), f"round-trip 失敗: {got}"
    assert 11 not in got, "問い合わせていない id まで返っている"


def _check8(ctx):
    reps = ctx["reps"]
    ids = np.arange(len(reps), dtype=np.int64)
    index = H.build_ip_idmap(reps, ids)
    with tempfile.TemporaryDirectory() as d:
        db = str(pathlib.Path(d) / "m.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE VectorMapping (faiss_id INTEGER PRIMARY KEY, frame_id TEXT, cluster_idx INTEGER)")
        conn.executemany(
            "INSERT INTO VectorMapping VALUES (?,?,?)",
            [(int(i), f"f{i // 3}", int(i % 3)) for i in ids],
        )
        conn.commit()
        conn.close()
        res = ex8_search_regions(index, db, reps[2:3], 3)
    assert len(res) == 3 and all(isinstance(r, tuple) for r in res), f"top-k 形式が違う: {res}"
    assert res[0] == ("f0", 2), f"自己検索の top-1 メタが違う: {res[0]}"


def _check9(ctx):
    assert ex9_bounded_put(list(range(10)), maxsize=3) == 7, "ドロップ数が違う"
    assert ex9_bounded_put([1, 2], maxsize=5) == 0, "満杯でないのにドロップしている"


def _check10(ctx):
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:15] = True  # 5x10 = 50 画素
    cov = ex10_coverage_ratio(mask, (5, 5, 15, 15))  # bbox 面積 10x10=100
    assert abs(cov - 0.5) < 1e-6, f"カバレッジが 0.5 でない: {cov}"
    assert ex10_coverage_ratio(mask, (5, 5, 15, 10)) == 1.0, "完全被覆で 1.0 にならない"


CHECKS = [
    ("Q1 l2_normalize_rows", _check1),
    ("Q2 tokens_to_feature_map", _check2),
    ("Q3 mean_pool_clusters", _check3),
    ("Q4 cluster_regions_connected", _check4),
    ("Q5 connectivity_components", _check5),
    ("Q6 self_search_top1", _check6),
    ("Q7 sqlite_roundtrip", _check7),
    ("Q8 search_regions", _check8),
    ("Q9 bounded_put", _check9),
    ("Q10 coverage_ratio", _check10),
]


def run() -> int:
    print("[setup] 合成特徴マップ・代表ベクトルを準備中 ...")
    ctx = _build_context()
    passed = 0
    for name, check in CHECKS:
        try:
            check(ctx)
            print(f"  PASS  {name}")
            passed += 1
        except NotImplementedError:
            print(f"  TODO  {name}  (未実装)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n採点: {passed}/{len(CHECKS)} PASS")
    return 0  # 未完成でも exit 0（繰り返し挑戦できるように）


if __name__ == "__main__":
    raise SystemExit(run())
