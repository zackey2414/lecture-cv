"""41_cluster_clip_pipeline 演習 — 模範解答（全 9 問 PASS）。

exercises.py の各 ex*_ を実装したもの。
    uv run python lectures/41_cluster_clip_pipeline/exercises_solutions.py
で全 PASS することを確認できる。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cv2  # noqa: F401
import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_clip_helpers as H  # noqa: E402


# ===========================================================================
# 解答
# ===========================================================================


def ex1_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Q1: 行ごとに L2 正規化（float32）。"""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / (norm + 1e-12)).astype(np.float32)


def ex2_patches_to_samples(feat_chw: np.ndarray) -> np.ndarray:
    """Q2: [C, H, W] -> [H*W, C]（パッチ (i,j) → row i*W+j）。"""
    c, h, w = feat_chw.shape
    return feat_chw.transpose(1, 2, 0).reshape(h * w, c)


def ex3_cluster_representatives(samples: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Q3: 各クラスタの平均 → L2 正規化（空クラスタは全体平均で代用）。"""
    samples = np.asarray(samples, dtype=np.float32)
    global_mean = samples.mean(axis=0)
    reps = np.empty((k, samples.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        v = samples[mask].mean(axis=0) if mask.any() else global_mean
        reps[c] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
    return reps


def ex4_build_and_search(vectors: np.ndarray, ids: np.ndarray, query: np.ndarray) -> int:
    """Q4: IndexFlatIP + IDMap を構築し、query の最近傍 ID を返す。"""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, ids)

    q = np.ascontiguousarray(query, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    _, indices = index.search(q, 1)
    return int(indices[0, 0])


def ex5_filter_valid_hits(distances: np.ndarray, indices: np.ndarray) -> list[tuple[int, float]]:
    """Q5: -1 を除外してスコア降順に整える。"""
    hits = []
    for rank in range(indices.shape[1]):
        fid = int(indices[0, rank])
        if fid == -1:
            continue
        hits.append((fid, float(distances[0, rank])))
    hits.sort(key=lambda t: t[1], reverse=True)
    return hits


def ex6_sqlite_lookup(db_path: Path, faiss_id: int) -> tuple[str, int]:
    """Q6: VectorMapping から faiss_id で (frame_id, cluster_idx) を引く。"""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT frame_id, cluster_idx FROM VectorMapping WHERE faiss_id = ?", (faiss_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"faiss_id {faiss_id} が見つかりません")
    return row[0], int(row[1])


def ex7_mask_coverage(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Q7: coverage = (BBox 内で mask=True の画素数) / (BBox 面積)。"""
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area == 0:
        return 0.0
    inter = int(mask[y1:y2, x1:x2].sum())
    return inter / area


def ex8_should_keep(prev_hist: np.ndarray, cur_hist: np.ndarray, threshold: float) -> bool:
    """Q8: delta = 1 - sum(min(prev, cur)) が threshold 以上なら True。"""
    intersection = float(np.minimum(prev_hist, cur_hist).sum())
    delta = max(0.0, 1.0 - intersection)
    return delta >= threshold


def ex9_region_search(build_dir: Path, query_vec: np.ndarray, k: int) -> list[dict]:
    """Q9: FAISS + SQLite を結線した領域検索エンジン。"""
    q = np.ascontiguousarray(query_vec, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]

    index = faiss.read_index(str(build_dir / H.INDEX_NAME))
    distances, indices = index.search(q, k)

    score_map: dict[int, float] = {}
    for rank in range(indices.shape[1]):
        fid = int(indices[0, rank])
        if fid == -1:  # 該当近傍が足りない場合は -1 が来る → 捨てる
            continue
        score_map[fid] = float(distances[0, rank])
    if not score_map:
        return []

    conn = sqlite3.connect(build_dir / H.DB_NAME)
    placeholders = ",".join("?" * len(score_map))
    rows = conn.execute(
        f"""
        SELECT v.faiss_id, v.cluster_idx, f.frame_id
        FROM VectorMapping v JOIN Frames f ON v.frame_id = f.frame_id
        WHERE v.faiss_id IN ({placeholders})
        """,
        list(score_map.keys()),
    ).fetchall()
    conn.close()

    results = [
        {"frame_id": frame_id, "cluster_idx": int(cluster_idx), "score": score_map[fid]}
        for fid, cluster_idx, frame_id in rows
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


# ===========================================================================
# 自己採点ハーネス（exercises.py と同一。全 PASS を確認する）
# ===========================================================================


def _build_context():
    H.set_seed(0)
    device = H.pick_device()
    base = H.outputs_dir("exercises")
    frames = H.run_split(base / "frames", 4, video_name="synthA")
    model, preprocess, _ = H.load_clip(device)
    build_dir = base / "build"
    H.run_build(build_dir, frames, model, preprocess, n_clusters=5, device=device)

    reps0 = np.load(build_dir / "vectors" / "synthA_frame000000_vecs.npy").astype(np.float32)
    hist0 = H.HistogramDeltaSampler._compute_hist(H.synth_frame(0, 4))
    hist2 = H.HistogramDeltaSampler._compute_hist(H.synth_frame(2, 4))
    return {
        "build_dir": build_dir,
        "reps0": reps0,
        "query_region": reps0[0:1],
        "hist0": hist0,
        "hist2": hist2,
    }


def _check1(ctx):
    x = np.random.RandomState(0).randn(6, 8).astype(np.float64)
    out = ex1_l2_normalize_rows(x)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def _check2(ctx):
    feat = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    out = ex2_patches_to_samples(feat)
    assert out.shape == (6, 4)
    assert np.allclose(out[5], feat[:, 1, 2])


def _check3(ctx):
    samples = np.array([[1, 0], [1, 0], [0, 2], [0, 2]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    reps = ex3_cluster_representatives(samples, labels, k=2)
    assert reps.shape == (2, 2)
    assert np.allclose(np.linalg.norm(reps, axis=1), 1.0, atol=1e-5)
    assert np.allclose(reps[0], [1, 0], atol=1e-5) and np.allclose(reps[1], [0, 1], atol=1e-5)


def _check4(ctx):
    vecs = H.l2_normalize(np.random.RandomState(1).randn(5, 16))
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    assert int(ex4_build_and_search(vecs, ids, vecs[2])) == 30


def _check5(ctx):
    distances = np.array([[0.9, 0.5, 0.7, 0.1]], dtype=np.float32)
    indices = np.array([[5, -1, 8, 2]], dtype=np.int64)
    out = ex5_filter_valid_hits(distances, indices)
    assert [i for i, _ in out] == [5, 8, 2]
    assert abs(out[0][1] - 0.9) < 1e-6


def _check6(ctx):
    frame_id, cluster_idx = ex6_sqlite_lookup(ctx["build_dir"] / H.DB_NAME, 1)
    assert frame_id == "synthA_frame000000" and cluster_idx == 0


def _check7(ctx):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    assert abs(ex7_mask_coverage(mask, (2, 2, 6, 6)) - 1.0) < 1e-6
    assert abs(ex7_mask_coverage(mask, (4, 4, 8, 8)) - 0.25) < 1e-6
    assert ex7_mask_coverage(mask, (3, 3, 3, 3)) == 0.0


def _check8(ctx):
    assert ex8_should_keep(ctx["hist0"], ctx["hist0"], 0.1) is False
    assert ex8_should_keep(ctx["hist0"], ctx["hist2"], 0.1) is True


def _check9(ctx):
    out = ex9_region_search(ctx["build_dir"], ctx["query_region"], k=3)
    assert isinstance(out, list) and len(out) >= 1
    assert set(out[0].keys()) >= {"frame_id", "cluster_idx", "score"}
    assert out[0]["frame_id"] == "synthA_frame000000" and out[0]["cluster_idx"] == 0
    assert out[0]["score"] > 0.99
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)


CHECKS = [
    ("Q1 l2_normalize_rows", _check1),
    ("Q2 patches_to_samples", _check2),
    ("Q3 cluster_representatives", _check3),
    ("Q4 build_and_search", _check4),
    ("Q5 filter_valid_hits", _check5),
    ("Q6 sqlite_lookup", _check6),
    ("Q7 mask_coverage", _check7),
    ("Q8 should_keep", _check8),
    ("Q9 region_search", _check9),
]


def run() -> int:
    print("[setup] CLIP と 4 フレームの小さな Build を準備中 ...")
    ctx = _build_context()
    passed = 0
    for name, check in CHECKS:
        try:
            check(ctx)
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n採点: {passed}/{len(CHECKS)} PASS")
    assert passed == len(CHECKS), "模範解答は全 PASS のはず"
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
