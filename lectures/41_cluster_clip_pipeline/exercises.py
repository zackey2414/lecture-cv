"""41_cluster_clip_pipeline 演習 — 自己採点ワークシート（講座の総仕上げ）。

各 ex*_ 関数の TODO を埋めて
    uv run python lectures/41_cluster_clip_pipeline/exercises.py
を回すと PASS / FAIL / TODO が表示される。全 9 問 PASS で合格。模範解答は exercises_solutions.py。

カバー範囲（学んだ全要素の統合）:
  Q1 正規化 / Q2 dense レイアウト / Q3 クラスタ代表ベクトル / Q4 FAISS 構築・検索 /
  Q5 検索結果の -1 ガード / Q6 SQLite 引き / Q7 カバレッジ評価 / Q8 適応サンプリング /
  Q9 検索エンジン統合（FAISS + SQLite を結線して領域検索を完成させる）。

難易度: Q1-2 易 / Q3-6 中 / Q7-9 難。
ヒント: 「コサイン類似は L2 正規化 + 内積」「FAISS 入力は float32・C 連続」「-1 は捨てる」。

実行:
    uv run python lectures/41_cluster_clip_pipeline/exercises.py
"""

from __future__ import annotations

import sqlite3  # noqa: F401  (解答で使う)
import sys
from pathlib import Path

import cv2  # noqa: F401
import faiss  # noqa: F401  (解答で使う)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_clip_helpers as H  # noqa: E402


# ===========================================================================
# ここを実装する（各関数の中身を TODO から置き換える）
# ===========================================================================


def ex1_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Q1(易): 行ごとに L2 正規化し、float32 で返す（各行のノルムが 1 になる）。

    ヒント: norm = np.linalg.norm(x, axis=1, keepdims=True); x / (norm + 1e-12)
    """
    raise NotImplementedError("Q1: 行ごとの L2 正規化を実装してください")


def ex2_patches_to_samples(feat_chw: np.ndarray) -> np.ndarray:
    """Q2(易): dense 特徴 [C, H, W] を、クラスタリング入力の [H*W, C] に並べ替える。

    パッチ (i, j) が 1 サンプル。出力 row = i*W + j が feat_chw[:, i, j] に一致すること。
    ヒント: feat_chw.transpose(1, 2, 0).reshape(H*W, C)
    """
    raise NotImplementedError("Q2: [C,H,W] -> [H*W,C] を実装してください")


def ex3_cluster_representatives(samples: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Q3(中): パッチ特徴 samples[N,C] とラベル labels[N] から、各クラスタの代表ベクトル [k,C] を作る。

    各クラスタの「平均 → L2 正規化」が代表ベクトル。空クラスタは全体平均で代用。float32 で返す。
    """
    raise NotImplementedError("Q3: クラスタ平均 + 正規化を実装してください")


def ex4_build_and_search(vectors: np.ndarray, ids: np.ndarray, query: np.ndarray) -> int:
    """Q4(中): IndexFlatIP + IDMap を作って vectors を add_with_ids し、query の最近傍 ID を返す。

    入力は L2 正規化済みとする（内積=コサイン類似）。float32・C 連続に直してから渡すこと。
    query は [C] か [1,C]。返すのは int の faiss_id。
    """
    raise NotImplementedError("Q4: FAISS の構築と検索を実装してください")


def ex5_filter_valid_hits(distances: np.ndarray, indices: np.ndarray) -> list[tuple[int, float]]:
    """Q5(中): faiss の戻り値 (distances[1,k], indices[1,k]) から、ID が -1 の行を捨てて

    (faiss_id, score) のリストをスコア降順で返す。-1 をそのまま使うと DB 参照でクラッシュする。
    """
    raise NotImplementedError("Q5: -1 を除外してスコア降順に整えてください")


def ex6_sqlite_lookup(db_path: Path, faiss_id: int) -> tuple[str, int]:
    """Q6(中): VectorMapping テーブルから faiss_id に対応する (frame_id, cluster_idx) を引いて返す。

    ヒント: SELECT frame_id, cluster_idx FROM VectorMapping WHERE faiss_id = ?
    """
    raise NotImplementedError("Q6: SQLite から faiss_id でメタを引いてください")


def ex7_mask_coverage(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Q7(難): クラスタマスク(bool, H x W) が BBox [x1,y1,x2,y2] をどれだけ覆うか（カバレッジ）を返す。

    coverage = (BBox 内で mask=True の画素数) / (BBox の面積)。面積 0 のときは 0.0。
    参考実装の評価（クラスタマスク ∩ BBox / BBox 面積）の最小版。
    """
    raise NotImplementedError("Q7: カバレッジ（mask∩bbox / bbox面積）を実装してください")


def ex8_should_keep(prev_hist: np.ndarray, cur_hist: np.ndarray, threshold: float) -> bool:
    """Q8(難): 適応サンプリングの判定。L1 正規化済みヒストグラム同士の交差から変化量を測り、

    delta = 1 - sum(min(prev, cur)) が threshold 以上なら True（= 場面が変わったので残す）。
    """
    raise NotImplementedError("Q8: ヒストグラム交差差分の判定を実装してください")


def ex9_region_search(build_dir: Path, query_vec: np.ndarray, k: int) -> list[dict]:
    """Q9(難・統合): 検索エンジンを完成させる。build_dir の FAISS と SQLite を結線し、

    query_vec[1,C] の上位 k 件を {"frame_id","cluster_idx","score"} の dict リスト（スコア降順）で返す。
    手順: read_index → search → -1 を捨てる → VectorMapping JOIN Frames でメタ引き → 並べ替え。
    """
    raise NotImplementedError("Q9: FAISS + SQLite を結線して領域検索を完成させてください")


# ===========================================================================
# 自己採点ハーネス（編集不要）
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
        "reps0": reps0,                 # frame0 の代表ベクトル [5, C]
        "query_region": reps0[0:1],     # frame0 cluster0（自己一致テスト用）
        "hist0": hist0,
        "hist2": hist2,
    }


def _check1(ctx):
    x = np.random.RandomState(0).randn(6, 8).astype(np.float64)
    out = ex1_l2_normalize_rows(x)
    assert out.dtype == np.float32, "float32 で返すこと"
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5), "各行のノルムが 1 でない"


def _check2(ctx):
    feat = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)  # [C=4,H=2,W=3]
    out = ex2_patches_to_samples(feat)
    assert out.shape == (6, 4), f"shape 不正: {out.shape}"
    # パッチ (i=1, j=2) -> row 1*3+2 = 5 が feat[:,1,2] と一致
    assert np.allclose(out[5], feat[:, 1, 2]), "パッチ→サンプルの対応が違う"


def _check3(ctx):
    samples = np.array([[1, 0], [1, 0], [0, 2], [0, 2]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    reps = ex3_cluster_representatives(samples, labels, k=2)
    assert reps.shape == (2, 2), f"shape 不正: {reps.shape}"
    assert np.allclose(np.linalg.norm(reps, axis=1), 1.0, atol=1e-5), "代表ベクトルが正規化されていない"
    assert np.allclose(reps[0], [1, 0], atol=1e-5) and np.allclose(reps[1], [0, 1], atol=1e-5), "平均→正規化が違う"


def _check4(ctx):
    vecs = H.l2_normalize(np.random.RandomState(1).randn(5, 16))
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    top = ex4_build_and_search(vecs, ids, vecs[2])
    assert int(top) == 30, f"自分自身(id=30)が最近傍にならない: {top}"


def _check5(ctx):
    distances = np.array([[0.9, 0.5, 0.7, 0.1]], dtype=np.float32)
    indices = np.array([[5, -1, 8, 2]], dtype=np.int64)
    out = ex5_filter_valid_hits(distances, indices)
    assert [i for i, _ in out] == [5, 8, 2], f"-1 除外/降順が違う: {out}"
    assert abs(out[0][1] - 0.9) < 1e-6, "スコアの対応が違う"


def _check6(ctx):
    # AUTOINCREMENT なので最初に入った faiss_id=1 は frame0 の cluster0
    frame_id, cluster_idx = ex6_sqlite_lookup(ctx["build_dir"] / H.DB_NAME, 1)
    assert frame_id == "synthA_frame000000" and cluster_idx == 0, f"引いた値が違う: {frame_id}, {cluster_idx}"


def _check7(ctx):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True  # 4x4 の True 領域
    # bbox [x1,y1,x2,y2] = (2,2,6,6) 面積16、全部覆う -> 1.0
    assert abs(ex7_mask_coverage(mask, (2, 2, 6, 6)) - 1.0) < 1e-6, "完全被覆で 1.0 にならない"
    # bbox (4,4,8,8) 面積16、重なりは (4..6,4..6)=2x2=4 -> 0.25
    assert abs(ex7_mask_coverage(mask, (4, 4, 8, 8)) - 0.25) < 1e-6, "部分被覆の計算が違う"
    assert ex7_mask_coverage(mask, (3, 3, 3, 3)) == 0.0, "面積0で 0.0 を返すこと"


def _check8(ctx):
    assert ex8_should_keep(ctx["hist0"], ctx["hist0"], 0.1) is False, "同一フレームは残さない(False)"
    assert ex8_should_keep(ctx["hist0"], ctx["hist2"], 0.1) is True, "別ショットは残す(True)"


def _check9(ctx):
    out = ex9_region_search(ctx["build_dir"], ctx["query_region"], k=3)
    assert isinstance(out, list) and len(out) >= 1, "結果が空"
    assert set(out[0].keys()) >= {"frame_id", "cluster_idx", "score"}, "dict のキーが不足"
    assert out[0]["frame_id"] == "synthA_frame000000" and out[0]["cluster_idx"] == 0, "自己領域が最上位でない"
    assert out[0]["score"] > 0.99, f"自己一致スコアが低い: {out[0]['score']}"
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True), "スコア降順になっていない"


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
        except NotImplementedError:
            print(f"  TODO  {name}  (未実装)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n採点: {passed}/{len(CHECKS)} PASS")
    return 0  # 未完成でも exit 0（繰り返し挑戦できるように）


if __name__ == "__main__":
    raise SystemExit(run())
