"""第27回 演習の模範解答（このファイルを実行すると全問 PASS する）。

exercises.py の各 TODO を埋めた参照実装。まずは自力で exercises.py を解き、
詰まったらここで答え合わせをすること。

実行: uv run python lectures/27_depth_pose_flow/exercises_solutions.py
"""

from __future__ import annotations

import numpy as np

COCO_SIGMAS = np.array(
    [.026, .025, .025, .035, .035, .079, .079, .072, .072,
     .062, .062, .107, .107, .087, .087, .089, .089],
    dtype=np.float32,
)


# =====================================================================
# 模範解答
# =====================================================================

def ex1_minmax_normalize(depth: np.ndarray) -> np.ndarray:
    """min-max 正規化。max==min なら 0 を返す（0 除算回避）。"""
    d = np.asarray(depth, dtype=np.float64)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-12:
        return np.zeros_like(d)
    return (d - lo) / (hi - lo)


def ex2_round_up_to_multiple(value: int, multiple: int = 8) -> int:
    """value 以上で multiple の倍数になる最小の整数。"""
    if value % multiple == 0:
        return value
    return value + (multiple - value % multiple)


def ex3_abs_rel(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """AbsRel = mean(|pred-gt| / gt)。gt は eps で下限クリップ。"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return float(np.mean(np.abs(pred - gt) / np.clip(gt, eps, None)))


def ex4_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """RMSE = sqrt(mean((pred-gt)^2))。"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def ex5_delta_accuracy(pred: np.ndarray, gt: np.ndarray, thr: float = 1.25,
                       eps: float = 1e-6) -> float:
    """δ<thr: max(pred/gt, gt/pred) < thr の割合。"""
    pred = np.clip(np.asarray(pred, dtype=np.float64), eps, None)
    gt = np.clip(np.asarray(gt, dtype=np.float64), eps, None)
    ratio = np.maximum(pred / gt, gt / pred)
    return float(np.mean(ratio < thr))


def ex6_align_scale_median(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """中央値スケール合わせ pred * median(gt)/median(pred)。"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mp = float(np.median(pred))
    if abs(mp) < 1e-8:
        return pred.copy()
    return pred * (float(np.median(gt)) / mp)


def ex7_endpoint_error(flow_pred: np.ndarray, flow_gt: np.ndarray) -> float:
    """EPE = mean(sqrt(du^2 + dv^2))。最後の軸が (u, v)。"""
    diff = np.asarray(flow_pred, dtype=np.float64) - np.asarray(flow_gt, dtype=np.float64)
    return float(np.mean(np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)))


def ex8_oks(pred_xy: np.ndarray, gt: np.ndarray, area: float,
            sigmas: np.ndarray = COCO_SIGMAS) -> float:
    """OKS = Σ exp(-d^2/(2·area·(2σ)^2))·δ(v>0) / Σ δ(v>0)。"""
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt = np.asarray(gt, dtype=np.float64)
    vis = gt[:, 2] > 0
    if not np.any(vis):
        return 0.0
    k = 2.0 * np.asarray(sigmas, dtype=np.float64)
    d2 = (pred_xy[:, 0] - gt[:, 0]) ** 2 + (pred_xy[:, 1] - gt[:, 1]) ** 2
    ks = np.exp(-d2 / (2.0 * float(area) * (k ** 2) + 1e-12))
    return float(np.sum(ks[vis]) / np.sum(vis))


def ex9_pck(pred_xy: np.ndarray, gt: np.ndarray, ref_len: float, alpha: float = 0.2) -> float:
    """PCK@alpha: d_i <= alpha*ref_len を満たす可視点の割合。"""
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt = np.asarray(gt, dtype=np.float64)
    vis = gt[:, 2] > 0
    if not np.any(vis):
        return 0.0
    d = np.sqrt((pred_xy[:, 0] - gt[:, 0]) ** 2 + (pred_xy[:, 1] - gt[:, 1]) ** 2)
    return float(np.mean(d[vis] <= (alpha * float(ref_len))))


# =====================================================================
# 自己採点ランナー（exercises.py と同一の検証）
# =====================================================================

def _make_kp(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    return np.concatenate([xy, vis.reshape(-1, 1)], axis=1).astype(np.float32)


def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        out = ex1_minmax_normalize(np.array([[0.0, 5.0], [10.0, 5.0]]))
        const = ex1_minmax_normalize(np.full((2, 2), 3.0))
        return np.allclose(out, [[0, 0.5], [1, 0.5]]) and np.allclose(const, 0.0), f"out={out.tolist()}"
    check("ex1_minmax_normalize", _c1)

    def _c2():
        ok = (ex2_round_up_to_multiple(125, 8) == 128 and ex2_round_up_to_multiple(128, 8) == 128
              and ex2_round_up_to_multiple(1, 8) == 8 and ex2_round_up_to_multiple(130, 8) == 136)
        return ok, f"125->{ex2_round_up_to_multiple(125, 8)}"
    check("ex2_round_up_to_multiple", _c2)

    def _c3():
        v = ex3_abs_rel(np.array([2.0, 4.0]), np.array([1.0, 2.0]))
        z = ex3_abs_rel(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        return abs(v - 1.0) < 1e-9 and z == 0.0, f"AbsRel={v:.4f}"
    check("ex3_abs_rel", _c3)

    def _c4():
        v = ex4_rmse(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 6]))
        return abs(v - 1.0) < 1e-9, f"RMSE={v:.4f}"
    check("ex4_rmse", _c4)

    def _c5():
        v = ex5_delta_accuracy(np.array([1.0, 2, 4]), np.array([1.0, 2, 2]), 1.25)
        return abs(v - 2 / 3) < 1e-9, f"delta={v:.4f}"
    check("ex5_delta_accuracy", _c5)

    def _c6():
        out = ex6_align_scale_median(np.array([1.0, 2, 3]), np.array([2.0, 4, 6]))
        return np.allclose(out, [2, 4, 6]), f"out={np.asarray(out).tolist()}"
    check("ex6_align_scale_median", _c6)

    def _c7():
        v = ex7_endpoint_error(np.array([[[3.0, 4.0]]]), np.array([[[0.0, 0.0]]]))
        z = ex7_endpoint_error(np.zeros((4, 2)), np.zeros((4, 2)))
        return abs(v - 5.0) < 1e-9 and z == 0.0, f"EPE={v:.4f}"
    check("ex7_endpoint_error", _c7)

    def _c8():
        xy = np.array([[10.0, 10.0]] + [[0.0, 0.0]] * 16, dtype=np.float32)
        vis = np.array([2] + [0] * 16, dtype=np.float32)
        gt = _make_kp(xy, vis)
        perfect = ex8_oks(xy, gt, area=1000.0)
        pred = xy.copy()
        pred[0, 0] += 0.5
        got = ex8_oks(pred, gt, area=1000.0)
        k0 = 2 * float(COCO_SIGMAS[0])
        expect = float(np.exp(-(0.5 ** 2) / (2 * 1000.0 * k0 ** 2)))
        zero = ex8_oks(np.zeros_like(xy), _make_kp(xy, np.zeros(17, np.float32)), area=1000.0)
        ok = abs(perfect - 1.0) < 1e-9 and abs(got - expect) < 1e-6 and zero == 0.0
        return ok, f"perfect={perfect:.4f}, got={got:.4f}, expect={expect:.4f}"
    check("ex8_oks", _c8)

    def _c9():
        xy = np.zeros((17, 2), np.float32)
        vis = np.zeros(17, np.float32)
        vis[:4] = 2
        pred = xy.copy()
        pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0] = 1.0, 2.0, 10.0, 10.0
        v = ex9_pck(pred, _make_kp(xy, vis), ref_len=10.0, alpha=0.2)
        return abs(v - 0.5) < 1e-9, f"PCK={v:.4f}"
    check("ex9_pck", _c9)

    print("=== 採点結果（模範解答）===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\n（想定外）模範解答が FAIL しました。")


if __name__ == "__main__":
    _grade()
