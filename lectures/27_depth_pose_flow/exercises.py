"""第27回 演習問題（深度・姿勢・フローの“評価と前処理”を手で書けるようにする）。

使い方:
  1. 各関数の TODO を自分で実装する（最初は NotImplementedError だが、採点ランナーが
     拾うのでプロセスは落ちず、FAIL と表示されるだけ＝必ず exit 0）。
  2. 自己採点:  uv run python lectures/27_depth_pose_flow/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 模範解答（実行すると全 PASS）:
        uv run python lectures/27_depth_pose_flow/exercises_solutions.py

この9問は本モジュールの“評価コア＆前処理コア”を易→難で1つずつ抜き出したもの（モデル DL 不要）:
  ex1: 深度マップの min-max 正規化（可視化の前処理）                         … 01
  ex2: RAFT 用「8 の倍数」への切り上げ（フロー前処理の鉄則）               … 03
  ex3: AbsRel = mean(|pred-gt|/gt)（深度の主指標）                          … 01/04
  ex4: RMSE = sqrt(mean((pred-gt)^2))（深度）                              … 04
  ex5: δ<1.25 = max(p/g, g/p) < 1.25 の割合（threshold accuracy）          … 01/04
  ex6: 中央値スケール合わせ pred*median(gt)/median(pred)（相対深度の整合）   … 01/04
  ex7: EPE = mean(||flow_pred - flow_gt||)（フローの端点誤差）              … 03/04
  ex8: OKS（物体スケールと per-keypoint σ で正規化した姿勢類似度）          … 02/04
  ex9: PCK@α（参照長×α 以内に入ったキーポイントの割合）                     … 02/04
"""

from __future__ import annotations

import numpy as np

# COCO 17 点の per-keypoint 定数 σ_i（ex8 のデフォルトで使う）
COCO_SIGMAS = np.array(
    [.026, .025, .025, .035, .035, .079, .079, .072, .072,
     .062, .062, .107, .107, .087, .087, .089, .089],
    dtype=np.float32,
)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_minmax_normalize(depth: np.ndarray) -> np.ndarray:
    """演習1: 深度マップを min-max で [0,1] に正規化して返す（可視化前の定番）。

    - out = (depth - min) / (max - min)。
    - max == min（定数マップ）のときは 0 除算を避けて全て 0.0 を返す。
    例: [[0,5],[10,5]] -> [[0.0,0.5],[1.0,0.5]]
    """
    # TODO: min-max 正規化した float 配列を返す
    raise NotImplementedError


def ex2_round_up_to_multiple(value: int, multiple: int = 8) -> int:
    """演習2: value 以上で multiple の倍数になる最小の整数を返す（RAFT の 8 の倍数要件）。

    - 既に倍数ならそのまま返す。
    例: (125, 8) -> 128, (128, 8) -> 128, (1, 8) -> 8, (130, 8) -> 136
    """
    # TODO: 切り上げた整数を返す
    raise NotImplementedError


def ex3_abs_rel(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """演習3: AbsRel = mean(|pred - gt| / gt) を返す（gt は eps で下限クリップ）。

    例: pred=[2,4], gt=[1,2] -> (|2-1|/1 + |4-2|/2)/2 = (1+1)/2 = 1.0
    """
    # TODO: AbsRel を float で返す
    raise NotImplementedError


def ex4_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """演習4: RMSE = sqrt(mean((pred - gt)^2)) を返す。

    例: pred=[1,2,3,4], gt=[1,2,3,6] -> sqrt((0+0+0+4)/4) = 1.0
    """
    # TODO: RMSE を float で返す
    raise NotImplementedError


def ex5_delta_accuracy(pred: np.ndarray, gt: np.ndarray, thr: float = 1.25,
                       eps: float = 1e-6) -> float:
    """演習5: δ<thr = 各要素で max(pred/gt, gt/pred) < thr となる割合を返す。

    - pred, gt は eps で下限クリップしてから比をとる。
    例: pred=[1,2,4], gt=[1,2,2], thr=1.25 -> 比は [1,1,2]、2<1.25 は False -> 2/3
    """
    # TODO: threshold accuracy を float で返す
    raise NotImplementedError


def ex6_align_scale_median(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """演習6: 中央値スケール合わせ pred * (median(gt)/median(pred)) を返す。

    - 相対深度を GT のスケールへ寄せる定石。median(pred) が 0 に近いときは pred をそのまま返す。
    例: pred=[1,2,3], gt=[2,4,6] -> median比 = 4/2 = 2 -> [2,4,6]
    """
    # TODO: スケール合わせ後の配列を返す
    raise NotImplementedError


def ex7_endpoint_error(flow_pred: np.ndarray, flow_gt: np.ndarray) -> float:
    """演習7: EPE = mean(sqrt(du^2 + dv^2)) を返す（フローの平均端点誤差）。

    - flow_*: (..., 2) 形状で最後の軸が (u, v)。du = pred_u - gt_u, dv = pred_v - gt_v。
    例: flow_pred=[[[3,4]]], flow_gt=[[[0,0]]] -> sqrt(9+16)=5.0
    """
    # TODO: EPE を float で返す
    raise NotImplementedError


def ex8_oks(pred_xy: np.ndarray, gt: np.ndarray, area: float,
            sigmas: np.ndarray = COCO_SIGMAS) -> float:
    """演習8: OKS を返す。COCO 定義:

        KS_i = exp(-d_i^2 / (2 * area * k_i^2)),  k_i = 2*σ_i
        OKS  = Σ_i KS_i·δ(v_i>0) / Σ_i δ(v_i>0)

    - pred_xy: (17,2) 予測座標。gt: (17,3=x,y,v)。可視 v>0 の点だけ分母に数える。
    - 可視点が無ければ 0.0 を返す。完全一致(pred_xy==gt[:, :2])なら 1.0 になる。
    """
    # TODO: OKS を float で返す
    raise NotImplementedError


def ex9_pck(pred_xy: np.ndarray, gt: np.ndarray, ref_len: float, alpha: float = 0.2) -> float:
    """演習9: PCK@alpha = (距離 d_i <= alpha*ref_len を満たす可視キーポイントの割合) を返す。

    - pred_xy: (17,2), gt: (17,3=x,y,v)。可視 v>0 の点だけ対象。
    - 可視点が無ければ 0.0。
    例: 可視4点の距離=[1,2,10,10], ref_len=10, alpha=0.2 -> 閾値2.0 -> [T,T,F,F] -> 0.5
    """
    # TODO: PCK を float で返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _make_kp(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(x,y) と可視フラグから (N,3) の keypoint 配列を作るテスト用ヘルパ。"""
    return np.concatenate([xy, vis.reshape(-1, 1)], axis=1).astype(np.float32)


def _grade() -> None:
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
        out = ex1_minmax_normalize(np.array([[0.0, 5.0], [10.0, 5.0]]))
        const = ex1_minmax_normalize(np.full((2, 2), 3.0))
        ok = np.allclose(out, [[0, 0.5], [1, 0.5]]) and np.allclose(const, 0.0)
        return ok, f"out={out.tolist()}"
    check("ex1_minmax_normalize", _c1)

    def _c2():
        ok = (ex2_round_up_to_multiple(125, 8) == 128 and ex2_round_up_to_multiple(128, 8) == 128
              and ex2_round_up_to_multiple(1, 8) == 8 and ex2_round_up_to_multiple(130, 8) == 136)
        return ok, f"125->{ex2_round_up_to_multiple(125, 8)}"
    check("ex2_round_up_to_multiple", _c2)

    def _c3():
        v = ex3_abs_rel(np.array([2.0, 4.0]), np.array([1.0, 2.0]))
        z = ex3_abs_rel(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        return abs(v - 1.0) < 1e-9 and z == 0.0, f"AbsRel={v:.4f}, same={z}"
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
        # 完全一致 -> 1.0
        xy = np.array([[10.0, 10.0]] + [[0.0, 0.0]] * 16, dtype=np.float32)
        vis = np.array([2] + [0] * 16, dtype=np.float32)
        gt = _make_kp(xy, vis)
        perfect = ex8_oks(xy, gt, area=1000.0)
        # 1点だけ可視で dx=0.5 ずらす -> exp(-(0.5^2)/(2*1000*(2*σ0)^2)) と一致するはず
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
        gt_xy = xy.copy()
        # 4 点だけ可視にし、距離が [1,2,10,10] になるよう pred をずらす
        vis = np.zeros(17, np.float32)
        vis[:4] = 2
        pred = xy.copy()
        pred[0, 0] = 1.0
        pred[1, 0] = 2.0
        pred[2, 0] = 10.0
        pred[3, 0] = 10.0
        v = ex9_pck(pred, _make_kp(gt_xy, vis), ref_len=10.0, alpha=0.2)
        return abs(v - 0.5) < 1e-9, f"PCK={v:.4f}"
    check("ex9_pck", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok
          else "\nまだ未達の演習があります。TODO を埋めましょう（解答: exercises_solutions.py）。")


if __name__ == "__main__":
    _grade()
