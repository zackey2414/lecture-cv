"""第43回 演習問題（色空間と画像の調整）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/43_color_spaces_and_adjustments/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動で答え合わせ:
         SHOW_SOLUTION=1 uv run python lectures/43_color_spaces_and_adjustments/exercises.py
     模範解答そのものは exercises_solutions.py にもある（実行すると全 PASS）。
     （まずは自力で！）

難易度は易 → 難（ex1〜ex3 が基礎、ex4〜ex7 が応用、ex8〜ex10 が総合）。
未実装でも自己採点は例外で落ちず、その問だけ FAIL 表示で最後まで走る（exit 0）。

覚えておく数値:
  - OpenCV 8bit: HSV は H=0-179, S/V=0-255。Lab/YCrCb は各ch 0-255。
  - 彩度の掛け算や WB は uint8 のままだとオーバーフローする → float32 で計算して clip→uint8。
  - ガンマ: 出力 = 255*(入力/255)^(1/gamma)。gamma>1 で明るくなる。
  - 輝度だけ動かす(YCrCb の Y / Lab の L)と色が崩れない。
  - ΔE*76 は float の L*a*b* のユークリッド距離（uint8 Lab で測ってはいけない）。
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import make_photo_scene_bgr  # noqa: E402


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """演習1【基礎】BGR画像を HSV に変換して返す。

    返り値は (H, W, 3) の uint8。OpenCV の H は 0-179 に収まる。
    ヒント: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)。
    """
    # TODO: BGR→HSV へ変換して返す
    raise NotImplementedError


def ex2_brightness_contrast(bgr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """演習2【基礎】線形調整 out = alpha*in + beta を飽和演算で行って返す。

    alpha=コントラスト(傾き), beta=明るさ(オフセット)。255 を超えたら頭打ち。
    ヒント: cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)。
    """
    # TODO: convertScaleAbs で線形調整して返す
    raise NotImplementedError


def ex3_gamma_lut(gamma: float) -> np.ndarray:
    """演習3【基礎】ガンマ補正用の 256 要素 LUT（uint8, shape=(256,)）を返す。

    LUT[i] = round(255 * (i/255)^(1/gamma)) を 0-255 にクリップ。
    手順:
      1. x = arange(256)/255 を float で作る。
      2. (x ** (1/gamma)) * 255 を計算し、np.clip(..., 0, 255)。
      3. astype(np.uint8) で返す。
    """
    # TODO: gamma 用の LUT を作って返す
    raise NotImplementedError


def ex4_scale_saturation(bgr: np.ndarray, factor: float) -> np.ndarray:
    """演習4【応用】彩度(HSV の S)だけを factor 倍して返す（色相・明度は保つ）。

    手順:
      1. BGR→HSV にして float32 にする（uint8 のまま掛けるとオーバーフロー）。
      2. S チャンネル(index 1)に factor を掛け、np.clip(..., 0, 255)。
      3. uint8 に戻して HSV→BGR で返す。
    """
    # TODO: S だけスケールして BGR で返す
    raise NotImplementedError


def ex5_shift_hue(bgr: np.ndarray, delta: int) -> np.ndarray:
    """演習5【応用】色相 H を delta(0-179スケール)だけ回して返す（mod 180）。

    手順:
      1. BGR→HSV。
      2. H チャンネル(index 0)に delta を足し、% 180 で巻き戻す（int で計算してから uint8）。
      3. HSV→BGR で返す。
    """
    # TODO: 色相を delta だけ回して返す
    raise NotImplementedError


def ex6_equalize_luminance(bgr: np.ndarray) -> np.ndarray:
    """演習6【応用】輝度チャンネル(YCrCb の Y)だけを平坦化して返す（色を崩さない）。

    手順:
      1. BGR→YCrCb。
      2. Y チャンネル(index 0)に cv2.equalizeHist を掛ける。
      3. YCrCb→BGR で返す。
    """
    # TODO: Y だけ equalizeHist して返す
    raise NotImplementedError


def ex7_gray_world_wb(bgr: np.ndarray) -> np.ndarray:
    """演習7【総合】gray-world ホワイトバランスを実装して返す。

    各チャンネル平均を、全体平均(灰)に揃えるゲインを掛けて色かぶりを打ち消す。
    手順:
      1. float32 にする。
      2. (mB,mG,mR) = 各chの平均、gray = それらの平均。
      3. gains = gray / 各ch平均（0割り回避に np.clip(means, 1e-6, None)）。
      4. clip(f*gains, 0,255).astype(uint8) を返す。
    """
    # TODO: gray-world WB を実装して返す
    raise NotImplementedError


def ex8_skin_mask_ycrcb(bgr: np.ndarray) -> np.ndarray:
    """演習8【総合】YCrCb の定番レンジで肌色マスク(0/255, (H,W))を返す。

    レンジ: Cr ∈ [133,173], Cb ∈ [77,127]、Y は全域(0-255)。
    ヒント: cv2.cvtColor → cv2.inRange(ycrcb, (0,133,77), (255,173,127))。
    """
    # TODO: YCrCb の肌色レンジで inRange マスクを返す
    raise NotImplementedError


def ex9_mean_delta_e(bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
    """演習9【総合】2画像の平均 ΔE*76（float L*a*b* のユークリッド距離）を返す。

    最重要: 必ず float の Lab で計算する。手順:
      1. 各画像を astype(float32)/255 にして cv2.COLOR_BGR2Lab。
      2. 画素ごとに sqrt(((labA-labB)**2).sum(axis=2))。
      3. その平均を float で返す。
    """
    # TODO: float Lab で平均 ΔE を計算して返す
    raise NotImplementedError


def ex10_clahe_luminance(bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """演習10【総合】Lab の L チャンネルにだけ CLAHE を掛けて返す（色を保つ）。

    手順:
      1. BGR→Lab。
      2. clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile,tile))。
      3. L チャンネル(index 0)に clahe.apply を適用。
      4. Lab→BGR で返す。
    """
    # TODO: L チャンネルにだけ CLAHE を掛けて返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _close(a: np.ndarray, b: np.ndarray, tol: int = 1) -> bool:
    """2つの uint8 画像が「ほぼ一致」か（最大絶対差 <= tol）。float 経路の丸め差を吸収。"""
    if a is None or b is None or a.shape != b.shape:
        return False
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()) <= tol


def _grade() -> None:
    bgr = make_photo_scene_bgr()  # (320, 480, 3)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, ok, detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        got = ex1_to_hsv(bgr)
        ref = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return (got is not None and np.array_equal(got, ref)
                and int(got[:, :, 0].max()) <= 179, "BGR→HSV, H<=179")
    check("ex1_to_hsv", _c1)

    def _c2():
        got = ex2_brightness_contrast(bgr, 1.4, 20)
        ref = cv2.convertScaleAbs(bgr, alpha=1.4, beta=20)
        return (_close(got, ref, 0), "out=alpha*in+beta（飽和）")
    check("ex2_brightness_contrast", _c2)

    def _c3():
        got = ex3_gamma_lut(2.0)
        x = np.arange(256, dtype=np.float32) / 255.0
        ref = np.clip((x ** 0.5) * 255.0, 0, 255).astype(np.uint8)
        return (got is not None and got.shape == (256,) and _close(got, ref, 0),
                "gamma LUT (shape=256)")
    check("ex3_gamma_lut", _c3)

    def _c4():
        got = ex4_scale_saturation(bgr, 0.0)  # 彩度0 → S が全画素 0 になる
        s = cv2.cvtColor(got, cv2.COLOR_BGR2HSV)[:, :, 1] if got is not None else None
        more = ex4_scale_saturation(bgr, 1.5)
        s0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        s_more = cv2.cvtColor(more, cv2.COLOR_BGR2HSV)[:, :, 1].mean() if more is not None else -1
        return (s is not None and int(s.max()) == 0 and s_more > s0,
                "S だけスケール（0で無彩・1.5で増加）")
    check("ex4_scale_saturation", _c4)

    def _c5():
        got = ex5_shift_hue(bgr, 30)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int32) + 30) % 180).astype(np.uint8)
        ref = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return (_close(got, ref, 1), "色相 +30（mod 180）")
    check("ex5_shift_hue", _c5)

    def _c6():
        got = ex6_equalize_luminance(bgr)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
        ref = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        return (_close(got, ref, 1), "Y だけ equalizeHist")
    check("ex6_equalize_luminance", _c6)

    def _c7():
        casted = (bgr.astype(np.float32) * np.array([1.4, 1.0, 0.8])).clip(0, 255).astype(np.uint8)
        got = ex7_gray_world_wb(casted)
        if got is None:
            return (False, "None")
        m = got.reshape(-1, 3).mean(axis=0)
        # WB 後は各ch平均がほぼ揃う（ばらつきが小さい）。
        return (float(m.max() - m.min()) < 2.0, "gray-world で各ch平均が揃う")
    check("ex7_gray_world_wb", _c7)

    def _c8():
        got = ex8_skin_mask_ycrcb(bgr)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ref = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))
        return (got is not None and got.shape == bgr.shape[:2]
                and np.array_equal(got, ref), "YCrCb 肌色 inRange")
    check("ex8_skin_mask_ycrcb", _c8)

    def _c9():
        other = cv2.convertScaleAbs(bgr, alpha=1.0, beta=20)
        got = ex9_mean_delta_e(bgr, other)
        la = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        lb = cv2.cvtColor(other.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        ref = float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())
        same = ex9_mean_delta_e(bgr, bgr)
        return (got is not None and abs(got - ref) < 1e-3 and abs(same) < 1e-6,
                "float Lab の平均ΔE（同一画像は0）")
    check("ex9_mean_delta_e", _c9)

    def _c10():
        got = ex10_clahe_luminance(bgr, 2.0, 8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        ref = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        return (_close(got, ref, 1), "L にだけ CLAHE")
    check("ex10_clahe_luminance", _c10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {n_pass}/{len(results)} 問 PASS")
    print("ALL PASS 🎉" if all_ok else "まだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替えて実行）
# 完全な解答は exercises_solutions.py にもある。まずは自力で解いてから見ること。
# =====================================================================

def _sol_ex1(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _sol_ex2(bgr, alpha, beta):
    return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)


def _sol_ex3(gamma):
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip((x ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)


def _sol_ex4(bgr, factor):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _sol_ex5(bgr, delta):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int32) + delta) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _sol_ex6(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def _sol_ex7(bgr):
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    gains = gray / np.clip(means, 1e-6, None)
    return np.clip(f * gains[None, None, :], 0, 255).astype(np.uint8)


def _sol_ex8(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))


def _sol_ex9(bgr_a, bgr_b):
    la = cv2.cvtColor(bgr_a.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lb = cv2.cvtColor(bgr_b.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())


def _sol_ex10(bgr, clip_limit=2.0, tile=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_to_hsv"] = _sol_ex1
    g["ex2_brightness_contrast"] = _sol_ex2
    g["ex3_gamma_lut"] = _sol_ex3
    g["ex4_scale_saturation"] = _sol_ex4
    g["ex5_shift_hue"] = _sol_ex5
    g["ex6_equalize_luminance"] = _sol_ex6
    g["ex7_gray_world_wb"] = _sol_ex7
    g["ex8_skin_mask_ycrcb"] = _sol_ex8
    g["ex9_mean_delta_e"] = _sol_ex9
    g["ex10_clahe_luminance"] = _sol_ex10


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
