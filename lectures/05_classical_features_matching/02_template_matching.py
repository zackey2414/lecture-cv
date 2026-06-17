"""02: テンプレートマッチング — matchTemplate/minMaxLoc とマルチスケール探索、その限界。

学ぶこと:
  - 最も単純な物体探索: テンプレート画像をシーン上で滑らせ、相関が最大の位置を探す。
  - cv2.matchTemplate の出力は「相関マップ」、cv2.minMaxLoc で最良位置を取る。
  - 手法ごとの最大/最小の向き: TM_CCOEFF/CCORR_NORMED は max、TM_SQDIFF は min。
  - 限界: 回転・スケールに不変でない。対象が回ると/大きさが変わるとスコアが急落する。
  - マルチスケール探索: テンプレートを複数サイズに変えて総当たりし、大きさ違いを吸収する。
  - これが「だから局所特徴量(SIFT/ORB)が要る」という動機づけになる。

実行:
  uv run python lectures/05_classical_features_matching/02_template_matching.py
結果は lectures/05_classical_features_matching/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_object_scene, output_dir  # noqa: E402


def find_best_match(scene: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    """TM_CCOEFF_NORMED で 1 回だけマッチングし、(最大スコア, 左上座標(x,y)) を返す。

    TM_CCOEFF_NORMED は正規化相互相関の一種で、明るさの違いに比較的強く、
    値域がおおむね -1〜1。1 に近いほど良い一致なので minMaxLoc の「最大」を採用する。
    """
    result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
    # minMaxLoc は (最小値, 最大値, 最小位置, 最大位置) を返す。CCOEFF 系は最大位置が答え。
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    return float(max_val), max_loc


def main() -> None:
    out = output_dir()

    # === 1. シーンとテンプレートを用意（正解位置を保持） ====================
    scene, template, true_tl, true_br = make_object_scene()
    th, tw = template.shape[:2]
    cv2.imwrite(str(out / "02_scene.png"), scene)
    cv2.imwrite(str(out / "02_template.png"), template)
    print(f"[1] テンプレート {tw}x{th}px、正解の左上={true_tl}")

    # === 2. 基本のマッチング（同一サイズ・無回転なら一発で決まる） ==========
    score, top_left = find_best_match(scene, template)
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    vis = scene.copy()
    cv2.rectangle(vis, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imwrite(str(out / "02_found_basic.png"), vis)
    # 正解位置とのズレ（ピクセル）。同条件なら 0 付近になるはず。
    err = abs(top_left[0] - true_tl[0]) + abs(top_left[1] - true_tl[1])
    print(f"[2] 基本マッチング: score={score:.3f}, 検出左上={top_left}, 正解とのズレ={err}px")

    # === 3. 手法による「最大/最小」の違い ===================================
    # TM_SQDIFF/SQDIFF_NORMED は「差の二乗和」なので小さいほど良い→ minLoc を使う。
    print("\n[3] 6 手法のスコアと最良位置（NORMED 系を使うのが無難）")
    methods = [
        ("TM_CCOEFF_NORMED", cv2.TM_CCOEFF_NORMED, "max"),
        ("TM_CCORR_NORMED", cv2.TM_CCORR_NORMED, "max"),
        ("TM_SQDIFF_NORMED", cv2.TM_SQDIFF_NORMED, "min"),
    ]
    for name, method, which in methods:
        res = cv2.matchTemplate(scene, template, method)
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
        loc = min_l if which == "min" else max_l
        val = min_v if which == "min" else max_v
        print(f"  {name:18s} 採用={which:3s} 値={val:7.3f} 位置={loc}")

    # === 4. 限界その1: 回転に弱い =========================================
    # シーンを 35 度回しただけで、同じテンプレートのスコアは大きく落ちる。
    h, w = scene.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), 35, 1.0)
    rotated = cv2.warpAffine(scene, rot_mat, (w, h), borderValue=(60, 60, 60))
    rot_score, _ = find_best_match(rotated, template)
    cv2.imwrite(str(out / "02_scene_rotated.png"), rotated)
    print(f"\n[4] 限界(回転): 無回転 score={score:.3f} → 35度回転 score={rot_score:.3f}"
          "（急落。テンプレートマッチングは回転不変でない）")

    # === 5. 限界その2: スケールに弱い → マルチスケールで吸収 ================
    # シーンを 1.3 倍に拡大すると、固定サイズのテンプレートでは一致が悪くなる。
    big = cv2.resize(scene, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
    single_score, _ = find_best_match(big, template)

    # マルチスケール: テンプレートを複数倍率に変え、最もスコアの高い倍率を探す。
    print("\n[5] スケール限界とマルチスケール探索（1.3 倍に拡大したシーンを探す）")
    print(f"  {'scale':>6s} {'score':>7s}")
    best = (-1.0, 1.0, None)  # (score, scale, top_left)
    for scale in np.linspace(0.6, 1.6, 21):
        # 縮小は INTER_AREA、拡大は INTER_CUBIC が定石。
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        tmpl_s = cv2.resize(template, None, fx=scale, fy=scale, interpolation=interp)
        # テンプレートがシーンより大きいと matchTemplate はエラーになるのでスキップ。
        if tmpl_s.shape[0] >= big.shape[0] or tmpl_s.shape[1] >= big.shape[1]:
            continue
        s, loc = find_best_match(big, tmpl_s)
        if abs(scale - round(scale, 1)) < 1e-9 and round(scale, 1) in (0.6, 0.8, 1.0, 1.3, 1.6):
            print(f"  {scale:6.2f} {s:7.3f}")
        if s > best[0]:
            best = (s, scale, loc)

    print(f"  → 単一スケール(1.0倍テンプレート)の score={single_score:.3f}")
    print(f"  → マルチスケール最良: score={best[0]:.3f} at scale={best[1]:.2f}"
          "（正しい倍率を当てるとスコアが回復する）")

    # マルチスケール最良の検出枠を可視化。
    if best[2] is not None:
        s_best = best[1]
        bw = int(template.shape[1] * s_best)
        bh = int(template.shape[0] * s_best)
        tl = best[2]
        vis_big = big.copy()
        cv2.rectangle(vis_big, tl, (tl[0] + bw, tl[1] + bh), (0, 255, 0), 2)
        cv2.imwrite(str(out / "02_found_multiscale.png"), vis_big)

    print("\n[6] まとめ: テンプレートマッチングは「同じ見た目・同じ向き・同じ大きさ」なら最強に簡単。")
    print("     だが回転・スケール・視点変化には弱い。そこを乗り越えるのが次の局所特徴量(SIFT/ORB)。")
    print("\n完了。lectures/05_classical_features_matching/outputs/ の 02_*.png を確認してください。")


if __name__ == "__main__":
    main()
