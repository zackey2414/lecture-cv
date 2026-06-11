"""01: 局所特徴量で 2 枚をマッチング — SIFT/ORB・knnMatch・Lowe 比率テスト・RANSAC 評価。

学ぶこと:
  - 局所特徴量の流れ: detectAndCompute で「キーポイント＋記述子」を得る。
  - 記述子の種類で距離が変わる: SIFT は浮動小数(NORM_L2)、ORB は2進(NORM_HAMMING)。
  - knnMatch(k=2) + Lowe の比率テスト(0.75) で誤対応を捨てる定番パターン。
  - findHomography(RANSAC) のインライア率でマッチング品質を「数値で」評価する。
  - 比率しきい値を 0.5〜0.9 で振って、なぜ 0.75 が定番なのかを実験で確かめる。
  - FlannBasedMatcher（近似最近傍）と crossCheck の使いどころ。
  - SIFT と ORB の速度・特徴点数・頑健性の違いを同じ画像で比較する。

実行:
  uv run python lectures/05_classical_features_matching/01_sift_orb_match.py
結果は outputs/05_classical_features_matching/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys
import time

import cv2
import numpy as np

# 同じフォルダの cv_helpers.py を確実に import できるようにする。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_scene_bgr, output_dir, warp_to_view  # noqa: E402


def ratio_test(knn_matches: list, ratio: float = 0.75) -> list:
    """Lowe の比率テスト。最近傍 m が 2 番目の近傍 n より十分近いものだけ残す。

    knnMatch は各クエリ点に対し近い順 2 個 [m, n] を返す。m.distance < ratio * n.distance なら
    「2 番目を大きく引き離して一番近い＝信頼できる対応」とみなす。
    knnMatch は近傍が 1 個しか無いと長さ 1 のペアを返すことがあるので、len==2 を必ず確認する。
    """
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue  # 近傍が 1 個しかない＝比較できないので捨てる（落とさず早期 continue）。
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def match_and_evaluate(detector, norm_type: int, gray1: np.ndarray, gray2: np.ndarray,
                       kp1=None, des1=None) -> dict:
    """1 つの検出器で「検出→knnMatch→比率テスト→RANSAC」を一気に走らせ、指標を辞書で返す。

    返り値: keypoints 数、good マッチ数、インライア数、インライア率、所要時間、可視化に使う中間物。
    SIFT/ORB の両方で同じ手順を踏むので、ここに 1 つにまとめて使い回す。
    """
    t0 = time.perf_counter()
    # detectAndCompute はキーポイント列と記述子行列(N, D)を同時に返す。
    if kp1 is None or des1 is None:
        kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # BFMatcher は総当たりで最近傍を探す（小〜中規模なら十分速く、正確）。
    # norm_type を記述子に合わせるのが肝: SIFT=NORM_L2, ORB=NORM_HAMMING。
    bf = cv2.BFMatcher(norm_type)
    knn = bf.knnMatch(des1, des2, k=2)
    good = ratio_test(knn, ratio=0.75)
    elapsed = time.perf_counter() - t0

    # 良マッチが 4 組以上あればホモグラフィを推定できる（最低 4 点必要）。
    inliers = 0
    mask = None
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # RANSAC は外れ値に強い推定法。mask[i]==1 がインライア（幾何的に整合する対応）。
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
        if mask is not None:
            inliers = int(mask.sum())

    total = len(good)
    return {
        "kp1": kp1, "kp2": kp2,
        "n_kp1": len(kp1), "n_kp2": len(kp2),
        "good": good, "n_good": total,
        "inliers": inliers,
        "inlier_ratio": (inliers / total) if total else 0.0,
        "mask": mask,
        "time": elapsed,
    }


def draw_inlier_matches(img1, kp1, img2, kp2, good, mask) -> np.ndarray:
    """RANSAC のインライアだけを線で結んだ対応図を作る（外れ値を描かないので見やすい）。"""
    if mask is None:
        matches_mask = None
    else:
        matches_mask = mask.ravel().tolist()  # 1=描く, 0=描かない
    return cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        matchColor=(0, 255, 0), singlePointColor=None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def main() -> None:
    out = output_dir()

    # === 1. 同一シーンの「別視点ペア」を用意する =============================
    # 正解のホモグラフィ H が分かっている 2 枚を作る（評価の物差しになる）。
    scene = make_scene_bgr()
    view, H_true = warp_to_view(scene)
    cv2.imwrite(str(out / "01_scene.png"), scene)
    cv2.imwrite(str(out / "01_view.png"), view)

    # 特徴点検出はグレースケールで行うのが定石（色ではなく明暗の構造で特徴を拾う）。
    g1 = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

    print("[1] 別視点ペアを作成。正解ホモグラフィ H_true を保持（評価の基準）。")

    # === 2. SIFT と ORB を同じ画像で比較 ===================================
    # SIFT: 浮動小数記述子 → NORM_L2。精度・スケール/回転不変性が高いが計算は重め。
    # ORB : 2 進記述子      → NORM_HAMMING。特許フリーで CPU でも非常に高速。
    detectors = [
        ("SIFT", cv2.SIFT_create(), cv2.NORM_L2),
        ("ORB", cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING),
    ]

    print("\n[2] SIFT vs ORB（同一ペアでの検出・マッチング・RANSAC 評価）")
    print(f"  {'検出器':6s} {'kp1':>5s} {'kp2':>5s} {'good':>6s} {'inliers':>8s} "
          f"{'inlier率':>8s} {'時間[s]':>8s}")
    results = {}
    for name, det, norm in detectors:
        r = match_and_evaluate(det, norm, g1, g2)
        results[name] = r
        print(f"  {name:6s} {r['n_kp1']:5d} {r['n_kp2']:5d} {r['n_good']:6d} "
              f"{r['inliers']:8d} {r['inlier_ratio']:8.2f} {r['time']:8.3f}")

        # キーポイント可視化（向きと大きさを円で表示）。BGR のまま保存。
        kp_vis = cv2.drawKeypoints(
            scene, r["kp1"], None, color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(str(out / f"01_{name.lower()}_keypoints.png"), kp_vis)

        # インライア対応の可視化。
        match_vis = draw_inlier_matches(scene, r["kp1"], view, r["kp2"], r["good"], r["mask"])
        cv2.imwrite(str(out / f"01_{name.lower()}_matches.png"), match_vis)

    # === 3. FlannBasedMatcher（近似最近傍）の使い方 =========================
    # BFMatcher は総当たりで正確だが点数が増えると遅い。FLANN は近似で高速。
    # SIFT(浮動小数)は KDTree、ORB(2進)は LSH と、記述子で index 種別が変わる点に注意。
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),   # algorithm=1: KDTree（SIFT の浮動小数記述子向け）
        dict(checks=50),              # 探索の打ち切り回数（大きいほど正確・遅い）
    )
    flann_good = ratio_test(flann.knnMatch(d1, d2, k=2), ratio=0.75)
    print(f"\n[3] FLANN(KDTree)+SIFT の good マッチ数: {len(flann_good)}"
          f"（BF と近い数になる。ORB で FLANN を使うなら algorithm=6 の LSH を指定する）")

    # === 4. crossCheck という別の誤対応除去 ================================
    # 比率テストの代わりに「A→B の最近傍と B→A の最近傍が一致する対応だけ残す」方法。
    # crossCheck=True のときは knnMatch ではなく match を使う（k=1 相当のため）。
    bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB_create(nfeatures=2000)
    ok1, od1 = orb.detectAndCompute(g1, None)
    ok2, od2 = orb.detectAndCompute(g2, None)
    cross = bf_cross.match(od1, od2)
    print(f"[4] crossCheck(ORB) の対応数: {len(cross)}"
          f"（比率テストと crossCheck はどちらか一方を使う。両者の併用はしない）")

    # === 5. 比率しきい値スイープ（なぜ 0.75 か） ============================
    # しきい値を上げるほど多く残るが誤対応も増え、インライア率は下がりやすい。
    # 0.75 付近が「十分な数」と「高い純度」のバランス点になることを数値で確認する。
    print("\n[5] Lowe 比率しきい値のスイープ（ORB, 値が小さいほど厳しい）")
    print(f"  {'ratio':>6s} {'good':>6s} {'inliers':>8s} {'inlier率':>8s}")
    knn_orb = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(od1, od2, k=2)
    for ratio in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        good = ratio_test(knn_orb, ratio=ratio)
        inl, tot = 0, len(good)
        if tot >= 4:
            src = np.float32([ok1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([ok2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inl = int(mask.sum()) if mask is not None else 0
        print(f"  {ratio:6.2f} {tot:6d} {inl:8d} {(inl / tot if tot else 0):8.2f}")

    # === 6. まとめ ========================================================
    sift_r, orb_r = results["SIFT"], results["ORB"]
    print("\n[6] まとめ")
    print(f"  - 所要時間: SIFT={sift_r['time']:.3f}s / ORB={orb_r['time']:.3f}s"
          "（ORB は nfeatures=2000 と点数を多く取らせているので単純比較はできないが、"
          "1 点あたりの計算は ORB が圧倒的に軽い）。")
    print(f"  - 検出点数: SIFT={sift_r['n_kp1']} / ORB={orb_r['n_kp1']}。"
          "SIFT は少数精鋭、ORB は点数で押す戦略。")
    print(f"  - インライア率: SIFT={sift_r['inlier_ratio']:.2f} / ORB={orb_r['inlier_ratio']:.2f}。"
          "どちらも比率テスト＋RANSAC で高純度の対応が得られている。")
    print("\n完了。outputs/05_classical_features_matching/ の *_keypoints.png / *_matches.png を確認。")


if __name__ == "__main__":
    main()
