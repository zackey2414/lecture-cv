"""mini_project.py — 章末ミニプロジェクト: 散らかったシーンの中から平面物体を検出する。

この回の学びを 1 本のパイプラインに統合する「完成形」:

    特徴に富んだ物体画像(object) を既知のホモグラフィ H_place で回転＋遠近変形し、
    妨害物だらけの背景に貼り込んだ「散らかったシーン(scene)」を作る。
      → ① 特徴抽出: ORB / SIFT で object・scene の双方からキーポイント＋記述子を得る
      → ② マッチング: knnMatch(k=2) + Lowe の比率テスト(0.75) で良マッチを選ぶ
      → ③ 位置推定: findHomography(RANSAC) で object→scene の変換 H を推定
      → ④ 投影: object の 4 隅を perspectiveTransform で scene に投影し、枠を描く
      → ⑤ 評価: 真の貼り込み 4 隅との「コーナー誤差(px)」とインライア率を数値化
      → ⑥ 対比: 素朴なテンプレートマッチングが「回転＋遠近」で破綻することを確認

外部データもネットも GPU も不要（合成画像のみ・CPU 完結）。実運用では object を
「探したい商品パッケージ/看板/書影」に、scene を「カメラ画像」に差し替えればそのまま動く。
これは次回(第6回)のホモグラフィ・パノラマ合成へ直結する中核部品でもある。

実行:
    uv run python lectures/05_classical_features_matching/mini_project.py
結果（画像・JSON）は outputs/05_classical_features_matching/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg を指定する。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同じフォルダの cv_helpers.py を確実に import できるようにする。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_scene_bgr, output_dir  # noqa: E402


# =====================================================================
# 部品（単一責務で小さく分ける）
# =====================================================================

def ratio_test(knn_matches: list, ratio: float = 0.75) -> list:
    """Lowe の比率テスト。最近傍 m が 2 番目 n を ratio 倍以上引き離す対応だけ残す。

    knnMatch は近傍が 1 個しか無いと長さ 1 のペアを返すことがあるので、
    len==2 を必ず確認してから展開する（怠ると ValueError で落ちる）。
    """
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def make_object_and_scene(
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """探したい「物体画像」と、それを回転＋遠近変形して貼り込んだ「散らかったシーン」を作る。

    返り値: (object_bgr, scene_bgr, true_corners)。
      - object_bgr: 特徴に富んだ平面物体（make_scene_bgr を流用して角・文字・ブロブを増やす）。
      - scene_bgr : 妨害物（ランダム図形）だらけの背景に object を H_place で貼り込んだ画像。
      - true_corners: シーン中での object の 4 隅の真値 (4,2) float32（評価の物差し）。
    """
    # --- 探したい物体（小さめ・特徴に富む） ---
    obj = make_scene_bgr(height=240, width=320, seed=seed)
    oh, ow = obj.shape[:2]

    # --- 背景キャンバス（妨害物を散らす） ---
    rng = np.random.default_rng(seed + 1)
    sh, sw = 720, 1000
    scene = np.zeros((sh, sw, 3), dtype=np.uint8)
    gx = np.linspace(20, 110, sw, dtype=np.uint8)  # ゆるい横グラデ背景
    scene[:] = np.repeat(gx[None, :, None], sh, axis=0)
    for _ in range(60):  # 妨害物（誤マッチを誘う）
        x = int(rng.integers(0, sw))
        y = int(rng.integers(0, sh))
        size = int(rng.integers(15, 70))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(scene, (x, y), (x + size, y + size), color, thickness=-1)
        else:
            cv2.circle(scene, (x, y), size // 2, color, thickness=-1)

    # --- 物体を「回転＋遠近」で貼り込む（真のホモグラフィ H_place は分かっている） ---
    src_quad = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
    # シーン内に傾けた台形として配置（4 隅をわざと歪ませてパースを作る）。
    dst_quad = np.float32([[520, 90], [930, 180], [860, 560], [470, 470]])
    h_place = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(obj, h_place, (sw, sh))
    # 物体領域だけを合成するためのマスク（255 で塗った同形状を同じ H で飛ばす）。
    mask = cv2.warpPerspective(np.full((oh, ow), 255, np.uint8), h_place, (sw, sh))
    scene[mask > 0] = warped[mask > 0]
    return obj, scene, dst_quad


def detect_object(
    obj_gray: np.ndarray, scene_gray: np.ndarray, detector, norm_type: int
) -> dict:
    """1 つの検出器で「特徴抽出→比率テスト→RANSAC→4 隅投影」を実行し、結果を辞書で返す。

    object を query、scene を train としてマッチングする（object の点を scene へ投影したい）。
    返り値には可視化に使う中間物（キーポイント・良マッチ・投影四隅）も含める。
    """
    kp_o, des_o = detector.detectAndCompute(obj_gray, None)
    kp_s, des_s = detector.detectAndCompute(scene_gray, None)

    result: dict = {
        "n_kp_obj": len(kp_o), "n_kp_scene": len(kp_s),
        "n_good": 0, "inliers": 0, "inlier_ratio": 0.0,
        "kp_obj": kp_o, "kp_scene": kp_s, "good": [], "mask": None,
        "proj_corners": None, "found": False,
    }
    # 記述子が空（特徴ゼロ）なら早期 return（落とさない）。
    if des_o is None or des_s is None:
        return result

    knn = cv2.BFMatcher(norm_type).knnMatch(des_o, des_s, k=2)
    good = ratio_test(knn, ratio=0.75)
    result["good"] = good
    result["n_good"] = len(good)
    if len(good) < 4:  # findHomography には最低 4 組必要
        return result

    src = np.float32([kp_o[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return result

    inliers = int(mask.sum())
    result["mask"] = mask
    result["inliers"] = inliers
    result["inlier_ratio"] = inliers / len(good)

    # object の 4 隅を scene 座標へ投影（perspectiveTransform は (N,1,2) を取る）。
    oh, ow = obj_gray.shape[:2]
    corners = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    result["proj_corners"] = proj
    result["found"] = True
    return result


def corner_error(proj: np.ndarray, true_corners: np.ndarray) -> float:
    """投影した 4 隅と真の 4 隅の平均ユークリッド距離（px）。小さいほど正確。"""
    return float(np.mean(np.linalg.norm(proj - true_corners, axis=1)))


def template_baseline(scene_bgr: np.ndarray, obj_bgr: np.ndarray) -> tuple[float, tuple[int, int]]:
    """対比用: 素朴なテンプレートマッチング。回転＋遠近の物体には当たらないことを示す。"""
    res = cv2.matchTemplate(scene_bgr, obj_bgr, cv2.TM_CCOEFF_NORMED)
    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
    return float(max_v), max_l


# =====================================================================
# メイン
# =====================================================================

def main() -> None:
    out = output_dir()

    # === 1. 課題シーンを用意（真の 4 隅を保持） ============================
    obj, scene, true_corners = make_object_and_scene()
    cv2.imwrite(str(out / "mini_project_object.png"), obj)
    cv2.imwrite(str(out / "mini_project_scene.png"), scene)
    obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    print("=" * 64)
    print("章末ミニプロジェクト: 散らかったシーンからの平面物体検出")
    print(f"  物体サイズ={obj.shape[1]}x{obj.shape[0]}px / シーン={scene.shape[1]}x{scene.shape[0]}px")

    # === 2. ORB と SIFT で検出（特徴→比率テスト→RANSAC→4 隅投影） ========
    detectors = [
        ("ORB", cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING),
        ("SIFT", cv2.SIFT_create(), cv2.NORM_L2),
    ]
    metrics: dict = {"detectors": {}}
    vis_results: dict = {}
    print("\n  検出器ごとの結果:")
    print(f"  {'name':5s} {'kp_obj':>7s} {'kp_scn':>7s} {'good':>5s} "
          f"{'inl':>4s} {'inl率':>6s} {'隅誤差px':>8s} {'判定':>5s}")
    for name, det, norm in detectors:
        r = detect_object(obj_gray, scene_gray, det, norm)
        err = corner_error(r["proj_corners"], true_corners) if r["found"] else float("nan")
        # コーナー誤差が小さい（=正しい場所を当てた）かを判定の物差しにする。
        ok = bool(r["found"] and err < 25.0)
        metrics["detectors"][name] = {
            "n_kp_obj": r["n_kp_obj"], "n_kp_scene": r["n_kp_scene"],
            "n_good": r["n_good"], "inliers": r["inliers"],
            "inlier_ratio": round(r["inlier_ratio"], 4),
            "corner_error_px": None if r["found"] is False else round(err, 2),
            "localized": ok,
        }
        print(f"  {name:5s} {r['n_kp_obj']:7d} {r['n_kp_scene']:7d} {r['n_good']:5d} "
              f"{r['inliers']:4d} {r['inlier_ratio']:6.2f} "
              f"{err:8.2f} {'OK' if ok else 'NG':>5s}")

        # 可視化: 投影四隅をシーンに描く（ある検出器の結果を 1 枚に）。
        vis = scene.copy()
        color = (0, 255, 0) if name == "ORB" else (0, 200, 255)
        if r["found"]:
            poly = r["proj_corners"].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=3,
                          lineType=cv2.LINE_AA)
            cv2.putText(vis, f"{name} inl={r['inliers']} err={err:.1f}px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        # 真の四隅も細い白線で重ねる（答え合わせ用）。
        cv2.polylines(vis, [true_corners.astype(np.int32).reshape(-1, 1, 2)],
                      isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(out / f"mini_project_detect_{name.lower()}.png"), vis)
        vis_results[name] = vis

    # === 3. 対比: 素朴なテンプレートマッチングは破綻する ==================
    tm_score, tm_loc = template_baseline(scene, obj)
    metrics["template_baseline"] = {"best_score": round(tm_score, 3), "best_loc": list(tm_loc)}
    print("\n  対比（テンプレートマッチング, 回転＋遠近の物体を未変形テンプレで探索）:")
    print(f"    best score={tm_score:.3f}（特徴点法のインライア率に比べ低い＝当てにならない）")
    print("    → 回転・スケール・遠近に弱いテンプレ法では平面物体を頑健に追えない。")
    print("       だから回転/スケール不変な局所特徴量(ORB/SIFT)が要る、という動機が閉じる。")

    # === 4. まとめ図（matplotlib, Agg, BGR→RGB に注意） ===================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(cv2.cvtColor(obj, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("object (template)")
    axes[0, 1].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("cluttered scene (warped object hidden)")
    axes[1, 0].imshow(cv2.cvtColor(vis_results["ORB"], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("ORB detection")
    axes[1, 1].imshow(cv2.cvtColor(vis_results["SIFT"], cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("SIFT detection")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "mini_project_summary.png"), dpi=110)
    plt.close(fig)

    # === 5. 指標を JSON で保存（再現性・自動評価のため） ==================
    json_path = out / "mini_project_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n  保存物:")
    print(f"    - 検出結果図: mini_project_detect_orb.png / mini_project_detect_sift.png")
    print(f"    - まとめ図  : mini_project_summary.png")
    print(f"    - 指標 JSON : {json_path.name}")
    print("\n完成: 特徴抽出→比率テスト→RANSAC→四隅投影、の一連を外部依存ゼロで回せた。")
    print("発展: object を実物の書影/看板に、scene をカメラ画像に差し替えれば実運用へ。")


if __name__ == "__main__":
    main()
