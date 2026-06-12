"""第6回 章末ミニプロジェクト — 4枚パノラマ合成 + 平面物体のまっすぐ化 + 品質レポート。

この回の学び（特徴マッチ → findHomography(RANSAC) → 再投影誤差 → キャンバス → warp →
フェザー合成 → 複数枚のホモグラフィ合成 → cv2.Stitcher 比較 → SSIM 評価 → 平面物体の
位置合わせ/まっすぐ化）を1本のパイプラインに統合する総合課題の完成形です。

やること（すべて CPU・合成データ・ネット/カメラ不要・headless 安全）:
  [A] 同じ平面シーンから重なりのある4視点を合成生成する。
  [B] 手作りパイプラインで4枚を1枚のパノラマに合成し、品質を数値化する。
  [C] cv2.Stitcher（自動）でも合成し、手作り版と重なり領域 SSIM で比較する。
  [D] 平面物体（書類カード）を散らかったシーンに射影で貼り込み、検出して四辺形を描き、
      逆ホモグラフィで「正面（まっすぐ）」に復元（rectify）して忠実度を SSIM で測る。
  [E] すべての指標を JSON に、図を PNG にまとめて outputs/06_homography_panorama/ に保存。

実行:
  uv run python lectures/06_homography_panorama/mini_project.py

注: 01〜03 の番号付きスクリプトは import できないため、Stitcher 実行・重なり SSIM・
平面物体の検出/復元はこのファイル内に自己完結で実装している。共通の合成データ生成や
findHomography/再投影誤差/SSIM などの低レベル部品だけ cv_helpers から借りる。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import cv2
import numpy as np

# 表示はせず保存するので matplotlib は Agg バックエンド（headless 安全）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同じフォルダの cv_helpers.py（数字始まりでないので import 可）から部品を借りる。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    build_panorama,
    estimate_homography,
    make_scene,
    orb_match,
    output_dir,
    points_from_matches,
    reprojection_error,
    ssim,
    warp_corners,
)

# RANSAC・ORB の乱数を固定して、毎回同じ結果（再現可能）にする。
cv2.setRNGSeed(0)


# =====================================================================
# [A] 同じ平面シーンから重なりのある N 視点を合成する（自己完結）
# =====================================================================

def _camera_quad(center_x: int, skew: int, top: int = 80, bottom: int = 680,
                 half_width: int = 360) -> np.ndarray:
    """シーン座標上で「カメラが切り取る四辺形」を作る（左上→右上→右下→左下）。

    skew で上下辺を左右にずらして台形にし、平行移動でなく本物の射影変換にする。
    """
    x0, x1 = center_x - half_width, center_x + half_width
    return np.float32([
        [x0 + skew, top + 20],     # 左上
        [x1 + skew, top - 10],     # 右上
        [x1 - skew, bottom + 15],  # 右下
        [x0 - skew, bottom - 5],   # 左下
    ])


def make_overlapping_views(n: int = 4, out_w: int = 640, out_h: int = 480,
                           seed: int = 0) -> list[np.ndarray]:
    """同じ平面シーンを左→右に少しずつずらして撮った n 視点（BGR）を返す。"""
    scene = make_scene(seed=seed)  # 特徴の豊富な平面（1700x760）
    width = scene.shape[1]
    margin = 360
    centers = np.linspace(margin, width - margin, n).astype(int)
    skews = [22 if i % 2 == 0 else -16 for i in range(n)]  # 首振りを交互に
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    views = []
    for cx, skew in zip(centers, skews):
        quad = _camera_quad(int(cx), skew)
        H = cv2.getPerspectiveTransform(quad, dst)  # シーン→視点
        views.append(cv2.warpPerspective(scene, H, (out_w, out_h)))
    return views


# =====================================================================
# [C] cv2.Stitcher（自動合成）と重なり領域 SSIM（自己完結）
# =====================================================================

_STATUS = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


def run_stitcher(images: list[np.ndarray]) -> tuple[np.ndarray | None, str, float]:
    """PANORAMA で試し、ダメなら SCANS にフォールバックして自動合成する。

    status を必ず確認し、失敗は握りつぶさず None を返す（呼び出し側で exit を正常に保つ）。
    """
    for mode_name, mode in (("PANORAMA", cv2.Stitcher_PANORAMA),
                            ("SCANS", cv2.Stitcher_SCANS)):
        stitcher = cv2.Stitcher_create(mode)
        t0 = time.perf_counter()
        status, pano = stitcher.stitch(images)
        dt = time.perf_counter() - t0
        print(f"  Stitcher({mode_name}): {_STATUS.get(status, status)}  {dt:.2f}s")
        if status == cv2.Stitcher_OK:
            return pano, mode_name, dt
    return None, "FAILED", 0.0


def overlap_ssim(pano_ref: np.ndarray, pano_other: np.ndarray) -> float | None:
    """2つのパノラマを位置合わせ(other→ref)し、両方が中身を持つ重なり領域で SSIM を測る。

    キャンバスの座標系が違うので、まず ORB+findHomography で other を ref に重ねる。
    対応が取れない/重なりが狭すぎる場合は None（評価不能でも exit は正常）。
    """
    kp1, kp2, good = orb_match(pano_ref, pano_other, ratio=0.8)
    if len(good) < 10:
        return None
    pts_ref, pts_other = points_from_matches(kp1, kp2, good)
    H, _ = estimate_homography(pts_other, pts_ref)
    if H is None:
        return None
    size = (pano_ref.shape[1], pano_ref.shape[0])
    aligned = cv2.warpPerspective(pano_other, H, size)
    valid = cv2.warpPerspective(np.full(pano_other.shape[:2], 255, np.uint8), H, size)
    both = (valid > 0) & (pano_ref.sum(axis=2) > 0)
    if both.sum() < 1000:
        return None
    g_ref = cv2.cvtColor(pano_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_other = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_ref = np.where(both, g_ref, 0)
    g_other = np.where(both, g_other, 0)
    return ssim(g_ref, g_other)


# =====================================================================
# [D] 平面物体（書類カード）の合成・検出・まっすぐ化（自己完結）
# =====================================================================

def make_document(width: int = 360, height: int = 480) -> np.ndarray:
    """特徴の多い「書類カード」を合成する（タイトル・本文行・図形・枠）。"""
    doc = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(doc, (10, 10), (width - 10, height - 10), (150, 60, 60), 5)
    cv2.putText(doc, "REPORT", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (30, 30, 190), 3, cv2.LINE_AA)
    cv2.putText(doc, "homography", (28, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (60, 120, 30), 2, cv2.LINE_AA)
    # 本文を模した横線（角が ORB の良い特徴になる）。
    for i, y in enumerate(range(160, height - 60, 26)):
        x2 = width - 30 if i % 3 else width - 90
        cv2.line(doc, (28, y), (x2, y), (90, 90, 90), 2, cv2.LINE_AA)
    cv2.circle(doc, (width - 60, 60), 22, (0, 150, 220), -1)
    cv2.rectangle(doc, (28, height - 70), (130, height - 30), (200, 120, 0), -1)
    return doc


def make_doc_background(height: int, width: int, seed: int = 5) -> np.ndarray:
    """書類を貼る「低周波な背景」を作る。

    平面物体検出は『貼った物体の特徴が背景の特徴より際立つ』ことが肝心。
    背景に角(コーナー)が多いと ORB が背景にも食いついて誤対応が増えるので、
    ここでは大きな色面＋強めのぼかしで角をほぼ消し、書類カードを主役にする。
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(2, 2, 3)).astype(np.uint8)
    bg = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
    for _ in range(8):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(70, 150))
        color = tuple(int(c) for c in rng.integers(40, 210, size=3))
        overlay = bg.copy()
        cv2.circle(overlay, (cx, cy), r, color, -1)
        bg = cv2.addWeighted(overlay, 0.35, bg, 0.65, 0)
    return cv2.GaussianBlur(bg, (31, 31), 0)  # 角を消して背景の特徴を弱める


def paste_planar(scene: np.ndarray, obj: np.ndarray, dst_quad: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """obj をシーンの四辺形 dst_quad へ射影変換で貼り付け、(合成シーン, obj→scene の H) を返す。"""
    oh, ow = obj.shape[:2]
    src_quad = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
    H = cv2.getPerspectiveTransform(src_quad, dst_quad)  # obj 座標 → scene 座標
    warped = cv2.warpPerspective(obj, H, (scene.shape[1], scene.shape[0]))
    mask = cv2.warpPerspective(np.full((oh, ow), 255, np.uint8), H,
                               (scene.shape[1], scene.shape[0]))
    out = scene.copy()
    out[mask > 0] = warped[mask > 0]
    return out, H


def detect_and_rectify(doc: np.ndarray, scene: np.ndarray
                       ) -> dict:
    """シーン内の平面物体を検出し、四辺形を描き、逆 H で正面へ復元(rectify)する。

    返り値 dict: good, inliers, reproj_error, quad, detected画像, rectified画像,
    rectify_ssim（復元した正面 vs 元カードの構造類似度）。検出失敗時は found=False。
    """
    dh, dw = doc.shape[:2]
    kp1, kp2, good = orb_match(doc, scene, ratio=0.75)
    info: dict = {"found": False, "good": len(good)}
    if len(good) < 4:
        return info
    pts_doc, pts_scene = points_from_matches(kp1, kp2, good)
    H, mask = estimate_homography(pts_doc, pts_scene)  # doc → scene
    if H is None:
        return info
    inliers = int(mask.sum())
    keep = mask.ravel().astype(bool)
    err = reprojection_error(H, pts_doc[keep], pts_scene[keep])

    # 検出枠（doc の四隅を scene 座標へ投影した四辺形）。
    quad = warp_corners(H, dw, dh)
    detected = scene.copy()
    cv2.polylines(detected, [quad.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # まっすぐ化: scene → doc の逆ホモグラフィで正面へ引き戻す。
    rectified = cv2.warpPerspective(scene, np.linalg.inv(H), (dw, dh))

    # 忠実度: 復元した正面と元カードのグレースケール SSIM（1 に近いほど元通り）。
    g_doc = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_rec = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY).astype(np.float64)
    rectify_ssim = ssim(g_doc, g_rec)

    info.update(found=True, inliers=inliers, reproj_error=err, quad=quad,
                detected=detected, rectified=rectified, rectify_ssim=rectify_ssim)
    return info


# =====================================================================
# 図のまとめ
# =====================================================================

def save_summary(out: pathlib.Path, manual: np.ndarray, auto: np.ndarray | None,
                 auto_mode: str, doc: np.ndarray, det: dict) -> None:
    """パノラマ2種＋書類の検出/復元を1枚に並べて保存（BGR→RGB に変換）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def show(ax, img_bgr, title):
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    show(axes[0, 0], manual, f"Manual panorama  {manual.shape[1]}x{manual.shape[0]}")
    if auto is not None:
        show(axes[0, 1], auto, f"cv2.Stitcher ({auto_mode})  {auto.shape[1]}x{auto.shape[0]}")
    else:
        axes[0, 1].text(0.5, 0.5, "Stitcher failed", ha="center", va="center")
        axes[0, 1].axis("off")
    if det.get("found"):
        show(axes[1, 0], det["detected"], "Planar object detected (quad)")
        show(axes[1, 1], det["rectified"],
             f"Rectified to frontal  SSIM={det['rectify_ssim']:.3f}")
    else:
        for ax in (axes[1, 0], axes[1, 1]):
            ax.text(0.5, 0.5, "detection skipped", ha="center", va="center")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "mp_summary.png"), dpi=110)
    plt.close(fig)


def _to_jsonable(obj):
    """numpy の型を JSON で書ける素の Python 型に変換する小ヘルパ。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# =====================================================================
# 統合パイプライン
# =====================================================================

def main() -> None:
    out = output_dir()
    report: dict = {}

    # [A] 4視点を合成 -----------------------------------------------------
    views = make_overlapping_views(n=4)
    for i, v in enumerate(views):
        cv2.imwrite(str(out / f"mp_view_{i}.png"), v)
    print(f"[A] 4視点を生成（各 {views[0].shape[1]}x{views[0].shape[0]}）")
    report["n_views"] = len(views)

    # [B] 手作りパノラマ + 品質数値化 ------------------------------------
    print("\n[B] 手作りパイプライン（build_panorama）")
    t0 = time.perf_counter()
    manual, inliers = build_panorama(views)
    manual_dt = time.perf_counter() - t0
    cv2.imwrite(str(out / "mp_panorama_manual.png"), manual)
    # 先頭ペアの再投影誤差とインライア比も測る（品質の一次指標）。
    kp1, kp2, good = orb_match(views[0], views[1])
    p0, p1 = points_from_matches(kp1, kp2, good)
    H01, mask01 = estimate_homography(p1, p0)
    err01 = reprojection_error(H01, p1, p0, mask01)
    inlier_ratio0 = float(mask01.sum()) / max(len(good), 1)
    print(f"  size={manual.shape[1]}x{manual.shape[0]} / ペア別インライア={inliers} / {manual_dt:.2f}s")
    print(f"  先頭ペア: 再投影誤差={err01:.3f}px / インライア比={inlier_ratio0:.1%}")
    report["manual"] = {
        "size": [manual.shape[1], manual.shape[0]],
        "pair_inliers": [int(x) for x in inliers],
        "seconds": round(manual_dt, 3),
        "first_pair_reproj_error_px": round(err01, 3),
        "first_pair_inlier_ratio": round(inlier_ratio0, 4),
    }

    # [C] cv2.Stitcher + 重なり SSIM ------------------------------------
    print("\n[C] cv2.Stitcher（自動）と重なり領域 SSIM")
    auto, mode, auto_dt = run_stitcher(views)
    overlap = None
    if auto is not None:
        cv2.imwrite(str(out / "mp_panorama_stitcher.png"), auto)
        overlap = overlap_ssim(manual, auto)
        print(f"  採用モード={mode} / size={auto.shape[1]}x{auto.shape[0]} / {auto_dt:.2f}s")
        print(f"  手作り vs Stitcher の重なり SSIM = "
              f"{'計測不可' if overlap is None else f'{overlap:.4f}'}")
        report["stitcher"] = {
            "mode": mode,
            "size": [auto.shape[1], auto.shape[0]],
            "seconds": round(auto_dt, 3),
            "overlap_ssim_vs_manual": None if overlap is None else round(overlap, 4),
        }
    else:
        print("  Stitcher は全モードで失敗（手作り版のみで続行）")
        report["stitcher"] = {"mode": "FAILED"}

    # [D] 平面物体の検出 + まっすぐ化 -----------------------------------
    print("\n[D] 平面物体（書類）の検出と正面復元（rectify）")
    doc = make_document()
    cv2.imwrite(str(out / "mp_document.png"), doc)
    scene = make_doc_background(height=560, width=860, seed=5)
    dst_quad = np.float32([[150, 90], [610, 150], [650, 470], [120, 410]])
    scene, _ = paste_planar(scene, doc, dst_quad)
    cv2.imwrite(str(out / "mp_document_scene.png"), scene)
    det = detect_and_rectify(doc, scene)
    if det.get("found"):
        cv2.imwrite(str(out / "mp_document_detected.png"), det["detected"])
        cv2.imwrite(str(out / "mp_document_rectified.png"), det["rectified"])
        print(f"  good={det['good']} / インライア={det['inliers']} "
              f"/ 再投影誤差={det['reproj_error']:.3f}px")
        print(f"  正面復元の忠実度 SSIM={det['rectify_ssim']:.4f}（元カードにどれだけ戻せたか）")
        report["planar_object"] = {
            "good": det["good"],
            "inliers": det["inliers"],
            "reproj_error_px": round(det["reproj_error"], 3),
            "rectify_ssim": round(det["rectify_ssim"], 4),
            "detected_quad": _to_jsonable(np.round(det["quad"], 1)),
        }
    else:
        print(f"  検出スキップ（good={det['good']} で対応不足）")
        report["planar_object"] = {"found": False, "good": det["good"]}

    # [E] 図と JSON の保存 ----------------------------------------------
    save_summary(out, manual, auto, mode, doc, det)
    with open(out / "mp_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[E] mp_summary.png と mp_report.json を保存しました。")
    print("完了。outputs/06_homography_panorama/ の mp_*.png / mp_report.json を確認してください。")


if __name__ == "__main__":
    main()
