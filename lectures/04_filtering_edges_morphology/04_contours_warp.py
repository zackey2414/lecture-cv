"""04: 輪郭抽出と形状解析 → 透視変換で「書類まっすぐ化」（このモジュールの完成物）。

学ぶこと:
  - findContours は OpenCV 4 系で返り値が (contours, hierarchy) の【2つ】。
    （3系の3つ返しサンプルをコピペすると ValueError になる、という頻出の罠）
  - 形状解析: contourArea / arcLength / boundingRect / approxPolyDP / convexHull。
  - approxPolyDP で輪郭を4点に近似 → 4隅を並べ替え → getPerspectiveTransform で
    透視変換行列を作り warpPerspective で正面化する、という古典の前処理連鎖。
  - 透視変換（射影）と アフィン変換（getRotationMatrix2D/warpAffine）の違い。

二値化 → 輪郭検出 → 4隅推定 → 透視変換、という流れを通して、
傾いて撮影された書類を正面のスキャン画像へ補正するパイプラインを自力で組む。

実行:
  uv run python lectures/04_filtering_edges_morphology/04_contours_warp.py
結果は outputs/04_filtering_edges_morphology/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

MODULE_ID = "04_filtering_edges_morphology"
OUT_DIR = os.path.join("outputs", MODULE_ID)


def panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 300)) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w, h = size
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def make_document(h: int = 340, w: int = 250) -> np.ndarray:
    """正面から見た「書類」を合成する（白い紙＋ヘッダ＋本文）。BGR。"""
    doc = np.full((h, w, 3), 245, dtype=np.uint8)  # 白い紙（背景の暗さと十分コントラスト）
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(doc, (10, 10), (w - 11, 58), (170, 110, 40), -1)  # 青系のヘッダ帯
    cv2.putText(doc, "INVOICE", (24, 44), f, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for i, line in enumerate(["No. 04-2026", "OpenCV  4.x", "warpPerspective", "straighten!"]):
        cv2.putText(doc, line, (22, 100 + i * 42), f, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.circle(doc, (w - 50, h - 55), 32, (40, 50, 200), 3)  # 朱印っぽい円
    return doc


def make_tilted_photo(doc: np.ndarray, ch: int = 520, cw: int = 470) -> np.ndarray:
    """書類を透視変換で傾けて暗い背景の上に置き、「撮影された写真」を作る。"""
    h, w = doc.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])  # 紙の4隅(TL,TR,BR,BL)
    # 撮影で傾いた四角形（凸な四隅）。これが「検出して戻す」対象になる。
    dst = np.float32([[95, 70], [410, 120], [360, 470], [70, 405]])
    m = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(
        doc, m, (cw, ch), borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 55, 55)
    )
    # 撮影ノイズを少しだけ乗せて現実味を出す（二値化が壊れない程度）。
    noise = np.random.default_rng(4).normal(0, 6, photo.shape)
    return np.clip(photo.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """4点を [左上, 右上, 右下, 左下] の順に並べ替える（和と差の符号で判定する定石）。"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)  # x+y: 最小=左上, 最大=右下
    d = (pts[:, 0] - pts[:, 1])  # x-y: 最大=右上, 最小=左下
    return np.float32([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


def find_document_quad(gray: np.ndarray) -> np.ndarray | None:
    """二値化→輪郭→最大輪郭→4点近似 で書類の四隅を返す。見つからなければ None。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 明るい書類 vs 暗い背景。Otsu で自動二値化（紙=白）。
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ★ OpenCV 4 系は返り値が2つ: (contours, hierarchy)。3系は3つ返しだった点に注意。
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)  # 面積最大の輪郭＝書類
    peri = cv2.arcLength(largest, True)  # 周囲長。近似の許容誤差をこれに比例させる
    # epsilon を変えながら「ちょうど4点」になる近似を探す
    for factor in (0.02, 0.03, 0.04, 0.05, 0.01):
        approx = cv2.approxPolyDP(largest, factor * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    # どうしても4点にならなければ最小外接矩形で代用（堅牢化）
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return order_corners(box)


def warp_to_front(photo: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """四隅 quad(TL,TR,BR,BL) を正面の長方形へ透視変換する。"""
    tl, tr, br, bl = quad
    # 出力サイズは、対辺の長い方から決める（つぶれを防ぐ）
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(photo, m, (width, height))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # === 0. 入力（傾いた書類写真）を合成 =================================
    doc = make_document()
    photo = make_tilted_photo(doc)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    # === 1. 形状解析の基本（最大輪郭で各種計測） =========================
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[1] findContours の返り値は2つ: contours={len(contours)}本, hierarchy.shape={None if hierarchy is None else hierarchy.shape}")
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    peri = cv2.arcLength(largest, True)
    x, y, bw, bh = cv2.boundingRect(largest)  # 軸並行の外接矩形
    hull = cv2.convexHull(largest)  # 凸包
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    print(f"    contourArea     = {area:.0f} px^2")
    print(f"    arcLength       = {peri:.0f} px")
    print(f"    boundingRect    = (x={x}, y={y}, w={bw}, h={bh})")
    print(f"    convexHull 点数  = {len(hull)}")
    print(f"    approxPolyDP 頂点= {len(approx)}  （書類なら4になってほしい）")

    # === 2. 四隅を検出して可視化 =========================================
    quad = find_document_quad(gray)
    if quad is None:
        print("    書類の四隅を検出できませんでした。")
        return
    viz = photo.copy()
    cv2.drawContours(viz, [largest], -1, (0, 200, 255), 2)  # 輪郭（黄）
    cv2.rectangle(viz, (x, y), (x + bw, y + bh), (255, 120, 0), 2)  # 外接矩形（青）
    for i, (px, py) in enumerate(quad.astype(int)):  # 4隅に番号付きの点
        cv2.circle(viz, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(viz, str(i), (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === 3. 透視変換で正面化（＝完成物の心臓部） =========================
    straight = warp_to_front(photo, quad)
    print(f"\n[3] 正面化後のサイズ: {straight.shape[1]} x {straight.shape[0]} (W x H)")

    # === 4. 透視変換 vs アフィン変換 =====================================
    # アフィン: 平行線は平行のまま（回転・拡縮・せん断・平行移動）。3点で決まる。
    sh, sw = straight.shape[:2]
    rot_m = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle=15, scale=1.0)  # 2x3 行列
    rotated = cv2.warpAffine(straight, rot_m, (sw, sh), borderValue=(255, 255, 255))

    # === まとめて保存 ====================================================
    cv2.imwrite(os.path.join(OUT_DIR, "04_doc_input.png"), photo)
    cv2.imwrite(os.path.join(OUT_DIR, "04_doc_binary.png"), binary)
    cv2.imwrite(os.path.join(OUT_DIR, "04_doc_detected.png"), viz)
    cv2.imwrite(os.path.join(OUT_DIR, "04_doc_straightened.png"), straight)

    sheet = grid(
        [
            panel(photo, "1. input (tilted photo)"),
            panel(binary, "2. Otsu binary"),
            panel(viz, "3. contour + 4 corners"),
            panel(straight, "4. warpPerspective (front)"),
            panel(rotated, "affine: rotate 15deg"),
        ],
        3,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "04_document_pipeline.png"), sheet)

    print("\n完了。outputs/04_filtering_edges_morphology/ の 04_*.png を確認してください。")
    print("  特に 04_doc_straightened.png が、傾いた写真から起こした正面スキャンです。")


if __name__ == "__main__":
    main()
