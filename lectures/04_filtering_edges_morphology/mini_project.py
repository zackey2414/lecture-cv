"""章末ミニプロジェクト: 「スマホ写真 → きれいなスキャン」書類スキャナの完成形。

この回で学んだ全要素を 1 本のパイプラインに統合する総合課題:
  平滑化(ノイズ除去) → エッジ/閾値(対象抽出) → モルフォロジー(整形)
  → 輪郭抽出+approxPolyDP(四隅推定) → 透視変換(正面化)
  → CLAHE(コントラスト補正) → 適応的閾値(二値スキャン化)

入力は「合成したスマホ写真」:
  - 透視で傾いた書類（撮影角度）
  - 照明ムラ（片側が暗い）
  - ガウスノイズ＋ごま塩ノイズ（センサ/圧縮ノイズ相当）
  - 低コントラスト

出力は outputs/04_filtering_edges_morphology/ に:
  - mini_project_pipeline.png : 各工程を並べた 1 枚（工程の流れを目で追える）
  - mini_project_scan.png     : 最終成果物（正面化された二値スキャン）
  - mini_project_hist.png     : CLAHE 前後の輝度ヒストグラム（matplotlib, Agg）
  - mini_project_report.json  : 各工程の定量指標（四隅座標・面積・コントラスト等）

実行:
  uv run python lectures/04_filtering_edges_morphology/mini_project.py
すべて CPU・ネット非依存・合成データ。cv2.imshow は使わずファイルに保存する。
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

# matplotlib は pyplot を import する前に Agg（画面不要）へ固定する。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "04_filtering_edges_morphology"
OUT_DIR = os.path.join("outputs", MODULE_ID)


# ---------------------------------------------------------------------------
# 可視化の小道具（このファイル内で自己完結。図タイトルは ASCII）
# ---------------------------------------------------------------------------
def panel(img: np.ndarray, title: str, size: tuple[int, int] = (340, 300)) -> np.ndarray:
    """画像をタイトル付きパネルにする（グレーは 3ch 化して横並びできるように）。"""
    if img.ndim == 2:  # (H, W) のグレーは 3ch にしないと hstack できない
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w, h = size
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)  # タイトル用の黒帯
    cv2.putText(p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    """パネル列を cols 列のグリッドに連結する。端数は黒で埋める。"""
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# 入力（劣化したスマホ写真）の合成
# ---------------------------------------------------------------------------
def make_document(h: int = 360, w: int = 260) -> np.ndarray:
    """正面から見た理想の書類を合成する（白い紙＋ヘッダ＋本文＋朱印）。BGR。"""
    doc = np.full((h, w, 3), 246, dtype=np.uint8)  # 白い紙
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(doc, (10, 10), (w - 11, 62), (170, 110, 40), -1)  # 青系ヘッダ帯
    cv2.putText(doc, "RECEIPT", (22, 46), f, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    lines = ["No. 04-2026", "lecture-cv", "document", "scanner", "warp + clahe", "straighten!"]
    for i, line in enumerate(lines):
        cv2.putText(doc, line, (20, 104 + i * 40), f, 0.66, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.circle(doc, (w - 48, h - 52), 30, (40, 50, 200), 3)  # 朱印っぽい円
    return doc


def make_phone_photo(doc: np.ndarray, ch: int = 540, cw: int = 480) -> np.ndarray:
    """理想の書類を「劣化したスマホ写真」に変換する。

    透視で傾け → 暗い背景に配置 → 照明ムラ → ガウス＋ごま塩ノイズ → 低コントラスト。
    これらを一気に乗せることで、この回の各工程が「なぜ必要か」が体感できる入力を作る。
    """
    h, w = doc.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    # 撮影で傾いた四隅（凸な台形）。これを検出して正面へ戻すのがゴール。
    dst = np.float32([[110, 60], [430, 120], [380, 500], [70, 430]])
    m = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(
        doc, m, (cw, ch), borderMode=cv2.BORDER_CONSTANT, borderValue=(55, 52, 50)
    )

    # 照明ムラ: 左上が明るく右下が暗い。固定/Otsu 閾値を苦しめ、適応的の出番を作る。
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    illum = 1.05 - 0.45 * (xx / cw) - 0.25 * (yy / ch)
    photo = np.clip(photo.astype(np.float32) * illum[:, :, None], 0, 255).astype(np.uint8)

    # ガウスノイズ（センサノイズ相当）
    rng = np.random.default_rng(42)
    photo = np.clip(photo.astype(np.float32) + rng.normal(0, 8, photo.shape), 0, 255).astype(np.uint8)

    # ごま塩ノイズ（圧縮/伝送の突発ノイズ相当）。medianBlur の出番を作る。
    sp = rng.random(photo.shape[:2])
    photo[sp < 0.01] = 0
    photo[sp > 0.99] = 255

    # 低コントラスト化（値域を圧縮）。CLAHE の出番を作る。
    return np.clip(photo.astype(np.float32) * 0.78 + 26.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 工程 1: ノイズ除去（ごま塩→median、仕上げ→bilateral でエッジ保持）
# ---------------------------------------------------------------------------
def denoise(photo: np.ndarray) -> np.ndarray:
    """ごま塩は中央値で確実に潰し、残りはエッジ保持の bilateral で均す。"""
    med = cv2.medianBlur(photo, 3)  # 突発的な粒（0/255）を中央値で除去
    return cv2.bilateralFilter(med, d=7, sigmaColor=60, sigmaSpace=7)  # エッジを残して平滑化


# ---------------------------------------------------------------------------
# 工程 2: 書類の四隅を検出（二値化→モルフォロジー→輪郭→4点近似）
# ---------------------------------------------------------------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    """4点を [左上, 右上, 右下, 左下] へ並べ替える（和と差の符号で判定する定石）。"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)  # x+y: 最小=左上, 最大=右下
    d = pts[:, 0] - pts[:, 1]  # x-y: 最大=右上, 最小=左下
    return np.float32([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


def detect_document(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """書類の四隅 quad と、途中経過（二値画像・最大輪郭）を返す。

    手順: ぼかし → Otsu 二値化（明るい紙=白）→ クロージングで穴埋め
          → findContours(4系は2返し) → 最大輪郭 → approxPolyDP で4点。
    4点に近似できなければ最小外接矩形 minAreaRect で代用して堅牢化する。
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 紙面の文字や朱印で空く穴をクロージングで埋め、輪郭を1枚の塊にする
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ★ OpenCV 4 系は (contours, hierarchy) の【2つ】返し（3系の3つ返しと違う）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)  # 面積最大の輪郭＝書類

    peri = cv2.arcLength(largest, True)
    for factor in (0.02, 0.03, 0.04, 0.05, 0.01):  # epsilon を振って「ちょうど4点」を探す
        approx = cv2.approxPolyDP(largest, factor * peri, True)
        if len(approx) == 4:
            return order_corners(approx), binary, largest
    # 4点にならなければ最小外接矩形で代用
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return order_corners(box), binary, largest


# ---------------------------------------------------------------------------
# 工程 3: 透視変換で正面化
# ---------------------------------------------------------------------------
def warp_to_front(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """四隅 quad(TL,TR,BR,BL) を正面の長方形へ射影変換する。"""
    tl, tr, br, bl = quad
    # 出力サイズは対辺の長い方から決める（つぶれを防ぐ）
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (width, height))


# ---------------------------------------------------------------------------
# 工程 4: コントラスト補正(CLAHE) → 適応的閾値で二値スキャン化 → 整形
# ---------------------------------------------------------------------------
def to_clean_scan(front_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """正面化した書類を「黒文字・白地」のきれいな二値スキャンに仕上げる。

    返り値: (CLAHE 後のグレー, 最終二値スキャン)。
    """
    gray = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # 局所コントラスト補正
    enhanced = clahe.apply(gray)
    # 適応的閾値: 照明ムラが残っても局所的に文字を拾える。黒文字/白地 → THRESH_BINARY。
    scan = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=12
    )
    # オープニングで二値化由来の白い粒ノイズを除去（地をきれいに）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scan = cv2.morphologyEx(scan, cv2.MORPH_OPEN, kernel)
    return enhanced, scan


# ---------------------------------------------------------------------------
# 指標の算出（JSON 用）
# ---------------------------------------------------------------------------
def variance_of_laplacian(gray: np.ndarray) -> float:
    """Laplacian の分散＝鮮鋭度の簡易指標（大きいほどシャープ/エッジが立つ）。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ink_ratio(scan: np.ndarray) -> float:
    """二値スキャン中の黒（インク=0）画素の割合。"""
    return float(np.mean(scan == 0))


def _to_native(obj):
    """numpy 型を JSON 化できる Python ネイティブ型へ再帰的に変換する。"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# メイン: 全工程を 1 本に統合
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 0) 入力（劣化したスマホ写真）を合成
    doc = make_document()
    photo = make_phone_photo(doc)

    # 1) ノイズ除去
    clean = denoise(photo)
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

    # 2) 四隅検出
    quad, binary, largest = detect_document(clean_gray)

    # 検出結果の可視化（輪郭＝黄、四隅＝赤い番号付き点）
    viz = clean.copy()
    cv2.drawContours(viz, [largest], -1, (0, 200, 255), 2)
    for i, (px, py) in enumerate(quad.astype(int)):
        cv2.circle(viz, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(viz, str(i), (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 3) 透視変換で正面化（ノイズ除去後のカラーを起こす）
    front = warp_to_front(clean, quad)

    # 4) CLAHE → 適応的閾値 → モルフォロジーで二値スキャン化
    enhanced, scan = to_clean_scan(front)
    front_gray = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY)

    # ---- 指標を集計 -------------------------------------------------------
    report = {
        "input": {
            "photo_shape_hw": list(photo.shape[:2]),
            "input_sharpness_varlap": round(variance_of_laplacian(clean_gray), 2),
        },
        "detect": {
            "num_contours_was_2_tuple": True,  # 4系は (contours, hierarchy) の2返し
            "largest_contour_area_px": round(float(cv2.contourArea(largest)), 1),
            "approx_corners": 4,
            "corners_tl_tr_br_bl": np.round(quad, 1),
        },
        "warp": {
            "front_size_wh": [int(front.shape[1]), int(front.shape[0])],
        },
        "enhance": {
            "front_gray_std": round(float(front_gray.std()), 2),
            "clahe_std": round(float(enhanced.std()), 2),
            "clahe_sharpness_varlap": round(variance_of_laplacian(enhanced), 2),
        },
        "scan": {
            "ink_ratio": round(ink_ratio(scan), 4),
            "scan_size_wh": [int(scan.shape[1]), int(scan.shape[0])],
        },
    }

    with open(os.path.join(OUT_DIR, "mini_project_report.json"), "w", encoding="utf-8") as fp:
        json.dump(_to_native(report), fp, ensure_ascii=False, indent=2)

    # ---- 可視化を保存 -----------------------------------------------------
    # (a) パイプライン全工程グリッド
    sheet = grid(
        [
            panel(photo, "1. phone photo (degraded)"),
            panel(clean, "2. denoise (median+bilateral)"),
            panel(binary, "3. otsu + close"),
            panel(viz, "4. contour + 4 corners"),
            panel(front, "5. warpPerspective (front)"),
            panel(enhanced, "6. CLAHE enhanced"),
            panel(scan, "7. adaptive -> clean scan"),
        ],
        4,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "mini_project_pipeline.png"), sheet)
    # (b) 最終成果物だけ単独でも保存
    cv2.imwrite(os.path.join(OUT_DIR, "mini_project_scan.png"), scan)

    # (c) CLAHE 前後の輝度ヒストグラム（matplotlib, Agg）
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, im, name in zip(axes, (front_gray, enhanced), ("front gray", "after CLAHE")):
        hist = cv2.calcHist([im], [0], None, [256], [0, 256])
        ax.plot(hist, color="black")
        ax.set_title(name)
        ax.set_xlim(0, 255)
        ax.set_xlabel("pixel value")
    fig.suptitle("CLAHE spreads the histogram (contrast up)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mini_project_hist.png"), dpi=110)
    plt.close(fig)

    # ---- ログ -------------------------------------------------------------
    print("=== ミニプロジェクト: 書類スキャナ パイプライン ===")
    print(f"[1] 入力写真サイズ(H,W)      : {tuple(photo.shape[:2])}")
    print(f"[2] ノイズ除去後の鮮鋭度(varLap): {report['input']['input_sharpness_varlap']}")
    print(f"[3] 最大輪郭の面積           : {report['detect']['largest_contour_area_px']} px^2")
    print(f"[4] 検出した四隅(TL,TR,BR,BL):\n{report['detect']['corners_tl_tr_br_bl']}")
    print(f"[5] 正面化サイズ(W,H)        : {tuple(report['warp']['front_size_wh'])}")
    print(
        f"[6] コントラスト std: {report['enhance']['front_gray_std']} "
        f"→ CLAHE後 {report['enhance']['clahe_std']}（広がる＝コントラスト向上）"
    )
    print(f"[7] 最終スキャンのインク比率 : {report['scan']['ink_ratio']}")
    print("\n完了。outputs/04_filtering_edges_morphology/ に以下を保存しました:")
    print("  mini_project_pipeline.png / mini_project_scan.png / "
          "mini_project_hist.png / mini_project_report.json")


if __name__ == "__main__":
    main()
