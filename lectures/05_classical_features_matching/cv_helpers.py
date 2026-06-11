"""第5回モジュール共通ヘルパ（合成画像の生成と出力先の管理）。

このモジュール（特徴点検出とマッチング）は、ネット非依存・サンプル画像不要で動くことを
最優先にしています。そのため「学習に使う画像」をすべて numpy / cv2 でその場で合成します。
合成生成のロジックは 01〜03 のスクリプトと exercises.py で共有するので、ここに 1 箇所だけ
置いて使い回します（他モジュールからは import しない＝この回だけで自己完結）。

ここに置く責務は次の 4 つだけに絞っています（賢すぎる抽象化はしない方針）。
  1. 出力先 outputs/05_classical_features_matching/ の場所を返す
  2. 特徴量マッチング用の「特徴に富んだシーン画像」を作る（make_scene_bgr）
  3. その画像を既知のホモグラフィで歪ませた「別視点の画像」を作る（warp_to_view）
  4. テンプレートマッチング用 / Hough 用の合成画像を作る

なお data/ に画像があれば優先利用してもよい設計ですが、この回の主題（同一物体の別視点ペアや
直線・円が明確な画像）には合成画像の方が向くため、既定では常に合成画像を使います。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

# --- パス定義 -----------------------------------------------------------
# このファイルは lectures/05_classical_features_matching/ にある。
#   parents[2] がプロジェクトルート（lecture-cv/）。__file__ 基準なので実行場所に依存しない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODULE_ID = "05_classical_features_matching"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    その保存先をここで一元管理する。存在しなければ自動で作成する。
    """
    d = OUTPUT_ROOT / MODULE_ID
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_scene_bgr(height: int = 480, width: int = 640, seed: int = 42) -> np.ndarray:
    """特徴点マッチング用の「特徴に富んだシーン画像」を BGR uint8 で作る。

    SIFT/ORB は「角（コーナー）」や「斑点（ブロブ）」のように、周囲と区別しやすい局所パターンを
    特徴点として拾う。そこで、四角・円・線・文字を大量に配置して角とブロブを増やし、
    マッチングが意味を持つ画像にする。seed を固定して毎回同じ画像が出るようにしている
    （実験の再現性のため）。
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 背景に横方向グラデーションを敷く（平坦な真っ黒だと特徴点が出にくいため）。
    gx = np.linspace(30, 200, width, dtype=np.uint8)
    img[:] = np.repeat(gx[None, :, None], height, axis=0)

    # 塗りつぶしの四角と円をランダム配置。コーナーとブロブの両方を作る。
    for _ in range(40):
        x = int(rng.integers(20, width - 60))
        y = int(rng.integers(20, height - 60))
        size = int(rng.integers(20, 70))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + size, y + size), color, thickness=-1)
        else:
            cv2.circle(img, (x, y), size // 2, color, thickness=-1)

    # 斜めの線を何本か引く（エッジ方向のバリエーションを増やす）。
    for _ in range(15):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        cv2.line(img, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

    # 文字も置く。文字の角は SIFT/ORB が好む良い特徴点になる。
    labels = [("CV", (60, 120)), ("ORB", (300, 200)), ("SIFT", (410, 360)),
              ("2026", (120, 410)), ("FEAT", (480, 90))]
    for text, org in labels:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    return img


def warp_to_view(img: np.ndarray, brightness_beta: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """画像を「既知のホモグラフィ」で歪ませ、別カメラ視点の画像を作って返す。

    返り値は (歪ませた画像, 真のホモグラフィ行列 H)。
    4 隅を内側にずらした台形へ射影することで、回転・遠近（パース）を含む視点変化を模す。
    さらに明るさを少し変えて、現実のカメラ条件差（露出違い）も再現する。

    こうして「正解の H が分かっている 2 枚」を用意すると、マッチングから推定した H の
    良し悪し（インライア率）を客観的に評価できる。
    """
    h, w = img.shape[:2]
    # 変換前後の対応 4 点を決める。src=元画像の 4 隅、dst=少しすぼめて傾けた台形。
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 30], [w - 20, 50], [w - 60, h - 25], [15, h - 60]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h))
    # 露出差を模す（alpha=コントラスト, beta=明るさ）。飽和演算なので 255 で頭打ち。
    warped = cv2.convertScaleAbs(warped, alpha=0.9, beta=brightness_beta)
    return warped, H


def make_object_scene(
    height: int = 480, width: int = 640, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """テンプレートマッチング用のシーンと、その中の「探したい物体」テンプレートを作る。

    返り値は (シーン画像, テンプレート画像, 物体の左上(x,y), 物体の右下(x,y))。
    背景に紛らわしい円を散らした上で、はっきり区別できる「バッジ」を 1 つ既知の位置に置く。
    テンプレートはそのバッジ部分を切り出したもの。正解位置が分かるので検出の当たり外れを判定できる。
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 60, dtype=np.uint8)

    # 背景に紛らわしい小さな円を散らす（誤検出を誘う要素）。
    for _ in range(30):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        cv2.circle(img, (x, y), int(rng.integers(5, 15)), color, thickness=-1)

    # 探したい物体（バッジ）。中心 (cx, cy) に矩形＋円＋文字で作る。
    cx, cy = 430, 300
    half_w, half_h = 40, 30
    top_left = (cx - half_w, cy - half_h)
    bottom_right = (cx + half_w, cy + half_h)
    cv2.rectangle(img, top_left, bottom_right, (20, 180, 255), thickness=-1)
    cv2.circle(img, (cx, cy), 18, (255, 40, 40), thickness=-1)
    cv2.putText(img, "T", (cx - 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    # テンプレート = 物体部分の切り出し（numpy スライスは [y, x] 順）。
    template = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
    return img, template, top_left, bottom_right


def make_shapes_bgr(height: int = 480, width: int = 640) -> np.ndarray:
    """Hough 変換用に、直線と円がはっきり写った画像を作る。

    白背景に黒で線分（ほぼ水平・ほぼ垂直の数本）と円（半径違い 3 個）を描く。
    パラメータをいじったときの検出数の変化を観察するのにちょうどよい構成。
    """
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    # 直線（端点を少し傾けて現実味を出す）。
    cv2.line(img, (50, 60), (600, 80), (0, 0, 0), thickness=2)
    cv2.line(img, (60, 400), (580, 420), (0, 0, 0), thickness=2)
    cv2.line(img, (100, 50), (120, 440), (0, 0, 0), thickness=2)
    cv2.line(img, (520, 60), (540, 430), (0, 0, 0), thickness=2)
    # 円（中心と半径を変えて 3 個）。
    for center, radius in [((200, 250), 50), ((400, 200), 40), ((480, 340), 60)]:
        cv2.circle(img, center, radius, (0, 0, 0), thickness=2)
    return img
