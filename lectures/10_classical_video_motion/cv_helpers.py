"""第10回モジュール共通ヘルパ（古典的な動画処理 — 動き解析）。

このファイルは「動き解析の章で何度も書く定型処理」をまとめた小さな道具箱です。
本講座の方針として、各スクリプト（01〜04）と exercises.py はこのヘルパを import します。
（※ import するのは "同じモジュール内" のこのファイルだけ。他モジュールには依存しません。
   ネット・GPU・カメラ・サンプル動画が無くても、ここで全部を numpy/cv2 で合成生成します。）

ここに置く責務は次の4つだけに絞っています。
  1. 「静止した背景」を作る（背景差分が成立するよう、テクスチャはあるが動かない）。
  2. その背景の上を「決まった軌道で動く物体」が通る連番フレーム列を作る。
       - テクスチャ付きスプライト（角が多い）= 疎/密オプティカルフロー向き。
       - 緑色の円（色が背景と明確に違う）   = meanshift/camshift の色追跡向き。
  3. 既知の平行移動だけの画像ペアを作る（密フローの終点誤差 EPE を答え合わせするため）。
  4. 出力先 lectures/10_classical_video_motion/outputs/ の管理と、フロー場の HSV 可視化。

コーディング原則: 各関数は1つのことだけを行い、賢すぎる抽象化は避けます。
スクリプト本体側は、学習上重要な処理（calcOpticalFlowPyrLK 等）をあえて生のまま書いて
「読んで学べる」ようにしつつ、ここの部品を必要に応じて呼びます。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/10_classical_video_motion/ にある。
#   parents[2] = プロジェクトルート（lecture-cv/）。どこから実行しても場所がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
OUTPUT_ROOT = _HERE.parent / "outputs"
MODULE_ID = "10_classical_video_motion"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    存在しなければ自動で作成する（os.makedirs 相当）。
    """
    d = OUTPUT_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- 1. 静止した背景 -----------------------------------------------------
def make_background(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """全フレームで共通の「動かない背景」を作る（BGR・uint8）。

    背景差分（MOG2/KNN）は「背景は静止している」前提で成り立つので、
    ここで作った背景はフレーム間で一切変えない。低周波のグラデに弱いノイズと、
    角の多い静止ランドマークを少しだけ載せて、フローの『動かない領域』の対照にする。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # ゆるい縞のグラデーション（チャンネルごとに少し倍率を変えて色味をつける）。
    base = 120 + 35 * np.sin(xx / 55.0) + 25 * np.cos(yy / 45.0)
    bg = np.stack([base * 0.85, base, base * 1.05], axis=-1)
    bg += rng.normal(0, 2.0, bg.shape)  # ごく弱い撮像ノイズ
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # 静止ランドマーク（角が出る小さな模様）。LK で「動かない短い軌跡」を見せる対照になる。
    for (cx, cy) in [(40, 40), (280, 50), (300, 210)]:
        cv2.rectangle(bg, (cx - 8, cy - 8), (cx + 8, cy + 8), (210, 210, 210), -1)
        cv2.rectangle(bg, (cx - 4, cy - 4), (cx + 4, cy + 4), (40, 40, 40), -1)
    return bg


def make_sprite(seed: int = 1) -> np.ndarray:
    """角（コーナー）の多いテクスチャ付きの小さな物体を作る。

    goodFeaturesToTrack が拾える『追跡しやすい角』を多く含むよう、市松模様＋図形にする。
    これを毎フレーム同じ向きのまま位置だけ動かす（剛体並進）ので、
    疎フロー(LK)も密フロー(Farneback)も素直に動きを拾える。
    """
    sp = np.zeros((34, 42, 3), np.uint8)
    tile = 7
    for j in range(0, 34, tile):
        for i in range(0, 42, tile):
            if ((i // tile) + (j // tile)) % 2 == 0:
                sp[j:j + tile, i:i + tile] = (230, 230, 230)
    cv2.rectangle(sp, (6, 6), (20, 18), (30, 30, 210), -1)
    cv2.circle(sp, (30, 24), 6, (210, 140, 20), -1)
    return sp


def make_motion_frames(
    n: int = 40, h: int = 240, w: int = 320, seed: int = 0
) -> list[np.ndarray]:
    """静止背景の上を2つの物体が動く連番フレーム列を作る（BGR・uint8 のリスト）。

    - テクスチャ付きスプライト: 左→右に等速並進＋上下に小さく揺れる（フロー解析の主役）。
    - 緑の円: 別の軌道で動き、半径が徐々に増える（色追跡 meanshift と、
      スケール変化を見る camshift の主役）。
    返り値はフレームのリスト（= 動画を1枚ずつバラした連番画像。動画とは静止画の連続）。
    """
    bg = make_background(h, w, seed)
    sprite = make_sprite()
    sh, sw = sprite.shape[:2]
    frames: list[np.ndarray] = []
    for i in range(n):
        f = bg.copy()
        t = i / (n - 1)  # 0.0 → 1.0 へ進む正規化時間
        # 1) テクスチャ付きスプライト（剛体並進）。
        sx = int(20 + (w - 40 - sw) * t)
        sy = int(h * 0.25 + 18 * np.sin(i / 6.0))
        f[sy:sy + sh, sx:sx + sw] = sprite
        # 2) 緑の円（背景と色が大きく違うので色ヒストグラム追跡に向く）。
        cx = int(40 + (w - 80) * t)
        cy = int(h * 0.7 - 22 * np.sin(i / 7.0))
        r = int(14 + 10 * t)  # 半径が 14→24 と増える（camshift のスケール追従を見せる）
        cv2.circle(f, (cx, cy), r, (40, 200, 60), -1, cv2.LINE_AA)
        frames.append(f)
    return frames


def green_window(frame: np.ndarray) -> tuple[int, int, int, int]:
    """フレーム中の緑の円を囲む初期追跡窓 (x, y, w, h) を返す。

    実務では追跡窓は人手やオブジェクト検出器が与えるが、ここは合成なので
    HSV の緑レンジで領域を取り、その外接矩形を初期窓にする（自己完結のため）。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # OpenCV の H は 0-179。緑はおおよそ 40-80 付近。
    mask = cv2.inRange(hsv, (40, 80, 80), (85, 255, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # 念のためのフォールバック（必ず妥当な窓を返し、呼び出し側を落とさない）。
        return (40, 150, 40, 40)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return (x0, y0, x1 - x0, y1 - y0)


def make_translating_pair(
    h: int = 160, w: int = 200, shift: tuple[int, int] = (6, 3), seed: int = 2
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """既知の平行移動だけで関係づく画像ペア (a, b, (dx, dy)) を作る。

    大きなテクスチャ画像から、ずらした2つの窓を切り出すだけ。
    こうすると「真のフローは全画素が (dx, dy)」と分かっているので、
    密フローの終点誤差(EPE)を答え合わせできる（27回の深層フローの定量評価の予習）。
    """
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 255, (h + 40, w + 40, 3), dtype=np.uint8)
    big = cv2.GaussianBlur(big, (0, 0), 1.2)  # 少しぼかして自然な勾配にする
    dx, dy = shift
    a = big[20:20 + h, 20:20 + w].copy()
    b = big[20 - dy:20 - dy + h, 20 - dx:20 - dx + w].copy()
    return a, b, (dx, dy)


# --- 4. フロー場の可視化 -------------------------------------------------
def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    """密フロー (H, W, 2) を HSV 配色で BGR 画像にする（OpenCV 公式チュートリアル流）。

      色相(H) = 動きの『向き』（角度を 0-179 にスケール）
      明度(V) = 動きの『大きさ』（大きいほど明るい。フレームごとに正規化）
      彩度(S) = 255 固定
    こうすると一目で「どの向きにどれだけ動いたか」が分かる。
    """
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def endpoint_error(flow: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """各画素の終点誤差 (EPE) = 推定フローと真のフローのユークリッド距離を返す (H, W)。"""
    return np.linalg.norm(flow - gt, axis=2)


if __name__ == "__main__":
    # 単体実行でスモークテスト（合成フレームが作れるか・各 API が動くかを確認）。
    out = output_dir()
    frames = make_motion_frames()
    print(f"frames: {len(frames)} 枚 / shape={frames[0].shape} / dtype={frames[0].dtype}")
    cv2.imwrite(str(out / "_smoke_frame00.png"), frames[0])
    cv2.imwrite(str(out / "_smoke_frame20.png"), frames[20])
    win = green_window(frames[0])
    print(f"緑円の初期窓 (x,y,w,h) = {win}")
    a, b, shift = make_translating_pair()
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    gt = np.zeros_like(flow)
    gt[..., 0], gt[..., 1] = shift
    print(f"既知シフト {shift} に対する EPE 平均 = {endpoint_error(flow, gt).mean():.3f} px")
    print(f"スモークOK。lectures/{MODULE_ID}/outputs/ に _smoke_*.png を保存しました。")
