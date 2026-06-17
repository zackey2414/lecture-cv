"""第11回モジュール共通ヘルパ（リアルタイム・ストリーム処理）。

このファイルは「ストリーム処理の演習で何度も書くことになる定型処理」を
小さくまとめた道具箱です。01〜04 の各スクリプトはこれを import して使います。

ここに置く責務は次の5つだけに絞っています（過度な抽象化はしない）。
  1. 出力先 lectures/11_realtime_stream/outputs/ の場所を返す
  2. ネット非依存・カメラ不要で動かすための「合成フレーム生成」
     （静止した背景の上を図形が動く＝背景差分の格好の練習台）
  3. cv2.VideoCapture に似せた合成ソース（カメラ無しでループを学ぶため）
  4. 既定は合成・CV_CAM=1 のときだけ本物のWebカメラを開く source ファクトリ
  5. リアルタイム性能を測る道具（処理FPSの EMA、レイテンシのパーセンタイル、
     1フレームあたりの『重い処理』を本物のCPU演算で模擬する関数）

なぜヘルパにまとめるのか:
  - リアルタイム処理の本質は「I/O ループ」「計測」「ドロップ」で、これらは
    どのスクリプトでも共通。学習の主眼（背景差分・スレッド分離・再接続）に
    集中できるよう、土台はここに固める。
  - 合成ソースを用意することで、Webカメラも動画ファイルもネットも無い環境（CI/Docker）で
    全スクリプトがそのまま exit 0 で完走できる。

注意:
  各スクリプト本体では、学習上重要な処理（MOG2 の apply、grab/retrieve、
  queue の put_nowait など）はあえて生のまま書いて「読んで学べる」ようにしています。
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Iterator

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/11_realtime_stream/ にある。
#   parents[0] = 11_realtime_stream / parents[1] = lectures / parents[2] = プロジェクトルート
# __file__ を基準にするので、どのディレクトリから実行しても出力先がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
MODULE_ID = "11_realtime_stream"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    その保存先をここで一元管理する。存在しなければ自動で作成する。
    """
    d = _HERE.parent / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- 合成フレーム生成 ----------------------------------------------------

def make_static_background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """動かない背景（テクスチャ付き）を BGR uint8 で作る。

    背景差分は「動かない背景」を統計モデルで学習し、そこから外れた画素を前景とみなす。
    そのため練習台にはノッペリした単色ではなく、ゆるい勾配＋細かな斑（ざらつき）のある
    『いかにも背景らしい』画を用意する。これは毎フレーム同じ＝動かない。
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    grad = xx / width * 90.0 + yy / height * 60.0          # ゆるい斜め勾配
    base = np.stack([grad + 40, grad + 50, grad + 60], axis=-1)  # B/G/R で少し色味を変える
    speckle = rng.normal(0.0, 6.0, size=(height, width, 1)).astype(np.float32)  # 固定の斑
    return np.clip(base + speckle, 0, 255).astype(np.uint8)


def synthetic_frames(
    num_frames: int = 120,
    size: tuple[int, int] = (480, 640),
    seed: int = 0,
    noise_sigma: float = 4.0,
) -> Iterator[tuple[int, np.ndarray]]:
    """合成フレーム列を1枚ずつ生成するジェネレータ（カメラ/動画/ネット不要）。

    静止した背景の上を、複数の明るい円がバウンドしながら動く。さらに毎フレーム
    弱いガウスノイズ（センサノイズの模擬）を乗せる。これにより:
      - 動く円 ＝ 前景（背景差分で抽出したいもの）
      - ノイズ ＝ モルフォロジーで消したい細かな誤検出
    という、ストリーム処理の練習に必要な要素がそろう。

    size は (高さ H, 幅 W) の順（numpy/cv2 の shape と同じ並び）に注意。
    yield するのは (フレーム番号, BGR画像) のタプル。
    """
    height, width = size
    rng = np.random.default_rng(seed)
    background = make_static_background(height, width, rng)

    # 動く物体（中心座標 pos・速度 vel・半径 r・色 color）を3つ用意する。
    objects = []
    for _ in range(3):
        objects.append(
            {
                "pos": rng.uniform([width * 0.2, height * 0.2], [width * 0.8, height * 0.8]),
                "vel": rng.uniform(-6.0, 6.0, size=2),
                "r": int(rng.integers(18, 34)),
                "color": tuple(int(c) for c in rng.integers(180, 256, size=3)),
            }
        )

    for idx in range(num_frames):
        frame = background.copy()
        for o in objects:
            o["pos"] = o["pos"] + o["vel"]
            # 壁で反射（index 0=x は幅 width、index 1=y は高さ height で跳ね返す）
            for k, lim in ((0, width), (1, height)):
                if o["pos"][k] < o["r"]:
                    o["pos"][k] = o["r"]
                    o["vel"][k] *= -1
                elif o["pos"][k] > lim - o["r"]:
                    o["pos"][k] = lim - o["r"]
                    o["vel"][k] *= -1
            cx, cy = int(o["pos"][0]), int(o["pos"][1])
            cv2.circle(frame, (cx, cy), o["r"], o["color"], thickness=-1, lineType=cv2.LINE_AA)

        if noise_sigma > 0:
            noise = rng.normal(0.0, noise_sigma, size=frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        yield idx, frame


class SyntheticCapture:
    """cv2.VideoCapture に似せた合成ソース（カメラ/動画ファイル不要でループを学ぶため）。

    本物の cv2.VideoCapture と同じく isOpened() / read() / get() / release() を持つので、
    『カメラを開く → while で read() → 処理 → release()』という正準ループを、
    実機が無くてもそのまま体験できる。read() は (成功フラグ, BGRフレーム) を返す。
    """

    def __init__(
        self,
        num_frames: int = 120,
        size: tuple[int, int] = (480, 640),
        seed: int = 0,
        src_fps: float = 30.0,
    ) -> None:
        self._gen = synthetic_frames(num_frames, size, seed)
        self._opened = True
        self._fps = src_fps
        self._size = size

    def isOpened(self) -> bool:  # noqa: N802 (cv2 互換の名前にあえて合わせる)
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            _, frame = next(self._gen)
            return True, frame
        except StopIteration:
            self._opened = False
            return False, None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[0])
        return 0.0

    def release(self) -> None:
        self._opened = False


def open_source(
    num_frames: int = 120,
    size: tuple[int, int] = (480, 640),
    seed: int = 0,
):
    """入力ソースを開いて返す。既定は合成、CV_CAM=1 のときだけ本物のWebカメラ。

    返り値は read()/get()/release() を持つ『capture ライク』なオブジェクト。
    こうしておくと、呼び出し側のループは合成でも実カメラでも全く同じコードで書ける。
    """
    if os.environ.get("CV_CAM") == "1":
        # OS の既定バックエンドでデバイス番号 0 を開く（Linux=V4L2 / mac=AVFOUNDATION 等）。
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[source] Webカメラ(0) を使用します（CV_CAM=1）")
            return cap
        print("[source] CV_CAM=1 ですがカメラを開けませんでした。合成ソースにフォールバックします。")
    return SyntheticCapture(num_frames=num_frames, size=size, seed=seed)


# --- 計測の道具 ----------------------------------------------------------

class FpsMeter:
    """処理FPS（自分のパイプラインが1秒に何枚さばけているか）を EMA で滑らかに測る。

    ソースFPS（cap.get(CAP_PROP_FPS)）と処理FPSは別物。ここで測るのは後者で、
    フレーム間の経過時間 dt から瞬間FPS=1/dt を出し、指数移動平均(EMA)でならす。
    瞬間値は跳ねやすいので、EMA で『今おおよそ何FPS出ているか』を安定して表示する。
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha          # EMA の追従度（大きいほど新しい値を重視）
        self.ema: float | None = None
        self._last: float | None = None

    def tick(self) -> None:
        """1フレーム処理し終えるたびに呼ぶ。経過時間から瞬間FPSを計算して EMA を更新。"""
        now = time.perf_counter()   # 壁時計より安定した高分解能タイマ
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                fps = 1.0 / dt
                self.ema = fps if self.ema is None else (1 - self.alpha) * self.ema + self.alpha * fps
        self._last = now

    @property
    def fps(self) -> float:
        return self.ema or 0.0


def percentiles(values_ms: list[float]) -> tuple[float, float]:
    """レイテンシ（ミリ秒）のリストから (p50, p99) を返す。

    平均だけ見ると『たまに大きく詰まる』瞬間を見落とす。p99（上位1%の遅さ）まで見ると、
    リアルタイム要件を満たせない外れ値（カクつき）を捉えられる。
    """
    if not values_ms:
        return 0.0, 0.0
    arr = np.asarray(values_ms, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 99))


def simulate_heavy_work(frame: np.ndarray, passes: int = 3) -> np.ndarray:
    """1フレームあたりの『重い処理』を本物のCPU演算で模擬する（例: 検出・セグメ相当）。

    わざと解像度に比例して重くなる演算（ガウシアン平滑化×数回 → グレー化 → Canny）にしてある。
    こうすると『早めに縮小すると速くなる』ことを実測で示せる（画素数が減れば計算も減る）。
    返り値は可視化用のエッジ画像（1ch）。
    """
    work = frame
    for _ in range(passes):
        work = cv2.GaussianBlur(work, (7, 7), 0)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)
