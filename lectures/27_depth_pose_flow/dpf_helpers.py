"""第27回 共有ヘルパ: 合成データ生成・可視化・評価指標をまとめた“道具箱”。

このモジュールは深度(depth)・姿勢(pose)・フロー(flow)の3スクリプトと mini_project が
共通で使う部品を集めたもの。方針は次のとおり。

  - 入力（画像/連番フレーム/画像対）は **すべて合成生成**（動く図形・人型図形・並進テクスチャ）。
    ネットに出るのは「モデル重みのDL」だけ、という講座のルールを守るため。
  - 可視化は headless 環境でも動くよう matplotlib / PIL / cv2(描画のみ) を使い、imshow は呼ばない。
  - 評価指標（AbsRel/RMSE/δ<1.25, EPE, OKS, PCK）は scripts 側が import して使う「実装版」。
    同じ式を学習者が手で書き直す課題が exercises.py にある（答え合わせ用に二重化している）。

座標規約: 画像配列は RGB の uint8 (H, W, 3)。キーポイントは (x, y, v) で v は可視フラグ
(0=未ラベル, 1=ラベルあり不可視, 2=可視)。COCO の人体17点の並びに従う。
"""

from __future__ import annotations

import pathlib

import cv2
import matplotlib
import numpy as np

# =====================================================================
# 0) 出力ディレクトリ / 乱数
# =====================================================================

MODULE_ID = "27_depth_pose_flow"
# このファイルから見て lecture-cv リポジトリのルートを辿り outputs/<module_id>/ を指す。
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_DIR = _REPO_ROOT / "outputs" / MODULE_ID
DATA_DIR = _REPO_ROOT / "data" / MODULE_ID


def ensure_output_dir() -> pathlib.Path:
    """outputs/<module_id>/ を作って返す（既にあってもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def set_seed(seed: int = 0) -> None:
    """再現性のため numpy の乱数を固定する（合成データを毎回同じにする）。"""
    np.random.seed(seed)


# =====================================================================
# 1) COCO 人体17点の定数（姿勢推定 keypoint R-CNN / OKS で使う）
# =====================================================================

# torchvision keypointrcnn_resnet50_fpn が返すキーポイントの並び（COCO 17 点）。
COCO_KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# スケルトンの辺（0始まりの index 対）。描画と「つながり」の理解のため。
COCO_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 顔（鼻-目-耳）
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 両腕（肩-肘-手首）
    (5, 11), (6, 12), (11, 12),               # 胴（肩-腰）
    (11, 13), (13, 15), (12, 14), (14, 16),   # 両脚（腰-膝-足首）
]

# COCO 公式の per-keypoint 定数 σ_i（OKS の正規化に使う。可視/小関節ほど小さい）。
COCO_SIGMAS: np.ndarray = np.array(
    [.026, .025, .025, .035, .035, .079, .079, .072, .072,
     .062, .062, .107, .107, .087, .087, .089, .089],
    dtype=np.float32,
)


# =====================================================================
# 2) 合成シーン: 単眼深度用（RGB 画像 + 真の深度マップ）
# =====================================================================

def make_depth_scene(h: int = 216, w: int = 288, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """擬似的な「奥行きのある室内」シーンを合成する。

    返り値:
      rgb     : (H, W, 3) uint8。床が手前→奥へ伸び、奥に壁、手前に箱、中景に球。
      gt_depth: (H, W) float32。メートル相当の“真の深度”（手前=小, 奥=大）。

    注意: Depth Anything V2 が出すのは相対(逆)深度で絶対距離ではない。ここで作る gt_depth は
    あくまで「定義を確かめるための合成正解」。実モデル評価ではスケール合わせが必要になる。
    """
    rng = np.random.default_rng(seed)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    gt = np.zeros((h, w), dtype=np.float32)

    horizon = int(h * 0.45)  # 地平線（この上が壁=遠景、下が床）
    rows = np.arange(h)[:, None]

    # --- 壁（遠景）: 一様な遠い深度 + ほんのり明るいグレー ---
    far = 9.0
    gt[:horizon, :] = far
    rgb[:horizon, :, :] = (205, 208, 214)

    # --- 床: 行が下に行くほど手前（深度が小さい）。線形に near..horizon を割り当てる ---
    near = 1.2
    floor_rows = np.arange(horizon, h)
    # 地平線で far、最下段で near になる線形勾配
    floor_depth = np.linspace(far, near, num=floor_rows.size, dtype=np.float32)
    gt[horizon:, :] = floor_depth[:, None]
    # 床の明るさは深度に反比例（手前ほど明るい）+ チェッカ模様で特徴を作る
    base = np.interp(rows, [horizon, h], [120, 190]).astype(np.float32)
    checker = (((np.arange(w)[None, :] // 24) + (rows // 16)) % 2) * 18.0
    floor_gray = np.clip(base + checker, 0, 255)
    rgb[horizon:, :, :] = floor_gray[horizon:, :, None].astype(np.uint8)

    # --- 手前の箱（近い物体）: 単色 + 真の深度を上書き ---
    bx0, by0, bx1, by1 = int(w * 0.12), int(h * 0.60), int(w * 0.40), int(h * 0.92)
    rgb[by0:by1, bx0:bx1] = (60, 150, 160)
    gt[by0:by1, bx0:bx1] = 1.8

    # --- 中景の球（楕円で近似）---
    cx, cy, rr = int(w * 0.72), int(h * 0.62), int(min(h, w) * 0.13)
    yy, xx = np.ogrid[:h, :w]
    ball = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr ** 2
    rgb[ball] = (210, 140, 60)
    gt[ball] = 3.5

    # --- 軽いノイズで“写真らしい”質感を足す（モデルが特徴を拾えるように）---
    noise = rng.normal(0, 5, size=rgb.shape)
    rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return rgb, gt


# =====================================================================
# 3) 合成シーン: オプティカルフロー用（並進する画像対 + 真のフロー）
# =====================================================================

def make_translating_pair(
    dx: int = 6, dy: int = 3, h: int = 128, w: int = 192, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """テクスチャ画像を (dx, dy) 並進させた2フレームと、真のフロー場を返す。

    返り値:
      frame1, frame2: (H, W, 3) uint8。frame2 は frame1 の内容が右/下に (dx, dy) だけ動いたもの。
      gt_flow       : (H, W, 2) float32。全画素で一定 (dx, dy)（グローバル並進なので真値が自明）。

    作り方: 大きなテクスチャ canvas を作り、(dy, dx) だけずらした2つの窓で切り出す。こうすると
    両フレームとも“はみ出し”が無く全画素が有効になり、EPE の計算がきれいになる。
    """
    rng = np.random.default_rng(seed)
    margin = max(abs(dx), abs(dy)) + 4
    H2, W2 = h + 2 * margin, w + 2 * margin

    # 追跡しやすいよう「色付きブロブ + 斜めグラデ + ランダム矩形」でリッチなテクスチャを作る
    canvas = (rng.normal(128, 28, size=(H2, W2, 3))).astype(np.float32)
    grad = np.linspace(0, 60, W2)[None, :, None] + np.linspace(0, 40, H2)[:, None, None]
    canvas += grad
    for _ in range(40):
        x = rng.integers(0, W2 - 14)
        y = rng.integers(0, H2 - 14)
        s = int(rng.integers(6, 14))
        color = rng.integers(0, 256, size=3).astype(np.float32)
        canvas[y:y + s, x:x + s] = color
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # frame2[r,c] = frame1[r-dy, c-dx] になるよう、窓の原点を (-dy, -dx) ずらす → フロー = (dx, dy)
    r0, c0 = margin, margin
    frame1 = canvas[r0:r0 + h, c0:c0 + w].copy()
    frame2 = canvas[r0 - dy:r0 - dy + h, c0 - dx:c0 - dx + w].copy()

    gt_flow = np.zeros((h, w, 2), dtype=np.float32)
    gt_flow[..., 0] = dx
    gt_flow[..., 1] = dy
    return frame1, frame2, gt_flow


# =====================================================================
# 4) 合成シーン: 姿勢推定用（人型図形のアニメ + 真のキーポイント）
# =====================================================================

def _humanoid_joints(t: float, h: int, w: int) -> np.ndarray:
    """アニメ位相 t∈[0,1] に応じた COCO17 キーポイント (17, 3) を計算する。

    単純な棒人間モデル: 頭→肩→腰→膝→足首を固定比率で置き、腕だけ t で振らせる。
    返す v(可視フラグ)はすべて 2（可視）。座標は (x, y)。
    """
    cx = w * 0.5
    head_y = h * 0.16
    shoulder_y = h * 0.30
    hip_y = h * 0.55
    knee_y = h * 0.74
    ankle_y = h * 0.93
    sw = w * 0.16   # 肩幅の半分
    hw = w * 0.10   # 腰幅の半分

    # 腕の角度を t で振る（手を上げ下げ）。angle は肩から手首への方向。
    swing = np.sin(t * 2 * np.pi)
    arm_dx = w * 0.16
    elbow_y = shoulder_y + h * 0.10 - h * 0.06 * (swing + 1)
    wrist_y = shoulder_y + h * 0.20 - h * 0.16 * (swing + 1)

    kp = np.zeros((17, 3), dtype=np.float32)

    def put(i: int, x: float, y: float) -> None:
        kp[i] = (x, y, 2)

    put(0, cx, head_y)                       # nose
    put(1, cx - w * 0.03, head_y - h * 0.02)  # left_eye
    put(2, cx + w * 0.03, head_y - h * 0.02)  # right_eye
    put(3, cx - w * 0.06, head_y)             # left_ear
    put(4, cx + w * 0.06, head_y)             # right_ear
    put(5, cx - sw, shoulder_y)              # left_shoulder
    put(6, cx + sw, shoulder_y)              # right_shoulder
    put(7, cx - sw - arm_dx * 0.5, elbow_y)  # left_elbow
    put(8, cx + sw + arm_dx * 0.5, elbow_y)  # right_elbow
    put(9, cx - sw - arm_dx, wrist_y)        # left_wrist
    put(10, cx + sw + arm_dx, wrist_y)       # right_wrist
    put(11, cx - hw, hip_y)                  # left_hip
    put(12, cx + hw, hip_y)                  # right_hip
    put(13, cx - hw, knee_y)                 # left_knee
    put(14, cx + hw, knee_y)                 # right_knee
    put(15, cx - hw, ankle_y)                # left_ankle
    put(16, cx + hw, ankle_y)                # right_ankle
    return kp


def make_humanoid_frames(
    n: int = 4, h: int = 360, w: int = 240, seed: int = 2
) -> list[tuple[np.ndarray, np.ndarray]]:
    """腕を振る人型図形を n フレーム合成する。各フレーム (rgb, gt_keypoints(17,3)) を返す。

    keypoint R-CNN は実写の人で学習されているため、この“おもちゃ図形”では検出が 0 になりやすい。
    その domain gap（合成→実写のズレ）自体が重要な学びなので、ここでは「真の関節位置」を
    こちらが知っている状態で図形を描き、スケルトン描画・OKS の説明に使えるようにする。
    """
    rng = np.random.default_rng(seed)
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n):
        t = k / max(n - 1, 1)
        kp = _humanoid_joints(t, h, w)
        rgb = np.full((h, w, 3), 235, dtype=np.uint8)
        rgb += rng.normal(0, 3, size=rgb.shape).astype(np.int16).clip(-8, 8).astype(np.uint8)
        skin = (235, 200, 170)
        cloth = (70, 110, 170)

        # 胴（肩中点-腰中点）を太線で
        sh_mid = ((kp[5, :2] + kp[6, :2]) / 2).astype(int)
        hip_mid = ((kp[11, :2] + kp[12, :2]) / 2).astype(int)
        cv2.line(rgb, tuple(sh_mid), tuple(hip_mid), cloth, int(w * 0.16))
        # 四肢
        for a, b, thick in [
            (5, 7, 0.07), (7, 9, 0.06), (6, 8, 0.07), (8, 10, 0.06),
            (11, 13, 0.08), (13, 15, 0.07), (12, 14, 0.08), (14, 16, 0.07),
        ]:
            pa = tuple(kp[a, :2].astype(int))
            pb = tuple(kp[b, :2].astype(int))
            cv2.line(rgb, pa, pb, cloth, int(w * thick))
        # 頭
        head = tuple(kp[0, :2].astype(int))
        cv2.circle(rgb, head, int(w * 0.085), skin, -1)
        frames.append((rgb, kp))
    return frames


# =====================================================================
# 5) 可視化
# =====================================================================

def colorize_depth(depth: np.ndarray, cmap: str = "magma", invert: bool = False) -> np.ndarray:
    """深度マップ(H,W)をカラーマップで (H,W,3) uint8 RGB に。percentile で外れ値に頑健に正規化。"""
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    rgba = matplotlib.colormaps[cmap](norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def draw_skeleton(
    rgb: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int] = (0, 200, 0),
    vis_thresh: float = 0.0,
) -> np.ndarray:
    """COCO17 キーポイント (17,3=x,y,v) をスケルトンとして描画した RGB 画像を返す（元は破壊しない）。"""
    canvas = rgb.copy()
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    for a, b in COCO_SKELETON:
        if kp[a, 2] > vis_thresh and kp[b, 2] > vis_thresh:
            pa = (int(kp[a, 0]), int(kp[a, 1]))
            pb = (int(kp[b, 0]), int(kp[b, 1]))
            cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)
    for i in range(kp.shape[0]):
        if kp[i, 2] > vis_thresh:
            cv2.circle(canvas, (int(kp[i, 0]), int(kp[i, 1])), 3, (255, 60, 60), -1, cv2.LINE_AA)
    return canvas


def draw_flow_arrows(
    rgb: np.ndarray, flow: np.ndarray, step: int = 16, color: tuple[int, int, int] = (0, 220, 0)
) -> np.ndarray:
    """密フロー (H,W,2) を一定間隔の矢印で重ね描き（quiver 風）。可視化確認用。"""
    canvas = rgb.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = float(flow[y, x, 0]), float(flow[y, x, 1])
            cv2.arrowedLine(
                canvas, (x, y), (int(x + fx), int(y + fy)), color, 1, cv2.LINE_AA, tipLength=0.4
            )
    return canvas


# =====================================================================
# 6) 評価指標（scripts が使う実装版。exercises は同じ式を手で書き直す）
# =====================================================================

def align_scale_median(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """相対深度を絶対深度に近づけるための中央値スケール合わせ: pred * median(gt)/median(pred)。"""
    mp = float(np.median(pred))
    mg = float(np.median(gt))
    if abs(mp) < 1e-8:
        return pred.copy()
    return pred * (mg / mp)


def abs_rel(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """AbsRel = mean(|pred-gt| / gt)。深度評価の主指標の1つ（相対誤差の平均）。"""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.abs(pred - gt) / np.clip(gt, eps, None)))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """RMSE = sqrt(mean((pred-gt)^2))。大きな誤差を強く罰する二乗平均平方根誤差。"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def delta_accuracy(pred: np.ndarray, gt: np.ndarray, thr: float = 1.25, eps: float = 1e-6) -> float:
    """δ<thr: 各画素で max(pred/gt, gt/pred) < thr となる割合（threshold accuracy）。"""
    pred = np.clip(np.asarray(pred, dtype=np.float64), eps, None)
    gt = np.clip(np.asarray(gt, dtype=np.float64), eps, None)
    ratio = np.maximum(pred / gt, gt / pred)
    return float(np.mean(ratio < thr))


def endpoint_error(flow_pred: np.ndarray, flow_gt: np.ndarray) -> float:
    """EPE = mean(||flow_pred - flow_gt||_2)。各画素のフローベクトル差のユークリッド長の平均。"""
    diff = np.asarray(flow_pred, dtype=np.float64) - np.asarray(flow_gt, dtype=np.float64)
    return float(np.mean(np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)))


def object_keypoint_similarity(
    pred_xy: np.ndarray, gt: np.ndarray, area: float, sigmas: np.ndarray = COCO_SIGMAS
) -> float:
    """OKS を計算する。COCO 定義: KS_i = exp(-d_i^2 / (2 s^2 k_i^2)), k_i = 2σ_i, s^2 = area。

    可視(v>0)の点だけを分母に数え、その平均を返す（IoU の姿勢版に当たる類似度, 1 が完全一致）。
    pred_xy: (17,2), gt: (17,3=x,y,v)。
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt = np.asarray(gt, dtype=np.float64)
    vis = gt[:, 2] > 0
    if not np.any(vis):
        return 0.0
    k = 2.0 * np.asarray(sigmas, dtype=np.float64)
    d2 = (pred_xy[:, 0] - gt[:, 0]) ** 2 + (pred_xy[:, 1] - gt[:, 1]) ** 2
    e = d2 / (2.0 * float(area) * (k ** 2) + 1e-12)
    ks = np.exp(-e)
    return float(np.sum(ks[vis]) / np.sum(vis))


def pck(
    pred_xy: np.ndarray, gt: np.ndarray, ref_len: float, alpha: float = 0.2
) -> float:
    """PCK@alpha: d_i <= alpha*ref_len を満たす可視キーポイントの割合（正しく当たった点の率）。"""
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt = np.asarray(gt, dtype=np.float64)
    vis = gt[:, 2] > 0
    if not np.any(vis):
        return 0.0
    d = np.sqrt((pred_xy[:, 0] - gt[:, 0]) ** 2 + (pred_xy[:, 1] - gt[:, 1]) ** 2)
    correct = d[vis] <= (alpha * float(ref_len))
    return float(np.mean(correct))


def round_up_to_multiple(value: int, multiple: int = 8) -> int:
    """value 以上で multiple の倍数になる最小の整数（RAFT の「8 の倍数」要件などに使う）。"""
    if value % multiple == 0:
        return value
    return value + (multiple - value % multiple)
