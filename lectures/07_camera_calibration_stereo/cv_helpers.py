"""第7回 共有ヘルパ（カメラ校正・ステレオ・エピポーラ幾何）。

このモジュールの 3 本のスクリプトと exercises.py から共通で使う部品を集めた。
ポイントは「実データ（撮影画像）が無くても完走できる」よう、
すべてのサンプル（チェスボード画像・歪み画像・ステレオ対）を numpy/cv2 で
**合成生成**していること。各スクリプトは他モジュールへ import せず自己完結する。

合成の核となる考え方:
  - カメラは「内部行列 K（焦点距離 fx,fy と主点 cx,cy）」と「歪み係数 dist」で表す。
  - 平面（Z=0）に置いたチェスボードを、既知の姿勢 (rvec, tvec) のカメラで
    cv2.projectPoints により**歪みを含めて**投影すれば、本物の写真と同じ
    幾何のチェスボード画像が作れる。全視点が同じ K, dist を共有するので、
    calibrateCamera は物理的に整合した解（≒真値）へ収束する。

このファイル単体を実行すると、合成と検出のスモークテストになる:
  uv run python lectures/07_camera_calibration_stereo/cv_helpers.py
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/07_camera_calibration_stereo/ にある。
#   parents[2] = プロジェクトルート（lecture-cv/）。どこから実行しても場所がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
MODULE_ID = "07_camera_calibration_stereo"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    出力先はこのスクリプト隣の outputs/（lectures/07_camera_calibration_stereo/outputs/）。
    """
    d = _HERE.parent / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- 真値カメラ（合成の基準） -------------------------------------------
# 校正で「当てに行く」答え。fx=fy=600px, 主点は画像中心、k1<0 のたる型歪みを入れる。
# dist = (k1, k2, p1, p2, k3)。k1=-0.28 はスマホ広角レンズくらいの中程度の歪み。
IMAGE_SIZE = (640, 480)  # (幅 W, 高さ H)。OpenCV の多くの API は (W, H) 順。
TRUE_K = np.array(
    [[600.0, 0.0, 320.0],
     [0.0, 600.0, 240.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
TRUE_DIST = np.array([-0.28, 0.10, 0.0, 0.0, 0.0], dtype=np.float64)

# チェスボードの「内側の角」の数 (列, 行)。10x7 マスの盤の内部頂点が 9x6=54 個。
PATTERN_SIZE = (9, 6)
SQUARE = 1.0  # 1 マスの物理サイズ（任意単位。校正の K は単位に依存しない）。


def chessboard_object_points(
    pattern_size: tuple[int, int] = PATTERN_SIZE, square: float = SQUARE
) -> np.ndarray:
    """チェスボード内側角の 3D 座標 (N,3) float32 を作る。盤は平面なので Z=0。

    並びは findChessboardCorners の既定（x が先に進む行優先）に合わせる。
    例: (0,0,0),(1,0,0),...,(8,0,0),(0,1,0),... ×square。
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    # np.mgrid[0:cols,0:rows].T で (r,c) を行優先に並べ、x=c, y=r を作る。
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def make_camera_poses(
    n_views: int,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square: float = SQUARE,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """盤を色々な角度・距離から見た姿勢 (rvec, tvec) を決定的に生成する。

    校正には「視点に角度のばらつき」があるほど良い（傾けた盤が歪みを露わにする）。
    盤の中心が光軸近くに来るよう tvec を組み、回転は ±0.35rad 程度に散らす。
    """
    cols, rows = pattern_size
    # 盤の中心（物体座標）。回転をこの周りで効かせると盤が画面内に収まりやすい。
    center = np.array([(cols - 1) / 2 * square, (rows - 1) / 2 * square, 0.0])
    rng = np.random.default_rng(seed)
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_views):
        ax = rng.uniform(-0.28, 0.28)
        ay = rng.uniform(-0.28, 0.28)
        az = rng.uniform(-0.22, 0.22)
        rvec = np.array([[ax], [ay], [az]], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        Z = rng.uniform(14.0, 17.0)  # 盤(幅約10単位)が白縁付きで画面に収まる距離。
        jx = rng.uniform(-0.8, 0.8)  # 中心からのずらし（毎回ど真ん中だと単調なので）。
        jy = rng.uniform(-0.6, 0.6)
        # 盤中心が (jx, jy, Z) に来るよう平行移動を逆算: t = 目標 - R·center。
        t = np.array([jx, jy, Z]) - R @ center
        poses.append((rvec, t.reshape(3, 1)))
    return poses


def render_chessboard_view(
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square: float = SQUARE,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """姿勢 (rvec,tvec) のカメラから見たチェスボード写真を合成する（歪み込み）。

    各マスの 4 頂点を projectPoints で（歪みを含めて）画像へ投影し、黒マスだけ塗る。
    背景＝白なので、盤の周りには自動的に白い「静音域(quiet zone)」ができ、
    findChessboardCorners が検出しやすい。頂点位置は幾何的に厳密なので、
    cornerSubPix 後のコーナーは理論値とほぼ一致し、校正が真値へ収束する。
    """
    cols, rows = pattern_size
    W, H = image_size
    img = np.full((H, W, 3), 255, dtype=np.uint8)  # 白背景＝静音域。

    # 1マスの境界を「各辺を細かく分割した曲線」として作る。歪みで辺が湾曲するため、
    # 4頂点だけだと辺が直線になりコーナー位置がわずかにずれる（特に k2,k3 の推定が悪化）。
    # 辺を 6 分割して投影すると、真の歪み曲線に沿った忠実なマスになる。
    steps = 6

    def square_boundary(i: int, j: int) -> np.ndarray:
        x0, y0 = i * square, j * square
        x1, y1 = (i + 1) * square, (j + 1) * square
        edge = np.linspace(0.0, 1.0, steps, endpoint=False)
        top = [(x0 + (x1 - x0) * t, y0) for t in edge]
        right = [(x1, y0 + (y1 - y0) * t) for t in edge]
        bottom = [(x1 - (x1 - x0) * t, y1) for t in edge]
        left = [(x0, y1 - (y1 - y0) * t) for t in edge]
        ring = top + right + bottom + left
        return np.array([[x, y, 0.0] for x, y in ring], dtype=np.float32)

    # 内側角 (0..cols-1, 0..rows-1) を内部頂点に持つには、マスを -1..cols-1 で張る。
    for i in range(-1, cols):
        for j in range(-1, rows):
            if (i + j) % 2 != 0:
                continue  # 黒マスだけ塗る（白マスは背景のまま）。
            proj, _ = cv2.projectPoints(square_boundary(i, j), rvec, tvec, K, dist)
            poly = proj.reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(img, [poly], (35, 35, 35), lineType=cv2.LINE_AA)
    return img


def detect_corners(
    img: np.ndarray, pattern_size: tuple[int, int] = PATTERN_SIZE
) -> tuple[bool, np.ndarray | None]:
    """findChessboardCorners → cornerSubPix の正準形。検出可否とサブピクセル角を返す。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    # サブピクセル精緻化（±0.001px または 30 反復で停止）。窓は (11,11)。
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def calibrate(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int] = IMAGE_SIZE,
    flags: int = cv2.CALIB_FIX_K3,
) -> tuple[float, np.ndarray, np.ndarray, list, list]:
    """calibrateCamera を呼ぶだけの薄いラッパ。戻り値の先頭が RMS 再投影誤差(px)。

    既定で CALIB_FIX_K3（k3=0 に固定）にしている。非魚眼レンズでは k3 は
    画像隅まで盤を写さないと不定になりやすく、固定するのが実務の定石。
    """
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )
    return ret, K, dist, rvecs, tvecs


def mean_reprojection_error(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list,
    tvecs: list,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """全視点の RMS 再投影誤差を自前で計算する（calibrateCamera の戻り値と一致するはず）。

    各点を projectPoints で再投影し、検出点との二乗距離を全点で合算 → 点数で割って sqrt。
    「総二乗誤差 / 総点数 の平方根」が calibrateCamera の定義する RMS。
    """
    total_sq = 0.0
    total_n = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        det = imgp.reshape(-1, 2)
        total_sq += float(np.sum((proj - det) ** 2))  # dx^2+dy^2 を全点合算。
        total_n += len(det)
    return float(np.sqrt(total_sq / total_n)) if total_n else 0.0


def make_distorted_image(
    clean: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """歪みの無い画像 clean から、レンズ歪みを加えた画像を合成する。

    仕組み: 歪み画像の各画素 p_d が「本来どこ(p_ideal)を見ているか」は
    cv2.undistortPoints(p_d, K, dist, P=K) が返す（= 歪み除去後の理想位置）。
    よって distorted[p_d] = clean[p_ideal] を remap で集めれば歪み画像になる。
    こうして作った画像は cv2.undistort(K,dist) でちょうど元の clean に戻る。
    """
    h, w = clean.shape[:2]
    # 歪み画像側の全画素座標 (u,v) を並べる。
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    # P=K を渡すと「理想（歪み無し）画像の画素座標」で返る。
    ideal = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    map_x = ideal[..., 0].astype(np.float32)
    map_y = ideal[..., 1].astype(np.float32)
    return cv2.remap(
        clean, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def make_calibration_grid(image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """歪みが目で見える「直線格子＋チェッカ」画像を作る（undistort デモ用）。

    直線は歪むと曲がるので、歪み・歪み補正の効果を一目で確認できる。
    """
    W, H = image_size
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    # 等間隔の縦横グリッド線（黒）。直線性が歪みの有無で一目瞭然。
    for x in range(0, W + 1, 40):
        cv2.line(img, (x, 0), (x, H), (40, 40, 40), 1, cv2.LINE_AA)
    for y in range(0, H + 1, 40):
        cv2.line(img, (0, y), (W, y), (40, 40, 40), 1, cv2.LINE_AA)
    # 外周の太枠（歪みで最も曲がる部分なので強調）。
    cv2.rectangle(img, (2, 2), (W - 3, H - 3), (200, 60, 60), 3, cv2.LINE_AA)
    # 角に色つき円を置いて、補正後にどこへ動いたか追えるようにする。
    for cx, cy, col in [(40, 40, (0, 0, 220)), (W - 40, 40, (0, 160, 0)),
                        (40, H - 40, (220, 120, 0)), (W - 40, H - 40, (160, 0, 160))]:
        cv2.circle(img, (cx, cy), 12, col, -1, cv2.LINE_AA)
    return img


# --- ステレオ（平行化済みの水平視差ペア） -------------------------------
# 既に平行化(rectify)された理想ステレオを直接合成する。歪み=0, 行は完全に一致し、
# 視差は水平方向のみ。視差 d と深度 Z は d = f*baseline / Z で結ばれる。
STEREO_F = 600.0       # 焦点距離 [px]
STEREO_CX = 320.0
STEREO_CY = 240.0
STEREO_BASELINE = 0.10  # 左右カメラ間距離 [m]。f*baseline=60 なので d=60/Z。


def make_stereo_pair(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """平行化済みステレオ対 (left, right) と、各物体の正解 (領域・視差・深度) を返す。

    シーンは奥行きの違う複数の長方形（高周波テクスチャ付き）。右画像は左画像と
    「同じテクスチャを視差ぶん左へずらした」もの。だからブロックマッチングが効き、
    SGBM は各物体の視差 d を復元できる。f*baseline=60 で d=60/Z。
    """
    W, H = 640, 480
    rng = np.random.default_rng(seed)
    max_disp = 64

    def texture(h: int, w: int, tint: tuple[int, int, int]) -> np.ndarray:
        """高周波ノイズ＋色味のテクスチャ。マッチングが一意に決まるよう細かくする。"""
        noise = rng.integers(0, 90, size=(h, w, 1), dtype=np.int16)
        base = np.array(tint, dtype=np.int16).reshape(1, 1, 3)
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    # 背景は横長に作り、左右で開始位置をずらして「視差 12 の遠景」を表現する。
    bg_disp = 12  # Z=5.0m
    bg = texture(H, W + max_disp, (60, 70, 90))
    left = bg[:, 0:W].copy()
    right = bg[:, bg_disp:bg_disp + W].copy()

    # 手前の物体（領域は左画像基準）。視差が大きいほど手前。奥→手前の順に描く。
    objects = [
        dict(x0=70, y0=110, x1=230, y1=330, disp=24, tint=(40, 120, 160)),   # Z=2.5
        dict(x0=360, y0=70, x1=560, y1=300, disp=40, tint=(160, 80, 60)),    # Z=1.5
        dict(x0=250, y0=250, x1=420, y1=440, disp=64, tint=(70, 160, 70)),   # Z=0.9375
    ]
    gt: list[dict] = [dict(name="background", x0=0, y0=0, x1=W, y1=H,
                           disp=bg_disp, depth=STEREO_F * STEREO_BASELINE / bg_disp)]
    for k, ob in enumerate(objects):
        w = ob["x1"] - ob["x0"]
        h = ob["y1"] - ob["y0"]
        tex = texture(h, w, ob["tint"])
        d = ob["disp"]
        # 左画像: そのままの位置へ。
        left[ob["y0"]:ob["y1"], ob["x0"]:ob["x1"]] = tex
        # 右画像: 同じテクスチャを視差 d だけ左へ。
        xr0, xr1 = ob["x0"] - d, ob["x1"] - d
        if xr0 >= 0:
            right[ob["y0"]:ob["y1"], xr0:xr1] = tex
        gt.append(dict(name=f"object{k + 1}", **ob,
                       depth=STEREO_F * STEREO_BASELINE / d))

    # ごく弱いセンサノイズ（マッチングを壊さない程度）。
    left = np.clip(left.astype(np.int16) + rng.integers(-2, 3, left.shape), 0, 255).astype(np.uint8)
    right = np.clip(right.astype(np.int16) + rng.integers(-2, 3, right.shape), 0, 255).astype(np.uint8)
    return left, right, gt


def reproject_matrix(
    f: float = STEREO_F, cx: float = STEREO_CX, cy: float = STEREO_CY,
    baseline: float = STEREO_BASELINE,
) -> np.ndarray:
    """平行化済みステレオ（cx=cx'）の再投影行列 Q（4x4）を手作りする。

    [X,Y,Z,W]^T = Q·[u,v,d,1]^T。実3D点は (X/W, Y/W, Z/W)。
    この Q では W=d/baseline, Z=f なので 実Z = f*baseline/d となり視差→深度に一致。
    """
    return np.array(
        [[1.0, 0.0, 0.0, -cx],
         [0.0, 1.0, 0.0, -cy],
         [0.0, 0.0, 0.0, f],
         [0.0, 0.0, 1.0 / baseline, 0.0]],
        dtype=np.float64,
    )


def _smoke_test() -> None:
    """合成と検出が動くかの簡易確認（このファイルを直接実行したとき）。"""
    print("[smoke] チェスボード合成 + コーナー検出")
    poses = make_camera_poses(6)
    objp = chessboard_object_points()
    n_found = 0
    for rvec, tvec in poses:
        img = render_chessboard_view(TRUE_K, TRUE_DIST, rvec, tvec)
        found, _ = detect_corners(img)
        n_found += int(found)
    print(f"  検出成功 {n_found}/{len(poses)} 視点 / objp.shape={objp.shape}")

    print("[smoke] 歪み画像の合成")
    clean = make_calibration_grid()
    distorted = make_distorted_image(clean, TRUE_K, TRUE_DIST)
    print(f"  clean={clean.shape}, distorted={distorted.shape}")

    print("[smoke] ステレオ対の合成")
    left, right, gt = make_stereo_pair()
    print(f"  left={left.shape}, right={right.shape}, 物体数={len(gt)}")
    print("  Q=\n", reproject_matrix())
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
