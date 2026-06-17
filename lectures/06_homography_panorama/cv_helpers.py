"""第6回モジュール共通ヘルパ（ホモグラフィ推定とパノラマ合成）。

このファイルは「ホモグラフィの章で何度も書く定型処理」をまとめた小さな道具箱です。
本講座の方針として、各スクリプト（01〜03）と exercises.py はこのヘルパを import します。
（※ import するのは "同じモジュール内" のこのファイルだけ。他モジュールには依存しません。
   ネット・GPU・サンプル画像が無くても、ここで全部を numpy/cv2 で合成生成します。）

ここに置く責務は次の5つだけに絞っています。
  1. 特徴の豊富な合成シーン画像を作る（ORB/SIFT が点を見つけられるよう、角・テキストを多めに）
  2. そのシーンを既知のホモグラフィで切り出し「重なりのある複数視点」を作る
     （= 真のホモグラフィが分かるので、推定結果を答え合わせできる）
  3. ORB マッチング＋Lowe 比率テストで対応点を得る（章の入口）
  4. findHomography(RANSAC)・再投影誤差・キャンバス計算・フェザー合成といった部品
  5. 出力先 lectures/06_homography_panorama/outputs/ の管理と、評価用の SSIM

コーディング原則: 各関数は1つのことだけを行い、賢すぎる抽象化は避けます。
スクリプト本体側は、学習上重要な処理（findHomography 等）をあえて生のまま書いて
「読んで学べる」ようにしつつ、ここの部品を必要に応じて呼びます。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/06_homography_panorama/ にある。
#   parents[2] = プロジェクトルート（lecture-cv/）。どこから実行しても場所がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
OUTPUT_ROOT = _HERE.parent / "outputs"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    存在しなければ自動で作成する。
    """
    d = OUTPUT_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d


# =====================================================================
# 1. 合成シーン（特徴が豊富な「世界」）を作る
# =====================================================================

def make_scene(height: int = 760, width: int = 1700, seed: int = 0) -> np.ndarray:
    """ORB/SIFT が角点を多数拾える、特徴の豊富な平面シーンを BGR uint8 で作る。

    パノラマ合成は「2枚に共通して写っている模様」を頼りに位置合わせする。
    そのため、のっぺりした画像ではなく、角・エッジ・テキストが各所に散らばった
    平面（ポスターの壁のようなもの）を生成する。乱数は seed で固定し、毎回同じ絵にする。
    """
    rng = np.random.default_rng(seed)
    # ベースは薄いグレー。完全な単色より少しムラがある方が自然。
    img = np.full((height, width, 3), 235, dtype=np.uint8)

    # (a) 大きめの色付き矩形をタイル状に敷く（強いエッジ＝コーナーの素になる）。
    for _ in range(70):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        w = int(rng.integers(40, 160))
        h = int(rng.integers(40, 160))
        color = tuple(int(c) for c in rng.integers(20, 230, size=3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)

    # (b) 塗りつぶし円をばらまく（丸いブロブもコーナー検出を助ける）。
    for _ in range(80):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        r = int(rng.integers(8, 36))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(img, (cx, cy), r, color, thickness=-1)

    # (c) 斜め線でグリッド模様（直線の交点は良い特徴になる）。
    for _ in range(40):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(c) for c in rng.integers(0, 120, size=3))
        cv2.line(img, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

    # (d) テキスト（文字の角は ORB が大好物。位置合わせの目印にもなる）。
    words = ["CV", "OpenCV", "HOMOGRAPHY", "RANSAC", "PANORAMA", "warp", "ORB", "SIFT"]
    for i in range(24):
        text = words[int(rng.integers(0, len(words)))]
        org = (int(rng.integers(20, width - 160)), int(rng.integers(40, height - 20)))
        scale = float(rng.uniform(0.7, 1.6))
        color = tuple(int(c) for c in rng.integers(0, 80, size=3))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    return img


def _scene_quad(center_x: int, skew: int, top: int = 80, bottom: int = 680,
                half_width: int = 360) -> np.ndarray:
    """シーン座標上に「カメラが切り取る四辺形」を作る（左上→右上→右下→左下の順）。

    skew で上辺を左右にずらし、わずかな台形にすることで『カメラの首振り（ヨー）』を模す。
    こうすると2視点の関係が単純な平行移動ではなく本物の射影変換（ホモグラフィ）になり、
    findHomography を学ぶ題材として適切になる。
    """
    x0, x1 = center_x - half_width, center_x + half_width
    return np.float32(
        [
            [x0 + skew, top + 20],     # 左上
            [x1 + skew, top - 10],     # 右上
            [x1 - skew, bottom + 15],  # 右下
            [x0 - skew, bottom - 5],   # 左下
        ]
    )


def _view_from_scene(scene: np.ndarray, quad: np.ndarray,
                     out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    """シーンの四辺形 quad を out_w×out_h の長方形へ射影変換して「1視点」を作る。

    返り値は (視点画像, H)。H は「シーン座標 → 視点画像座標」のホモグラフィ。
    getPerspectiveTransform は4点対応から3x3行列を解いて返す（最低4点必要の原典）。
    """
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    view = cv2.warpPerspective(scene, H, (out_w, out_h))
    return view, H


def make_two_views(out_w: int = 640, out_h: int = 480, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """重なりのある2視点 (A=左, B=右) と、真のホモグラフィ H_BtoA を返す。

    H_BtoA は「B 画像の画素座標 → A 画像の画素座標」へ写す 3x3 行列。
    両視点とも同じ平面シーンを別アングルで撮ったものなので、両者は厳密にホモグラフィで結ばれる。
    推定した H をこの真値と比べれば、自分の実装の正しさを定量的に確認できる。
    """
    scene = make_scene(seed=seed)
    quad_a = _scene_quad(center_x=430, skew=25)   # 左カメラ
    quad_b = _scene_quad(center_x=820, skew=-18)  # 右カメラ（少し違う首振り）
    view_a, h_a = _view_from_scene(scene, quad_a, out_w, out_h)  # シーン→A
    view_b, h_b = _view_from_scene(scene, quad_b, out_w, out_h)  # シーン→B
    # A = h_a · scene, scene = inv(h_b) · B  ⇒  A = h_a · inv(h_b) · B
    h_b_to_a = h_a @ np.linalg.inv(h_b)
    h_b_to_a = h_b_to_a / h_b_to_a[2, 2]  # スケールを正規化（[2,2]=1 に揃える）
    return view_a, view_b, h_b_to_a.astype(np.float64)


def make_three_views(out_w: int = 640, out_h: int = 480, seed: int = 0
                     ) -> list[np.ndarray]:
    """左→中→右に重なる3視点を返す（パノラマを順次つなぐ題材）。"""
    scene = make_scene(seed=seed)
    centers_skews = [(380, 22), (820, -16), (1260, 24)]
    views = []
    for cx, skew in centers_skews:
        quad = _scene_quad(center_x=cx, skew=skew)
        view, _ = _view_from_scene(scene, quad, out_w, out_h)
        views.append(view)
    return views


# =====================================================================
# 2. 特徴点マッチング（ORB + Lowe 比率テスト）
# =====================================================================

def orb_match(img1: np.ndarray, img2: np.ndarray, ratio: float = 0.75,
              nfeatures: int = 3000):
    """ORB で2枚から対応を取り、(kp1, kp2, good_matches) を返す。

    流れ（第5回の復習）:
      1. detectAndCompute でキーポイントと記述子（ORB はバイナリ記述子）。
      2. BFMatcher(NORM_HAMMING) で各点の上位2近傍を knnMatch。
      3. Lowe の比率テスト: 最近傍が2番目より十分近い(距離比<ratio)対応だけ残す。
    比率テストは誤対応をここで大量に間引くための定石。残った good を後段の RANSAC に渡す。
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return kp1, kp2, []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue  # 近傍が1個しか無いと比率テストできない
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return kp1, kp2, good


def points_from_matches(kp1, kp2, matches) -> tuple[np.ndarray, np.ndarray]:
    """マッチ列を findHomography が受け取る形 (N,1,2) float32 の点配列2つに変換する。

    pts1 は img1 側、pts2 は img2 側の対応座標（同じ index 同士が対応）。
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2


# =====================================================================
# 3. ホモグラフィ推定・評価・幾何の部品
# =====================================================================

def estimate_homography(pts_src: np.ndarray, pts_dst: np.ndarray,
                        thresh: float = 3.0):
    """findHomography(RANSAC) で pts_src → pts_dst のホモグラフィを推定。

    返り値 (H, mask)。
      - H: 3x3。pts_dst ≈ H · pts_src を満たす（4点未満や退化時は None）。
      - mask: (N,1) の 0/1。1 が RANSAC のインライア（モデルに合った対応）。
    ransacReprojThreshold は「何ピクセルまでをインライアとみなすか」のしきい値。
    """
    if pts_src is None or len(pts_src) < 4:
        return None, None  # ホモグラフィは最低4対応点が必要
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, thresh)
    return H, mask


def reprojection_error(H: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray,
                       mask: np.ndarray | None = None) -> float:
    """H で pts_src を写した点と pts_dst の平均ユークリッド距離（px）を返す。

    mask を渡すとインライアだけで誤差を測る（外れ値を含めると無意味に大きくなるため）。
    値が小さいほど「ぴったり重なる」良いホモグラフィ。
    """
    if H is None:
        return float("inf")
    projected = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H).reshape(-1, 2)
    target = pts_dst.reshape(-1, 2)
    dist = np.linalg.norm(projected - target, axis=1)
    if mask is not None:
        keep = mask.ravel().astype(bool)
        if keep.any():
            dist = dist[keep]
    return float(dist.mean())


def warp_corners(H: np.ndarray, width: int, height: int) -> np.ndarray:
    """画像(width×height)の四隅を H で写した先の座標 (4,2) を返す。

    perspectiveTransform は点群にホモグラフィを適用する関数。
    パノラマのキャンバスサイズ計算や、平面物体の枠（バウンディング四角形）描画に使う。
    """
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H).reshape(-1, 2)


def canvas_for(images: list[np.ndarray], homographies: list[np.ndarray]
               ) -> tuple[np.ndarray, tuple[int, int]]:
    """各画像を共通フレームへ写したとき、全部が収まるキャンバスの平行移動 T とサイズを返す。

    warpPerspective は出力が原点(0,0)から始まる。負の座標へはみ出す画像があると切れるため、
    全画像の四隅を集めて最小座標を求め、それを 0 に押し込む平行移動 T を作る。
    返り値 (T, (W, H))。実際の warp では「T @ 各ホモグラフィ」を渡す。
    """
    all_corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        all_corners.append(warp_corners(H, w, h))
    pts = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.floor(pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(pts.max(axis=0)).astype(int)
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    size = (int(x_max - x_min), int(y_max - y_min))
    return T, size


def feather_weight(shape: tuple[int, int]) -> np.ndarray:
    """画像中央ほど重く、縁ほど 0 に近い重みマップ（フェザー合成用）を返す。

    距離変換で「画像の縁からの距離」を計算して正規化する。重なり領域でこの重みを使うと、
    継ぎ目（シーム）が滑らかに溶け合い、単純な上書きで出る段差を消せる。
    """
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist = dist / (dist.max() + 1e-6)
    return dist.astype(np.float32)


def build_panorama(images: list[np.ndarray], ratio: float = 0.75
                   ) -> tuple[np.ndarray, list[int]]:
    """左→右に重なる複数画像を、先頭画像を基準に順次つないでパノラマを返す。

    手順:
      1. 隣り合う (i-1, i) で ORB マッチ → findHomography で「i → i-1」の H を推定。
      2. それらを掛け合わせ、各画像 i を基準フレーム(画像0)へ写す行列 M_i を作る。
      3. 全画像の四隅からキャンバスを決め、フェザー重みで加重平均合成する。
    返り値 (パノラマ画像, 各ペアのインライア数リスト)。
    """
    n = len(images)
    m_to_ref = [np.eye(3, dtype=np.float64)]  # M_0 = 単位行列
    inliers = []
    for i in range(1, n):
        kp1, kp2, good = orb_match(images[i - 1], images[i], ratio)
        pts_prev, pts_cur = points_from_matches(kp1, kp2, good)
        # 「現在画像 cur → ひとつ前 prev」へ写す H を推定。
        H, mask = estimate_homography(pts_cur, pts_prev)
        if H is None:
            raise RuntimeError(f"画像 {i - 1} と {i} の位置合わせに失敗（対応点不足）")
        inliers.append(int(mask.sum()))
        m_to_ref.append(m_to_ref[-1] @ H)  # 合成して基準フレームへ

    T, size = canvas_for(images, m_to_ref)
    acc = np.zeros((size[1], size[0], 3), dtype=np.float64)
    wsum = np.zeros((size[1], size[0]), dtype=np.float64)
    for img, M in zip(images, m_to_ref):
        full = T @ M
        warped = cv2.warpPerspective(img, full, size)
        weight = feather_weight(img.shape[:2])
        wwarp = cv2.warpPerspective(weight, full, size)
        acc += warped.astype(np.float64) * wwarp[:, :, None]
        wsum += wwarp
    wsum[wsum == 0] = 1.0  # ゼロ割り回避（どの画像も無い空白部）
    pano = (acc / wsum[:, :, None]).clip(0, 255).astype(np.uint8)
    return pano, inliers


# =====================================================================
# 4. 評価用 SSIM（参照 scikit-image 無しで cv2 だけで実装）
# =====================================================================

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """2枚のグレースケール画像の SSIM（構造的類似度, 1 に近いほど似ている）を返す。

    scikit-image を使わず、ガウシアン窓での局所平均・分散・共分散から定義通り計算する。
    パノラマの「重なり領域で2枚がどれだけ一致しているか」= 位置合わせの良さの指標に使う。
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    k = (11, 11)
    s = 1.5
    mu_a = cv2.GaussianBlur(a, k, s)
    mu_b = cv2.GaussianBlur(b, k, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = cv2.GaussianBlur(a * a, k, s) - mu_a2
    var_b = cv2.GaussianBlur(b * b, k, s) - mu_b2
    cov_ab = cv2.GaussianBlur(a * b, k, s) - mu_ab
    num = (2 * mu_ab + c1) * (2 * cov_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


if __name__ == "__main__":
    # 直接実行すると、合成サンプルを生成して outputs/ に保存するスモークテスト。
    out = output_dir()
    a, b, h_true = make_two_views()
    cv2.imwrite(str(out / "helper_view_a.png"), a)
    cv2.imwrite(str(out / "helper_view_b.png"), b)
    print("[cv_helpers] サンプル2視点を生成しました:", out)
    print("  真のホモグラフィ H_BtoA =\n", np.round(h_true, 4))
