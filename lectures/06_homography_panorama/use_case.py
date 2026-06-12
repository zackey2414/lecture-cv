"""use_case.py — 実践ユースケース: 手持ち写真をつなぐ「パノラマ作成ツール」。

この章で学んだ「特徴マッチ → findHomography(RANSAC) → warpPerspective → ブレンド」を、
"自分の写真フォルダを 1 枚の横長パノラマに変換する小さな CLI ツール" として完結させた
ものです。教材のデモではなく、明日からあなたの写真で使える"道具"の最小実装です。

--------------------------------------------------------------------------
mini_project.py との違い（役割分担）
--------------------------------------------------------------------------
  * mini_project.py … 章の学びを"採点・検証"するための総合課題。合成データ専用で、
                      4枚パノラマ＋平面物体のまっすぐ化＋SSIM 評価＋JSON レポートまで
                      盛り込んだ「教材の完成形」。
  * use_case.py（本ファイル）… "実データを食わせて成果物を1つ出す"現実の小ツール。
                      data/ に置いた自分の重なり写真を読み、横長パノラマ 1 枚を作る。
                      手作りパイプライン（cv_helpers.build_panorama）を本筋に、失敗したら
                      cv2.Stitcher に自動フォールバックして、できるだけ"絵を出し切る"。
                      引数（入力フォルダ・出力先・比率テストしきい値など）で実運用できる。

--------------------------------------------------------------------------
実データの置き方（実データ優先・無ければ合成で必ず完走）
--------------------------------------------------------------------------
  1. プロジェクト直下に data/06_homography_panorama/ を作る（無ければ自動作成）。
  2. そこへ「左→右に少しずつ重なる」写真を入れる。三脚でその場で首を振って（パンして）
     撮ると上手くつながる。重なりは隣どうしで 30〜50% 程度が目安。
     例:  data/06_homography_panorama/01_left.jpg
          data/06_homography_panorama/02_mid.jpg
          data/06_homography_panorama/03_right.jpg
  3. 並び順は「ファイル名の昇順 = 左→右」と解釈する。01_, 02_, ... と番号を付けると確実。
  4. 画像が 2 枚未満なら、特徴の豊富な合成 3 視点（cv_helpers が生成）に自動で切り替えて
     必ず最後まで動く（exit 0）。実写でも合成でも、同じパイプラインがそのまま走る。

--------------------------------------------------------------------------
実行例
--------------------------------------------------------------------------
  # 既定（data/06_homography_panorama/ を読む。無ければ合成データで完走）
  uv run python lectures/06_homography_panorama/use_case.py

  # 入力フォルダ・出力先・比率テストしきい値を指定
  uv run python lectures/06_homography_panorama/use_case.py \
      --input data/06_homography_panorama --output outputs/06_homography_panorama/my_pano.png \
      --ratio 0.8 --max-width 1280

  # 手作りが苦しい難しめの写真は、最初から cv2.Stitcher（自動）で
  uv run python lectures/06_homography_panorama/use_case.py --stitcher

  # ローカル GUI があれば完成パノラマを Tkinter ウィンドウでプレビュー（headless では自動スキップ）
  uv run python lectures/06_homography_panorama/use_case.py --show

--------------------------------------------------------------------------
拡張アイデア（ここから練習で育てる）
--------------------------------------------------------------------------
  * 自動順序推定: 総当たりで隣接ペアのインライア数を測り、つながりが強い順に並べ替えて
    「フォルダに入れる順番を気にしないツール」にする。
  * 露出補正: 隣接画像の重なり領域で平均輝度を合わせてから合成し、明るさ段差を減らす。
  * クロップ: 合成後に黒余白を contour/boundingRect で自動トリミングして長方形に整える。
  * 360 度対応: 平面合成ではなく cv2.Stitcher_PANORAMA（円筒/球面投影）に寄せる。
  * EXIF 尊重: スマホ写真は ImageOps.exif_transpose で向きを正してから読む（横倒し防止）。
  * 失敗診断: ペアごとのインライア数/比をログに出し、つながらない箇所を特定できるようにする。

すべて CPU・追加依存なし・ネット非依存・headless 安全（画面表示はせず保存が基本）。
日本語コメントを厚めにしているので、読みながら自分の道具へ改造してください。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import cv2
import numpy as np

# 表示はせず保存が基本なので matplotlib は Agg バックエンド（headless 安全）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同じフォルダの cv_helpers.py（数字始まりでないので import 可）から部品を借りる。
# ※ 01〜03 の番号付きスクリプトと mini_project.py は import できない仕様なので、
#   ここで使う低レベル部品はすべて cv_helpers から取るか、本ファイル内に自己完結で書く。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    PROJECT_ROOT,
    build_panorama,
    estimate_homography,
    make_three_views,
    orb_match,
    output_dir,
    points_from_matches,
    reprojection_error,
)

# RANSAC・ORB の乱数を固定して、毎回同じ結果（再現可能）にする。
cv2.setRNGSeed(0)

# 読み込む画像拡張子（小文字で比較）。
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =====================================================================
# 1. 入力（実データ優先・合成フォールバック）
# =====================================================================

def default_input_dir() -> pathlib.Path:
    """既定の入力フォルダ data/06_homography_panorama/ を返す（無ければ作る）。"""
    d = PROJECT_ROOT / "data" / "06_homography_panorama"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _imread_unicode(path: pathlib.Path) -> np.ndarray | None:
    """日本語など非 ASCII パスでも読めるように np.fromfile + imdecode で読む。

    cv2.imread は環境によってマルチバイトパスを読めないことがある（章の落とし穴）。
    実ツールでは現実のファイル名に強い読み方をしておくと事故が減る。失敗時は None。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR で返る
    except OSError:
        return None


def _resize_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """横幅が max_width を超える画像だけ縮小する（CPU と warp のメモリ節約）。

    縮小には INTER_AREA が定石（モアレが出にくい）。十分小さければそのまま返す。
    """
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    new_size = (max_width, max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def load_images(input_dir: pathlib.Path, max_width: int
                ) -> tuple[list[np.ndarray], list[str]]:
    """入力フォルダから画像をファイル名昇順で読み込み、(画像列, 名前列) を返す。

    並び順 = 左→右と解釈する（番号付きファイル名推奨）。読めないファイルは黙って飛ばす。
    """
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    images: list[np.ndarray] = []
    names: list[str] = []
    for p in files:
        img = _imread_unicode(p)
        if img is None:
            print(f"  [skip] 読み込めませんでした: {p.name}")
            continue
        images.append(_resize_to_width(img, max_width))
        names.append(p.name)
    return images, names


def synthetic_views() -> tuple[list[np.ndarray], list[str]]:
    """実データが無いときの合成フォールバック（左→中→右の重なり 3 視点）。

    cv_helpers.make_three_views() は特徴の豊富な同一平面シーンを別アングルで切り出すので、
    実写が無くてもパノラマ合成パイプラインが必ず最後まで動く（exit 0 を保証）。
    """
    views = make_three_views()
    names = [f"synthetic_{i:02d}.png" for i in range(len(views))]
    return views, names


# =====================================================================
# 2. つなぐ（手作りを本筋に、ダメなら cv2.Stitcher へ自動フォールバック）
# =====================================================================

_STITCHER_STATUS = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


def stitch_with_opencv(images: list[np.ndarray]) -> np.ndarray | None:
    """cv2.Stitcher（自動）で合成。PANORAMA→SCANS の順に試し、status を必ず確認する。

    全モード失敗なら None（呼び出し側で握りつぶさず、正常終了を保てるように）。
    """
    for mode_name, mode in (("PANORAMA", cv2.Stitcher_PANORAMA),
                            ("SCANS", cv2.Stitcher_SCANS)):
        stitcher = cv2.Stitcher_create(mode)
        status, pano = stitcher.stitch(images)
        label = _STITCHER_STATUS.get(status, str(status))
        print(f"  cv2.Stitcher({mode_name}): {label}")
        if status == cv2.Stitcher_OK:
            return pano
    return None


def make_panorama(images: list[np.ndarray], ratio: float, force_stitcher: bool
                  ) -> tuple[np.ndarray | None, str]:
    """画像列を 1 枚のパノラマに合成し、(パノラマ, 使った手法名) を返す。

    方針:
      1. force_stitcher が False なら、まず手作りパイプライン build_panorama を試す
         （仕組みが全部見えるのが利点。本章の核）。
      2. 手作りが対応点不足などで失敗（RuntimeError）したら、cv2.Stitcher へ自動で降りる。
      3. それも失敗したら None を返す（呼び出し側でメッセージを出し正常終了）。
    """
    if not force_stitcher:
        try:
            pano, inliers = build_panorama(images, ratio=ratio)
            print(f"  手作りパイプライン成功: ペアごとのインライア数 = {inliers}")
            return pano, "manual"
        except RuntimeError as e:
            print(f"  手作りパイプライン失敗（{e}）→ cv2.Stitcher にフォールバック")

    pano = stitch_with_opencv(images)
    if pano is not None:
        return pano, "cv2.Stitcher"
    return None, "failed"


def first_pair_quality(images: list[np.ndarray], ratio: float) -> str:
    """先頭ペアの「インライア数/比・再投影誤差」を一行に要約して返す（品質の一次指標）。

    つながり具合をざっと見るためのログ。失敗しても評価不能の旨を返すだけで例外は投げない。
    """
    if len(images) < 2:
        return "ペアが無いため品質評価なし"
    kp1, kp2, good = orb_match(images[0], images[1], ratio=ratio)
    if len(good) < 4:
        return f"対応点が少なく評価不能（good={len(good)}）"
    pts0, pts1 = points_from_matches(kp1, kp2, good)
    H, mask = estimate_homography(pts1, pts0)  # 2枚目 → 1枚目
    if H is None or mask is None:
        return f"ホモグラフィ推定不可（good={len(good)}）"
    inliers = int(mask.sum())
    err = reprojection_error(H, pts1, pts0, mask)
    ratio_in = inliers / max(len(good), 1)
    return (f"先頭ペア: good={len(good)} / インライア={inliers}({ratio_in:.0%}) "
            f"/ 再投影誤差={err:.2f}px")


# =====================================================================
# 3. 出力（パノラマ本体＋入力サムネ付きの確認図）
# =====================================================================

def save_overview(out_path: pathlib.Path, inputs: list[np.ndarray],
                  names: list[str], pano: np.ndarray, method: str) -> None:
    """入力サムネ（上段）と完成パノラマ（下段）を 1 枚にまとめて保存（確認用）。

    matplotlib は RGB 前提なので cv2.cvtColor で BGR→RGB に直してから渡す。
    図タイトルは環境差で文字化けしないよう ASCII に限定する。
    """
    n = len(inputs)
    fig = plt.figure(figsize=(min(4 * n, 16), 7))
    for i, (img, name) in enumerate(zip(inputs, names)):
        ax = fig.add_subplot(2, n, i + 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # ファイル名は非 ASCII を含み得るので index 表示に留める（タイトル ASCII 維持）。
        ax.set_title(f"input {i}")
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
    ax.set_title(f"panorama ({method})  {pano.shape[1]}x{pano.shape[0]}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)


def preview_with_tkinter(pano: np.ndarray) -> bool:
    """完成パノラマを Tkinter ウィンドウで軽くプレビューする（任意・成功で True）。

    headless（DISPLAY 無し）や Tkinter/PIL が無い環境では何もせず False を返す。
    imshow は使わない方針なので、ローカル GUI が欲しい人向けの"おまけ"として分離している。
    どんな失敗でも例外を外へ出さず、ツール本体の正常終了（exit 0）を壊さない。
    """
    # Linux で DISPLAY が無ければ GUI は出せない（サーバ/CI/Docker 想定）。
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("  [preview] DISPLAY が無いためプレビューはスキップ（保存済みファイルを見てください）")
        return False
    try:
        import tkinter as tk
        from PIL import Image, ImageTk

        # 大きすぎると画面に収まらないので幅 960 程度へ縮小して表示。
        rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 960:
            scale = 960 / w
            rgb = cv2.resize(rgb, (960, max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        root = tk.Tk()
        root.title("panorama preview (close window to exit)")
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        tk.Label(root, image=photo).pack()
        root.mainloop()
        return True
    except Exception as e:  # noqa: BLE001  GUI 周りは何が来ても本体を止めない
        print(f"  [preview] プレビュー不可（{type(e).__name__}）→ 保存済みファイルを見てください")
        return False


# =====================================================================
# 4. CLI 本体
# =====================================================================

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="重なり写真をつなぐパノラマ作成ツール（実データ優先・合成フォールバック）")
    p.add_argument("--input", type=str, default=None,
                   help="入力画像フォルダ（既定: data/06_homography_panorama/）")
    p.add_argument("--output", type=str, default=None,
                   help="出力パノラマPNG（既定: outputs/06_homography_panorama/use_case_panorama.png）")
    p.add_argument("--ratio", type=float, default=0.75,
                   help="Lowe 比率テストのしきい値（緩めると対応が増える。既定 0.75）")
    p.add_argument("--max-width", type=int, default=1280,
                   help="この幅を超える入力は縮小（CPU/メモリ節約。既定 1280）")
    p.add_argument("--stitcher", action="store_true",
                   help="手作りを使わず最初から cv2.Stitcher（自動）で合成する")
    p.add_argument("--show", action="store_true",
                   help="完成パノラマを Tkinter でプレビュー（headless では自動スキップ）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = output_dir()
    out_path = (pathlib.Path(args.output) if args.output
                else out_dir / "use_case_panorama.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 入力を決める（実データ優先・合成フォールバック） ---------------
    input_dir = pathlib.Path(args.input) if args.input else default_input_dir()
    print(f"[1] 入力フォルダ: {input_dir}")
    images, names = load_images(input_dir, args.max_width)
    if len(images) < 2:
        print(f"  実画像が {len(images)} 枚（2枚未満）。合成データに切り替えます。")
        print(f"  → {input_dir} に左→右へ重なる写真を入れると、それを使います。")
        images, names = synthetic_views()
        source = "synthetic"
    else:
        print(f"  実画像 {len(images)} 枚を読み込み: {', '.join(names)}")
        source = "real"

    # --- 品質の一次指標（先頭ペア）をログ ------------------------------
    print(f"[2] {first_pair_quality(images, args.ratio)}")

    # --- つなぐ -------------------------------------------------------
    print("[3] パノラマ合成")
    pano, method = make_panorama(images, ratio=args.ratio,
                                 force_stitcher=args.stitcher)
    if pano is None:
        # ここまで来ても合成できないのは、重なり/特徴が極端に乏しい実写など。
        # 失敗を握りつぶさずメッセージを出し、入力サムネだけ残して正常終了する。
        print("  すべての手法で合成に失敗しました。")
        print("  ヒント: 隣どうしの重なりを増やす / 順番を左→右に / --ratio 0.8 を試す。")
        contact = cv2.hconcat([cv2.resize(im, (320, 240)) for im in images])
        cv2.imwrite(str(out_dir / "use_case_inputs.png"), contact)
        print(f"  入力一覧を保存: {out_dir / 'use_case_inputs.png'}")
        return 0  # ツールとしては"診断を出して正常終了"を正とする

    # --- 保存 ---------------------------------------------------------
    cv2.imwrite(str(out_path), pano)
    overview_path = out_dir / "use_case_overview.png"
    save_overview(overview_path, images, names, pano, method)
    print(f"[4] 完成: 手法={method} / size={pano.shape[1]}x{pano.shape[0]} / source={source}")
    print(f"  パノラマ: {out_path}")
    print(f"  確認図:   {overview_path}")

    # --- 任意プレビュー（headless では自動スキップ） --------------------
    if args.show:
        preview_with_tkinter(pano)

    print("完了。outputs/06_homography_panorama/ を確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
