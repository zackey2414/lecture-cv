"""実践ユースケース: ポートレート背景ぼかし（深度 bokeh）ツール。

スマホの「ポートレートモード」を 1 枚画像から再現する小ツール。単眼深度
（Depth Anything V2 Small）で各画素の手前/奥を推定し、**ピントを合わせた面から
遠い画素ほど強くぼかす**ことで、被写体を浮き立たせる背景ボケ（bokeh）を作ります。

------------------------------------------------------------------
これは何か / mini_project.py との違い
------------------------------------------------------------------
- mini_project.py は「深度→姿勢→フロー」を 1 本に通した“学習用の総合レポート”。
- 本 use_case.py は **深度だけを使う現実の小ツール**として完結します。出力 use_case_bokeh.png は
  そのまま SNS アイコンやサムネに使える“成果物”で、パラメータ（焦点面・被写界深度・ボケ量）を
  いじって遊べる「動く出発点」です。番号付きスクリプトは import せず、深度推定は本ファイル内に
  自己完結で実装しています（借用するのは dpf_helpers の合成シーン/可視化だけ）。

------------------------------------------------------------------
データの置き方（実データ優先・無ければ合成で必ず完走）
------------------------------------------------------------------
- data/27_depth_pose_flow/ に画像（*.png / *.jpg / *.jpeg）を 1 枚以上置くと、その先頭を
  自動で使います。**人物が手前にいるポートレート**を置くと効果が分かりやすいです。
- 置かなければ合成室内シーン（手前の箱・中景の球・奥の壁）を生成して動きます。
- ネット不通などでモデル重み DL に失敗しても、合成シーンなら GT 深度、実画像ならランプ深度に
  フォールバックして最後まで通ります（必ず exit 0）。

------------------------------------------------------------------
拡張アイデア（ここを書き換えて練習）
------------------------------------------------------------------
- 焦点面を動かす: apply_depth_bokeh(focus_pct=...) を変え「奥にピント」「手前にピント」を試す。
- 被写界深度(DoF)の幅: band を小さくすると“ピント面が薄く”背景が大きくボケる。
- 円形ボケ（玉ボケ）: GaussianBlur を円盤カーネル(cv2.filter2D)に替えると点光源が玉になる。
- レイヤ DoF: ボケ量を数段に量子化し段ごとに別カーネルで合成すると、より自然な遠近ボケに。
- バッチ化: data/ の全画像をループして一括加工する CLI に拡張する。
- Tkinter スライダで focus_pct/band を対話調整（headless では本ファイルのファイル保存にフォールバック）。

実行: uv run python lectures/27_depth_pose_flow/use_case.py
出力: lectures/27_depth_pose_flow/outputs/use_case_*.png / use_case_bokeh.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")  # headless 前提: 画面が無くても図を PNG 保存できるバックエンド
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import dpf_helpers as H  # noqa: E402

# Depth Anything V2 Small（01 と同じ小型モデル。CPU でも数秒）。
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# =====================================================================
# 1) 入力の取得（実データ優先・合成フォールバック）
# =====================================================================

def load_input() -> tuple[np.ndarray, np.ndarray | None]:
    """入力 RGB と（合成のときだけ）GT 深度を返す。

    data/ に実画像があればそれを優先（GT は無し）。無ければ合成室内シーンを生成。
    """
    H.DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        p for p in H.DATA_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if candidates:
        img = np.array(Image.open(candidates[0]).convert("RGB"))
        print(f"[入力] data/ の実画像を使用: {candidates[0].name}（人物ポートレート推奨）")
        return img, None
    print("[入力] 合成室内シーン（手前=箱 / 中景=球 / 奥=壁）を生成")
    rgb, gt = H.make_depth_scene(h=288, w=384, seed=0)
    return rgb, gt


# =====================================================================
# 2) 単眼深度の推定（モデル → 失敗時フォールバック）
# =====================================================================

def estimate_inverse_depth(
    rgb: np.ndarray, gt_depth: np.ndarray | None
) -> tuple[np.ndarray, str]:
    """各画素の“逆深度”(H,W)を返す。慣習として **値が大きいほどカメラに近い**。

    1) Depth Anything V2 で推定（pipeline）。出力は入力と解像度が違う場合があるので必ずリサイズ。
    2) 失敗時: 合成なら GT 深度の逆数、実画像なら縦ランプでフォールバック（exit 0 を死守）。
    """
    h, w = rgb.shape[:2]
    try:
        from transformers import pipeline

        # device 省略で CPU 実行。pipeline が前処理〜後処理まで面倒を見る。
        pipe = pipeline("depth-estimation", model=MODEL_ID)
        out = pipe(Image.fromarray(rgb))
        pred = np.asarray(out["predicted_depth"], dtype=np.float32)
        # predicted_depth はモデル解像度のことがある → 入力サイズへ合わせる（深度を画像に重ねるため）。
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
        return pred, "Depth-Anything-V2-Small"
    except Exception as e:  # noqa: BLE001  ネット不通/未導入でも止めない
        print(f"[警告] 深度モデル実行に失敗（{type(e).__name__}: {e}）。フォールバックで継続。")
        if gt_depth is not None:
            # 合成シーンは GT 深度を持つので、逆深度(=1/depth)に変換すれば意味のあるボケが作れる。
            inv = 1.0 / (gt_depth.astype(np.float32) + 1e-3)
            return inv, "synthetic-GT(fallback)"
        # 実画像で GT が無い場合: 上=遠/下=近の縦ランプ（パイプラインの形だけ保つ）。
        ramp = np.linspace(0.1, 1.0, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
        return ramp, "ramp(fallback)"


def _normalize01(inv_depth: np.ndarray) -> np.ndarray:
    """逆深度を percentile で頑健に [0,1] へ（外れ値に強い・near=1, far=0）。"""
    d = inv_depth.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


# =====================================================================
# 3) 深度ボケ（被写界深度シミュレーション）
# =====================================================================

def apply_depth_bokeh(
    rgb: np.ndarray,
    inv_depth: np.ndarray,
    focus_pct: float = 82.0,
    band: float = 0.16,
    blur_ksize: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """深度に応じて背景をぼかしたポートレート風画像を作る。

    考え方（カメラの被写界深度の近似）:
      - 逆深度を [0,1] に正規化（near=1）。
      - 「ピントを合わせる面」focus_val を分位点 focus_pct で決める（既定は手前の被写体）。
      - focus_val から ±band 以内はくっきり、外れるほどボケ強度 alpha を 0→1 に上げる。
      - alpha をぼかして被写体の輪郭が硬く切れないよう“ふち取り”を柔らかくする。
      - 出力 = くっきり画像*(1-alpha) + 全面ぼかし画像*alpha のアルファ合成。

    返り値: (bokeh_rgb(H,W,3) uint8, blur_strength(H,W) float32[0,1], focus_val float)
    """
    h, w = rgb.shape[:2]
    norm = _normalize01(inv_depth)  # near=1, far=0

    # ピント面（分位点で決める）。focus_pct を上げるほど“より手前”にピントが合う。
    focus_val = float(np.percentile(norm, focus_pct))

    # ピント面からの距離 → ボケ強度。band 以内は 0（くっきり）、外側は線形に 1 へ。
    dist = np.abs(norm - focus_val)
    blur_strength = np.clip((dist - band) / max(band, 1e-6), 0.0, 1.0).astype(np.float32)
    # 被写体エッジで硬く切れないよう alpha を軽くぼかす（ふち取りのフェザリング）。
    blur_strength = cv2.GaussianBlur(blur_strength, (0, 0), sigmaX=max(1.0, w / 200.0))

    # 背景用の強いボケ。カーネルは画像幅に比例（奇数化）。点光源を玉ボケにしたいなら円盤カーネルへ。
    if blur_ksize is None:
        r = max(5, w // 36)
        blur_ksize = 2 * r + 1
    blurred = cv2.GaussianBlur(rgb, (blur_ksize, blur_ksize), 0)

    alpha = blur_strength[..., None]  # (H,W,1) でブロードキャスト
    out = rgb.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), blur_strength, focus_val


# =====================================================================
# 4) 保存・可視化
# =====================================================================

def save_montage(
    out_dir: pathlib.Path,
    rgb: np.ndarray,
    depth_vis: np.ndarray,
    blur_strength: np.ndarray,
    bokeh: np.ndarray,
    source: str,
) -> pathlib.Path:
    """入力 / 深度 / ボケ強度マスク / 仕上がり を 2x2 の 1 枚に（ASCII タイトル）。"""
    mask_vis = (np.clip(blur_strength, 0, 1) * 255).astype(np.uint8)
    mask_rgb = np.repeat(mask_vis[..., None], 3, axis=2)  # グレースケールを 3ch に
    titles = [
        "input",
        f"depth ({source})",
        "blur mask (white=blur)",
        "portrait bokeh result",
    ]
    imgs = [rgb, depth_vis, mask_rgb, bokeh]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, im, ti in zip(axes.ravel(), imgs, titles):
        ax.imshow(im)
        ax.set_title(ti, fontsize=11)
        ax.axis("off")
    fig.suptitle("27_depth_pose_flow : portrait depth bokeh", fontsize=13)
    fig.tight_layout()
    path = out_dir / "use_case_montage.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    out = H.ensure_output_dir()
    H.set_seed(0)
    print("=== 実践ユースケース: ポートレート背景ぼかし（深度 bokeh）===")

    # 1) 入力
    rgb, gt_depth = load_input()

    # 2) 深度推定（逆深度: 大=近い）
    inv_depth, source = estimate_inverse_depth(rgb, gt_depth)
    print(f"[深度] source={source}, shape={inv_depth.shape}, "
          f"min={inv_depth.min():.3f}, max={inv_depth.max():.3f}（大きいほど近い）")

    # 3) 深度ボケ生成（手前=被写体にピント・奥をぼかす）
    focus_pct, band = 82.0, 0.16
    bokeh, blur_strength, focus_val = apply_depth_bokeh(
        rgb, inv_depth, focus_pct=focus_pct, band=band
    )
    sharp_ratio = float(np.mean(blur_strength < 0.5))
    print(f"[ボケ] focus_pct={focus_pct} (focus_val={focus_val:.3f}), band={band}")
    print(f"       くっきり領域の割合={sharp_ratio:.2%}（被写体が浮き出る面積の目安）")

    # 4) 保存（成果物 + 可視化 + メタ JSON）
    depth_vis = H.colorize_depth(inv_depth)
    Image.fromarray(rgb).save(out / "use_case_input.png")
    Image.fromarray(depth_vis).save(out / "use_case_depth.png")
    Image.fromarray(bokeh).save(out / "use_case_bokeh.png")
    montage = save_montage(out, rgb, depth_vis, blur_strength, bokeh, source)

    meta = {
        "depth_source": source,
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "focus_pct": focus_pct,
        "focus_val": round(focus_val, 4),
        "band": band,
        "sharp_ratio": round(sharp_ratio, 4),
    }
    (out / "use_case_bokeh.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out/'use_case_bokeh.png'} (成果物)")
    print(f"       {montage}, {out/'use_case_bokeh.json'}")
    print("完成: 1 枚画像から深度を推定し、背景だけぼかしたポートレートを生成できた。")
    print("ヒント: data/27_depth_pose_flow/ に人物写真を置くと効果が分かりやすい。")


if __name__ == "__main__":
    main()
