"""実践ユースケース: ワンクリック写真自動補正ツール（バッチ処理）。

何をするツールか
  フォルダに放り込んだ「色かぶり・露出不足・低コントラスト・くすみ」で撮れた写真を、
  人手のパラメータ調整なしに 1 枚ずつ自動補正して一括で書き出す小さな現場ツールです。
  補正の流れは本章の王道どおり:
      ホワイトバランス(gray-world) → ガンマ(露出) → CLAHE(局所コントラスト) → 彩度
  ただし各段の強さ(ガンマ値・彩度倍率)を **画像自身の統計から自動で決める** のが肝です。

mini_project.py との違い（ここが本ファイルの存在意義）
  - mini_project.py は「正解の見え方(target)」が手元にある教材です。劣化版を合成し、
    ガンマ=1.7・彩度=1.2 といった **固定パラメータ** を当て、目標との ΔE で良し悪しを測ります。
  - 一方この use_case.py は **正解画像が無い現実の状況** を想定します。スマホやドラレコの
    写真には「本来こう見えるはず」という基準などありません。そこで補正量を、その画像の
    平均輝度・平均彩度から **自分で推定** します（no-reference な自動補正）。さらに 1 枚では
    なく **フォルダ単位のバッチ** として完結させ、補正前後を自己統計(mean_Y/std_Y/mean_S)で
    確認します。＝「教材のデモ」ではなく「そのまま使える小ツール」になっています。

実データの置き方（実データ優先・無ければ合成で必ず完走）
  プロジェクト直下に
      data/43_color_spaces_and_adjustments/
  を作り、補正したい写真（.jpg / .jpeg / .png / .bmp / .webp）を入れて実行してください。
  そこにあるものを優先して読み込みます（日本語・空白を含むパスでも読めるよう自前デコード）。
  画像が 1 枚も無ければ、色かぶり違いの合成劣化写真を 3 枚作ってデモし、必ず exit 0 します。

実行
  uv run python lectures/43_color_spaces_and_adjustments/use_case.py
  補正後画像と before/after モンタージュは outputs/43_color_spaces_and_adjustments/ に保存します
  （headless 前提・画面表示はしません）。

拡張アイデア（練習用の出発点としてどうぞ）
  - WB を white-patch（99 パーセンタイル最大値→255）や Shades-of-Gray に差し替えて比較する。
  - 自動ガンマを「平均輝度」ではなく「中央値」や「暗部 10 パーセンタイルが潰れない」基準にする。
  - 彩度を全体一律でなく、肌色領域(YCrCb inRange)だけ控えめにする選択的彩度に拡張する。
  - argparse で入力フォルダ・目標輝度・clipLimit を CLI 引数化し、本物の CLI ツールにする。
  - Pillow の ImageOps.exif_transpose でスマホ写真の向きを正規化してから補正する。
  - 基準画像(参照)がある場合だけ color_helpers.mean_delta_e で ΔE も出す分岐を足す。
  - 補正パラメータ(ガンマ・彩度倍率)を JSON でログ出力し、後でレシピを再現できるようにする。
"""

from __future__ import annotations

import math
import pathlib
import sys

import cv2
import numpy as np

# 番号付きスクリプト(01_..〜05_..)は import 不可だが、色道具箱の color_helpers は import できる。
# 本章の関数を借りて重複実装を避ける（補正ロジックは helpers に集約済み）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import (  # noqa: E402
    MODULE_ID,
    PROJECT_ROOT,
    adjust_gamma,
    apply_color_cast,
    clahe_luminance,
    gray_world_white_balance,
    imwrite_unicode,
    luminance,
    make_photo_scene_bgr,
    mean_saturation,
    output_dir,
    save_montage,
    scale_saturation,
)

# 入力フォルダ（実写真をここに置く）。無くてもよい（合成にフォールバックする）。
DATA_DIR = PROJECT_ROOT / "data" / MODULE_ID
# モンタージュに並べる最大枚数（多すぎる図は重く読みにくいので頭打ちにする）。
MAX_PANELS = 4


# =====================================================================
# 入力の読み込み（実データ優先・合成フォールバック）
# =====================================================================

def imread_unicode(path: pathlib.Path) -> np.ndarray | None:
    """日本語・空白を含むパスでも安全に読む imread の代替（失敗時は None）。

    cv2.imread は内部の C ファイル API の都合で非 ASCII パスを読めないことがある。
    「バイト列を np.fromfile で読む → cv2.imdecode で画像化」に分解して回避する。
    読み込み失敗（壊れ・非対応形式）でも例外を投げず None を返す（呼び出し側で弾く）。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # 常に BGR 3ch で返る
    except Exception:
        return None


def make_synthetic_degraded() -> list[tuple[str, np.ndarray]]:
    """data/ が空のときのフォールバック。色かぶり違いの劣化写真を 3 枚合成する。

    自動補正が「種類の違う劣化」に効くことを 1 回の実行で見せるため、
    暖色かぶり・寒色かぶり・全体くすみ の 3 パターンを用意する（いずれも露出不足ぎみ）。
    元シーンは seed 固定なので毎回同じ画像になる（出力の再現性を確保）。
    """
    base = make_photo_scene_bgr()
    warm = apply_color_cast(cv2.convertScaleAbs(base, alpha=0.55, beta=-12), (0.78, 1.0, 1.45))
    cool = apply_color_cast(cv2.convertScaleAbs(base, alpha=0.60, beta=8), (1.45, 1.0, 0.80))
    dull = scale_saturation(cv2.convertScaleAbs(base, alpha=0.50, beta=-4), 0.45)
    return [("synthetic_warm_dark", warm),
            ("synthetic_cool_flat", cool),
            ("synthetic_dim_lowsat", dull)]


def load_input_photos() -> list[tuple[str, np.ndarray]]:
    """補正対象の (名前, BGR画像) リストを返す。実写真があれば優先、無ければ合成。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos: list[tuple[str, np.ndarray]] = []
    if DATA_DIR.is_dir():
        for p in sorted(DATA_DIR.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            img = imread_unicode(p)
            if img is not None:
                photos.append((p.stem, img))
    if photos:
        print(f"[load] 実データ {len(photos)} 枚を data/{MODULE_ID}/ から読み込みました。")
        return photos
    print(f"[load] data/{MODULE_ID}/ に画像が無いので合成劣化写真でデモします。")
    print(f"       （実写真を data/{MODULE_ID}/ に置けばそれを自動補正します）")
    return make_synthetic_degraded()


# =====================================================================
# 補正量の自動推定（no-reference: 画像自身の統計から決める）
# =====================================================================

def estimate_auto_gamma(bgr: np.ndarray, target_mean: float = 0.5) -> float:
    """平均輝度を目標(既定 0.5)へ寄せるガンマ値を自動推定する。

    本章のガンマ定義 out=(in)^(1/gamma) を正規化(0-1)で考えると、
    平均輝度 mean を target に合わせたいとき近似的に mean^(1/gamma)=target。
    両辺の log を取ると gamma = log(mean)/log(target)。
      - 暗い写真(mean<target) → gamma>1 で持ち上げ、
      - 明るすぎ(mean>target) → gamma<1 で締める、
    と自動で向きが決まる。極端値で破綻しないよう [0.4, 3.0] にクリップする。
    """
    y = luminance(bgr).astype(np.float32) / 255.0
    mean = float(np.clip(y.mean(), 1e-3, 0.999))  # log の発散を避ける
    gamma = math.log(mean) / math.log(target_mean)
    return float(np.clip(gamma, 0.4, 3.0))


def estimate_auto_saturation(bgr: np.ndarray, target_s: float = 120.0) -> float:
    """平均彩度を目標(既定 120/255)へ寄せる彩度倍率を自動推定する。

    くすんだ写真(mean_S 小)は強めに、もともと鮮やかな写真は控えめに補正されるよう、
    倍率 = target_s / 現在の mean_S とする。やり過ぎ・色潰れを避けるため [0.8, 2.2] に制限。
    """
    s = mean_saturation(bgr)
    factor = target_s / max(s, 1.0)  # 0 割りを避ける
    return float(np.clip(factor, 0.8, 2.2))


def auto_correct(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """ワンクリック自動補正の本体。各段の強さは画像統計から自動で決める。

    順序に意味がある（mini_project と同じ思想）:
      1. WB を最初に当て、後段の露出/彩度が「かぶり色」を増幅しないようにする。
      2. 自動ガンマで露出を目標輝度へ。
      3. CLAHE(L チャンネル)で局所コントラストを回復（色は崩さない）。
      4. 自動彩度でくすみを補う。
    返り値: (補正後BGR, 使ったパラメータの記録)。記録はログ/レシピ再現に使える。
    """
    wb = gray_world_white_balance(bgr)
    gamma = estimate_auto_gamma(wb)
    expo = adjust_gamma(wb, gamma)
    local = clahe_luminance(expo, clip_limit=2.0, tile=8)
    factor = estimate_auto_saturation(local)
    corrected = scale_saturation(local, factor)
    return corrected, {"gamma": gamma, "sat": factor}


# =====================================================================
# 自己評価（正解画像が無いので、画像自身の統計で前後を比べる）
# =====================================================================

def photo_stats(bgr: np.ndarray) -> dict[str, float]:
    """補正の効きを測る自己統計。mean_Y=明るさ, std_Y=コントラスト, mean_S=鮮やかさ。"""
    y = luminance(bgr).astype(np.float32)
    return {"mean_Y": float(y.mean()), "std_Y": float(y.std()), "mean_S": mean_saturation(bgr)}


def main() -> None:
    out = output_dir()
    photos = load_input_photos()

    panels: list[tuple[str, np.ndarray]] = []
    print("\n[auto-correct] 1 枚ずつ自動補正（パラメータは画像統計から決定）")
    print(f"  {'name':22s} {'gamma':>6s} {'sat':>5s} | "
          f"{'Y(before>after)':>17s} {'std(b>a)':>13s} {'S(b>a)':>13s}")

    for name, bgr in photos:
        corrected, params = auto_correct(bgr)
        before, after = photo_stats(bgr), photo_stats(corrected)

        # 補正後画像をファイルに保存（実ツールとしての成果物）。
        save_name = f"use_case_{name}_corrected.png"
        imwrite_unicode(out / save_name, corrected)

        print(f"  {name[:22]:22s} {params['gamma']:6.2f} {params['sat']:5.2f} | "
              f"{before['mean_Y']:6.1f}>{after['mean_Y']:<6.1f}    "
              f"{before['std_Y']:5.1f}>{after['std_Y']:<5.1f}  "
              f"{before['mean_S']:5.1f}>{after['mean_S']:<5.1f}")

        # 最初の数枚だけ before/after をモンタージュに積む（図が重くなりすぎないように）。
        if len(panels) < MAX_PANELS * 2:
            panels.append((f"{name[:16]} (before)", bgr))
            panels.append((f"{name[:16]} (after)", corrected))

    montage_path = out / "use_case_one_click.png"
    save_montage(panels, montage_path,
                 suptitle="One-click auto correction (WB -> auto gamma -> CLAHE -> auto saturation)",
                 cols=2)

    print(f"\n[done] 補正後画像 {len(photos)} 枚と before/after モンタージュを保存しました。")
    print(f"       {montage_path}")
    print("       目安: mean_Y は中庸(約128)へ寄り、くすんだ写真ほど mean_S が上がる。")
    print("       （色かぶり写真は WB で水増しされた彩度が正されるため mean_S が下がることもある）")
    print(f"       実写真を data/{MODULE_ID}/ に置くと、それを自動補正します。")


if __name__ == "__main__":
    main()
