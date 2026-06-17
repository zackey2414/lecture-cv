"""use_case.py — 実践ユースケース: 「良品フォルダ」で覚えて検査フォルダを一括検品する PaDiM 検品ツール。

────────────────────────────────────────────────────────────────────────
これは何か（use_case_theme: 不良品検出 / 品質ゲート）
  正常（良品）画像だけで PaDiM を学習し、検査したい画像を 1 枚ずつ採点して
  **OK / NG 判定 + 異常ヒートマップ + 欠陥位置（バウンディングボックス）** を出す、
  現場寄りの「外観検査 CLI」です。ラベル（正解の不良/良品）は不要で、
  しきい値は良品の一部を学習に使わず取り置く **hold-out 較正 + マージン**で
  ラベル無しに決めます（実運用では不良サンプルが手に入らないことが多いため）。
  各画像の結果オーバーレイ PNG、一覧コンタクトシート、`inspection_report.csv`
  / `.json`（1 行 = 1 製品）を outputs/ に書き出します。

mini_project.py との違い（ここが大事）
  - mini_project.py … 教材の集大成。**ラベル付き評価セット**で AUROC/AUPR/PRO を
    測り、品質ゲート→異常検知→評価→合否の一連を「学ぶ」ためのデモ。
  - use_case.py（本ファイル）… **ラベル不要**。良品フォルダと検査フォルダを渡すと、
    各製品に OK/NG とヒートマップ・欠陥位置を付けて **検品レポートを吐く現場ツール**。
    フォルダを差し替えるだけで自分のデータに使える「動く出発点」。評価指標ではなく
    「1 個ずつの判定と成果物（レポート/画像）」を主役にしているのが違いです。

データの置き方（実データ優先・無ければ合成で必ず完走）
  data/32_anomaly_iqa/
    ├── train/   … 良品（正常）画像だけを入れる（PaDiM の学習用。10〜50 枚程度）
    └── test/    … 検査したい画像を入れる（OK/NG が混在していてよい）
  対応拡張子: .png .jpg .jpeg .bmp .webp。どちらかが空/無ければ **合成データに自動
  フォールバック**して必ず exit 0（CI でも動く）。実画像は同じ製品を同じ構図で撮ると
  PaDiM の「位置合わせ前提」が効いて精度が上がります。

  # 例: 自分のデータで検品（フォルダ指定）
  uv run python lectures/32_anomaly_iqa/use_case.py \
      --train-dir data/32_anomaly_iqa/train --test-dir data/32_anomaly_iqa/test --margin 1.3

拡張アイデア（練習用）
  1. しきい値較正を分位点ベース（hold-out 良品の 99 パーセンタイル）に変える／中央値+MAD の
     ロバスト統計にするなど、ノイズに強い動作点の決め方を試す。
  2. PaDiM を PatchCore（lab.patchcore_fit / patchcore_score）に差し替えて、位置ずれに
     強い検品にする（実データで効きやすい）。
  3. 入口に撮影品質ゲート（lab.variance_of_laplacian）を足し、ボケ画像は検査前に弾く。
  4. 欠陥 bbox を COCO/YOLO 形式で書き出し、二次アノテーションの種にする。
  5. --margin / --keep-dims をフォルダごとにチューニングして過検出・見逃しを調整する。

制約: CPU・headless（imshow 不使用, matplotlib=Agg）・図タイトルは ASCII・出力は
lectures/32_anomaly_iqa/outputs/。ネットに出るのは resnet18 重み DL のみ（失敗時はランダム初期化へ
フォールバックして完走）。重い依存（anomalib/pyiqa）は使わず lab の自前 PaDiM を借用。

実行（引数なしでも合成データで完走）:
  uv run python lectures/32_anomaly_iqa/use_case.py
出力:
  lectures/32_anomaly_iqa/outputs/usecase_overlays/*.png … 製品ごとの 入力+ヒートマップ+欠陥枠
  lectures/32_anomaly_iqa/outputs/usecase_contact_sheet.png … 検査結果の一覧（OK=緑枠 / NG=赤枠）
  lectures/32_anomaly_iqa/outputs/inspection_report.csv / .json … 1 行=1 製品の判定レポート
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless 用。GUI を一切開かないバックエンド
import matplotlib.pyplot as plt
import numpy as np

import anomaly_iqa_lab as lab

# 読み込む画像拡張子（大文字小文字は無視）。
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ============================================================================
# 1) データ取り込み — 実フォルダがあれば読む / 無ければ合成にフォールバック
# ============================================================================
def load_folder(path: str) -> tuple[list[np.ndarray], list[str]]:
    """フォルダ内の画像を RGB uint8 (IMG_SIZE 角) に揃えて読み込む。

    返り値: (画像リスト, ファイル名リスト)。読めない/無ければ空リスト。
    cv2 は BGR で読むので RGB へ変換する（この回は一貫して RGB を使う）。
    """
    if not os.path.isdir(path):
        return [], []
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS))
    imgs: list[np.ndarray] = []
    names: list[str] = []
    for f in files:
        bgr = lab.cv2.imread(os.path.join(path, f), lab.cv2.IMREAD_COLOR)
        if bgr is None:  # 壊れたファイルは飛ばす
            continue
        rgb = lab.cv2.cvtColor(bgr, lab.cv2.COLOR_BGR2RGB)
        rgb = lab.cv2.resize(rgb, (lab.IMG_SIZE, lab.IMG_SIZE), interpolation=lab.cv2.INTER_AREA)
        imgs.append(rgb)
        names.append(f)
    return imgs, names


def load_inputs(train_dir: str, test_dir: str) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    """良品(train)・検査(test)画像を用意する。実データが無ければ合成にフォールバック。

    返り値: (train, test, test_names, used_synthetic)。
    実データは「train と test の両方に 1 枚以上」あって初めて採用する（片方欠けは合成）。
    """
    train_imgs, _ = load_folder(train_dir)
    test_imgs, test_names = load_folder(test_dir)

    if train_imgs and test_imgs:
        return np.stack(train_imgs), np.stack(test_imgs), test_names, False

    # --- 合成フォールバック（必ず完走させるため）------------------------------
    ds = lab.make_dataset()  # 正常(train) + 正常/異常(test) + GT（GT は本ツールでは未使用）
    names = [f"normal_{i:02d}.png" for i in range(int((ds["labels"] == 0).sum()))]
    names += [f"defect_{i:02d}.png" for i in range(int((ds["labels"] == 1).sum()))]
    return ds["train"], ds["test"], names, True


# ============================================================================
# 2) 検品エンジン — 良品で PaDiM を学習し、検査画像を採点
# ============================================================================
def split_calibration(n: int, calib_frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """良品を「学習用」と「しきい値較正用(hold-out)」に分けるインデックスを返す。

    PaDiM は学習に使った良品を低く採点してしまう（過剰適合）ので、しきい値は
    **学習に使わない良品**で較正するのが正しい。良品が少ない(<6枚)ときは分けず、
    全量学習＋大きめマージンで代用する（その場合 calib は空配列）。
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    if n < 6:  # 較正用を取り置く余裕が無い → 全量学習にフォールバック
        return perm, np.array([], dtype=np.int64)
    n_calib = max(2, int(round(n * calib_frac)))
    return perm[n_calib:], perm[:n_calib]  # (fit_idx, calib_idx)


def fit_inspector(
    train_fit: np.ndarray, keep_dims: int = 100
) -> tuple[lab.FeatureExtractor, dict, np.ndarray]:
    """良品画像だけで PaDiM を学習し、(特徴抽出器, モデル, 使用チャネルidx) を返す。"""
    ext = lab.FeatureExtractor(seed=0)  # resnet18（重みDL失敗時はランダム初期化で完走）
    emb = ext.embed(train_fit)  # (N, Hc, Wc, C)
    idx = lab.select_dims(emb.shape[-1], keep_dims, seed=0)  # PaDiM の次元削減
    model = lab.padim_fit(emb[..., idx])  # 位置別ガウシアン（正常のみ）
    return ext, model, idx


def score_images(
    ext: lab.FeatureExtractor, model: dict, idx: np.ndarray, imgs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """画像群 → (異常マップ (N,H,W), image スコア (N,))。"""
    emb = ext.embed(imgs)
    maps = lab.upsample_maps(lab.padim_score(emb[..., idx], model))
    return maps, lab.image_scores(maps)


def decide_threshold(calib_scores: np.ndarray, margin: float) -> float:
    """較正用良品スコアの最大値 × マージンでしきい値を決める（ラベル不要）。

    学習に使っていない良品（未知の正常）の中で最も異常寄りの素点に安全マージン
    （既定 1.3 倍）を掛け、「正常のばらつきの少し外側」を NG の境界にする。
    margin を上げれば見逃し方向（NG を出しにくい）、下げれば過検出方向に振れる。
    """
    return float(calib_scores.max()) * float(margin)


# ============================================================================
# 3) 欠陥の位置 — 異常マップの高い領域を囲むバウンディングボックス
# ============================================================================
def locate_defect(amap: np.ndarray, rel: float = 0.5) -> tuple[int, int, int, int] | None:
    """異常マップで「最大値の rel 倍以上」の画素を囲む bbox (x0,y0,x1,y1) を返す。

    欠陥が無い（ほぼ平坦）な画像では None。現場では「どこを見ればいいか」の手がかりになる。
    """
    hi = amap >= (float(amap.max()) * rel)
    ys, xs = np.where(hi)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ============================================================================
# 4) 成果物 — 製品ごとのオーバーレイ / 一覧コンタクトシート / レポート
# ============================================================================
def save_overlay(img: np.ndarray, amap: np.ndarray, vmax: float, verdict: str, score_ratio: float,
                 bbox: tuple[int, int, int, int] | None, path: str) -> None:
    """1 製品分の図（左=入力, 右=ヒートマップ重ね+欠陥枠）を保存する。"""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(img)
    axes[0].set_title("input", fontsize=9)
    axes[1].imshow(img)
    axes[1].imshow(amap, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
    axes[1].set_title("anomaly heatmap", fontsize=9)
    if bbox is not None and verdict == "NG":  # NG のときだけ欠陥枠を描く
        x0, y0, x1, y1 = bbox
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=2)
        axes[1].add_patch(rect)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    color = "red" if verdict == "NG" else "green"
    # ASCII タイトル（日本語は文字化けするので英語）。score/thr 比で異常度を示す。
    fig.suptitle(f"[{verdict}] score/thr = {score_ratio:.2f}", color=color, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_contact_sheet(imgs: np.ndarray, verdicts: list[str], ratios: np.ndarray,
                       names: list[str], path: str, max_show: int = 12) -> None:
    """検査結果の一覧（OK=緑枠 / NG=赤枠）を 1 枚に並べる。"""
    n = min(len(imgs), max_show)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n:
            ax.axis("off")
            continue
        ax.imshow(imgs[k])
        color = "red" if verdicts[k] == "NG" else "green"
        ax.set_title(f"{verdicts[k]} {ratios[k]:.2f}", color=color, fontsize=9)
        for spine in ax.spines.values():  # 枠線を判定色で太く
            spine.set_color(color)
            spine.set_linewidth(3)
    fig.suptitle(f"Inspection results ({n} of {len(imgs)} shown; green=OK red=NG)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(rows: list[dict], csv_path: str, json_path: str, meta: dict) -> None:
    """1 行=1 製品の検品レポートを CSV と JSON で書き出す。"""
    fields = ["name", "verdict", "score", "threshold", "score_ratio", "defect_bbox"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row["defect_bbox"] = "" if r["defect_bbox"] is None else "|".join(map(str, r["defect_bbox"]))
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "items": rows}, f, ensure_ascii=False, indent=2)


# ============================================================================
# 5) パイプライン本体
# ============================================================================
def run(train_dir: str, test_dir: str, margin: float, keep_dims: int) -> None:
    out = lab.ensure_out_dir()
    overlay_dir = os.path.join(out, "usecase_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print("=" * 64)
    print("実践ユースケース: PaDiM 外観検品ツール（良品で覚えて OK/NG 判定）")

    # 1) 入力（実データ優先・無ければ合成）
    train, test, names, used_synth = load_inputs(train_dir, test_dir)
    src = "合成データ（フォールバック）" if used_synth else f"実データ {train_dir} / {test_dir}"
    print(f"  入力: {src}")
    print(f"  良品(学習) {len(train)} 枚 / 検査(test) {len(test)} 枚")

    # 2) 良品を「学習用」と「較正用(hold-out)」に分け、PaDiM を学習 → しきい値を決定
    fit_idx, calib_idx = split_calibration(len(train))
    ext, model, idx = fit_inspector(train[fit_idx], keep_dims=keep_dims)
    print(f"  特徴抽出器: resnet18 ({ext.mode}) / 次元 {len(idx)} に削減")
    # 較正用が取れたら未知良品スコアで、取れなければ学習良品スコアで較正（少数データ時）。
    calib_src = train[calib_idx] if len(calib_idx) else train[fit_idx]
    _, calib_scores = score_images(ext, model, idx, calib_src)
    thr = decide_threshold(calib_scores, margin)
    how = f"hold-out {len(calib_idx)} 枚" if len(calib_idx) else "学習良品で代用"
    print(f"  しきい値: 良品({how})の最大×{margin:.2f} = {thr:.2f}（ラベル不要）")

    # 3) 検査画像を採点 → OK/NG と欠陥位置
    maps, scores = score_images(ext, model, idx, test)
    vmax = float(maps.max()) if maps.size else 1.0
    verdicts: list[str] = []
    ratios = scores / max(thr, 1e-6)
    rows: list[dict] = []
    for i, name in enumerate(names):
        verdict = "NG" if scores[i] >= thr else "OK"
        verdicts.append(verdict)
        bbox = locate_defect(maps[i]) if verdict == "NG" else None
        # 製品ごとのオーバーレイ画像を保存
        stem = os.path.splitext(name)[0]
        save_overlay(test[i], maps[i], vmax, verdict, float(ratios[i]), bbox,
                     os.path.join(overlay_dir, f"{stem}.png"))
        rows.append({
            "name": name,
            "verdict": verdict,
            "score": round(float(scores[i]), 4),
            "threshold": round(thr, 4),
            "score_ratio": round(float(ratios[i]), 4),
            "defect_bbox": bbox,
        })

    n_ng = verdicts.count("NG")
    print(f"  判定: NG {n_ng} 枚 / OK {len(verdicts) - n_ng} 枚")

    # 4) 一覧コンタクトシート + レポート
    sheet_path = os.path.join(out, "usecase_contact_sheet.png")
    csv_path = os.path.join(out, "inspection_report.csv")
    json_path = os.path.join(out, "inspection_report.json")
    save_contact_sheet(test, verdicts, ratios, names, sheet_path)
    meta = {
        "source": "synthetic" if used_synth else "real",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "threshold": round(thr, 4),
        "margin": float(margin),
        "keep_dims": int(len(idx)),
        "feature_extractor_mode": ext.mode,
        "n_ng": int(n_ng),
    }
    write_report(rows, csv_path, json_path, meta)

    print(f"  保存: {overlay_dir}/ （製品ごとのヒートマップ {len(names)} 枚）")
    print(f"  保存: {sheet_path}")
    print(f"  保存: {csv_path}")
    print(f"  保存: {json_path}")
    print("  → 良品フォルダを差し替えれば、自分の製品の検品ツールとしてそのまま使える出発点。")


def main() -> None:
    parser = argparse.ArgumentParser(description="PaDiM 外観検品ツール（良品で覚えて OK/NG 判定）")
    parser.add_argument("--train-dir", default=os.path.join(lab.DATA_DIR, "train"),
                        help="良品(正常)画像フォルダ。無ければ合成にフォールバック")
    parser.add_argument("--test-dir", default=os.path.join(lab.DATA_DIR, "test"),
                        help="検査したい画像フォルダ。無ければ合成にフォールバック")
    parser.add_argument("--margin", type=float, default=1.3,
                        help="しきい値マージン（大きいほど NG を出しにくい）")
    parser.add_argument("--keep-dims", type=int, default=100,
                        help="PaDiM の次元削減で残すチャネル数")
    args = parser.parse_args()
    run(args.train_dir, args.test_dir, args.margin, args.keep_dims)


if __name__ == "__main__":
    main()
