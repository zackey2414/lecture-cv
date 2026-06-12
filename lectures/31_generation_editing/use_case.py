"""use_case.py — EC 商品写真スタジオ（出品用バリエーション生成 + 不要物除去）の動く出発点。

────────────────────────────────────────────────────────────────────────
これは何のツールか
────────────────────────────────────────────────────────────────────────
ネットショップ（EC）の出品担当者が、1 つの商品について
  (A) 出品サムネイル用の **背景・ライティング違いのバリエーション画像**を SD-Turbo の
      text-to-image でまとめて作り（A/B テスト用の素材出し）、
  (B) 商品写真に写り込んだ **不要物（値札シール・透かし・汚れ）をマスク指定で消す**
      （cv2.inpaint によるクリーンアップ）、
という日々の「商品画像づくり」を 1 コマンドで回すための小ツールです。
成果物は「出品キット」として、バリエーションのコンタクトシート・クリーンアップ
before/after 図・キット内容を記した JSON を outputs/ に書き出します。

さらに各バリエーションを **CLIPScore（プロンプト整合度）でランク付け**し、
「一番『商品らしく』写ったヒーロー画像」を自動で選びます（出品の主画像候補）。

────────────────────────────────────────────────────────────────────────
実データの置き方（実データ優先・無ければ合成で必ず完走）
────────────────────────────────────────────────────────────────────────
  data/31_generation_editing/
    ├─ <任意名>.png / .jpg   … ベース商品写真（最初の 1 枚を使用）
    └─ *mask*.png            … 「消したい領域」のマスク（白=255 が除去対象。任意）
  - ベース画像が無ければ、簡単な「ボトル商品」を cv2 で合成して使います。
  - マスクが無ければ、商品にダミーの値札シールを貼ってからそれを除去する
    デモになります（自分の写真とマスクに差し替えれば実運用に近づきます）。
  外部ネットアクセスはモデル重みの初回 DL のみ。重みが取れない/壊れた環境でも
  古典フォールバック（色変換バリエーション + cv2.inpaint）で必ず exit 0 で終わります。

────────────────────────────────────────────────────────────────────────
mini_project.py との違い
────────────────────────────────────────────────────────────────────────
  mini_project.py … 章の全工程（生成→劣化→超解像復元→インペイント→PSNR/SSIM/CLIP 評価）を
                    1 本に通す「学習用の総合パイプライン」。指標の確認が主目的。
  use_case.py     … 「EC の商品画像づくり」という単一ユースケースに絞った現実の小ツール。
                    バリエーション“群”を作って CLIPScore でヒーローを選び、不要物を消す、という
                    出品ワークフローそのもの。指標は評価ではなく「採用画像の選定」に使う。

────────────────────────────────────────────────────────────────────────
SD-Turbo の CPU 設定（厳守）
────────────────────────────────────────────────────────────────────────
  float32 / num_inference_steps=1〜2 / guidance_scale=0.0 / 256〜384px /
  enable_attention_slicing()。guidance を上げると蒸留モデルでは絵が崩れる。

────────────────────────────────────────────────────────────────────────
拡張アイデア（練習）
────────────────────────────────────────────────────────────────────────
  - バリエーションを t2i ではなく **img2img（gl.load_img2img_pipeline）** にして、
    「同じ商品のまま背景だけ変える」一貫性を高める（IP-Adapter/ControlNet が本筋）。
  - 不要物マスクを手描きでなく **自動検出**（白い透かしを輝度しきい値で抽出、
    23 章の文プロンプトセグメンテーションで「price tag」を指定）に置き換える。
  - 大穴・物体除去は古典 cv2.inpaint では跡が残るので **LaMa / 拡散インペイント**へ。
  - 採用基準を CLIPScore 単独でなく「無参照 IQA（鮮鋭さ）× 整合」で複合スコア化。
  - 生成サイズ・スタイル一覧・採用枚数を商品カテゴリごとにプリセット化して量産。

実行例:
  uv run python lectures/31_generation_editing/use_case.py
  uv run python lectures/31_generation_editing/use_case.py --mode variations --num 4 --prompt "a product photo of a perfume bottle"
  uv run python lectures/31_generation_editing/use_case.py --mode cleanup
結果: outputs/31_generation_editing/use_case_variations.png / use_case_cleanup.png / use_case_listing.json
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless（imshow は使わない。ファイル保存のみ）
import matplotlib.pyplot as plt  # noqa: E402  Agg を設定してから読む

import gen_lab as gl  # 番号なしヘルパは import 可（合成データ・モデルローダ・指標を共有）

# 出品サムネ用の「背景・ライティング」スタイル一覧（ラベルはすべて ASCII＝図タイトル安全）。
STYLES: list[tuple[str, str]] = [
    ("marble", "on a white marble table, soft studio light, e-commerce product photo"),
    ("wood", "on a rustic wooden desk, warm natural light, lifestyle product photo"),
    ("studio", "on a seamless white studio background, sharp product photography"),
    ("nature", "on a stone with green leaves, outdoor natural light"),
    ("dark", "on a dark slate surface, dramatic moody lighting"),
    ("pastel", "on a pastel pink background, minimal flat lay"),
]

DEFAULT_PROMPT = "a product photo of a cosmetic bottle"


# ----------------------------------------------------------------------------
# ベース商品画像（data/ 優先・無ければ合成ボトル）
# ----------------------------------------------------------------------------
def make_product(size: int = 320, seed: int = 0) -> np.ndarray:
    """淡いスタジオ背景に「ボトル商品」を 1 つ描いた RGB uint8 画像（合成フォールバック）。

    バリエーション/クリーンアップの題材になるよう、ラベル・キャップ・影など
    “消したり差し替えたりしたくなる要素”を入れている。
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), np.uint8)
    # 背景: 上から下への淡いグレーグラデ（スタジオ撮影の無地背景イメージ）
    for y in range(size):
        v = int(232 - 40 * (y / size))
        img[y, :] = (v, v, v + 4)
    # 接地影（楕円をぼかす）
    shadow = np.zeros((size, size), np.uint8)
    cv2.ellipse(shadow, (size // 2, int(size * 0.82)), (size // 5, size // 22), 0, 0, 360, 60, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), size / 40)
    img = (img.astype(np.int16) - shadow[..., None] // 1).clip(0, 255).astype(np.uint8)
    # ボトル本体（角丸の縦長）＋キャップ
    bx, bw = size // 2 - size // 10, size // 5
    by, bh = int(size * 0.30), int(size * 0.48)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (70, 130, 190), -1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (40, 80, 120), 2, cv2.LINE_AA)
    cap_w = bw // 2
    cv2.rectangle(
        img, (size // 2 - cap_w // 2, by - size // 14), (size // 2 + cap_w // 2, by), (35, 35, 45), -1
    )
    # ラベル（明るい帯＋文字）。ここを差し替えたくなる＝編集対象の典型
    ly = by + bh // 3
    cv2.rectangle(img, (bx + 4, ly), (bx + bw - 4, ly + bh // 4), (240, 240, 235), -1)
    cv2.putText(
        img, "AURA", (bx + 10, ly + bh // 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 70), 1, cv2.LINE_AA
    )
    # ハイライト（光沢）と微ノイズで質感を足す
    cv2.line(img, (bx + 6, by + 6), (bx + 6, by + bh - 6), (200, 220, 240), 2, cv2.LINE_AA)
    noise = rng.randint(-4, 5, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def load_base_product(size: int, seed: int) -> tuple[np.ndarray, str]:
    """data/ の最初の商品画像（mask を除く）を読み、無ければ合成ボトルを返す。返り値 (RGB, 由来)。"""
    if os.path.isdir(gl.DATA_DIR):
        for f in sorted(os.listdir(gl.DATA_DIR)):
            if "mask" in f.lower():
                continue
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                bgr = cv2.imread(os.path.join(gl.DATA_DIR, f))
                if bgr is not None:
                    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
                    return rgb, f"real:{f}"
    return make_product(size, seed), "synthetic"


def find_data_mask(size: int) -> tuple[np.ndarray, str] | None:
    """data/ に *mask* を含む画像があれば「除去対象マスク（白=255）」として読む（無ければ None）。"""
    if not os.path.isdir(gl.DATA_DIR):
        return None
    for f in sorted(os.listdir(gl.DATA_DIR)):
        if "mask" in f.lower() and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            m = cv2.imread(os.path.join(gl.DATA_DIR, f), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (size, size))
                return (m > 127).astype(np.uint8) * 255, f"real:{f}"
    return None


# ----------------------------------------------------------------------------
# (A) 出品バリエーション生成（SD-Turbo t2i／失敗時は古典の色変換で代替）
# ----------------------------------------------------------------------------
def classical_variation(base_rgb: np.ndarray, idx: int) -> np.ndarray:
    """重みが無い環境向けの古典フォールバック: 色相シフト＋明度＋色付きトーンでバリエーション化。"""
    hsv = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + (idx * 30) % 180) % 180  # 色相を回す
    hsv[..., 2] = np.clip(hsv[..., 2] * (0.85 + 0.1 * (idx % 3)), 0, 255)  # 明るさ
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    tints = [(60, 120, 180), (180, 120, 60), (120, 180, 120), (150, 150, 150), (80, 80, 120), (200, 160, 200)]
    overlay = np.full_like(out, tints[idx % len(tints)])
    return cv2.addWeighted(out, 0.85, overlay, 0.15, 0)


def generate_variations(
    base_prompt: str, base_rgb: np.ndarray, styles: list[tuple[str, str]], size: int, steps: int, seed: int
) -> tuple[list[dict], str]:
    """スタイルごとに 1 枚ずつ生成。返り値 (バリエーションのリスト, 使用エンジン名)。

    各要素: {"label", "image"(RGB), "origin", "prompt"}。
    SD-Turbo は steps=1〜2 / guidance=0.0 を厳守（蒸留モデルの作法）。
    """
    pipe = gl.load_t2i_pipeline()  # 失敗時 None（ガード済み）
    results: list[dict] = []
    if pipe is not None:
        import torch

        for i, (label, style) in enumerate(styles):
            prompt = f"{base_prompt}, {style}"
            with torch.inference_mode():
                img = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,  # 1〜2
                    guidance_scale=0.0,  # ★ Turbo は 0.0 固定
                    height=size,
                    width=size,
                    generator=torch.manual_seed(seed + i),  # スタイルごとに別シードで多様化
                ).images[0]
            results.append({"label": label, "image": gl.to_np(img), "origin": "sd-turbo-t2i", "prompt": prompt})
        return results, "sd-turbo-t2i"

    # フォールバック: ベース商品を色変換してバリエーション化（必ず完走するため）
    for i, (label, style) in enumerate(styles):
        results.append(
            {
                "label": label,
                "image": classical_variation(base_rgb, i),
                "origin": "classical-fallback",
                "prompt": f"{base_prompt}, {style}",
            }
        )
    return results, "classical-fallback"


def score_and_pick_hero(results: list[dict], base_prompt: str) -> str | None:
    """各バリエーションを CLIPScore（ベースプロンプト整合）で採点し、最高点のラベルを返す。

    出品の「主画像（ヒーロー）」をデータドリブンに選ぶための採用ロジック。
    CLIP が無い環境では採点を None にし、先頭をヒーローにする。
    """
    scorer = gl.load_clip_scorer()
    best_label, best_score = (results[0]["label"] if results else None), -1.0
    for r in results:
        s = None if scorer is None else round(scorer(r["image"], base_prompt), 4)
        r["clip_score"] = s
        if s is not None and s > best_score:
            best_score, best_label = s, r["label"]
    return best_label


# ----------------------------------------------------------------------------
# (B) 不要物除去（クリーンアップ）: マスク指定で cv2.inpaint
# ----------------------------------------------------------------------------
def add_unwanted_object(base_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """商品写真にダミーの「値札シール」を貼り、(貼り付け後画像, 除去マスク) を返す。

    自分の写真とマスクを data/ に置けばこの合成は不要（実運用は find_data_mask が優先）。
    """
    h, w = base_rgb.shape[:2]
    before = base_rgb.copy()
    mask = np.zeros((h, w), np.uint8)
    # 左上に黄色いシール＋赤帯＋文字（いかにも消したい異物）
    x0, y0, x1, y1 = int(w * 0.08), int(h * 0.10), int(w * 0.42), int(h * 0.26)
    cv2.rectangle(before, (x0, y0), (x1, y1), (60, 200, 240), -1)
    cv2.rectangle(before, (x0, y0), (x1, y1), (40, 40, 200), 2)
    cv2.putText(
        before, "SALE 50%", (x0 + 6, (y0 + y1) // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 160), 2, cv2.LINE_AA
    )
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # 境界の残渣を巻き込む
    return before, mask


def cleanup_object(base_rgb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """不要物をマスク指定で除去。返り値 (除去前, マスク, 除去後, マスク由来)。"""
    found = find_data_mask(size)
    if found is not None:
        mask, mask_origin = found
        before = base_rgb.copy()
    else:
        before, mask = add_unwanted_object(base_rgb)
        mask_origin = "synthetic-sticker"
    after = cv2.inpaint(before, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return before, mask, after, mask_origin


# ----------------------------------------------------------------------------
# 可視化（matplotlib=Agg・図タイトルは ASCII）と保存
# ----------------------------------------------------------------------------
def save_variations_figure(base_rgb: np.ndarray, results: list[dict], hero: str | None, name: str) -> str:
    """ベース＋各バリエーションを 1 枚のコンタクトシートにし、ヒーローを明示して保存。"""
    panels = [("base", base_rgb, None)] + [(r["label"], r["image"], r.get("clip_score")) for r in results]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (label, img, score) in zip(axes, panels):
        ax.imshow(img)
        ax.axis("off")
        title = label if score is None else f"{label}\nclip {score:.2f}"
        if label == hero:
            title = "[HERO] " + title
        ax.set_title(title, fontsize=9)
    fig.suptitle("EC product variations (pick hero for thumbnail)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(gl.OUT_DIR, name)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_cleanup_figure(before: np.ndarray, mask: np.ndarray, after: np.ndarray, name: str) -> str:
    """不要物除去の before / mask / after を 3 枚並べて保存。"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    axes[0].imshow(before); axes[0].set_title("before (with unwanted object)", fontsize=9); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("removal mask (white=remove)", fontsize=9); axes[1].axis("off")
    axes[2].imshow(after); axes[2].set_title("after cv2.inpaint (TELEA)", fontsize=9); axes[2].axis("off")
    fig.suptitle("Product photo cleanup (object/watermark removal)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(gl.OUT_DIR, name)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EC 商品写真スタジオ: バリエーション生成 + 不要物除去")
    p.add_argument("--mode", choices=["both", "variations", "cleanup"], default="both", help="実行する工程")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="商品説明（ベースプロンプト）")
    p.add_argument("--num", type=int, default=4, help="生成するバリエーション数（1〜6）")
    p.add_argument("--size", type=int, default=320, help="生成サイズ px（256〜384 推奨）")
    p.add_argument("--steps", type=int, default=1, help="SD-Turbo の推論ステップ（1〜2）")
    p.add_argument("--seed", type=int, default=0, help="基準シード")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # CPU 制約を厳守するためのクランプ（はみ出した指定は安全側へ丸める）
    num = max(1, min(args.num, len(STYLES)))
    size = max(256, min(args.size, 384))
    steps = max(1, min(args.steps, 2))
    styles = STYLES[:num]

    gl.ensure_dirs()
    print("=" * 70)
    print(f"EC 商品写真スタジオ  mode={args.mode}  size={size}  steps={steps}")
    base, base_origin = load_base_product(size, args.seed)
    gl.save_rgb(base, "use_case_base.png")
    print(f"  ベース商品: 由来={base_origin}")

    report: dict = {
        "tool": "ec_product_studio",
        "mode": args.mode,
        "base_prompt": args.prompt,
        "base_origin": base_origin,
        "size": size,
        "files": {"base": "use_case_base.png"},
        "notes": "CPU・実データ優先/合成フォールバック完結。SD-Turbo は steps=1-2/guidance=0.0。",
    }

    # (A) バリエーション生成 + ヒーロー選定
    if args.mode in ("both", "variations"):
        print(f"[A] バリエーション生成 x{num} ...")
        results, engine = generate_variations(args.prompt, base, styles, size, steps, args.seed)
        hero = score_and_pick_hero(results, args.prompt)
        var_files = []
        for i, r in enumerate(results):
            fname = f"use_case_var_{i + 1:02d}_{r['label']}.png"
            gl.save_rgb(r["image"], fname)
            var_files.append(fname)
        fig_path = save_variations_figure(base, results, hero, "use_case_variations.png")
        report["variation_engine"] = engine
        report["hero"] = hero
        report["variations"] = [
            {"label": r["label"], "file": var_files[i], "clip_score": r.get("clip_score"), "origin": r["origin"]}
            for i, r in enumerate(results)
        ]
        report["files"]["variations_sheet"] = os.path.basename(fig_path)
        report["files"]["variations"] = var_files
        print(f"      エンジン={engine} / ヒーロー(主画像候補)={hero}")
        for r in results:
            print(f"      - {r['label']:7}: CLIPScore={r.get('clip_score')}")

    # (B) 不要物除去（クリーンアップ）
    if args.mode in ("both", "cleanup"):
        print("[B] 不要物除去（マスク指定 cv2.inpaint）...")
        before, mask, after, mask_origin = cleanup_object(base, size)
        gl.save_rgb(after, "use_case_cleanup_after.png")
        fig_path = save_cleanup_figure(before, mask, after, "use_case_cleanup.png")
        report["cleanup"] = {
            "method": "cv2.inpaint(INPAINT_TELEA)",
            "mask_origin": mask_origin,
            "after_file": "use_case_cleanup_after.png",
        }
        report["files"]["cleanup_figure"] = os.path.basename(fig_path)
        report["files"]["cleanup_after"] = "use_case_cleanup_after.png"
        print(f"      マスク由来={mask_origin} → use_case_cleanup_after.png")

    # 出品キットの目録（JSON）
    path = os.path.join(gl.OUT_DIR, "use_case_listing.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[done] 出品キットを保存: {gl.OUT_DIR}/use_case_*.{{png,json}}")


if __name__ == "__main__":
    main()
