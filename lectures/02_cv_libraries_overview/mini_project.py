"""mini_project.py — 章末ミニプロジェクト: ライブラリ選定レポート生成器。

この回（ライブラリの地図）の学びを 1 本に束ねる総合課題の「完成形」です。
合成画像 1 枚を入力に、次の 4 つを実測してレポート（JSON + 図）にまとめます。

  (1) 同一の「正準リサイズ」を OpenCV / Pillow / scikit-image(任意) で実行し、
      処理時間と、OpenCV を基準にした数値差(MAD: 平均絶対差) を測る。
      → 「同じ処理でもライブラリで速度も画素値も違う」を数値で確認（02 の発展）。
  (2) cv2(BGR) → PIL(RGB) → numpy → cv2(BGR) の往復、および
      cv2 → Tensor(C,H,W) → cv2 の往復(torch があれば) が無損失かを検証する。
      → 相互運用で最重要の「色順」「軸順」を取り違えていないかの自動チェック（地図の核）。
  (3) 自作拡張パイプラインを多数回かけ、明るさ分布がどれだけ広がるかを定量化する。
      → 拡張がデータの多様性を増やす様子を std/range で数値化（03 の評価ポイント）。
  (4) 「課題 → まず選ぶライブラリ」の意思決定表が自己整合かを点検する。

出力（すべて outputs/02_cv_libraries_overview/ へ。画面表示なし・ネット不要・CPU 完結）:
  - mini_project_report.json   … 機械可読な集計（時間/MAD/往復/分布/環境）
  - mini_project_summary.png   … タイミング棒グラフ ＋ 明るさ分布ヒストグラム

実行:
  uv run python lectures/02_cv_libraries_overview/mini_project.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageFilter

import matplotlib

matplotlib.use("Agg")  # pyplot より前に Agg 固定（headless 対応）
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir, probe, to_rgb  # noqa: E402


# =====================================================================
# (1) 同一の「正準リサイズ」をライブラリ横断で実行し、時間と数値差を測る
# =====================================================================

def _time_call(fn, repeats: int = 30) -> float:
    """fn() を repeats 回呼び、1 回あたりの最小実行時間[ミリ秒]を返す。

    最小値を採るのは、GC や OS のノイズに左右されにくいから（ベンチの定石）。
    """
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def resize_opencv(bgr: np.ndarray, dst_hw: tuple[int, int]) -> np.ndarray:
    """OpenCV で縮小（基準）。dsize は (幅, 高さ) 順、縮小は INTER_AREA が定石。"""
    h, w = dst_hw
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return to_rgb(resized)  # 比較のため RGB(uint8) に揃える


def resize_pillow(bgr: np.ndarray, dst_hw: tuple[int, int]) -> np.ndarray:
    """Pillow で縮小。size も (幅, 高さ) 順。Pillow 12 は LANCZOS を明示する。"""
    h, w = dst_hw
    pil = Image.fromarray(to_rgb(bgr))            # RGB に直してから PIL 化
    resized = pil.resize((w, h), Image.Resampling.LANCZOS)
    return np.asarray(resized)


def resize_skimage(bgr: np.ndarray, dst_hw: tuple[int, int], skimage_mod) -> np.ndarray:
    """scikit-image で縮小。出力サイズは (高さ, 幅) 順（cv2/PIL と逆！）・値域 float[0,1]。"""
    transform = skimage_mod.transform
    util = skimage_mod.util
    f = util.img_as_float(to_rgb(bgr))            # uint8 → float[0,1]
    resized = transform.resize(f, dst_hw, anti_aliasing=True)  # (H, W) 順
    return util.img_as_ubyte(np.clip(resized, 0, 1))


def benchmark_resize(bgr: np.ndarray) -> dict:
    """各ライブラリの正準リサイズを時間計測し、OpenCV 基準の MAD を求める。"""
    h, w = bgr.shape[:2]
    dst_hw = (h // 2, w // 2)                      # 半分に縮小（(高さ, 幅) で持つ）

    base = resize_opencv(bgr, dst_hw)              # OpenCV を基準画像にする
    rows: list[dict] = []

    # OpenCV（基準）
    rows.append({
        "library": "OpenCV",
        "time_ms": round(_time_call(lambda: resize_opencv(bgr, dst_hw)), 3),
        "mad_vs_opencv": 0.0,
    })
    # Pillow
    pil_out = resize_pillow(bgr, dst_hw)
    rows.append({
        "library": "Pillow",
        "time_ms": round(_time_call(lambda: resize_pillow(bgr, dst_hw)), 3),
        "mad_vs_opencv": round(_mad(base, pil_out), 3),
    })
    # scikit-image（任意）
    sk = probe("scikit-image", "skimage", "uv add --group aug scikit-image")
    if sk.available:
        sk_out = resize_skimage(bgr, dst_hw, sk.module)
        rows.append({
            "library": "scikit-image",
            "time_ms": round(_time_call(lambda: resize_skimage(bgr, dst_hw, sk.module)), 3),
            "mad_vs_opencv": round(_mad(base, sk_out), 3),
        })
    return {"dst_hw": list(dst_hw), "rows": rows}


def _mad(a: np.ndarray, b: np.ndarray) -> float:
    """平均絶対差（0-255 スケール）。shape が違えば NaN を返す（比較不能の印）。"""
    if a.shape != b.shape:
        return float("nan")
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


# =====================================================================
# (2) 相互変換が無損失かを検証する（色順・軸順の自動チェック）
# =====================================================================

def roundtrip_cv_pil_cv(bgr: np.ndarray) -> bool:
    """cv2(BGR) → PIL(RGB) → numpy → cv2(BGR) の往復で元に戻るか。

    境界をまたぐたびに BGR↔RGB を正しく入れれば、画素は完全一致するはず。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)    # 行きで RGB へ
    pil = Image.fromarray(rgb)
    back_rgb = np.asarray(pil)
    back_bgr = cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR)  # 帰りで BGR へ戻す
    return bool(np.array_equal(bgr, back_bgr))


def roundtrip_cv_tensor_cv(bgr: np.ndarray, torch_mod) -> bool:
    """cv2(BGR,HWC) → Tensor(CHW,RGB) → cv2(BGR,HWC) の往復で元に戻るか。

    Tensor は (C, H, W) 軸順なので permute が要る。色順(BGR↔RGB)も忘れない。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    chw = torch_mod.from_numpy(rgb).permute(2, 0, 1)      # (H,W,C) → (C,H,W)
    back_rgb = chw.permute(1, 2, 0).numpy()               # (C,H,W) → (H,W,C)
    back_bgr = cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR)
    return bool(np.array_equal(bgr, back_bgr))


def check_interop(bgr: np.ndarray) -> dict:
    """相互変換の往復チェックをまとめて行い、結果の辞書を返す。"""
    result = {"cv2<->PIL<->cv2": roundtrip_cv_pil_cv(bgr)}
    tp = probe("torch", "torch", "uv add --group dl torch")
    if tp.available:
        result["cv2<->Tensor<->cv2"] = roundtrip_cv_tensor_cv(bgr, tp.module)
    else:
        result["cv2<->Tensor<->cv2"] = None  # torch 未導入はスキップ（None）
    return result


# =====================================================================
# (3) 自作拡張パイプラインで明るさ分布の広がりを定量化する
# =====================================================================

def augment_once(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """確率的な小さな拡張（反転・明るさ/コントラスト・回転）を 1 回かける。

    どのライブラリの拡張も、内部はこういう「確率的な画素操作の連なり」にすぎない。
    """
    out = rgb
    if rng.random() < 0.5:
        out = out[:, ::-1].copy()                          # 左右反転
    alpha = rng.uniform(0.7, 1.3)                          # コントラスト
    beta = rng.uniform(-40, 40)                            # 明るさ
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), 1.0)
    return cv2.warpAffine(out, mat, (w, h), borderMode=cv2.BORDER_REFLECT)


def measure_spread(rgb: np.ndarray, rng: np.random.Generator, n: int = 300) -> dict:
    """拡張を n 回かけ、平均明るさの分布（中心・広がり・範囲）を返す。"""
    means = [float(augment_once(rgb, rng).mean()) for _ in range(n)]
    arr = np.asarray(means)
    return {
        "samples": n,
        "original_mean": round(float(rgb.mean()), 3),
        "aug_mean": round(float(arr.mean()), 3),
        "aug_std": round(float(arr.std()), 3),       # std>0 = 多様性が増えた証拠
        "aug_min": round(float(arr.min()), 3),
        "aug_max": round(float(arr.max()), 3),
        "_means": means,                             # 図用（JSON には出さない）
    }


# =====================================================================
# (4) 意思決定表（課題 → まず選ぶライブラリ）の自己整合チェック
# =====================================================================

DECISION_TABLE = {
    "touch_pixels": "NumPy",
    "fast_classic_cv": "OpenCV",
    "intuitive_edit": "Pillow",
    "scientific_repro": "scikit-image",
    "bbox_mask_aug": "albumentations",
    "torch_training_aug": "torchvision",
    "differentiable_gpu_aug": "kornia",
    "read_video": "OpenCV",
    "headless_view": "matplotlib",
}


def pick_library(task: str) -> str:
    """課題キーワードから「まず選ぶライブラリ」を返す（未知キーは OpenCV）。"""
    return DECISION_TABLE.get(task, "OpenCV")


def check_decision_table() -> dict:
    """意思決定表が引けること・未知キーが OpenCV に落ちることを点検する。"""
    all_known_ok = all(pick_library(k) == v for k, v in DECISION_TABLE.items())
    unknown_ok = pick_library("totally_unknown_task") == "OpenCV"
    return {"entries": len(DECISION_TABLE),
            "all_known_ok": bool(all_known_ok),
            "unknown_defaults_opencv": bool(unknown_ok)}


# =====================================================================
# 図とレポートの保存
# =====================================================================

def save_summary_figure(out: pathlib.Path, bench: dict, spread: dict) -> pathlib.Path:
    """タイミング棒グラフ ＋ 明るさ分布ヒストグラムの 2 枚組を保存する。"""
    fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(12, 4.6))

    # 左: ライブラリ別リサイズ時間（ms）。棒の上に MAD を注記する。
    libs = [r["library"] for r in bench["rows"]]
    times = [r["time_ms"] for r in bench["rows"]]
    mads = [r["mad_vs_opencv"] for r in bench["rows"]]
    bars = ax_t.bar(libs, times, color=["#1565c0", "#ef6c00", "#2e7d32"][: len(libs)])
    for bar, mad in zip(bars, mads):
        ax_t.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"MAD={mad:.1f}", ha="center", va="bottom", fontsize=9)
    ax_t.set_ylabel("resize time [ms] (lower is faster)")
    ax_t.set_title("Canonical resize: speed & pixel diff (MAD vs OpenCV)")

    # 右: 拡張後の平均明るさ分布。元画像は縦線で示す。
    ax_h.hist(spread["_means"], bins=30, color="#ef6c00", alpha=0.8,
              label=f"augmented (n={spread['samples']})")
    ax_h.axvline(spread["original_mean"], color="#1565c0", linewidth=2.5, label="original")
    ax_h.set_xlabel("mean brightness (0-255)")
    ax_h.set_ylabel("count")
    ax_h.set_title(f"Augmentation spread (std={spread['aug_std']})")
    ax_h.legend()

    fig.suptitle("CV libraries mini-project: same-op / interop / augmentation")
    fig.tight_layout()
    path = out / "mini_project_summary.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def environment_snapshot() -> dict:
    """主要ライブラリの導入状況をスナップショットにする（レポートに残す）。"""
    libs = [
        ("OpenCV", "cv2"), ("Pillow", "PIL"), ("NumPy", "numpy"),
        ("matplotlib", "matplotlib"), ("scikit-image", "skimage"),
        ("torch", "torch"), ("torchvision", "torchvision"),
        ("albumentations", "albumentations"), ("kornia", "kornia"),
    ]
    snap = {}
    for name, imp in libs:
        p = probe(name, imp, "")
        snap[name] = p.version if p.available else None
    return snap


def main() -> None:
    out = output_dir()
    rng = np.random.default_rng(42)
    bgr = get_sample_bgr()
    print("入力画像 shape:", bgr.shape, "(H, W, C) ※ OpenCV は BGR")

    # (1) 書き比べベンチ
    print("\n=== (1) 同一リサイズの速度 & 数値差（MAD vs OpenCV）===")
    bench = benchmark_resize(bgr)
    for r in bench["rows"]:
        print(f"  {r['library']:13s}: {r['time_ms']:7.3f} ms / MAD={r['mad_vs_opencv']}")

    # (2) 相互変換の往復チェック
    print("\n=== (2) 相互変換は無損失か（色順・軸順チェック）===")
    interop = check_interop(bgr)
    for k, v in interop.items():
        state = "OK(無損失)" if v is True else ("FAIL(壊れた)" if v is False else "skip(未導入)")
        print(f"  {k:22s}: {state}")

    # (3) 拡張の分布の広がり
    print("\n=== (3) 拡張は分布をどれだけ広げるか ===")
    spread = measure_spread(bgr, rng)
    print(f"  元の平均明るさ={spread['original_mean']} → "
          f"拡張後 mean={spread['aug_mean']} std={spread['aug_std']} "
          f"range=[{spread['aug_min']}, {spread['aug_max']}]")

    # (4) 意思決定表の自己整合
    print("\n=== (4) 意思決定表の点検 ===")
    decision = check_decision_table()
    print(f"  既知 {decision['entries']} 件すべて整合={decision['all_known_ok']} / "
          f"未知キーは OpenCV={decision['unknown_defaults_opencv']}")

    # 図 & JSON 保存（_means は図にだけ使い、JSON からは外す）
    print("\n=== 図とレポートを保存 ===")
    fig_path = save_summary_figure(out, bench, spread)
    print("  保存:", fig_path.name)

    spread_for_json = {k: v for k, v in spread.items() if k != "_means"}
    report = {
        "module": "02_cv_libraries_overview",
        "input_shape": list(bgr.shape),
        "resize_benchmark": bench,
        "interop_roundtrip": interop,
        "augmentation_spread": spread_for_json,
        "decision_table": decision,
        "environment": environment_snapshot(),
    }
    json_path = out / "mini_project_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  保存:", json_path.name)

    print("\n完了。outputs/02_cv_libraries_overview/ の "
          "mini_project_summary.png と mini_project_report.json を確認してください。")


if __name__ == "__main__":
    main()
