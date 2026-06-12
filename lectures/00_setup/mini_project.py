"""章末ミニプロジェクト: 『環境まるごと検証』。

この回で学んだ要素を 1 本に統合し、「本講座を走らせる環境が整っているか」を
自動で総点検する完成形。各チェックは PASS/FAIL を返し、最後に総合判定・
JSON レポート・サマリ図を outputs/00_setup/ に出力する。

検証する 6 項目:
  (1) 必須ライブラリの導入と版（numpy/cv2/PIL/matplotlib）
  (2) device 自動判定（cpu/mps/cuda）とスレッド固定
  (3) torch の実計算 sanity（選んだ device 上で誤差なく計算できるか）
  (4) 画像パイプライン往復（cv2 BGR → RGB → PIL → numpy → cv2 が一致するか）
  (5) headless 保存（matplotlib=Agg と cv2.imwrite が画面なしで動くか）
  (6) HuggingFace キャッシュ場所(HF_HOME)の確認

実行:
    uv run python lectures/00_setup/mini_project.py
成果物（JSON・PNG）は outputs/00_setup/ に保存される。CPUのみ・合成データ・ネット不要。
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from device import configure_threads, hf_home, pick_device, summary  # noqa: E402


def _output_dir() -> pathlib.Path:
    d = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "00_setup"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =====================================================================
# 個々のチェック（それぞれ (ok: bool, detail: str) を返す単一責務）
# =====================================================================

def check_required_libs() -> tuple[bool, str]:
    """必須 4 ライブラリが import でき、版が取れるか。"""
    import matplotlib

    vers = {
        "numpy": np.__version__,
        "cv2": cv2.__version__,
        "PIL": Image.__version__,
        "matplotlib": matplotlib.__version__,
    }
    detail = " ".join(f"{k}={v}" for k, v in vers.items())
    return True, detail


def check_device() -> tuple[bool, str]:
    """device 自動判定とスレッド固定が機能するか。"""
    n = configure_threads()
    dev = pick_device()
    ok = str(dev) in {"cpu", "cuda", "mps"}
    return ok, f"device={dev} threads={n}"


def check_torch_compute(device: torch.device) -> tuple[bool, str]:
    """選んだ device 上で行列積が数学的に正しく計算できるか（sanity）。"""
    a = torch.eye(8, device=device)
    b = torch.randn(8, 8, device=device)
    out = (a @ b).cpu()             # 単位行列 × b == b になるはず
    err = float((out - b.cpu()).abs().max())
    ok = err < 1e-5
    return ok, f"max_abs_err={err:.2e} on {device}"


def check_image_roundtrip() -> tuple[bool, str]:
    """cv2(BGR) → RGB → PIL → numpy → cv2(BGR) と 1 周して画素が一致するか。"""
    bgr = _make_sample_bgr()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)        # 外の世界(RGB)へ
    pil = Image.fromarray(rgb)                         # numpy → PIL
    back = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)  # PIL → numpy → cv2(BGR)
    ok = bool(np.array_equal(bgr, back))
    return ok, f"roundtrip_equal={ok} shape={bgr.shape}"


def check_headless_save(out: pathlib.Path) -> tuple[bool, str]:
    """画面なし(headless)でも matplotlib(Agg) と cv2.imwrite で保存できるか。"""
    import matplotlib

    matplotlib.use("Agg")  # DISPLAY が無くても savefig が動く
    import matplotlib.pyplot as plt

    bgr = _make_sample_bgr()
    cv_path = out / "mp_opencv_save.png"
    cv2.imwrite(str(cv_path), bgr)                     # 方法A: cv2(BGR のまま)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(rgb)                                     # 方法B: matplotlib(RGB に変換してから)
    ax.set_title("headless save OK")
    ax.axis("off")
    plt_path = out / "mp_matplotlib_save.png"
    fig.savefig(plt_path, dpi=90)
    plt.close(fig)

    ok = cv_path.exists() and plt_path.exists()
    return ok, f"saved {cv_path.name}, {plt_path.name}"


def check_hf_cache() -> tuple[bool, str]:
    """HF_HOME の場所が解決できるか（存在は問わない＝初回 DL 時に作られる）。"""
    home = hf_home()
    return True, f"HF_HOME={home} exists={home.exists()}"


# =====================================================================
# 補助
# =====================================================================

def _make_sample_bgr(h: int = 120, w: int = 160) -> np.ndarray:
    """合成 BGR 画像（サンプル不要で完走するため）。原色を四隅に置く。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, : w // 2] = (0, 0, 255)     # 左上=赤(BGR)
    img[: h // 2, w // 2 :] = (0, 255, 0)     # 右上=緑
    img[h // 2 :, : w // 2] = (255, 0, 0)     # 左下=青
    img[h // 2 :, w // 2 :] = (0, 255, 255)   # 右下=黄
    return img


def _save_report_chart(out: pathlib.Path, rows: list[tuple[str, bool, str]]) -> None:
    """各チェックの PASS/FAIL を 1 枚の図にまとめる（headless: Agg）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r[0] for r in rows]
    vals = [1 if r[1] else 0 for r in rows]
    colors = ["tab:green" if v else "tab:red" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.set_title("environment verification (PASS=green / FAIL=red)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "mini_project.png", dpi=110)
    plt.close(fig)


def main() -> None:
    out = _output_dir()
    device = pick_device()
    print("=" * 60)
    print("ミニプロジェクト: 環境まるごと検証")
    print("=" * 60)

    rows: list[tuple[str, bool, str]] = []

    def run(name: str, fn) -> None:
        """1 チェックを実行し、例外でも落とさず FAIL として記録する。"""
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - 検証ツール自体は止めない
            ok, detail = False, f"例外: {type(e).__name__}: {e}"
        rows.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} {detail}")

    run("required_libs", check_required_libs)
    run("device", check_device)
    run("torch_compute", lambda: check_torch_compute(device))
    run("image_roundtrip", check_image_roundtrip)
    run("headless_save", lambda: check_headless_save(out))
    run("hf_cache", check_hf_cache)

    all_ok = all(ok for _, ok, _ in rows)

    # --- レポート保存 -------------------------------------------------
    report = {
        "all_pass": all_ok,
        "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in rows],
        "environment": summary(),
    }
    (out / "mini_project_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_report_chart(out, rows)

    print("\n保存: mini_project_report.json / mini_project.png")
    if all_ok:
        print("\n総合判定: ALL PASS 🎉  本講座を走らせる環境が整っている。")
    else:
        print("\n総合判定: 未達あり。FAIL の detail を見て README の対処法を参照。")


if __name__ == "__main__":
    main()
