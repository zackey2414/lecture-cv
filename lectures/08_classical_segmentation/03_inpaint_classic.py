"""03: 古典 inpaint（cv2.inpaint）で傷消し・物体除去を行い、PSNR/SSIM で評価する。

学ぶこと:
  - cv2.inpaint(src, mask, radius, flags) は、mask>0 の領域を、その「周囲の画素」
    から滑らかに埋め戻す古典復元。flags は 2 種:
      INPAINT_TELEA: Fast Marching 法。境界から距離の近い順に内側へ埋める。速い。
      INPAINT_NS   : Navier-Stokes 法。等照度線(isophote)を伸ばすように埋める。
  - 「傷・細い線・小さな文字(透かし)」のような細い欠損には古典 inpaint が強い
    （周囲から色を引いて来れば十分だから）。
  - 一方「大きな穴・構造のある領域」は苦手（周囲の色を引き伸ばすだけなので
    のっぺりボケる）。この限界を体感し、深層インペイント LaMa(29回) への動機にする。
  - 評価は合成した「元のキレイな画像」との比較で:
      PSNR（大きいほど良い・dB）と SSIM（1.0 が完全一致・構造の近さ）。
    どちらも numpy/cv2 だけで自前実装する（skimage 等に依存しない）。

実行:
  uv run python lectures/08_classical_segmentation/03_inpaint_classic.py
結果は lectures/08_classical_segmentation/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "08_classical_segmentation"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def panel(img: np.ndarray, title: str, size: tuple[int, int] = (340, 270)) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w, h = size
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# =====================================================================
# 評価指標（PSNR / SSIM を numpy/cv2 だけで実装）
# =====================================================================

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """ピーク信号対雑音比[dB]。大きいほど元画像に近い。"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """構造的類似度（グレースケール、Wang らのガウス窓版）。1.0 で完全一致。"""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64) if a.ndim == 3 else a.astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64) if b.ndim == 3 else b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    var_b = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    cov = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    num = (2 * mu_ab + c1) * (2 * cov + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


# =====================================================================
# サンプル生成（キレイな元画像 + 2種類の欠損）
# =====================================================================

def make_clean() -> np.ndarray:
    """グラデ背景に図形と文字を置いた、復元の「正解」になる BGR 画像。"""
    h, w = 300, 400
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (200 - xx / w * 120).astype(np.uint8)
    img[..., 1] = (60 + yy / h * 150).astype(np.uint8)
    img[..., 2] = (90 + xx / w * 120).astype(np.uint8)
    cv2.circle(img, (110, 120), 55, (40, 180, 240), -1)
    cv2.rectangle(img, (220, 70), (340, 200), (230, 120, 60), -1)
    cv2.putText(img, "PHOTO", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                cv2.LINE_AA)
    return img


def make_scratch_mask(shape: tuple[int, int]) -> np.ndarray:
    """細い傷 + 透かし文字の欠損マスク(0/255)。古典 inpaint が得意な細い欠損。"""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(11)
    for _ in range(6):  # ランダムな引っかき傷（折れ線）
        pts = rng.integers([0, 0], [w, h], size=(4, 2))
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, thickness=2)
    cv2.putText(mask, "(C)2026", (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3, cv2.LINE_AA)
    return mask


def make_hole_mask(shape: tuple[int, int]) -> np.ndarray:
    """大きな矩形の穴マスク。構造を含む大穴 → 古典 inpaint が苦手なケース。"""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (210, 60), (350, 210), 255, -1)  # 四角い領域をまるごと欠損に
    return mask


def damage(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """マスク領域を白(=傷/欠損)で塗りつぶした「破損画像」を作る。"""
    out = clean.copy()
    out[mask > 0] = (255, 255, 255)
    return out


def restore_and_score(clean: np.ndarray, mask: np.ndarray, tag: str) -> dict:
    """破損→TELEA/NS で復元→PSNR/SSIM を計算し、結果一式を返す。"""
    broken = damage(clean, mask)
    telea = cv2.inpaint(broken, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    ns = cv2.inpaint(broken, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return {
        "tag": tag,
        "broken": broken,
        "telea": telea,
        "ns": ns,
        "psnr_broken": psnr(clean, broken),
        "psnr_telea": psnr(clean, telea),
        "psnr_ns": psnr(clean, ns),
        "ssim_telea": ssim(clean, telea),
        "ssim_ns": ssim(clean, ns),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    clean = make_clean()
    shape = clean.shape[:2]

    # --- ケースA: 細い傷+透かし（古典が得意）-----------------------------
    scratch = make_scratch_mask(shape)
    a = restore_and_score(clean, scratch, "scratch")

    # --- ケースB: 大きな穴（古典が苦手）---------------------------------
    hole = make_hole_mask(shape)
    b = restore_and_score(clean, hole, "big-hole")

    # --- 可視化 ---------------------------------------------------------
    sheet = grid(
        [
            panel(clean, "clean (ground truth)"),
            panel(a["broken"], "A: scratched + watermark"),
            panel(a["telea"], f"A: TELEA  PSNR={a['psnr_telea']:.1f}"),
            panel(a["ns"], f"A: NS  SSIM={a['ssim_ns']:.3f}"),
            panel(b["broken"], "B: big hole (removed)"),
            panel(b["telea"], f"B: TELEA  PSNR={b['psnr_telea']:.1f}"),
        ],
        3,
    )
    cv2.imwrite(os.path.join(OUT_DIR, "03_inpaint_pipeline.png"), sheet)
    cv2.imwrite(os.path.join(OUT_DIR, "03_inpaint_scratch_telea.png"), a["telea"])
    cv2.imwrite(os.path.join(OUT_DIR, "03_inpaint_mask_scratch.png"), scratch)

    # --- PSNR 比較の棒グラフ（破損 → TELEA/NS でどれだけ回復したか）------
    cases = ["scratch\n(thin)", "big-hole\n(large)"]
    broken_p = [a["psnr_broken"], b["psnr_broken"]]
    telea_p = [a["psnr_telea"], b["psnr_telea"]]
    ns_p = [a["psnr_ns"], b["psnr_ns"]]
    pos = np.arange(len(cases))
    plt.figure(figsize=(6.4, 3.8))
    plt.bar(pos - 0.25, broken_p, width=0.25, label="broken")
    plt.bar(pos, telea_p, width=0.25, label="TELEA")
    plt.bar(pos + 0.25, ns_p, width=0.25, label="NS")
    plt.xticks(pos, cases)
    plt.ylabel("PSNR [dB] (higher = closer to original)")
    plt.title("cv2.inpaint: thin scratch recovers well, big hole poorly")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_inpaint_psnr.png"), dpi=110)
    plt.close()

    # --- ログ -----------------------------------------------------------
    print("=== 古典 inpaint（TELEA / NS）と PSNR/SSIM 評価 ===")
    for case in (a, b):
        print(f"[{case['tag']}]")
        print(f"  破損時 PSNR        : {case['psnr_broken']:.2f} dB")
        print(f"  TELEA PSNR / SSIM  : {case['psnr_telea']:.2f} dB / {case['ssim_telea']:.3f}")
        print(f"  NS    PSNR / SSIM  : {case['psnr_ns']:.2f} dB / {case['ssim_ns']:.3f}")
    print("\n  → 細い傷は PSNR が大きく回復、大穴は周囲色の引き伸ばしでボケて回復が小さい。")
    print("  これが古典 inpaint の限界で、構造を補完したいときは深層 LaMa(29回) に進む。")
    print("\n完了。lectures/08_classical_segmentation/outputs/ の 03_*.png を確認してください。")


if __name__ == "__main__":
    main()
