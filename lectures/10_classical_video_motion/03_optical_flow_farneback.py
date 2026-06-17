"""03: Farneback 密オプティカルフロー — 全画素の動きを推定して HSV で可視化する。

学ぶこと:
  - calcOpticalFlowFarneback で『全画素』の移動ベクトル (u, v) を一気に求める。
  - フロー場を HSV 配色（色相=向き / 明度=大きさ）で1枚の画像にする定番可視化。
  - 疎フロー(LK)との違い: LK は選んだ点だけ、Farneback は画面全部。密だが重い。
  - 終点誤差 EPE（End-Point Error）の概念: 既知の平行移動ペアで答え合わせする。
    （EPE の定量比較そのものは深層フロー RAFT を扱う第27回で本格的に行う予習。）

実行:
  uv run python lectures/10_classical_video_motion/03_optical_flow_farneback.py
結果は lectures/10_classical_video_motion/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import matplotlib

matplotlib.use("Agg")  # headless 環境で確実に保存できるよう、GUI 無しの Agg を使う
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    endpoint_error,
    flow_to_bgr,
    make_motion_frames,
    make_translating_pair,
    output_dir,
)

# Farneback のパラメータ（OpenCV チュートリアルの定番値）。
# pyr_scale=0.5（各段で半分に縮小）, levels=3, winsize=15, iterations=3,
# poly_n=5, poly_sigma=1.2, flags=0。
FB_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)


def farneback(prev_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
    """2フレームから密フロー (H, W, 2) を計算する（グレースケール入力が必要）。"""
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, **FB_PARAMS)


def demo_dense_on_scene(frames: list[np.ndarray], out: pathlib.Path) -> None:
    """動く図形のシーンで密フローを計算し、HSV 可視化と矢印を保存する。"""
    prev, nxt = frames[18], frames[19]
    flow = farneback(prev, nxt)
    mag = np.linalg.norm(flow, axis=2)
    print("[1] シーン（動く図形）の密フロー")
    print(f"  フロー shape = {flow.shape}（最後の2 = 各画素の u,v）")
    print(f"  動きの大きさ: 最大 {mag.max():.2f}px / 平均 {mag.mean():.2f}px")
    print("  → 物体の所だけ大きさが立ち、背景は 0 付近（＝動いていない）。")

    vis = flow_to_bgr(flow)
    cv2.imwrite(str(out / "03_farneback_hsv.png"), vis)
    cv2.imwrite(str(out / "03_farneback_frame.png"), nxt)

    # まばらに矢印を描いて『どっち向きにどれだけ』を直接見せる（quiver 風）。
    arrows = nxt.copy()
    step = 16
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            if fx * fx + fy * fy < 1.0:  # 動きが小さい所は描かない（見やすさ優先）
                continue
            cv2.arrowedLine(arrows, (x, y), (int(x + fx), int(y + fy)),
                            (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.4)
    cv2.imwrite(str(out / "03_farneback_arrows.png"), arrows)


def demo_epe_on_known_shift(out: pathlib.Path) -> None:
    """既知の平行移動ペアで EPE（終点誤差）を計算し、答え合わせする。"""
    shift = (6, 3)
    a, b, true_shift = make_translating_pair(shift=shift)
    flow = farneback(a, b)

    # 真のフロー: 全画素が (dx, dy) だけ動いている、と分かっている。
    gt = np.zeros_like(flow)
    gt[..., 0], gt[..., 1] = true_shift
    epe = endpoint_error(flow, gt)

    print("\n[2] 既知の平行移動で EPE（終点誤差）を測る")
    print(f"  真のシフト = {true_shift} px（全画素共通）")
    print(f"  EPE 平均 = {epe.mean():.3f}px / 中央値 = {np.median(epe):.3f}px（0 に近いほど良い）")
    print("  → 合成なので真値が分かり、推定フローがほぼ一致することを数値で確認できる。")

    # 入力ペア・EPE マップを matplotlib で1枚に（BGR→RGB を忘れずに）。
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    axes[0].set_title("frame A")
    axes[1].imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
    axes[1].set_title("frame B (shifted)")
    im = axes[2].imshow(epe, cmap="magma")
    axes[2].set_title(f"EPE map (mean={epe.mean():.2f}px)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "03_farneback_epe.png"), dpi=110)
    plt.close(fig)


def demo_color_wheel(out: pathlib.Path) -> None:
    """HSV 可視化の凡例（カラーホイール）を作る。どの色がどの向きかを示す。"""
    size = 160
    yy, xx = np.mgrid[-size:size, -size:size].astype(np.float32)
    wheel_flow = np.stack([xx, yy], axis=-1)  # 中心から外向きの放射フロー
    legend = flow_to_bgr(wheel_flow)
    cv2.putText(legend, "right", (2 * size - 48, size), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, "down", (size - 20, 2 * size - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "03_flow_color_wheel.png"), legend)
    print("\n[3] HSV 可視化の凡例（カラーホイール）を保存")
    print("  色=向き / 明るさ=大きさ。03_flow_color_wheel.png と HSV 図を見比べると読めます。")


def main() -> None:
    out = output_dir()
    frames = make_motion_frames()
    demo_dense_on_scene(frames, out)
    demo_epe_on_known_shift(out)
    demo_color_wheel(out)
    print("\n完了。lectures/10_classical_video_motion/outputs/ の 03_*.png を確認してください。")


if __name__ == "__main__":
    main()
