"""03: 深層オプティカルフロー ― torchvision RAFT(small) と 古典法(Farneback/LK)の対比。

オプティカルフロー = 連続2フレーム間で各画素がどこへ動いたかのベクトル場 (u, v)。

学ぶこと（RAFT の3つの勘所）:
  1. 入力は **[-1,1] 正規化**（weights.transforms() がやってくれる）。
  2. 高さ・幅は **8 の倍数**（内部で 1/8 に下げるため。最低 128px 必要）。
  3. 出力は **反復ごとのフローのリスト**で、採用するのは **最後の要素 flows[-1]**。
  そして torchvision.utils.flow_to_image で色相=向き・彩度=大きさの定番可視化にする。

古典法との対比:
  - Farneback(密) と Lucas-Kanade(疎) は CPU だけで動き重み DL も不要。いずれも
    「輝度一定仮定（同じ点は明るさを保つ）」と「滑らかさ」に依存し、大変位や無テクスチャに弱い。
  - 合成の並進ペア（真のフローが一定値で自明）に対し、各手法の EPE を比較する。

実行: uv run python lectures/27_depth_pose_flow/03_raft_optical_flow.py
出力: lectures/27_depth_pose_flow/outputs/03_*.png / 03_flow_metrics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import dpf_helpers as H  # noqa: E402


def farneback_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """古典・密フロー（Farneback）。グレースケール2枚から (H,W,2) のフローを返す。"""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    # 既定的なパラメータ。pyr_scale=0.5, levels=3, winsize=15 など定番値。
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow.astype(np.float32)


def lucas_kanade_epe(frame1: np.ndarray, frame2: np.ndarray, gt_flow: np.ndarray) -> float | None:
    """疎フロー（Lucas-Kanade）。特徴点だけ追跡し、その点群での EPE を返す（点が無ければ None）。"""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    pts = cv2.goodFeaturesToTrack(g1, maxCorners=200, qualityLevel=0.01, minDistance=7)
    if pts is None:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None)
    status = status.reshape(-1).astype(bool)
    if not np.any(status):
        return None
    p0 = pts.reshape(-1, 2)[status]
    p1 = nxt.reshape(-1, 2)[status]
    disp = p1 - p0  # 各点の (du, dv)
    # GT は一定並進なので、追跡点の位置の GT フローと比較
    gt_pts = np.stack(
        [gt_flow[int(np.clip(y, 0, gt_flow.shape[0] - 1)),
                 int(np.clip(x, 0, gt_flow.shape[1] - 1))] for x, y in p0]
    )
    err = np.sqrt(np.sum((disp - gt_pts) ** 2, axis=1))
    return float(np.mean(err))


def raft_flow(frame1: np.ndarray, frame2: np.ndarray) -> tuple[np.ndarray | None, str]:
    """RAFT(small) で密フロー。前処理3点（[-1,1]/8の倍数/最後の反復）を踏む。失敗時 None。"""
    try:
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        torch.set_num_threads(4)
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False).eval()
        transforms = weights.transforms()

        h, w = frame1.shape[:2]
        # 8 の倍数かつ最低 128px へ。ここでは生成側で 128x192 にしてあるのでそのまま使える。
        Ht = max(H.round_up_to_multiple(h, 8), 128)
        Wt = max(H.round_up_to_multiple(w, 8), 128)
        a = cv2.resize(frame1, (Wt, Ht))
        b = cv2.resize(frame2, (Wt, Ht))

        def to_tensor(x: np.ndarray) -> "torch.Tensor":
            return torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        ta, tb = transforms(to_tensor(a), to_tensor(b))  # ここで [-1,1] 正規化される
        with torch.inference_mode():
            flows = model(ta, tb)  # 反復ごとのリスト
        flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()  # 最後を採用 → (H,W,2)

        # 入力をリサイズしていた場合はフローを元解像度へ戻し、ベクトルも倍率補正
        if (Ht, Wt) != (h, w):
            flow = cv2.resize(flow, (w, h))
            flow[..., 0] *= w / Wt
            flow[..., 1] *= h / Ht
        return flow.astype(np.float32), "raft_small"
    except Exception as e:  # noqa: BLE001
        print(f"[警告] RAFT 実行に失敗（{type(e).__name__}: {e}）。古典法のみで継続。")
        return None, "fallback(skip)"


def save_flow_vis(flow: np.ndarray, path: pathlib.Path) -> None:
    """torchvision.utils.flow_to_image で色可視化して保存（無ければ HSV 自前可視化）。"""
    try:
        import torch
        from torchvision.utils import flow_to_image

        t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
        img = flow_to_image(t)[0].permute(1, 2, 0).numpy().astype(np.uint8)
    except Exception:  # noqa: BLE001
        # HSV: 色相=向き, 明度=大きさ
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    Image.fromarray(img).save(path)


def main() -> None:
    out = H.ensure_output_dir()
    H.set_seed(1)

    dx, dy = 6, 3
    frame1, frame2, gt_flow = H.make_translating_pair(dx=dx, dy=dy)
    Image.fromarray(frame1).save(out / "03_frame1.png")
    Image.fromarray(frame2).save(out / "03_frame2.png")
    print("=== 03: オプティカルフロー（RAFT vs 古典）===")
    print(f"合成: 全画素が一定並進 (dx,dy)=({dx},{dy}) の画像対。真のフローは自明。")
    save_flow_vis(gt_flow, out / "03_flow_gt.png")

    metrics: dict = {"gt_translation": [dx, dy]}

    # --- RAFT(small) ---
    flow_raft, src = raft_flow(frame1, frame2)
    if flow_raft is not None:
        epe = H.endpoint_error(flow_raft, gt_flow)
        metrics["raft_epe"] = epe
        save_flow_vis(flow_raft, out / "03_flow_raft.png")
        med = np.median(flow_raft.reshape(-1, 2), axis=0)
        print(f"[RAFT] EPE={epe:.3f}px  推定中央値フロー≈({med[0]:.2f},{med[1]:.2f})  "
              f"saved: {out/'03_flow_raft.png'}")
    else:
        print("[RAFT] スキップ（重み DL 不可）。")

    # --- Farneback(密・古典) ---
    flow_fb = farneback_flow(frame1, frame2)
    epe_fb = H.endpoint_error(flow_fb, gt_flow)
    metrics["farneback_epe"] = epe_fb
    save_flow_vis(flow_fb, out / "03_flow_farneback.png")
    print(f"[Farneback] EPE={epe_fb:.3f}px  saved: {out/'03_flow_farneback.png'}")

    # --- Lucas-Kanade(疎・古典) ---
    epe_lk = lucas_kanade_epe(frame1, frame2, gt_flow)
    metrics["lk_epe"] = epe_lk
    if epe_lk is not None:
        print(f"[Lucas-Kanade] 追跡点での EPE={epe_lk:.3f}px（疎: 特徴点のみ）")

    print("\n[考察] 一定並進かつ十分なテクスチャがあるこの易しい設定では古典法でも当たりやすい。"
          "大変位・無テクスチャ・遮蔽が入ると RAFT の優位が出る（次の発展で実験）。")

    (out / "03_flow_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {out/'03_flow_metrics.json'}")


if __name__ == "__main__":
    main()
