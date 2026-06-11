"""02: 姿勢/キーポイント推定 ― torchvision keypoint R-CNN で COCO 17 点を取る。

なぜ MediaPipe ではなく keypoint R-CNN か:
  設計メモでは MediaPipe Pose も挙がるが、mediapipe は numpy<2 ピンや protobuf 競合を起こし
  やすく、本講座の numpy2 系環境と衝突する。そこで **実行経路は torchvision の
  keypointrcnn_resnet50_fpn（COCO 17 キーポイント）** で完結させ、MediaPipe は「概念＋任意
  導入」に留める（このスクリプト末尾で import を try/except ガードして案内のみ）。

学ぶこと:
  - keypoint R-CNN は「人を検出 → その人の 17 関節を回帰」する2段構え。出力 keypoints は
    (人数, 17, 3=x,y,score)。COCO 17 点の並びとスケルトン(辺)の張り方を体で覚える。
  - **domain gap の落とし穴**: 実写の人で学習したモデルは“おもちゃの合成図形”をほぼ検出できない
    （検出数 0 になりやすい）。本スクリプトはそれを実演し、検出が無い場合は「真の関節位置」で
    スケルトンを描いて表現自体は学べるようにする。data/ に実写人物画像を置けばモデルが発火する。

入力は腕を振る人型図形の連番フレーム（合成“動画”）。

実行: uv run python lectures/27_depth_pose_flow/02_pose_keypoints.py
出力: outputs/27_depth_pose_flow/02_*.png / 02_pose.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import dpf_helpers as H  # noqa: E402


def load_frames() -> tuple[list[tuple[np.ndarray, np.ndarray | None]], bool]:
    """(rgb, gt_keypoints) のリストと「合成かどうか」を返す。data/ の実画像があれば優先。"""
    H.DATA_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in H.DATA_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if imgs:
        print(f"[入力] data/ の実画像 {len(imgs)} 枚を使用（GT キーポイントは無し）")
        return [(np.array(Image.open(p).convert("RGB")), None) for p in imgs], False
    print("[入力] 腕を振る人型図形 4 フレームを合成（GT キーポイントつき）")
    return [(rgb, kp) for rgb, kp in H.make_humanoid_frames(4)], True


def load_model():
    """keypoint R-CNN をロード。失敗時は None（フォールバックで GT 描画に切替）。"""
    try:
        import torch
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights,
            keypointrcnn_resnet50_fpn,
        )

        torch.set_num_threads(4)
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights, progress=False).eval()
        return model
    except Exception as e:  # noqa: BLE001
        print(f"[警告] keypoint R-CNN のロードに失敗（{type(e).__name__}: {e}）。GT 描画のみ。")
        return None


def infer_keypoints(model, rgb: np.ndarray, score_thresh: float = 0.7):
    """1 枚に対し推論し、最高スコアの人の keypoints(17,3) を返す。検出なしなら None。"""
    import torch

    # uint8 RGB → [0,1] float の (3,H,W)。keypoint R-CNN は正規化を内部で行うので [0,1] でよい。
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    with torch.inference_mode():
        result = model([tensor])[0]
    scores = result["scores"]
    if len(scores) == 0 or float(scores[0]) < score_thresh:
        return None, float(scores[0]) if len(scores) else 0.0
    best = int(torch.argmax(scores))
    kpts = result["keypoints"][best].cpu().numpy()  # (17,3)
    return kpts, float(scores[best])


def main() -> None:
    out = H.ensure_output_dir()
    H.set_seed(2)

    frames, synthetic = load_frames()
    model = load_model()

    print("\n=== 02: 姿勢/キーポイント推定 (keypoint R-CNN, COCO17) ===")
    print(f"keypoint 名: {H.COCO_KEYPOINT_NAMES}")

    panels: list[np.ndarray] = []
    summary: list[dict] = []
    for i, (rgb, gt_kp) in enumerate(frames):
        used = "none"
        kp_for_draw = None
        if model is not None:
            pred, top_score = infer_keypoints(model, rgb)
            if pred is not None:
                kp_for_draw = pred
                used = f"model(score={top_score:.2f})"
            else:
                print(f"  frame{i}: 人を検出できず（top_score={top_score:.2f}）→ "
                      "合成図形は学習分布外。GT 関節で代替描画。")
        if kp_for_draw is None and gt_kp is not None:
            kp_for_draw = gt_kp
            used = "ground-truth(fallback)"

        if kp_for_draw is not None:
            vis = H.draw_skeleton(rgb, kp_for_draw)
        else:
            vis = rgb.copy()  # 実画像で検出ゼロのときは素のまま
        # フレーム高さを 240 に揃えて横連結できるよう縮小
        scale = 240 / vis.shape[0]
        vis_r = np.array(Image.fromarray(vis).resize((int(vis.shape[1] * scale), 240)))
        panels.append(vis_r)
        Image.fromarray(vis).save(out / f"02_frame{i}.png")
        summary.append({"frame": i, "source": used})

    montage = np.concatenate(panels, axis=1)
    Image.fromarray(montage).save(out / "02_pose_montage.png")
    print(f"\nsaved: {out/'02_pose_montage.png'} と 02_frame*.png")

    # --- MediaPipe は概念＋任意（実行経路では使わない）。導入有無だけ案内 ---
    mp_note = mediapipe_note()
    print(f"\n[MediaPipe(任意)] {mp_note}")

    (out / "02_pose.json").write_text(
        json.dumps({"synthetic": synthetic, "frames": summary, "mediapipe": mp_note},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {out/'02_pose.json'}")


def mediapipe_note() -> str:
    """mediapipe の導入有無を try/except で確認し、案内文を返す（実行経路では使わない）。"""
    try:
        import mediapipe  # noqa: F401

        return ("mediapipe 検出。Pose Landmarker は人体33点を CPU 実時間で推定でき、"
                "IMAGE/VIDEO/LIVE_STREAM の RunningMode を持つ。ただし numpy<2/protobuf 競合に注意。")
    except Exception:
        return ("未導入。`uv add --group pose mediapipe` で概念実験は可能（別環境推奨）。"
                "本講座の実行経路は keypoint R-CNN で完結させ衝突を回避している。")


if __name__ == "__main__":
    main()
