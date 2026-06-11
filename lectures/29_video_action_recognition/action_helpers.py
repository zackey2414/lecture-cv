"""第29回 動画理解・行動認識 — 共通の道具箱（helpers）。

このモジュールは「クリップ(複数フレーム)から行動クラスを推定する」ための
共通処理を一手に引き受ける。各スクリプト(01〜03 / mini_project)はここを import し、
本質（モデルの使い方・前処理・評価）だけに集中できるようにしている。

ここに集約している責務:
  - device 判定 / スレッド数の調整（CPU 前提）
  - 合成「行動クリップ」の生成（動く図形。ネットもデータも不要）
  - mp4 への書き出し / cv2.VideoCapture での読み戻し（read_video 廃止のため）
  - フレームサンプリング（uniform / clip_len+frame_rate=stride）
  - r3d_18 用の専用正規化と (1,C,T,H,W) への整形
  - r3d_18 / VideoMAE のロード（VideoMAE は v5 のバイアス取りこぼしを手当て）
  - top-k accuracy / 混同行列 / 可視化

合成クリップには「本物の Kinetics ラベル」は無い。そこで評価では
「正しい前処理で得た予測」を各クリップの基準ラベル(pseudo-GT)とし、
前処理を変えると予測がどれだけ崩れるかを top-1/top-5 で測る方針を採る
（詳しくは 03_action_topk_eval.py と README 第8節）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless 環境でも図を保存できるよう、画面を持たないバックエンドにする
import matplotlib.pyplot as plt
import numpy as np
import torch

# 出力先（図・json はすべてここへ）。プロジェクト直下の outputs/<module_id>/ に統一。
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "29_video_action_recognition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# r3d_18(Kinetics-400) の専用正規化定数。torchvision の R3D_18_Weights.transforms() と同値。
# 「画像 ImageNet とは別の平均/分散」である点が重要（誤ると無意味な出力になる）。
R3D_MEAN = (0.43216, 0.394666, 0.37645)
R3D_STD = (0.22803, 0.22145, 0.216989)
R3D_SIZE = 112  # r3d_18 の入力解像度（112x112）


# ---------------------------------------------------------------------------
# device / スレッド
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """利用可能な最良の device を返す（本講座は CPU 前提だが定石は共通化する）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def limit_cpu_threads(n: int = 4) -> None:
    """CPU 推論が極端に遅くならないようスレッド数を穏当に固定する（任意）。"""
    try:
        torch.set_num_threads(max(1, n))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 合成「行動クリップ」の生成（動く図形）
# ---------------------------------------------------------------------------
# 各 "kind" は別々の運動シグネチャを持つ。行動認識モデルは時間方向の動きを見るので、
# 「何がどう動くか」を変えると別クラスへ分類される（＝混同行列に意味が出る）。
ACTION_KINDS: tuple[str, ...] = (
    "circle_right",  # 0: 円が左→右へ並進
    "square_down",  # 1: 四角が上→下へ並進
    "circle_expand",  # 2: 円が中心から拡大（ズームイン的）
    "line_spin",  # 3: 線分が回転
    "circle_zigzag",  # 4: 円がジグザグに往復
    "bar_sweep",  # 5: 縦帯が左→右へスイープ（ワイパー的）
)


def make_action_clip(
    kind: int,
    num_frames: int = 16,
    height: int = 120,
    width: int = 160,
    seed: int = 0,
) -> np.ndarray:
    """合成行動クリップを (T, H, W, 3) uint8 RGB で返す。

    背景に弱いノイズを乗せ、kind に応じた「動く図形」を描く。色は RGB 順で扱う
    （torch / transformers / matplotlib はすべて RGB 前提のため）。
    """
    rng = np.random.default_rng(seed)
    base_bg = rng.integers(18, 55, size=(height, width, 3), dtype=np.uint8)
    frames = []
    for t in range(num_frames):
        f = base_bg.copy()
        # フレームごとにごく弱いノイズを足し、完全な静止画の連続にしない
        f = np.clip(f.astype(np.int16) + rng.integers(-6, 7, f.shape), 0, 255).astype(np.uint8)
        phase = t / max(1, num_frames - 1)  # 0.0 → 1.0
        if kind == 0:  # 円が左→右
            x = int(width * 0.15 + phase * width * 0.7)
            cv2.circle(f, (x, height // 2), 16, (255, 200, 40), -1)
        elif kind == 1:  # 四角が上→下
            y = int(height * 0.15 + phase * height * 0.7)
            cv2.rectangle(f, (width // 2 - 16, y - 16), (width // 2 + 16, y + 16), (60, 230, 90), -1)
        elif kind == 2:  # 円が拡大
            r = int(6 + phase * 34)
            cv2.circle(f, (width // 2, height // 2), r, (240, 80, 200), -1)
        elif kind == 3:  # 線分が回転
            a = phase * 2.0 * math.pi
            x2 = int(width // 2 + 42 * math.cos(a))
            y2 = int(height // 2 + 42 * math.sin(a))
            cv2.line(f, (width // 2, height // 2), (x2, y2), (250, 250, 250), 5)
        elif kind == 4:  # 円がジグザグ往復
            x = int(width * 0.2 + abs((phase * 2 - 1)) * width * 0.6)
            y = int(height * 0.3 + math.sin(phase * 4 * math.pi) * height * 0.2)
            cv2.circle(f, (x, y), 14, (80, 180, 255), -1)
        elif kind == 5:  # 縦帯が左→右スイープ
            x = int(phase * (width - 20))
            cv2.rectangle(f, (x, 0), (x + 20, height), (255, 120, 120), -1)
        else:
            raise ValueError(f"unknown kind: {kind}")
        frames.append(f)
    return np.stack(frames, axis=0)


def make_clip_dataset(
    num_frames: int = 16, height: int = 120, width: int = 160, seeds_per_kind: int = 2
) -> tuple[list[np.ndarray], list[int]]:
    """全 kind × 複数 seed の小さな「行動データセット」を作る。

    返り値: (clips, kind_labels)。kind_labels は「どの運動で生成したか」の整数ラベル。
    （注: これは生成元のラベルであって Kinetics ラベルではない。評価方針は README 第8節）
    """
    clips: list[np.ndarray] = []
    labels: list[int] = []
    for k in range(len(ACTION_KINDS)):
        for s in range(seeds_per_kind):
            clips.append(make_action_clip(k, num_frames, height, width, seed=k * 100 + s))
            labels.append(k)
    return clips, labels


# ---------------------------------------------------------------------------
# mp4 への書き出し / cv2.VideoCapture での読み戻し
# ---------------------------------------------------------------------------
def write_clip_mp4(clip_rgb: np.ndarray, path: str | Path, fps: float = 8.0) -> Path:
    """(T,H,W,3) RGB のクリップを mp4 に書き出す。

    cv2 は BGR 前提なので RGB→BGR に変換してから書き込む。
    torchvision 0.26 で内蔵デコーダが廃止されたため、動画 I/O は cv2 に統一する。
    """
    path = Path(path)
    t, h, w, _ = clip_rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter を開けませんでした: {path}")
    for frame in clip_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def read_clip_mp4(path: str | Path) -> np.ndarray:
    """mp4 を cv2.VideoCapture で読み、(T,H,W,3) RGB の numpy 配列で返す。

    読み出しは BGR で来るので RGB へ戻す。これが torchvision read_video の代替。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture を開けませんでした: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"フレームを 1 枚も読めませんでした: {path}")
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# フレームサンプリング（行動認識の心臓部）
# ---------------------------------------------------------------------------
def uniform_indices(total: int, num_frames: int) -> np.ndarray:
    """全長 total から num_frames 枚を等間隔で選ぶインデックス列を返す。

    np.linspace で端から端まで均等に取る。動画全体を「広く薄く」見たいとき向き。
    total < num_frames でも端点を重複させて必ず num_frames 枚返す（落ちない）。
    """
    if total <= 0:
        raise ValueError("total は 1 以上が必要です")
    idx = np.linspace(0, total - 1, num=num_frames)
    return np.round(idx).astype(int)


def strided_indices(
    total: int, num_frames: int, frame_rate: int = 1, start: int | None = None
) -> np.ndarray:
    """clip_len(=num_frames) + frame_rate(=ストライド) 方式のサンプリング。

    start から frame_rate おきに num_frames 枚を取る（VideoMAE/3D CNN の標準作法）。
    末尾を超えたら最終フレームにクランプする。start 未指定なら中央寄せにする。
    frame_rate を変えると「どれだけ速い動きを捉えるか」が変わり、予測が変化する。
    """
    span = (num_frames - 1) * frame_rate + 1
    if start is None:
        start = max(0, (total - span) // 2)  # クリップ中央に寄せる
    idx = start + np.arange(num_frames) * frame_rate
    return np.clip(idx, 0, total - 1).astype(int)


# ---------------------------------------------------------------------------
# r3d_18 用の前処理（専用正規化 + (1,C,T,H,W) 整形）
# ---------------------------------------------------------------------------
def preprocess_for_r3d(
    clip_rgb: np.ndarray,
    indices: np.ndarray | None = None,
    size: int = R3D_SIZE,
    normalize: bool = True,
    mean: tuple[float, float, float] = R3D_MEAN,
    std: tuple[float, float, float] = R3D_STD,
) -> torch.Tensor:
    """(T,H,W,3) uint8 RGB → r3d_18 が食える (1, C, T, H, W) float32 へ。

    手順: フレーム選択 → [0,1] スケール → size×size へリサイズ → 専用正規化 →
    (T,H,W,C) から (C,T,H,W) へ並べ替え → バッチ次元を付ける。
    normalize=False や mean/std を変えると「壊れた前処理」を再現できる（実験用）。
    """
    if indices is not None:
        clip_rgb = clip_rgb[indices]
    # 各フレームを size×size にリサイズ（cv2 は (W,H) 指定）
    resized = np.stack(
        [cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in clip_rgb]
    )  # (T, size, size, 3)
    x = torch.from_numpy(resized).float() / 255.0  # [0,1] スケール
    if normalize:
        m = torch.tensor(mean).view(1, 1, 1, 3)
        s = torch.tensor(std).view(1, 1, 1, 3)
        x = (x - m) / s
    # (T,H,W,C) -> (C,T,H,W) -> (1,C,T,H,W)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return x


# ---------------------------------------------------------------------------
# モデルのロード
# ---------------------------------------------------------------------------
def load_r3d():
    """torchvision r3d_18(Kinetics-400) をロードして (model, categories) を返す。

    初回のみ重み(~127MB)を ~/.cache へダウンロードする。以後はキャッシュから即起動。
    """
    from torchvision.models.video import R3D_18_Weights, r3d_18

    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights).eval()
    categories = list(weights.meta["categories"])  # 400 個の行動名
    return model, categories


def _patch_videomae_biases(model, model_name: str) -> int:
    """VideoMAE の q_bias/v_bias 取りこぼしを手当てする（transformers v5 の落とし穴）。

    VideoMAE の自己注意は q_bias と v_bias だけを学習し k_bias は 0 に固定する特殊構造。
    transformers v5 は attention を query/key/value の Linear に分解した結果、チェックポイントの
    q_bias/v_bias を query.bias/value.bias へ正しく移せず、バイアスが 0 のままになることがある。
    そこで元チェックポイントから直接読み込んで埋め直す。失敗しても致命的ではないので 0 を返す。
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(model_name, "model.safetensors")
        sd = load_file(path)
    except Exception:
        return 0
    patched = 0
    for i, layer in enumerate(model.videomae.encoder.layer):
        attn = layer.attention.attention
        qb = sd.get(f"videomae.encoder.layer.{i}.attention.attention.q_bias")
        vb = sd.get(f"videomae.encoder.layer.{i}.attention.attention.v_bias")
        if qb is not None and vb is not None and hasattr(attn, "query"):
            with torch.no_grad():
                attn.query.bias.copy_(qb)
                attn.value.bias.copy_(vb)
            patched += 1
    return patched


def load_videomae(model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics"):
    """VideoMAE(kinetics finetuned) をロードして (model, processor, id2label, n_patched) を返す。

    初回のみ重み(~330MB)をダウンロード。読み込み後に q_bias/v_bias を手当てする。
    """
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name).eval()
    n_patched = _patch_videomae_biases(model, model_name)
    id2label = model.config.id2label
    return model, processor, id2label, n_patched


# ---------------------------------------------------------------------------
# 推論結果のデコード / 評価
# ---------------------------------------------------------------------------
def topk_from_logits(logits: torch.Tensor, labels, k: int = 5):
    """1 サンプルの logits から top-k の (ラベル名, 確率) を返す。"""
    probs = logits.softmax(dim=-1)[0]
    top = probs.topk(min(k, probs.numel()))
    out = []
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        name = labels[i] if not isinstance(labels, dict) else labels[i]
        out.append((name, float(p)))
    return out


def topk_accuracy(logits: np.ndarray, gt_indices: np.ndarray, k: int = 1) -> float:
    """top-k accuracy = (gt が上位 k 予測に入っているサンプルの割合)。

    logits: (N, C)、gt_indices: (N,)。行動認識の標準評価指標（top-1 / top-5）。
    """
    logits = np.asarray(logits)
    gt_indices = np.asarray(gt_indices)
    if logits.ndim != 2:
        raise ValueError("logits は (N, C) が必要です")
    # 各行で上位 k のクラスインデックスを取る（argsort 降順）
    topk = np.argsort(-logits, axis=1)[:, :k]
    hits = [gt in row for gt, row in zip(gt_indices, topk)]
    return float(np.mean(hits)) if len(hits) else 0.0


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """混同行列 (num_classes, num_classes) を素朴に作る。行=正解、列=予測。"""
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gt, pred):
        cm[int(g), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------
def save_clip_grid(clip_rgb: np.ndarray, path: str | Path, title: str = "", max_frames: int = 8):
    """クリップを横並びの静止画グリッドで保存（動きが時間順に見える）。"""
    t = clip_rgb.shape[0]
    idx = uniform_indices(t, min(max_frames, t))
    n = len(idx)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.0))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.imshow(clip_rgb[i])
        ax.set_title(f"t={i}", fontsize=8)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_bar(names, values, path: str | Path, title: str, ylabel: str = "value", ylim=None):
    """棒グラフを保存（top-k やスコア比較に使う）。"""
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(names)), 3.4))
    bars = ax.bar(range(len(names)), values, color="#4c72b0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_confusion(cm: np.ndarray, labels, path: str | Path, title: str):
    """混同行列をヒートマップで保存。"""
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(labels), 1.0 + 0.7 * len(labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground-truth")
    ax.set_title(title, fontsize=10)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thr else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_json(obj, path: str | Path):
    """結果を json で保存（再現性のため）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 自己点検（モデル不要・純計算）
# ---------------------------------------------------------------------------
def _self_check() -> None:
    """道具箱の健全性を確認する。モデル DL なしで動き、合成データの確認図を出す。"""
    print("=" * 70)
    print("action_helpers 自己点検（モデル不要）")
    print("=" * 70)
    print(f"device           : {get_device()}")
    print(f"出力ディレクトリ : {OUT_DIR}")

    # 1) 合成クリップ生成 → グリッド保存
    clip = make_action_clip(kind=0, num_frames=16)
    print(f"\n合成クリップ shape: {clip.shape} dtype={clip.dtype}（kind=circle_right）")
    save_clip_grid(clip, OUT_DIR / "helpers_clip_grid.png", title="synthetic clip: circle_right")

    # 2) mp4 書き出し → 読み戻し（cv2 ラウンドトリップ）
    mp4 = write_clip_mp4(clip, OUT_DIR / "helpers_clip.mp4", fps=8.0)
    back = read_clip_mp4(mp4)
    print(f"mp4 ラウンドトリップ: 書込 {clip.shape[0]} 枚 → 読戻 {back.shape[0]} 枚 ({mp4.name})")

    # 3) サンプリングの違いを表示
    total = 32
    print(f"\nフレームサンプリング（全長 total={total} を 8 枚に）:")
    print(f"  uniform           : {uniform_indices(total, 8).tolist()}")
    print(f"  strided(rate=1)   : {strided_indices(total, 8, frame_rate=1).tolist()}")
    print(f"  strided(rate=4)   : {strided_indices(total, 8, frame_rate=4).tolist()}")

    # 4) 前処理の整形（形だけ確認、モデルには通さない）
    x = preprocess_for_r3d(clip, uniform_indices(clip.shape[0], 16))
    print(f"\nr3d 前処理後 tensor: {tuple(x.shape)}（(1,C,T,H,W) になっていれば OK）")

    # 5) top-k accuracy / 混同行列の動作確認（ダミー）
    #   row1: gt=1 が top1、row2: gt=0 は top1 を外すが top2 に入る、row3: gt=1 が top1
    logits = np.array([[0.1, 0.9, 0.0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
    gt = np.array([1, 0, 1])
    print(f"\ntop-1 acc(ダミー)  : {topk_accuracy(logits, gt, k=1):.3f}（期待 0.667）")
    print(f"top-2 acc(ダミー)  : {topk_accuracy(logits, gt, k=2):.3f}（期待 1.000）")
    cm = confusion_matrix(gt, logits.argmax(1), num_classes=3)
    print(f"混同行列(ダミー)\n{cm}")

    print("\n自己点検 OK。図は outputs/29_video_action_recognition/ を参照。")


if __name__ == "__main__":
    limit_cpu_threads(4)
    _self_check()
