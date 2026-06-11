"""39_clip_distillation 共通ヘルパ。

このモジュール（CLIP/VLM 蒸留）で各スクリプトが共有する道具をここに集約する。
単一責務の小関数を並べ、本編スクリプトからは「何をしているか」だけ読めば済む状態にする。

提供するもの:
  - 乱数固定 / 図の保存（matplotlib Agg、ASCII タイトル）
  - cv2 合成図形データセット（色 + 形のクラス）
  - teacher CLIP（openai/clip-vit-base-patch32, transformers v5）のロードと埋め込み計算
  - 小型 student CNN（teacher の画像埋め込み次元 512 に合わせる）
  - 蒸留損失（コサイン + MSE）と評価指標（ゼロショット精度 / Recall@k / 整合度）

設計メモ:
  CPU 前提。teacher は eval() + inference_mode() で凍結し、勾配を流さない。
  「埋め込みは L2 正規化してから比較する」を全関数で徹底する（スケールずれ防止）。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 画面を持たない環境でも図を保存できるよう Agg を明示（imshow は使わない）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# 定数（このモジュール全体で共有）
# ---------------------------------------------------------------------------

# teacher = openai/clip-vit-base-patch32 の画像/テキスト射影次元。student もこれに合わせる。
EMBED_DIM = 512
# student に与える画像サイズ。teacher(224) より小さくして「安く・速く」を体現する。
STUDENT_IMG = 64
TEACHER_NAME = "openai/clip-vit-base-patch32"

# クラスは「色 + 形」の組。色は CLIP が強く認識する手がかりで、teacher のゼロショットが安定する。
# (クラス名, ゼロショット用プロンプト, 描画色 RGB, 形タグ)
CLASSES: list[tuple[str, str, tuple[int, int, int], str]] = [
    ("red circle", "a photo of a red circle", (220, 40, 40), "circle"),
    ("green square", "a photo of a green square", (40, 180, 60), "square"),
    ("blue triangle", "a photo of a blue triangle", (50, 70, 220), "triangle"),
]
CLASS_NAMES = [c[0] for c in CLASSES]
CLASS_PROMPTS = [c[1] for c in CLASSES]
NUM_CLASSES = len(CLASSES)


def repo_root() -> Path:
    """リポジトリ直下を返す（lectures/<id>/ から 2 つ上）。"""
    return Path(__file__).resolve().parents[2]


def out_dir() -> Path:
    """このモジュールの成果物保存先 outputs/39_clip_distillation を作って返す。"""
    d = repo_root() / "outputs" / "39_clip_distillation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int = 0) -> None:
    """再現性のため numpy / torch の乱数を固定する。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_fig(fig, name: str) -> Path:
    """図を outputs に保存してパスを返す（タイトルは ASCII 推奨でフォント警告回避）。"""
    path = out_dir() / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def pick_device() -> torch.device:
    """本講座は CPU 前提。明示的に CPU を返す。"""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 合成データセット（cv2 で色 + 形の図形を描く）
# ---------------------------------------------------------------------------


def _draw_one(rng: np.random.Generator, cls_idx: int, size: int = 128) -> np.ndarray:
    """1 枚の合成画像（uint8 RGB, HxWx3）を描く。背景は薄いグレー。"""
    _, _, color, shape = CLASSES[cls_idx]
    # 背景はクラスごとに固定せず、薄いランダムグレーにして「色 = 前景」を学ばせる。
    bg = int(rng.integers(205, 235))
    img = np.full((size, size, 3), bg, dtype=np.uint8)

    cx = int(rng.integers(size // 3, size * 2 // 3))
    cy = int(rng.integers(size // 3, size * 2 // 3))
    r = int(rng.integers(size // 5, size // 3))
    col = (int(color[0]), int(color[1]), int(color[2]))

    if shape == "circle":
        cv2.circle(img, (cx, cy), r, col, thickness=-1)
    elif shape == "square":
        # わずかに回転させて多様性を出す（同一クラス内の見た目を散らす）。
        rect = cv2.boxPoints(((cx, cy), (2 * r, 2 * r), float(rng.integers(-20, 20))))
        cv2.fillConvexPoly(img, rect.astype(np.int32), col)
    else:  # triangle
        ang = rng.uniform(0, 2 * np.pi)
        pts = []
        for k in range(3):
            a = ang + k * 2 * np.pi / 3
            pts.append([cx + int(r * np.cos(a)), cy + int(r * np.sin(a))])
        cv2.fillConvexPoly(img, np.array(pts, dtype=np.int32), col)
    return img


def make_dataset(n_per_class: int, seed: int = 0, size: int = 128):
    """合成データセットを作る。

    返り値: (images_uint8, labels)
      images_uint8: list[np.ndarray]  各 (size,size,3) RGB uint8
      labels: np.ndarray[int]         クラス index
    """
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for cls_idx in range(NUM_CLASSES):
        for _ in range(n_per_class):
            images.append(_draw_one(rng, cls_idx, size))
            labels.append(cls_idx)
    return images, np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# teacher CLIP（transformers v5）
# ---------------------------------------------------------------------------


def load_teacher(device: torch.device | None = None):
    """teacher CLIP（openai/clip-vit-base-patch32）と processor をロードする。

    eval() で凍結。重みは初回のみ HF からダウンロード（以降キャッシュ）。
    """
    from transformers import CLIPModel, CLIPProcessor

    device = device or pick_device()
    model = CLIPModel.from_pretrained(TEACHER_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(TEACHER_NAME)
    return model, processor, device


def teacher_logit_scale(model) -> float:
    """teacher の logit_scale（温度）の exp 値を返す。CLIP は学習で 100 付近に収束する。"""
    with torch.inference_mode():
        return float(model.logit_scale.exp())


@torch.inference_mode()
def teacher_image_embeds(model, processor, images_uint8, device, batch: int = 32):
    """teacher の画像埋め込みを L2 正規化して返す（[N, 512]）。

    注意: get_image_features().pooler_output は *未正規化*（ノルム約 10〜12）。
    比較・蒸留の前に必ず F.normalize する（スケールずれ防止の最重要ポイント）。
    """
    from PIL import Image

    embs = []
    for i in range(0, len(images_uint8), batch):
        chunk = images_uint8[i : i + batch]
        pil = [Image.fromarray(a) for a in chunk]
        inputs = processor(images=pil, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        embs.append(F.normalize(feats, dim=-1))
    return torch.cat(embs, dim=0)


@torch.inference_mode()
def teacher_text_embeds(model, processor, prompts, device):
    """teacher のテキスト埋め込みを L2 正規化して返す（[C, 512]）。"""
    inputs = processor(text=list(prompts), return_tensors="pt", padding=True).to(device)
    out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# student（小型 CNN）と前処理
# ---------------------------------------------------------------------------


class StudentCNN(nn.Module):
    """teacher の画像埋め込み（512 次元）を模倣する小型 CNN。

    64x64 入力 → 3 つの conv ブロック → GAP → Linear で 512 次元へ。
    パラメータは数十万程度で、teacher の画像タワー（約 8800 万）より桁違いに軽い。
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),   # 64 -> 32
            block(32, 64),  # 32 -> 16
            block(64, 128), # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)  # 正規化は呼び出し側で行う（用途に応じて使い分けるため）


def student_preprocess(images_uint8, device) -> torch.Tensor:
    """合成画像（uint8 RGB list）を student 用テンソル [N,3,64,64] に変換する。

    teacher の重い前処理（224 + CLIP 正規化）と違い、軽い [-1,1] 正規化で済ませる。
    """
    arr = np.stack(
        [cv2.resize(a, (STUDENT_IMG, STUDENT_IMG), interpolation=cv2.INTER_AREA) for a in images_uint8]
    ).astype(np.float32)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2) / 255.0  # [N,3,H,W] in [0,1]
    t = (t - 0.5) / 0.5  # -> [-1,1]
    return t.to(device)


def count_params(model: nn.Module) -> int:
    """学習対象パラメータ数を数える。"""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 蒸留損失
# ---------------------------------------------------------------------------


def embedding_distill_loss(
    student_emb: torch.Tensor, teacher_emb: torch.Tensor, mse_weight: float = 1.0
):
    """埋め込み蒸留の損失（コサイン距離 + MSE）。

    student/teacher を L2 正規化し、(1 - cos) を主損失、MSE を補助に重み付き加算する。
    正規化後の MSE はコサインと等価に近いが、両方入れると初期収束が安定する。
    """
    s = F.normalize(student_emb, dim=-1)
    t = F.normalize(teacher_emb, dim=-1)
    cos = (s * t).sum(dim=-1)  # [N]
    cos_loss = (1.0 - cos).mean()
    mse_loss = F.mse_loss(s, t)
    return cos_loss + mse_weight * mse_loss, cos_loss.detach(), mse_loss.detach()


# ---------------------------------------------------------------------------
# 評価指標
# ---------------------------------------------------------------------------


def zeroshot_logits(img_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: float):
    """画像埋め込み x テキスト埋め込みのゼロショット類似度（温度付き）を返す。"""
    s = F.normalize(img_emb, dim=-1)
    t = F.normalize(text_emb, dim=-1)
    return logit_scale * s @ t.t()  # [N, C]


def zeroshot_accuracy(img_emb, text_emb, labels, logit_scale: float) -> float:
    """ゼロショット分類精度。各画像を最も近いテキストプロンプトに割り当てる。"""
    logits = zeroshot_logits(img_emb, text_emb, logit_scale)
    pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == labels).mean())


def zeroshot_predict(img_emb, text_emb, logit_scale: float) -> np.ndarray:
    """ゼロショットの予測クラス index を返す。"""
    logits = zeroshot_logits(img_emb, text_emb, logit_scale)
    return logits.argmax(dim=1).cpu().numpy()


def mean_cosine_alignment(a: torch.Tensor, b: torch.Tensor) -> float:
    """2 つの埋め込み集合の行ごとコサイン類似度の平均（teacher との整合度）。"""
    s = F.normalize(a, dim=-1)
    t = F.normalize(b, dim=-1)
    return float((s * t).sum(dim=-1).mean())


def retrieval_recall_at_k(img_emb: torch.Tensor, labels: np.ndarray, k: int = 5) -> float:
    """画像 -> 画像検索の Recall@k（leave-one-out, 同一クラスが上位 k に入る割合）。"""
    s = F.normalize(img_emb, dim=-1)
    sim = (s @ s.t()).cpu().numpy()
    np.fill_diagonal(sim, -np.inf)  # 自分自身は除外
    n = sim.shape[0]
    hits = 0
    for i in range(n):
        topk = np.argsort(-sim[i])[:k]
        if np.any(labels[topk] == labels[i]):
            hits += 1
    return hits / n


def agreement(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """2 つの予測列の一致率（student が teacher と同じ判断をした割合）。"""
    return float((pred_a == pred_b).mean())


# ---------------------------------------------------------------------------
# 簡易セルフテスト（このファイルを直接実行したとき）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(0)
    imgs, labels = make_dataset(2, seed=1)
    print(f"[helpers] dataset: {len(imgs)} images, classes={CLASS_NAMES}")
    student = StudentCNN()
    x = student_preprocess(imgs, pick_device())
    y = student(x)
    print(f"[helpers] student out shape={tuple(y.shape)} params={count_params(student):,}")
    print(f"[helpers] out_dir={out_dir()}")
    print("[helpers] OK")
