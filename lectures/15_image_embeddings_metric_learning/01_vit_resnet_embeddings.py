"""01: 画像埋め込みの取り出し — ViTModel / ResNetModel / timm の特徴とその「形」。

学ぶこと:
  - 分類ヘッド無しの ViTModel / ResNetModel から「埋め込み（ベクトル）」を取り出す。
  - 出力の "形" の違いを取り違えない:
      ViT   : last_hidden_state = (B, 1+num_patches, hidden)。先頭[:,0]が CLS トークン。
              全パッチの平均 (mean pooling) も定番の埋め込み。
      ResNet: last_hidden_state = (B, C, H, W) の「特徴マップ」。pooler_output = (B, C, 1, 1)
              の Global Average Pooling 済みベクトル（flatten して (B, C) で使う）。
  - 落とし穴: google/vit-base-patch16-224 のチェックポイントには学習済み pooler が無く、
    ViTModel の pooler_output 用の重みは「新規初期化（未学習）」になる（ロード時の
    LOAD REPORT に pooler.dense.* = MISSING と出る）。pooler は CLS の線形変換+tanh
    なので形上は埋め込みが出るが、未学習ゆえ品質の保証が無い。再現性・意味づけの面から
    ViT の埋め込みは CLS（last_hidden_state[:,0]）か mean pooling を使うのが定石。
  - timm.create_model(num_classes=0) はプール済み埋め込み、forward_features は未プール特徴。
  - F.normalize で L2 正規化 → 内積がコサイン類似度になり、同クラスが高く出ることを確認。

実行:
  uv run python lectures/15_image_embeddings_metric_learning/01_vit_resnet_embeddings.py
初回はモデル重みを HuggingFace / timm からダウンロードする（以後はキャッシュ）。
結果は lectures/15_image_embeddings_metric_learning/outputs/ に保存（画面表示はしない）。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # 画面の無い環境でも図を保存できるよう、必ず pyplot より前に設定する
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, ResNetModel, ViTModel

import embed_helpers as eh

VIT_ID = "google/vit-base-patch16-224"  # 重すぎる場合は WinKawaks/vit-tiny-patch16-224 などでも可
RESNET_ID = "microsoft/resnet-18"  # 小型・CPU で軽い CNN


def build_small_labeled_set() -> tuple[list, np.ndarray, list[str]]:
    """4 クラス × 2 枚 = 8 枚の小さな確認用セットを、同クラスが隣り合う順で作る。

    こうしておくと、コサイン類似度行列が 2x2 の明るいブロックとして見え、
    『同クラスが近い』ことが目で分かる。
    """
    images, labels, names = eh.make_dataset(n_per_class=2, seed=123)
    # make_dataset は 6 クラス×2枚。先頭 4 クラスだけ使う（=先頭 8 枚）。
    keep = labels < 4
    images = [im for im, k in zip(images, keep) if k]
    labels = labels[keep]
    return images, labels, [names[i] for i in range(4)]


@torch.inference_mode()  # 推論だけ。勾配を切ってメモリ・時間を節約（CPU では特に効く）
def vit_embeddings(images: list, device: torch.device) -> dict[str, np.ndarray]:
    """ViTModel から CLS / mean pooling / pooler_output の 3 通りの埋め込みを取り出す。"""
    proc = AutoImageProcessor.from_pretrained(VIT_ID)  # transformers v5 は fast 実装（torchvision）
    model = ViTModel.from_pretrained(VIT_ID).to(device).eval()

    inputs = proc(images=images, return_tensors="pt").to(device)  # pixel_values: (B,3,224,224)
    out = model(**inputs)
    last = out.last_hidden_state  # (B, 1+196, 768): [:,0]=CLS, [:,1:]=各パッチ
    print(f"  ViT last_hidden_state : {tuple(last.shape)}  (先頭が CLS、残り 196 がパッチ)")
    print(f"  ViT pooler_output     : {tuple(out.pooler_output.shape)}  ← この CKPT では未学習(新規初期化)")

    cls = last[:, 0]  # CLS トークン: 系列全体を要約するよう学習されたトークン
    mean = last[:, 1:].mean(dim=1)  # パッチトークンの平均（mean pooling）
    pooler = out.pooler_output  # ★ pooler.dense は CKPT に無く新規初期化（未学習）。tanh(W·CLS)。
    return {
        "vit_cls": cls.cpu().numpy(),
        "vit_mean": mean.cpu().numpy(),
        "vit_pooler(untrained)": pooler.cpu().numpy(),
    }


@torch.inference_mode()
def resnet_embeddings(images: list, device: torch.device) -> dict[str, np.ndarray]:
    """ResNetModel(HF) の特徴マップと GAP 済みベクトルを取り出す。"""
    proc = AutoImageProcessor.from_pretrained(RESNET_ID)
    model = ResNetModel.from_pretrained(RESNET_ID).to(device).eval()

    inputs = proc(images=images, return_tensors="pt").to(device)
    out = model(**inputs)
    fmap = out.last_hidden_state  # (B, 512, 7, 7) ← これは「特徴マップ」。ベクトルではない！
    pooled = out.pooler_output  # (B, 512, 1, 1) = GAP 済み
    print(f"  ResNet last_hidden_state: {tuple(fmap.shape)}  ← (B,C,H,W) の特徴マップ")
    print(f"  ResNet pooler_output    : {tuple(pooled.shape)}  → flatten して (B,512) で使う")

    gap = pooled.flatten(1)  # (B, 512): 末尾の (1,1) を潰してベクトルにする
    return {"resnet_gap": gap.cpu().numpy()}


@torch.inference_mode()
def timm_embeddings(images: list, device: torch.device) -> dict[str, np.ndarray]:
    """timm.create_model(num_classes=0) でプール済み埋め込みを得る。前処理もモデル固有値で。"""
    model = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
    # ★ 前処理は手打ちせず、モデル固有の平均/分散/サイズを自動生成する（ズレ事故を防ぐ）
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)

    batch = torch.stack([transform(im) for im in images]).to(device)
    feat = model(batch)  # num_classes=0 → プール済み (B, 512)
    fmap = model.forward_features(batch)  # 未プール特徴 (B, 512, 7, 7)
    print(f"  timm resnet18 num_classes=0 : {tuple(feat.shape)}  (プール済み埋め込み)")
    print(f"  timm forward_features       : {tuple(fmap.shape)}  (未プール特徴マップ)")
    return {"timm_resnet18": feat.cpu().numpy()}


def save_cosine_heatmap(emb: np.ndarray, labels: np.ndarray, names: list[str], title: str, path: str) -> None:
    """L2 正規化してコサイン類似度行列を作り、ヒートマップとして保存する。"""
    # F.normalize でも np でも同じ。ここでは「正規化してから内積=コサイン類似度」を示す。
    t = F.normalize(torch.from_numpy(emb).float(), p=2, dim=1)  # ★ L2 正規化が肝
    sim = (t @ t.T).numpy()

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="viridis")
    tick_labels = [names[c] for c in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    os.makedirs(eh.OUT_DIR, exist_ok=True)
    device = eh.get_device()
    print(f"device = {device}")

    images, labels, names = build_small_labeled_set()
    print(f"確認用セット: {len(images)} 枚 / {len(names)} クラス {names}\n")

    print("[ViT] 埋め込みの取り出し")
    vit = vit_embeddings(images, device)
    print("\n[ResNet] 埋め込みの取り出し")
    res = resnet_embeddings(images, device)
    print("\n[timm] 埋め込みの取り出し")
    tim = timm_embeddings(images, device)

    all_emb = {**vit, **res, **tim}

    # 各埋め込みについて「同クラス平均類似度 − 異クラス平均類似度」を計算し、分離度を比べる。
    print("\n[分離度] same(同クラス) と diff(異クラス) のコサイン類似度の平均差（大きいほど良い埋め込み）")
    same_mask = labels[:, None] == labels[None, :]
    off = ~np.eye(len(labels), dtype=bool)  # 対角（自分自身）を除く
    for key, emb in all_emb.items():
        nrm = eh.l2_normalize(emb)
        sim = nrm @ nrm.T
        same = sim[same_mask & off].mean()  # 同クラス（自分自身を除く）の平均類似度
        diff = sim[~same_mask].mean()  # 異クラスの平均類似度
        print(f"  {key:22s} same={same:+.3f}  diff={diff:+.3f}  gap={same - diff:+.3f}")

    # ヒートマップ保存（CLS / pooler(未学習) / ResNet GAP を見比べる。pooler は未学習で要注意）
    save_cosine_heatmap(
        vit["vit_cls"], labels, names, "ViT CLS (good)",
        os.path.join(eh.OUT_DIR, "01_cosine_vit_cls.png"),
    )
    save_cosine_heatmap(
        vit["vit_pooler(untrained)"], labels, names, "ViT pooler_output (untrained)",
        os.path.join(eh.OUT_DIR, "01_cosine_vit_pooler.png"),
    )
    save_cosine_heatmap(
        res["resnet_gap"], labels, names, "ResNet GAP (good)",
        os.path.join(eh.OUT_DIR, "01_cosine_resnet_gap.png"),
    )

    print("\n読み取り方:")
    print("  - vit_mean / vit_cls / resnet_gap / timm は gap が大きい（同クラスがちゃんと近い）。")
    print("    特に mean pooling は CLS より gap が大きく出やすい（全パッチの情報を均すため）。")
    print("  - vit_pooler(untrained) は CLS の tanh(W·) なので形は出るが、重みが未学習で")
    print("    品質保証が無い。再現性のため ViT 埋め込みは CLS か mean を使うのが定石。")
    print(f"\n完了。{eh.OUT_DIR}/01_cosine_*.png を見比べてください。")


if __name__ == "__main__":
    main()
