"""第13回 / 02: pipeline を分解する — processor + model を手書きで動かす。

01 の pipeline は便利だが中身がブラックボックスだった。ここでは前処理(processor / transforms)と
推論(model)を自分の手で分離して、次の2つを体得する。

  [A] 同じ画像を3つのエコシステムで分類する
      - torchvision: ResNet-18（CNN。畳み込み+残差接続）
      - HuggingFace: ViT-tiny（パッチ埋め込み + CLS トークンの Transformer）
      正準フロー: 画像 → (processor/transforms) → pixel_values → model → logits → argmax → id2label

  [B] 分類ヘッドを外して「埋め込み(特徴ベクトル)」を取り出す
      - torchvision: 最終 fc を Identity に差し替えて penultimate(512次元)を得る
      - timm:        num_classes=0 で埋め込み、forward_features で未プーリングの特徴マップ
      - HF ViT:      last_hidden_state の CLS トークン / 全パッチ平均 / pooler_output
      最後にコサイン類似度で「同クラスは近く、別クラスは遠い」ことを確認（15回への布石）。

transformers 5.x の要点:
  - AutoImageProcessor を使う（AutoFeatureExtractor は廃止）。画像 processor は torchvision 製の
    fast 実装のみで、use_fast 引数は不要（指定すると将来警告/エラー）。
  - 推論は必ず model.eval() + torch.inference_mode() で勾配を切る。

実行:
  uv run python lectures/13_classification_transfer_learning/02_resnet_vit_manual.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoImageProcessor, AutoModel, ViTForImageClassification

from dl_helpers import (
    SHAPE_CLASSES,
    ensure_out_dir,
    get_device,
    load_demo_images,
    make_shape_image,
    save_prediction_panel,
    set_seed,
)

transformers.logging.set_verbosity_error()

VIT_ID = "WinKawaks/vit-tiny-patch16-224"  # CPU で軽い小型 ViT


# =====================================================================
# [A] 同じ画像を CNN(torchvision) と ViT(HF) で分類する
# =====================================================================

def classify_torchvision(img, device: torch.device) -> tuple[str, torch.Tensor]:
    """torchvision の ResNet-18 で分類し (ラベル, logits) を返す。"""
    weights = ResNet18_Weights.DEFAULT  # ImageNet-1k で学習済みの重みセット
    model = resnet18(weights=weights).to(device).eval()
    preprocess = weights.transforms()  # 重みに紐づく正準な前処理(resize/centercrop/normalize)
    categories = weights.meta["categories"]  # 1000クラスのラベル名（ImageNet-1k 標準の並び）

    # transforms.v2 のパイプラインが PIL → 正規化済みテンソル (3,H,W) を作る。unsqueeze でバッチ次元。
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)  # (1, 1000)
    idx = int(logits.argmax(-1).item())
    return categories[idx], logits.squeeze(0)


def classify_vit(img, device: torch.device) -> tuple[str, torch.Tensor, torch.Tensor]:
    """HF の ViT-tiny で分類し (ラベル, logits, pixel_values) を返す。"""
    processor = AutoImageProcessor.from_pretrained(VIT_ID)  # fast 実装（torchvision バックエンド）
    model = ViTForImageClassification.from_pretrained(VIT_ID).to(device).eval()

    # processor が resize→rescale(0-1)→normalize をまとめて行い pixel_values (1,3,224,224) を作る。
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits  # (1, 1000)
    idx = int(logits.argmax(-1).item())
    return model.config.id2label[idx], logits.squeeze(0), inputs["pixel_values"]


# =====================================================================
# [B] 埋め込み（特徴ベクトル）の取り出し
# =====================================================================

def embed_torchvision(img, device: torch.device) -> torch.Tensor:
    """ResNet-18 の最終 fc を Identity に差し替え、penultimate 512次元を取り出す。"""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()  # 1000クラスの分類層を「素通し」にすると直前の512次元が出力になる
    model = model.to(device).eval()
    x = weights.transforms()(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(x).squeeze(0)  # (512,)


def build_timm_embedder(device: torch.device):
    """timm の ResNet-18 を「埋め込み器」として作る（num_classes=0 で分類ヘッドを外す）。"""
    # num_classes=0 にすると、global average pooling 後の特徴ベクトルが出力になる。
    model = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
    # モデル固有の前処理（入力サイズ・平均/分散）を自動生成する。手打ちでズラす事故を防ぐ定石。
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform


def embed_timm(model, transform, img, device: torch.device) -> tuple[torch.Tensor, tuple[int, ...]]:
    """timm モデルで埋め込み(プール済み)と forward_features(未プーリング特徴マップ)を返す。"""
    x = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        pooled = model(x).squeeze(0)  # (512,) ← global average pooling 済み
        feat_map = model.forward_features(transform(img).unsqueeze(0).to(device))  # (1,512,7,7)
    return pooled, tuple(feat_map.shape)


def embed_vit_base(img, device: torch.device) -> dict[str, torch.Tensor]:
    """ViT の base モデル(分類ヘッド無し)から CLS / mean-pool / pooler の3種を取り出す。"""
    processor = AutoImageProcessor.from_pretrained(VIT_ID)
    model = AutoModel.from_pretrained(VIT_ID).to(device).eval()  # ViTModel（重みは分類版とキャッシュ共有）
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    last = out.last_hidden_state  # (1, 197, 192) = 1 CLS + 196 patch トークン, hidden=192
    return {
        "cls": last[:, 0].squeeze(0),  # 先頭の CLS トークン（分類用に集約された表現）
        "mean": last[:, 1:].mean(dim=1).squeeze(0),  # パッチトークンの平均（mean pooling）
        "pooler": out.pooler_output.squeeze(0),  # pooler 出力。※下の注記参照
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 正規化してから内積 = コサイン類似度。"""
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return float((a * b).sum().item())


def main() -> None:
    set_seed(0)
    out_dir = ensure_out_dir()
    device = get_device()
    print(f"[info] device = {device}")

    images, names = load_demo_images(n=4)
    demo = images[0]
    demo_name = names[0]

    # ---------- [A] 分類: CNN vs ViT ----------
    print("\n=== [A] 同じ画像を CNN(torchvision ResNet-18) と ViT(HF vit-tiny) で分類 ===")
    panel_imgs, panel_titles = [], []
    cls_records = []
    for img, name in zip(images, names):
        tv_label, tv_logits = classify_torchvision(img, device)
        vit_label, vit_logits, pixel_values = classify_vit(img, device)
        print(f"\n[{name}]")
        print(f"  torchvision ResNet-18 : {tv_label}")
        print(f"  HF ViT-tiny           : {vit_label}")
        print(f"  pixel_values shape     : {tuple(pixel_values.shape)}  (前処理後の入力テンソル)")
        print(f"  logits shape           : {tuple(tv_logits.shape)} (1000クラス分のスコア)")
        panel_imgs.append(img)
        panel_titles.append(
            f"CNN: {tv_label.split(',')[0]}\nViT: {vit_label.split(',')[0]}"
        )
        cls_records.append({"input": name, "resnet18": tv_label, "vit_tiny": vit_label})

    save_prediction_panel(
        panel_imgs,
        panel_titles,
        out_dir / "02_cnn_vs_vit.png",
        suptitle="Manual classify: CNN(ResNet-18) vs ViT(vit-tiny), same image",
    )
    print(
        "\n  → 合成図形は ImageNet ラベルを持たないので予測は\"それっぽい別物\"。"
        "学ぶのは『画像→pixel_values→logits→argmax→id2label』の手順です。"
    )

    # ---------- [B] 埋め込みの取り出し ----------
    print("\n=== [B] 分類ヘッドを外して埋め込みを取り出す ===")
    tv_emb = embed_torchvision(demo, device)
    timm_model, timm_tf = build_timm_embedder(device)
    timm_emb, timm_featmap = embed_timm(timm_model, timm_tf, demo, device)
    vit_embs = embed_vit_base(demo, device)

    print(f"\n対象画像: {demo_name}")
    print(f"  torchvision penultimate (fc=Identity) : {tuple(tv_emb.shape)}")
    print(f"  timm num_classes=0 (pooled)           : {tuple(timm_emb.shape)}")
    print(f"  timm forward_features (未プーリング)    : {timm_featmap}  ← (B,C,H,W) の特徴マップ")
    print(f"  HF ViT  CLS トークン                   : {tuple(vit_embs['cls'].shape)}")
    print(f"  HF ViT  mean-pool(パッチ平均)          : {tuple(vit_embs['mean'].shape)}")
    print(f"  HF ViT  pooler_output                  : {tuple(vit_embs['pooler'].shape)}")
    print(
        "\n  注: vit-tiny の pooler はチェックポイントに含まれず乱数初期化のことがある"
        "（ロード時の MISSING 警告）。\n"
        "      埋め込みには学習済みの CLS トークンか mean-pool を使うのが安全。"
    )

    # ---------- [C] コサイン類似度デモ（15回への布石） ----------
    print("\n=== [C] 埋め込みのコサイン類似度: 同クラスは近く・別クラスは遠い ===")
    rng_imgs = {
        "red_circle #1": make_shape_image(SHAPE_CLASSES[0], size=224),
        "red_circle #2": make_shape_image(SHAPE_CLASSES[0], size=224),
        "blue_triangle": make_shape_image("blue_triangle", size=224),
    }
    embs = {
        k: embed_timm(timm_model, timm_tf, v, device)[0] for k, v in rng_imgs.items()
    }
    sim_same = cosine(embs["red_circle #1"], embs["red_circle #2"])
    sim_diff = cosine(embs["red_circle #1"], embs["blue_triangle"])
    print(f"  cos(red_circle#1, red_circle#2) = {sim_same:.3f}  (同クラス → 高い)")
    print(f"  cos(red_circle#1, blue_triangle)= {sim_diff:.3f}  (別クラス → 低い)")
    print(f"  → 同クラスの方が類似度が高い: {sim_same > sim_diff}")

    save_prediction_panel(
        list(rng_imgs.values()),
        [
            f"{k}\ncos vs #1 = {cosine(embs['red_circle #1'], v):.2f}"
            for k, v in embs.items()
        ],
        out_dir / "02_embedding_cosine.png",
        suptitle="timm ResNet-18 embeddings: same-class is more similar",
        ncols=3,
    )

    # ---------- 保存 ----------
    summary = {
        "device": str(device),
        "classification": cls_records,
        "embedding_dims": {
            "torchvision_penultimate": list(tv_emb.shape),
            "timm_pooled": list(timm_emb.shape),
            "timm_feature_map": list(timm_featmap),
            "vit_cls": list(vit_embs["cls"].shape),
            "vit_mean": list(vit_embs["mean"].shape),
        },
        "cosine_demo": {"same_class": round(sim_same, 4), "diff_class": round(sim_diff, 4)},
    }
    (out_dir / "02_manual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] saved figures and 02_manual_summary.json to {out_dir}")


if __name__ == "__main__":
    main()
