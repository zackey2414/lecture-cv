"""04: open_clip で MobileCLIP/TinyCLIP に触れ、参照リポの mobileclip_blt.ts に接続する。

ここまでで「埋め込み蒸留」を自前実装した。実務では、蒸留済みの効率 CLIP が
open_clip から使える。代表が TinyCLIP（重み継承 + 親和性蒸留）と MobileCLIP/MobileCLIP2
（DataCompDR による『データセット強化』で、アーキよりデータで効率化）。

このスクリプトは:
  1) open_clip.list_pretrained() で MobileCLIP/TinyCLIP の事前学習キーを確認
  2) MobileCLIP アーキを *オフラインで* 構築（pretrained=None）し、teacher との
     パラメータ数・推論レイテンシを比較（重み無しでも「効率」は測れる）
  3) 事前学習重みのロードを try/except で試す（ネットがあれば DL、無ければ概念で代替）
  4) 参照リポ /home/user/cluster-clip の mobileclip_blt.ts（TorchScript）に
     言及し、エッジ展開（torch.jit.load で Python 非依存配布）の位置づけを説明

実行:
    uv run python lectures/39_clip_distillation/04_open_clip_mobileclip.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_distill_helpers as H  # noqa: E402

REF_REPO = Path("/home/user/cluster-clip")
REF_TS = REF_REPO / "mobileclip_blt.ts"

# オフライン構築を試す候補（軽い順）。最初に成功したものを使う。
CANDIDATES = ["MobileCLIP-S1", "MobileCLIP-S2", "MobileCLIP2-S0", "TinyCLIP-ViT-40M-32-Text-19M"]


def list_efficient_models() -> None:
    """効率 CLIP の事前学習キーを列挙（list_pretrained で確認するのが正攻法）。"""
    import open_clip

    pretrained = open_clip.list_pretrained()
    eff = [(n, t) for (n, t) in pretrained if ("MobileCLIP" in n or "TinyCLIP" in n)]
    print(f"[open_clip] version={open_clip.__version__}  efficient pretrained keys ({len(eff)}):")
    for n, t in eff[:12]:
        print(f"    create_model_and_transforms('{n}', pretrained='{t}')")


def build_offline_student():
    """MobileCLIP/TinyCLIP アーキを重み無し(pretrained=None)で構築する。

    重みが無くても nn.Module は作れるので、パラメータ数や API（encode_image）は確認できる。
    """
    import open_clip

    for name in CANDIDATES:
        try:
            model = open_clip.create_model(name, pretrained=None)
            model.eval()
            return name, model
        except Exception as e:  # noqa: BLE001
            print(f"    skip {name}: {type(e).__name__}")
    return None, None


def _encode_image(model, x):
    """encode_image を持つモデル(open_clip/HFラッパ)も、forward だけの student も同じ呼び口にする。"""
    if hasattr(model, "encode_image"):
        return model.encode_image(x)
    return model(x)


def latency(model, image_size, device, runs: int = 5) -> float:
    """画像エンコードの中央値レイテンシ（ms）。ウォームアップ込み。"""
    import statistics

    size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
    x = torch.randn(1, 3, size[0], size[1], device=device)
    with torch.inference_mode():
        _encode_image(model, x)  # warmup
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _encode_image(model, x)
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def try_pretrained(name: str) -> bool:
    """事前学習重みのロードを試す（ネットが無ければ False で概念に切り替え）。"""
    import open_clip

    pretrained = open_clip.list_pretrained()
    tag = next((t for (n, t) in pretrained if n == name), None)
    if tag is None:
        print(f"[pretrained] {name} に対応する事前学習タグがありません")
        return False
    try:
        print(f"[pretrained] try load '{name}' pretrained='{tag}' (数百MB DL の可能性) ...")
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=tag)
        model.eval()
        with torch.inference_mode():
            f = model.encode_image(torch.randn(1, 3, 256, 256))
        print(f"[pretrained] OK: 蒸留済み効率 CLIP をロード。embed_dim={f.shape[-1]}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[pretrained] スキップ（ネット未接続/重み未取得）: {type(e).__name__}: {str(e)[:80]}")
        print("[pretrained] -> 概念で代替: TinyCLIP=重み継承+親和性蒸留 / MobileCLIP=DataCompDR データ強化")
        return False


def reference_repo_note() -> None:
    """参照リポ mobileclip_blt.ts（TorchScript）への接続を説明する。"""
    print("\n=== 参照リポ接続: cluster-clip の mobileclip ===")
    if REF_TS.exists():
        mb = REF_TS.stat().st_size / 1e6
        print(f"  {REF_TS}  ({mb:.0f} MB, TorchScript)")
    else:
        print(f"  {REF_TS} は見つかりません（概念のみ）。")
    print("  - .ts は torch.jit.script/trace で固めた Python 非依存の配布形。")
    print("    エッジ(Jetson 等)では `m = torch.jit.load('mobileclip_blt.ts')` で読み、")
    print("    Python の open_clip 実装が無くても encode_image/encode_text を実行できる。")
    print("  - 参照リポは src/adaptive_cluster_clip/build/models.py の load_clip_model() で")
    print("    open_clip.create_model_and_transforms(force_quick_gelu=True) により効率 CLIP を読み、")
    print("    CenterCrop を排した独自 preprocess で dense 特徴(クラスタリング用)を取り出している。")
    print("  - 本章の自前 student と同じ『L2 正規化 + logit_scale で温度付与』を共有しており、")
    print("    『小さく速い CLIP をエッジで回す』という同一目的に接続する。")
    print("  注意: 599MB の TorchScript ロードは重いため本スクリプトでは実ロードしない")
    print("        （試すなら別途 `torch.jit.load` を inference_mode で）。")


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()

    # 1) 事前学習キーの確認
    list_efficient_models()

    # 2) オフライン構築 + 効率比較（teacher vs MobileCLIP arch vs 自前 student）
    print("\n=== 効率比較（パラメータ数 / レイテンシ） ===")
    name, eff = build_offline_student()
    rows = []

    print("[load] teacher CLIP (openai/clip-vit-base-patch32) ...")
    teacher, _, device = H.load_teacher(device)
    t_params = H.count_params(teacher.vision_model) + teacher.visual_projection.weight.numel()
    rows.append(("teacher ViT-B/32 (HF)", t_params, latency(_wrap_hf(teacher), 224, device)))

    if eff is not None:
        rows.append((f"{name} arch (open_clip)", H.count_params(eff), latency(eff, eff.visual.image_size, device)))
    else:
        print("[open_clip] MobileCLIP/TinyCLIP の構築に失敗（概念のみ続行）")

    own = H.StudentCNN().to(device).eval()
    rows.append(("our student CNN (64px)", H.count_params(own), latency(own, 64, device)))

    print(f"\n  {'model':<28}{'params':>14}{'img-enc ms':>12}")
    for n_, p_, ms_ in rows:
        print(f"  {n_:<28}{p_:>14,}{ms_:>12.1f}")

    # 3) 事前学習重みのロードを試す（ネット依存・guarded）
    print()
    if name is not None:
        try_pretrained(name)

    # 4) 参照リポ接続
    reference_repo_note()

    print("\n[done] 効率 CLIP の読み方と、自前蒸留 student / 参照リポ mobileclip の接続を確認した")
    return 0


class _wrap_hf:
    """HF CLIPModel を open_clip 風の encode_image インターフェースに薄くラップ（計測用）。"""

    def __init__(self, hf_model):
        self.m = hf_model

    @torch.inference_mode()
    def encode_image(self, pixel_values):
        return self.m.get_image_features(pixel_values=pixel_values).pooler_output


if __name__ == "__main__":
    raise SystemExit(main())
