"""01: device 自動判定ユーティリティの使い方（全回で使う定石）。

実行:
    uv run python lectures/00_setup/01_device_util.py

学ぶこと:
  - cpu/mps/cuda をどう自動判定するか（device.pick_device）
  - 選んだ device に Tensor / Module を載せる正準形（.to(device)）
  - 「同じコードが CPU でも GPU でも動く」書き方の型
  - prefer 指定が使えない時に黙ってフォールバックする挙動

このスクリプトは GPU が無い CPU 環境でも完走する（cpu が選ばれるだけ）。
"""

from __future__ import annotations

import pathlib
import sys

import torch
from torch import nn

# 同じディレクトリの device.py を import できるようにパスを通す（定型）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from device import available_devices, pick_device  # noqa: E402


class TinyNet(nn.Module):
    """device 非依存で書けることを示すための極小モデル（学習はしない）。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main() -> None:
    print("=== 1. 使えるデバイスの一覧（優先度順 cuda > mps > cpu）===")
    print(f"  available_devices() = {available_devices()}")

    # --- 2. 自動判定 ---------------------------------------------------
    # 何も指定しなければ最良のデバイスが選ばれる。CPU 環境なら 'cpu'。
    device = pick_device()
    print(f"\n=== 2. 自動判定 ===\n  pick_device() -> {device}")

    # --- 3. prefer 指定とフォールバック --------------------------------
    # 使えないデバイスを希望しても落ちず、自動判定にフォールバックする。
    print("\n=== 3. prefer 指定（使えなければフォールバック）===")
    for want in ("cpu", "cuda", "mps"):
        got = pick_device(prefer=want)
        note = "希望どおり" if str(got) == want else "使えないのでフォールバック"
        print(f"  prefer={want:5s} -> {got}  ({note})")

    # --- 4. Tensor / Module を device に載せる定石 ---------------------
    # ここがポイント: 入力もモデルも同じ device に置く。片方だけだと
    # 'Expected all tensors to be on the same device' で落ちる。
    print("\n=== 4. Tensor / Module を device に載せる ===")
    x = torch.randn(3, 4).to(device)          # 入力を device へ
    model = TinyNet().to(device)              # モデルも同じ device へ
    with torch.inference_mode():              # 推論は勾配不要 → inference_mode で軽く
        y = model(x)
    print(f"  入力 x.device  = {x.device}")
    print(f"  重み .device   = {next(model.parameters()).device}")
    print(f"  出力 y.shape   = {tuple(y.shape)}  y.device = {y.device}")

    # --- 5. CPU フォールバックの一般形 --------------------------------
    # .to(device) は CPU では実質ノーオペ。だからこの 1 本のコードが
    # CPU/MPS/CUDA すべてで動く。これが「環境の入口を 1 つにする」効能。
    print("\n=== 5. まとめ ===")
    print("  入力・モデルを同じ device に載せれば、コードは device 非依存になる。")
    print("  全回でこの device.py を import し、環境差をここ 1 箇所に閉じ込める。")


if __name__ == "__main__":
    main()
