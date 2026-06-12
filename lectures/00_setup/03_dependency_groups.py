"""03: uv の依存グループ運用と opencv の排他を読み解く。

実行:
    uv run python lectures/00_setup/03_dependency_groups.py

学ぶこと:
  - pyproject.toml の [project.dependencies]（常に入る本体）と
    [dependency-groups]（uv sync --group で足す任意グループ）の違い
  - PyTorch CPU ホイールを引く [[tool.uv.index]] + [tool.uv.sources] の仕組み
  - opencv-python（full）と opencv-python-headless の排他をどう検出するか
  - この 00 回が必要とするグループ（dl）の確認

pyproject.toml を tomllib（標準ライブラリ）で読むだけ。ネット不要・CPU で完走。
"""

from __future__ import annotations

import pathlib
import sys
import tomllib
from importlib import metadata

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict:
    """プロジェクトルートの pyproject.toml を辞書として読む（単一責務）。"""
    path = PROJECT_ROOT / "pyproject.toml"
    with path.open("rb") as f:  # tomllib はバイナリで開く
        return tomllib.load(f)


def detect_opencv_variant() -> str:
    """インストール済みの opencv パッケージから variant を判定する。

    返り値: "headless" / "full" / "conflict"（両方入っている）/ "none"。
    両方を同時に入れると cv2 名前空間が衝突するため、必ずどちらか一方にする。
    """
    installed = {dist.metadata["Name"].lower() for dist in metadata.distributions()}
    has_full = "opencv-python" in installed
    has_headless = "opencv-python-headless" in installed
    if has_full and has_headless:
        return "conflict"
    if has_headless:
        return "headless"
    if has_full:
        return "full"
    return "none"


def main() -> None:
    data = _load_pyproject()

    # --- 1. 本体依存（常に入る） --------------------------------------
    print("=== 1. [project.dependencies]（uv sync で常に入る本体）===")
    for dep in data["project"]["dependencies"]:
        print(f"  - {dep}")
    print("  → 画像の基礎トラック(00〜09)はこの本体だけで CPU 完走できる。")

    # --- 2. 任意の依存グループ ----------------------------------------
    print("\n=== 2. [dependency-groups]（uv sync --group <name> で足す）===")
    groups: dict[str, list[str]] = data.get("dependency-groups", {})
    for name, pkgs in groups.items():
        print(f"  [{name:9s}] {len(pkgs):2d} 個: {', '.join(p.split('>')[0].split('=')[0] for p in pkgs)}")
    print("  → 例: 深層の土台は `uv sync --group dl`（torch/torchvision）。")

    # --- 3. PyTorch CPU インデックス ----------------------------------
    print("\n=== 3. PyTorch CPU ホイールの引き方 ===")
    indexes = data.get("tool", {}).get("uv", {}).get("index", [])
    for idx in indexes:
        print(f"  [[tool.uv.index]] name={idx.get('name')} explicit={idx.get('explicit')}")
        print(f"                    url={idx.get('url')}")
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    for pkg, rule in sources.items():
        print(f"  [tool.uv.sources] {pkg} -> {rule}")
    print("  → Linux は巨大な CUDA 版を避け CPU ホイールを明示。mac は PyPI 既定(CPU/MPS)。")

    # --- 4. この回が必要とするグループ --------------------------------
    needs = {"dl"}
    print("\n=== 4. この 00 回が必要とするグループ ===")
    present = set(groups)
    for g in sorted(needs):
        mark = "○ 定義あり" if g in present else "× 未定義"
        print(f"  needs '{g}': {mark}  → `uv sync --group {g}`")

    # --- 5. opencv の排他検出 -----------------------------------------
    variant = detect_opencv_variant()
    print("\n=== 5. opencv の排他（headless / full はどちらか一方）===")
    print(f"  検出された variant: {variant}")
    if variant == "conflict":
        print("  ⚠ full と headless が同居している。cv2 名前空間が衝突するので片方を外すこと。")
    elif variant == "headless":
        print("  ○ headless。Docker/CI 向けの既定。imshow は使えない（保存で確認する）。")
    elif variant == "full":
        print("  ○ full。ローカルで cv2.imshow が使える。ただし本講座は保存運用が既定。")
    else:
        print("  × opencv 未導入。`uv sync` で本体依存を入れる。")


if __name__ == "__main__":
    sys.exit(main())
