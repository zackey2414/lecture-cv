"""環境スモークテスト: 必須/任意ライブラリの導入状況・版・デバイスを表示する。

使い方:
    uv run python lectures/00_setup/check_env.py

「画像の基礎トラック(00〜09)」は numpy / opencv / pillow / matplotlib だけで動く。
深層・各タスクのトラックに進むときは `uv sync --group dl --group hf ...` で任意依存を足す。
"""

from __future__ import annotations

import platform
import sys


def _ver(modname: str) -> str:
    """モジュールを import できれば版を、できなければ理由を返す（単一責務）。"""
    try:
        mod = __import__(modname)
    except Exception as exc:  # noqa: BLE001 - 未導入も含めて状況を見せたい
        return f"× 未導入 ({exc.__class__.__name__})"
    return f"○ {getattr(mod, '__version__', '?')}"


def main() -> None:
    print("=== Python / Platform ===")
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    print("\n=== 必須（画像の基礎トラック 00-09）===")
    for name in ("numpy", "cv2", "PIL", "matplotlib"):
        print(f"  {name:12}: {_ver(name)}")

    print("\n=== 任意（深層・各タスクのトラックで使用）===")
    for name in ("torch", "torchvision", "transformers", "faiss"):
        print(f"  {name:12}: {_ver(name)}")

    # PyTorch があれば利用可能なデバイスを表示（CPU / MPS / CUDA フォールバック）。
    try:
        import torch
    except Exception:
        print("\nPyTorch 未導入のためデバイス判定はスキップ（`uv sync --group dl` で導入）")
        return

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nPyTorch のデバイス: {device}")


if __name__ == "__main__":
    main()
