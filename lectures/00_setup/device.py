"""device.py — cpu/mps/cuda 自動判定と CPU スレッド調整の共通ユーティリティ。

本講座の「環境の入口」。深層を扱う回（13 回以降）のスクリプトは、冒頭でここを
呼ぶだけで、CPU / Apple Silicon(MPS) / NVIDIA(CUDA) のどれでも **同じコード**が
動くようにする。これがこの 00 回の最重要成果物（deliverable）であり、全回で再利用する。

他モジュールからの使い方（パスを通して import するだけ）:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_setup"))
    from device import pick_device, configure_threads
    device = pick_device()        # cpu/mps/cuda を自動選択（cuda > mps > cpu）
    configure_threads()           # CPU スレッド数を妥当値に固定
    model = model.to(device)      # 以降は device 非依存で書ける

単体実行で環境診断を表示:
    uv run python lectures/00_setup/device.py
"""

from __future__ import annotations

import os
import pathlib

import torch


# =====================================================================
# デバイス判定（cuda > mps > cpu の優先順）
# =====================================================================

def available_devices() -> list[str]:
    """この環境で使えるアクセラレータ名を優先度順で返す（cpu は常に末尾に含む）。

    判定は「速い順」= cuda → mps → cpu。cpu は最後の砦として常に使えるので必ず含める。
    GPU 判定 API はバックエンドごとに別物なので、ここで一箇所に閉じ込めておく。
    """
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    # mps は古い torch / 非 mac では backends.mps 自体が無いので getattr で安全に確認する。
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def pick_device(prefer: str | None = None) -> torch.device:
    """cpu/mps/cuda を自動判定して torch.device を返す（全回で使う定石）。

    - 既定は優先度順（cuda > mps > cpu）の先頭。
    - prefer に "cpu"/"cuda"/"mps" を渡すと、それが利用可能ならそれを優先する。
      利用不可なら黙って自動判定へフォールバックする（落とさない）。
    - mps を選んだ場合は、未対応演算を CPU に逃がす保険(env)も自動で有効化する。
    """
    devices = available_devices()
    # ユーザ希望が使えるならそれを尊重（早期確定）
    if prefer is not None and prefer in devices:
        chosen = prefer
    else:
        chosen = devices[0]  # 優先度順で先頭が最良
    if chosen == "mps":
        enable_mps_fallback()
    return torch.device(chosen)


def enable_mps_fallback() -> None:
    """Apple Silicon(MPS) で未実装の演算に当たったとき CPU へ自動フォールバックさせる。

    PYTORCH_ENABLE_MPS_FALLBACK=1 を立てておくと、MPS 未対応の演算で
    NotImplementedError を投げて止まる代わりに、その演算だけ CPU で実行される。
    MPS は「多くは速い、一部は未対応」が現実なので、学習/推論を止めないための保険。
    既に設定済みなら尊重する（setdefault）。
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# =====================================================================
# CPU スレッド調整（torch と OpenMP を揃える）
# =====================================================================

def configure_threads(num_threads: int | None = None) -> int:
    """CPU の計算スレッド数を決めて torch と OpenMP に反映し、実際に使う数を返す。

    - num_threads=None: 環境変数 OMP_NUM_THREADS があればそれ、無ければ
      os.cpu_count() を上限に妥当値へ丸める。
    - torch.set_num_threads と OMP_NUM_THREADS の **両方**を揃えるのが肝心。
      片方だけ設定すると numpy/cv2(OpenMP) と torch がスレッド数で食い違い、
      オーバーサブスクライブ（コア数以上のスレッド乱立）でかえって遅くなる。
    """
    n = _resolve_thread_count(num_threads)
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    return n


def _resolve_thread_count(num_threads: int | None) -> int:
    """スレッド数を 1..cpu_count の範囲へ正規化する（単一責務の補助）。"""
    cpu = os.cpu_count() or 1
    if num_threads is None:
        env = os.environ.get("OMP_NUM_THREADS")
        num_threads = int(env) if env and env.isdigit() else cpu
    # 1 未満（0/負）や cpu 超過は事故の元なのでクランプする。
    return max(1, min(int(num_threads), cpu))


# =====================================================================
# HuggingFace キャッシュ
# =====================================================================

def hf_home() -> pathlib.Path:
    """HuggingFace のキャッシュ場所(HF_HOME)を返す（未設定なら既定 ~/.cache/huggingface）。

    モデル重みはここに落ちる。Docker では HF_HOME をボリュームにマウントして永続化すると、
    コンテナを作り直すたびの再ダウンロードを避けられる。HF_HUB_OFFLINE=1 を併用すると、
    キャッシュ済みの重みだけでネット非依存に動かせる（CI/オフライン検証で有用）。
    """
    env = os.environ.get("HF_HOME")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "huggingface"


# =====================================================================
# 診断
# =====================================================================

def summary() -> dict:
    """環境の要約を dict で返す（ログ/JSON 化しやすい形）。"""
    dev = pick_device()
    return {
        "torch": torch.__version__,
        "available": available_devices(),
        "selected": str(dev),
        "num_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "hf_home": str(hf_home()),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "0"),
    }


def main() -> None:
    """単体実行時: スレッドを構成し、判定結果を一覧表示する。"""
    n = configure_threads()
    info = summary()
    print("=== device / 環境ユーティリティ ===")
    print(f"torch          : {info['torch']}")
    print(f"使えるデバイス : {info['available']}")
    print(f"選択デバイス   : {info['selected']}")
    print(f"スレッド数     : {n}  (torch.get_num_threads={info['num_threads']}, cpu={info['cpu_count']})")
    print(f"HF_HOME        : {info['hf_home']}")
    print(f"HF_HUB_OFFLINE : {info['hf_hub_offline']}")
    print("\nこの device.py を全回で import して『環境の入口』を 1 つに統一する。")


if __name__ == "__main__":
    main()
