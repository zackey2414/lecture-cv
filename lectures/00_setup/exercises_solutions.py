"""第0回 演習の模範解答（全 PASS）。

実行:
    uv run python lectures/00_setup/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま import して使い、
ここでは各 exN_*() を「正しく実装した版」に差し替えて採点する。
まずは自力で exercises.py を解いてから答え合わせに使うこと。
"""

from __future__ import annotations

import pathlib
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  （採点ランナーを再利用）

_PRIORITY = ["cuda", "mps", "cpu"]  # 速い順


# =====================================================================
# 模範解答
# =====================================================================

def ex1_select_device(available: list[str], prefer: str | None) -> str:
    """prefer が使えるならそれ、無ければ優先度順で available にある最初を返す。"""
    if prefer is not None and prefer in available:
        return prefer
    for name in _PRIORITY:
        if name in available:
            return name
    return "cpu"  # 念のための最後の砦


def ex2_clamp_threads(requested: int, cpu_count: int) -> int:
    """1..cpu_count にクランプ。0/負は 1、超過は cpu_count。"""
    return max(1, min(requested, cpu_count))


def ex3_opencv_variant(installed: list[str]) -> str:
    """full / headless の有無で 4 通りを判定する。"""
    names = set(installed)
    has_full = "opencv-python" in names
    has_headless = "opencv-python-headless" in names
    if has_full and has_headless:
        return "conflict"
    if has_headless:
        return "headless"
    if has_full:
        return "full"
    return "none"


def ex4_build_hf_env(hf_home: str, offline: bool) -> dict[str, str]:
    """HF_HOME と HF_HUB_OFFLINE（文字列 "1"/"0"）を組み立てる。"""
    return {"HF_HOME": hf_home, "HF_HUB_OFFLINE": "1" if offline else "0"}


def ex5_mps_fallback_env(device: str) -> dict[str, str]:
    """mps のときだけ未対応演算を CPU に逃がす env を返す。"""
    if device == "mps":
        return {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    return {}


def ex6_count_group_packages(pyproject_text: str) -> dict[str, int]:
    """[dependency-groups] の各グループのパッケージ数を集計する。"""
    data = tomllib.loads(pyproject_text)
    groups = data.get("dependency-groups", {})
    return {name: len(pkgs) for name, pkgs in groups.items()}


def ex7_missing_groups(required: list[str], installed: list[str]) -> list[str]:
    """必要だが未導入のグループを重複除去・昇順で返す。"""
    return sorted(set(required) - set(installed))


def ex8_tensor_device(device: str) -> str:
    """指定 device 上にテンソルを作り、その device 種別を返す。"""
    import torch

    t = torch.zeros(2, 2).to(torch.device(device))
    return t.device.type


def solution_funcs() -> dict:
    return {
        "ex1": ex1_select_device,
        "ex2": ex2_clamp_threads,
        "ex3": ex3_opencv_variant,
        "ex4": ex4_build_hf_env,
        "ex5": ex5_mps_fallback_env,
        "ex6": ex6_count_group_packages,
        "ex7": ex7_missing_groups,
        "ex8": ex8_tensor_device,
    }


if __name__ == "__main__":
    grade(solution_funcs())
