"""第0回 演習問題（環境構築 / device 判定・スレッド・依存グループ・キャッシュ）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点（未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0）:
         uv run python lectures/00_setup/exercises.py
  3. 全問の模範解答（全 PASS）はこちら:
         uv run python lectures/00_setup/exercises_solutions.py

テーマ（易→難）:
  1 デバイス選択ロジック    2 スレッド数クランプ      3 opencv variant 判定
  4 HF 環境変数の組み立て   5 MPS フォールバック env  6 依存グループのパッケージ数集計
  7 不足グループの抽出      8 device 上のテンソル計算

採点は純粋な入出力（一部だけ torch）で決定的に行うので速く再現的。
本物の使い方は device.py・01〜03・mini_project.py を参照。
"""

from __future__ import annotations

# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_select_device(available: list[str], prefer: str | None) -> str:
    """演習1: 使えるデバイス一覧と希望から、実際に使うデバイス名を返す。

    ルール:
      - prefer が available にあれば prefer を採用。
      - 無ければ優先度 ["cuda", "mps", "cpu"] の順で available にある最初を返す。
    例: select(["cuda","mps","cpu"], None) -> "cuda" / select(["mps","cpu"], "cuda") -> "mps"
    """
    # TODO: prefer が使えるならそれ、なければ優先度順で available に最初に現れるものを返す
    raise NotImplementedError


def ex2_clamp_threads(requested: int, cpu_count: int) -> int:
    """演習2: 要求スレッド数を 1..cpu_count の範囲にクランプして返す。

    0 や負数は 1 に、cpu_count 超過は cpu_count に丸める。
    例: clamp(100, 8) -> 8 / clamp(0, 8) -> 1 / clamp(4, 8) -> 4
    """
    # TODO: max(1, min(requested, cpu_count)) を返す
    raise NotImplementedError


def ex3_opencv_variant(installed: list[str]) -> str:
    """演習3: インストール済みパッケージ名一覧から opencv の variant を判定する。

    返り値: "conflict"（両方）/ "headless" / "full" / "none"。
    対象名: "opencv-python"（full）, "opencv-python-headless"（headless）。
    両方入っていると cv2 名前空間が衝突する（=conflict）。
    """
    # TODO: installed に full / headless が含まれるかで 4 通りを返す
    raise NotImplementedError


def ex4_build_hf_env(hf_home: str, offline: bool) -> dict[str, str]:
    """演習4: HuggingFace 用の環境変数 dict を組み立てて返す。

    返り値: {"HF_HOME": hf_home, "HF_HUB_OFFLINE": "1" if offline else "0"}。
    値は必ず文字列（環境変数は文字列）にすること。
    """
    # TODO: HF_HOME と HF_HUB_OFFLINE（"1"/"0"）を入れた dict を返す
    raise NotImplementedError


def ex5_mps_fallback_env(device: str) -> dict[str, str]:
    """演習5: device が "mps" のときだけ MPS フォールバック env を返す。

    返り値: device=="mps" なら {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}、それ以外は {}。
    """
    # TODO: device が "mps" のときだけ env を返し、他は空 dict を返す
    raise NotImplementedError


def ex6_count_group_packages(pyproject_text: str) -> dict[str, int]:
    """演習6: pyproject.toml の文字列から [dependency-groups] の各グループの個数を返す。

    例: {"dl": 2, "viz": 1, ...}。tomllib.loads でパースし、各グループのリスト長を数える。
    [dependency-groups] が無ければ空 dict。
    """
    # TODO: tomllib.loads(pyproject_text) して "dependency-groups" を取り、各値の len を集計
    raise NotImplementedError


def ex7_missing_groups(required: list[str], installed: list[str]) -> list[str]:
    """演習7: 必要グループのうち未導入のものを、ソート済みリストで返す。

    例: missing(["dl","hf"], ["dl"]) -> ["hf"]。重複は除き、昇順で返す。
    """
    # TODO: required にあって installed に無いものを sorted(set(...)) で返す
    raise NotImplementedError


def ex8_tensor_device(device: str) -> str:
    """演習8: 指定 device 上にテンソルを作り、その device 種別("cpu"/"cuda"/"mps")を返す。

    torch.tensor(...).to(torch.device(device)) の .device.type を文字列で返す。
    cpu は常に成功する（GPU 不要で採点できる）。
    """
    # TODO: torch でテンソルを作って device に載せ、.device.type を返す
    raise NotImplementedError


# =====================================================================
# 採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

_SAMPLE_PYPROJECT = """
[project]
name = "demo"
dependencies = ["numpy", "pillow"]

[dependency-groups]
dl = ["torch", "torchvision"]
viz = ["matplotlib"]
hf = ["transformers", "huggingface-hub", "safetensors"]
"""


def grade(funcs: dict) -> bool:
    """funcs（名前→関数）を採点して PASS/FAIL を表示。全 PASS なら True。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def c1():
        a = funcs["ex1"](["cuda", "mps", "cpu"], None) == "cuda"
        b = funcs["ex1"](["mps", "cpu"], "cuda") == "mps"   # prefer 不可 → 優先度順
        c = funcs["ex1"](["cuda", "mps", "cpu"], "mps") == "mps"
        d = funcs["ex1"](["cpu"], None) == "cpu"
        return (a and b and c and d, "優先度順 + prefer フォールバック")

    def c2():
        ok = (funcs["ex2"](100, 8) == 8 and funcs["ex2"](0, 8) == 1
              and funcs["ex2"](4, 8) == 4 and funcs["ex2"](-3, 8) == 1)
        return (ok, "1..cpu_count にクランプ")

    def c3():
        a = funcs["ex3"](["opencv-python", "opencv-python-headless"]) == "conflict"
        b = funcs["ex3"](["opencv-python-headless", "numpy"]) == "headless"
        c = funcs["ex3"](["opencv-python"]) == "full"
        d = funcs["ex3"](["numpy"]) == "none"
        return (a and b and c and d, "headless/full/conflict/none")

    def c4():
        on = funcs["ex4"]("/c/hf", True)
        off = funcs["ex4"]("/c/hf", False)
        ok = (on == {"HF_HOME": "/c/hf", "HF_HUB_OFFLINE": "1"}
              and off == {"HF_HOME": "/c/hf", "HF_HUB_OFFLINE": "0"})
        return (ok, "HF_HOME + HF_HUB_OFFLINE(文字列)")

    def c5():
        ok = (funcs["ex5"]("mps") == {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
              and funcs["ex5"]("cpu") == {} and funcs["ex5"]("cuda") == {})
        return (ok, "mps のときだけ fallback env")

    def c6():
        counts = funcs["ex6"](_SAMPLE_PYPROJECT)
        ok = counts == {"dl": 2, "viz": 1, "hf": 3}
        return (ok, f"counts={counts}")

    def c7():
        a = funcs["ex7"](["dl", "hf"], ["dl"]) == ["hf"]
        b = funcs["ex7"](["dl"], ["dl"]) == []
        c = funcs["ex7"](["hf", "dl", "hf"], []) == ["dl", "hf"]  # 重複除去 + 昇順
        return (a and b and c, "不足グループ抽出(重複除去・昇順)")

    def c8():
        t = funcs["ex8"]("cpu")
        return (t == "cpu", f"device.type={t}")

    check("ex1_select_device", c1)
    check("ex2_clamp_threads", c2)
    check("ex3_opencv_variant", c3)
    check("ex4_build_hf_env", c4)
    check("ex5_mps_fallback_env", c5)
    check("ex6_count_group_packages", c6)
    check("ex7_missing_groups", c7)
    check("ex8_tensor_device", c8)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
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
    grade(current_funcs())
