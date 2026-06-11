"""第12回 演習問題（PyTorch 画像テンソルとデータ拡張）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出るが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ）。
  2. 実装できたら自己採点:
         uv run python lectures/12_data_pipeline_augmentation/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/12_data_pipeline_augmentation/exercises.py

この5問は、本モジュールの核を1つずつ抜き出したもの:
  ex1: HWC uint8 → CHW float[0,1] への手書き変換（並び替え＋スケール）   … 01
  ex2: 正規化 (x-mean)/std をチャンネルごとに手書き                      … 01
  ex3: 逆正規化 x*std+mean（可視化のために元へ戻す）                     … 01
  ex4: 推論用の“決定論”transform を v2 で組む                           … 02
  ex5: class/xxx.png フォルダを走査して (path,label) 一覧を作る           … 02
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_to_chw_float(np_hwc_uint8: np.ndarray) -> torch.Tensor:
    """演習1: HWC・uint8(0..255) の numpy 画像を、CHW・float32(0..1) のテンソルにする。

    手順:
      1. torch.from_numpy で numpy → Tensor（この時点では HWC・uint8）。
      2. .permute(2, 0, 1) で (H,W,C) → (C,H,W) に並び替える。
      3. .float() で float32 にし、255.0 で割って [0,1] にする。
    返り値は dtype=torch.float32、shape=(C,H,W)、値域 [0,1]。
    """
    # TODO: 上の3手順を実装する
    raise NotImplementedError


def ex2_normalize(t_chw_float: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """演習2: CHW float[0,1] テンソルを (x-mean)/std でチャンネルごとに正規化する。

    mean, std は長さ C のタプル。チャンネル方向にブロードキャストするため、
    torch.tensor(mean).view(-1, 1, 1) のように (C,1,1) へ整形してから引き算/割り算する。
    """
    # TODO: mean_t, std_t を (C,1,1) に整形し、(t - mean_t) / std_t を返す
    raise NotImplementedError


def ex3_denormalize(t_norm: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """演習3: 正規化の逆 x*std + mean で [0,1] スケールに戻す（可視化のため）。

    ex2 と同じく mean/std を (C,1,1) に整形し、t*std + mean を返す。
    """
    # TODO: t_norm * std_t + mean_t を返す
    raise NotImplementedError


def ex4_build_eval_transform(size: int):
    """演習4: 推論/評価用の“決定論的”な v2.Compose を組んで返す。

    要件（ランダム要素を入れないこと＝同じ入力なら毎回同じ出力）:
      v2.ToImage()
      v2.Resize((size, size), antialias=True)
      v2.ToDtype(torch.float32, scale=True)
      v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    これを v2.Compose([...]) で包んで返す。
    """
    # TODO: torchvision.transforms.v2 を import し、上の4段の Compose を返す
    raise NotImplementedError


def ex5_scan_folder(root: pathlib.Path) -> tuple[list[tuple[pathlib.Path, int]], dict]:
    """演習5: root/<class>/*.png を走査し (画像パス, ラベル) 一覧と class_to_idx を返す。

    手順:
      1. root 直下のサブフォルダ名を sorted で取り、0,1,2... のラベルを振る（class_to_idx）。
      2. 各クラスの *.png を sorted で集め、(path, label) を samples に追加する。
    返り値は (samples, class_to_idx)。samples は『クラス順→ファイル名順』に並ぶこと。
    """
    # TODO: classes を sorted で取得 → class_to_idx を作る → 各クラスの png を sorted で集める
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _make_temp_dataset() -> pathlib.Path:
    """採点用に class/xxx.png レイアウトの一時フォルダを作る（クラス名は b,a,c）。"""
    from PIL import Image

    root = pathlib.Path(tempfile.mkdtemp(prefix="ex5_"))
    for name in ("b", "a", "c"):  # わざと非ソート順で作る → 実装側で sorted されるか確認
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for i in (1, 0):  # わざと逆順 → ファイル名順に並ぶか確認
            Image.new("RGB", (8, 8)).save(d / f"img_{i}.png")
    return root


def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 並び替え＋スケール。255 の画素は 1.0、0 の画素は 0.0 に。
    def _c1():
        arr = np.array([[[0, 128, 255]]], dtype=np.uint8)  # (H=1,W=1,C=3)
        got = ex1_to_chw_float(arr)
        ref = _sol_ex1(arr)
        ok = (
            isinstance(got, torch.Tensor)
            and got.dtype == torch.float32
            and tuple(got.shape) == (3, 1, 1)
            and torch.allclose(got, ref, atol=1e-6)
        )
        return ok, f"shape={tuple(got.shape) if isinstance(got, torch.Tensor) else '-'}"
    check("ex1_to_chw_float", _c1)

    # ex2: 正規化。手計算と一致するか。
    def _c2():
        t = torch.full((3, 2, 2), 0.5)
        got = ex2_normalize(t, _MEAN, _STD)
        ref = _sol_ex2(t, _MEAN, _STD)
        return isinstance(got, torch.Tensor) and torch.allclose(got, ref, atol=1e-6), \
            "(x-mean)/std"
    check("ex2_normalize", _c2)

    # ex3: 逆正規化は ex2 を打ち消す（round-trip で元に戻る）。
    def _c3():
        t = torch.rand(3, 4, 4)
        norm = _sol_ex2(t, _MEAN, _STD)
        got = ex3_denormalize(norm, _MEAN, _STD)
        return isinstance(got, torch.Tensor) and torch.allclose(got, t, atol=1e-5), \
            "denorm(norm(x)) == x"
    check("ex3_denormalize", _c3)

    # ex4: 決定論（2回適用が一致）・shape・正規化されている。
    def _c4():
        from PIL import Image
        tf = ex4_build_eval_transform(64)
        img = Image.fromarray(np.random.randint(0, 255, (50, 70, 3), dtype=np.uint8))
        a = tf(img)
        b = tf(img)
        ok = (
            isinstance(a, torch.Tensor)
            and tuple(a.shape) == (3, 64, 64)
            and a.dtype == torch.float32
            and torch.allclose(a, b)        # 決定論
            and float(a.min()) < 0.0        # 正規化されていれば負値を含む
        )
        return ok, f"shape={tuple(a.shape) if isinstance(a, torch.Tensor) else '-'}, deterministic"
    check("ex4_build_eval_transform", _c4)

    # ex5: クラスは sorted で a,b,c、samples はクラス順→ファイル名順。
    def _c5():
        root = _make_temp_dataset()
        samples, c2i = ex5_scan_folder(root)
        ref_samples, ref_c2i = _sol_ex5(root)
        ok = (
            c2i == ref_c2i == {"a": 0, "b": 1, "c": 2}
            and [(p.name, lb) for p, lb in samples] == [(p.name, lb) for p, lb in ref_samples]
            and len(samples) == 6
        )
        return ok, f"classes={list(c2i.keys()) if isinstance(c2i, dict) else '-'}, n={len(samples)}"
    check("ex5_scan_folder", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。
# =====================================================================

def _sol_ex1(np_hwc_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np_hwc_uint8)        # HWC, uint8
    t = t.permute(2, 0, 1)                     # HWC -> CHW
    return t.float() / 255.0                   # 0..255 -> 0..1


def _sol_ex2(t_chw_float: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=t_chw_float.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=t_chw_float.dtype).view(-1, 1, 1)
    return (t_chw_float - mean_t) / std_t


def _sol_ex3(t_norm: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=t_norm.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=t_norm.dtype).view(-1, 1, 1)
    return t_norm * std_t + mean_t


def _sol_ex4(size: int):
    from torchvision.transforms import v2
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_MEAN, std=_STD),
    ])


def _sol_ex5(root: pathlib.Path) -> tuple[list[tuple[pathlib.Path, int]], dict]:
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples: list[tuple[pathlib.Path, int]] = []
    for name in classes:
        for path in sorted((root / name).glob("*.png")):
            samples.append((path, class_to_idx[name]))
    return samples, class_to_idx


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_to_chw_float"] = _sol_ex1
    g["ex2_normalize"] = _sol_ex2
    g["ex3_denormalize"] = _sol_ex3
    g["ex4_build_eval_transform"] = _sol_ex4
    g["ex5_scan_folder"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
