"""第2回 演習問題（画像・動画処理ライブラリの地図）。

この回は「概念回」なので、演習も「ライブラリ間の橋渡し」と「選び方」を中心に置きます。
新しい画像処理アルゴリズムを覚えるより、

  - cv2(BGR) ↔ PIL(RGB) ↔ numpy ↔ Tensor の相互変換を淀みなく書けること
  - 「幅・高さ」と「shape の (H, W)」「Tensor の (C, H, W)」の軸順を取り違えないこと
  - uint8 のオーバーフロー/飽和、補間の使い分けなど「ライブラリの癖」を押さえること
  - 課題に応じて適切なライブラリを選べること（地図の暗記）

を体に入れるのが目的です。易しい問題（ex1〜）から難しい問題（〜ex10）へ並べてあります。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/02_cv_libraries_overview/exercises.py
     全問が pass すれば "ALL PASS" と表示される。未実装でも例外で落ちず exit 0。
  3. どうしても分からない時は、模範解答の挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/02_cv_libraries_overview/exercises.py
     模範解答そのものは exercises_solutions.py にもあります（まずは自力で！）。

ヒント: サンプル画像は cv_helpers.get_sample_bgr() で得られる（BGR uint8）。
出力の保存先は cv_helpers.output_dir() を使うと outputs/02_cv_libraries_overview/ になる。
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir  # noqa: E402


# =====================================================================
# 演習（ここを実装する）— 易 → 難
# =====================================================================

def ex1_bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """演習1【易】OpenCV の BGR 配列を、色が正しい PIL.Image（RGB）に変換して返す。

    ポイント: cv2 は BGR、PIL は RGB。fromarray に渡す前に BGR→RGB へ変換する。
    変換を忘れると赤と青が入れ替わった画像になる。
    """
    # TODO: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) してから Image.fromarray で PIL 化して返す
    raise NotImplementedError


def ex2_pil_size(rgb_array: np.ndarray) -> tuple[int, int]:
    """演習2【易】RGB の numpy 配列 (H, W, 3) から、PIL の size タプル (幅 W, 高さ H) を返す。

    ポイント: numpy の shape は (H, W, 3)、PIL の size は (W, H) で軸順が逆。
    実際に Image.fromarray して .size を返しても、shape から組み立ててもよい。
    """
    # TODO: (幅, 高さ) すなわち (shape[1], shape[0]) を返す（PIL.size と一致させる）
    raise NotImplementedError


def ex3_resize_cv2(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """演習3【易】cv2.resize で (width, height) のサイズへ縮小し、結果を返す。

    ポイント: cv2.resize の dsize は (幅, 高さ) の順。shape の (高さ, 幅) と逆なので、
    返り値の shape は必ず (height, width, 3) になること。縮小なので INTER_AREA を使う。
    """
    # TODO: cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA) を返す
    raise NotImplementedError


def ex4_hflip_brighten(rgb: np.ndarray, beta: int) -> np.ndarray:
    """演習4【中】最小の自作拡張。左右反転してから明るさを +beta する（飽和させる）。

    ポイント: 反転は numpy スライス [:, ::-1]。明るさ加算は uint8 のままだと
    オーバーフローするので float で計算 → np.clip(..,0,255) → uint8 に戻す。
    返り値の shape は入力と同じであること。
    """
    # TODO: flipped = rgb[:, ::-1]; out = clip(flipped.astype(float32)+beta,0,255).astype(uint8)
    raise NotImplementedError


def ex5_pick_library(task: str) -> str:
    """演習5【易】課題キーワードから「まず選ぶライブラリ名」を返す（地図の暗記）。

    対応表（この通りに返すこと）:
      "differentiable_gpu_aug" -> "kornia"          # 学習ループ内・GPU・微分可能な拡張
      "bbox_mask_aug"          -> "albumentations"  # 検出/セグメ用に bbox/mask ごと拡張
      "torch_training_aug"     -> "torchvision"     # PyTorch 学習の前処理/拡張(v2)
      "intuitive_edit"         -> "Pillow"          # 直感的な画像編集・フォント描画
      "fast_classic_cv"        -> "OpenCV"          # 高速な古典CV・動画I/O
    上記以外のキーは "OpenCV"（迷ったらまず OpenCV）を返す。
    """
    # TODO: 上の対応表を dict で持ち、task に対応する値を返す（未知キーは "OpenCV"）
    raise NotImplementedError


def ex6_to_chw(rgb: np.ndarray) -> np.ndarray:
    """演習6【中】RGB の (H, W, 3) 配列を、PyTorch Tensor 流の (3, H, W) 軸順へ並べ替える。

    ポイント: numpy/OpenCV は (H, W, C)、PyTorch の Tensor は (C, H, W)。
    値は変えず、軸の順番だけを (2, 0, 1) に入れ替える（np.transpose / ndarray.transpose）。
    返り値の shape は (3, H, W) になること。
    """
    # TODO: np.transpose(rgb, (2, 0, 1)) を返す（チャンネルを先頭へ）
    raise NotImplementedError


def ex7_saturating_add(rgb: np.ndarray, value: int) -> np.ndarray:
    """演習7【中】uint8 画像に value を「飽和加算」する（cv2.add と同じ 255 頭打ち）。

    ポイント: numpy の素の `rgb + value` は uint8 のままだと 255 を超えた分が
    0 へ回り込む（ラップアラウンド）。飽和（255 で頭打ち）にするには float などへ
    広げてから加算 → np.clip(0, 255) → uint8 に戻す。負の value（減算）も clip で 0 止め。
    返り値の shape/dtype は入力と同じ（uint8）であること。
    """
    # TODO: clip(rgb.astype(int16)+value, 0, 255).astype(uint8) を返す
    raise NotImplementedError


def ex8_resize_long_side(bgr: np.ndarray, long_side: int) -> np.ndarray:
    """演習8【中】アスペクト比を保ったまま「長辺＝long_side」になるよう縮小して返す。

    ポイント: 元の (H, W) のうち長い方を long_side に合わせ、短い方は比率で決める。
    新しいサイズは四捨五入（round）で整数化。dsize は (幅, 高さ) 順に渡すこと。
    縮小前提なので補間は INTER_AREA。返り値の長辺は long_side になる。
    （元画像が long_side 以下なら拡大はせず、そのまま返してよい。）
    """
    # TODO: scale = long_side / max(H, W); 拡大不要なら return bgr;
    #       new_w, new_h = round(W*scale), round(H*scale);
    #       cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    raise NotImplementedError


def ex9_roundtrip_lossless(bgr: np.ndarray) -> np.ndarray:
    """演習9【中】cv2(BGR) → PIL(RGB) → numpy → cv2(BGR) の往復をして元配列を再現する。

    ポイント: 行きは BGR→RGB してから PIL 化、帰りは RGB→BGR に戻す。色順を正しく
    入れれば、往復しても画素は完全一致する（戻り値は入力 bgr と array_equal になる）。
    どこかで cvtColor を入れ忘れると赤青が入れ替わって不一致になる。
    """
    # TODO: rgb=cvtColor(bgr,BGR2RGB); pil=Image.fromarray(rgb);
    #       back=np.asarray(pil); return cvtColor(back, RGB2BGR)
    raise NotImplementedError


def ex10_reproducible_aug(rgb: np.ndarray, seed: int) -> np.ndarray:
    """演習10【難】seed から決まる「再現可能な」拡張を 1 回かけて返す。

    要件:
      - 乱数は np.random.default_rng(seed) だけを使う（同じ seed なら毎回同じ出力）。
      - 内容は最低限「coin（rng.random()<0.5）で左右反転」＋「明るさ +beta（飽和）」。
        beta は rng.integers(-50, 51) のように seed 依存で決める（毎回必ず適用）。
      - 返り値の shape は入力と同じ・dtype は uint8。
    ねらい: 拡張の「確率的だが seed で固定できる（=実験を再現できる）」性質を体得する。
    同じ seed では一致し、異なる seed では（多くの場合）異なる出力になる。
    """
    # TODO: rng=np.random.default_rng(seed);
    #       out = rgb[:, ::-1].copy() if rng.random()<0.5 else rgb;
    #       beta=int(rng.integers(-50,51));
    #       return clip(out.astype(int16)+beta,0,255).astype(uint8)
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    bgr = get_sample_bgr()
    output_dir()  # 出力先を作っておく（演習で保存したくなった時のため）
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: BGR→PIL(RGB)。PIL 画像を numpy に戻すと cvtColor の結果と一致するはず。
    def _check_ex1():
        pil = ex1_bgr_to_pil(bgr)
        if not isinstance(pil, Image.Image):
            return False, "PIL.Image を返すこと"
        got = np.asarray(pil)
        want = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.array_equal(got, want), "BGR→RGB→PIL の色一致"
    check("ex1_bgr_to_pil", _check_ex1)

    # ex2: PIL の size (W, H) を返せているか。
    def _check_ex2():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        got = ex2_pil_size(rgb)
        want = Image.fromarray(rgb).size  # (W, H)
        return tuple(got) == tuple(want), f"size {want} を返す"
    check("ex2_pil_size", _check_ex2)

    # ex3: cv2.resize の dsize 順。出力 shape が (H, W, 3) になるか。
    def _check_ex3():
        out = ex3_resize_cv2(bgr, 100, 60)  # (幅100, 高さ60)
        return out.shape == (60, 100, 3), f"shape={out.shape} 期待(60,100,3)"
    check("ex3_resize_cv2", _check_ex3)

    # ex4: 反転＋明るさ加算（飽和）。
    def _check_ex4():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        got = ex4_hflip_brighten(rgb, 50)
        flipped = rgb[:, ::-1]
        want = np.clip(flipped.astype(np.float32) + 50, 0, 255).astype(np.uint8)
        return (got.shape == rgb.shape and np.array_equal(got, want)), "反転+明るさ飽和"
    check("ex4_hflip_brighten", _check_ex4)

    # ex5: ライブラリ選択（地図の暗記）。
    def _check_ex5():
        cases = {
            "differentiable_gpu_aug": "kornia",
            "bbox_mask_aug": "albumentations",
            "torch_training_aug": "torchvision",
            "intuitive_edit": "Pillow",
            "fast_classic_cv": "OpenCV",
            "something_unknown": "OpenCV",
        }
        ok = all(ex5_pick_library(k) == v for k, v in cases.items())
        return ok, "課題→ライブラリの対応"
    check("ex5_pick_library", _check_ex5)

    # ex6: (H,W,C) → (C,H,W) の軸並べ替え。
    def _check_ex6():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        got = ex6_to_chw(rgb)
        h, w = rgb.shape[:2]
        want = np.transpose(rgb, (2, 0, 1))
        return (got.shape == (3, h, w) and np.array_equal(got, want)), "(3,H,W) へ並べ替え"
    check("ex6_to_chw", _check_ex6)

    # ex7: 飽和加算が cv2.add と一致するか（オーバーフロー対策）。
    def _check_ex7():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        got = ex7_saturating_add(rgb, 60)
        want = cv2.add(rgb, np.full_like(rgb, 60))  # cv2.add は飽和演算
        ok = (got.dtype == np.uint8 and got.shape == rgb.shape
              and np.array_equal(got, want))
        return ok, "255 頭打ちの飽和加算（cv2.add 一致）"
    check("ex7_saturating_add", _check_ex7)

    # ex8: 長辺合わせの縮小。長辺が指定値・アスペクト比保持。
    def _check_ex8():
        out = ex8_resize_long_side(bgr, 160)  # 元 (240,320) → 長辺 320 を 160 へ
        h, w = out.shape[:2]
        long_ok = max(h, w) == 160
        # 元のアスペクト比 320/240 = 4/3。短辺 120 になるはず。
        ratio_ok = abs((w / h) - (320 / 240)) < 0.03
        return (long_ok and ratio_ok), f"out={out.shape[:2]} 長辺160・比保持"
    check("ex8_resize_long_side", _check_ex8)

    # ex9: cv2↔PIL の往復が無損失か。
    def _check_ex9():
        got = ex9_roundtrip_lossless(bgr)
        return (got.shape == bgr.shape and np.array_equal(got, bgr)), "往復で完全一致（無損失）"
    check("ex9_roundtrip_lossless", _check_ex9)

    # ex10: 再現性（同 seed で一致）＋ seed による多様性。
    def _check_ex10():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = ex10_reproducible_aug(rgb, 7)
        b = ex10_reproducible_aug(rgb, 7)        # 同じ seed → 完全一致
        reproducible = np.array_equal(a, b)
        shape_ok = (a.shape == rgb.shape and a.dtype == np.uint8)
        # 複数 seed の出力が「全部同一」でないこと（拡張が seed に依存している証拠）。
        outs = [ex10_reproducible_aug(rgb, s) for s in (0, 1, 2, 3, 4)]
        varied = any(not np.array_equal(outs[0], o) for o in outs[1:])
        return (reproducible and shape_ok and varied), "同seed一致 & seedで変化"
    check("ex10_reproducible_aug", _check_ex10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} 問 PASS")
    print("ALL PASS 🎉" if all_ok else "まだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答の差し込み（SHOW_SOLUTION=1 のとき / exercises_solutions.py から呼ばれる）
# まずは自力で解いてから見ること。模範解答本体は exercises_solutions.py にあります。
# =====================================================================

def _install_solutions() -> None:
    """exercises_solutions.py の模範解答で TODO 関数を差し替える（答え合わせ用）。

    解答コードの単一の置き場所を exercises_solutions.py に集約し、重複を避ける。
    万一そのファイルが無くても、本ファイルだけで（未実装＝FAIL のまま）動くようガードする。
    """
    try:
        from exercises_solutions import SOLUTIONS  # 同ディレクトリ
    except Exception as e:  # noqa: BLE001
        print("（模範解答ファイルを読み込めませんでした:", e, "）")
        return
    globals().update(SOLUTIONS)


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
