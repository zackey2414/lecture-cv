"""第9回モジュール共通ヘルパ（動画I/Oの基礎）。

このファイルは「動画I/Oの章で何度も書く定型処理」をまとめた小さな道具箱です。
本講座の方針として、各スクリプト（01〜03）と exercises.py はこのヘルパだけを import します。
（※ import するのは "同じモジュール内" のこのファイルだけ。他モジュールには依存しません。
   ネット・GPU・Webカメラ・サンプル動画が無くても、ここで全部を numpy/cv2 で合成生成します。）

ここに置く責務は次の3つだけに絞っています。
  1. 出力先 lectures/09_video_io_basics/outputs/ の管理（headless では「保存して後で見る」が基本）。
  2. 「動く図形＋フレーム番号テキスト」の合成フレーム列を作る（シークの確認がしやすい）。
  3. その合成フレーム列を動画ファイルとして書き出す（mp4v→MJPG→連番PNG の順にフォールバック）。

コーディング原則: 各関数は1つのことだけを行い、賢すぎる抽象化は避けます。
スクリプト本体側は、学習上重要な処理（VideoCapture のループや VideoWriter）をあえて
生のまま書いて「読んで学べる」ようにしつつ、ここの "素材づくり" だけを呼びます。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/09_video_io_basics/ にある。
#   parents[2] = プロジェクトルート（lecture-cv/）。どこから実行しても場所がぶれない。
MODULE_ID = "09_video_io_basics"
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
OUTPUT_ROOT = _HERE.parent / "outputs"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す（cv2.imshow）」代わりに「ファイルに保存して後で見る」
    のが基本。存在しなければ親ごと自動で作成する。
    """
    d = OUTPUT_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_moving_frames(num: int = 60, width: int = 320, height: int = 240
                       ) -> list[np.ndarray]:
    """「左から右へ動くオレンジの円」＋「フレーム番号テキスト」の連番フレームを作る。

    動画とは連続した静止画（フレーム）である、という本章の核心を体感するための素材。
    各フレームに番号を描いておくので、あとでシーク（指定フレームへ飛ぶ）した結果が
    正しいかを目視で確認できる。返すのはすべて (height, width, 3) uint8 の BGR 配列。
    """
    frames: list[np.ndarray] = []
    for i in range(num):
        # 濃いグレーの背景。OpenCV の色順は (B, G, R) であることに注意。
        frame = np.full((height, width, 3), 30, dtype=np.uint8)

        # 画面外周に固定の枠を描く（動かない基準。シークの確認がしやすくなる）。
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (70, 70, 70), 2)

        # フレーム番号に応じて円の x 座標を左→右へ動かす。
        #   i=0 で左端、i=num-1 で右端。割り算のゼロ除算を避けるため max(...,1)。
        cx = 20 + int(i / max(num - 1, 1) * (width - 40))
        cy = height // 2
        cv2.circle(frame, (cx, cy), 16, (0, 165, 255), thickness=-1)  # 塗りつぶしのオレンジ

        # 何フレーム目かを左下に描く（シーク結果の答え合わせ用）。
        cv2.putText(frame, f"frame {i:03d}", (15, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(frame)
    return frames


def make_demo_video(out_dir: pathlib.Path, name: str = "demo_source",
                    fps: float = 24.0, num: int = 60,
                    width: int = 320, height: int = 240
                    ) -> tuple[str, float, int, str]:
    """合成フレーム列を動画ファイルに書き出し、VideoCapture で開ける source を返す。

    headless の opencv-python-headless でも書ける fourcc を順に試す:
      1. "mp4v" (.mp4)  2. "MJPG" (.avi)  どちらも本体同梱で移植性が高い。
    いずれの VideoWriter も開けない環境では、最後の手段として「連番PNG」を書き、
    cv2.VideoCapture が画像シーケンスとして読める "%04d.png" パターンを source にする。

    返り値: (source, fps, num_frames, mode)
      - source : cv2.VideoCapture(source) にそのまま渡せる文字列（動画パス or PNGパターン）
      - mode   : 実際に使われた書き出し方法（"mp4v.mp4" / "MJPG.avi" / "png-sequence"）
    """
    frames = make_moving_frames(num=num, width=width, height=height)

    # --- まず動画コンテナ＋コーデックで書けるか試す ---------------------
    for ext, cc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = out_dir / f"{name}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*cc)
        # VideoWriter は (幅, 高さ) の順。フレームの shape は (高さ, 幅) で逆なので要注意。
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            # 開けなければ次のコーデックへ（壊れた空ファイルは残さない方針で続行）。
            writer.release()
            continue
        for f in frames:
            writer.write(f)
        writer.release()
        return str(path), fps, len(frames), f"{cc}{ext}"

    # --- フォールバック: 連番PNG（VideoCapture は %04d.png を読める） ----
    seq = out_dir / f"{name}_frames"
    seq.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(seq / f"{i:04d}.png"), f)
    # 画像シーケンスは fps を持たないため 0 が返ることがある点に注意（mode で判別可能）。
    return str(seq / "%04d.png"), fps, len(frames), "png-sequence"


def fourcc_to_str(fourcc_int: int) -> str:
    """CAP_PROP_FOURCC で得た 32bit 整数を 4 文字のコーデック名へ復元する。

    fourcc は 4 つの ASCII 文字を 1 バイトずつ詰めた整数。下位 8bit から順に取り出す。
    """
    code = int(fourcc_int)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def _smoke_test() -> None:
    """このファイルを直接実行したときの動作確認（合成→書き出し→読み戻し）。"""
    out = output_dir()
    source, fps, num, mode = make_demo_video(out, name="helper_smoke", num=20)
    cap = cv2.VideoCapture(source)
    count = 0
    while True:
        ret, _frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    print(f"[cv_helpers] mode={mode} source={source}")
    print(f"[cv_helpers] 書き出し {num} フレーム / 読み戻し {count} フレーム / fps={fps}")
    print("[cv_helpers] OK（このファイルは各スクリプトから import して使います）")


if __name__ == "__main__":
    _smoke_test()
