"""01: 画像・動画処理ライブラリの「地図」と早見表。

学ぶこと:
  - 主要ライブラリ（OpenCV / Pillow / NumPy / scikit-image / imageio / PyAV /
    torchvision transforms v2 / albumentations / kornia / matplotlib）の役割分担。
  - それぞれの「色順・データ表現・GPU/微分可能性・強み・使いどころ」を一覧で比較する。
  - いま自分の環境に何が入っているかを import で点検し、無いものは導入コマンドを案内する。
  - 役割を 2 軸（低レベル⇔高レベル / CPUのみ⇔GPU・微分可能）の散布図で\"地図\"として可視化する。

この回は概念回なので、難しいアルゴリズムは出てきません。代わりに
「どの課題でどのライブラリを選ぶか」を即答できる手札を作るのが狙いです。

実行:
  uv run python lectures/02_cv_libraries_overview/01_library_map.py
結果は outputs/02_cv_libraries_overview/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

# matplotlib は pyplot を import する前に必ず Agg（画面不要）バックエンドへ固定する。
# これで DISPLAY が無い Docker/SSH/CI でも savefig が確実に動く。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import output_dir, probe  # noqa: E402


# === 早見表のデータ ========================================================
# 各ライブラリを 1 行のレコードにする。列の意味:
#   role        : 一番の役割
#   data        : 主に扱うデータ表現（ndarray / PIL.Image / Tensor）
#   color       : 既定の色順（BGR は OpenCV だけ。残りは RGB）
#   gpu_diff    : GPU / 微分可能性（学習パイプラインに組み込めるか）
#   strength    : 強み
#   use_when    : こういう時に選ぶ
#   import_name : import 名（導入チェック用）
#   hint        : 未導入時の導入コマンド
LIBRARIES = [
    {
        "name": "NumPy", "import_name": "numpy",
        "role": "配列計算の土台", "data": "ndarray", "color": "-",
        "gpu_diff": "CPU", "strength": "全画像の実体。スライス/ブロードキャスト",
        "use_when": "画素を直接いじる・自作処理",
        "use_en": "touch pixels / custom ops",
        "hint": "uv add numpy",
    },
    {
        "name": "OpenCV", "import_name": "cv2",
        "role": "古典CV総合", "data": "ndarray(BGR)", "color": "BGR",
        "gpu_diff": "CPU", "strength": "高速・機能網羅・動画I/O・カメラ",
        "use_when": "前処理/検出/動画。まず第一候補",
        "use_en": "preprocess / detect / video (1st)",
        "hint": "uv add opencv-python-headless",
    },
    {
        "name": "Pillow", "import_name": "PIL",
        "role": "画像編集I/O", "data": "PIL.Image(RGB)", "color": "RGB",
        "gpu_diff": "CPU", "strength": "直感的・フォント描画・EXIF・GIF",
        "use_when": "読み書き/簡単な加工/学習の前処理",
        "use_en": "read-write / edit / font / EXIF",
        "hint": "uv add pillow",
    },
    {
        "name": "scikit-image", "import_name": "skimage",
        "role": "研究寄り画像処理", "data": "ndarray(RGB,float)", "color": "RGB",
        "gpu_diff": "CPU", "strength": "アルゴリズム豊富・float[0,1]・科学計算親和",
        "use_when": "セグメンテーション/計測/論文の再現",
        "use_en": "segmentation / measure / repro",
        "hint": "uv add --group aug scikit-image",
    },
    {
        "name": "imageio", "import_name": "imageio",
        "role": "汎用I/O", "data": "ndarray(RGB)", "color": "RGB",
        "gpu_diff": "CPU", "strength": "画像/動画/GIFを統一APIで読み書き",
        "use_when": "色々な形式を手軽に入出力",
        "use_en": "easy I/O, many formats",
        "hint": "uv add --group aug imageio",
    },
    {
        "name": "PyAV(av)", "import_name": "av",
        "role": "動画I/O(低レベル)", "data": "ndarray/Frame", "color": "RGB*",
        "gpu_diff": "CPU", "strength": "FFmpeg同梱・正確なtimestamp/コーデック制御",
        "use_when": "精密な動画デコード/エンコード",
        "use_en": "precise video decode/encode",
        "hint": "uv add --group aug av",
    },
    {
        "name": "torchvision v2", "import_name": "torchvision",
        "role": "学習用前処理/拡張", "data": "Tensor", "color": "RGB",
        "gpu_diff": "GPU/微分可", "strength": "transforms.v2・bbox/mask対応・学習と一体",
        "use_when": "PyTorch学習の前処理・データ拡張",
        "use_en": "PyTorch training aug (v2)",
        "hint": "uv add --group dl torchvision",
    },
    {
        "name": "albumentations", "import_name": "albumentations",
        "role": "データ拡張専門", "data": "ndarray(RGB)", "color": "RGB",
        "gpu_diff": "CPU", "strength": "高速・Compose・bbox/mask/keypoint同時変換",
        "use_when": "検出/セグメの本格的な拡張パイプライン",
        "use_en": "detection/seg aug (bbox/mask)",
        "hint": "uv add --group aug albumentations",
    },
    {
        "name": "kornia", "import_name": "kornia",
        "role": "微分可能CV", "data": "Tensor", "color": "RGB",
        "gpu_diff": "GPU/微分可", "strength": "GPUバッチ拡張・勾配が流れる",
        "use_when": "拡張を学習ループ内・GPUで回す",
        "use_en": "differentiable aug in train loop",
        "hint": "uv add --group aug kornia",
    },
    {
        "name": "matplotlib", "import_name": "matplotlib",
        "role": "可視化", "data": "ndarray(RGB)", "color": "RGB",
        "gpu_diff": "CPU", "strength": "Aggで画面不要・図/比較/ヒストグラム",
        "use_when": "headlessでの結果確認・図の保存",
        "use_en": "headless display / save figs",
        "hint": "uv add matplotlib",
    },
]


def print_cheatsheet() -> None:
    """早見表をコンソールへ。導入状況（✓/×）も付ける。"""
    print("=== 主要ライブラリ早見表 ===")
    header = f"{'ライブラリ':16s} {'導入':6s} {'色順':9s} {'データ':16s} {'GPU/微分':10s} 役割"
    print(header)
    print("-" * len(header))
    for lib in LIBRARIES:
        p = probe(lib["name"], lib["import_name"], lib["hint"])
        mark = f"✓{p.version}" if p.available else "×"
        # バージョン文字列が長いと崩れるので軽く詰める。
        mark = mark[:6]
        print(
            f"{lib['name']:16s} {mark:6s} {lib['color']:9s} "
            f"{lib['data']:16s} {lib['gpu_diff']:10s} {lib['role']}"
        )


def save_table_figure(out: pathlib.Path) -> None:
    """早見表を画像（表）として保存する。

    図のラベルは日本語フォント非搭載環境でも崩れないよう、英語/短い記号で書く。
    （詳しい日本語解説は README とコンソール出力に置く方針。）
    """
    # 図のセルは英語表記にして、日本語フォント非搭載環境でも文字化け(豆腐)しないようにする。
    cols = ["Library", "Installed", "Color", "Data", "GPU/diff", "Use when"]
    gpu_en = {"CPU": "CPU", "GPU/微分可": "GPU/diff"}
    rows = []
    colors = []
    for lib in LIBRARIES:
        p = probe(lib["name"], lib["import_name"], lib["hint"])
        installed = "yes" if p.available else "no"
        rows.append([
            lib["name"], installed, lib["color"],
            lib["data"], gpu_en.get(lib["gpu_diff"], lib["gpu_diff"]), lib["use_en"],
        ])
        # 導入済みは薄緑、未導入は薄赤で行を塗り分け、環境が一目で分かるようにする。
        tint = "#e8f5e9" if p.available else "#fdecea"
        colors.append([tint] * len(cols))

    fig, ax = plt.subplots(figsize=(15, 0.55 * len(rows) + 1.2))
    ax.axis("off")
    # 各列に幅を割り当てる（合計 1.0）。最後の "Use when" を広めに取り、右端での見切れを防ぐ。
    col_widths = [0.14, 0.09, 0.07, 0.16, 0.10, 0.44]
    table = ax.table(
        cellText=rows, colLabels=cols, cellColours=colors,
        colWidths=col_widths, loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # ヘッダ行を太字＆色付けする。
    for c in range(len(cols)):
        cell = table[0, c]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#37474f")
    ax.set_title("CV library cheat sheet (green=installed / red=not installed)", pad=14)
    fig.tight_layout()
    path = out / "01_library_cheatsheet.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  保存:", path.name)


def save_quadrant_map(out: pathlib.Path) -> None:
    """役割を 2 軸の散布図で\"地図\"化する。

    x 軸: 低レベル(配列を直接いじる) ←→ 高レベル(学習向け・抽象度が高い)
    y 軸: CPUのみ ←→ GPU・微分可能(学習ループに組み込める)
    近い場所にあるライブラリは「守備範囲が似ている=競合しやすい」と読める。
    """
    # (x, y) は役割の位置づけを手で配置した代表値（厳密な指標ではなく地図上の目安）。
    points = {
        "NumPy": (0.6, 1.0, "core"),
        "OpenCV": (2.6, 1.6, "classic"),
        "Pillow": (4.2, 1.0, "classic"),
        "scikit-image": (3.0, 2.3, "classic"),
        "imageio/PyAV (I/O)": (1.4, 0.5, "io"),
        "matplotlib (viz)": (5.2, 1.2, "viz"),
        "albumentations": (6.2, 3.6, "aug"),
        "torchvision v2": (7.2, 6.2, "aug"),
        "kornia": (8.6, 8.6, "aug"),
    }
    palette = {
        "core": "#455a64", "classic": "#1565c0", "io": "#6a1b9a",
        "viz": "#00838f", "aug": "#ef6c00",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, (x, y, cat) in points.items():
        ax.scatter(x, y, s=240, color=palette[cat], alpha=0.85, edgecolors="white", zorder=3)
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10)

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 10)
    ax.set_xlabel("low-level (touch pixels)  --->  high-level (learning-oriented)")
    ax.set_ylabel("CPU only  --->  GPU / differentiable")
    ax.set_title("Map of CV libraries: who does what")
    ax.grid(True, linestyle="--", alpha=0.4)
    # 4 象限の意味を薄く書き込む。
    ax.text(0.3, 9.2, "GPU + low-level\n(rare)", color="gray", fontsize=9)
    ax.text(7.0, 9.2, "GPU augmentation\n(kornia / torchvision)", color="gray", fontsize=9)
    ax.text(0.3, -0.3, "CPU classic CV\n(OpenCV / Pillow / skimage)", color="gray", fontsize=9)
    fig.tight_layout()
    path = out / "01_library_quadrant_map.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  保存:", path.name)


def print_decision_guide() -> None:
    """「課題 → まず試すライブラリ」の意思決定ガイドをコンソールへ。"""
    print("\n=== ざっくり選択ガイド（迷ったらこの順で考える） ===")
    guide = [
        ("画素を直接いじる・自作の処理を書く", "NumPy（+ OpenCV）"),
        ("読み込み/保存/リサイズ/ぼかしなど定番処理を速く", "OpenCV"),
        ("直感的に画像を編集・フォント描画・EXIF対応", "Pillow"),
        ("論文のアルゴリズム再現・科学計算寄りの処理", "scikit-image"),
        ("検出/セグメ用に bbox/mask ごと拡張したい", "albumentations"),
        ("PyTorch 学習の前処理・データ拡張", "torchvision transforms v2"),
        ("拡張を GPU・学習ループ内で微分可能に回す", "kornia"),
        ("動画/カメラを手早く読む", "OpenCV VideoCapture"),
        ("動画を精密に（timestamp/コーデック）扱う", "PyAV(av) / imageio-ffmpeg"),
        ("headless で結果を見る・図にする", "matplotlib(Agg) / 保存"),
    ]
    for task, lib in guide:
        print(f"  ・{task:36s} → {lib}")


def main() -> None:
    out = output_dir()
    print_cheatsheet()
    print_decision_guide()
    print("\n=== 図を保存 ===")
    save_table_figure(out)
    save_quadrant_map(out)
    print("\n完了。outputs/02_cv_libraries_overview/ の早見表と地図を確認してください。")


if __name__ == "__main__":
    main()
