# 00_setup: 環境構築 — uv + Docker + CPU版PyTorch + HuggingFaceキャッシュ + device判定

> トラック: **環境構築** ／ レベル: **入門** ／ 必要な依存グループ: `dl`
> 前提: なし（ここが講座の出発点）／ 関連: 全回（この回で作る `device.py` を以降ずっと使う）

---

## 🎯 この章のゴール

本章のゴールは、uvの依存グループとDocker(python:3.12-slim + libgl1/ffmpeg)で本講座の実行環境を誰でも再現できるようにすることです。そのうえで、Linuxで巨大なCUDA版torchを避けてCPUホイールを入れる方法、HF_HOMEキャッシュの永続化、そしてcpu/mps/cudaを自動判定するtorch.deviceの定石を、いずれも全回で再利用できる形で確立します。

到達点を一言でいえば、**「自分の手元（CPU の Mac でも、GPU の Linux でも、ディスプレイの無い Docker でも）で、本講座のどのスクリプトも同じコマンドで動かせる」**状態を作ることです。そして、その土台となる環境差吸収用の共通ユーティリティ `device.py` を、自分の言葉で説明でき・そらでも書ける——これがこの章の合格ラインです。

---


## 0. 直感 — なぜ「環境構築」にまるごと1回を割くのか

機械学習・CV のプロジェクトで、初学者が最初に溶かす時間の大半は「環境」です。`torch` が CUDA 版で 2GB ダウンロードされて固まる、`cv2.imshow` が Docker でプロセスごと落ちる、別の PC では動くのに自分の Mac では動かない、モデルが毎回ダウンロードし直される——これらは**コードの問題ではなく環境の問題**で、放置すると学習の本筋に入る前に消耗します。

本講座の方針は明快です。**環境差を一箇所（`device.py` と `pyproject.toml`）に閉じ込め、各回のスクリプトは環境に依存しない形で書く**。そして「CPU・合成データ・ネット不要」を基本に据えることで、GPU が無くても・サンプル画像が無くても・オフラインでも、全教材が完走するようにします。この回で作る土台が、以降 42 回分すべての足場になります。

## 1. uv の依存グループ運用 — `[project.dependencies]` と `[dependency-groups]`

本講座のパッケージ管理は **uv**（Rust 製の高速なパッケージマネージャ）に統一しています。uv は `pyproject.toml` を唯一の真実とし、解決結果を `uv.lock` に固定するので、**誰の環境でも同じバージョンが再現**されます。そのうえで、依存は次の 2 階層に分けて管理します。

- **`[project.dependencies]`**: `uv sync` で**常に**入る本体。本講座では `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` の 4 つだけ。**画像の基礎トラック（00〜09）はこの本体だけで CPU 完走**します。
- **`[dependency-groups]`**: `uv sync --group <name>` で**必要になったときに足す**任意グループ。`dl`（torch/torchvision）、`hf`（transformers 一式）、`vector`（faiss）… のように、回ごとに使うものを隔離しています。重い依存・衝突しやすい依存を本体に混ぜないための仕組みです。

```bash
uv sync                      # 本体だけ（00〜09 はこれで足りる）
uv sync --group dl           # 深層の土台（torch/torchvision）を追加 ← この00回で使う
uv sync --group dl --group hf  # 複数グループを同時に
uv add --group hf accelerate # グループに新パッケージを足す（pyproject と lock を更新）
```

`03_dependency_groups.py` は、実際に `pyproject.toml` を `tomllib`（Python 標準）で読み、本体依存・各グループ・後述の PyTorch インデックスを一覧表示します。「いまどのグループに何が入っているか」を**コードで確認できる**のがポイントです。

## 2. PyTorch を CPU で入れる — `[[tool.uv.index]]` と `[tool.uv.sources]`

PyTorch の罠は、**PyPI 既定の `torch` が Linux では CUDA 版**で、GPU が無くても巨大な CUDA ランタイム（数 GB）を引いてしまうことです。本講座は CPU 前提なので、これを避けて**CPU ホイール**を明示的に引きます。uv ではこれを宣言的に書けます（`pyproject.toml` に設定済み）。

```toml
# CPU ホイールの配布元を「明示的に指定したときだけ使う index」として定義
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true                       # explicit=true: 名指ししたパッケージにだけ適用

# torch / torchvision を、Linux のときだけ上の index から引く
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "platform_system == 'Linux'" }]
torchvision = [{ index = "pytorch-cpu", marker = "platform_system == 'Linux'" }]
```

ポイントは `explicit = true` と `marker` の 2 つです。まず `explicit=true` は、「この index は名指しされたパッケージ（torch/torchvision）にだけ使い、他の普通のパッケージは PyPI から引く」という意味です。一方 `marker = "platform_system == 'Linux'"` は、「Linux のときだけ CPU index を使う」という条件であり、**macOS は PyPI 既定のまま**にします（mac の wheel は CPU と Apple Silicon の **MPS** を両方含むため、そちらが正解だからです）。なお、GPU を使いたい場合だけ、この URL を `cu126` などの CUDA index に差し替えます。

## 3. device 自動判定 — `cpu` / `mps` / `cuda` を 1 行で

環境が違えば、使えるアクセラレータも違います。とはいえ、その判定ロジックを各スクリプトで毎回書くのは無駄なので、**判定を `device.py` 一箇所に閉じ込め**ます。優先順位は**速い順に `cuda > mps > cpu`** とし、そのうえで `cpu` は最後の砦として常に選べるようにします。

```python
from device import pick_device, configure_threads
device = pick_device()        # cuda があれば cuda、Mac なら mps、無ければ cpu
configure_threads()           # CPU スレッド数を妥当値に固定
model = model.to(device)      # 入力もモデルも同じ device に載せれば device 非依存
x = x.to(device)
```

判定の中核 API は 3 つだけ。`torch.cuda.is_available()`（NVIDIA）、`torch.backends.mps.is_available()`（Apple Silicon）、そして `torch.device("cpu"/"cuda"/"mps")`。`device.py` の `available_devices()` がこの判定を 1 箇所に集約し、`pick_device(prefer=...)` が希望デバイス（使えなければ自動フォールバック）を返します。`01_device_util.py` で、CPU 環境でも `.to(device)` の定石が動くことを体験します。

**MPS の保険**: Apple Silicon の MPS は「多くの演算は速いが、一部は未対応」が現実です。未対応演算で `NotImplementedError` を出して止まらないよう、`PYTORCH_ENABLE_MPS_FALLBACK=1` を立てておくと、その演算だけ CPU に逃がせます。`device.py` は `mps` を選んだとき自動でこれを有効にします。

## 4. CPU スレッド調整 — `torch.set_num_threads` と `OMP_NUM_THREADS` を揃える

CPU 推論の速度は**スレッド数**に大きく左右されます。重要なのは、torch とその下で動く OpenMP（numpy/cv2 が使う）の**スレッド数を揃える**こと。片方だけ設定すると、両者が別々にスレッドを立てて物理コア数を超える「**オーバーサブスクライブ**」が起き、かえって遅くなります。

```python
import torch, os
torch.set_num_threads(4)          # torch の計算スレッド
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP(numpy/cv2) のスレッド ← 同じ値に揃える
```

`device.py` の `configure_threads()` がこの両方を 1 回で揃えます。`02_threads_and_cache.py` では、スレッド数を 1→2→4→全コアと変えて行列積の時間を**実測**し、「増やせば必ず速くなるわけではない（小さい行列・メモリ帯域で頭打ち）」ことを目で確認します。**推測するな、測れ**——これは第34回の推論プロファイリングにそのまま繋がる態度です。

## 5. HuggingFace キャッシュ — `HF_HOME` と `HF_HUB_OFFLINE`

`transformers` などで事前学習モデルを使うと、重みは初回に **`HF_HOME`** が指す場所（既定 `~/.cache/huggingface`）へダウンロードされます。ここを意識的に管理すると、2 つの嬉しさがあります。

- **永続化**: Docker では `HF_HOME` をボリュームにマウントしておくと、コンテナを作り直しても**再ダウンロードを避けられる**（巨大モデルで効く）。
- **オフライン/CI**: `HF_HUB_OFFLINE=1` を立てると、**キャッシュ済みの重みだけ**でネット非依存に動く。CI やオフライン検証で「ネットに繋がらず固まる」事故を防げます。

```bash
export HF_HOME=/cache/hf          # 重みの集約先を固定
export HF_HUB_OFFLINE=1           # キャッシュ済みだけで動かす（事前 DL が前提）
```

`device.py` の `hf_home()` がこの場所を返し、`02_threads_and_cache.py` が現在値を表示します。本講座の画像基礎トラックは HF を使わないので、ここは「深層トラックに入ったときの備え」として理解しておけば十分です。

## 6. opencv の排他 — `headless` と `full` はどちらか一方

OpenCV の Python パッケージには 2 種類あり、**同じ `cv2` 名前空間を共有するため同居させてはいけません**。

| パッケージ | 用途 | `cv2.imshow` | 本講座 |
| --- | --- | --- | --- |
| `opencv-python`（full） | ローカルで対話的に学ぶ（GUI/Qt 依存を含む） | 使える | 任意 |
| `opencv-python-headless` | Docker/サーバ/CI 配布（GUI 依存なし・軽量） | **無い** | **既定** |

本講座は headless を既定にし、`albumentations` 等との衝突も避けています。**結果は画面に出さず `outputs/` に保存して確認する**のが方針なので、`imshow` が無くても困りません。`03_dependency_groups.py` の `detect_opencv_variant()` は、インストール済みパッケージのメタデータから variant（headless/full/conflict/none）を判定し、両方入っている危険な状態を検出します。

## 7. Docker — `python:3.12-slim + libgl1/ffmpeg`

配布・再現の最終形が Docker です。`python:3.12-slim` をベースに、**OpenCV(headless) が動的リンクで必要とする `libgl1` / `libglib2.0-0`** と、**動画 I/O 用の `ffmpeg`** だけを足し、uv で依存を再現インストールします。同梱の `Dockerfile`（参考実装）は、依存定義を先にコピーしてレイヤキャッシュを効かせ、`HF_HOME=/cache/hf` をボリューム化して再 DL を防ぐ構成です。既定コマンドは「環境まるごと検証」ミニプロジェクトで、ビルド直後に健全性を確認できます。

## 8. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば理解が積み上がります。すべて CPU・合成データ・ネット不要で完走し、図/JSON は `outputs/00_setup/` に保存します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `device.py` | **この回の成果物**。cpu/mps/cuda 判定・スレッド調整・HF_HOME を返す共通ユーティリティ（全回で import） |
| `check_env.py` | 必須/任意ライブラリの導入状況・版・デバイスを表示する環境スモークテスト |
| `01_device_util.py` | `pick_device` の使い方、`.to(device)` の定石、prefer フォールバック |
| `02_threads_and_cache.py` | スレッド数 × 速度の実測、torch と OMP を揃える理由、HF_HOME/HF_HUB_OFFLINE |
| `03_dependency_groups.py` | pyproject の本体依存/グループ/PyTorch CPU index を読み解く、opencv 排他の検出 |
| `mini_project.py` | 章末ミニプロジェクト：環境まるごと検証（6 項目を PASS/FAIL 判定し JSON/図を出力） |
| `exercises.py` | TODO 形式の演習 8 問（自己採点・未実装でも exit 0） |
| `exercises_solutions.py` | 演習の模範解答（実行で全 PASS） |
| `Dockerfile` | 参考 Dockerfile（python:3.12-slim + libgl1/ffmpeg + uv + CPU torch + HF キャッシュ） |

---

## 🛠 章末ミニプロジェクト — 環境まるごと検証

この回の要素を 1 本に統合する総合課題が `mini_project.py` です。**「本講座を走らせる環境が整っているか」を自動で総点検**し、各項目を PASS/FAIL 判定して、JSON レポートとサマリ図を `outputs/00_setup/` に出力します。

検証する 6 項目：
1. **必須ライブラリ**の導入と版（numpy/cv2/PIL/matplotlib）
2. **device 自動判定**（cpu/mps/cuda）とスレッド固定
3. **torch の実計算 sanity**（選んだ device 上で単位行列積 `I @ b == b` が誤差なく成立するか）
4. **画像パイプライン往復**（cv2 BGR → RGB → PIL → numpy → cv2 が完全一致するか）
5. **headless 保存**（matplotlib=Agg と cv2.imwrite が画面なしで動くか）
6. **HuggingFace キャッシュ**場所（HF_HOME）の確認

```bash
uv run python lectures/00_setup/mini_project.py
```

`device.py`（device/スレッド/HF）と、後の 01 回で深掘りする BGR/RGB 往復、headless 保存を**ひとつのワークフローに束ねる**ことで、「環境が整う」とは具体的に何が動くことなのかを体で理解します。成果物は `outputs/00_setup/mini_project_report.json`（各チェックの結果＋環境サマリ）と `mini_project.png`（PASS/FAIL の一覧図）。

---

## ✅ 到達チェックリスト

- [ ] `[project.dependencies]`（常に入る本体）と `[dependency-groups]`（`uv sync --group` で足す任意）の違いを説明できる
- [ ] 画像基礎トラック(00〜09)が本体 4 ライブラリだけで完走する理由を言える
- [ ] `uv sync` / `uv sync --group dl` / `uv add --group hf <pkg>` を使い分けられる
- [ ] Linux で CUDA 版 torch を避けて CPU ホイールを引く `[[tool.uv.index]] explicit=true` + `[tool.uv.sources]` の仕組みを説明できる
- [ ] mac は PyPI 既定（CPU/MPS）を使う理由（`marker = platform_system == 'Linux'`）を言える
- [ ] `cuda > mps > cpu` の優先順で `torch.device` を自動判定でき、`.to(device)` の定石をそらで書ける
- [ ] `PYTORCH_ENABLE_MPS_FALLBACK=1` が何のための保険か説明できる
- [ ] `torch.set_num_threads` と `OMP_NUM_THREADS` を**揃える**べき理由（オーバーサブスクライブ）を言える
- [ ] `HF_HOME` と `HF_HUB_OFFLINE` でキャッシュ永続化・オフライン動作を制御できる
- [ ] `opencv-python` と `opencv-python-headless` の排他と、headless を既定にする方針を説明できる
- [ ] Docker で `libgl1`/`ffmpeg` がなぜ要るか（OpenCV の動的リンク・動画 I/O）を言える
- [ ] `device.py` を他モジュールから import して使え、`mini_project.py` で環境を総点検できる

---

## ❓ 落とし穴・FAQ・デバッグ

**Q1. `uv sync` したのに `import torch` できない。**
A. torch は本体依存ではなく `dl` グループ。`uv sync --group dl` を実行する。画像基礎トラック(00〜09)は torch 不要なので、本体だけだと入らないのは正常。

**Q2. Linux で torch のインストールが巨大／遅い。CUDA 版が入った。**
A. `[[tool.uv.index]]` の `pytorch-cpu` が効いていない可能性。`explicit = true` と `[tool.uv.sources]` の `marker = "platform_system == 'Linux'"` が両方そろっているか確認する。CPU ホイールは桁違いに軽い。新しめの uv なら `uv sync --group dl --torch-backend cpu` も使える。

**Q3. Mac で `pick_device()` が `mps` を返すが一部の処理で落ちる。**
A. その演算が MPS 未対応。`PYTORCH_ENABLE_MPS_FALLBACK=1` を立てる（`device.py` は `mps` 選択時に自動で立てる）。それでも不安定なら `pick_device(prefer="cpu")` で CPU に固定して切り分ける。

**Q4. `cv2.imshow` が Docker / SSH で固まる・プロセスごと落ちる。**
A. headless 環境に GUI バックエンドが無いため。本講座は `opencv-python-headless` を既定にし、結果は `cv2.imwrite` / matplotlib(Agg) で**保存して確認**する。`imshow` は使わない。

**Q5. `cv2` が import できない／`libGL.so.1: cannot open shared object file`。**
A. Docker で `libgl1`（と `libglib2.0-0`）が入っていない。`apt-get install -y libgl1 libglib2.0-0` を足す（同梱 `Dockerfile` 参照）。headless 版でも動的リンクでこれらが要る。

**Q6. `opencv-python` と `opencv-python-headless` を両方入れてしまった。**
A. `cv2` 名前空間が衝突する危険な状態。`03_dependency_groups.py` の `detect_opencv_variant()` が `conflict` を返す。どちらか一方を `uv remove` で外す（本講座は headless を残す）。

**Q7. CPU なのに遅い／たまに激遅。**
A. スレッドのオーバーサブスクライブを疑う。`configure_threads()` で torch と `OMP_NUM_THREADS` を**同じ値**に揃える。`02_threads_and_cache.py` の実測で、自分のマシンの最適スレッド数を把握しておく。

**Q8. モデルが毎回ダウンロードされる／CI でネットに繋がらず固まる。**
A. `HF_HOME` がコンテナ再生成で消えている。ボリュームにマウントして永続化する。オフライン前提なら事前に重みを落としてから `HF_HUB_OFFLINE=1`。

**Q9. matplotlib が `savefig` で固まる／`no display name` エラー。**
A. `import matplotlib.pyplot` の**前に** `matplotlib.use("Agg")` を呼ぶ。本講座の図はすべて Agg バックエンドで画面非依存に保存している。

---

## 🚀 発展トピック・参考

- **`uv.lock` による完全再現**: `uv lock` で解決を固定し、`uv sync --frozen` で lock どおりに再現。CI・Docker では `--frozen` を使うと「手元では動くのに CI で違う版が入る」を防げる。
- **`uv run --group <name>`**: その実行のときだけグループを有効化する。`uv run --group site python tools/build_site.py` のように、常用しないツールを一時的に使える。
- **`uv python pin`**: プロジェクトの Python 版を `.python-version` に固定。チームで Python バージョンを揃える。
- **CUDA を使う場合**: `[[tool.uv.index]]` の URL を `https://download.pytorch.org/whl/cu126` などに差し替え、`marker` を環境に合わせる。`pick_device()` はそのまま `cuda` を拾う。
- **スレッドの上級設定**: `torch.set_num_interop_threads`（演算子間並列）、`MKL_NUM_THREADS`、NUMA 配置など。CPU 推論を詰めるなら第34回（推論プロファイリング）へ。
- **公式ドキュメント**: [uv (Astral)](https://docs.astral.sh/uv/) / [uv の PyTorch 連携](https://docs.astral.sh/uv/guides/integration/pytorch/) / [PyTorch get-started](https://pytorch.org/get-started/locally/) / [HuggingFace cache 管理](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) / [opencv-python(PyPI)](https://pypi.org/project/opencv-python-headless/)。

---

## ▶ 動かし方

このモジュールは本体依存に加え `dl` グループ（CPU 版 torch）を使います。GPU もネット接続も不要です。

```bash
# 依存をインストール（初回のみ）。深層の土台 dl まで入れる
uv sync --group dl

# 環境スモークテスト（まずこれで全体像を掴む）
uv run python lectures/00_setup/check_env.py

# 本編（番号順）。すべて CPU・合成データ・ネット不要
uv run python lectures/00_setup/device.py                 # device ユーティリティ単体診断
uv run python lectures/00_setup/01_device_util.py         # device 自動判定と .to(device)
uv run python lectures/00_setup/02_threads_and_cache.py   # スレッド実測 + HF キャッシュ
uv run python lectures/00_setup/03_dependency_groups.py   # 依存グループ + opencv 排他

# 章末ミニプロジェクト（環境まるごと検証 → JSON/図を outputs/00_setup/ に出力）
uv run python lectures/00_setup/mini_project.py

# 演習（自己採点。未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/00_setup/exercises.py
uv run python lectures/00_setup/exercises_solutions.py
```

成果物（図・JSON）は `outputs/00_setup/` に保存されます（matplotlib=Agg、OpenCV は headless）。`mini_project_report.json` を開いて、自分の環境のデバイス・スレッド数・各チェック結果を確認してください。

---

> 参照ライブラリ: **opencv-python-headless 4.13** / **Pillow 12.2** / **numpy 2.4** / **matplotlib 3.10**（torch は **2.12+cpu**）
> （CPU 前提・合成データ・ネット不要、Linux は CPU ホイール明示／mac は MPS、matplotlib=Agg、headless OpenCV） — 2026-06
