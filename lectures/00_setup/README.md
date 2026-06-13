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

本講座の方針は明快です。**環境差を一箇所（`device.py` と `pyproject.toml`）に閉じ込め、各回のスクリプトは環境に依存しない形で書く**。そして「CPUのみ・合成データ・ネット不要」を基本に据えることで、GPU が無くても・サンプル画像が無くても・オフラインでも、全教材が完走するようにします。この回で作る土台が、以降 42 回分すべての足場になります。

## 1. uv の依存グループ運用 — `[project.dependencies]` と `[dependency-groups]`

本講座のパッケージ管理は **uv**（Rust 製の高速なパッケージマネージャ）に統一しています。uv は `pyproject.toml` を唯一の真実とし、解決結果を `uv.lock` に固定するので、**誰の環境でも同じバージョンが再現**されます。そのうえで、依存は次の 2 階層に分けて管理します。

- **`[project.dependencies]`**: `uv sync` で**常に**入る本体。本講座では `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` の 4 つだけ。**画像の基礎トラック（00〜09）はこの本体だけで CPU 完走**します。
- **`[dependency-groups]`**: `uv sync --group <name>` で**必要になったときに足す**任意グループ。`dl`（torch/torchvision）、`hf`（transformers 一式）、`vector`（faiss）… のように、回ごとに使うものを隔離しています。重い依存・衝突しやすい依存を本体に混ぜないための仕組みです。

<figure class="lec-fig"><svg viewBox="0 0 660 264" role="img" aria-label="uvの依存は2階層。本体はuv syncで常に入り、dl/hf/vectorなどの任意グループはuv sync --groupで必要時だけ足す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#2563eb">任意グループ ── uv sync --group で必要時だけ追加</text><rect x="48" y="44" width="176" height="74" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="136" y="76" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">dl</text><text x="136" y="100" text-anchor="middle" font-size="12.5" fill="#3f3f46">torch / torchvision</text><rect x="242" y="44" width="176" height="74" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="330" y="76" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">hf</text><text x="330" y="100" text-anchor="middle" font-size="12.5" fill="#3f3f46">transformers ほか</text><rect x="436" y="44" width="176" height="74" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="524" y="76" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">vector</text><text x="524" y="100" text-anchor="middle" font-size="12.5" fill="#3f3f46">faiss</text><polygon points="136,138 128,122 144,122" fill="#71717a"/><polygon points="330,138 322,122 338,122" fill="#71717a"/><polygon points="524,138 516,122 532,122" fill="#71717a"/><rect x="48" y="148" width="564" height="92" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><text x="64" y="176" font-size="15" font-weight="700" fill="#c2410c">本体 [project.dependencies] ── uv sync で常に入る</text><rect x="64" y="190" width="132" height="34" rx="6" fill="#ffffff" stroke="#ea580c" stroke-width="1.5"/><text x="130" y="212" text-anchor="middle" font-size="13" fill="#18181b">numpy</text><rect x="208" y="190" width="184" height="34" rx="6" fill="#ffffff" stroke="#ea580c" stroke-width="1.5"/><text x="300" y="212" text-anchor="middle" font-size="13" fill="#18181b">opencv-headless</text><rect x="404" y="190" width="92" height="34" rx="6" fill="#ffffff" stroke="#ea580c" stroke-width="1.5"/><text x="450" y="212" text-anchor="middle" font-size="13" fill="#18181b">pillow</text><rect x="508" y="190" width="92" height="34" rx="6" fill="#ffffff" stroke="#ea580c" stroke-width="1.5"/><text x="554" y="212" text-anchor="middle" font-size="13" fill="#18181b">matplotlib</text></svg><figcaption>uv の依存は <b>2 階層</b>です。<b>本体 [project.dependencies]</b>（numpy・opencv-headless・pillow・matplotlib の 4 つ）は <code>uv sync</code> で常に入り、<b>画像基礎トラック(00〜09)はこれだけで完走</b>します。<code>dl</code> / <code>hf</code> / <code>vector</code> などの任意グループは <code>uv sync --group</code> で、その回が必要になったときだけ足します。</figcaption></figure>

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

<figure class="lec-fig"><svg viewBox="0 0 660 268" role="img" aria-label="torchのCPUインストール分岐。LinuxはCPU index指定でCPUホイール、macOSはPyPI既定でCPUとMPSのホイール" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="24" y="106" width="116" height="56" rx="8" fill="#fafafa" stroke="#52525b" stroke-width="2"/><text x="82" y="139" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">torch・torchvision</text><line x1="140" y1="134" x2="158" y2="134" stroke="#71717a" stroke-width="2"/><line x1="158" y1="60" x2="158" y2="208" stroke="#71717a" stroke-width="2"/><line x1="158" y1="60" x2="176" y2="60" stroke="#71717a" stroke-width="2"/><polygon points="182,60 172,55 172,65" fill="#71717a"/><line x1="158" y1="208" x2="176" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="182,208 172,203 172,213" fill="#71717a"/><rect x="184" y="38" width="96" height="44" rx="7" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="232" y="65" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">Linux</text><line x1="280" y1="60" x2="312" y2="60" stroke="#71717a" stroke-width="2"/><polygon points="318,60 308,55 308,65" fill="#71717a"/><rect x="320" y="38" width="150" height="44" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="395" y="65" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">pytorch-cpu index</text><line x1="470" y1="60" x2="502" y2="60" stroke="#71717a" stroke-width="2"/><polygon points="508,60 498,55 498,65" fill="#71717a"/><rect x="510" y="38" width="128" height="44" rx="7" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="574" y="65" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">CPU ホイール</text><text x="320" y="116" font-size="12" fill="#dc2626">設定なし＝CUDA版（数GB）を引く ← 避ける</text><rect x="184" y="186" width="96" height="44" rx="7" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="232" y="213" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">macOS</text><line x1="280" y1="208" x2="312" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="318,208 308,203 308,213" fill="#71717a"/><rect x="320" y="186" width="150" height="44" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="395" y="213" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">PyPI 既定</text><line x1="470" y1="208" x2="502" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="508,208 498,203 498,213" fill="#71717a"/><rect x="510" y="186" width="128" height="44" rx="7" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="574" y="206" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">CPU + MPS</text><text x="574" y="222" text-anchor="middle" font-size="11" fill="#15803d">ホイール</text></svg><figcaption>PyTorch を <b>CPU で</b>入れる分岐です。<b>Linux</b> では <code>[[tool.uv.index]] explicit=true</code> と <code>marker=platform_system=='Linux'</code> で <b>pytorch-cpu</b> index を名指しし、軽い <b>CPU ホイール</b>を引きます（設定しないと GPU が無くても巨大な <b>CUDA 版</b>が入る）。<b>macOS</b> は PyPI 既定のままでよく、<b>CPU+MPS</b> 入りのホイールが選ばれます。</figcaption></figure>

## 3. device 自動判定 — `cpu` / `mps` / `cuda` を 1 行で

環境が違えば、使えるアクセラレータも違います。とはいえ、その判定ロジックを各スクリプトで毎回書くのは無駄なので、**判定を `device.py` 一箇所に閉じ込め**ます。優先順位は**速い順に `cuda > mps > cpu`** とし、そのうえで `cpu` は最後の砦として常に選べるようにします。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="pick_deviceの優先順位。速い順にcuda、mps、cpuを試し、使えなければ次へフォールバックする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" fill="#3f3f46">pick_device() ── 速い順に試し、使えなければ次へ</text><rect x="40" y="78" width="160" height="120" rx="10" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><text x="120" y="128" text-anchor="middle" font-size="22" font-weight="700" fill="#c2410c">cuda</text><text x="120" y="156" text-anchor="middle" font-size="13" fill="#3f3f46">NVIDIA GPU</text><text x="120" y="178" text-anchor="middle" font-size="12" fill="#71717a">いちばん速い</text><line x1="200" y1="138" x2="244" y2="138" stroke="#71717a" stroke-width="2"/><polygon points="250,138 240,133 240,143" fill="#71717a"/><text x="222" y="128" text-anchor="middle" font-size="11" fill="#71717a">無ければ</text><rect x="250" y="78" width="160" height="120" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2.5"/><text x="330" y="128" text-anchor="middle" font-size="22" font-weight="700" fill="#1d4ed8">mps</text><text x="330" y="156" text-anchor="middle" font-size="13" fill="#3f3f46">Apple Silicon</text><text x="330" y="178" text-anchor="middle" font-size="12" fill="#71717a">Mac で速い</text><line x1="410" y1="138" x2="454" y2="138" stroke="#71717a" stroke-width="2"/><polygon points="460,138 450,133 450,143" fill="#71717a"/><text x="432" y="128" text-anchor="middle" font-size="11" fill="#71717a">無ければ</text><rect x="460" y="78" width="160" height="120" rx="10" fill="#ffffff" stroke="#16a34a" stroke-width="2.5"/><text x="540" y="128" text-anchor="middle" font-size="22" font-weight="700" fill="#15803d">cpu</text><text x="540" y="156" text-anchor="middle" font-size="13" fill="#3f3f46">どこでも必ず動く</text><text x="540" y="178" text-anchor="middle" font-size="12" fill="#16a34a">最後の砦</text></svg><figcaption><code>pick_device()</code> の優先順位です。<b>速い順に cuda → mps → cpu</b> を試し、<b>使えなければ自動で次へフォールバック</b>します。判定は <code>torch.cuda.is_available()</code>（NVIDIA）→ <code>torch.backends.mps.is_available()</code>（Apple Silicon）→ どこでも必ず動く <b>cpu</b>（最後の砦）の順。だから同じコードが GPU でも Mac でも Docker でも動きます。</figcaption></figure>

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

<figure class="lec-fig"><svg viewBox="0 0 660 286" role="img" aria-label="スレッド数の揃え方。1コア1スレッドなら最適、torchとOMPが別々に全コア分立てるとコア数超過で競合し遅くなる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" fill="#3f3f46">CPU 物理コア数に スレッド数を揃える</text><rect x="24" y="64" width="296" height="200" rx="10" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="172" y="90" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">✓ 一致：1 コア = 1 スレッド</text><rect x="50" y="104" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="80" y="120" width="60" height="30" rx="4" fill="#f97316"/><rect x="180" y="104" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="210" y="120" width="60" height="30" rx="4" fill="#f97316"/><rect x="50" y="178" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="80" y="194" width="60" height="30" rx="4" fill="#f97316"/><rect x="180" y="178" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="210" y="194" width="60" height="30" rx="4" fill="#f97316"/><text x="172" y="258" text-anchor="middle" font-size="12" fill="#15803d">4 コア = 4 スレッド（最適）</text><rect x="340" y="64" width="296" height="200" rx="10" fill="#fafafa" stroke="#dc2626" stroke-width="2"/><text x="488" y="90" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">✗ バラバラ：1 コアに 2 スレッド</text><rect x="366" y="104" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="372" y="118" width="52" height="34" rx="4" fill="#ea580c"/><rect x="428" y="118" width="52" height="34" rx="4" fill="#c2410c"/><rect x="496" y="104" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="502" y="118" width="52" height="34" rx="4" fill="#ea580c"/><rect x="558" y="118" width="52" height="34" rx="4" fill="#c2410c"/><rect x="366" y="178" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="372" y="192" width="52" height="34" rx="4" fill="#ea580c"/><rect x="428" y="192" width="52" height="34" rx="4" fill="#c2410c"/><rect x="496" y="178" width="120" height="62" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="502" y="192" width="52" height="34" rx="4" fill="#ea580c"/><rect x="558" y="192" width="52" height="34" rx="4" fill="#c2410c"/><text x="488" y="258" text-anchor="middle" font-size="12" fill="#dc2626">4 コアに 8 スレッドが競合</text></svg><figcaption>CPU 推論では <b>torch のスレッド数</b>と <b>OpenMP(numpy/cv2) のスレッド数</b>を物理コア数に<b>揃える</b>のが要点です。左のように <b>1 コア = 1 スレッド</b>なら最適（灰色＝コア／オレンジ＝スレッド）。揃え忘れて両者が別々に全コア分立てると、右のように <b>コア数を超えるスレッドが競合</b>（オーバーサブスクライブ）してかえって遅くなります。<code>configure_threads()</code> が両方を同じ値に揃えます。</figcaption></figure>

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

<figure class="lec-fig"><svg viewBox="0 0 520 322" role="img" aria-label="Dockerfileは上から順に積むビルド層。python3.12-slimベース、apt libgl1/ffmpeg、依存コピーとuv sync、HF_HOMEボリューム化、CMDで環境検証" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="260" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Dockerfile ── 上から順に積むビルド層</text><rect x="100" y="40" width="320" height="42" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="260" y="59" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">FROM python:3.12-slim</text><text x="260" y="76" text-anchor="middle" font-size="11.5" fill="#52525b">軽量な公式ベースイメージ</text><rect x="100" y="96" width="320" height="42" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="260" y="115" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">apt: libgl1 / glib / ffmpeg</text><text x="260" y="132" text-anchor="middle" font-size="11.5" fill="#52525b">OpenCV の動的リンク・動画 I/O</text><rect x="100" y="152" width="320" height="42" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="260" y="171" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">依存定義をコピー → uv sync</text><text x="260" y="188" text-anchor="middle" font-size="11.5" fill="#52525b">レイヤキャッシュで再ビルドを高速化</text><rect x="100" y="208" width="320" height="42" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="260" y="227" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">HF_HOME=/cache/hf をボリューム化</text><text x="260" y="244" text-anchor="middle" font-size="11.5" fill="#52525b">モデル重みの再ダウンロードを防ぐ</text><rect x="100" y="264" width="320" height="42" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2.5"/><text x="260" y="283" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">CMD: 環境まるごと検証</text><text x="260" y="300" text-anchor="middle" font-size="11.5" fill="#52525b">ビルド直後に健全性チェック</text><line x1="260" y1="82" x2="260" y2="90" stroke="#71717a" stroke-width="2"/><polygon points="260,96 255,86 265,86" fill="#71717a"/><line x1="260" y1="138" x2="260" y2="146" stroke="#71717a" stroke-width="2"/><polygon points="260,152 255,142 265,142" fill="#71717a"/><line x1="260" y1="194" x2="260" y2="202" stroke="#71717a" stroke-width="2"/><polygon points="260,208 255,198 265,198" fill="#71717a"/><line x1="260" y1="250" x2="260" y2="258" stroke="#71717a" stroke-width="2"/><polygon points="260,264 255,254 265,254" fill="#71717a"/></svg><figcaption>同梱の <code>Dockerfile</code> は<b>上から順に積むレイヤ</b>で構成します。<b>python:3.12-slim</b> をベースに、OpenCV の動的リンクと動画 I/O に必要な <code>libgl1</code>・<code>ffmpeg</code> を <code>apt</code> で足し、<b>依存定義を先にコピーしてから <code>uv sync</code></b>（レイヤキャッシュが効く）、<code>HF_HOME=/cache/hf</code> をボリューム化して再 DL を防ぎ、最後に <b>環境まるごと検証</b>を既定コマンドにします。</figcaption></figure>

## 8. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば理解が積み上がります。すべて CPUのみ・合成データ・ネット不要で完走し、図/JSON は `outputs/00_setup/` に保存します。

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

<figure class="lec-fig"><svg viewBox="0 0 660 312" role="img" aria-label="mini_project.pyは6項目を順に検証しレポートへ合流するワークフロー。必須ライブラリ・device判定・torch計算・画像往復・headless保存・HFキャッシュ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">mini_project.py ── 6 項目を順に検証し 1 枚のレポートへ</text><rect x="78" y="50" width="148" height="52" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="152" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">1. 必須ライブラリ</text><text x="152" y="91" text-anchor="middle" font-size="11.5" fill="#52525b">numpy / cv2 / PIL</text><rect x="256" y="50" width="148" height="52" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="330" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">2. device 判定</text><text x="330" y="91" text-anchor="middle" font-size="11.5" fill="#52525b">cpu / mps / cuda</text><rect x="434" y="50" width="148" height="52" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="508" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">3. torch 実計算</text><text x="508" y="91" text-anchor="middle" font-size="11.5" fill="#52525b">単位行列積の検算</text><rect x="434" y="170" width="148" height="52" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="508" y="192" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">4. 画像往復</text><text x="508" y="211" text-anchor="middle" font-size="11.5" fill="#52525b">BGR↔RGB が一致</text><rect x="256" y="170" width="148" height="52" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="330" y="192" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">5. headless 保存</text><text x="330" y="211" text-anchor="middle" font-size="11.5" fill="#52525b">Agg / imwrite</text><rect x="78" y="170" width="148" height="52" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="152" y="192" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">6. HF キャッシュ</text><text x="152" y="211" text-anchor="middle" font-size="11.5" fill="#52525b">HF_HOME を確認</text><line x1="226" y1="76" x2="250" y2="76" stroke="#71717a" stroke-width="2"/><polygon points="256,76 246,71 246,81" fill="#71717a"/><line x1="404" y1="76" x2="428" y2="76" stroke="#71717a" stroke-width="2"/><polygon points="434,76 424,71 424,81" fill="#71717a"/><line x1="508" y1="102" x2="508" y2="164" stroke="#71717a" stroke-width="2"/><polygon points="508,170 503,160 513,160" fill="#71717a"/><line x1="434" y1="196" x2="410" y2="196" stroke="#71717a" stroke-width="2"/><polygon points="404,196 414,191 414,201" fill="#71717a"/><line x1="256" y1="196" x2="232" y2="196" stroke="#71717a" stroke-width="2"/><polygon points="226,196 236,191 236,201" fill="#71717a"/><line x1="152" y1="222" x2="152" y2="254" stroke="#71717a" stroke-width="2"/><polygon points="152,260 147,250 157,250" fill="#71717a"/><rect x="78" y="260" width="440" height="44" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/><text x="298" y="287" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">PASS / FAIL レポート（JSON + 一覧図）</text></svg><figcaption><code>mini_project.py</code> は環境の健全性を <b>6 項目</b>順に検証するワークフローです。<b>必須ライブラリ → device 判定 → torch 実計算 → 画像往復 → headless 保存 → HF キャッシュ</b>の順に <b>PASS/FAIL</b> を判定し、最後に <code>mini_project_report.json</code> と一覧図 <code>mini_project.png</code> へ<b>合流</b>させて出力します。</figcaption></figure>

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

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/00_setup/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 使えるデバイス一覧と希望から実際に使うデバイス名を返す（`ex1_select_device` の TODO）。prefer が available にあればそれを採用し、無ければ `cuda > mps > cpu` の優先度順で最初に該当するものを返す。
2. 要求スレッド数を `1..cpu_count` の範囲にクランプして返す（`ex2_clamp_threads` の TODO）。0 や負数は 1 に、cpu_count 超過は cpu_count に丸める。
3. インストール済みパッケージ一覧から opencv の variant を判定する（`ex3_opencv_variant` の TODO）。`conflict`（両方）/`headless`/`full`/`none` の 4 通りを返す。
4. HuggingFace 用の環境変数 dict を組み立てて返す（`ex4_build_hf_env` の TODO）。`HF_HOME` と `HF_HUB_OFFLINE`（offline なら "1"、それ以外は "0" の文字列）を入れる。
5. device が `mps` のときだけ MPS フォールバック env を返す（`ex5_mps_fallback_env` の TODO）。mps なら `{"PYTORCH_ENABLE_MPS_FALLBACK": "1"}`、それ以外は空 dict。
6. pyproject.toml 文字列の `[dependency-groups]` から各グループのパッケージ個数を集計して返す（`ex6_count_group_packages` の TODO）。`tomllib` でパースし、各グループのリスト長を数える。
7. 必要グループのうち未導入のものを抽出する（`ex7_missing_groups` の TODO）。重複を除き、昇順のソート済みリストで返す。
8. 指定 device 上にテンソルを作り、その device 種別を返す（`ex8_tensor_device` の TODO）。`.device.type`（"cpu"/"cuda"/"mps"）を文字列で返す。

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

# 本編（番号順）。すべて CPUのみ・合成データ・ネット不要
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
