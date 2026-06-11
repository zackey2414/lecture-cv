# 34_inference_profiling: 推論高速化の地図 — 計測ファースト・プロファイリング・autocast・torch.compile

> 前提: 第13回「分類と転移学習」。`model.eval()` / `torch.inference_mode()` / `.to(device)` を一度触っていると理解が速い。
> このモジュールは「最適化・デプロイ」トラックの入口であり、続く第35回（量子化・枝刈り）/第36回（ONNX）/第37回（エッジ最適化）の **共通の物差し（ベンチの作法）** を作る回です。

---

## 🎯 この章のゴール

- **「測ってから直す」原則（計測ファースト）** を体得する。最適化の前にまず *正しく測れる* ようにならないと、改善の真偽（速くなったのか、計測がブレただけか）を判定できない。
- **レイテンシ（p50/p99）とスループット（img/s）の違い** を区別し、用途（対話応答 vs バッチ処理）でどちらを最適化すべきか言える。
- **正しいベンチ手順** ＝「ウォームアップ → 多数回反復 → （GPUは同期）→ 分位点で評価」を自分の手で書ける。`time.perf_counter` と `torch.utils.benchmark.Timer` の両方を使える。
- **`model.eval()` と `torch.inference_mode()`** が「結果の正しさ」と「速度・メモリ」の両面で何を変えるかを実測で説明できる。
- **`torch.profiler`** で演算子別の自己CPU時間を集計し（`key_averages().table`）、`record_function` で区間注釈、`export_chrome_trace` でタイムライン化して **律速演算子を特定** できる。
- **数値精度と `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`** を使い、CPU では fp16 ではなく **bf16** が現実的だと（速度と結果のズレを同時に測って）理解する。
- **`torch.compile`（Inductor）** の効果と落とし穴（初回コンパイルコスト・グラフブレイク・動的shape・**C++ツールチェーン依存**）を知り、効くかどうかを必ず実測する。

この章の成果物は **「推論ベンチマーク・ユーティリティ」**（条件別 latency p50/p99・スループット・profiler 律速特定を一枚の表/図/JSON にまとめる）です。`mini_project.py` が完成形として動きます。

---

## 1. 直感 — なぜ「測ってから直す」のか

最適化でいちばん多い失敗は、**測らずに直すこと** です。「`half()` を付けたら速いはず」「`torch.compile` すれば速いはず」——こうした思い込みは、しばしば外れます。CPU では fp16 がむしろ遅いことがあり、`torch.compile` は初回コンパイルが重く、1回しか推論しない用途では割に合いません。直す前に *その変更が本当に効いたか* を判定できないと、最適化は「効いた気がする」で終わってしまいます。だから最初の道具は profiler でも量子化でもなく、**正しいベンチマーク** です。

そして「速い／遅い」を語る前に、**何を速くしたいのか** を決める必要があります。指標は大きく2つ。**レイテンシ**は「1件の入力に答えるまでの時間」で、対話的な応答（カメラ1フレーム、1リクエスト）で効きます。**スループット**は「単位時間に何件さばけるか（img/s, tok/s）」で、大量の画像を一括処理するバッチ用途で効きます。両者はトレードオフの関係にあり、バッチサイズを上げるとレイテンシ（1件の待ち時間）は伸びますが、スループット（全体の処理効率）は上がります。**どちらを最適化するかで打つ手が変わる** ——これを最初に体に入れます。

最後に、ベンチの結果は「1つの数字」ではなく「分布」で見ます。同じ処理を何度測っても、OSのスケジューラやGC、キャッシュの状態でブレます。だから平均（mean）ではなく **分位点** で語ります。**p50（中央値）** は「典型的な速さ」、**p99** は「100回に1回起こる遅さ（最悪寄り）」。本番のサービス品質（SLA）は普通 **p99** で設計します。「たまに遅い」を平均は隠してしまうからです。この章では `01_benchmark_basics.py` で、ウォームアップを省くと p99 が膨れ上がる様子を実際に見せます。

---

## 2. 理論 — 正しいベンチの4原則と eval/inference_mode が変えるもの

正しいベンチマークは、たった4つの原則に集約できます。**(1) ウォームアップ**：初回の呼び出しは、遅延ロード・メモリ確保・キャッシュ miss・（compile 経路なら）JITコンパイルを含むため遅い。最初の数回は計測から **捨てます**。**(2) 多数回反復**：1回計測はブレるので、数十回測って分布を得ます。**(3) 同期**：GPU はカーネルを *非同期* に発行するため、`perf_counter` で挟むだけでは GPU の処理完了を待たずに時刻を読み、**爆速に誤計測** します。計測の前後で `torch.cuda.synchronize()` を呼んで初めて正しい壁時計時間になります（CPU は同期実行なので不要）。**(4) 分位点で評価**：平均ではなく p50/p99 で語る。この4つを `profiling_lab.py` の `benchmark()` に閉じ込め、各スクリプトは「何を比較するか」だけに集中します。

`model.eval()` は **推論結果そのもの** を変えるスイッチです。学習時、**Dropout** はニューロンを確率的に間引き、**BatchNorm** はそのバッチの統計（平均・分散）で正規化します。`eval()` を呼ぶと、Dropout は *素通し*（決定的）に、BN は学習中に蓄積した *移動統計* を使う（バッチに依存しない）モードに切り替わります。つまり `eval()` を忘れると、同じ画像を2回推論しても答えが揺れ、BN もバッチの中身で結果が変わる——これは速度以前の **正しさのバグ** です。`02_eval_inference_mode.py` では、Dropout を持つ小さなネットで「train だと2回の出力が一致せず、eval だと一致する」ことを実測して見せます。

`torch.inference_mode()`（および古くからの `torch.no_grad()`）は **速度とメモリ** のスイッチです。これらの文脈の外で `model(x)` を呼ぶと、PyTorch は逆伝播に備えて **自動微分の計算グラフ** を構築します。推論ではこのグラフは完全に無駄で、時間とメモリを食うだけ。`inference_mode()` はグラフ構築を止め、さらに `no_grad()` より強く（テンソルのバージョン管理なども省くため）軽量です。実測でも `eval() + inference_mode()` は「タダで得られる高速化」になります（本章の計測では grad あり比で 1.3〜1.4 倍）。**推論は必ず `model.eval()` と `torch.inference_mode()` の中で** ——これがすべての出発点です。

---

## 3. 正準 API — プロファイラ・autocast・compile の“最小の正しい使い方”

**`torch.profiler`** は「推測するな、計測せよ」を支える道具です。`profile(activities=[ProfilerActivity.CPU])` の文脈で推論を回し、`record_function("名前")` で自分の関心区間に注釈を付け、`key_averages().table(sort_by="self_cpu_time_total")` で演算子別の時間を降順に並べます。ここで2つの時間を区別します。**self_cpu_time** は「その演算子 *自身* が使った時間（内部で呼ぶ子演算子を含まない）」で、律速の特定にはこれを使います。**cpu_time_total** は「子を含む合計」です。resnet18 を測ると `aten::mkldnn_convolution`（最適化された畳み込み）が支配的だと一目でわかり、「次の一手は畳み込みを狙え（bf16/量子化/ONNX）」という戦略が立ちます。`export_chrome_trace()` で書き出した JSON は `chrome://tracing` や [Perfetto UI](https://ui.perfetto.dev) でタイムラインとして開けます。

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = build_model("resnet18").eval()          # 重み DL 不要のランダム初期化
x = make_input(batch_size=2)

for _ in range(3):                               # ★プロファイルでもウォームアップは要る
    with torch.inference_mode():
        model(x)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("resnet18_inference"):  # 区間に名前を付ける
        for _ in range(10):
            with torch.inference_mode():
                model(x)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")           # Perfetto / chrome://tracing で開く
```

**`torch.autocast`** は混合精度推論を1行で導入します。`with torch.autocast(device_type="cpu", dtype=torch.bfloat16):` の中で推論すると、matmul/conv など対象演算だけが bf16 で実行され、数値的に敏感な箇所は fp32 に保たれます。**CPU では fp16 を使ってはいけません** ——fp16 はソフトウェア寄りの実装で遅い・未対応 op が多く、CPU の低精度は **bf16**（指数幅が fp32 と同じで範囲が広く安定）か **int8**（第35回）が正解です。精度を落とす最適化では、**速度・メモリ** と **結果のズレ**（最大絶対誤差・argmax 一致率）を必ずセットで測ります。本章の計測では bf16 で argmax 一致率 100% を保ちつつ 1.7 倍前後に高速化しました（**効くかは CPU の命令対応 ＝ AVX512-BF16/AMX 次第なので、必ず実測** します）。

```python
with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    y_bf16 = model(x)                            # 対象演算だけ bf16 で実行
# 精度は速度とセットで評価する
with torch.inference_mode():
    y_fp32 = model(x)
max_abs = (y_fp32 - y_bf16.float()).abs().max().item()
argmax_match = (y_fp32.argmax(-1) == y_bf16.argmax(-1)).float().mean().item()
```

**`torch.compile`** は TorchDynamo でグラフを捕捉し、Inductor が C++/OpenMP カーネルを生成・コンパイルして高速化します。CPU でも効くことがありますが、3つの注意があります。**初回コンパイルが重い**（数十秒〜）ので、必ずウォームアップしてから定常状態を測る。**グラフブレイク**（Python の動的分岐や未対応 op でグラフが分断される）と **動的 shape**（入力サイズが変わるたび再コンパイル）で期待した効果が出ないことがある。そして本講座の環境で実際に起きるのが、**C++ ツールチェーン依存** です。Inductor の CPU バックエンドは g++ と **Python 開発ヘッダ（`Python.h`）** を必要とし、最小コンテナ/CI ではこれが無くて *最初の呼び出し時* に `CppCompileError` で失敗します。`05_torch_compile.py` はこれを `try/except` で安全に握りつぶし、失敗したら eager に退避して **exit 0 を保ちます**（現実の事象をそのまま教材化）。

```python
# make_compiled_runner はトリガ呼び出し（＝初回コンパイル）まで含めて try/except する
compiled = torch.compile(model)                  # mode="default"/"reduce-overhead"/"max-autotune"
try:
    with torch.inference_mode():
        compiled(x)                              # ← ここで初めてコンパイルが走る（失敗するならここ）
    # 成功: ウォームアップ後に eager と公平にベンチする
except Exception as e:                            # C++ ツールチェーン/Python.h 不在など
    ...                                          # eager に退避（exit 0 維持）
```

---

## 4. 実装を1つずつ — スクリプトで段階的に組む

各スクリプトは独立に動き、**ネット不要**（resnet18 は `weights=None`、入力は合成）。結果は `outputs/34_inference_profiling/` に保存されます。共通部品は `profiling_lab.py`（device 判定・同期・`benchmark()`・`profile_ops()`・`make_compiled_runner()`・表/図ユーティリティ・純関数群）。

- **`01_benchmark_basics.py` — 正しいベンチの全部**。ウォームアップ有無で同じ処理の p99/mean が変わること、mean vs p50/p99、バッチサイズ↑でレイテンシは伸びるがスループットは上がること、`torch.set_num_threads` で CPU 速度が激変すること、`torch.utils.benchmark.Timer` での答え合わせ（**Timer は既定 num_threads=1** なので揃えて比較）。図 `01_throughput_vs_batch.png`。
- **`02_eval_inference_mode.py` — 正しさと速さの両面**。Dropout を持つ小ネットで train/eval の決定性の差を見せ、`requires_grad` で勾配グラフの有無を確認し、resnet18 で grad / no_grad / inference_mode のレイテンシを比較。図 `02_eval_inference_mode.png`。
- **`03_torch_profiler.py` — 律速を特定する**。`profile` + `record_function` + `key_averages().table`、`group_by_input_shape`、`top_self_cpu_ops` で律速演算子を機械可読に取り出し、`export_chrome_trace` を保存。`aten::mkldnn_convolution` が支配的という事実を「見る」。出力 `03_profiler_table.txt` / `03_chrome_trace.json` / `03_top_ops.png`。
- **`04_autocast_bf16.py` — 数値精度の最適化**。bf16 autocast の速度と結果のズレ（最大絶対誤差・argmax 一致率）を同時に測り、CPU fp16 が遅い/不安定なことを安全に確認。図 `04_autocast_bf16.png`。
- **`05_torch_compile.py` — Inductor の効果と落とし穴**。`make_compiled_runner` で安全にコンパイルを試し（不可なら eager 退避）、初回コンパイルコストとグラフブレイク/動的shape の注意を整理。図 `05_torch_compile.png`。

```python
# profiling_lab.py の道具で“正しいベンチ”を最小コードで
from profiling_lab import build_model, make_input, benchmark, profile_ops, top_self_cpu_ops

model = build_model("resnet18"); x = make_input(batch_size=4)
run = lambda: model(x)                                   # 推論クロージャ（実際は inference_mode 内で）
res = benchmark(run, label="resnet18", batch_size=4, n_warmup=8, n_iter=30)
print(res.p50_ms, res.p99_ms, res.throughput_img_s)      # p50/p99/スループット
prof = profile_ops(run, n_warmup=3, n_iter=10)
print(top_self_cpu_ops(prof, k=5))                       # [(演算子, 自己CPU時間ms), ...]
```

---

## 🛠 章末ミニプロジェクト — 推論ベンチマーク・ユーティリティ

`mini_project.py` が完成形です。任意モデル（既定 resnet18）の推論を **同一手順で公平に** 計測し、次を一気通貫で行います:

1. **計測手順の検証**: ウォームアップ有無だけが違う2条件を測り、省くと p99/mean が汚れることを示す。
2. **最適化条件の公平比較**: `A: train+grad`（推論なのに train＆勾配＝よくある誤り） / `B: eval+inference_mode` / `C: + bf16 autocast` / `D: + torch.compile`（不可なら自動スキップ）の latency p50/p99・スループット・speedup を一枚の表に。
3. **律速演算子の特定**: 条件 B を `torch.profiler` で測り、自己CPU時間トップ5を出す（→ 第35〜37回で狙う先）。
4. **レポート保存**: 全数値と環境情報（torch版・スレッド数・platform）を `mini_report.json`、条件別 p50/p99 横棒を `mini_benchmark.png` に保存。

```bash
uv run python lectures/34_inference_profiling/mini_project.py
```

実行例（CPU・20スレッド）では `A: train+grad → B → C` で `1.00x → 1.42x → 1.77x` と段階的に速くなり、律速は `aten::mkldnn_convolution` でした（環境で数値は変わります）。

**腕試し（発展課題）**: ①`build_model("mobilenet_v3_small")` に差し替え、resnet18 と律速演算子がどう変わるか見る。②`batch_size` を 1 と 16 で回し、レイテンシ最適とスループット最適で結論（最良条件）が変わることを確認する。③`mini_report.json` を読み、複数モデル×条件の speedup ヒートマップを描く。④`torch.set_num_threads` をループに組み込み「スレッド数×バッチ」のスループット表を作る。⑤Dockerfile に `build-essential` と `python3-dev` を入れた環境で再実行し、条件 D（torch.compile）が有効化されるか確かめる。

---

## ✅ 到達チェックリスト

- [ ] 「最適化の前にまず正しく計測する（計測ファースト）」理由を自分の言葉で説明できる。
- [ ] **レイテンシ（p50/p99）** と **スループット（img/s）** の違いと、用途別にどちらを最適化するか言える。
- [ ] 正しいベンチの **4原則**（ウォームアップ・多数回・GPUは同期・分位点）を実装できる。
- [ ] `model.eval()` が **Dropout/BN** を切り替え、忘れると **結果が揺れる**（速度以前の正しさ）と説明できる。
- [ ] `torch.inference_mode()` が **勾配グラフを作らない**（速さ・省メモリ）ことを `requires_grad` で確認できる。
- [ ] `torch.profiler` で **self_cpu_time** 上位＝律速演算子を特定でき、`export_chrome_trace` を Perfetto で開ける。
- [ ] **CPU は fp16 ではなく bf16**（または int8）が正解だと理解し、`autocast` の速度と **結果のズレ** をセットで測れる。
- [ ] `torch.compile` の **初回コスト・グラフブレイク・動的shape・C++ツールチェーン依存** を挙げ、効くかを実測する姿勢がある。
- [ ] **GPU 計測では `torch.cuda.synchronize()` 必須**（忘れると爆速に誤計測）だと知っている。

---

## ❓ 落とし穴・FAQ・デバッグ

- **ウォームアップを省いて誤計測する**: 初回は遅延ロード/キャッシュ miss/（compile なら）JIT を含むので遅い。最初の数回を捨てる。`01` で no-warmup の p99 が膨れる様子を確認。
- **GPU で同期を忘れて「爆速」と誤解する**: CUDA はカーネルを非同期発行する。`perf_counter` の前後で `torch.cuda.synchronize()` を呼ぶ。逆に **CPU で `synchronize()` を呼ぶのは無意味**（no-op）。`sync(device)` のように device を見て分岐する。
- **平均（mean）で語ってブレに騙される**: 外れ値に mean は弱い。p50（典型）と p99（最悪寄り、SLA設計用）で語る。
- **`eval()` を忘れる**: Dropout が確率的なまま・BN がバッチ統計のままになり、推論結果が揺れる。**速度の問題ではなく正しさのバグ**。
- **`inference_mode()`/`no_grad()` を忘れる**: 無駄に勾配グラフを構築して遅く・メモリ食いになる。推論は必ずどちらかの文脈で。
- **CPU で `half()`/fp16 autocast を使って逆に遅くなる**: CPU の低精度は **bf16** か **int8**。fp16 の真価は GPU（Tensor Core）。
- **bf16 autocast が速くならない**: bf16 の高速化は CPU の命令対応（AVX512-BF16 / AMX 等）次第。`autocast を当てれば速い、ではない`。必ず実測する。
- **`torch.compile` が `CppCompileError` で落ちる**: Inductor の CPU バックエンドが g++ と **`Python.h`（python3-dev / python3.12-dev）** を要求する。最小コンテナ/CI で頻発。Dockerfile に `build-essential` と `python3-dev` を入れると解決。入れられない環境では eager に退避する（本章の `make_compiled_runner` の方針）。
- **`torch.compile` の初回が異常に遅い**: 初回はコンパイル時間を含む。必ずウォームアップ後の定常状態を測る。1回しか推論しないなら割に合わない（サーバ常駐向き）。
- **`torch.utils.benchmark.Timer` の数字が自前ループと食い違う**: Timer は **既定 `num_threads=1`**。自前ループ（既定スレッド数）と比べるなら `num_threads=torch.get_num_threads()` を渡して揃える。
- **profiler のテーブルが空/薄い**: ウォームアップ前に測っている、あるいは反復回数が少ない。`profile_ops` のように温めてから複数回回す。`record_shapes=True` や `group_by_input_shape=True` で shape 別に割ると分析しやすい。
- **スループットをバッチ1のまま比較して「遅い」と誤結論**: 並列性を活かせていないだけ。バッチを上げてスループットを測る（`01` の曲線）。

---

## 🚀 発展トピック・参考

- **次の一手（このトラックの地図）**: 本章で律速（多くは畳み込み/行列積）を特定したら、第35回で **量子化（int8）・枝刈り**、第36回で **ONNX エクスポート + onnxruntime（CPUで最も手軽に効く int8 動的量子化）**、第37回で **OpenVINO/CoreML/LiteRT/TensorRT** とランタイム選択へ進む。推奨の意思決定順序は **eval+inference_mode → bf16+torch.compile → ONNX動的量子化 → 静的PTQ → 必要なら QAT/pruning**。
- **`torch.profiler` の可視化**: `export_chrome_trace` の JSON を [Perfetto UI](https://ui.perfetto.dev) で開くとタイムラインで「どの演算がいつ走ったか」が見える。TensorBoard プラグイン（`torch.profiler.tensorboard_trace_handler`）でステップ単位の集計も可能。
- **`torch.compile` の調査**: `fullgraph=True`（グラフブレイクで例外化して原因を炙り出す）、`TORCH_LOGS="graph_breaks"` / `dynamic=False`（再コンパイル抑制）、`mode="reduce-overhead"`（起動オーバヘッド削減）/ `"max-autotune"`（探索に時間をかけ最速狙い）。
- **GPU を使う受講者向けの差分**: 計測の前後で `torch.cuda.synchronize()` を必ず入れる。低精度は fp16/bf16 の両方が選択肢（Tensor Core）。`Timer` は GPU でも同期を内部処理してくれる。
- **公式ドキュメント**: [torch.profiler](https://docs.pytorch.org/docs/stable/profiler.html) / [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html) / [Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html) / [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html)。

---

## ▶ 動かし方

```bash
# 依存（未導入なら）: 深層学習の土台（CPU 版 torch / torchvision）
uv sync --group dl

# 本編（番号順）。ネット不要・CPUのみで数秒〜十数秒
uv run python lectures/34_inference_profiling/01_benchmark_basics.py
uv run python lectures/34_inference_profiling/02_eval_inference_mode.py
uv run python lectures/34_inference_profiling/03_torch_profiler.py
uv run python lectures/34_inference_profiling/04_autocast_bf16.py
uv run python lectures/34_inference_profiling/05_torch_compile.py   # compile 不可環境でも exit 0

# 章末ミニプロジェクト（推論ベンチマーク・ユーティリティの完成形）
uv run python lectures/34_inference_profiling/mini_project.py

# 演習（自己採点。未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/34_inference_profiling/exercises.py
uv run python lectures/34_inference_profiling/exercises_solutions.py
```

成果物（図・JSON・chrome trace・profiler テーブル）は `outputs/34_inference_profiling/` に保存される。

> **補足（torch.compile を有効化したい場合）**: Inductor の CPU バックエンドには C++ コンパイラと Python 開発ヘッダが要る。Debian/Ubuntu 系なら `apt-get install build-essential python3-dev`（本講座の Python 3.12 なら `python3.12-dev`）。これらが無い環境では `05`/`mini_project` は自動で eager に退避する（落ちない）。

---

> 参照ライブラリ: **torch 2.12+cpu** / **torchvision 0.27+cpu**（題材 resnet18）/ **onnx 1.21** / **onnxruntime 1.26**（本章では未使用、第36回で使用）
> （headless OpenCV は本章では未使用、matplotlib=Agg、CPU・`model.eval()`+`torch.inference_mode()`、torch.compile は C++ ツールチェーン不在環境では eager 退避） — 2026-06
