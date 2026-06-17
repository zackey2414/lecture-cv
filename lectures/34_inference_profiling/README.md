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

最適化でいちばん多い失敗は、**測らずに直すこと** です。「`half()` を付ければ速いはず」「`torch.compile` すれば速いはず」——こうした思い込みは、しばしば外れます。実際、CPU では fp16 がむしろ遅いことがありますし、`torch.compile` も初回コンパイルが重いため、1回しか推論しない用途では割に合いません。そして、直す前に *その変更が本当に効いたのか* を判定できなければ、最適化は「効いた気がする」で終わってしまいます。だからこそ、最初に手にすべき道具は profiler でも量子化でもなく、**正しいベンチマーク** なのです。

もっとも、「速い／遅い」を語る前に、**何を速くしたいのか** を決めておく必要があります。指標は大きく2つです。**レイテンシ**は「1件の入力に答えるまでの時間」を指し、対話的な応答（カメラ1フレーム、1リクエスト）で効きます。一方の**スループット**は「単位時間に何件さばけるか（img/s, tok/s）」を指し、大量の画像を一括処理するバッチ用途で効きます。両者はトレードオフの関係にあり、バッチサイズを上げると、レイテンシ（1件の待ち時間）は伸びる一方でスループット（全体の処理効率）は上がります。つまり、**どちらを最適化するかで打つ手が変わる** ——これを最初に体に入れます。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="バッチを上げるとレイテンシは伸びスループットは上がって飽和するトレードオフのグラフ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">バッチを上げると両者はトレードオフ</text><rect x="70" y="56" width="490" height="184" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><line x1="70" y1="240" x2="562" y2="240" stroke="#71717a" stroke-width="2"/><polygon points="570,240 560,235 560,245" fill="#71717a"/><line x1="70" y1="240" x2="70" y2="48" stroke="#71717a" stroke-width="2"/><polygon points="70,40 65,50 75,50" fill="#71717a"/><polyline points="100,228 180,190 280,146 400,120 520,110" fill="none" stroke="#2563eb" stroke-width="3"/><polyline points="100,214 180,196 280,166 400,118 520,70" fill="none" stroke="#ea580c" stroke-width="3"/><circle cx="520" cy="70" r="5" fill="#ea580c"/><circle cx="520" cy="110" r="5" fill="#2563eb"/><line x1="92" y1="74" x2="116" y2="74" stroke="#ea580c" stroke-width="3"/><text x="122" y="78" font-size="12.5" font-weight="700" fill="#c2410c">レイテンシ（p99 ↑）</text><line x1="92" y1="96" x2="116" y2="96" stroke="#2563eb" stroke-width="3"/><text x="122" y="100" font-size="12.5" font-weight="700" fill="#1d4ed8">スループット（img/s ↑・飽和）</text><text x="315" y="266" text-anchor="middle" font-size="14" fill="#3f3f46">バッチサイズ →</text><text x="100" y="258" text-anchor="middle" font-size="12" fill="#71717a">1</text><text x="520" y="258" text-anchor="middle" font-size="12" fill="#71717a">16</text></svg><figcaption>バッチサイズを上げると、<b>レイテンシ</b>（1件に答えるまでの待ち時間 / p99）は伸びる一方で、<b>スループット</b>（単位時間あたりの処理枚数 img/s）は上がり、やがて<b>飽和</b>します。両者は<b>トレードオフ</b>の関係にあり、<b>対話応答</b>ならレイテンシを、<b>バッチ処理</b>ならスループットを最適化します。</figcaption></figure>

最後に、ベンチの結果は「1つの数字」ではなく「分布」として見ます。というのも、同じ処理を何度測っても、OSのスケジューラやGC、キャッシュの状態によって値はブレるからです。そのため、平均（mean）ではなく **分位点** で語ります。たとえば **p50（中央値）** は「典型的な速さ」を、**p99** は「100回に1回起こる遅さ（最悪寄り）」を表します。本番のサービス品質（SLA）を普通 **p99** で設計するのは、「たまに遅い」を平均が隠してしまうからです。この章では `01_benchmark_basics.py` を通じて、ウォームアップを省くと p99 が膨れ上がる様子を実際に見せます。

---

## 2. 理論 — 正しいベンチの4原則と eval/inference_mode が変えるもの

正しいベンチマークは、たった4つの原則に集約できます。**(1) ウォームアップ**：初回の呼び出しは、遅延ロード・メモリ確保・キャッシュ miss・（compile 経路なら）JITコンパイルを含むため遅い。最初の数回は計測から **捨てます**。**(2) 多数回反復**：1回計測はブレるので、数十回測って分布を得ます。**(3) 同期**：GPU はカーネルを *非同期* に発行するため、`perf_counter` で挟むだけでは GPU の処理完了を待たずに時刻を読み、**爆速に誤計測** します。計測の前後で `torch.cuda.synchronize()` を呼んで初めて正しい壁時計時間になります（CPU は同期実行なので不要）。**(4) 分位点で評価**：平均ではなく p50/p99 で語る。この4つを `profiling_lab.py` の `benchmark()` に閉じ込めておけば、各スクリプトは「何を比較するか」だけに集中できます。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="ウォームアップで最初の数回を捨て多数回計測して分布のp50とp99を見るベンチ手順" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">温めて → 多数回 → 分位点で見る</text><line x1="42" y1="210" x2="312" y2="210" stroke="#71717a" stroke-width="1.5"/><rect x="48" y="115" width="18" height="95" fill="#d4d4d8" stroke="#71717a" stroke-width="1.2" stroke-dasharray="3 2"/><rect x="72" y="144" width="18" height="66" fill="#d4d4d8" stroke="#71717a" stroke-width="1.2" stroke-dasharray="3 2"/><rect x="96" y="166" width="18" height="44" fill="#d4d4d8" stroke="#71717a" stroke-width="1.2" stroke-dasharray="3 2"/><rect x="120" y="186" width="18" height="24" fill="#f97316"/><rect x="144" y="190" width="18" height="20" fill="#f97316"/><rect x="168" y="184" width="18" height="26" fill="#f97316"/><rect x="192" y="188" width="18" height="22" fill="#f97316"/><rect x="216" y="170" width="18" height="40" fill="#f97316"/><rect x="240" y="189" width="18" height="21" fill="#f97316"/><rect x="264" y="185" width="18" height="25" fill="#f97316"/><rect x="288" y="187" width="18" height="23" fill="#f97316"/><text x="84" y="104" text-anchor="middle" font-size="13" fill="#52525b">ウォームアップ</text><text x="207" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">計測（多数回）</text><line x1="318" y1="180" x2="350" y2="180" stroke="#71717a" stroke-width="3"/><polygon points="362,180 350,173 350,187" fill="#71717a"/><line x1="372" y1="210" x2="612" y2="210" stroke="#71717a" stroke-width="1.5"/><polygon points="620,210 610,205 610,215" fill="#71717a"/><polygon points="378,210 396,205 410,182 426,138 442,160 466,186 500,199 545,205 585,208 600,210" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><line x1="426" y1="210" x2="426" y2="132" stroke="#16a34a" stroke-width="2.5"/><text x="420" y="122" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">p50（典型）</text><line x1="545" y1="210" x2="545" y2="176" stroke="#dc2626" stroke-width="2.5"/><text x="608" y="168" text-anchor="end" font-size="12.5" font-weight="700" fill="#dc2626">p99（最悪寄り）</text><text x="495" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">レイテンシ →</text></svg><figcaption>正しいベンチの作法です。最初の数回は遅延ロードやキャッシュの影響で遅いので<b>ウォームアップとして捨て</b>、その後を<b>多数回</b>反復して<b>分布</b>として見ます。代表値は平均ではなく<b>分位点</b>で語り、<b>p50</b>（典型の速さ）と <b>p99</b>（100回に1回の遅さ＝SLA設計用）を読みます。<br>GPU では計測の前後で <code>torch.cuda.synchronize()</code> が必須です（忘れると爆速に誤計測）。</figcaption></figure>

`model.eval()` は **推論結果そのもの** を変えるスイッチです。学習時、**Dropout** はニューロンを確率的に間引き、**BatchNorm** はそのバッチの統計（平均・分散）で正規化します。ここで `eval()` を呼ぶと、Dropout は *素通し*（決定的）に、BN は学習中に蓄積した *移動統計* を使う（バッチに依存しない）モードへ切り替わります。逆に言えば、`eval()` を忘れると、同じ画像を2回推論しても答えが揺れ、BN もバッチの中身で結果が変わってしまう——これは速度以前の **正しさのバグ** です。`02_eval_inference_mode.py` では、Dropout を持つ小さなネットを使い、「train だと2回の出力が一致せず、eval だと一致する」ことを実測して見せます。

一方、`torch.inference_mode()`（および古くからの `torch.no_grad()`）は **速度とメモリ** のスイッチです。これらの文脈の外で `model(x)` を呼ぶと、PyTorch は逆伝播に備えて **自動微分の計算グラフ** を構築します。しかし推論ではこのグラフは完全に無駄で、時間とメモリを食うだけです。`inference_mode()` はそのグラフ構築を止め、しかも `no_grad()` よりさらに強く（テンソルのバージョン管理なども省くため）軽量です。実測でも `eval() + inference_mode()` は「タダで得られる高速化」になります（本章の計測では grad あり比で 1.3〜1.4 倍）。だからこそ、**推論は必ず `model.eval()` と `torch.inference_mode()` の中で** ——これがすべての出発点です。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="model.evalは結果の正しさ、torch.inference_modeは速さと省メモリを担う別々のスイッチ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="28" y="58" width="276" height="150" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="166" y="90" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">model.eval()</text><text x="166" y="118" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">＝ 結果の正しさ</text><text x="166" y="160" text-anchor="middle" font-size="12.5" fill="#52525b">Dropout→素通し ・ BN→移動統計</text><rect x="336" y="58" width="276" height="150" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="474" y="90" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">torch.inference_mode()</text><text x="474" y="118" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">＝ 速さ・省メモリ</text><text x="474" y="160" text-anchor="middle" font-size="12.5" fill="#52525b">自動微分グラフを作らない</text><rect x="308" y="128" width="24" height="6" fill="#52525b"/><rect x="317" y="119" width="6" height="24" fill="#52525b"/><rect x="28" y="232" width="584" height="46" rx="8" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="320" y="260" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">両方そろえて『正しく・速い』推論</text></svg><figcaption><code>model.eval()</code> と <code>torch.inference_mode()</code> は<b>別々の役割</b>を持つ2つのスイッチです。<b>eval()</b> は <b>Dropout / BatchNorm</b> を推論モードへ切り替えて<b>結果の正しさ</b>を保ち（忘れると同じ入力でも出力が揺れる）、<b>inference_mode()</b> は<b>自動微分グラフを作らず</b>に<b>速度とメモリ</b>を改善します。推論ではこの<b>両方</b>を必ず併用します。</figcaption></figure>

---

## 3. 正準 API — プロファイラ・autocast・compile の“最小の正しい使い方”

**`torch.profiler`** は「推測するな、計測せよ」を支える道具です。`profile(activities=[ProfilerActivity.CPU])` の文脈で推論を回し、`record_function("名前")` で自分の関心区間に注釈を付け、`key_averages().table(sort_by="self_cpu_time_total")` で演算子別の時間を降順に並べます。このとき、2種類の時間を区別することが大切です。**self_cpu_time** は「その演算子 *自身* が使った時間（内部で呼ぶ子演算子を含まない）」で、律速の特定にはこちらを使います。一方、**cpu_time_total** は「子を含む合計」です。実際に resnet18 を測ると `aten::mkldnn_convolution`（最適化された畳み込み）が支配的だと一目でわかり、「次の一手は畳み込みを狙え（bf16/量子化/ONNX）」という戦略が立ちます。なお、`export_chrome_trace()` で書き出した JSON は `chrome://tracing` や [Perfetto UI](https://ui.perfetto.dev) でタイムラインとして開けます。

<figure class="lec-fig"><svg viewBox="0 0 580 300" role="img" aria-label="cpu_time_totalは子を含む合計、self_cpu_timeは自分だけの時間で律速の特定に使う" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="290" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">プロファイラの self と total</text><polyline points="40,58 40,52 540,52 540,58" fill="none" stroke="#2563eb" stroke-width="1.5"/><text x="290" y="44" text-anchor="middle" font-size="12.5" fill="#1d4ed8">cpu_time_total ＝ 区間の合計（自分＋子）</text><rect x="40" y="66" width="500" height="40" rx="4" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="290" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">resnet18_inference（区間）</text><rect x="40" y="124" width="280" height="46" rx="4" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><text x="180" y="152" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">aten::mkldnn_convolution</text><rect x="320" y="124" width="220" height="46" rx="4" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="430" y="152" text-anchor="middle" font-size="12.5" fill="#52525b">relu・add・他</text><polyline points="40,180 40,186 320,186 320,180" fill="none" stroke="#c2410c" stroke-width="1.5"/><text x="180" y="206" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">self_cpu_time 大 ＝ 律速（畳み込み）</text><text x="290" y="250" text-anchor="middle" font-size="12.5" fill="#3f3f46">→ 次の一手は畳み込みを狙う（bf16 / 量子化 / ONNX）</text></svg><figcaption>プロファイラの2つの時間の違いです。<b>cpu_time_total</b> は「その区間が使った合計時間（<b>自分＋呼び出した子演算子</b>）」で、行全体の幅にあたります。<b>self_cpu_time</b> は「その演算子<b>自身だけ</b>が使った時間」で、<b>律速の特定にはこちら</b>を使います。resnet18 では <code>aten::mkldnn_convolution</code>（最適化された畳み込み）の self が支配的なので、次の一手は畳み込みを狙います。</figcaption></figure>

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

**`torch.autocast`** は混合精度推論を1行で導入できます。`with torch.autocast(device_type="cpu", dtype=torch.bfloat16):` の中で推論すると、matmul/conv など対象演算だけが bf16 で実行され、数値的に敏感な箇所は fp32 に保たれます。ただし、**CPU では fp16 を使ってはいけません** ——fp16 はソフトウェア寄りの実装で遅く、未対応の op も多いからです。CPU の低精度は **bf16**（指数幅が fp32 と同じで範囲が広く安定）か **int8**（第35回）が正解です。また、精度を落とす最適化では、**速度・メモリ** と **結果のズレ**（最大絶対誤差・argmax 一致率）を必ずセットで測ります。本章の計測では bf16 で argmax 一致率 100% を保ちつつ 1.7 倍前後に高速化できました（ただし **効くかは CPU の命令対応 ＝ AVX512-BF16/AMX 次第なので、必ず実測** します）。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="fp32 bf16 fp16のビット配置。bf16は指数8bitでfp32と同じ範囲、fp16は指数5bitで範囲が狭い" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">CPUは fp16 より bf16（指数部に注目）</text><text x="190" y="62" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">指数部 ＝ 範囲</text><text x="376" y="62" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">仮数部 ＝ 精度</text><rect x="130" y="74" width="12" height="40" fill="#52525b"/><rect x="142" y="74" width="96" height="40" fill="#f97316"/><rect x="238" y="74" width="276" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="130" y="144" width="12" height="40" fill="#52525b"/><rect x="142" y="144" width="96" height="40" fill="#f97316"/><rect x="238" y="144" width="84" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="130" y="214" width="12" height="40" fill="#52525b"/><rect x="142" y="214" width="60" height="40" fill="#f97316"/><rect x="202" y="214" width="120" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><line x1="238" y1="70" x2="238" y2="262" stroke="#c2410c" stroke-width="1.4" stroke-dasharray="5 3" opacity="0.7"/><text x="120" y="99" text-anchor="end" font-size="15" font-weight="700" fill="#18181b">fp32</text><text x="120" y="169" text-anchor="end" font-size="15" font-weight="700" fill="#18181b">bf16</text><text x="120" y="239" text-anchor="end" font-size="15" font-weight="700" fill="#18181b">fp16</text><text x="335" y="240" font-size="12" font-weight="700" fill="#dc2626">← 指数 5bit ＝ 範囲が狭い</text></svg><figcaption>浮動小数点のビット配置です（色: <b>符号＝灰 / 指数部＝橙 / 仮数部＝青</b>）。<b>指数部</b>が<b>表せる範囲</b>を、<b>仮数部</b>が<b>精度</b>を決めます。<b>bf16</b> は指数部が <b>8bit</b> と <b>fp32 と同じ</b>幅なので範囲が広く安定し、CPUでも安全です。一方 <b>fp16</b> は指数部が <b>5bit</b> と狭く範囲が小さいため、CPU では<b>遅く・不安定</b>になりがちです。だから CPU の低精度は <code>bf16</code>（または int8）を選びます。</figcaption></figure>

```python
with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    y_bf16 = model(x)                            # 対象演算だけ bf16 で実行
# 精度は速度とセットで評価する
with torch.inference_mode():
    y_fp32 = model(x)
max_abs = (y_fp32 - y_bf16.float()).abs().max().item()
argmax_match = (y_fp32.argmax(-1) == y_bf16.argmax(-1)).float().mean().item()
```

**`torch.compile`** は TorchDynamo でグラフを捕捉し、Inductor が C++/OpenMP カーネルを生成・コンパイルして高速化します。CPU でも効くことがありますが、注意点は3つあります。第一に、**初回コンパイルが重い**（数十秒〜）ので、必ずウォームアップしてから定常状態を測ること。第二に、**グラフブレイク**（Python の動的分岐や未対応 op でグラフが分断される）や **動的 shape**（入力サイズが変わるたび再コンパイル）によって、期待した効果が出ないことがあること。そして第三に、本講座の環境で実際に起きるのが **C++ ツールチェーン依存** です。Inductor の CPU バックエンドは g++ と **Python 開発ヘッダ（`Python.h`）** を必要とするため、最小コンテナ/CI ではこれが無く、*最初の呼び出し時* に `CppCompileError` で失敗します。そこで `05_torch_compile.py` はこれを `try/except` で安全に握りつぶし、失敗したら eager に退避して **exit 0 を保ちます**（現実の事象をそのまま教材化しています）。

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

各スクリプトは独立に動き、**ネット不要** です（resnet18 は `weights=None`、入力は合成）。結果は `lectures/34_inference_profiling/outputs/` に保存されます。共通部品は `profiling_lab.py` にまとめてあります（device 判定・同期・`benchmark()`・`profile_ops()`・`make_compiled_runner()`・表/図ユーティリティ・純関数群）。

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

<figure class="lec-fig"><svg viewBox="0 0 640 260" role="img" aria-label="章末ミニプロジェクトの4ステップ。計測手順の検証から最適化条件の比較、律速演算子の特定を経てレポート保存へ流れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="34" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">ミニプロジェクト — 4 ステップで一気通貫</text><rect x="16" y="78" width="128" height="96" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="176" y="78" width="128" height="96" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="336" y="78" width="128" height="96" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="496" y="78" width="128" height="96" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="80" y="110" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">①</text><text x="80" y="136" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">計測手順の検証</text><text x="80" y="157" text-anchor="middle" font-size="10.5" fill="#71717a">warmup 有無で p99</text><text x="240" y="110" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">②</text><text x="240" y="136" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">最適化条件の比較</text><text x="240" y="157" text-anchor="middle" font-size="10.5" fill="#71717a">A／B／C／D</text><text x="400" y="110" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">③</text><text x="400" y="136" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">律速演算子の特定</text><text x="400" y="157" text-anchor="middle" font-size="10.5" fill="#71717a">自己CPU時間 top5</text><text x="560" y="110" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">④</text><text x="560" y="136" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">レポート保存</text><text x="560" y="157" text-anchor="middle" font-size="10.5" fill="#71717a">JSON ＋ 横棒図</text><line x1="144" y1="126" x2="164" y2="126" stroke="#71717a" stroke-width="2"/><polygon points="175,126 165,121 165,131" fill="#71717a"/><line x1="304" y1="126" x2="324" y2="126" stroke="#71717a" stroke-width="2"/><polygon points="335,126 325,121 325,131" fill="#71717a"/><line x1="464" y1="126" x2="484" y2="126" stroke="#71717a" stroke-width="2"/><polygon points="495,126 485,121 485,131" fill="#71717a"/></svg><figcaption><b>章末ミニプロジェクト</b>は入口の1枚を、<b>① 計測手順の検証</b>（ウォームアップ有無で p99 が汚れるか）<b>→ ② 最適化条件の比較</b>（A／B／C／D を同一手順で公平に）<b>→ ③ 律速演算子の特定</b>（profiler の自己CPU時間トップ5）<b>→ ④ レポート保存</b>（<code>mini_report.json</code> と条件別 p50/p99 の横棒図）の順に一気通貫で流します。①〜③が計測、青い<b>④</b>が結果を JSON と図へ束ねる出力ステップです。</figcaption></figure>

```bash
uv run python lectures/34_inference_profiling/mini_project.py
```

実行例（CPU・20スレッド）では、`A: train+grad → B → C` の順に `1.00x → 1.42x → 1.77x` と段階的に速くなり、律速は `aten::mkldnn_convolution` でした（数値は環境によって変わります）。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="最適化条件A train+gradからB eval+inference mode、C bf16、D torch.compileへ積み増すほど速くなる累積speedup" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">条件を積み増すほど速い（累積 speedup）</text><rect x="26" y="70" width="116" height="92" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><rect x="190" y="70" width="116" height="92" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="354" y="70" width="116" height="92" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><rect x="518" y="70" width="116" height="92" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2" stroke-dasharray="5 3"/><text x="84" y="98" text-anchor="middle" font-size="11.5" font-weight="700" fill="#52525b">A：train+grad</text><text x="84" y="133" text-anchor="middle" font-size="19" font-weight="700" fill="#18181b">1.00x</text><text x="84" y="153" text-anchor="middle" font-size="10" fill="#dc2626">基準・誤用</text><text x="248" y="98" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">B：eval+infer</text><text x="248" y="133" text-anchor="middle" font-size="19" font-weight="700" fill="#c2410c">1.42x</text><text x="248" y="153" text-anchor="middle" font-size="10" fill="#71717a">推論の基本</text><text x="412" y="98" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">C：＋bf16</text><text x="412" y="133" text-anchor="middle" font-size="19" font-weight="700" fill="#c2410c">1.77x</text><text x="412" y="153" text-anchor="middle" font-size="10" fill="#71717a">混合精度</text><text x="576" y="98" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1d4ed8">D：＋compile</text><text x="576" y="132" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">環境次第</text><text x="576" y="153" text-anchor="middle" font-size="10" fill="#71717a">C++ 依存</text><text x="166" y="104" text-anchor="middle" font-size="10.5" fill="#3f3f46">eval</text><text x="330" y="104" text-anchor="middle" font-size="10.5" fill="#3f3f46">bf16</text><text x="494" y="104" text-anchor="middle" font-size="10.5" fill="#3f3f46">compile</text><line x1="142" y1="116" x2="178" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="189,116 179,111 179,121" fill="#71717a"/><line x1="306" y1="116" x2="342" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="353,116 343,111 343,121" fill="#71717a"/><line x1="470" y1="116" x2="506" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="517,116 507,111 507,121" fill="#71717a"/><line x1="70" y1="208" x2="596" y2="208" stroke="#16a34a" stroke-width="2.5"/><polygon points="606,208 596,203 596,213" fill="#16a34a"/><text x="333" y="231" text-anchor="middle" font-size="12.5" fill="#15803d">条件を積み増すほど速い（speedup ↑）</text></svg><figcaption>推論ベンチマークの<b>核心</b>は、最適化を積み増すほど速くなる様子を<b>同一手順で公平に</b>比べることです。<b>A：train+grad</b>（推論なのに学習＆勾配＝よくある誤り。<b>1.00x</b> の基準）<b>→ B：eval + inference_mode</b>（<b>1.42x</b>）<b>→ C：＋bf16 autocast</b>（<b>1.77x</b>）<b>→ D：＋torch.compile</b>（C++ ツールチェーン依存で<b>環境次第</b>・不可なら自動スキップ）。数値は環境によって変わります。</figcaption></figure>

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

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/34_inference_profiling/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. レイテンシ列の指定パーセンタイル（p50/p99 など）を `np.percentile` の線形補間に一致するように返す（`ex1_percentile` の TODO）。平均ではなく分位点で語るのがベンチの作法。
2. 1反復のレイテンシ（秒）とバッチ枚数から スループット（img/s ＝ `batch_size / latency_s`）を求める（`ex2_throughput` の TODO）。
3. 計測列の先頭 `n_warmup` 個（ウォームアップ）を捨て、残りの列を返す（`ex3_drop_warmup` の TODO）。
4. baseline と candidate のレイテンシから speedup（`baseline / candidate`、>1 で高速化）を計算する（`ex4_speedup` の TODO）。
5. レイテンシ列から `p50`/`p99`/`mean` をまとめた dict を作って返す（`ex5_summarize` の TODO）。
6. 「ウォームアップ→反復計測→中央値」を行い、中央値（ms）と計測した反復数を返すベンチ本体（`ex6_run_benchmark` の TODO）。`time.perf_counter` で各反復を挟む。
7. （演算子名, 自己CPU時間）のリストを時間の降順に並べ、上位 k 件（律速演算子）を取り出す（`ex7_top_ops` の TODO）。
8. `torch.inference_mode()` 下で `model(x)` を実行し、出力が勾配を追跡していないか（`requires_grad` が False か）を bool で返す（`ex8_no_grad_output` の TODO）。
9. 条件名→p50(ms) の dict から speedup 付きの比較行リストを作り、p50 の昇順（速い順）に並べる（baseline 自身の speedup は 1.0）（`ex9_comparison_table` の TODO）。

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

<figure class="lec-fig"><svg viewBox="0 0 620 250" role="img" aria-label="最適化トラックの意思決定順序。eval inference modeからbf16 compile、ONNX動的量子化、静的PTQ、QAT枝刈りへ手軽な順に試す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="310" y="32" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">最適化トラックの意思決定順序（手軽な順に試す）</text><rect x="14" y="78" width="100" height="92" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="137" y="78" width="100" height="92" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><rect x="260" y="78" width="100" height="92" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><rect x="383" y="78" width="100" height="92" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="506" y="78" width="100" height="92" rx="8" fill="#f4f4f5" stroke="#52525b" stroke-width="2"/><text x="64" y="106" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">eval +</text><text x="64" y="128" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">inference_mode</text><text x="64" y="151" text-anchor="middle" font-size="9.5" fill="#71717a">本章・即効</text><text x="187" y="106" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">bf16 +</text><text x="187" y="128" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">torch.compile</text><text x="187" y="151" text-anchor="middle" font-size="9.5" fill="#71717a">本章</text><text x="310" y="106" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">ONNX</text><text x="310" y="128" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">動的量子化</text><text x="310" y="151" text-anchor="middle" font-size="9.5" fill="#71717a">第36回・手軽</text><text x="433" y="106" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">静的 PTQ</text><text x="433" y="128" text-anchor="middle" font-size="11" fill="#3f3f46">校正データ要</text><text x="433" y="151" text-anchor="middle" font-size="9.5" fill="#71717a">第35/36回</text><text x="556" y="106" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3f3f46">QAT・枝刈り</text><text x="556" y="128" text-anchor="middle" font-size="11" fill="#3f3f46">最終手段</text><text x="556" y="151" text-anchor="middle" font-size="9.5" fill="#71717a">第35回</text><line x1="114" y1="124" x2="126" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="136,124 126,119 126,129" fill="#71717a"/><line x1="237" y1="124" x2="249" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="259,124 249,119 249,129" fill="#71717a"/><line x1="360" y1="124" x2="372" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="382,124 372,119 372,129" fill="#71717a"/><line x1="483" y1="124" x2="495" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="505,124 495,119 495,129" fill="#71717a"/><text x="310" y="214" text-anchor="middle" font-size="12.5" fill="#3f3f46">左ほど手軽・即効 ／ 右ほど効果大・手間大</text></svg><figcaption>本章で律速を特定したら、最適化は<b>手軽で即効なものから順に</b>試すのが定石です。<b>① eval + inference_mode</b>（本章・タダで速い）<b>→ ② bf16 + torch.compile</b>（本章）<b>→ ③ ONNX 動的量子化</b>（第36回・CPUで最も手軽に効く）<b>→ ④ 静的 PTQ</b>（校正データが要る）<b>→ ⑤ QAT・枝刈り</b>（手間が大きいので必要なときだけ）。<b>左ほど手軽</b>・<b>右ほど効果は大きいが手間も増える</b>、という順序です。</figcaption></figure>

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

成果物（図・JSON・chrome trace・profiler テーブル）は `lectures/34_inference_profiling/outputs/` に保存される。

> **補足（torch.compile を有効化したい場合）**: Inductor の CPU バックエンドには C++ コンパイラと Python 開発ヘッダが要る。Debian/Ubuntu 系なら `apt-get install build-essential python3-dev`（本講座の Python 3.12 なら `python3.12-dev`）。これらが無い環境では `05`/`mini_project` は自動で eager に退避する（落ちない）。

---

> 参照ライブラリ: **torch 2.12+cpu** / **torchvision 0.27+cpu**（題材 resnet18）/ **onnx 1.21** / **onnxruntime 1.26**（本章では未使用、第36回で使用）
> （headless OpenCV は本章では未使用、matplotlib=Agg、CPU・`model.eval()`+`torch.inference_mode()`、torch.compile は C++ ツールチェーン不在環境では eager 退避） — 2026-06
