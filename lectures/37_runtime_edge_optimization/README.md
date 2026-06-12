# 37_runtime_edge_optimization: ランタイム/エッジ最適化 — TorchScript・torch.compile・ONNX Runtime・OpenVINO/CoreML/LiteRT/TensorRT

> トラック: **最適化・デプロイ** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `onnx`
> 前提: `36_onnx_runtime`(ONNX化と数値検証)・`35_quantization_pruning`(量子化/枝刈り)・`34_inference_profiling`(計測ファースト)

---

## 🎯 この章のゴール

学習済みモデルを「速く・小さく・正確なまま」本番やエッジに届けるには、**どのランタイムへ・どの順番で**最適化を積むかという地図が要ります。この章のゴールは次の5つを身につけることです。

1. **TorchScript(trace / script)** でモデルを Python 非依存の `.pt` に固め、配布形にできる。trace と script の違い(「実行をなぞる」vs「ソースを解析する」)と、データ依存分岐での**trace の罠**を説明できる。
2. **torch.compile(Inductor)** の仕組み(捕捉→生成→融合)を理解し、初回コンパイルコスト・グラフブレイク・C++ ツールチェイン依存をふまえて**ガードして**使える。
3. **ONNX Runtime をランタイムとして** 使い、グラフ最適化・スレッド設定・int8 動的量子化まで含めて eager torch と公平にベンチできる。
4. **エッジ/プラットフォーム別ランタイム**(OpenVINO / CoreML / LiteRT / TensorRT)の立ち位置・使い方・CPUのみ環境での実習可否を整理し、重い依存を**任意ガード**で扱える。
5. 速度・サイズ・精度の三角関係を**意思決定の順序**(eval+inference_mode → スレッド → bf16+compile → ONNX → 動的量子化 → 静的PTQ → QAT/pruning → エッジ専用)に落とし、目標到達で止められる。

この章は34〜36の集大成です。「計測ファースト(34)」「量子化/枝刈り(35)」「ONNX化(36)」で得た部品を、**複数ランタイムの横並びベンチ**と**用途別の意思決定**へ統合します。

---


## 1. ランタイムの地図 — なぜ「出口」を選ぶのか

学習が終わったモデルは、そのまま `model(x)` で動かす(eager 実行)のが一番手軽です。しかし eager は1演算ずつ Python を経由し、メモリ往復も多く、配布には Python とソースコードが要ります。本番やエッジでは「Python 非依存」「グラフ最適化済み」「ターゲットのハードに最適化済み」のランタイムへ**変換して**載せ替えるのが定石です。これが「出口(deployment target)を選ぶ」ということです。

出口は大きく分けて次のとおりです。**サーバCPU / Intel** なら OpenVINO(と ONNX Runtime)、**Mac(Apple Silicon)** なら CoreML、**モバイル(Android/組込)** なら LiteRT(`.tflite`)、**NVIDIA GPU** なら TensorRT。どれも「PyTorch → 中間表現(ONNX か TorchScript)→ ターゲット形式」という共通の流れで、中間ハブに ONNX を据えると移植性が高まります。重要なのは、**変換した瞬間に「速くなった」と思い込まない**こと。変換後は必ず①元モデルとの数値一致(atol/rtol)と②正しい手順でのベンチを取り、速度・サイズ・精度の3つを同時に確認します。

<figure class="lec-fig"><svg viewBox="0 0 640 330" role="img" aria-label="PyTorchモデルを中間ハブ(ONNX/TorchScript)経由でOpenVINO・CoreML・LiteRT・TensorRTへ変換する地図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="20" y="132" width="124" height="66" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="82" y="160" text-anchor="middle" fill="#18181b"><tspan x="82" dy="0" font-size="16" font-weight="700">PyTorch</tspan><tspan x="82" dy="20" font-size="12" font-weight="400" fill="#52525b">eager 実行</tspan></text><line x1="144" y1="165" x2="188" y2="165" stroke="#71717a" stroke-width="2"/><polygon points="196,165 186,160 186,170" fill="#71717a"/><rect x="200" y="120" width="152" height="90" rx="10" fill="#fff7ed" stroke="#ea580c" stroke-width="2.2"/><text x="276" y="158" text-anchor="middle"><tspan x="276" dy="0" font-size="15" font-weight="700" fill="#c2410c">中間ハブ</tspan><tspan x="276" dy="22" font-size="13" font-weight="600" fill="#3f3f46">ONNX · TorchScript</tspan></text><line x1="352" y1="158" x2="432" y2="55" stroke="#71717a" stroke-width="1.8"/><line x1="352" y1="162" x2="432" y2="126" stroke="#71717a" stroke-width="1.8"/><line x1="352" y1="168" x2="432" y2="200" stroke="#71717a" stroke-width="1.8"/><line x1="352" y1="172" x2="432" y2="271" stroke="#71717a" stroke-width="1.8"/><polygon points="438,55 428,50 428,60" fill="#71717a"/><polygon points="438,126 428,121 428,131" fill="#71717a"/><polygon points="438,200 428,195 428,205" fill="#71717a"/><polygon points="438,271 428,266 428,276" fill="#71717a"/><rect x="438" y="26" width="180" height="52" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="438" y="100" width="180" height="52" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="438" y="174" width="180" height="52" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="438" y="248" width="180" height="52" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="528" y="48" text-anchor="middle"><tspan x="528" dy="0" font-size="15" font-weight="700" fill="#18181b">OpenVINO</tspan><tspan x="528" dy="17" font-size="11" font-weight="400" fill="#52525b">Intel / サーバ CPU</tspan></text><text x="528" y="122" text-anchor="middle"><tspan x="528" dy="0" font-size="15" font-weight="700" fill="#18181b">CoreML</tspan><tspan x="528" dy="17" font-size="11" font-weight="400" fill="#52525b">Mac · iOS</tspan></text><text x="528" y="196" text-anchor="middle"><tspan x="528" dy="0" font-size="14" font-weight="700" fill="#18181b">LiteRT (.tflite)</tspan><tspan x="528" dy="17" font-size="11" font-weight="400" fill="#52525b">モバイル · 組込</tspan></text><text x="528" y="270" text-anchor="middle"><tspan x="528" dy="0" font-size="15" font-weight="700" fill="#18181b">TensorRT</tspan><tspan x="528" dy="17" font-size="11" font-weight="400" fill="#52525b">NVIDIA GPU</tspan></text></svg><figcaption>学習済み <b>PyTorch モデル</b> は、まず移植性の高い<b>中間ハブ（ONNX / TorchScript）</b>に固め、出口（デプロイ先）に応じて変換します。<b>OpenVINO</b> は Intel/サーバ CPU、<b>CoreML</b> は Mac/iOS、<b>LiteRT</b>(<code>.tflite</code>)はモバイル/組込、<b>TensorRT</b> は NVIDIA GPU 向けです。どれも「中間ハブ → ターゲット形式」という共通の流れで、<b>変換した瞬間に速くなったと思い込まず</b>、毎回 数値一致・速度・サイズ・精度を実測します。</figcaption></figure>

この章の実習環境はCPUのみ(MacBook を含む想定)なので、**確実に動く**ランタイム=eager / TorchScript / ONNX Runtime を主役にして横並びベンチを組みます。一方、OpenVINO・CoreML・LiteRT・TensorRT は重く、プラットフォーム依存も強いため、import をガードして「導入済みなら実演・未導入なら概念紹介」に徹します(`04_edge_runtimes_concept.py`)。

## 2. TorchScript — trace と script(直感 → 理論 → 正準API)

**直感**: TorchScript は PyTorch モデルを「Python が無くても実行できる中間表現」に固める仕組みです。固めた `.pt` は `torch.jit.load` でロードでき、C++(LibTorch)やサーバ配布でそのまま動きます。捕まえ方は2通りあります。`torch.jit.trace` は**実際に1回 forward を走らせて通った道を記録**し、`torch.jit.script` は**ソースコードを解析して制御フローごとグラフ化**します。

**理論と罠**: trace は速くて手軽ですが「通った道」しか残りません。`if x.sum() > 0:` のような**データ依存の分岐**は、trace 時に通った側だけが焼き込まれ、別の入力では**間違った分岐のまま**動きます(これが trace の罠)。`01_torchscript_trace_script.py` では、入力 `+1` で trace したモデルに `-1` を入れると、本来 `-x = +1` であるべき出力が `-2`(=`x*2` の道に固定)になることを実測で見せます。一方 `script` は分岐を保持するので正しく `+1` を返します。判断基準はシンプルで、**分岐の無い純粋な計算(多くの CNN)は trace で十分、入力で経路が変わるモデルは script** を使います。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="同じ分岐モデルでもtraceは通った1本だけ焼き込むので別入力で誤り、scriptは両方の分岐を保持して正しく動く" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="14" y="40" width="276" height="244" rx="10" fill="#fff7ed"/><rect x="310" y="40" width="276" height="244" rx="10" fill="#eff6ff"/><text x="300" y="26" text-anchor="middle" font-size="14.5" font-weight="700" fill="#18181b">同じ分岐モデルを2通りで固める</text><text x="152" y="72" text-anchor="middle" font-size="17" font-weight="700" fill="#c2410c">trace</text><circle cx="152" cy="98" r="7" fill="#ffffff" stroke="#3f3f46" stroke-width="2"/><line x1="152" y1="105" x2="106" y2="182" stroke="#ea580c" stroke-width="3"/><circle cx="106" cy="188" r="8" fill="#ea580c"/><line x1="152" y1="105" x2="200" y2="182" stroke="#d4d4d8" stroke-width="2.5" stroke-dasharray="5 4"/><circle cx="200" cy="188" r="8" fill="#ffffff" stroke="#d4d4d8" stroke-width="2"/><rect x="44" y="216" width="216" height="50" rx="9" fill="#ffffff" stroke="#dc2626" stroke-width="2"/><text x="152" y="246" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">✗ 別入力で誤った道</text><text x="448" y="72" text-anchor="middle" font-size="17" font-weight="700" fill="#15803d">script</text><circle cx="448" cy="98" r="7" fill="#ffffff" stroke="#3f3f46" stroke-width="2"/><line x1="448" y1="105" x2="402" y2="182" stroke="#2563eb" stroke-width="3"/><line x1="448" y1="105" x2="494" y2="182" stroke="#2563eb" stroke-width="3"/><circle cx="402" cy="188" r="8" fill="#2563eb"/><circle cx="494" cy="188" r="8" fill="#2563eb"/><rect x="340" y="216" width="216" height="50" rx="9" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="448" y="246" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">✓ 両方の道が正しく動く</text></svg><figcaption><b>trace</b>（実行をなぞる）は forward を1回走らせて<b>通った1本の道だけ</b>を焼き込むため、<code>if x.sum()…</code> のような<b>データ依存の分岐</b>があると別の入力でも同じ道を実行して誤ります（例: <code>+1</code> で trace すると <code>-1</code> 入力に誤って <code>-2</code> を返す）。<b>script</b>（ソース解析）は<b>分岐ごとグラフ化</b>するのでどの入力でも正しい道を選びます。分岐の無い CNN は trace、経路が変わるモデルは script。</figcaption></figure>

**正準API**: 変換後は `torch.jit.freeze`(eval 済み前提で定数畳み込み等を適用)で推論用に軽くし、`.save()` で保存します。そのうえで、必ず `eager` 出力との最大絶対誤差を確認してから配布します。resnet18 では trace/script とも eager と完全一致(誤差 0)し、ロード後も誤差 1e-6 程度に収まります。

```python
traced = torch.jit.trace(model.eval(), example)     # 実行をなぞる
traced = torch.jit.freeze(traced)                    # 推論用に最適化
traced.save("model.pt")                              # Python 非依存で配布
reloaded = torch.jit.load("model.pt")                # 元コード不要でロード
# scripted = torch.jit.script(model)                 # 分岐を保持したいとき
```

## 3. torch.compile — Inductor の仕組みと現実(直感 → 落とし穴)

**直感**: `torch.compile(model)` は PyTorch 2 の目玉で、モデルを書き換えずに1行で高速化を狙えます。中身は2段で、**TorchDynamo** が Python バイトコードをフックして計算グラフを切り出し(捕捉)、**Inductor** がそのグラフを最適化して**CPU では C++/OpenMP カーネルを生成**(GPU では Triton カーネル)します。複数の演算を1カーネルに**融合**してメモリ往復を減らすのが速さの源です。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="torch.compileの2段。Dynamoが多数の小演算を計算グラフに捕捉し、Inductorが1本の融合カーネルに生成する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">torch.compile：捕捉 → 生成・融合 の2段</text><rect x="20" y="66" width="132" height="118" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="36" y="82" width="100" height="18" rx="3" fill="#dbeafe" stroke="#d4d4d8" stroke-width="1"/><rect x="36" y="106" width="100" height="18" rx="3" fill="#ffedd5" stroke="#d4d4d8" stroke-width="1"/><rect x="36" y="130" width="100" height="18" rx="3" fill="#dbeafe" stroke="#d4d4d8" stroke-width="1"/><rect x="36" y="154" width="100" height="18" rx="3" fill="#ffedd5" stroke="#d4d4d8" stroke-width="1"/><text x="86" y="206" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1d4ed8">eager：小演算が多数</text><line x1="154" y1="128" x2="226" y2="128" stroke="#52525b" stroke-width="2.2"/><polygon points="234,128 224,122 224,134" fill="#52525b"/><text x="194" y="104" text-anchor="middle" fill="#3f3f46"><tspan x="194" dy="0" font-size="12" font-weight="700">Dynamo</tspan><tspan x="194" dy="13" font-size="10.5" font-weight="400">捕捉</tspan></text><rect x="236" y="66" width="128" height="118" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><circle cx="272" cy="104" r="9" fill="#ffffff" stroke="#ea580c" stroke-width="2"/><circle cx="328" cy="104" r="9" fill="#ffffff" stroke="#ea580c" stroke-width="2"/><circle cx="300" cy="150" r="9" fill="#ffffff" stroke="#ea580c" stroke-width="2"/><line x1="281" y1="104" x2="319" y2="104" stroke="#ea580c" stroke-width="1.6"/><line x1="280" y1="111" x2="294" y2="143" stroke="#ea580c" stroke-width="1.6"/><line x1="320" y1="111" x2="306" y2="143" stroke="#ea580c" stroke-width="1.6"/><text x="300" y="206" text-anchor="middle" font-size="12.5" font-weight="600" fill="#c2410c">計算グラフ</text><line x1="366" y1="128" x2="438" y2="128" stroke="#52525b" stroke-width="2.2"/><polygon points="446,128 436,122 436,134" fill="#52525b"/><text x="406" y="104" text-anchor="middle" fill="#3f3f46"><tspan x="406" dy="0" font-size="12" font-weight="700">Inductor</tspan><tspan x="406" dy="13" font-size="10.5" font-weight="400">融合</tspan></text><rect x="448" y="66" width="132" height="118" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="2.2"/><rect x="464" y="110" width="100" height="34" rx="5" fill="#f97316" stroke="#c2410c" stroke-width="1.5"/><text x="514" y="206" text-anchor="middle"><tspan x="514" dy="0" font-size="12.5" font-weight="600" fill="#c2410c">融合カーネル</tspan><tspan x="514" dy="15" font-size="10.5" font-weight="400" fill="#52525b">C++ / OpenMP</tspan></text></svg><figcaption><code>torch.compile</code> は2段で動きます。<b>TorchDynamo</b> が Python バイトコードをフックして<b>多数の小さな演算を計算グラフに捕捉</b>し、<b>Inductor</b> がそのグラフから <b>1本の融合カーネル</b>（CPU は C++/OpenMP、GPU は Triton）を生成します。<b>演算を融合してメモリ往復を減らす</b>のが速さの源ですが、初回はコンパイルで遅く、CPU では <code>g++</code> と <code>Python.h</code> が要る点に注意します。</figcaption></figure>

**落とし穴**: ここが本章の山場です。第一に、**初回呼び出しでコンパイルが走るため非常に遅い**(数秒〜)。ベンチで初回を混ぜると「遅い」と誤判定します。必ずウォームアップ後の定常レイテンシで測ります。第二に、CPU では Inductor が**g++ と Python 開発ヘッダ(`Python.h`)を必要**とし、これらが無い環境では**初回呼び出しで `CppCompileError` を出して失敗**します(本講座の実習環境がまさにこれ)。そのため教材では `torch.compile` を**try/except でガード**し、使えなければ表から自動的に外して、概念だけ残します(`02_torch_compile_cpu.py` / `bench_lab.build_torch_compile`)。第三に、**グラフブレイク**(`print` や未対応構文・データ依存分岐でグラフが分断され Python に戻る)や、形状が変わるたびの**再コンパイル**で、期待した高速化が出ないことがあります。`fullgraph=True` で隠れたブレイクを例外として炙り出し、固定形なら `dynamic=False`、可変なら `dynamic=True` で挙動を制御します。

実務では「`torch.compile` は使えれば効くが、デプロイ経路の必須にはしない」と捉えるのが安全です。確実に効かせたいなら、次に説明する ONNX Runtime のほうが移植性・再現性で勝ります。

## 4. ONNX Runtime を「ランタイム」として使う(正準API → 実装)

36章では ONNX を「交換フォーマット」として学びましたが、本章では**実行ランタイム**として横並びベンチに組み込みます。ポイントは3つ。**①エクスポートは安定経路で**: torch 2.9+ では `torch.onnx.export` の既定が dynamo 経路(`onnxscript` 必須)になりましたが、本講座は onnxscript 非依存で確実に動かすため `dynamo=False`(legacy 経路)+ `opset_version=17` を使います。`dynamic_axes` でバッチ次元を可変にすると、レイテンシ(batch=1)とスループット(batch=N)を**同じモデル**で測れます。**②セッション最適化**: `SessionOptions` で `graph_optimization_level = ORT_ENABLE_ALL`(演算子融合・定数畳み込み)を有効化し、`intra_op_num_threads` を実機に合わせて固定します。**③int8 動的量子化**: `quantize_dynamic(..., weight_type=QuantType.QUInt8)` で重みを int8 化します(CPU は U8X8 経路のため QUInt8)。

```python
torch.onnx.export(model, (example,), "m.onnx", opset_version=17, dynamo=False,
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
sess = ort.InferenceSession("m.onnx", so, providers=["CPUExecutionProvider"])
y = sess.run(None, {"input": x_numpy})[0]
```

実測(resnet18 / CPU / 4スレッド固定、`03_runtime_bench.py`)では、**ONNX Runtime fp32 が eager の約 1.7〜2.0倍**で最速、bf16 autocast が約 1.2倍でスループットが伸び、TorchScript はほぼ等速(配布形としての価値が主)という典型的な並びになります。

## 5. 量子化の「実効速度の罠」をランタイム視点で再確認

35章で「非構造化 pruning はマスクを掛けるだけで実速度は縮まない」という罠を学びました。ランタイム視点では、**int8 動的量子化にも似た罠**があります。`03_runtime_bench.py` の実測では、ONNX int8 動的量子化は**サイズが約 1/4(46.7MB → 11.7MB)に確実に縮む**一方で、レイテンシは eager の **0.6〜0.85倍(=むしろ遅い)** になりました。理由は、動的量子化が **Linear/MatMul 主体(Transformer など)で効きやすく**、resnet18 のような **Conv 主体の CNN では quant/dequant のオーバーヘッドが利得を食う**ためです。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="int8動的量子化はサイズが約4分の1に縮むが、CNNでは速度がeagerの0.6から0.85倍で速くならない" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="32" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">int8 動的量子化（CNN）：サイズは縮むが速くならない</text><text x="30" y="105" font-size="13" font-weight="700" fill="#3f3f46">サイズ</text><rect x="96" y="68" width="360" height="24" rx="3" fill="#2563eb"/><text x="106" y="85" font-size="13" font-weight="700" fill="#ffffff">fp32  46.7 MB</text><rect x="96" y="100" width="92" height="24" rx="3" fill="#16a34a"/><text x="106" y="117" font-size="12.5" font-weight="700" fill="#ffffff">int8 11.7</text><text x="200" y="118" font-size="13.5" font-weight="700" fill="#15803d">✓ ≈ 1/4 に縮む</text><text x="30" y="197" font-size="13" font-weight="700" fill="#3f3f46">速度(CNN)</text><rect x="96" y="164" width="320" height="24" rx="3" fill="#2563eb"/><text x="106" y="181" font-size="13" font-weight="700" fill="#ffffff">eager  1.00x</text><rect x="96" y="196" width="224" height="24" rx="3" fill="#dc2626"/><text x="106" y="213" font-size="13" font-weight="700" fill="#ffffff">int8  ≈0.7x</text><line x1="416" y1="164" x2="416" y2="220" stroke="#71717a" stroke-width="1.4" stroke-dasharray="4 3"/><text x="330" y="213" font-size="13.5" font-weight="700" fill="#dc2626">✗ むしろ遅い</text></svg><figcaption>resnet18（CNN）の <b>int8 動的量子化</b>の実測です。<b>サイズは約 1/4（46.7 → 11.7MB）に確実に縮む</b>一方、<b>速度は eager の 0.6〜0.85倍で速くなりません</b>（むしろ遅い）。動的量子化は <b>Linear/MatMul 主体（Transformer）で効き</b>、Conv 主体の CNN では quant/dequant のオーバーヘッドが利得を食うためです。<b>サイズと速度は必ず分けて測る</b>のが鉄則です。</figcaption></figure>

ここから得る教訓は2つ。第一に「**サイズが縮む≠速くなる**」を常に分けて測ること。エッジでメモリ/配布サイズが制約なら int8 は大正解ですが、レイテンシが制約なら CNN では効かないことがあります。第二に「**手法は題材に依存する**」こと。同じ int8 でも、Transformer 系なら速度も伸び、CNN なら静的PTQ(activation もキャリブレーションして固定)のほうが効きます。だからこそ「精度・速度・サイズを毎回実測する」習慣が決定的に重要です。

## 6. エッジ/プラットフォーム別ランタイム(任意ガードで概念)

`04_edge_runtimes_concept.py` は4つのエッジ系ランタイムを、import ガード付きで整理します。**OpenVINO**(`uv add --group edge openvino nncf`)は Intel CPU 最適(AMD x86 でも動作)で、`ov.convert_model` → `ov.compile_model('CPU')` の2行で推論でき、NNCF の `nncf.quantize` で PTQ(INT8)まで載せられます。**CPU実習の主力**で、Intel機では ONNX Runtime と並ぶ第一候補です。**CoreML**(`coremltools`)は Mac/iOS 向けで、`ct.convert(traced, ...)` → `.mlpackage` 保存。**変換は Linux でも通ることがあるが、予測実行は macOS 必須**という非対称に注意します。**LiteRT**(`litert-torch`、旧 `ai-edge-torch` から改名)は PyTorch → `.tflite` でモバイル向け。**TensorRT**(`torch-tensorrt`)は NVIDIA GPU 専用で**CPU では実行不可**のため、本講座では概念・図解・netron 可視化に留めます。

どのランタイムも「ONNX/TorchScript を中間ハブにして変換 → 変換後に数値一致を検証」という基本線は同じです。`netron`(`uv add --group viz netron`)で `.onnx` を可視化すると、融合された層・shape・量子化ノードを目で確認でき、デバッグの強力な相棒になります。スクリプトは可視化用に `resnet18_for_netron.onnx` を書き出すので、未導入でも <https://netron.app> にドラッグ&ドロップすれば層構造を読めます。

## 7. 手法選択の意思決定順序(実務の使い分け)

最適化は「全部盛り」ではなく**順番**が命です。コストが低く壊れにくい手から試し、各段階で必ず計測して、**目標に届いたら止める**。やみくもに量子化や pruning から入ると、精度を壊した割に速くならない事故が起きます。推奨順序は次のとおりです。

> **0.計測ファースト → 1.eval()+inference_mode(タダ・必須) → 2.スレッド数調整 → 3.bf16 autocast(+compile) → 4.ONNX Runtime(fp32) → 5.ONNX 動的量子化(int8) → 6.静的PTQ → 7.QAT/構造化pruning → 8.エッジ専用ランタイム**

`05_decision_order.py` は、この順序を小さな**意思決定エンジン**として実装します。`Target`(許容 latency / size / 精度保持率)と現状の `State` を入れると、「精度が割れていたら最優先で一段戻す」「未達なら効果が大きく壊れにくい次の手を1つ提案する」「達成したら止める」を早期 return で判定します。**精度保持を最優先のガード**に置くのがコツで、速度・サイズだけを追って精度を割る事故を構造的に防げます。

<figure class="lec-fig"><svg viewBox="0 0 640 320" role="img" aria-label="最適化の意思決定順序。安く安全な手から始め、目標に届いたら止め、精度が割れたら1段戻すガードを置く" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">順番が命：安い・安全な手から → 目標で止める</text><rect x="232" y="62" width="176" height="30" rx="15" fill="#16a34a"/><text x="320" y="82" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">目標に届いたら STOP</text><line x1="320" y1="92" x2="320" y2="114" stroke="#16a34a" stroke-width="2"/><polygon points="320,120 314,110 326,110" fill="#16a34a"/><rect x="14" y="120" width="88" height="58" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><rect x="116" y="120" width="88" height="58" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><rect x="218" y="120" width="88" height="58" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="1.8"/><rect x="320" y="120" width="88" height="58" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="1.8"/><rect x="422" y="120" width="88" height="58" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/><rect x="524" y="120" width="88" height="58" rx="8" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="58" y="154" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">計測</text><text x="160" y="146" text-anchor="middle" fill="#18181b"><tspan x="160" dy="0" font-size="12.5" font-weight="700">eval</tspan><tspan x="160" dy="16" font-size="11" font-weight="600" fill="#3f3f46">スレッド</tspan></text><text x="262" y="146" text-anchor="middle" fill="#18181b"><tspan x="262" dy="0" font-size="12.5" font-weight="700">bf16</tspan><tspan x="262" dy="16" font-size="11" font-weight="600" fill="#3f3f46">compile</tspan></text><text x="364" y="146" text-anchor="middle" fill="#18181b"><tspan x="364" dy="0" font-size="12.5" font-weight="700">ONNX</tspan><tspan x="364" dy="16" font-size="11" font-weight="600" fill="#3f3f46">fp32</tspan></text><text x="466" y="146" text-anchor="middle" fill="#18181b"><tspan x="466" dy="0" font-size="12.5" font-weight="700">int8</tspan><tspan x="466" dy="16" font-size="11" font-weight="600" fill="#3f3f46">量子化</tspan></text><text x="568" y="146" text-anchor="middle" fill="#18181b"><tspan x="568" dy="0" font-size="11.5" font-weight="700">PTQ・QAT</tspan><tspan x="568" dy="16" font-size="11" font-weight="600" fill="#3f3f46">エッジ専用</tspan></text><line x1="102" y1="149" x2="113" y2="149" stroke="#71717a" stroke-width="1.8"/><polygon points="116,149 109,145 109,153" fill="#71717a"/><line x1="204" y1="149" x2="215" y2="149" stroke="#71717a" stroke-width="1.8"/><polygon points="218,149 211,145 211,153" fill="#71717a"/><line x1="306" y1="149" x2="317" y2="149" stroke="#71717a" stroke-width="1.8"/><polygon points="320,149 313,145 313,153" fill="#71717a"/><line x1="408" y1="149" x2="419" y2="149" stroke="#71717a" stroke-width="1.8"/><polygon points="422,149 415,145 415,153" fill="#71717a"/><line x1="510" y1="149" x2="521" y2="149" stroke="#71717a" stroke-width="1.8"/><polygon points="524,149 517,145 517,153" fill="#71717a"/><path d="M 466 180 Q 415 202 368 184" fill="none" stroke="#dc2626" stroke-width="2"/><polygon points="362,184 372,180 371,190" fill="#dc2626"/><line x1="22" y1="208" x2="600" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="608,208 598,203 598,213" fill="#71717a"/><text x="312" y="228" text-anchor="middle" font-size="12.5" font-weight="600" fill="#52525b">← 安く・壊れにくい ／ 高コスト・壊れやすい →</text><rect x="150" y="252" width="340" height="36" rx="9" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="320" y="275" text-anchor="middle" font-size="13.5" font-weight="700" fill="#dc2626">精度が割れたら 1 段戻す（最優先ガード）</text></svg><figcaption>最適化は「全部盛り」ではなく<b>順番</b>が命です。<b>コストが低く壊れにくい手から</b>試し（計測 → <code>eval()</code>/スレッド → bf16/compile → ONNX fp32 → int8量子化 → 静的PTQ/QAT/エッジ専用）、各段で計測して<b>目標に届いたら止めます</b>。最重要は<b>精度保持のガード</b>で、<b>精度が割れたら最優先で1段戻し</b>、速度・サイズだけを追って精度を割る事故を構造的に防ぎます。</figcaption></figure>

---

## 🛠 章末ミニプロジェクト — マルチランタイム・ベンチ＆意思決定レポート

`mini_project.py` は本章の統合課題です。**同一モデル(resnet18)を複数ランタイムへ変換し、同一指標で公平比較し、用途別の最適ランタイムを自動で意思決定**して、表(Markdown)と図を成果物として残します。

含むランタイムは、この環境で使えるものを自動収集します: `eager fp32` / `eager bf16 autocast` / `TorchScript(trace/script)` / `torch.compile`(ツールチェインがあれば) / `ONNX Runtime(fp32 / int8 動的量子化)` / `OpenVINO`(導入済みなら)。各ランタイムを「numpy in → logits numpy out」の共通インタフェース(`bench_lab.Runtime`)に揃えることで、1つのループで横並び比較できます。

評価軸は4つを必ずセットで見ます: **レイテンシ(p50/p99, batch=1)**・**スループット(img/s, batched)**・**サイズ(MB)**・**精度保持(fp32 eager 基準の top1 一致率と最大絶対誤差)**。意思決定は「①レイテンシ最小 ②サイズ最小(精度保持≥99%) ③総合おすすめ(精度を保ちつつ最速)」を根拠つきで選び、`outputs/37_runtime_edge_optimization/mini_project_report.md` と `mini_project_bench.png` に出力します。

実行(数十秒):

```bash
uv run python lectures/37_runtime_edge_optimization/mini_project.py
```

典型的な出力(CPU/4スレッド固定)では、レイテンシ最小=`onnxruntime_fp32`(eager 比 ~2.0x)、サイズ最小=`onnxruntime_int8_dynamic`(~11.7MB)、総合おすすめ=`onnxruntime_fp32` と判定されます。`torch.compile` が使えない環境では自動的に表から外れ、それでも結論は変わらない=**移植性の高い最適化(ONNX)を軸にすべき**という実務的な学びが残ります。

---

## ✅ 到達チェックリスト

- [ ] `torch.jit.trace` と `torch.jit.script` の違いを説明でき、**trace の罠**(データ依存分岐)を例で示せる。
- [ ] TorchScript を `freeze` → `save` → `load` し、eager との数値一致を検証してから配布できる。
- [ ] `torch.compile` の2段(Dynamo 捕捉 / Inductor 生成・融合)と、初回コンパイルコスト・グラフブレイク・C++ ツールチェイン依存を説明できる。
- [ ] `torch.compile` を **try/except でガード**し、使えない環境でも落ちないコードが書ける。
- [ ] `torch.onnx.export(dynamo=False, opset_version=17, dynamic_axes=...)` で ONNX 化し、`SessionOptions` で graph 最適化・スレッドを設定して onnxruntime 推論できる。
- [ ] `quantize_dynamic`(QUInt8)でサイズが ~1/4 に縮むことを実測し、**CNN では速度が伸びない/逆に遅い**ことがある(実効速度の罠)と説明できる。
- [ ] レイテンシは **p50/p99**、スループットは **batched img/s**、精度保持は **fp32 基準の top1 一致率** で測る、と区別できる。
- [ ] OpenVINO / CoreML / LiteRT / TensorRT の出口・使い方・CPUのみでの実習可否を整理できる。
- [ ] **意思決定の順序**(eval → スレッド → bf16/compile → ONNX → 動的量子化 → 静的PTQ → QAT/pruning → エッジ専用)を言え、目標到達で止められる。
- [ ] `mini_project.py` を動かし、用途別の最適ランタイムを根拠つきで選べる。

---

## ❓ 落とし穴・FAQ・デバッグ

**Q1. `torch.compile` が `CppCompileError` / `Can't find Python.h` で失敗する。**
CPU の Inductor は C++ カーネルを生成するため、`g++` と Python 開発ヘッダ(`python3-dev` の `Python.h`)が必要です。無い環境では初回呼び出しで失敗します。本講座のように **try/except でガード**して概念に切り替えるのが正解。実際に効かせたい場合は `apt install build-essential python3-dev`(または環境に応じた dev パッケージ)を入れてから再試行します。

**Q2. `torch.onnx.export` が `No module named 'onnxscript'` で落ちる。**
torch 2.9+ は既定が dynamo 経路で `onnxscript` を要求します。onnxscript を入れない方針なら **`dynamo=False`** を明示して legacy 経路を使ってください(本講座の全スクリプトはこれ)。

**Q3. trace したモデルが別入力で間違った結果を返す。**
データ依存分岐を trace で焼き込んだ典型(trace の罠)。`torch.jit.script` を使うか、分岐自体を消す(マスク演算で表現する等)で回避します。`01_torchscript_trace_script.py` の `[3]` が再現例です。

**Q4. int8 量子化したのに遅くなった。**
動的量子化は Linear/MatMul 主体で効き、Conv 主体の CNN では quant/dequant のオーバーヘッドで逆に遅くなることがあります。CNN を速くしたいなら**静的PTQ**(activation もキャリブレーション)を検討。サイズ削減だけが目的なら動的量子化で十分です。**速度とサイズは必ず分けて測る**こと。

**Q5. ベンチの数字が毎回ばらつく / 速く見えたり遅く見えたり。**
ウォームアップを省くと初回の JIT/コンパイル/キャッシュmiss が混ざります。必ず「初回を捨てる→多数回反復→中央値/分位点」で測り、スレッド数を固定(`torch.set_num_threads` / ORT `intra_op_num_threads`)します。CPU では `cuda.synchronize` は不要ですが、GPU では同期を忘れると非同期実行を「爆速」と誤認します。

**Q6. OpenVINO / CoreML を入れたいが依存が重い/環境を壊しそう。**
本講座の main pyproject には含めず、到達してから個別に追加します(`uv add --group edge openvino nncf` 等)。CoreML の**予測実行は macOS 必須**、TensorRT は **NVIDIA GPU 必須**なので、CPUのみ環境では概念に留めて問題ありません。

**Q7. ONNX 出力と torch 出力が微妙にずれる。**
浮動小数点の演算順序差で 1e-5〜1e-6 程度のずれは正常です。`atol`/`rtol` を決めて**最大絶対誤差**で検証します(`exercises.py` の ex5/ex10)。1e-3 を超えるなら opset・dynamic_axes・量子化設定を疑います。

---

## 🚀 発展トピック・参考

- **静的PTQ / QAT(PT2E フロー)**: `torch.export.export` + `prepare_pt2e` / `convert_pt2e` + `X86InductorQuantizer` で、activation もキャリブレーションする静的量子化。Conv 主体の CNN を CPU で速くしたいときの本命。QAT は `torchao` の `QATConfig` で精度回復を狙う(35章の発展)。
- **torchao の現行量子化API**: `torch.ao.quantization`(eager/FX)は将来削除予定で、`torchao.quantization.quantize_` + Config が現行推奨。int8 dynamic は CPU で動くが、int4/float8 の多くは CUDA/ARM(Apple Silicon)限定。
- **IOBinding**: onnxruntime で入出力のコピーを削減してさらに高速化(大入力/連続推論で効く)。
- **optimum + optimum-onnx**: HuggingFace Transformers を `ORTModelForXXX(export=True)` で ONNX 化し、ONNX Runtime で動かす(36章の発展)。
- **bitsandbytes**: LLM/VLM の 8bit/4bit(NF4)。歴史的に CUDA 専用、0.49 で CPU(AVX512)等が α対応だが成熟度は低い。CPUのみ環境では概念中心、実速度検証は torchao/ONNX に寄せるのが安全。
- **公式ドキュメント**: PyTorch <https://docs.pytorch.org/docs/stable/> / ONNX Runtime <https://onnxruntime.ai/docs/> / OpenVINO <https://docs.openvino.ai/2026/> / CoreML Tools <https://apple.github.io/coremltools/> / netron <https://github.com/lutzroeder/netron>

---

## ▶ 動かし方

```bash
# 依存(未導入なら): dl と onnx グループ
uv sync --group dl --group onnx

# 1本ずつ(各ファイル exit 0)
uv run python lectures/37_runtime_edge_optimization/01_torchscript_trace_script.py
uv run python lectures/37_runtime_edge_optimization/02_torch_compile_cpu.py
uv run python lectures/37_runtime_edge_optimization/03_runtime_bench.py
uv run python lectures/37_runtime_edge_optimization/04_edge_runtimes_concept.py
uv run python lectures/37_runtime_edge_optimization/05_decision_order.py

# 章末ミニプロジェクト(成果物: outputs/37_runtime_edge_optimization/ に表+図)
uv run python lectures/37_runtime_edge_optimization/mini_project.py

# 演習(自己採点)→ 模範解答(全PASS)
uv run python lectures/37_runtime_edge_optimization/exercises.py
uv run python lectures/37_runtime_edge_optimization/exercises_solutions.py

# (任意)エッジ系を実習したい場合
uv add --group edge openvino nncf      # Intel/AMD CPU で OpenVINO 実習
uv add --group viz netron              # モデルグラフ可視化
```

成果物・図はすべて `outputs/37_runtime_edge_optimization/` に保存されます(matplotlib は Agg、`cv2.imshow` は呼びません)。

> 版: torch 2.12+cpu / torchvision 0.27+cpu / onnx 1.21 / onnxruntime 1.26 / numpy 2.x ・ 2026-06
> 注: 本実習環境は CPU のみ。`torch.compile`(要 C++ ツールチェイン)・OpenVINO・CoreML・LiteRT・TensorRT は**実行経路で必須にせず**、未導入時は概念紹介にフォールバックします。
