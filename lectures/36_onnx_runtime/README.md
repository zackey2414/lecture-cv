# 36_onnx_runtime: ONNX エクスポートと onnxruntime — グラフ最適化・動的量子化

> トラック: **最適化・デプロイ** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `onnx`
> 前提: 第34回（推論プロファイリング・計測ファースト）／ 関連: 第35回（量子化と枝刈り）・第37回（エッジ最適化）

---

## 🎯 この章のゴール

この章では、学習した PyTorch モデルを、フレームワークに縛られない **ONNX（Open Neural Network Exchange）** に書き出し、**onnxruntime** という高速な推論エンジンで動かせるようになる。具体的には、次の内容を「AI 補助なしで書ける」ところまで身につける。

- `torch.onnx.export` でモデルを `.onnx` に書き出す（新既定 `dynamo=True` と旧 `dynamo=False` の違いを理解した上で、本講座環境では `dynamo=False` を正準に使う）
- エクスポートの正しさを **2 段階**で検証する: `onnx.checker.check_model`（静的整合性）と、**torch と onnxruntime の出力が atol/rtol 内で一致するか**（最大絶対誤差）
- `onnxruntime.InferenceSession`（`CPUExecutionProvider`）で推論し、`SessionOptions` の **グラフ最適化レベル**と **スレッド数**を制御する
- `onnxruntime.quantization.quantize_dynamic`（`QuantType.QUInt8`）で **CPU に最も手軽に効く int8 化**を行い、**速度・サイズ・精度の三角関係**で費用対効果を判断する
- `optimum` の `ORTModelForXXX(export=True)` で HuggingFace Transformers も ONNX 化できる（任意依存。概念と等価な自前実装を体験）
- `netron` でグラフを可視化する（概念）。`onnx` API で「テキスト版 netron」を作って構造を点検する

最終的に、章末ミニプロジェクトで **「export → 検証 → onnxruntime 最適化 → 動的量子化 → 公平比較 → 意思決定」** という実務のデプロイ・ワークフロー 1 周を完走する。

---


## 0. 直感 — なぜ ONNX と onnxruntime なのか

PyTorch はモデルを**研究・学習する**のに最高の道具だが、**本番で推論を回す**となるといくつか不都合がある。Python インタプリタと巨大な torch ランタイムを丸ごと抱える必要がある上に、デプロイ先（サーバ・エッジ・ブラウザ・モバイル）ごとに事情も異なるからだ。そこで「学習は PyTorch、推論は専用エンジン」という分業が定石になる。この分業を支えるのが ONNX と onnxruntime であり、ONNX が両者の橋渡しをする **共通の中間表現（計算グラフのフォーマット）**、onnxruntime がその ONNX を高速に実行する **推論専用エンジン**にあたる。

<figure class="lec-fig"><svg viewBox="0 0 660 210" role="img" aria-label="PyTorchで学習したモデルをtorch.onnx.exportでonnxに書き出し、onnxruntimeなど多様な推論エンジンで実行する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="26" y="78" width="128" height="64" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="90" y="106" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">PyTorch</text><text x="90" y="126" text-anchor="middle" font-size="12" fill="#52525b">学習・研究</text><line x1="154" y1="110" x2="222" y2="110" stroke="#71717a" stroke-width="2"/><polygon points="230,110 220,105 220,115" fill="#71717a"/><text x="192" y="100" text-anchor="middle" font-size="11.5" fill="#3f3f46">export</text><rect x="236" y="68" width="150" height="84" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><text x="311" y="105" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">model.onnx</text><text x="311" y="127" text-anchor="middle" font-size="11.5" fill="#52525b">計算グラフ（中間表現）</text><line x1="386" y1="110" x2="471" y2="79" stroke="#71717a" stroke-width="2"/><polygon points="480,76 472.3,84.1 468.9,74.7" fill="#71717a"/><line x1="386" y1="110" x2="471" y2="146" stroke="#71717a" stroke-width="2"/><polygon points="480,150 468.8,150.7 472.8,141.5" fill="#71717a"/><rect x="486" y="48" width="150" height="54" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2.5"/><text x="561" y="71" text-anchor="middle" font-size="15" font-weight="700" fill="#15803d">onnxruntime</text><text x="561" y="90" text-anchor="middle" font-size="11.5" fill="#52525b">本章・CPU で実行</text><rect x="486" y="124" width="150" height="54" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><text x="561" y="147" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">他ランタイム</text><text x="561" y="166" text-anchor="middle" font-size="11" fill="#52525b">OpenVINO / CoreML 等</text></svg><figcaption>学習は <b>PyTorch</b>、推論は専用エンジンという分業を、<b>ONNX</b>（<code>.onnx</code>）が橋渡しします。<code>torch.onnx.export</code> で書き出した 1 つの計算グラフは、本章の <b>onnxruntime</b>（CPU）だけでなく OpenVINO や TensorRT など多様な実行系に流せます（<b>可搬性</b>）。</figcaption></figure>

ONNX の嬉しさは大きく 3 つある。第一に **可搬性**: 一度 `.onnx` にすれば、onnxruntime・OpenVINO・TensorRT・CoreML・各種モバイルランタイムなど多様な実行系に流せる（第37回につながる）。第二に **速度**: onnxruntime は定数畳み込みや演算子融合といったグラフ最適化を行うため、CPU では素の eager PyTorch より速いことが多い（本章の実測でも TinyCNN で約 2〜3 倍）。第三に **軽さ**: 推論だけなら torch 本体を持ち込まずに済み、int8 量子化と組み合わせれば配布サイズも小さくできる。

とはいえ「ONNX 化すれば無条件に速くて正しい」わけではない。**変換が数値的に正しいか**は必ず検証しなければならないし、**最適化や量子化が本当に速くなるか**もモデルとハードで変わるので**必ず計測**する。この点で本章は終始、第34回の「計測ファースト」「推測するな、測れ」を引き継いでいる。

## 1. ONNX とは何か（理論: 計算グラフ・opset・IR）

ONNX ファイルの中身は、モデルの **計算グラフそのもの**だ。具体的には、`Conv → Relu → MaxPool → … → Gemm` のような**演算子（node）の有向グラフ**に、重み定数（initializer）と、入出力テンソルの名前・型・形（value_info）が添えられている。PyTorch の `nn.Module`（Python のクラス階層）とは違い、ONNX は**実行順に並んだ平らな演算列**なので、どの実行系でも素直に解釈・最適化できる。`04_graph_inspect_netron.py` では、`onnx` の Python API でこの中身（入出力・op の並び・initializer のサイズ）を実際に覗いてみる。

<figure class="lec-fig"><svg viewBox="0 0 660 244" role="img" aria-label="onnxファイルの中身は入力テンソルとConvReluMaxPoolGemmの演算ノード列と重み定数initializerと出力からなる計算グラフ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" fill="#3f3f46">.onnx の中身 ＝ 実行順に並んだ演算ノードの有向グラフ</text><rect x="44" y="57" width="86" height="50" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="87" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">input</text><text x="87" y="98" text-anchor="middle" font-size="10" fill="#52525b">[<tspan fill="#c2410c" font-weight="700">batch</tspan>,3,32,32]</text><line x1="130" y1="82" x2="150" y2="82" stroke="#71717a" stroke-width="1.8"/><polygon points="156,82 148,78 148,86" fill="#71717a"/><rect x="156" y="58" width="66" height="48" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="189" y="87" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">Conv</text><line x1="222" y1="82" x2="240" y2="82" stroke="#71717a" stroke-width="1.8"/><polygon points="246,82 238,78 238,86" fill="#71717a"/><rect x="246" y="58" width="60" height="48" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="276" y="87" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">Relu</text><line x1="306" y1="82" x2="324" y2="82" stroke="#71717a" stroke-width="1.8"/><polygon points="330,82 322,78 322,86" fill="#71717a"/><rect x="330" y="58" width="84" height="48" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="372" y="87" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">MaxPool</text><line x1="414" y1="82" x2="432" y2="82" stroke="#71717a" stroke-width="1.8"/><polygon points="438,82 430,78 430,86" fill="#71717a"/><rect x="438" y="58" width="66" height="48" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="471" y="87" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">Gemm</text><line x1="504" y1="82" x2="524" y2="82" stroke="#71717a" stroke-width="1.8"/><polygon points="530,82 522,78 522,86" fill="#71717a"/><rect x="530" y="57" width="86" height="50" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="573" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">logits</text><text x="573" y="98" text-anchor="middle" font-size="10" fill="#52525b">[<tspan fill="#c2410c" font-weight="700">batch</tspan>,10]</text><rect x="150" y="178" width="78" height="42" rx="5" fill="#f4f4f5" stroke="#71717a" stroke-width="1.6"/><text x="189" y="204" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">重み W,b</text><line x1="189" y1="178" x2="189" y2="112" stroke="#71717a" stroke-width="1.6"/><polygon points="189,106 185,114 193,114" fill="#71717a"/><rect x="432" y="178" width="78" height="42" rx="5" fill="#f4f4f5" stroke="#71717a" stroke-width="1.6"/><text x="471" y="204" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">重み W,b</text><line x1="471" y1="178" x2="471" y2="112" stroke="#71717a" stroke-width="1.6"/><polygon points="471,106 467,114 475,114" fill="#71717a"/></svg><figcaption><code>.onnx</code> の正体は計算グラフです。<b>入力テンソル</b>（型・形＝value_info）から <code>Conv → Relu → MaxPool → Gemm</code> のような<b>演算ノード（op）の列</b>を通って<b>出力</b>に至り、各演算には<b>重み定数（initializer）</b>がぶら下がります。形の先頭 <code>batch</code> を記号にすれば<b>動的バッチ</b>になり、export と違うバッチでも推論できます。</figcaption></figure>

ここで重要になる概念が **opset（operator set）version** だ。ONNX の演算子セットには版があり（本章は `opset=18`）、これをエクスポート時に指定する。opset が新しいほど表現できる演算が増えるが、その一方で **実行側ランタイムが対応していない opset を指定するとロードに失敗する**。逆に古すぎると新しい演算が表現できない。したがって「使う onnxruntime が安定に読める opset」を選ぶのが鉄則であり、上げ過ぎ・下げ過ぎはどちらも事故の元（落とし穴の定番）になる。なお **IR version**（グラフフォーマット自体の版）という版もあるが、こちらは export 時に自動で妥当な値が入るので普段は意識しなくてよい。

入出力の **形（shape）** は、固定値にも**動的軸（記号名）**にもできる。例えばバッチ次元を `'batch'` という記号にしておけば、export 時は batch=1 でも、推論時は batch=16 でも、同じモデルで通せる。この指定は `dynamic_axes`（旧 API）／`dynamic_shapes`（新 API）で行う。逆に動的軸を張り忘れると、「export したバッチサイズでしか動かない」硬いモデルになってしまうので注意したい。

## 2. 正準 API — `torch.onnx.export`（dynamo の有無）

PyTorch から ONNX への変換は `torch.onnx.export(model, args, path, ...)` 一本で行う。ただし、その前に**歴史的な分岐**を理解しておく必要がある。というのも、PyTorch 2.9 から、この関数の既定が **`dynamo=True`（torch.export を基盤にした新エクスポータ）** に変わったからだ。新エクスポータは制御フローやダイナミックな形をより正確に捉えられる将来本命だが、その代わり変換に **`onnxscript` パッケージ**を必須とする。

一方、本講座の標準環境には `onnxscript` を入れていない（依存を増やさない方針）。そこで本章は **旧来の TorchScript ベースのエクスポータ（`dynamo=False`）を正準**として使う。こちらは `onnxscript` が不要で枯れており、ResNet/小型 CNN/素朴な Transformer のような「素直なモデル」なら安定して動く。`dynamo=False` を指定すると `DeprecationWarning`（将来は新方式が既定）が出るが、機能上は問題ない。なお、新方式を使いたい場合は `uv add --group onnx onnxscript` を足し、`dynamo=True`（既定）にすればよい。

```python
import torch

model.eval()
example = torch.randn(1, 3, 32, 32)            # 形を伝えるダミー入力（バッチ1）
torch.onnx.export(
    model, (example,), "model.onnx",
    dynamo=False,                              # ★ onnxscript 不要の旧経路を明示
    opset_version=18,                          # 使う onnxruntime が読める版
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # バッチを動的軸に
)
```

**落とし穴になりやすい注意**: エクスポートは必ず `model.eval()` の **fp32 モデル**から行う。量子化済みモデルや MPS 上のモデルはエクスポートで失敗しやすいためで、量子化は ONNX 化の**後で** onnxruntime 側でやるのが安全だ。もう一つ、`nn.TransformerEncoder` は `eval()` かつ `inference_mode` 下で**融合カーネル**に切り替わり、旧エクスポータが対応しないことがある（`05_optimum_transformers.py` で実際にこれに遭遇し、トレースを grad 有効で行って回避している）。

## 3. 数値一致の検証 — `onnx.checker` と `atol/rtol`

エクスポートして「ファイルができた」で満足してはいけない。その先で **変換が数値的に正しいかを必ず検証**する必要があり、検証は 2 段構えで行う。まず **静的チェック**として `onnx.checker.check_model(onnx.load(path))` を通す。これはグラフの型・shape・opset の整合性を調べ、壊れたグラフなら例外を投げる。いわば「文法チェック」に相当するもので、通っても**数値が合う保証まではない**。

次に**動的チェック**として、**同じ入力**を torch と onnxruntime の両方に流し、出力が一致するかを `numpy.allclose(got, ref, atol, rtol)` で判定する。あわせて **最大絶対誤差** `max(|got - ref|)` も必ずログに出す。fp32 のまっとうな変換なら、誤差は `1e-6` オーダー（浮動小数の演算順序差程度）に収まるはずで、実際 `01_export_and_verify.py` でも最大絶対誤差 `~6e-7`、top-1 予測一致率 `1.000` になる。逆に **ここを省いて壊れた ONNX を本番に出す**のが、この分野で最も多くて最も痛い事故だ。

<figure class="lec-fig"><svg viewBox="0 0 640 272" role="img" aria-label="ONNX変換の検証は2段階。onnxcheckerで静的整合を確認した後、同じ入力をtorchとonnxruntimeに流しnpallcloseで数値一致を見る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="24" y="28" font-size="13" font-weight="700" fill="#c2410c">① 静的チェック（文法）</text><rect x="24" y="42" width="84" height="46" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><text x="66" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">model.onnx</text><line x1="108" y1="65" x2="148" y2="65" stroke="#71717a" stroke-width="1.8"/><polygon points="154,65 146,61 146,69" fill="#71717a"/><rect x="158" y="42" width="192" height="46" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="254" y="64" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">onnx.checker.check_model</text><text x="254" y="81" text-anchor="middle" font-size="10.5" fill="#52525b">型・shape・opset の整合</text><line x1="350" y1="65" x2="390" y2="65" stroke="#71717a" stroke-width="1.8"/><polygon points="396,65 388,61 388,69" fill="#71717a"/><rect x="400" y="42" width="170" height="46" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="485" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">整合 OK ✓</text><line x1="24" y1="108" x2="616" y2="108" stroke="#e4e4e7" stroke-width="1.5"/><text x="24" y="134" font-size="13" font-weight="700" fill="#c2410c">② 数値チェック（同じ入力で出力を比較）</text><rect x="24" y="176" width="86" height="44" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="67" y="203" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">同じ入力</text><line x1="110" y1="198" x2="158" y2="176" stroke="#71717a" stroke-width="1.8"/><polygon points="164,174 153,173 156,183" fill="#71717a"/><line x1="110" y1="198" x2="158" y2="220" stroke="#71717a" stroke-width="1.8"/><polygon points="164,222 153,213 156,223" fill="#71717a"/><rect x="166" y="154" width="150" height="40" rx="6" fill="#f4f4f5" stroke="#52525b" stroke-width="1.8"/><text x="241" y="179" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">torch（基準 ref）</text><rect x="166" y="202" width="150" height="40" rx="6" fill="#f4f4f5" stroke="#52525b" stroke-width="1.8"/><text x="241" y="227" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">onnxruntime（got）</text><line x1="316" y1="174" x2="358" y2="192" stroke="#71717a" stroke-width="1.8"/><polygon points="366,196 354.8,196.5 358.9,187.4" fill="#71717a"/><line x1="316" y1="222" x2="360" y2="202" stroke="#71717a" stroke-width="1.8"/><polygon points="366,200 355,199 356,209" fill="#71717a"/><rect x="368" y="170" width="124" height="56" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><text x="430" y="193" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">np.allclose</text><text x="430" y="211" text-anchor="middle" font-size="11" fill="#52525b">atol / rtol</text><line x1="492" y1="198" x2="520" y2="198" stroke="#71717a" stroke-width="1.8"/><polygon points="526,198 518,194 518,202" fill="#71717a"/><rect x="528" y="170" width="88" height="56" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="572" y="192" text-anchor="middle" font-size="11" fill="#52525b">max|Δ|</text><text x="572" y="212" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">≈6e-7 ✓</text></svg><figcaption>ONNX 変換の検証は <b>2 段階</b>です。①<b>静的チェック</b> <code>onnx.checker.check_model</code> は型・shape・opset の整合（文法）を見ますが、<b>数値が合う保証はありません</b>。②<b>数値チェック</b>は<b>同じ入力</b>を torch と onnxruntime に流し、<code>np.allclose(atol, rtol)</code> と<b>最大絶対誤差</b>で一致を確認します。fp32 なら誤差は <code>≈1e-6</code>、int8 量子化後は <code>≈0.05</code> 規模になるので <b>top-1 一致率</b>で評価します。</figcaption></figure>

`atol`（絶対許容）と `rtol`（相対許容）は、出力のスケールに応じて決める。例えば分類ロジットのように値域が ±数十なら `atol=1e-4, rtol=1e-3` 程度で十分だ。ただし int8 量子化後は誤差が桁違いに大きくなる（`~0.05` など）ので、量子化モデルの検証は「ロジットの一致」ではなく **「top-1 予測の一致率」や「accuracy の劣化幅」** で見るのが正しい（後述の三角評価）。

## 4. onnxruntime 推論とセッション最適化 — `InferenceSession` / `SessionOptions`

推論側の主役が `onnxruntime.InferenceSession` だ。`InferenceSession(path, sess_options, providers=["CPUExecutionProvider"])` でモデルをロードし、`sess.run(None, {入力名: numpy配列})` で実行する。入力名は `sess.get_inputs()[0].name` で取れる（export で付けた `"input"`）。引数の **プロバイダ**は実行バックエンドの指定で、pip 版 `onnxruntime` は CPU 用（`CPUExecutionProvider`）にあたる。GPU 用は別パッケージ `onnxruntime-gpu` だが、**両方を同時に入れるとプロバイダ競合でロードに失敗**するので入れない（落とし穴）。

この `SessionOptions` には二大ツマミがある。一つは `graph_optimization_level` で、**グラフ最適化の強さ**を `ORT_DISABLE_ALL`（無効）〜`ORT_ENABLE_ALL`（全部: 定数畳み込み・冗長ノード除去・演算子融合）の範囲で切り替える。もう一つは `intra_op_num_threads` で、**1 演算の内部並列スレッド数**を CPU の物理コア数に合わせて調整する。`02_ort_session_optimize.py` の実測では、TinyCNN で `DISABLE_ALL` → `ENABLE_ALL` が p50 で約 1.34 倍、スレッドを 1→2→4 と増やすとスループットが約 40k→70k→154k img/s と伸びた。

ここで肝心なのは、**「最適化レベルを上げれば必ず速い」「スレッドを増やせば必ず速い」は思い込み**だという点だ。実際、小さなモデルでは最適化の効果が誤差に埋もれたり、スレッド起動コストが勝って 1〜2 スレッドが最速だったりする。だからこそ推測せず計測する。加えて、**torch 側（`torch.set_num_threads`）と onnxruntime 側（`intra_op_num_threads`）のスレッド数を必ず揃えて**ベンチしないと比較が不公平になる（第34回の「公平なベンチ」原則の延長）。

## 5. グラフ最適化の中身と netron 可視化

では、onnxruntime の「グラフ最適化」は具体的に何をするのか。その代表が **演算子融合（operator fusion）** だ。例えば `Conv → ReLU` や `Gemm → ReLU` のように「重い演算 + 活性化」の並びは、1 つの融合演算（`FusedGemm` など）にまとめられる。融合すると中間結果をメモリに書いて読み直す往復が消え、カーネル呼び出し回数も減るので速くなる。これを確かめるため、`04_graph_inspect_netron.py` では `SessionOptions.optimized_model_filepath` を使って**最適化後のグラフ**をファイルに書き出し、最適化前後で op の数と種類を比較する。実測では `Relu` が消えてノード数が 10→9 になり、代わりに `FusedGemm` が現れる様子が見える。

<figure class="lec-fig"><svg viewBox="0 0 600 226" role="img" aria-label="グラフ最適化の演算子融合。GemmとReluの2ノードが1つのFusedGemmにまとまりノード数が10から9に減る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="150" y="32" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">最適化前（10 ノード）</text><rect x="48" y="76" width="88" height="52" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="92" y="108" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">Gemm</text><line x1="136" y1="102" x2="162" y2="102" stroke="#71717a" stroke-width="1.8"/><polygon points="168,102 160,98 160,106" fill="#71717a"/><rect x="170" y="76" width="80" height="52" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="210" y="108" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">Relu</text><line x1="272" y1="102" x2="350" y2="102" stroke="#c2410c" stroke-width="3"/><polygon points="360,102 346,95 346,109" fill="#c2410c"/><text x="314" y="90" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">融合 fusion</text><text x="314" y="124" text-anchor="middle" font-size="10.5" fill="#52525b">ENABLE_ALL</text><text x="470" y="32" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">最適化後（9 ノード）</text><rect x="382" y="72" width="172" height="60" rx="7" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><text x="468" y="98" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">FusedGemm</text><text x="468" y="118" text-anchor="middle" font-size="10.5" fill="#52525b">Gemm ＋ Relu を 1 つに</text><text x="300" y="174" text-anchor="middle" font-size="12.5" fill="#3f3f46">Relu が消え、中間結果のメモリ往復が減って <tspan fill="#15803d" font-weight="700">高速化</tspan></text><text x="300" y="200" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">ノード数 10 → 9</text></svg><figcaption><b>グラフ最適化</b>の代表が<b>演算子融合</b>です。<code>Gemm → Relu</code> のような「重い演算＋活性化」の並びは、<code>SessionOptions</code> の <code>ORT_ENABLE_ALL</code> で 1 つの <b>FusedGemm</b> に融合されます。中間結果をメモリへ書いて読み直す往復が消え、カーネル呼び出しも減るので<b>速くなります</b>（実測ではノード数 10 → 9、<code>Relu</code> が消滅）。</figcaption></figure>

こうしたグラフを**目で見る**定番ツールが **netron** だ。`pip install netron` して `netron.start("model.onnx")` するとブラウザでグラフが綺麗に表示され、各ノードの属性や形を対話的に確認できる（モデルが期待通りの構造か、変な分岐や余計なノードが無いかの点検に便利）。ただし本講座の標準環境には netron を入れていないため、本章では `onnx` の Python API で **「テキスト版 netron」**（入出力・op ヒストグラム・initializer サイズの要約）を作り、同じ目的を達する。

さらにグラフの点検は、**動的軸が意図通り張れているか**の確認にも使える。`04` の出力で `inputs: {'input': ['batch', 3, 32, 32]}` のようにバッチ次元が記号 `'batch'` になっていれば動的バッチは成功だ。逆に、ここが具体的な数値（例 `1`）に固定されていたら `dynamic_axes` の指定ミスを疑う。

## 6. 動的量子化 — CPU に最も手軽に効く int8 化

CPU 推論で**最も費用対効果が高い**圧縮が **動的量子化（dynamic quantization）** だ。`onnxruntime.quantization.quantize_dynamic(in_path, out_path, weight_type=QuantType.QUInt8)` の一行で、`Linear/MatMul/Conv` の**重みを int8 に量子化**し、**活性は推論時にその場で量子化**する。静的量子化と違って**キャリブレーション用データが不要**なため、導入が極めて簡単だ。`03_dynamic_quant.py` の実測では、TinyCNN のファイルサイズが `0.258MB → 0.072MB`（約 3.6 倍縮小）、accuracy はほぼ無劣化だった。

CPU の int8 行列演算は **U8(activation) × S8(weight)** の組み合わせが基本構成なので、`weight_type` には `QuantType.QUInt8` を指定する。ただし注意点もある。まず、**`Embedding` テーブルは量子化対象外**なので、語彙の大きいモデルではサイズ削減比が理論上限の 4 倍に届かない（`05` の Transformer は 1.53 倍）。さらに、**極小モデルでは量子化/逆量子化ノードの固定オーバーヘッドが勝ち、逆にサイズや時間が増える**こともある。だからここでも結論は「**必ず実測**」となる。

そして最重要の作法は、圧縮を **速度・サイズ・精度の三角関係**で評価することだ。サイズと速度だけ見て **accuracy を測らない**のは最悪のアンチパターンである（第35回と共通）。`mini_project.py` では `torch fp32 / ONNX fp32 / ONNX int8` の 3 者を **accuracy・latency(p50/p99)・throughput・model size(MB)** で並べて比較表にし、`03` では同じ 3 者を accuracy・p50・size に絞って比較表にする。こうして初めて「この用途ならどれを出すか」を意思決定できるようになる。なお **int8 が CPU で必ず速くなるわけではなく**、小モデル・少バッチでは fp32 の方が速いことも珍しくない。int8 の主効果は多くの場合「サイズ・メモリ削減」だと割り切るのが実務的だ。

<figure class="lec-fig"><svg viewBox="0 0 600 296" role="img" aria-label="圧縮は速度サイズ精度の三角関係で評価する。サイズや速度だけ見て精度を測らないのはアンチパターン" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><polygon points="300,64 96,236 504,236" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><circle cx="300" cy="64" r="7" fill="#16a34a"/><text x="300" y="44" text-anchor="middle" font-size="16" font-weight="700" fill="#15803d">精度 accuracy</text><circle cx="96" cy="236" r="7" fill="#2563eb"/><text x="96" y="262" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">速度 latency</text><circle cx="504" cy="236" r="7" fill="#ea580c"/><text x="504" y="262" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">サイズ size</text><text x="300" y="166" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">3 つ同時に測り</text><text x="300" y="190" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">用途で選ぶ</text><text x="300" y="288" text-anchor="middle" font-size="12" fill="#dc2626">サイズ・速度だけ見て精度を測らないのは ✗</text></svg><figcaption>圧縮の効果は <b>速度・サイズ・精度</b>の<b>三角関係</b>で評価します。サイズと速度だけ見て<b>精度（accuracy）を測らない</b>のは最悪のアンチパターンです。例えば int8 動的量子化は<b>サイズを大きく削減</b>（実測 3.6×）できますが、<b>速度は CPU で必ず速くなるとは限らず</b>、精度は必ず測って許容内か確認します。用途（レイテンシ／配布サイズ／精度のどれを最優先するか）に照らして選びます。</figcaption></figure>

## 7. Transformers の ONNX 化 — `optimum`（ORTModel）

HuggingFace Transformers のモデルを ONNX 化する正準ツールが **`optimum` / `optimum-onnx`** だ。`ORTModelForImageClassification.from_pretrained("microsoft/resnet-18", export=True)` のように `export=True` を付けるだけで、**ロード時にその場で ONNX へ変換**し、以降は onnxruntime で推論してくれる。`forward` の使い勝手は transformers のままで裏側だけ ORT に差し替わるので、既存コードからの移行が楽だ。さらに `save_pretrained(...)` で保存・量子化・配布もできる。

```python
# 概念（optimum は任意依存。導入: uv add --group onnx "optimum[onnx]"）
from optimum.onnxruntime import ORTModelForImageClassification
ort_model = ORTModelForImageClassification.from_pretrained("microsoft/resnet-18", export=True)
ort_model.save_pretrained("resnet18_onnx")
```

この `optimum` は重めの任意依存なので、本講座の標準環境には入れていない。そこで `05_optimum_transformers.py` では `import` を `try/except` で守り、**未導入なら概念紹介＋導入案内にフォールバック**しつつ、**同じ原理を自前の小型 Transformer エンコーダで実演**する: `torch.onnx.export → onnx.checker → onnxruntime で数値一致 → 動的量子化`。optimum が内部でやっているのも本質的にはこの **「export → verify → run（→ quantize）」** の流れであり、その原理を手で書けることが何より大事だからだ。なお Transformer は `Linear/MatMul` の比率が高いので、CPU では int8 動的量子化の費用対効果が（CNN 以上に）出やすいことも覚えておきたい。

## 8. 実務の使い分け（意思決定の順序）

最後に、この章の手法を**いつ使うか**を整理しておく。推論を速く・軽くしたいときは、いきなり量子化に飛びつくのではなく、**コストの低い順**に試すのが鉄則だ（第34〜37回を貫く意思決定順序）。具体的には次の順序で進める。

1. **まず `eval()` + `inference_mode()` と正しいベンチ**（第34回）。ここを外すと以降の比較が全部崩れる。
2. **CPU なら bf16 autocast / `torch.compile`** を試す（第34・37回）。
3. **ONNX 化して onnxruntime で回す**（本章）。CPU では eager torch より速いことが多く、可搬性も得られる。多くの実務では**ここまでで十分**。
4. **ONNX 動的量子化（int8）**（本章）。サイズ・メモリが課題なら有力。**accuracy を測って**許容内か確認。
5. それでも精度劣化が大きい/さらに速くしたいなら、**静的 PTQ・QAT**（第35回）や**枝刈り**、**プラットフォーム別ランタイム**（OpenVINO/CoreML/TensorRT、第37回）へ。

<figure class="lec-fig"><svg viewBox="0 0 680 250" role="img" aria-label="推論最適化はコストの低い順に試す。ベンチ整備、bf16やtorch.compile、ONNX化とonnxruntime、int8動的量子化、静的PTQやQATの順で、多くの実務はONNX化までで十分" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">最適化はコストの低い順に — 各段で速度・サイズ・精度を測る</text><rect x="12" y="62" width="120" height="84" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="146" y="62" width="120" height="84" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="280" y="62" width="120" height="84" rx="8" fill="#ffedd5" stroke="#16a34a" stroke-width="2.5"/><rect x="414" y="62" width="120" height="84" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="548" y="62" width="120" height="84" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="72" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">① ベンチ整備</text><text x="72" y="110" text-anchor="middle" font-size="10.5" fill="#52525b">eval()+計測</text><text x="72" y="128" text-anchor="middle" font-size="10.5" fill="#71717a">第34回</text><text x="206" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">② bf16/compile</text><text x="206" y="110" text-anchor="middle" font-size="10.5" fill="#52525b">autocast</text><text x="206" y="128" text-anchor="middle" font-size="10.5" fill="#71717a">第34・37回</text><text x="340" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">③ ONNX 化</text><text x="340" y="107" text-anchor="middle" font-size="10.5" fill="#52525b">+ onnxruntime</text><text x="340" y="127" text-anchor="middle" font-size="10.5" font-weight="700" fill="#15803d">多くはここまで</text><text x="474" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">④ int8 量子化</text><text x="474" y="110" text-anchor="middle" font-size="10.5" fill="#52525b">動的・要 acc 測定</text><text x="474" y="128" text-anchor="middle" font-size="10.5" fill="#71717a">本章</text><text x="608" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">⑤ PTQ/QAT 等</text><text x="608" y="110" text-anchor="middle" font-size="10.5" fill="#52525b">静的量子化・枝刈り</text><text x="608" y="128" text-anchor="middle" font-size="10.5" fill="#71717a">第35・37回</text><line x1="132" y1="104" x2="140" y2="104" stroke="#71717a" stroke-width="2"/><polygon points="146,104 138,100 138,108" fill="#71717a"/><line x1="266" y1="104" x2="274" y2="104" stroke="#71717a" stroke-width="2"/><polygon points="280,104 272,100 272,108" fill="#71717a"/><line x1="400" y1="104" x2="408" y2="104" stroke="#71717a" stroke-width="2"/><polygon points="414,104 406,100 406,108" fill="#71717a"/><line x1="534" y1="104" x2="542" y2="104" stroke="#71717a" stroke-width="2"/><polygon points="548,104 540,100 540,108" fill="#71717a"/><line x1="20" y1="200" x2="648" y2="200" stroke="#c2410c" stroke-width="2"/><polygon points="656,200 646,195 646,205" fill="#c2410c"/><text x="334" y="224" text-anchor="middle" font-size="12" fill="#c2410c">コスト・手間：小 → 大（迷ったら左から。各段で accuracy を必ず測る）</text></svg><figcaption>推論を速く・軽くするときは <b>コストの低い順</b>に試します。<b>① ベンチ整備（第34回）→ ② bf16 / torch.compile → ③ ONNX 化＋onnxruntime → ④ int8 動的量子化 → ⑤ 静的 PTQ・QAT・枝刈り</b> の順で、<b>多くの実務は ③ まで</b>で十分です。下に伸びる<b>コスト軸</b>のとおり右へ行くほど手間と精度劣化リスクが増えるので、<b>各段で必ず速度・サイズ・精度を測って</b>から次へ進みます。</figcaption></figure>

そして各段で必ず **「速度・サイズ・精度」の三角**を同一指標で測り、用途（レイテンシ最優先か、配布サイズ最優先か、精度最優先か）に照らして選ぶ。これがこの章の到達点だ。

---

## 🛠 章末ミニプロジェクト — PyTorch → ONNX デプロイ・ベンチ一式

`mini_project.py` は、この章の全要素を 1 本のデプロイ・ワークフローに統合した完成形（実際に動く）。

1. **学習済みモデル**を用意（TinyCNN を合成図形データで軽く学習）
2. **ONNX エクスポート**（`dynamo=False` / 動的バッチ）
3. **検証**: `onnx.checker` ＋ torch との **数値一致（atol/rtol・最大絶対誤差）**。一致しなければ**デプロイ中止**して理由を表示
4. **onnxruntime 推論**（グラフ最適化・スレッド固定）
5. **動的量子化（int8）**で圧縮
6. `torch fp32 / ONNX fp32 / ONNX int8` を **accuracy・p50・p99・throughput・size(MB)** で**公平比較**し、結果から**用途別の推奨を意思決定**。比較表・図（`mini_project.png`）・レポート（`mini_project_report.json`）を出力

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="ミニプロジェクトの6ステップ。学習済みモデルからONNXエクスポート、検証へ進み、検証で不一致ならデプロイ中止、一致ならonnxruntime推論、動的量子化、公平比較と意思決定へ進む" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — export から意思決定までを 1 周</text><rect x="28" y="62" width="180" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="62" width="180" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="452" y="62" width="180" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2.5"/><rect x="452" y="210" width="180" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="210" width="180" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="28" y="210" width="180" height="60" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="118" y="88" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 学習済みモデル</text><text x="118" y="108" text-anchor="middle" font-size="11" fill="#71717a">TinyCNN を軽く学習</text><text x="330" y="88" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② ONNX エクスポート</text><text x="330" y="108" text-anchor="middle" font-size="11" fill="#71717a">dynamo=False・動的バッチ</text><text x="542" y="88" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 検証</text><text x="542" y="108" text-anchor="middle" font-size="11" fill="#71717a">checker ＋ 数値一致</text><text x="542" y="236" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">④ onnxruntime 推論</text><text x="542" y="256" text-anchor="middle" font-size="11" fill="#71717a">最適化・スレッド固定</text><text x="330" y="236" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">⑤ 動的量子化 int8</text><text x="330" y="256" text-anchor="middle" font-size="11" fill="#71717a">quantize_dynamic</text><text x="118" y="236" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">⑥ 公平比較→意思決定</text><text x="118" y="256" text-anchor="middle" font-size="11" fill="#71717a">acc・p50・p99・size</text><line x1="208" y1="92" x2="234" y2="92" stroke="#71717a" stroke-width="2"/><polygon points="240,92 232,88 232,96" fill="#71717a"/><line x1="420" y1="92" x2="446" y2="92" stroke="#71717a" stroke-width="2"/><polygon points="452,92 444,88 444,96" fill="#71717a"/><line x1="542" y1="122" x2="542" y2="204" stroke="#16a34a" stroke-width="2"/><polygon points="542,210 537,200 547,200" fill="#16a34a"/><text x="556" y="170" font-size="11.5" font-weight="700" fill="#15803d">一致 ✓</text><line x1="500" y1="122" x2="410" y2="144" stroke="#dc2626" stroke-width="2"/><polygon points="400,146 408.6,138.8 410.9,148.5" fill="#dc2626"/><text x="470" y="134" text-anchor="middle" font-size="11.5" font-weight="700" fill="#dc2626">不一致 ✗</text><rect x="232" y="146" width="200" height="40" rx="8" fill="#ffffff" stroke="#dc2626" stroke-width="2"/><text x="332" y="171" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">デプロイ中止して理由表示</text><line x1="452" y1="240" x2="426" y2="240" stroke="#71717a" stroke-width="2"/><polygon points="420,240 428,236 428,244" fill="#71717a"/><line x1="240" y1="240" x2="214" y2="240" stroke="#71717a" stroke-width="2"/><polygon points="208,240 216,236 216,244" fill="#71717a"/></svg><figcaption><b>章末ミニプロジェクト</b>のデプロイ・ワークフローは <b>① 学習済みモデル → ② ONNX エクスポート → ③ 検証</b> と進み、③ で <b>torch と onnxruntime の数値が一致</b>すれば <b>④ onnxruntime 推論 → ⑤ 動的量子化(int8) → ⑥ 公平比較→意思決定</b> へ続きます。③ で<b>一致しなければデプロイを中止</b>して理由を表示する（赤）のが要で、壊れた ONNX を本番に出さないための関所です。</figcaption></figure>

実行例（実測の一例。環境で数値は変わる）:

```
[5] 公平比較（評価 240 枚・1 スレッド）
    手法               acc   p50(ms)   p99(ms)     img/s       MB
    torch fp32     0.846    16.096    17.628     14782    0.258
    ONNX fp32      0.846     5.966     6.882     39904    0.258
    ONNX int8      0.846     9.566    10.574     24968    0.072
[6] 意思決定: ONNX fp32 は eager torch より p50 が速い → CPU 配布の既定候補 / int8 はサイズ3.6x縮小・精度劣化±0.000
```

**発展課題**: (a) モデルを `torchvision.models.resnet18` に差し替えて同じワークフローを回す（重み DL あり）。(b) バッチサイズを 1/8/64 と変えてスループット曲線を描く。(c) `intra_op_num_threads` を物理コア数まで振って最速点を探す。(d) 静的量子化（onnxruntime の `quantize_static` + `CalibrationDataReader`）に挑戦し、動的量子化と精度・速度を比べる。

---

## ✅ 到達チェックリスト

- [ ] `torch.onnx.export` で `.onnx` を書き出せる。`dynamo=True`（新既定・onnxscript 必須）と `dynamo=False`（旧・本章の正準）の違いを説明できる
- [ ] `opset_version` の意味と「上げ過ぎ/下げ過ぎ」のリスクを説明できる
- [ ] `dynamic_axes` でバッチ次元を動的軸にし、export と違うバッチで推論できる
- [ ] エクスポート後に `onnx.checker.check_model` と **torch との数値一致（atol/rtol・最大絶対誤差）** の **両方**で検証できる
- [ ] `InferenceSession(providers=["CPUExecutionProvider"])` で推論でき、`onnxruntime` と `onnxruntime-gpu` を同時に入れない理由を言える
- [ ] `SessionOptions` の `graph_optimization_level` と `intra_op_num_threads` を設定し、効果を**計測**で確かめられる
- [ ] グラフ最適化（演算子融合）が何をするかを、最適化前後の op ヒストグラムで説明できる
- [ ] `quantize_dynamic(weight_type=QuantType.QUInt8)` で int8 化でき、サイズ削減を実測できる
- [ ] 量子化を **速度・サイズ・精度の三角**で評価し、「int8 は CPU で必ずしも速くならない」を理解している
- [ ] `optimum` の `ORTModelForXXX(export=True)` の役割と、その中身が「export→verify→run」であることを説明できる
- [ ] netron（または onnx API）でグラフ構造・動的軸を点検できる

---

## ❓ 落とし穴・FAQ・デバッグ

**Q1. `ModuleNotFoundError: No module named 'onnxscript'` が出てエクスポートできない。**
A. PyTorch 2.9+ では `torch.onnx.export` の既定が `dynamo=True` で、これは `onnxscript` を必須とする。本講座は `dynamo=False`（旧 TorchScript ベース）を明示して回避する。新方式を使いたいなら `uv add --group onnx onnxscript` を足す。

**Q2. `dynamo=False` にすると `DeprecationWarning` が出る。問題ない？**
A. 問題ない。将来は新エクスポータが既定になるという案内であって、機能上は安定して動く。本章はクリーンな出力のため `warnings.filterwarnings("ignore")` で抑制している（中身はこの README で解説済み）。

**Q3. ONNX 化したら出力が torch と少しズレる。**
A. fp32 で最大絶対誤差が `1e-5` 以下なら、浮動小数の演算順序差なので正常（`atol=1e-4, rtol=1e-3` で `allclose` 通過が目安）。**大きくズレる**場合は、(a) `eval()` 忘れ（Dropout/BN が動いている）、(b) 動的でない入力形の取り違え、(c) 非対応 op の近似変換、などを疑う。**int8 量子化後**は誤差が `0.05` 規模になるのが普通なので、ロジット一致ではなく **top-1 一致率/accuracy 劣化**で評価する。

**Q4. `quantize_dynamic` でサイズが小さくならない／逆に大きくなった。**
A. (a) `Embedding` は量子化対象外なので語彙が大きいと比が伸びない。(b) **極小モデル**では量子化ノードの固定オーバーヘッドが勝って増えることがある（`05` で実演）。サイズが本当に効くのは Linear/Conv の重みが支配的な、ある程度の大きさのモデル。

**Q5. int8 にしたのに CPU で速くならない／むしろ遅い。**
A. よくある。動的量子化は実行時に量子化/逆量子化を挟むため、小モデル・少バッチではそのコストが勝つ。int8 の主効果は「サイズ・メモリ削減」と割り切り、速度は**必ず実測**で判断する（第35回「実効速度の罠」と同根）。

**Q6. `optimum` をインポートできない。**
A. 任意依存なので標準では未導入。`uv add --group onnx "optimum[onnx]"` で追加できる。本章は未導入でも `05` が概念紹介＋自前実装にフォールバックして動く。

**Q7. onnxruntime のロードに失敗する／プロバイダが見つからない。**
A. `onnxruntime`（CPU）と `onnxruntime-gpu` を**同時にインストール**していないか確認する。競合してロードに失敗する。CPU 講座では `onnxruntime` のみ。`sess.get_providers()` で実際に使われているプロバイダを確認できる。

**Q8. `nn.TransformerEncoder` のエクスポートが `aten::_transformer_encoder_layer_fwd is not supported` で落ちる。**
A. `eval()` + `inference_mode` 下で融合カーネル（fast path）に切り替わり、旧エクスポータが対応しないため。トレースを **grad 有効（`inference_mode` で包まない）** で行うと分解された経路になり export できる（`05` で実装）。

**Q9. 計測がブレる／ベンチが不公平。**
A. (a) **ウォームアップ**を入れて初回の最適化・キャッシュ miss を除外する。(b) torch（`set_num_threads`）と ORT（`intra_op_num_threads`）の**スレッド数を揃える**。(c) p50 だけでなく **p99** も見る（テール遅延）。これらは第34回の手順をそのまま適用する。

---

## 🚀 発展トピック・参考

- **静的量子化（static PTQ）**: `onnxruntime.quantization.quantize_static` + `CalibrationDataReader`。代表データで activation 範囲を事前推定し、活性も固定 int8 にする。Conv 主体の CNN で動的量子化より効きやすいが、キャリブレーションデータが要る。
- **QAT（量子化を意識した学習）**: 学習時に fake-quant を挟んで精度を回復（第35回）。PTQ で精度が足りないときの次の一手。
- **IOBinding**: 入出力テンソルのコピーを省いてオーバーヘッドを削る上級 API（高頻度推論で効く）。
- **dynamo / `torch.export` 経路**: `uv add --group onnx onnxscript` で `dynamo=True`。制御フローや動的形をより正確に捉える将来本命。`torch.export.export` と `dynamic_shapes`（ネスト構造）も関連。
- **プラットフォーム別ランタイム（第37回）**: OpenVINO（Intel/CPU）・CoreML（Mac）・LiteRT（モバイル）・TensorRT（NVIDIA GPU）。ONNX はこれらへの共通入口になることが多い。
- **公式ドキュメント**: [onnxruntime](https://onnxruntime.ai/docs/) / [ONNX](https://onnx.ai/onnx/) / [torch.onnx](https://docs.pytorch.org/docs/stable/onnx.html) / [optimum-onnx](https://huggingface.co/docs/optimum-onnx) / [netron](https://github.com/lutzroeder/netron)。

---

## ▶ 動かし方

```bash
# 依存（未導入なら）: 深層学習の土台 + ONNX 一式
uv sync --group dl --group onnx

# 本編（番号順）。モデル DL は不要（合成データ・小型モデルで完結）
uv run python lectures/36_onnx_runtime/01_export_and_verify.py      # export と数値一致検証
uv run python lectures/36_onnx_runtime/02_ort_session_optimize.py   # グラフ最適化・スレッド
uv run python lectures/36_onnx_runtime/03_dynamic_quant.py          # int8 動的量子化(三角評価)
uv run python lectures/36_onnx_runtime/04_graph_inspect_netron.py   # グラフ点検 + netron 概念
uv run python lectures/36_onnx_runtime/05_optimum_transformers.py   # Transformers ONNX 化(optimum)

# 章末ミニプロジェクト（export→検証→最適化→量子化→公平比較→意思決定）
uv run python lectures/36_onnx_runtime/mini_project.py

# 演習（自己採点。未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/36_onnx_runtime/exercises.py
uv run python lectures/36_onnx_runtime/exercises_solutions.py
```

なお `01`〜`05`・`mini_project`・`exercises` は、共通土台 `onnx_helpers.py`（export/検証/ベンチ/合成データを提供）を `import` して再利用している。

成果物（図・JSON・`.onnx`）は `outputs/36_onnx_runtime/` に保存される（matplotlib=Agg、OpenCV headless）。

---

> 参照ライブラリ: **torch 2.12+cpu** / **torchvision 0.27+cpu** / **onnx 1.21** / **onnxruntime 1.26**（optimum・onnxscript・netron は任意）
> （CPU 前提・`model.eval()` + `torch.inference_mode()`、`torch.onnx.export(dynamo=False)` を正準、合成データ・小型モデルで DL 不要、matplotlib=Agg、headless OpenCV） — 2026-06
