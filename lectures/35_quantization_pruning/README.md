# 35_quantization_pruning: 量子化と枝刈り — PTQ(動的/静的)・QAT・torchao・pruning の「実効速度の罠」

> トラック: **最適化・デプロイ** ／ レベル: **中級** ／ 必要な依存グループ: `dl`（`quant`=torchao は任意）
> 前提: 第34回「推論プロファイリング」。`model.eval()` + `torch.inference_mode()`、ウォームアップ→多数回→分位点(p50/p99)の正しいベンチ手順を一度通していると、本章の before/after 計測がそのまま読めます。

---

## 🎯 この章のゴール

- **int8 量子化の正体**（`scale` / `zero_point`・対称/非対称・per-tensor/per-channel）を手計算で理解し、「実数を 256 段の整数格子へ丸める写像」だと体に入れる。
- **動的PTQ**（`quantize_dynamic`・Linear/LSTM 向け・キャリブレーション不要）と**静的PTQ**（`fuse_modules` + `QuantStub/DeQuantStub` + キャリブレーション・Conv にも効く）と**QAT**（fake-quant で微調整）を、**同じ小モデル**に適用して使い分けられる。
- 本章最大の山場 **「枝刈りの実効速度の罠」** を実測で確かめる:**非構造化プルーニングは 0 マスクを掛けるだけで、`prune.remove` しても密(dense)テンソルのまま**。だから CPU の実レイテンシも保存サイズも縮まない。実圧縮には**構造化プルーニング + 小さい密モデルへのリビルド**か量子化が要る。
- 圧縮を **三角関係（精度 / レイテンシ p50・p99 / モデルサイズ MB）** で評価し、用途に応じた**手法選択の意思決定順序**を持つ。
- 現行の正準量子化 API が `torch.ao.quantization`（非推奨）から **torchao の `quantize_`** へ移行中であること、ただし CPU での実効速度は限定的で x86 CPU は int8 系 / ONNX 動的量子化が確実、という現実を知る。

この章の成果物は **「圧縮手法ベンチ」**（同一モデルに各手法を適用し、精度/速度/サイズを一表で比較し意思決定する）です。`mini_project.py` が完成形として動きます。

---

## 1. 直感 — 「小さく・速く」には3つの独立した刃がある

学習し終えたモデルをデプロイする段になると、目的は「精度を上げる」ことから「同じ精度を**速く・小さく**動かす」ことへと変わります。このとき使える刃は大きく3つあります。まず**量子化**は、重みや活性を fp32(32bit) から int8(8bit) へ落として「1要素あたりの情報量」を減らします。次に**枝刈り(pruning)**は、重要でない重みを 0 にして「使う重みの数」を減らします。そして**蒸留**（第38/39回）は、小さいモデルに大きいモデルの知識を移します。本章はこのうち前2つ、量子化と枝刈りを扱います。いずれも「精度を少しだけ犠牲にして、速度とサイズを稼ぐ」取引であり、**どれだけ犠牲を払ったか**を必ず測るのが鉄則です。

量子化の効き目は直感的に理解できます。fp32 は 1 要素 4 バイト、int8 は 1 バイトなので、素朴に考えればモデルは 1/4 に縮みます。さらに int8 の積和は整数演算ユニットで高速に回せるため、対応するカーネルさえあれば速くもなります。落とし穴は「精度がどれだけ落ちるか」と「**そのカーネルが手元の CPU に本当にあるか**」の2点です。たとえば Conv を int8 で実行するには activation の範囲を事前に知る必要があり（静的PTQ）、Linear だけならその場での量子化で足ります（動的PTQ）。つまり、この「**何が量子化されるか**」がモデル構造（Conv 主体か Linear 主体か）と噛み合うかどうかで、効果は大きく変わるのです。

枝刈りは、量子化よりもさらに罠が深い手法です。重みの 90% を 0 にすればモデルは 1/10 になりそうに思えますが、**`torch.nn.utils.prune` が実際にするのは「0 のマスクを掛ける」ことだけ**です。テンソルの形も dtype も変わらないので、密行列の積（GEMM）がそのまま走り、0 を掛け算しても計算量は減りません。`prune.remove` を呼んでもマスクを重みに焼き付けるだけで、やはり密のままです。**疎(sparse)フォーマットに変換すれば「保存サイズ」は縮められますが、CPU には疎行列の高速カーネルがほぼ無いため「実行速度」は縮みません**。本当に速く・小さくしたいなら、チャネルやニューロンを丸ごと落とす**構造化枝刈り**によって、小さい密モデルに作り直す必要があります。本章では、この差を `04` で数字として突きつけます。

<figure class="lec-fig"><svg viewBox="0 0 660 290" role="img" aria-label="非構造化プルーニングは0マスクで密のまま速くも小さくもならず、構造化プルーニング+リビルドは小さい密モデルで実際に縮む" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">枝刈りの実効速度の罠 — マスクは密のまま／リビルドで本当に縮む</text><line x1="40" y1="150" x2="620" y2="150" stroke="#e4e4e7" stroke-width="1.5"/><text x="28" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626" style="writing-mode:vertical-rl;text-orientation:upright">非構造化</text><rect x="60" y="58" width="96" height="64" fill="#2563eb"/><line x1="76" y1="58" x2="76" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="92" y1="58" x2="92" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="108" y1="58" x2="108" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="124" y1="58" x2="124" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="140" y1="58" x2="140" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="60" y1="74" x2="156" y2="74" stroke="#ffffff" stroke-width="1.4"/><line x1="60" y1="90" x2="156" y2="90" stroke="#ffffff" stroke-width="1.4"/><line x1="60" y1="106" x2="156" y2="106" stroke="#ffffff" stroke-width="1.4"/><polygon points="170,82 188,90 170,98" fill="#ea580c"/><text x="179" y="74" text-anchor="middle" font-size="11" fill="#c2410c">0マスク</text><rect x="200" y="58" width="96" height="64" fill="#2563eb"/><rect x="216" y="58" width="16" height="16" fill="#e4e4e7"/><rect x="264" y="58" width="16" height="16" fill="#e4e4e7"/><rect x="200" y="74" width="16" height="16" fill="#e4e4e7"/><rect x="248" y="74" width="16" height="16" fill="#e4e4e7"/><rect x="280" y="74" width="16" height="16" fill="#e4e4e7"/><rect x="232" y="90" width="16" height="16" fill="#e4e4e7"/><rect x="280" y="90" width="16" height="16" fill="#e4e4e7"/><rect x="216" y="106" width="16" height="16" fill="#e4e4e7"/><rect x="264" y="106" width="16" height="16" fill="#e4e4e7"/><line x1="216" y1="58" x2="216" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="232" y1="58" x2="232" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="248" y1="58" x2="248" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="264" y1="58" x2="264" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="280" y1="58" x2="280" y2="122" stroke="#ffffff" stroke-width="1.4"/><line x1="200" y1="74" x2="296" y2="74" stroke="#ffffff" stroke-width="1.4"/><line x1="200" y1="90" x2="296" y2="90" stroke="#ffffff" stroke-width="1.4"/><line x1="200" y1="106" x2="296" y2="106" stroke="#ffffff" stroke-width="1.4"/><text x="178" y="142" text-anchor="middle" font-size="11.5" fill="#52525b">同じ (out, in) の密テンソル → GEMM そのまま</text><rect x="330" y="70" width="300" height="44" rx="6" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="480" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">速度も保存サイズも 変わらない</text><text x="480" y="107" text-anchor="middle" font-size="11" fill="#52525b">prune.remove しても密のまま</text><text x="28" y="202" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d" style="writing-mode:vertical-rl;text-orientation:upright">構造化</text><rect x="60" y="170" width="96" height="64" fill="#16a34a"/><line x1="76" y1="170" x2="76" y2="234" stroke="#ffffff" stroke-width="1.4"/><line x1="92" y1="170" x2="92" y2="234" stroke="#ffffff" stroke-width="1.4"/><line x1="108" y1="170" x2="108" y2="234" stroke="#ffffff" stroke-width="1.4"/><line x1="124" y1="170" x2="124" y2="234" stroke="#ffffff" stroke-width="1.4"/><line x1="140" y1="170" x2="140" y2="234" stroke="#ffffff" stroke-width="1.4"/><line x1="60" y1="186" x2="156" y2="186" stroke="#ffffff" stroke-width="1.4"/><line x1="60" y1="202" x2="156" y2="202" stroke="#ffffff" stroke-width="1.4"/><line x1="60" y1="218" x2="156" y2="218" stroke="#ffffff" stroke-width="1.4"/><line x1="60" y1="194" x2="156" y2="194" stroke="#dc2626" stroke-width="2.5"/><line x1="60" y1="226" x2="156" y2="226" stroke="#dc2626" stroke-width="2.5"/><polygon points="170,194 188,202 170,210" fill="#ea580c"/><text x="179" y="166" text-anchor="middle" font-size="11" fill="#c2410c">行（出力ch）を削除</text><rect x="200" y="186" width="96" height="32" fill="#16a34a"/><line x1="216" y1="186" x2="216" y2="218" stroke="#ffffff" stroke-width="1.4"/><line x1="232" y1="186" x2="232" y2="218" stroke="#ffffff" stroke-width="1.4"/><line x1="248" y1="186" x2="248" y2="218" stroke="#ffffff" stroke-width="1.4"/><line x1="264" y1="186" x2="264" y2="218" stroke="#ffffff" stroke-width="1.4"/><line x1="280" y1="186" x2="280" y2="218" stroke="#ffffff" stroke-width="1.4"/><line x1="200" y1="202" x2="296" y2="202" stroke="#ffffff" stroke-width="1.4"/><rect x="330" y="190" width="300" height="44" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><text x="480" y="210" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">小さい密モデル → 速く・小さい</text><text x="480" y="227" text-anchor="middle" font-size="11" fill="#52525b">行・チャネルを丸ごと削減</text></svg><figcaption>枝刈りの <b>実効速度の罠</b>です。<b>非構造化プルーニング</b>は重要でない重みに <b>0 のマスク</b>を掛けるだけで、テンソルの<b>形も dtype も変わらず密(dense)のまま</b>。密行列積（GEMM）がそのまま走るので <b>CPU では速くも小さくもなりません</b>（<code>prune.remove</code> はマスクを焼くだけ）。対して <b>構造化プルーニング+リビルド</b>は <b>行＝出力チャネルを丸ごと削除</b>して <b>小さい密モデル</b>に作り直すため、サイズもレイテンシも実際に縮みます。</figcaption></figure>

---

## 2. 理論 — scale / zero-point、対称/非対称、per-channel、PTQ と QAT

int8 量子化とは、**実数区間 `[x_min, x_max]` を整数区間 `[qmin, qmax]`（int8 なら 0..255 か -128..127）へ線形写像する**ことです。この写像は2つの数だけで決まります。`scale` は「整数 1 段あたりの実数の刻み幅」を、`zero_point` は「実数の 0.0 がどの整数に当たるか」を表します。これらを使うと、量子化は `q = round(x/scale) + zero_point`（範囲外はクランプ）、逆量子化は `x ≈ scale * (q - zero_point)` と書けます。ここで写像には2系統あります。**対称(symmetric)**は `zero_point=0` に固定し `|x|max` から scale を決める方式で、0 を中心に正負対称へ分布する**重み**に向いています。一方**非対称(asymmetric/affine)**は zero_point を使うことで、片側に寄った分布（ReLU 後の非負な**活性**など）にぴったり合わせられます。`01` では同じ非負ベクトルに両方を当て、非対称の方が MAE が小さくなる（zero_point によって 0.0 を整数へ正確に置けるため）ことを確かめます。

<figure class="lec-fig"><svg viewBox="0 0 640 280" role="img" aria-label="量子化は連続な実数を0〜255の整数格子へ丸める線形写像。scaleは1段の刻み幅、zero_pointは0.0が当たる整数位置" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="24" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">実数 x を int8 格子（256段）へ丸める線形写像</text><line x1="80" y1="82" x2="560" y2="82" stroke="#2563eb" stroke-width="2.5"/><polygon points="568,82 556,77 556,87" fill="#2563eb"/><text x="82" y="66" font-size="13" fill="#1d4ed8">実数 x（連続）</text><circle cx="200" cy="82" r="4.5" fill="#1d4ed8"/><text x="200" y="66" text-anchor="middle" font-size="12" fill="#3f3f46">0.0</text><circle cx="340" cy="82" r="5" fill="#ea580c"/><circle cx="430" cy="82" r="5" fill="#ea580c"/><circle cx="510" cy="82" r="5" fill="#ea580c"/><line x1="200" y1="88" x2="200" y2="182" stroke="#1d4ed8" stroke-width="1.6" stroke-dasharray="5 3"/><line x1="340" y1="88" x2="360" y2="182" stroke="#71717a" stroke-width="1.4" stroke-dasharray="5 3"/><line x1="430" y1="88" x2="440" y2="182" stroke="#71717a" stroke-width="1.4" stroke-dasharray="5 3"/><line x1="510" y1="88" x2="520" y2="182" stroke="#71717a" stroke-width="1.4" stroke-dasharray="5 3"/><rect x="80" y="182" width="480" height="32" fill="#ffedd5" stroke="#ea580c" stroke-width="1.8"/><line x1="120" y1="182" x2="120" y2="214" stroke="#f97316" stroke-width="1"/><line x1="160" y1="182" x2="160" y2="214" stroke="#f97316" stroke-width="1"/><line x1="240" y1="182" x2="240" y2="214" stroke="#f97316" stroke-width="1"/><line x1="280" y1="182" x2="280" y2="214" stroke="#f97316" stroke-width="1"/><line x1="320" y1="182" x2="320" y2="214" stroke="#f97316" stroke-width="1"/><line x1="360" y1="182" x2="360" y2="214" stroke="#f97316" stroke-width="1"/><line x1="400" y1="182" x2="400" y2="214" stroke="#f97316" stroke-width="1"/><line x1="440" y1="182" x2="440" y2="214" stroke="#f97316" stroke-width="1"/><line x1="480" y1="182" x2="480" y2="214" stroke="#f97316" stroke-width="1"/><line x1="520" y1="182" x2="520" y2="214" stroke="#f97316" stroke-width="1"/><line x1="200" y1="182" x2="200" y2="214" stroke="#c2410c" stroke-width="3"/><line x1="200" y1="176" x2="240" y2="176" stroke="#ea580c" stroke-width="1.4"/><line x1="200" y1="172" x2="200" y2="180" stroke="#ea580c" stroke-width="1.4"/><line x1="240" y1="172" x2="240" y2="180" stroke="#ea580c" stroke-width="1.4"/><text x="220" y="168" text-anchor="middle" font-size="12.5" fill="#ea580c">scale</text><text x="200" y="236" text-anchor="middle" font-size="12" fill="#c2410c">zero_point</text><text x="558" y="236" text-anchor="end" font-size="13" fill="#c2410c">int8 q（0〜255）</text><rect x="186" y="248" width="268" height="26" rx="5" fill="#f4f4f5"/><text x="320" y="266" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b" font-family="'JetBrains Mono', monospace">q = round(x / scale) + zero_point</text></svg><figcaption>int8 量子化は、<b>連続な実数 x</b> を <b>0〜255 の整数格子（256段）</b>へ丸める線形写像です。<code>scale</code> は<b>1段あたりの実数の刻み幅</b>、<code>zero_point</code> は<b>実数 0.0 が当たる整数の位置</b>を表します。<code>q = round(x/scale) + zero_point</code> で量子化し、<code>x ≈ scale·(q − zero_point)</code> で逆量子化します。</figcaption></figure>

もう一つの軸が、**per-tensor か per-channel か**という粒度の違いです。per-tensor はテンソル全体で scale を1つだけ持ち、per-channel は出力チャネル（重み行列の行）ごとに scale を持ちます。チャネル間で重みの振幅が桁違いに異なるとき（これはよくあります）、per-tensor では巨大チャネルに合わせた粗い scale のせいで、小振幅チャネルが潰れてしまいます。実際 `01` の実験では、行ごとに振幅が 0.01〜1000 倍まで違う重みに対し、per-channel が per-tensor の**約3倍**まで誤差を下げました。だからこそ、実用の重み量子化は **per-channel が既定**になっています（`get_default_qconfig` もそうなっています）。また、ビット幅を下げるほど段階数が減って誤差が増える（int8=256段、int4=16段、int2=4段）ことも合わせて押さえましょう。int8 は randn をほぼ保てる一方で、int2 の MAE は int8 の約100倍にもなりました。

<figure class="lec-fig"><svg viewBox="0 0 660 280" role="img" aria-label="per-tensorはテンソル全体でscale1つ、per-channelは行ごとにscale。小振幅チャネルはper-tensorで潰れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">per-tensor は scale 1つ ／ per-channel は行ごとに scale</text><line x1="330" y1="44" x2="330" y2="246" stroke="#e4e4e7" stroke-width="1.5"/><text x="175" y="52" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">per-tensor</text><rect x="70" y="66" width="210" height="20" fill="#2563eb"/><rect x="70" y="96" width="110" height="20" fill="#2563eb"/><rect x="70" y="126" width="26" height="20" fill="#2563eb"/><text x="104" y="140" font-size="11" fill="#dc2626">← 1段未満で潰れる</text><line x1="70" y1="156" x2="280" y2="156" stroke="#71717a" stroke-width="1.6"/><line x1="70" y1="156" x2="70" y2="164" stroke="#71717a" stroke-width="1.4"/><line x1="100" y1="156" x2="100" y2="164" stroke="#71717a" stroke-width="1.4"/><line x1="130" y1="156" x2="130" y2="164" stroke="#71717a" stroke-width="1.4"/><line x1="160" y1="156" x2="160" y2="164" stroke="#71717a" stroke-width="1.4"/><line x1="190" y1="156" x2="190" y2="164" stroke="#71717a" stroke-width="1.4"/><line x1="220" y1="156" x2="220" y2="164" stroke="#71717a" stroke-width="1.4"/><line x1="250" y1="156" x2="250" y2="164" stroke="#71717a" stroke-width="1.4"/><line x1="280" y1="156" x2="280" y2="164" stroke="#71717a" stroke-width="1.4"/><rect x="70" y="200" width="210" height="30" rx="5" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="175" y="220" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">小振幅チャネルが潰れる</text><text x="490" y="52" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">per-channel</text><rect x="370" y="66" width="210" height="20" fill="#16a34a"/><rect x="370" y="96" width="110" height="20" fill="#16a34a"/><rect x="370" y="126" width="26" height="20" fill="#16a34a"/><line x1="370" y1="90" x2="580" y2="90" stroke="#15803d" stroke-width="1.4"/><line x1="370" y1="90" x2="370" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="400" y1="90" x2="400" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="430" y1="90" x2="430" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="460" y1="90" x2="460" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="490" y1="90" x2="490" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="520" y1="90" x2="520" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="550" y1="90" x2="550" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="580" y1="90" x2="580" y2="95" stroke="#15803d" stroke-width="1.2"/><line x1="370" y1="120" x2="480" y2="120" stroke="#15803d" stroke-width="1.4"/><line x1="370" y1="120" x2="370" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="386" y1="120" x2="386" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="402" y1="120" x2="402" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="418" y1="120" x2="418" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="434" y1="120" x2="434" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="450" y1="120" x2="450" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="466" y1="120" x2="466" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="480" y1="120" x2="480" y2="125" stroke="#15803d" stroke-width="1.2"/><line x1="370" y1="150" x2="396" y2="150" stroke="#15803d" stroke-width="1.4"/><line x1="370" y1="150" x2="370" y2="155" stroke="#15803d" stroke-width="1.2"/><line x1="374" y1="150" x2="374" y2="155" stroke="#15803d" stroke-width="1.2"/><line x1="378" y1="150" x2="378" y2="155" stroke="#15803d" stroke-width="1.2"/><line x1="382" y1="150" x2="382" y2="155" stroke="#15803d" stroke-width="1.2"/><line x1="386" y1="150" x2="386" y2="155" stroke="#15803d" stroke-width="1.2"/><line x1="390" y1="150" x2="390" y2="155" stroke="#15803d" stroke-width="1.2"/><line x1="396" y1="150" x2="396" y2="155" stroke="#15803d" stroke-width="1.2"/><rect x="370" y="200" width="210" height="30" rx="5" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><text x="475" y="220" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">各行を最適に量子化</text><text x="330" y="262" text-anchor="middle" font-size="11.5" fill="#52525b">各バー＝重み行列の 1 行（出力チャネル）の値レンジ</text></svg><figcaption>重み量子化の <b>scale の粒度</b>です。<b>per-tensor</b> はテンソル全体で <code>scale</code> を1つだけ持つため、最大振幅の行に合わせた粗い刻みになり、<b>小振幅の行（チャネル）が数段に潰れて</b>誤差が増えます。<b>per-channel</b> は<b>行（出力チャネル）ごとに <code>scale</code></b> を持つので各行が 256 段をフルに使え、誤差が小さくなります。実用の重み量子化は per-channel が既定です。</figcaption></figure>

PTQ と QAT は、**いつ何を測って量子化するか**によって分かれます。**動的PTQ(dynamic)**は、重みだけを事前に int8 化し、活性は推論のたびにその場で範囲を測って量子化します。キャリブレーション不要で1行で済むため CPU で最も手軽ですが、対象は既定で Linear/LSTM なので **Conv 主体には効きません**。これに対し**静的PTQ(static)**は、代表データを流して活性の範囲を**事前に観測(キャリブレーション)**し固定します。これにより Conv も int8 で実行でき、その代わりに `fuse_modules`（conv→bn→relu を1演算に畳む）と `QuantStub/DeQuantStub`（量子化の境界）が必要になります。さらに**QAT**は、学習中に **fake-quant**（前向きで丸めを模擬し、逆伝播はストレートスルー）を挟んで、量子化誤差込みで重みを微調整します。これは PTQ で精度が落ちすぎる時の上位互換であり、学習コストを払う代わりに精度を取り戻します。したがって意思決定は「動的で足りるか → ダメなら静的 → それでも精度が足りなければ QAT」という順序になります。

<figure class="lec-fig"><svg viewBox="0 0 660 252" role="img" aria-label="圧縮手法の意思決定順序。動的PTQで足りなければ静的PTQ、それでも精度不足ならQATへ一段ずつ進み、各段で精度OKならデプロイする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">手法選択の意思決定順序 — 動的 → 静的 → QAT</text><rect x="24" y="68" width="170" height="66" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="250" y="68" width="170" height="66" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="476" y="68" width="160" height="66" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="109" y="100" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">動的PTQ</text><text x="109" y="120" text-anchor="middle" font-size="10.5" fill="#52525b">Linear/LSTM・キャリブ不要</text><text x="335" y="100" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">静的PTQ</text><text x="335" y="120" text-anchor="middle" font-size="10.5" fill="#52525b">Conv も int8・要キャリブ</text><text x="556" y="100" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">QAT</text><text x="556" y="120" text-anchor="middle" font-size="10.5" fill="#52525b">fake-quant で微調整</text><line x1="200" y1="101" x2="244" y2="101" stroke="#dc2626" stroke-width="2"/><polygon points="250,101 240,96 240,106" fill="#dc2626"/><line x1="426" y1="101" x2="470" y2="101" stroke="#dc2626" stroke-width="2"/><polygon points="476,101 466,96 466,106" fill="#dc2626"/><text x="222" y="94" text-anchor="middle" font-size="10.5" fill="#dc2626">不足なら</text><text x="448" y="94" text-anchor="middle" font-size="10.5" fill="#dc2626">不足なら</text><rect x="24" y="190" width="170" height="48" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><rect x="250" y="190" width="170" height="48" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><rect x="476" y="190" width="160" height="48" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><text x="109" y="219" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">精度OK → デプロイ</text><text x="335" y="219" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">精度OK → デプロイ</text><text x="556" y="219" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">回復 → デプロイ</text><line x1="109" y1="138" x2="109" y2="184" stroke="#16a34a" stroke-width="2"/><polygon points="109,190 104,180 114,180" fill="#16a34a"/><line x1="335" y1="138" x2="335" y2="184" stroke="#16a34a" stroke-width="2"/><polygon points="335,190 330,180 340,180" fill="#16a34a"/><line x1="556" y1="138" x2="556" y2="184" stroke="#16a34a" stroke-width="2"/><polygon points="556,190 551,180 561,180" fill="#16a34a"/></svg><figcaption>圧縮手法の<b>意思決定順序</b>です。まず最も手軽な <b>動的PTQ</b>（Linear/LSTM・キャリブレーション不要）を試し、Conv 主体などで<b>足りなければ静的PTQ</b>（fuse＋キャリブで Conv も int8 化）へ、それでも<b>精度が足りなければ QAT</b>（fake-quant で微調整）へと一段ずつ進みます。横（赤）の矢印が「不足なら次の手法へ」、下（緑）の矢印が「精度が許容範囲ならそこでデプロイ」を表します。</figcaption></figure>

---

## 3. 正準 API — `torch.ao.quantization` と `torch.nn.utils.prune`

本講座で CPU 上で確実に動く正準 API は、**`torch.ao.quantization`（eager mode）** です。実行すると「将来 torchao へ移行」という `DeprecationWarning` が出ますが、torch 2.12 でも問題なく動き、CPU の int8 実行カーネル（fbgemm/x86/qnnpack）に直結します。新しい `torchao.quantization.quantize_` は現行の推奨 API ではあるものの**任意依存**であり、int4/float8 の高速パスは主に GPU/ARM 向けです。x86 CPU で確実に効くのは int8 系なので、本章では torch.ao を主軸に据え、torchao は `05` で概念だけ紹介します。

```python
import torch, torch.nn as nn
from torch.ao.quantization import (
    quantize_dynamic, fuse_modules, QuantStub, DeQuantStub,
    get_default_qconfig, prepare, convert,
)
torch.backends.quantized.engine = "x86"   # x86=fbgemm系 / Apple Silicon・ARM= "qnnpack"

# --- 動的PTQ: Linear/LSTM を int8 に（キャリブレーション不要・1 行） ---
model_int8 = quantize_dynamic(model.eval(), {nn.Linear, nn.LSTM}, dtype=torch.qint8)

# --- 静的PTQ: fuse → qconfig → prepare(observer) → キャリブ → convert ---
q = copy.deepcopy(model).eval()
fuse_modules(q, [["conv1", "bn1", "relu1"]], inplace=True)   # conv-bn-relu を畳む
q.qconfig = get_default_qconfig("x86")                       # per-channel 重み + 活性観測
prepare(q, inplace=True)                                     # observer を挿入
with torch.inference_mode():
    for xb in calib_loader:        # ★ラベル不要。推論を流して活性の min/max を観測
        q(xb)
convert(q, inplace=True)                                     # 統計から scale/zp を確定し int8 化
```

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="静的PTQの流れ。fuseでconv-bn-reluを畳み、prepareでobserver挿入、キャリブで活性を観測、convertでint8化" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">静的PTQ の流れ — fuse → 観測（キャリブ）→ int8 変換</text><text x="86" y="84" text-anchor="middle" font-size="12.5" fill="#1d4ed8">fp32 モデル</text><text x="572" y="84" text-anchor="middle" font-size="12.5" fill="#15803d">int8 モデル</text><rect x="16" y="100" width="140" height="78" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="178" y="100" width="140" height="78" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="340" y="100" width="140" height="78" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="502" y="100" width="140" height="78" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><text x="86" y="134" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">① fuse</text><text x="86" y="158" text-anchor="middle" font-size="10.5" fill="#52525b">conv+bn+relu を融合</text><text x="248" y="134" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">② prepare</text><text x="248" y="158" text-anchor="middle" font-size="10.5" fill="#52525b">qconfig＋observer 挿入</text><text x="410" y="134" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">③ calibrate</text><text x="410" y="158" text-anchor="middle" font-size="10.5" fill="#c2410c">代表データを流し観測</text><text x="572" y="134" text-anchor="middle" font-size="16" font-weight="700" fill="#15803d">④ convert</text><text x="572" y="158" text-anchor="middle" font-size="10.5" fill="#52525b">scale/zp 確定 → int8</text><polygon points="160,131 174,139 160,147" fill="#ea580c"/><polygon points="322,131 336,139 322,147" fill="#ea580c"/><polygon points="484,131 498,139 484,147" fill="#ea580c"/><text x="410" y="198" text-anchor="middle" font-size="11.5" fill="#c2410c">★ ラベル不要</text></svg><figcaption><b>静的PTQ</b> は活性の範囲を事前に固定するため4手順を踏みます。① <code>fuse_modules</code> で conv→bn→relu を1演算に畳み、② <code>prepare</code> で <code>qconfig</code> と <b>observer</b> を挿入、③ <b>キャリブレーション</b>で代表データを流して活性の min/max を観測し（<b>ラベル不要</b>）、④ <code>convert</code> で <code>scale</code>・<code>zero_point</code> を確定して int8 化します。これにより <b>Conv も int8 実行</b>できます。</figcaption></figure>

QAT は `prepare_qat` 系を使い、**fuse を QAT 用の `fuse_modules_qat`**（train モード対応）で行う点だけが異なります（`fuse_modules` を train モードで呼ぶと `Fusion only for eval!` で落ちるのが定番の罠です）。流れは `q.qconfig = get_default_qat_qconfig("x86")` → `prepare_qat(q)` → 数エポック微調整 → `q.eval(); convert(q)` です。

枝刈りには **`torch.nn.utils.prune`** を使います。非構造化は `prune.l1_unstructured(module, "weight", amount)`（小さい重みから個別に 0 にする）、全層横断は `prune.global_unstructured([...], prune.L1Unstructured, amount)`、構造化は `prune.ln_structured(module, "weight", amount, n=2, dim=0)`（行=出力チャネル単位で落とす）で行い、`prune.remove(module, "weight")` でマスクを焼いて再パラメータ化を外します。ただし**重要なのは、これらがどれも密テンソルのままだ**という点です。`module.weight_mask` を見ればマスクが別バッファとして増えているのが分かり、`prune.remove` する前の `state_dict` は `weight_orig + weight_mask` を両方抱えるため**むしろ大きくなります**。

---

## 4. 実装を1つずつ — スクリプトで段階的に確かめる

各スクリプトは独立に動き、モデルは合成データで**その場で軽く学習**（CPU で 1〜2 秒・決定的）します。共通部品は `quant_lab.py` にまとめてあり（合成タスク・`TinyMLP`/`TinyConvNet`・学習・三角評価の道具・量子化/枝刈りヘルパ）、結果は `lectures/35_quantization_pruning/outputs/` に保存されます。

- **`01_quant_theory.py` — 量子化の正体**。`scale`/`zero_point` を手計算し、対称 vs 非対称の MAE 差、ビット幅(8/4/2)と誤差、per-tensor vs per-channel（チャネル振幅が桁違いの重みで per-channel が約3倍改善）を数字と図で確認。図 `01_quantization_levels.png`。
- **`02_dynamic_ptq.py` — 動的PTQ**。Linear 主体の `TinyMLP` で**サイズ約3.9倍縮小・精度維持**を確認。さらに Conv 主体の `TinyConvNet` では**約1.02倍しか縮まない**（Linear ヘッドだけ int8、Conv は fp32 のまま）ことを示し、「動的は CNN に効かない」を体感。バッチ1では int8 が fp32 より遅い**速度の逆転**も観察。図 `02_dynamic_quant.png`。
- **`03_static_ptq_qat.py` — 静的PTQ と QAT**。同じ `TinyConvNet` に fuse + キャリブレーション + convert を適用し、**Conv も int8 化して約2.8倍縮小、bs=128 で大幅に高速化**。QAT も併走させ、4方式（fp32 / dynamic / static / QAT）を一表で比較。図 `03_static_vs_qat.png`。
- **`04_pruning_speed_trap.py` — 実効速度の罠（本章の核心）**。非構造化グローバル L1 枝刈りでスパース率を 0→0.9 まで上げても、**密の保存サイズも p50 レイテンシも一定**。CSR(疎)に変換した保存サイズだけは高スパースで縮むが、それは「保存」の話で実行は速くならない。`prune.remove` 前の `state_dict` は2倍に増える。最後に**構造化枝刈り+リビルド**で size も p50 も本当に縮むことを対比。図 `04_pruning_speed_trap.png`。
- **`05_torchao_quantize.py` — 現行の正準 API（任意依存）**。`torch.ao` が非推奨で torchao の `quantize_` が後継であることを紹介。torchao があれば int8 dynamic を実演し、無ければ概念紹介で**正常終了(exit 0)**。重い/任意依存を実行経路に必須化しない設計。

```python
# quant_lab.py の部品で「同じ題材」を量子化/枝刈りする
from quant_lab import (get_trained_mlp, get_trained_cnn, dynamic_quantize,
                       static_quantize, qat_quantize, structured_compress_mlp,
                       accuracy, state_dict_size_mb, benchmark)

mlp = get_trained_mlp()
mlp_q = dynamic_quantize(mlp)                 # Linear→int8（手軽）
print(state_dict_size_mb(mlp), "->", state_dict_size_mb(mlp_q))   # 約 1/4

cnn = get_trained_cnn()
cnn_q = static_quantize(cnn, calib_x)         # Conv も int8（fuse+キャリブ+convert）
small = structured_compress_mlp(mlp, keep_frac=0.5)   # 構造化→小さい密モデル（本当に縮む）
```

---

## 🛠 章末ミニプロジェクト — 圧縮手法ベンチ（三角関係で選ぶ）

`mini_project.py` が完成形です。ここでは `CompressionBench` クラスが、**1つのベースモデルに複数の圧縮手法を適用し、accuracy / size(MB) / latency(p50,p99) を同一指標で表にする**ベンチを提供します。

1. Linear 主体(`TinyMLP`) と Conv 主体(`TinyConvNet`) を学習。
2. 適材適所で圧縮を適用:
   - MLP → 動的量子化 / 非構造化90%プルーニング(**罠: 縮まない**) / 構造化リビルド(**本当に縮む**)
   - CNN → 静的PTQ / QAT（torchao があれば int8 dynamic も自動追加）
3. 三指標を1つの表 + 図 + JSON にまとめる。
4. 速度・サイズ・精度のトレードオフから**意思決定順序**を提示する。

<figure class="lec-fig"><svg viewBox="0 0 680 286" role="img" aria-label="ミニプロジェクトの流れ。2モデルを学習し適材適所で圧縮を適用、三指標で比較して意思決定順序を提示。圧縮はMLPに動的量子化・非構造化90%・構造化、CNNに静的PTQ・QAT" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="26" text-anchor="middle" font-size="14.5" font-weight="700" fill="#18181b">ミニプロジェクト — 学習 → 圧縮 → 比較 → 意思決定</text><rect x="16" y="50" width="148" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="190" y="50" width="130" height="56" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="360" y="50" width="150" height="56" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="548" y="50" width="120" height="56" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="90" y="76" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">① 2モデルを学習</text><text x="90" y="95" text-anchor="middle" font-size="10" fill="#52525b">TinyMLP / TinyConvNet</text><text x="255" y="76" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">② 圧縮を適用</text><text x="255" y="95" text-anchor="middle" font-size="10" fill="#52525b">適材適所</text><text x="435" y="76" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">③ 三指標で比較</text><text x="435" y="95" text-anchor="middle" font-size="10" fill="#52525b">acc・size・p50/p99</text><text x="608" y="76" text-anchor="middle" font-size="13.5" font-weight="700" fill="#15803d">④ 意思決定</text><text x="608" y="95" text-anchor="middle" font-size="10" fill="#52525b">順序を提示</text><line x1="168" y1="78" x2="184" y2="78" stroke="#71717a" stroke-width="2"/><polygon points="190,78 180,73 180,83" fill="#71717a"/><line x1="326" y1="78" x2="354" y2="78" stroke="#71717a" stroke-width="2"/><polygon points="360,78 350,73 350,83" fill="#71717a"/><line x1="516" y1="78" x2="542" y2="78" stroke="#71717a" stroke-width="2"/><polygon points="548,78 538,73 538,83" fill="#71717a"/><line x1="255" y1="110" x2="255" y2="144" stroke="#ea580c" stroke-width="2"/><polygon points="255,150 250,140 260,140" fill="#ea580c"/><rect x="60" y="150" width="560" height="120" rx="10" fill="#fff7ed" stroke="#f97316" stroke-width="1.8"/><text x="80" y="172" font-size="11.5" font-weight="700" fill="#c2410c">② で適用する圧縮手法</text><text x="92" y="205" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">MLP</text><rect x="140" y="188" width="112" height="28" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.6"/><text x="196" y="206" text-anchor="middle" font-size="11" fill="#c2410c">動的量子化</text><rect x="264" y="188" width="146" height="28" rx="6" fill="#fff7ed" stroke="#dc2626" stroke-width="1.6"/><text x="337" y="206" text-anchor="middle" font-size="11" fill="#dc2626">非構造化90%（罠）</text><rect x="422" y="188" width="134" height="28" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="1.6"/><text x="489" y="206" text-anchor="middle" font-size="11" fill="#15803d">構造化リビルド</text><text x="92" y="251" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">CNN</text><rect x="140" y="234" width="104" height="28" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.6"/><text x="192" y="252" text-anchor="middle" font-size="11" fill="#c2410c">静的PTQ</text><rect x="256" y="234" width="78" height="28" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.6"/><text x="295" y="252" text-anchor="middle" font-size="11" fill="#c2410c">QAT</text><rect x="346" y="234" width="210" height="28" rx="6" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.4"/><text x="451" y="252" text-anchor="middle" font-size="10.5" fill="#71717a">+ torchao int8（任意）</text></svg><figcaption><b>章末ミニプロジェクト</b>（<code>CompressionBench</code>）の流れです。<b>① 2モデルを学習</b>（Linear 主体の <code>TinyMLP</code> と Conv 主体の <code>TinyConvNet</code>）→ <b>② 適材適所で圧縮を適用</b>（MLP に動的量子化／非構造化90%＝<b>縮まない罠</b>／構造化リビルド、CNN に静的PTQ／QAT）→ <b>③ 精度・サイズ(MB)・レイテンシ(p50/p99) の三指標で比較</b> → <b>④ トレードオフから手法選択の意思決定順序を提示</b>、と一気通貫で流れます。</figcaption></figure>

```bash
uv run python lectures/35_quantization_pruning/mini_project.py
```

出力される表はこんな形（数値は環境で変わります。`x` はサイズ縮小率）:

```
=== TinyMLP (Linear 主体) ===
    method                    acc  size(MB)     x  p50(ms)  p99(ms)  spars
    fp32 baseline           1.000    1.0605  1.0x    0.175    0.217   0.00
    dynamic int8            1.000    0.2715  3.9x    0.171    0.192   0.00
    unstructured 90%        0.938    1.0606  1.0x    0.171    0.218   0.90   ← 罠: 90%スパースでも縮まない
    structured keep=0.5     0.992    0.4005  2.6x    0.094    0.107   0.00   ← 本当に縮む & 速い
=== TinyConvNet (Conv 主体) ===
    static PTQ int8         1.000    0.0246  2.8x    0.293    ...           ← Conv も int8 化
```

**腕試し（発展課題）**: ①`structured_compress_mlp` の `keep_frac` を 0.1 まで下げ、精度がどこで崩れるか境界を探る。②`04` の CSR サイズが密を下回る「損益分岐スパース率」を求める（ヒント: 0% では CSR の方が大きい）。③`get_default_qconfig` を per-tensor に変えて精度劣化が増えるか比較する。④静的PTQの**キャリブレーション枚数**を 8→256 と変え、精度がどう安定するか観察する。⑤`mini_compression_bench.json` を読み、size×p50 の散布図で「パレート最適」な手法を可視化する。

---

## ✅ 到達チェックリスト

- [ ] 量子化を `q = round(x/scale)+zero_point` / `x ≈ scale*(q-zero_point)` で説明でき、**対称(重み)** と **非対称(活性)** を使い分けられる。
- [ ] **per-channel** がチャネル間スケール差を吸収して per-tensor より誤差が小さい理由を言える。
- [ ] **動的PTQ** は Linear/LSTM 向き・キャリブ不要、**Conv には効かない**ことを実測で示せる。
- [ ] **静的PTQ** に必要な部品（`fuse_modules` / `QuantStub`・`DeQuantStub` / `prepare`→キャリブ→`convert`）を並べられる。
- [ ] **QAT** が fake-quant で量子化誤差を学習に取り込む手法で、PTQ で精度が落ちる時の一手だと説明できる。
- [ ] **非構造化プルーニングは密のままで実速度・サイズが縮まない**（`prune.remove` も焼くだけ）ことを数字で示せる。
- [ ] **構造化プルーニング+リビルド**なら実際に小さい密モデルになり、size も latency も縮むと説明できる。
- [ ] 圧縮を **精度 / レイテンシ(p50,p99) / サイズ(MB)** の三角で評価し、手法選択の順序を持っている。
- [ ] `torch.ao` が非推奨で **torchao `quantize_`** が後継だが、CPU では int8 系 / ONNX 動的量子化が確実、と知っている。

---

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/35_quantization_pruning/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 非対称(affine) int8（uint8・qmin=0/qmax=255）の `scale` と `zero_point` を計算して返す。x_max==x_min の退化時は scale=1.0・zero_point=0（`ex1_affine_scale_zero_point`）。
2. テンソルを uint8(0..255) に量子化してから逆量子化した復元テンソル x̂ を返す（`ex2_quantize_dequantize`）。
3. 対称(symmetric) num_bits 量子化で往復させ、平均絶対誤差(MAE) を float で返す（`ex3_roundtrip_mae`）。
4. 重み行列 w(out,in) の per-channel（行ごと）対称 int8 `scale` を形 (out,) で返す（max(|w[i]|)==0 の行は 1.0）（`ex4_per_channel_scales`）。
5. 重みテンソルのスパース率（値が 0 の要素の割合, 0〜1）を返す（`ex5_weight_sparsity`）。
6. |w| が小さい順に amount 割合を 0 にした同じ形のテンソルを返す＝非構造化(magnitude)枝刈りの中身（`ex6_magnitude_prune`）。
7. num_params 個の重みを fp32→int8 にしたときに節約できるバイト数（1 要素あたり 3 byte）を返す（`ex7_int8_bytes_saved`）。
8. 構造化枝刈りで残す出力ニューロン（行）の index を、行ごとの L2 ノルム上位から選んで昇順で返す（`ex8_structured_keep_rows`）。
9. model_kind・キャリブデータの有無・精度劣化の許容可否から量子化手法（dynamic / static / qat）を選んで返す（`ex9_recommend_method`）。

---

## ❓ 落とし穴・FAQ・デバッグ

- **「90% 枝刈りしたのに `.pt` が大きくなった」**: `prune.remove` する前の `state_dict` は `weight_orig`(元の重み) + `weight_mask`(マスク) を両方持つので**約2倍**になります。`prune.remove` でマスクを焼いてから保存する。ただし焼いても密テンソル（0 が増えるだけ）でサイズは元と同じ。実圧縮は構造化+リビルドか量子化。
- **「枝刈りしたのに速くならない」**: 非構造化スパースは密 GEMM のまま。CPU には実用的な疎カーネルが（ほぼ）無く、`to_sparse_csr()` にしても**保存は縮むが実行は速くなりません**。速度が欲しいなら構造化プルーニングで密モデルを小さくする。
- **`Fusion only for eval!`**: `fuse_modules` を train モードで呼ぶと出ます。推論用 fuse は `model.eval()` してから。**QAT では `fuse_modules_qat`** を使う（train モード対応）。
- **動的量子化で CNN が縮まない**: 動的量子化の対象は既定で Linear/LSTM。Conv 主体モデルでは Conv が fp32 のまま残るので効きません。Conv を量子化するなら**静的PTQ**（要キャリブレーション）。
- **`NoQEngine`/`quantized engine` 系のエラーや精度劣化**: バックエンド未設定が原因。x86 は `torch.backends.quantized.engine="x86"`（または `"fbgemm"`）、Apple Silicon/ARM は `"qnnpack"` を**明示**する。`torch.backends.quantized.supported_engines` で確認。
- **静的PTQの精度が悪い**: キャリブレーションデータが本番分布と乖離している/枚数が少なすぎる、`fuse_modules` を忘れて BN が悪さをしている、per-tensor qconfig になっている、などが定番。まず fuse と per-channel(`get_default_qconfig` 既定)を確認し、キャリブ枚数を増やす。それでもダメなら QAT。
- **`DeprecationWarning: torch.ao.quantization is deprecated`**: torch 2.12 では出るが**動作はする**。新規は torchao `quantize_` 推奨ですが、x86 CPU の int8 実行は torch.ao が確実。本章は学習目的で torch.ao を正準にし、警告は `quiet_warnings()` で抑制しています。
- **量子化済みモデルを ONNX エクスポートしたい**: 量子化済み/特殊モジュールは ONNX 化で失敗しがち。**ONNX 化は fp32 モデルから行い、量子化は onnxruntime 側（次回36 `quantize_dynamic`）でやる**のが安全。
- **bs=1 で int8 が fp32 より遅い**: 小モデル/小バッチでは量子化・逆量子化のオーバヘッドが計算削減を上回ることがあります（`02` で実測）。速度評価は**実運用のバッチサイズ**で、ウォームアップ込みの p50/p99 で測る（第34回の作法）。

---

## 🚀 発展トピック・参考

- **torchao（現行の正準・任意 `quant` グループ）**: `from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig`。`quantize_(model, Config())` で in-place 変換し、`torch.compile` 併用でカーネル融合。int4/float8 は主に GPU/ARM 向け。導入: `uv add --group quant torchao`。
- **PT2E 量子化（静的PTQ/QAT の現行フロー）**: `torch.export.export` → `prepare_pt2e` / `convert_pt2e` + `X86InductorQuantizer`。FX/eager の後継で、`torch.compile` と相性が良い。
- **ONNX Runtime 動的量子化（次回36）**: `onnxruntime.quantization.quantize_dynamic(QuantType.QUInt8)` は **CPU 推論で最も費用対効果が高い int8 化**。本章で「CPU では効きにくい」と学んだ量子化が、ONNX ランタイムでは素直に効くことが多い。
- **構造化プルーニングの自動化**: `torch.nn.utils.prune.ln_structured` でチャネルを落とした後、依存するレイヤ（次の Conv の入力チャネル等）も整合させて密モデルを作り直す必要がある。実務では `torch-pruning` 等のライブラリが依存グラフを追って自動でリビルドしてくれる。
- **bitsandbytes（LLM/VLM の 8bit/4bit）**: `transformers` の `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`。歴史的に CUDA 前提で、CPU は α 対応・機能限定。CPU 講座では概念とコードの読み方を押さえ、実速度検証は torchao/ONNX に寄せるのが無難。
- **公式ドキュメント**: [PyTorch Quantization](https://docs.pytorch.org/docs/stable/quantization.html) / [torch.nn.utils.prune](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.html) / [torchao](https://docs.pytorch.org/ao/stable/) / [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。

---

## ▶ 動かし方

```bash
# 依存（未導入なら）: 深層学習の土台（CPU 版 torch / torchvision）
uv sync --group dl
# torchao を試す場合のみ（任意・CPU では効果限定）: uv add --group quant torchao

# 本編（番号順）。モデルは合成データでその場で軽く学習（数秒・決定的）
uv run python lectures/35_quantization_pruning/01_quant_theory.py
uv run python lectures/35_quantization_pruning/02_dynamic_ptq.py
uv run python lectures/35_quantization_pruning/03_static_ptq_qat.py
uv run python lectures/35_quantization_pruning/04_pruning_speed_trap.py
uv run python lectures/35_quantization_pruning/05_torchao_quantize.py   # 任意依存・未導入でも exit 0

# 章末ミニプロジェクト（圧縮手法ベンチの完成形）
uv run python lectures/35_quantization_pruning/mini_project.py

# 演習（自己採点。未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/35_quantization_pruning/exercises.py
uv run python lectures/35_quantization_pruning/exercises_solutions.py
```

成果物（図・JSON）は `lectures/35_quantization_pruning/outputs/` に保存されます。CPU 前提・`model.eval()` + `torch.inference_mode()`・headless（`imshow` は呼ばず matplotlib=Agg で保存）。量子化バックエンドは x86 を既定にし、Apple Silicon/ARM では `qnnpack` に切り替えてください。

---

> 参照ライブラリ: **torch 2.12+cpu**（`torch.ao.quantization` / `torch.nn.utils.prune`）/ **onnx 1.21** / **onnxruntime 1.26**（量子化の出口は次回36）/ torchao は任意（現行の正準 API・CPU では効果限定）
> （題材: 合成6クラス分類・`TinyMLP`(Linear主体)/`TinyConvNet`(Conv主体)、バックエンド x86/fbgemm/qnnpack、CPU・`model.eval()`+`torch.inference_mode()`、`torch.ao` は非推奨だが CPU int8 実行の正準として使用） — 2026-06
