# 38_knowledge_distillation: 知識蒸留の基礎(a) — 温度付きKD・特徴量蒸留(CPUトイ学習)

> トラック: **最適化・デプロイ** ／ レベル: **中級** ／ 必要な依存グループ: `dl`（`timm` は任意・`hf` グループ）

このモジュールが狙うのは、「大きな teacher の知識を、小さな student へ移す」**知識蒸留(Knowledge Distillation, KD)** を、合成図形データの**CPU トイ学習**として最初から最後まで自分の手で書ける状態にすることだ。温度付きソフトターゲットの KL 損失から特徴量蒸留、さらに DeiT の distillation token までを、`uv run` ですぐ動くスクリプトで一つずつ確かめていく。

---

## 🎯 この章のゴール

teacher(凍結)→student へ知識を移す蒸留の枠組みを理解した上で、二つの代表的な手法を CPU の小モデルで**学習ループを自分で書いて**完結できるようになることを目指す。一つは、温度 T でソフトターゲットの KL に `T^2` スケールを掛け、ハードラベル CE と alpha 結合する Hinton 蒸留。もう一つは、forward hook で中間特徴を捕まえ MSE で合わせる特徴量蒸留(FitNets、射影層必須)だ。さらに、同じ小 student を「素の教師あり学習」と「蒸留学習」の両方で訓練し、テスト accuracy の差として**暗黙知(dark knowledge)の移転**を定量化できるようにする。

到達点はシンプルだ。すなわち、`response_kd_loss = alpha * KD(soft) + (1-alpha) * CE(hard)` を**何も見ずに正しく書ける**こと、teacher を `eval() + requires_grad=False` で凍結する理由を説明できること、そして「蒸留ありが素の student を上回る」ことを自分のベンチで示せること、の三点である。

---

## 1. 直感 — 「正解は1つ」では捨ててしまう情報がある

ハードラベル（one-hot）は「この画像は猫だ」としか教えてくれない。ところが、よく訓練された teacher の softmax 出力を覗くと、「猫 90%・犬 8%・車 1.5%・飛行機 0.5%」のように、**猫は犬に似ていて車には似ていない**という**クラス間の類似構造**まで含まれている。Hinton はこれを *dark knowledge（暗黙知）* と呼んだ。つまり蒸留とは、この「2 番手以降の確率」を student に伝え、正解ラベル 1 ビットよりもはるかに豊かな信号で学ばせる営みなのだ。

<figure class="lec-fig"><svg viewBox="0 0 640 262" role="img" aria-label="ハードラベルは猫だけ1で他は0、teacherのsoft targetは猫90%犬8%車1.5%飛0.5%でクラス間の類似構造を残す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="24" y="46" width="288" height="202" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="328" y="46" width="288" height="202" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="168" y="70" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">ハードラベル (one-hot)</text><text x="472" y="70" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">teacher の soft target</text><line x1="56" y1="214" x2="296" y2="214" stroke="#71717a" stroke-width="1.5"/><line x1="360" y1="214" x2="600" y2="214" stroke="#71717a" stroke-width="1.5"/><rect x="70" y="94" width="34" height="120" fill="#ea580c"/><rect x="122" y="211" width="34" height="3" fill="#d4d4d8"/><rect x="174" y="211" width="34" height="3" fill="#d4d4d8"/><rect x="226" y="211" width="34" height="3" fill="#d4d4d8"/><text x="87" y="88" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">1.0</text><text x="87" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">猫</text><text x="139" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">犬</text><text x="191" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">車</text><text x="243" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">飛</text><rect x="374" y="106" width="34" height="108" fill="#ea580c"/><rect x="426" y="204" width="34" height="10" fill="#f97316"/><rect x="478" y="211" width="34" height="3" fill="#f97316"/><rect x="530" y="212" width="34" height="2" fill="#f97316"/><text x="391" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">90%</text><text x="443" y="198" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ea580c">8%</text><text x="495" y="205" text-anchor="middle" font-size="11" fill="#52525b">1.5%</text><text x="547" y="205" text-anchor="middle" font-size="11" fill="#52525b">0.5%</text><text x="391" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">猫</text><text x="443" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">犬</text><text x="495" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">車</text><text x="547" y="232" text-anchor="middle" font-size="13" fill="#3f3f46">飛</text></svg><figcaption>正解ラベルは <b>猫だけ 1、残りは 0</b> の one-hot で、クラス間の関係を捨ててしまいます。一方 teacher の <b>soft target</b> は <code>猫90% / 犬8% / 車1.5% / 飛0.5%</code> のように<b>2 番手以降の確率</b>を残し、「猫は犬に似て車には似ていない」という<b>クラス間の類似構造（dark knowledge）</b>まで student に伝えます。</figcaption></figure>

では、なぜそれが効くのか。小さな student は容量が限られるため、少ないデータでハードラベルだけを追うと、ノイズや暗記に陥りやすい。これに対し teacher の soft target は、各サンプルに連続的な「正解らしさの分布」を与えるので、実質的に**1 サンプルあたりの情報量を増やし**、正則化としても働く。そこで本モジュールのトイでは「少量・ラベル 40% ノイズ」という意地悪な状況をあえて用意し、クリーンなフルデータで育った teacher の soft target が student をノイズ過学習から救う様子を観察する。

そもそも蒸留は、モデル圧縮の **4 本柱（蒸留 / 量子化 / プルーニング / 低ランク分解）** の 1 本目にあたる。量子化やプルーニング(35回)が「同じモデルを物理的に削る」のに対し、蒸留は「**小さいモデルを賢く学習させる**」アプローチであり、他の 3 本と直交するため組み合わせて使える（蒸留した小モデルをさらに量子化、など）。`01_kd_overview.py` では、この全体地図と teacher 凍結を確認する。

---

## 2. 理論 — 温度付きソフトターゲットと KL ダイバージェンス

蒸留の心臓部は **温度 T による軟化** と **KL ダイバージェンス損失** だ。ロジット `z` をそのまま softmax すると分布が尖り、暗黙知が埋もれてしまう。そこで `softmax(z / T)` のように温度 T で割ってから softmax する。こうすると T を上げるほど分布はなだらかになり、2 番手以降の確率が持ち上がって読み取りやすくなる（T=1 は素の softmax、T→∞ で一様分布）。なお蒸留では T=2〜5 がよく使われる。

<figure class="lec-fig"><svg viewBox="0 0 640 262" role="img" aria-label="同じロジットでもT=1では1位が83%と尖り、T=4では41/25/19/15%となだらかになり2番手以降が持ち上がる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="22" y="46" width="252" height="202" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="366" y="46" width="252" height="202" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="148" y="70" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">T = 1（鋭い）</text><text x="492" y="70" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">T = 4（軟化）</text><line x1="48" y1="216" x2="262" y2="216" stroke="#71717a" stroke-width="1.5"/><line x1="380" y1="216" x2="600" y2="216" stroke="#71717a" stroke-width="1.5"/><rect x="56" y="108" width="30" height="108" fill="#ea580c"/><rect x="104" y="202" width="30" height="14" fill="#f97316"/><rect x="152" y="211" width="30" height="5" fill="#f97316"/><rect x="200" y="214" width="30" height="2" fill="#f97316"/><text x="71" y="102" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">83%</text><text x="119" y="196" text-anchor="middle" font-size="11" fill="#52525b">11%</text><text x="167" y="205" text-anchor="middle" font-size="11" fill="#52525b">4%</text><rect x="392" y="163" width="30" height="53" fill="#ea580c"/><rect x="440" y="183" width="30" height="33" fill="#f97316"/><rect x="488" y="191" width="30" height="25" fill="#f97316"/><rect x="536" y="196" width="30" height="20" fill="#f97316"/><text x="407" y="157" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">41%</text><text x="455" y="177" text-anchor="middle" font-size="11" fill="#52525b">25%</text><text x="503" y="185" text-anchor="middle" font-size="11" fill="#52525b">19%</text><text x="551" y="190" text-anchor="middle" font-size="11" fill="#52525b">15%</text><line x1="288" y1="150" x2="352" y2="150" stroke="#c2410c" stroke-width="3"/><polygon points="362,150 350,144 350,156" fill="#c2410c"/><text x="320" y="138" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">T を上げる</text><text x="320" y="170" text-anchor="middle" font-size="11" fill="#52525b">軟化</text></svg><figcaption>同じロジットでも温度 <b>T</b> で割ってから softmax すると分布の鋭さが変わります。<b>T=1</b> は 1 位が <code>83%</code> と尖って暗黙知が埋もれますが、<b>T=4</b> では <code>41 / 25 / 19 / 15%</code> と<b>なだらか</b>になり、2 番手以降の確率（クラス間の類似）が読み取りやすくなります。<b>T→∞ で一様分布</b>に近づきます。</figcaption></figure>

軟化した student の出力分布を teacher の soft target に近づける尺度が、KL ダイバージェンスである。その正準形は次の通りで、**順序・log・reduction・T² の 4 点を厳守する**:

```python
# soft = KL( softmax(teacher/T) || softmax(student/T) )
soft = F.kl_div(
    F.log_softmax(student_logits / T, dim=1),   # 入力は log_softmax(student)
    F.softmax(teacher_logits / T, dim=1),       # ターゲットは softmax(teacher)
    reduction="batchmean",                       # 数式の 1/N（'mean' は要素数で割れて誤り）
) * (T ** 2)                                      # 勾配が 1/T^2 に縮むのを補正
```

`T²` を掛けるのは、勾配スケールを補正するためだ。`softmax(z/T)` を微分すると勾配が `1/T²` のオーダーで縮むため、T を変えるたびに soft 損失の実効的な重み（実効学習率）まで変わってしまう。そこで `T²` を掛けておけば、T によらず勾配の大きさがほぼ揃い、T をハイパラとして安心して動かせる。そして最終的な損失は **ソフトとハードの線形結合** `loss = alpha * soft + (1 - alpha) * F.cross_entropy(student_logits, y)` で表され、alpha=0 が素の教師あり学習、alpha=1 がソフトのみに対応する。これら軟化・KL の向き・T² 補正・合成損失を一つずつ手で確かめるのが `02_hinton_soft_targets.py` だ。

<figure class="lec-fig"><svg viewBox="0 0 680 300" role="img" aria-label="同じ入力を凍結teacherと学習中studentに通し、softmaxのsoft損失KLにTの二乗を掛けたものとCEのhard損失をalpha結合して総損失にする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="16" y="132" width="54" height="44" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="43" y="159" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">入力</text><rect x="96" y="48" width="128" height="48" rx="6" fill="#eff6ff" stroke="#1d4ed8" stroke-width="2"/><text x="160" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">teacher</text><text x="160" y="87" text-anchor="middle" font-size="11" fill="#52525b">凍結・eval</text><rect x="96" y="200" width="128" height="48" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="160" y="222" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">student</text><text x="160" y="239" text-anchor="middle" font-size="11" fill="#52525b">学習対象</text><rect x="396" y="64" width="152" height="58" rx="6" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="472" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">soft 損失</text><text x="472" y="108" text-anchor="middle" font-size="12.5" fill="#3f3f46">KL · T²</text><rect x="396" y="196" width="152" height="58" rx="6" fill="#ffffff" stroke="#ea580c" stroke-width="2"/><text x="472" y="220" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">hard 損失</text><text x="472" y="240" text-anchor="middle" font-size="12.5" fill="#3f3f46">CE(z, y)</text><rect x="250" y="256" width="86" height="32" rx="5" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="293" y="277" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">正解 y</text><rect x="582" y="116" width="84" height="72" rx="6" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><text x="624" y="142" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">総損失</text><text x="624" y="162" text-anchor="middle" font-size="11" fill="#15803d">α·soft</text><text x="624" y="178" text-anchor="middle" font-size="11" fill="#c2410c">+(1−α)hard</text><line x1="70" y1="140" x2="96" y2="102" stroke="#71717a" stroke-width="1.8"/><polygon points="99,98 97,109 89,103" fill="#71717a"/><line x1="70" y1="166" x2="96" y2="194" stroke="#71717a" stroke-width="1.8"/><polygon points="99,198 88,194 96,187" fill="#71717a"/><line x1="224" y1="72" x2="392" y2="89" stroke="#1d4ed8" stroke-width="1.8"/><polygon points="396,90 386,94 387,84" fill="#1d4ed8"/><text x="306" y="68" text-anchor="middle" font-size="11" fill="#1d4ed8">softmax(z/T)</text><line x1="224" y1="208" x2="392" y2="119" stroke="#c2410c" stroke-width="1.8"/><polygon points="396,118 389,127 385,118" fill="#c2410c"/><text x="300" y="150" text-anchor="middle" font-size="10.5" fill="#c2410c">log_softmax(z/T)</text><line x1="224" y1="234" x2="392" y2="224" stroke="#c2410c" stroke-width="1.8"/><polygon points="396,224 386,230 386,220" fill="#c2410c"/><line x1="312" y1="262" x2="394" y2="248" stroke="#dc2626" stroke-width="1.6"/><polygon points="398,246 388,250 389,240" fill="#dc2626"/><line x1="548" y1="94" x2="579" y2="134" stroke="#16a34a" stroke-width="1.8"/><polygon points="582,138 572,133 580,127" fill="#16a34a"/><line x1="548" y1="224" x2="579" y2="170" stroke="#ea580c" stroke-width="1.8"/><polygon points="582,166 581,177 573,172" fill="#ea580c"/></svg><figcaption>レスポンス蒸留のデータフローです。<b>同じ入力</b>を、<b>凍結した teacher</b>（<code>eval</code> ＋勾配なし）と<b>学習中の student</b>の両方に通します。teacher 側の <code>softmax(z/T)</code>（soft target）と student 側の <code>log_softmax(z/T)</code> の <b>KL に T² を掛けた soft 損失</b>、正解 <code>y</code> との <b>CE（hard 損失）</b>を、<code>α·soft + (1−α)·hard</code> で合成します。optimizer に渡すのは <b>student だけ</b>です。</figcaption></figure>

蒸留は、伝える「知識の場所」によって 3 系統に分かれる。すなわち、**レスポンスベース**（出力ロジットを真似る＝Hinton 蒸留）、**特徴量ベース**（中間特徴を真似る＝FitNets）、**関係ベース**（サンプル間の距離・角度の関係を真似る＝RKD）の三つだ。それぞれ、レスポンスは最も簡単で効果も安定し、特徴量は teacher の内部表現まで写せる代わりに次元合わせ（射影層）が要り、関係は特徴次元が違っても比較できる、という棲み分けになっている。

---

## 3. 正準 API — `F.kl_div` / `register_forward_hook` / 凍結

蒸留の実装で覚える正準 API は、驚くほど少ない。まず**ソフト損失**は `F.log_softmax` / `F.softmax` / `F.kl_div(..., reduction="batchmean")`、**ハード損失**は `F.cross_entropy` で組み立てる。次に**特徴量蒸留**では、中間特徴を取り出す `module.register_forward_hook(fn)`、次元を合わせる射影層 `nn.Conv2d(student_ch, teacher_ch, 1)`（または `nn.Linear`）、スケール差を消す `F.normalize`、そして `nn.MSELoss` / `F.mse_loss` を使う。最適化は一貫して `torch.optim.AdamW` でよい。

<figure class="lec-fig"><svg viewBox="0 0 660 268" role="img" aria-label="hookでteacher64chとstudent16chの中間特徴を取り、1x1Conv射影で16から64chに合わせL2正規化してからMSEを取る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="18" y="40" width="104" height="48" rx="6" fill="#eff6ff" stroke="#1d4ed8" stroke-width="2"/><text x="70" y="62" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">teacher</text><text x="70" y="79" text-anchor="middle" font-size="11" fill="#52525b">凍結</text><rect x="168" y="40" width="120" height="48" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="228" y="60" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">teacher 特徴</text><text x="228" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="#2563eb">64 ch</text><rect x="18" y="192" width="104" height="48" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="70" y="214" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">student</text><text x="70" y="231" text-anchor="middle" font-size="11" fill="#52525b">学習</text><rect x="168" y="192" width="120" height="48" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="228" y="212" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">student 特徴</text><text x="228" y="230" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ea580c">16 ch</text><rect x="330" y="192" width="128" height="48" rx="6" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><text x="394" y="212" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">1×1 Conv 射影</text><text x="394" y="230" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">16 → 64 ch</text><rect x="500" y="106" width="142" height="64" rx="6" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="571" y="132" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">特徴 MSE 損失</text><text x="571" y="152" text-anchor="middle" font-size="11.5" fill="#3f3f46">L2 正規化してから</text><line x1="122" y1="64" x2="162" y2="64" stroke="#71717a" stroke-width="1.8"/><polygon points="168,64 158,59 158,69" fill="#71717a"/><text x="145" y="56" text-anchor="middle" font-size="10" font-weight="700" fill="#16a34a">hook</text><line x1="122" y1="216" x2="162" y2="216" stroke="#71717a" stroke-width="1.8"/><polygon points="168,216 158,211 158,221" fill="#71717a"/><text x="145" y="208" text-anchor="middle" font-size="10" font-weight="700" fill="#16a34a">hook</text><line x1="288" y1="216" x2="324" y2="216" stroke="#71717a" stroke-width="1.8"/><polygon points="330,216 320,211 320,221" fill="#71717a"/><line x1="288" y1="64" x2="496" y2="123" stroke="#2563eb" stroke-width="1.8"/><polygon points="500,124 489,126 492,117" fill="#2563eb"/><line x1="458" y1="208" x2="497" y2="152" stroke="#c2410c" stroke-width="1.8"/><polygon points="500,150 498,161 490,155" fill="#c2410c"/></svg><figcaption>特徴量蒸留（FitNets）の流れです。<code>register_forward_hook</code> で teacher と student の<b>中間特徴</b>を捕まえますが、チャネル数が <b>64 ch と 16 ch</b> で食い違います。そこで <b>1×1 Conv の射影層</b>で student 側を <code>16 → 64 ch</code> に合わせ、<b><code>F.normalize</code> でスケール差を消してから MSE</b> を取ります。射影や正規化を省くと次元が合わず、スケールの大きいチャネルに損失が支配されてしまいます。</figcaption></figure>

```python
# --- レスポンス蒸留（合成損失） ---
def response_kd_loss(s_logits, t_logits, y, T, alpha):
    log_p = F.log_softmax(s_logits / T, dim=1)
    q     = F.softmax(t_logits / T, dim=1)
    soft  = F.kl_div(log_p, q, reduction="batchmean") * (T ** 2)
    hard  = F.cross_entropy(s_logits, y)
    return alpha * soft + (1 - alpha) * hard

# --- 特徴量蒸留（forward hook で中間特徴を捕まえる） ---
feats = {}
h = teacher.features.register_forward_hook(lambda m, i, o: feats.__setitem__("t", o))
# ... teacher(x); student(x) の後 ...
proj = projector(feats["s"])                       # 1x1 Conv で student を teacher の次元へ
loss_feat = F.mse_loss(F.normalize(proj.flatten(1), dim=1),
                       F.normalize(feats["t"].flatten(1), dim=1))
h.remove()                                          # 使い終わったらフックを外す
```

一方、**teacher の凍結**は API というより規律の問題だ。具体的には、`teacher.eval()`（BN/dropout を固定）に加えて `for p in teacher.parameters(): p.requires_grad_(False)` を実行し、推論は `with torch.inference_mode()` で行い、`optimizer` には **student のパラメータだけ** を渡す。これを怠ると擬似ラベルが毎バッチ揺れ、最悪の場合 optimizer が teacher まで更新してしまう。そこで本講座では、こうした規律を `kd_lab.freeze_module()` に閉じ込めている。

---

## 4. 実装を1つずつ — スクリプトで段階的に確かめる

番号順に動かしていくと、概念→損失→学習→特徴→DeiT と段階的に積み上がる。いずれも合成図形データ（cv2 で描く円・四角・三角・楕円・十字・線分の 6 クラス）を題材とし、teacher の学習まで含めて CPU で数秒〜十数秒で完結する。なお、データ生成・モデル・損失・hook・評価といった共通部品は `kd_lab.py` にまとめてある。

<figure class="lec-fig"><svg viewBox="0 0 680 250" role="img" aria-label="番号順スクリプトの学習プログレッション。01概念02損失03学習04特徴05DeiTと左から右へ段階的に積み上がり03が評価の核" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">スクリプトを番号順に — 概念から DeiT へ積み上げる</text><rect x="8" y="58" width="120" height="150" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="68" y="98" text-anchor="middle" font-size="20" font-weight="700" fill="#ea580c">①</text><text x="68" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">概念</text><text x="68" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">全体地図・凍結</text><text x="68" y="176" text-anchor="middle" font-size="10" fill="#52525b">量子化と直交</text><rect x="144" y="58" width="120" height="150" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="204" y="98" text-anchor="middle" font-size="20" font-weight="700" fill="#ea580c">②</text><text x="204" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">損失</text><text x="204" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">温度・KL・T²</text><text x="204" y="176" text-anchor="middle" font-size="10" fill="#52525b">KL の向き</text><rect x="280" y="58" width="120" height="150" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><text x="340" y="98" text-anchor="middle" font-size="20" font-weight="700" fill="#c2410c">③</text><text x="340" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">学習</text><text x="340" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">素 vs 蒸留</text><text x="340" y="176" text-anchor="middle" font-size="10" fill="#52525b">T・α スイープ</text><rect x="416" y="58" width="120" height="150" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="476" y="98" text-anchor="middle" font-size="20" font-weight="700" fill="#ea580c">④</text><text x="476" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">特徴</text><text x="476" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">FitNets 特徴</text><text x="476" y="176" text-anchor="middle" font-size="10" fill="#52525b">hook+射影+MSE</text><rect x="552" y="58" width="120" height="150" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="612" y="98" text-anchor="middle" font-size="20" font-weight="700" fill="#ea580c">⑤</text><text x="612" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">DeiT</text><text x="612" y="150" text-anchor="middle" font-size="10" fill="#52525b">distill token</text><text x="612" y="176" text-anchor="middle" font-size="10" fill="#52525b">2 ヘッド平均</text><line x1="128" y1="130" x2="138" y2="130" stroke="#71717a" stroke-width="2"/><polygon points="144,130 134,125 134,135" fill="#71717a"/><line x1="264" y1="130" x2="274" y2="130" stroke="#71717a" stroke-width="2"/><polygon points="280,130 270,125 270,135" fill="#71717a"/><line x1="400" y1="130" x2="410" y2="130" stroke="#71717a" stroke-width="2"/><polygon points="416,130 406,125 406,135" fill="#71717a"/><line x1="536" y1="130" x2="546" y2="130" stroke="#71717a" stroke-width="2"/><polygon points="552,130 542,125 542,135" fill="#71717a"/></svg><figcaption>番号順のスクリプトは <b>概念 → 損失 → 学習 → 特徴 → DeiT</b> と段階的に積み上がります。<b>①</b> <code>01_kd_overview</code>（圧縮 4 本柱・teacher 凍結）、<b>②</b> <code>02_hinton_soft_targets</code>（温度・KL の向き・T²）、<b>③</b> <code>03_response_kd_train</code>（素 vs 蒸留・T/α スイープ＝<b>本章の評価の核</b>）、<b>④</b> <code>04_feature_distill</code>（FitNets の hook＋射影＋MSE）、<b>⑤</b> <code>05_distill_token_deit</code>（DeiT distillation token）。前の回の理解が次に乗るので、番号順に動かすのが近道です。</figcaption></figure>

```bash
uv run python lectures/38_knowledge_distillation/01_kd_overview.py        # 4本柱・teacher凍結・圧縮率
uv run python lectures/38_knowledge_distillation/02_hinton_soft_targets.py # 温度・KLの向き・T^2補正
uv run python lectures/38_knowledge_distillation/03_response_kd_train.py   # 素 vs 蒸留 + T/alpha スイープ
uv run python lectures/38_knowledge_distillation/04_feature_distill.py     # FitNets(hook+射影+MSE)+RKD概観
uv run python lectures/38_knowledge_distillation/05_distill_token_deit.py  # DeiT distillation token
```

なかでも `03` は本章の評価の核だ。**同じ StudentCNN** を (A) 素の教師あり学習 と (B) レスポンス蒸留 で訓練し、テスト accuracy を比べる。少量(132枚)・ラベル 40% ノイズという設定では、素の student がノイズに過学習して精度を落とす一方、teacher の soft target を混ぜた student はこれを明確に上回る（このトイでは概ね 0.42 → 0.73 程度）。さらに T と alpha をスイープすることで、これらが感度の高いハイパラであること、そして alpha=0（素の CE）が最低で「少しでもソフトを混ぜると改善する」ことを観察できる。

続く `04` では、中間特徴まで合わせる FitNets を足す。`register_forward_hook` で teacher(64ch) と student(16ch) の特徴を捕まえ、1x1 Conv の射影層で次元を合わせ、正規化してから MSE を取る、という流れだ。出力だけのレスポンス蒸留にこれを足すと、さらに精度が伸びやすい。最後の `05` は DeiT を扱う。class token に加えて **distillation token** を持たせ、teacher の argmax を hard distillation で学ぶ専用ヘッドを足し、推論時に 2 ヘッドを平均する——この仕組みを、timm の `deit_tiny_distilled_*`（構造のみ確認・重み DL なし）と、CPU で回る自前の 2 ヘッド student の両方で再現する。

### 実務の使い分け

まずは **レスポンス蒸留** から始めるとよい（最も簡単で安定）。出力ロジットと alpha・T だけで効くので、既存の学習ループに数行足すだけで済む。それで精度が足りなければ、**特徴量蒸留** を重ねる（teacher の内部表現まで写せる反面、射影層と「どの層をフックするか」の設計が要る）。さらに、teacher と student のアーキが大きく違う（CNN→ViT など）場合は、**DeiT 型の distillation token** や **関係ベース(RKD)** が候補になる。一方、teacher が無い／作れない場合には、self-distillation（自分自身や深い層から浅い層へ）も選択肢となる。いずれの場合も、「teacher は強く、凍結して使う」「T/alpha はスイープする」が共通の鉄則だ。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="実務の使い分け。まずレスポンス蒸留を試し精度が足りなければ特徴量蒸留を重ねアーキ差が大きいならDeiT tokenやRKDへ。共通の鉄則はteacher凍結とTalphaスイープ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">実務の使い分け — 簡単な手法から段階的に強化</text><rect x="16" y="108" width="174" height="92" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="103" y="138" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">① レスポンス蒸留</text><text x="103" y="162" text-anchor="middle" font-size="11" fill="#52525b">最も簡単・安定</text><text x="103" y="184" text-anchor="middle" font-size="10.5" fill="#52525b">出力ロジット + α・T</text><rect x="243" y="108" width="174" height="92" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="330" y="138" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">② ＋特徴量蒸留</text><text x="330" y="162" text-anchor="middle" font-size="11" fill="#52525b">内部表現も写す</text><text x="330" y="184" text-anchor="middle" font-size="10.5" fill="#52525b">hook + 射影層が必要</text><rect x="470" y="108" width="174" height="92" rx="6" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><text x="557" y="138" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">③ DeiT token / RKD</text><text x="557" y="162" text-anchor="middle" font-size="10.5" fill="#52525b">アーキ差が大きい時</text><text x="557" y="184" text-anchor="middle" font-size="10.5" fill="#52525b">token 蒸留 / 関係ベース</text><text x="216" y="100" text-anchor="middle" font-size="10.5" fill="#3f3f46">精度が足りなければ</text><text x="443" y="100" text-anchor="middle" font-size="10.5" fill="#3f3f46">アーキ差が大きいなら</text><line x1="190" y1="154" x2="237" y2="154" stroke="#71717a" stroke-width="2"/><polygon points="243,154 233,149 233,159" fill="#71717a"/><line x1="417" y1="154" x2="464" y2="154" stroke="#71717a" stroke-width="2"/><polygon points="470,154 460,149 460,159" fill="#71717a"/><rect x="16" y="210" width="628" height="24" rx="5" fill="#f4f4f5" stroke="#e4e4e7" stroke-width="1.5"/><text x="330" y="226" text-anchor="middle" font-size="11.5" font-weight="700" fill="#3f3f46">共通の鉄則: teacher は凍結 ／ T・alpha はスイープ</text></svg><figcaption>実務では <b>簡単な手法から段階的に強化</b>します。まず <b>① レスポンス蒸留</b>（出力ロジット＋α・T、最も簡単で安定）を試し、<b>精度が足りなければ ② 特徴量蒸留</b>（<code>hook</code> ＋射影層で内部表現も写す）を重ねます。teacher と student の<b>アーキが大きく違う</b>（CNN→ViT など）なら <b>③ DeiT distillation token や関係ベース(RKD)</b> へ進みます。teacher が無い／作れない場合は <b>self-distillation</b> も選択肢です。どの段階でも <b>teacher は凍結</b>し、<b>T・alpha はスイープ</b>するのが共通の鉄則です。</figcaption></figure>

---

## 🛠 章末ミニプロジェクト — 蒸留ベンチ（精度 × 圧縮率 × 速度）

`mini_project.py` は、本章の部品を 1 つに統合した **deliverable** だ。まずクリーンなフルデータで teacher を学習し、続いて同じ小 student を **4 通り**（(A) baseline / (B) response KD / (C) response+feature / (D) DeiT distillation token）で少量ノイジーデータに対して学習する。そのうえで **テスト accuracy・圧縮率・推論レイテンシ(p50/p99)** を同一指標の表にまとめ、図と JSON に保存する。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="蒸留ベンチの全体フロー。クリーン全データでteacherを学習し同じ小studentを4通り(A baseline B response KD C response+feature D DeiT distill token)で少量ノイジーデータに学習しacc圧縮率レイテンシを測定して表図JSONに出力する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト『蒸留ベンチ』の全体フロー</text><rect x="16" y="95" width="130" height="64" rx="6" fill="#eff6ff" stroke="#1d4ed8" stroke-width="2"/><text x="81" y="124" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">teacher 学習</text><text x="81" y="146" text-anchor="middle" font-size="11" fill="#52525b">クリーン全データ</text><rect x="172" y="52" width="164" height="150" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="254" y="74" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">同じ小 student を 4 通り</text><rect x="184" y="88" width="140" height="24" rx="4" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="254" y="104" text-anchor="middle" font-size="10.5" fill="#3f3f46">(A) baseline</text><rect x="184" y="118" width="140" height="24" rx="4" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><text x="254" y="134" text-anchor="middle" font-size="10.5" fill="#c2410c">(B) response KD</text><rect x="184" y="148" width="140" height="24" rx="4" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><text x="254" y="164" text-anchor="middle" font-size="10.5" fill="#c2410c">(C) response+feature</text><rect x="184" y="178" width="140" height="24" rx="4" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><text x="254" y="194" text-anchor="middle" font-size="10.5" fill="#c2410c">(D) DeiT distill token</text><rect x="380" y="95" width="120" height="64" rx="6" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="440" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">指標測定</text><text x="440" y="144" text-anchor="middle" font-size="10.5" fill="#52525b">acc・圧縮率・速度</text><rect x="528" y="95" width="116" height="64" rx="6" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><text x="586" y="122" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">表・図・JSON</text><text x="586" y="144" text-anchor="middle" font-size="11" fill="#52525b">に保存</text><line x1="146" y1="127" x2="166" y2="127" stroke="#71717a" stroke-width="2"/><polygon points="172,127 162,122 162,132" fill="#71717a"/><line x1="336" y1="127" x2="374" y2="127" stroke="#71717a" stroke-width="2"/><polygon points="380,127 370,122 370,132" fill="#71717a"/><line x1="500" y1="127" x2="522" y2="127" stroke="#71717a" stroke-width="2"/><polygon points="528,127 518,122 518,132" fill="#71717a"/></svg><figcaption>ミニプロジェクト『蒸留ベンチ』の全体フローです。まず <b>クリーンな全データで teacher を学習</b>し、続いて<b>同じ小 student を 4 通り</b>（<code>(A) baseline</code> / <code>(B) response KD</code> / <code>(C) response+feature</code> / <code>(D) DeiT distillation token</code>）で<b>少量ノイジーデータ</b>に学習します。その結果を <b>acc・圧縮率・推論レイテンシ(p50/p99)</b> で測定し、<b>表・図・JSON</b> に保存します。student のサイズは戦略を変えても不変で、蒸留は「同じ小ささのまま精度を底上げ」します。</figcaption></figure>

```bash
uv run python lectures/38_knowledge_distillation/mini_project.py
```

出力例（CPU・決定的）:

```
=== 蒸留ベンチ（同じ小 student・少量ノイジーデータ） ===
    teacher: acc=0.962  params=23,910

    strategy                    acc   params   compr   p50(ms)   p99(ms)
    (A) baseline              0.417    1,398   17.1x     2.28      2.62
    (B) response KD           0.725    1,398   17.1x     2.34      2.50  (+0.308)
    (C) response+feature      0.754    1,398   17.1x     2.31      2.74  (+0.337)
    (D) DeiT distill token    0.592    1,500   15.9x     2.37      3.06  (+0.175)
```

この表から読み取るべき結論は 2 つある。第一に、**student のサイズ（params・レイテンシ）は学習戦略を変えても不変**であり、蒸留はあくまで「同じ小ささのまま精度を底上げする」手法だということ。第二に、**どの蒸留も素の baseline を明確に上回っており**、teacher の暗黙知がノイズ過学習を抑えているということだ。こうして、まずレスポンス蒸留を試し、足りなければ特徴量や DeiT を重ねる、という意思決定が表から自然に導ける。

---

## ✅ 到達チェックリスト

- [ ] 蒸留が圧縮 4 本柱の 1 つで、量子化/プルーニングと直交することを説明できる。
- [ ] soft target（暗黙知）がハードラベルより豊かな信号である理由を言える。
- [ ] 温度 T の役割（軟化）と、T を上げるとエントロピーが増えることを説明できる。
- [ ] 温度付き KL を `log_softmax(student)` / `softmax(teacher)` / `batchmean` / `T**2` で**正しく書ける**。
- [ ] `T**2` を掛ける理由（勾配スケール補正）を説明できる。
- [ ] 合成損失 `alpha*soft + (1-alpha)*hard` を書け、alpha の意味を言える。
- [ ] teacher を `eval()+requires_grad=False+inference_mode()` で凍結し、optimizer に student だけ渡せる。
- [ ] `register_forward_hook` で中間特徴を取り、射影層+正規化+MSE で特徴量蒸留を書ける。
- [ ] レスポンス/特徴量/関係ベースの 3 系統と DeiT distillation token の違いを説明できる。
- [ ] 同じ student で「素 vs 蒸留」の accuracy 差を測り、蒸留の効果を定量化できる。

---

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/38_knowledge_distillation/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 温度 T でロジットを軟化した確率分布 `softmax(logits / T)` を `dim=1` で返す（`ex1_soften_logits` の TODO）。
2. 温度付き KL（Hinton 蒸留のソフト損失）を `log_softmax(student/T)`・`softmax(teacher/T)`・`reduction="batchmean"`・末尾の `T**2` の正準形で組み立て、スカラーで返す（`ex2_kd_kl_loss` の TODO）。
3. 合成損失 `alpha * ソフト(温度付きKL) + (1-alpha) * ハード(cross_entropy)` を線形結合して返す（`ex3_combined_kd_loss` の TODO）。
4. 圧縮率 = teacher のパラメータ数 / student のパラメータ数 を返す（`student_params` が 0 のときは 0.0）（`ex4_compression_ratio` の TODO）。
5. `softmax(logits / T)` の平均エントロピー(nats) を float で返す（`log(0)` を避けるため確率に微小値を足してから log を取る）（`ex5_soft_target_entropy` の TODO）。
6. 特徴量蒸留の MSE を返す（射影 → L2 正規化 → MSE の順で、student 特徴を proj_weight で teacher 次元へ合わせる）（`ex6_feature_distill_mse` の TODO）。
7. DeiT 推論時の出力として class ヘッドと distill ヘッドの平均ロジットを返す（`ex7_deit_inference_logits` の TODO）。
8. 関係蒸留(RKD) の正規化距離行列を返す（`cdist` でペア間距離を取り、対角を除いた正の要素平均で正規化）（`ex8_rkd_distance_matrix` の TODO）。

---

## ❓ 落とし穴・FAQ・デバッグ

- **KL の引数を逆にした / log を付け忘れた**: `F.kl_div` には、**入力に `log_softmax(student)`、ターゲットに `softmax(teacher)`** を渡す。順序を入れ替えたり log を付け忘れたりすると、別物の損失になって学習が崩れる。`02` の `[3]` で値が変わることを実演している。
- **`reduction='mean'` を使った**: 既定の `'mean'` は要素数（バッチ×クラス）で割るため、数式の `1/N`（バッチサイズで割る）とずれ、クラス数倍だけ小さくなってしまう。したがって必ず **`reduction='batchmean'`** を使うこと。
- **`T**2` を忘れた**: T を変えるたびにソフト損失の実効重みが変わり、最適な alpha も動いて不安定になる。`02` の `[4]` で勾配ノルムが揃うことを確認できる。
- **teacher を凍結し忘れた**: BN 統計や dropout が動いて擬似ラベルが毎回ブレる。さらに optimizer へ teacher のパラメータを渡すと、誤って更新されてしまう。よって `freeze_module()`（eval + requires_grad=False）を必ず通し、teacher の推論は `inference_mode()` で行う。
- **特徴量蒸留で次元が合わない**: student と teacher の中間特徴はチャネル数が違うため、**射影層（1x1 Conv か Linear）を必ず挟む**。さらに、チャネルごとのスケール差を `F.normalize` で吸収してから MSE を取らないと、スケールの大きいチャネルに損失が支配されてしまう。
- **フックを付けっぱなしにした**: `register_forward_hook` の戻り値（ハンドル）を保持しておき、学習後に `handle.remove()` する。付けっぱなしだと別の推論でも特徴が保存され続け、メモリと混乱の元になる。
- **「蒸留したのに精度が上がらない」**: teacher が student より十分強いか、T/alpha が適切かをまず疑う。**T と alpha はスイープ必須**だ（`03` の `[2][3]`）。なお、データがクリーンで student が単独でも十分学べる場合は、蒸留の上積みは小さい（蒸留が効くのは「容量が足りない / データが少ない・ノイジー」な状況である）。
- **DeiT で推論時にヘッドを平均し忘れた**: distillation token を持つモデルは、推論時に class ヘッドと distill ヘッドの**平均**を取る。片方だけ使うと性能を取りこぼす。なお timm の `*_distilled` は eval 時に自動で平均してくれる。

---

## 🚀 発展トピック・参考

- **関係ベース蒸留(RKD)**: 個々の特徴ではなくサンプル間の距離・角度の関係を合わせる。特徴次元が大きく違う teacher/student でも比較でき、`04` の `[2]` で距離行列の発想を紹介している。
- **self-distillation / Born-Again Networks**: teacher と student を同一アーキにして繰り返し蒸留すると、しばしば元 teacher を超える。深い層から浅い層へ蒸留する自己蒸留もある。
- **online / mutual distillation (DML)**: 固定 teacher を使わず、複数の student を同時に学習させて互いに soft target を教え合う。
- **VLM/CLIP の蒸留（次回39）**: 画像-テキスト埋め込みを小型エンコーダへ蒸留する TinyCLIP / MobileCLIP。本章の「ロジット模倣」が「埋め込み模倣（コサイン/L2 正規化 + logit_scale）」に変わる。
- **蒸留 × 量子化/プルーニング**: 蒸留で小モデルを賢く育て、さらに 35 回の量子化/プルーニング、36 回の ONNX 化を重ねるとエッジ展開に効く。
- 参考: Hinton et al. *Distilling the Knowledge in a Neural Network* (2015) / Romero et al. *FitNets* (2015) / Park et al. *Relational KD* (2019) / Touvron et al. *DeiT*（distillation token, 2021）/ timm token distillation。

---

## ▶ 動かし方

```bash
# 依存（未導入なら）: 深層学習の土台（CPU 版 torch / torchvision）
uv sync --group dl
# DeiT 構造の確認に timm を使う場合のみ（任意・未導入でも 05 は概念のみで exit 0）
uv sync --group hf

# 本編（番号順）。teacher/student は合成図形データでその場で軽く学習（数秒・決定的）
uv run python lectures/38_knowledge_distillation/01_kd_overview.py
uv run python lectures/38_knowledge_distillation/02_hinton_soft_targets.py
uv run python lectures/38_knowledge_distillation/03_response_kd_train.py
uv run python lectures/38_knowledge_distillation/04_feature_distill.py
uv run python lectures/38_knowledge_distillation/05_distill_token_deit.py

# 章末ミニプロジェクト（蒸留ベンチの完成形）
uv run python lectures/38_knowledge_distillation/mini_project.py

# 演習（自己採点。未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/38_knowledge_distillation/exercises.py
uv run python lectures/38_knowledge_distillation/exercises_solutions.py
```

成果物（図・JSON）は `lectures/38_knowledge_distillation/outputs/` に保存されます。実行は CPU 前提で、`model.eval()` + `torch.inference_mode()` を用い、headless（`imshow` は呼ばず matplotlib=Agg で保存）で動きます。また teacher は必ず `freeze_module()`（eval + requires_grad=False）で凍結します。

---

> 参照ライブラリ: **torch 2.12+cpu**（`F.kl_div` / `register_forward_hook` / `AdamW`）/ **timm 1.0.27**（任意・DeiT distillation token の構造確認）/ 次回39で **transformers 5.11** / **open_clip 3.3** による CLIP 蒸留へ接続。
> （題材: cv2 合成図形 6 クラス分類・`TeacherCNN`(やや大)/`StudentCNN`(小)、少量ノイジーデータで「素 vs 蒸留」を比較、CPU・`model.eval()`+`torch.inference_mode()`、teacher は `freeze_module` で凍結） — 2026-06
