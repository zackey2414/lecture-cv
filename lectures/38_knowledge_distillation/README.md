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

では、なぜそれが効くのか。小さな student は容量が限られるため、少ないデータでハードラベルだけを追うと、ノイズや暗記に陥りやすい。これに対し teacher の soft target は、各サンプルに連続的な「正解らしさの分布」を与えるので、実質的に**1 サンプルあたりの情報量を増やし**、正則化としても働く。そこで本モジュールのトイでは「少量・ラベル 40% ノイズ」という意地悪な状況をあえて用意し、クリーンなフルデータで育った teacher の soft target が student をノイズ過学習から救う様子を観察する。

そもそも蒸留は、モデル圧縮の **4 本柱（蒸留 / 量子化 / プルーニング / 低ランク分解）** の 1 本目にあたる。量子化やプルーニング(35回)が「同じモデルを物理的に削る」のに対し、蒸留は「**小さいモデルを賢く学習させる**」アプローチであり、他の 3 本と直交するため組み合わせて使える（蒸留した小モデルをさらに量子化、など）。`01_kd_overview.py` では、この全体地図と teacher 凍結を確認する。

---

## 2. 理論 — 温度付きソフトターゲットと KL ダイバージェンス

蒸留の心臓部は **温度 T による軟化** と **KL ダイバージェンス損失** だ。ロジット `z` をそのまま softmax すると分布が尖り、暗黙知が埋もれてしまう。そこで `softmax(z / T)` のように温度 T で割ってから softmax する。こうすると T を上げるほど分布はなだらかになり、2 番手以降の確率が持ち上がって読み取りやすくなる（T=1 は素の softmax、T→∞ で一様分布）。なお蒸留では T=2〜5 がよく使われる。

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

蒸留は、伝える「知識の場所」によって 3 系統に分かれる。すなわち、**レスポンスベース**（出力ロジットを真似る＝Hinton 蒸留）、**特徴量ベース**（中間特徴を真似る＝FitNets）、**関係ベース**（サンプル間の距離・角度の関係を真似る＝RKD）の三つだ。それぞれ、レスポンスは最も簡単で効果も安定し、特徴量は teacher の内部表現まで写せる代わりに次元合わせ（射影層）が要り、関係は特徴次元が違っても比較できる、という棲み分けになっている。

---

## 3. 正準 API — `F.kl_div` / `register_forward_hook` / 凍結

蒸留の実装で覚える正準 API は、驚くほど少ない。まず**ソフト損失**は `F.log_softmax` / `F.softmax` / `F.kl_div(..., reduction="batchmean")`、**ハード損失**は `F.cross_entropy` で組み立てる。次に**特徴量蒸留**では、中間特徴を取り出す `module.register_forward_hook(fn)`、次元を合わせる射影層 `nn.Conv2d(student_ch, teacher_ch, 1)`（または `nn.Linear`）、スケール差を消す `F.normalize`、そして `nn.MSELoss` / `F.mse_loss` を使う。最適化は一貫して `torch.optim.AdamW` でよい。

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

---

## 🛠 章末ミニプロジェクト — 蒸留ベンチ（精度 × 圧縮率 × 速度）

`mini_project.py` は、本章の部品を 1 つに統合した **deliverable** だ。まずクリーンなフルデータで teacher を学習し、続いて同じ小 student を **4 通り**（(A) baseline / (B) response KD / (C) response+feature / (D) DeiT distillation token）で少量ノイジーデータに対して学習する。そのうえで **テスト accuracy・圧縮率・推論レイテンシ(p50/p99)** を同一指標の表にまとめ、図と JSON に保存する。

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

成果物（図・JSON）は `outputs/38_knowledge_distillation/` に保存されます。実行は CPU 前提で、`model.eval()` + `torch.inference_mode()` を用い、headless（`imshow` は呼ばず matplotlib=Agg で保存）で動きます。また teacher は必ず `freeze_module()`（eval + requires_grad=False）で凍結します。

---

> 参照ライブラリ: **torch 2.12+cpu**（`F.kl_div` / `register_forward_hook` / `AdamW`）/ **timm 1.0.27**（任意・DeiT distillation token の構造確認）/ 次回39で **transformers 5.11** / **open_clip 3.3** による CLIP 蒸留へ接続。
> （題材: cv2 合成図形 6 クラス分類・`TeacherCNN`(やや大)/`StudentCNN`(小)、少量ノイジーデータで「素 vs 蒸留」を比較、CPU・`model.eval()`+`torch.inference_mode()`、teacher は `freeze_module` で凍結） — 2026-06
