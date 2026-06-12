# 第14回 分類の評価指標(A) — 混同行列・precision/recall/F1・ROC/PR・AUC

> トラック: 評価指標 ／ レベル: 初級 ／ 依存グループ: `dl`（torch）・`metrics`（scikit-learn / torchmetrics）。画像モデルもネット接続も不要で、合成した「予測スコアと正解ラベル」だけで完結します。

## 🎯 この章のゴール

この章を終えたとき、あなたは「分類器の良し悪しを accuracy 一発で語ってはいけない」という感覚を体に入れた状態になります。具体的には、すべての分類評価の最小単位である TP / FP / FN / TN と混同行列を自分の手で組み立てられ、そこから precision・recall・F1 を定義式どおりに導けること。さらに、macro / micro / weighted の3つの平均が「何を平等に扱っているのか」を説明でき、不均衡データでどれを見るべきかを判断できること。これらがこの章の土台です。

さらに一歩進んで、「あるしきい値で予測を 0/1 に固めた後」の指標（precision/recall）と、「しきい値を全部掃引した挙動をまとめた」しきい値非依存の指標（ROC-AUC と PR-AUC=AP）の違いを、曲線を自分で描けるレベルで理解します。とりわけ、クラス不均衡のときに ROC-AUC が楽観的に高く出るのに対し PR-AUC（AP）が実態を映す、という非対称を、数値とグラフの両方で説明できるようになります。

到達点を一言でいえば、**予測スコアと正解ラベルさえあれば、混同行列から ROC/PR 曲線まで AI 補助なしで numpy で書け、その値が scikit-learn や torchmetrics と一致することを自分で検証できる**ことです。指標を「ライブラリの関数名」ではなく「式と手順」で語れるようになることが、この章の合格ラインです。本講座は後段（第19回）で物体検出の mAP を一から実装しますが、その下地は、まさにこの章の「混同行列→PR曲線→面積」という流れにあります。

---

## 1. 評価の最小単位 — 混同行列と TP/FP/FN/TN

あらゆる分類評価は、たった4つの数 —— TP（真陽性）、FP（偽陽性）、FN（偽陰性）、TN（真陰性）—— から組み上がります。二値分類で「陽性を当てた」のが TP、「陰性を誤って陽性と言った」のが FP、「陽性を取りこぼした」のが FN、「陰性を正しく陰性と言った」のが TN です。この4つさえ数えられれば、precision も recall も F1 も accuracy も、あとは割り算で出てきます。逆にいえば、指標で詰まったときは必ずこの4つに立ち返れば良い、というのが評価指標を学ぶうえでの最大の足場になります。

多クラス（K クラス）になっても考え方は同じで、すべては **混同行列** `C` に集約されます。`C[i, j]` は「正解が i なのに j と予測した件数」で、対角（`i == j`）が正解、対角以外が間違いです。あるクラス c に注目して「c か、それ以外か」という二値問題に分解（one-vs-rest）すると、4つの数が混同行列の中に見えてきます。対角の `C[c,c]` が TP、**列 c の対角以外の合計**が FP（他クラスを c と誤った）、**行 c の対角以外の合計**が FN（c を他クラスと誤った）、そして残り全部が TN です。この「行＝正解、列＝予測」という向きと、FP は列・FN は行という対応を、最初に手で確かめておくと一生忘れません。

`01_confusion_matrix_prf.py` は、この混同行列を `np.add.at` で散布加算して自作し、`sklearn.metrics.confusion_matrix` と完全一致することを `assert` で確認します。下のスニペットがその核心で、ループを書かずに「(正解, 予測) の組ごとに +1」を一括で行っています。

```python
cm = np.zeros((k, k), dtype=int)
np.add.at(cm, (y_true, y_pred), 1)   # C[i,j] = 正解iを j と予測した件数
tp = np.diag(cm)                      # 対角＝正解
fp = cm.sum(axis=0) - tp              # 列方向 - 対角 ＝ 偽陽性
fn = cm.sum(axis=1) - tp              # 行方向 - 対角 ＝ 偽陰性
```

ここで `cm.sum(axis=0)` が列の和（予測側の総数）、`cm.sum(axis=1)` が行の和（正解側の総数）であることを必ず押さえてください。`axis` を取り違えると FP と FN が入れ替わり、precision と recall がまるごと逆になります。スクリプトは混同行列をヒートマップ（`01_confusion_matrix.png`）として保存するので、対角が濃く・非対角が薄いほど良い分類器、という見方も目で確認できます。

## 2. accuracy の罠 — クラス不均衡では多数派に張るだけで高く出る

accuracy（正解率）は `(TP+TN)/全件` で、最も直感的な指標です。ところが、これは**クラスの頻度が偏ると簡単に嘘をつきます**。例えば陽性がたった10%しかないデータでは、「何も考えず全部を陰性と予測する」だけの分類器が accuracy 0.90 を叩き出します。中身は空っぽ（陽性を1件も当てていない）なのに、数字だけを見ると優秀に見えてしまいます。これが実務で最も多い「評価の事故」です。

なぜこうなるかというと、accuracy は TN（多数派の正解）も TP（少数派の正解）も区別せずに足し込むため、多数派が多いほど多数派の正解だけで数字が膨らむからです。だからこそ、不均衡データでは accuracy 単独を信じず、**少数派クラスの recall（取りこぼし）や F1 を必ず併記する**のが鉄則になります。「accuracy が高い」と言われたら、まず「クラス比はどうなっているか」「少数派の recall はいくつか」を聞き返す癖をつけてください。

`01_confusion_matrix_prf.py` の `demo_accuracy_trap()` は、この罠を数値で突きつけます。実行すると次のような出力が得られ、「常に陰性」分類器は accuracy が高いのに陽性 recall がゼロである一方、しきい値0.5の中身のある分類器は accuracy ではむしろ下がるのに陽性をきちんと拾えている、という逆転が見て取れます。

```text
[不均衡での accuracy の罠] 陽性割合 = 9.4%
  常に陰性と予測  : accuracy=0.905  陽性recall=0.000  ← 高精度に見えるが無能
  しきい値0.5の分類: accuracy=0.542  陽性recall=0.910  ← recall を見て初めて差が出る
```

この出力の意味するところは重大です。accuracy という1つの数字は、評価軸を1本に潰してしまうため、「多数派をなぞるだけの怠慢」と「少数派を頑張って拾う有能さ」を区別できません。後述の per-class 指標や PR-AUC は、まさにこの潰れた軸を分解して見せるための道具です。

## 3. precision / recall / F1 と macro / micro / weighted 平均

precision（適合率）は `TP/(TP+FP)`、recall（再現率）は `TP/(TP+FN)` です。precision は「陽性と予測したもののうち、本当に陽性だった割合」＝**当てに行った予測がどれだけ正確か**、recall は「本当の陽性のうち、拾えた割合」＝**取りこぼしの少なさ**を表します。この2つはしばしばトレードオフの関係にあり、しきい値を下げれば recall は上がるが precision は下がります。両者の調和平均が F1 `= 2PR/(P+R)` で、precision と recall のバランスを1つの数にまとめたものです。どちらか一方が極端に低いと F1 も大きく下がる（調和平均の性質）ので、片肺飛行を見逃しにくいのが F1 の利点です。

多クラスでは、クラスごとに出た precision/recall/F1 を1つの代表値に**平均化**します。ここで3つの流儀があり、選び方を間違えると数字が比較不能になります。**macro** は各クラスの指標を単純平均する方式で、頻度に関係なく全クラスを対等に扱うため、少数クラスの出来も平等に効きます。**micro** は全クラスの TP/FP/FN を先に合算してから式に入れる方式で、件数の多いクラスほど強く効き、多クラス単一ラベルでは値が accuracy と一致します。**weighted** は各クラスの指標をその出現頻度（support）で重み付け平均する、macro と micro の中間です。

`01_confusion_matrix_prf.py` はこの3つを自作で計算し、`precision_recall_fscore_support` の `average="macro"/"micro"/"weighted"` と一致することを確認します。下の表は、それぞれが「何を平等に扱うか」と「いつ使うか」の早見表です。

| 平均化 | 計算の仕方 | 何を平等に扱う | 使いどころ |
| --- | --- | --- | --- |
| `macro` | クラスごとの指標を単純平均 | クラスを対等（少数派も1票） | 少数クラスも重要なとき・不均衡 |
| `micro` | 全クラスの TP/FP/FN を合算してから計算 | サンプルを対等（多数派が強い） | 全体の取りこぼしを見たいとき（=accuracy） |
| `weighted` | クラスごとの指標を support で加重平均 | 頻度に比例 | 全体傾向を1値で代表したいとき |

表の通り、「少数クラスを見逃したくない」なら macro、「全体の正解率に近い感覚が欲しい」なら micro、という使い分けになります。論文や他チームの数字と突き合わせるときは、**どの平均か**を必ず確認してください。同じ「F1」でも macro と micro で値が変わり、平均化を言わずに F1 だけ比べるのは無意味です。スクリプトは最後に `classification_report` も出力するので、クラス別の値と3つの平均を一望できます。

## 4. top-k accuracy と one-vs-rest 分解

クラス数が多いタスク（ImageNet の1000クラスなど）では、「1位が正解か」だけで評価するのは厳しすぎることがあります。そこで **top-k accuracy** —— 確率上位 k 個の予測の中に正解が含まれていれば正解とみなす —— を併用します。top-1 は通常の accuracy、top-5 は「上位5候補に入っていればOK」で、モデルが正解を“惜しいところ”まで絞れているかを測れます。計算は単純で、各サンプルについて確率を降順に並べ、上位 k 個のクラス番号に正解ラベルが入っているかを見るだけです。

```python
topk = np.argsort(-proba, axis=1)[:, :k]          # 各行の上位 k クラス番号
top_k_acc = (topk == y_true[:, None]).any(axis=1).mean()
```

このスニペットの `argsort(-proba)` が「確率の大きい順に並べ替えた添字」を返し、その上位 k 列に正解が含まれる行の割合を取っています。`01_confusion_matrix_prf.py` はこれを自作し、`sklearn.metrics.top_k_accuracy_score` と一致することを確認します。実行すると top-2 accuracy が top-1 より高く出る（候補を増やせば当たりやすくなる）ことが見て取れます。

なお、第1節で触れた **one-vs-rest（OvR）分解** は、この章を貫く考え方です。多クラスの混同行列を「クラス c か、それ以外か」という K 個の二値問題に分解すると、各クラスについて TP/FP/FN/TN・precision/recall が定義でき、次節の ROC/AUC も「クラスごとに二値の ROC を描いて平均する（macro OvR）」という形で多クラスへ自然に拡張できます。「多クラスは二値の束である」と捉えると、見通しが一気に良くなります。

## 5. しきい値非依存の指標 — ROC 曲線と ROC-AUC

ここまでの precision/recall/F1 は、すべて「あるしきい値で予測を 0/1 に確定させた後」の指標でした。しかし分類器が実際に出すのは連続値のスコア（陽性確率など）で、しきい値を 0.5 に置くか 0.3 に置くかで TP/FP は変わります。そこで、**しきい値を高い方から低い方へ全部掃引したときの挙動**を1本の曲線にまとめたのが ROC 曲線です。縦軸に TPR（=recall=`TP/(TP+FN)`）、横軸に FPR（=`FP/(FP+TN)`）を取り、しきい値を下げながら点を打っていきます。しきい値が高いうちは「ほぼ何も陽性と言わない」ので左下、下げきると「全部陽性」で右上に達し、(0,0)→(1,1) を結ぶ曲線になります。

この曲線の**下の面積が ROC-AUC** です。AUC は「ランダムに選んだ陽性サンプルのスコアが、ランダムに選んだ陰性サンプルより高い確率」と等価で、0.5 が当てずっぽう、1.0 が完璧を意味します。重要なのは、AUC が**順位（ランキング）の指標**であって、スコアの絶対値や確率較正とは別物だという点です。スコアを単調変換（例えばシグモイドを通す）しても順位は変わらないので AUC も変わりません。

`02_roc_pr_auc.py` は、スコアを降順に並べて TP/FP を累積し、(FPR, TPR) を打点して**台形則**で面積を取る、という手順をそのまま numpy で書きます。`sklearn.metrics.roc_auc_score` も内部は台形則（`auc(fpr, tpr)`）なので、自作とライブラリはほぼ完全に一致します（実行ログで「自作=0.8623 / sklearn=0.8623」のように並びます）。

```python
order = np.argsort(-score)                 # スコア降順 ＝ しきい値を高→低に下げる
tp = np.cumsum(y_true[order])              # ここまでで拾った陽性
fp = np.cumsum(1 - y_true[order])          # ここまでで誤った陰性
tpr = np.r_[0.0, tp / n_pos]; fpr = np.r_[0.0, fp / n_neg]
roc_auc = np.trapezoid(tpr, fpr)           # 台形則。numpy 2.x は trapz 廃止 → trapezoid
```

なお numpy 2.x では従来の `np.trapz` が廃止され、`np.trapezoid` に置き換わっています（本講座の numpy 2.4 では `np.trapz` は存在しません）。古いコードを写すときの落とし穴なので、面積計算は `np.trapezoid` を使ってください。同点（タイ）のスコアが多いと階段の作り方に注意が要りますが、本章の合成データは連続スコアなので同点はほぼ生じず、自作値がきれいにライブラリと揃います。

## 6. PR 曲線と PR-AUC(=AP)、不均衡での ROC との違い

ROC と並ぶもう一つのしきい値非依存指標が **PR 曲線**（precision-recall 曲線）です。同じしきい値掃引で、今度は横軸に recall、縦軸に precision を取って打点します。その下の面積が PR-AUC で、実務では **AP（Average Precision）** という呼び名がよく使われます。ここで一つ技術的に大事な注意があります。`sklearn.metrics.average_precision_score` は PR 曲線を**台形**で積むのではなく、`AP = Σ (recall_i − recall_{i−1}) × precision_i` という**短冊（ステップ）和**で計算します。台形則の `auc(recall, precision)` とは微妙に値が違い、論文比較で正準とされるのは前者の AP の方です。自作で AP を出すときは、この「ステップ和」を再現すると sklearn とぴったり合います。

ROC と PR の決定的な違いは、**クラス不均衡への感度**に出ます。ROC-AUC は TPR と FPR がそれぞれクラス内で正規化されているため、陽性が珍しくなっても値がほとんど変わりません（順位の指標なので頻度に鈍感）。一方 AP は precision を通じて陽性割合（prevalence）を直接織り込むため、陽性が珍しいほど下がります。無情報な分類器（全部に同じスコアを付ける）の AP は、ちょうど**陽性割合そのもの**になり、これが PR 曲線のベースライン（基準線）です。だから不均衡タスクでは、「ROC-AUC 0.86」よりも「AP 0.53、ベースライン 0.10」の方が、改善の余地と実力を正直に映します。

`02_roc_pr_auc.py` は、**分離度（gap）を同じに保ったまま陽性割合だけ 10% と 50% に変えた**2つのデータを比べます。実行ログを見ると、この非対称が一目で分かります。

```text
[不均衡] 陽性割合=10.6%  ROC-AUC: 0.8623   AP: 0.5267（ベースラインAP≈0.106）
[均衡]   陽性割合=50.9%  ROC-AUC: 0.8452   AP: 0.8505（ベースラインAP≈0.509）
```

ROC-AUC は不均衡（0.862）でも均衡（0.845）でもほぼ同じなのに、AP は不均衡で 0.53 まで落ち、ベースライン（0.106）と比べて初めて「無情報よりはずっと良い」と読めます。`02_roc_curve.png` と `02_pr_curve.png` を見比べると、PR 曲線にはクラス比に応じた点線のベースラインが引かれており、ROC では捉えきれない実態を PR が映していることが視覚的に納得できます。**不均衡タスクでは PR-AUC/AP を主指標に、ROC-AUC は補助に**、と覚えてください。

## 7. 自作実装とライブラリ値の突き合わせ — なぜ一致し、どこでズレるか

この章のスクリプトはどれも「numpy で自作した値」と「ライブラリの値」を `assert np.isclose(...)` で突き合わせます。これは単なる答え合わせではなく、**指標を式で理解できている証拠**です。混同行列・precision/recall/F1・top-k・ROC-AUC は、いずれも自作とライブラリが（浮動小数の丸め誤差の範囲で）一致します。一致を自分の目で確認することで、「ライブラリがブラックボックスに見える」状態から「中で何をしているか分かる」状態へ進めます。

一方で、**意図的にズレうる箇所**も知っておく価値があります。最大の落とし穴が前節の AP で、ステップ和（`average_precision_score`）と台形則（`auc(recall, precision)`）は別物です。値が微妙に食い違ったとき、「どちらの定義か」を確認できるだけでデバッグが早くなります。また、同点スコアが多いデータでは、ROC/PR の階段の作り方（同点をまとめるか）で曲線の点数が変わります。AUC の値自体は同じでも、`roc_curve` が返す点の数は sklearn 側が冗長点を間引くぶん自作より少なくなる、といった差が出ます。

```python
assert np.isclose(roc_auc_manual, roc_auc_score(y_true, score), atol=1e-6)   # ROC は台形で一致
assert np.isclose(ap_manual,      average_precision_score(y_true, score), atol=1e-6)  # AP はステップ和で一致
```

このスニペットのように、許容誤差 `atol` を添えて一致を主張するのが実務的な作法です。完全一致（`==`）を求めると浮動小数の最下位ビットの差で落ちることがあるため、指標の照合は必ず `np.isclose` を使います。「自作とライブラリが一致した」という体験を3つのスクリプトで積み重ねれば、第19回の mAP 自作実装にも臆せず進めます。

## 8. torchmetrics の update → compute → reset サイクル

scikit-learn は「全データが揃ってから一括で計算する」スタイルですが、深層学習の学習・評価ループでは「ミニバッチごとに少しずつ指標を貯めていき、エポックの最後にまとめて確定する」方が自然です。その用途に向くのが **torchmetrics** で、`update(preds, target)` でバッチを足し込み、`compute()` で最終値を出し、`reset()` で状態を空にする、という3拍子のサイクルで使います。`reset()` を忘れると次のエポックに前のエポックの統計が混ざり続けるので、エポックの頭で必ずリセットするのが鉄則です。

`03_torchmetrics_vs_manual.py` は、データをわざと2バッチに分けて `update` を2回呼び、最後に1回だけ `compute` します。そして得られた Accuracy / macro-F1 / macro-AUROC / macro-AP が、全件を一括で計算した sklearn の値と（許容誤差内で）一致することを確認します。「バッチに分けて貯めても、まとめて計算しても同じ」——この更新順・分割に依らない性質が、torchmetrics を学習ループに安心して組み込める理由です。多クラスの AUROC/AP はクラスごとの OvR を macro 平均したもので、`roc_auc_score(..., multi_class="ovr", average="macro")` と一致します。

```python
acc = MulticlassAccuracy(num_classes=k, average="micro").to(DEVICE)
for sl in (slice(0, half), slice(half, None)):   # ミニバッチを模して2回 update
    acc.update(preds[sl], target[sl])
print(acc.compute())   # まとめて確定 → 全件計算と一致
acc.reset()            # 次のエポックの前に必ず空にする
```

device の扱いも実務的なポイントです。評価指標はすべて CPU で完結し（GPU は大規模バッチを速くするだけ）、唯一の注意点は **`preds` と `target` を同じ device に揃える**ことだけです。本章は `device = "cuda" if ... else ("mps" if ... else "cpu")` の定石で自動判定しますが、CPU では `cpu` のまま動きます。`preds` を確率 `(n, K)`、`target` を整数ラベル `(n,)` で渡す、AUROC/AP はしきい値非依存なので確率（logits でも可）を渡す、という入力の型だけ押さえれば、あとは sklearn と同じ結果が得られます。スクリプトは3者（自作・sklearn・torchmetrics）が重なる棒グラフ `03_multiclass_compare.png` も保存します。

## 9. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「混同行列 → しきい値非依存の曲線 → ライブラリ連携」と理解が積み上がります。すべて `outputs/14_eval_classification/` に図と JSON を保存し、画面表示には依存しません。題材データ（合成した予測スコアと正解ラベル）の生成と出力先管理は `eval_helpers.py` にまとめ、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `eval_helpers.py` | 合成データ生成（多クラス確率・二値スコア）と `output_dir()`。各スクリプトが import する道具箱 |
| `01_confusion_matrix_prf.py` | 混同行列の自作、TP/FP/FN/TN、precision/recall/F1、macro/micro/weighted、top-k、不均衡の accuracy 罠 |
| `02_roc_pr_auc.py` | ROC/PR 曲線の自作、台形則の ROC-AUC、ステップ和の AP、不均衡での ROC と PR の違い |
| `03_torchmetrics_vs_manual.py` | torchmetrics の update→compute→reset、自作/sklearn/torchmetrics の三者一致、device の揃え方 |
| `mini_project.py` | 章末ミニプロジェクト。評価指標を統合した「モデル評価レポート」を生成（モデル選択・しきい値最適化・ブートストラップ信頼区間・6 パネルのダッシュボード） |
| `exercises.py` | TODO 形式の演習 8 問（易→難。自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答に差し替え） |
| `exercises_solutions.py` | 演習の模範解答（全 PASS）。採点ロジックは `exercises.py` の `grade()` を再利用（二重定義なし） |

表の通り `eval_helpers.py` だけは「読み物」ではなく「再利用する道具」です。中身も厚くコメントしてあるので、最初に一読してから 01 へ進むと、各スクリプトが何のデータで実験しているかが腑に落ちます。実データで試したい人は、自分の `(y_true, proba)` を作って同じ関数に流せばそのまま動きます。

## 🛠 章末ミニプロジェクト — 「分類モデル評価レポート」を一枚にまとめる

ここまでの 01〜03 は指標を一つずつ分解して学ぶ「部品」でした。`mini_project.py` は、それらを統合して**実務でそのまま提出できる評価レポートを 1 枚（6 パネルのダッシュボード）＋ JSON で吐く**総合課題です。題材は合成データ（完全 CPU・数秒で完走）ですが、設計は実プロジェクトの評価フローそのものです。digit 始まりの 01〜03 は import できないため、必要な計算は本ファイル内に自己完結で書き（共通の `eval_helpers` だけ借用）、自作値はすべて scikit-learn と `assert` で一致を確認しています。

このミニプロジェクトには、本編 01〜03 には無い**マスター要素を3つ**足してあります。

1. **モデル選択（accuracy の落とし穴の実演）** — 同じ正解データに対し「強い分類器（signal 大）」と「弱い分類器（signal 小）」を用意し、`accuracy` で選ぶのと `macro-F1 / macro-AP` で選ぶのを並べて表示します。不均衡では accuracy が多数派をなぞるだけで上振れするため、**少数クラスを対等に見る macro 系を主指標にする**、という選定の作法を数値で体感します。
2. **しきい値最適化（目的で最適点は変わる）** — 二値サブタスクでスコアを降順に掃引し、`TP/FP` の累積から全しきい値の `precision/recall/F1` と `TPR/FPR` を一気に評価します。そのうえで **F1 を最大化するしきい値**と **Youden-J（`TPR−FPR`）を最大化するしきい値**を別々に求め、両者が一致しないこと・既定の `0.5` が最適とは限らないことを示します。「陽性を当てたい」のか「誤受入(FPR)を抑えたい」のかで選ぶしきい値が変わる、という運用判断を学びます。
3. **ブートストラップ信頼区間（点推定で断じない）** — 評価セットを復元抽出して `macro-F1` を 500 回計算し、**95% 信頼区間**を求めます。評価値には必ずブレがあるため、0.001 の差で勝敗を言わず**区間で実力を語る**——これがマスター水準の評価作法です。

```bash
uv run python lectures/14_eval_classification/mini_project.py
```

出力は `outputs/14_eval_classification/mini_project_dashboard.png`（混同行列・クラス別 F1・ROC・PR・しきい値 vs F1・ブートストラップ分布の 6 枚）と `mini_project_report.json`（全指標・選定結果・最適しきい値・信頼区間）です。コンソールには上の3要素の要約が表示されます。**到達目標は「このレポートを自分で再現でき、各数値が何を意味し、なぜその指標で結論を出したかを説明できる」こと**です。

## ✅ 到達チェックリスト

この章を「マスターした」と言える基準です。すべて**手を動かして**確認してください（読んで分かった気にならない）。

- [ ] 混同行列 `C[i,j]` の向き（行＝正解・列＝予測）を説明でき、`np.add.at` で自作して sklearn と一致させられる。
- [ ] 混同行列から one-vs-rest で TP/FP/FN/TN を取り出し、precision/recall/F1 を**式から**計算できる（`FP は列・FN は行`）。
- [ ] macro / micro / weighted の3平均が「何を平等に扱うか」を即答でき、不均衡で macro を主指標に選ぶ理由を言える。
- [ ] top-k accuracy を `argsort` 1 行で書け、top-1 ≤ top-2 ≤ … となる理由を説明できる。
- [ ] 「不均衡で accuracy が嘘をつく」例（常に多数派予測で高 accuracy・少数派 recall=0）を自分で再現できる。
- [ ] ROC 曲線を降順掃引で打点し、台形則（`np.trapezoid`）で ROC-AUC を自作して `roc_auc_score` と一致させられる。
- [ ] AP（PR-AUC）が**ステップ和**（`Σ(recall差)×precision`）であることを知り、台形則の `auc(recall,precision)` と区別できる。
- [ ] 「ROC-AUC は不均衡に鈍感・AP は prevalence を映す」非対称を、曲線とベースライン（=陽性割合）で説明できる。
- [ ] torchmetrics の `update→compute→reset` を回し、2 バッチに分けても全件計算と一致することを確認できる。`reset()` の必要性を言える。
- [ ] F1 最適しきい値と Youden-J 最適しきい値を**別々に**求められ、両者が一致しないことを説明できる。
- [ ] ブートストラップで指標の 95% 信頼区間を出せ、「CI が重なる2モデルは誤差の範囲で同等」と判断できる。
- [ ] 演習 `exercises.py` を 8 問すべて自力で PASS させ、各自作値が scikit-learn と一致することを確認した。

## ❓ よくある落とし穴・FAQ・デバッグ

セクション 11 に「症状 → 原因 → 対処」の早見表があります。ここではそれを補う **Q&A** と、詰まったときの **デバッグ手順** をまとめます。

**FAQ**

- **Q. macro-F1 と weighted-F1、結局どっちを報告すべき？** — 「少数クラスの出来も同じ重みで評価したい」なら macro、「全体傾向を1値で代表したい・support の偏りを反映したい」なら weighted。**どちらか一方ではなく両方**出し、論文・他者比較では必ず `average` を明記します。多クラス単一ラベルでは micro-F1 = accuracy です。
- **Q. 自作 AP が `average_precision_score` と微妙に合わない。** — ほぼ確実に**台形則で積んでいる**のが原因です。AP は短冊（ステップ）和 `Σ(recall_i − recall_{i−1})·precision_i`。`auc(recall, precision)`（台形）とは別物で、正準は前者です。`02_roc_pr_auc.py` / `ex7` の実装と見比べてください。
- **Q. しきい値はどう決めればいい？** — 目的次第です。F1 を上げたいなら F1 最適しきい値、誤受入(FPR)を一定以下に抑えたいなら「FAR を固定して閾値を探索」、ROC 上で総合的にバランスを取りたいなら Youden-J。既定 `0.5` は確率較正済みでない限り最適とは限りません（`mini_project.py` で実演）。
- **Q. ROC-AUC が高いのにモデルが使い物にならない。** — 不均衡が原因のことが多いです。ROC-AUC は順位指標で頻度に鈍感なので、**PR-AUC(AP) とベースライン（=陽性割合）を併記**し、少数クラスの recall / F1 を必ず見ます。
- **Q. torchmetrics の値がエポックをまたいで増え続ける。** — `reset()` 忘れです。エポックの頭で `metric.reset()` を呼びます。`compute()` は状態を消さない点に注意。
- **Q. `np.trapz` が無いと言われる。** — numpy 2.x で廃止されました。`np.trapezoid(y, x)` を使います（本講座は numpy 2.4）。

**デバッグの定石**

- 指標が想定とズレたら、**まず混同行列を print** する。precision と recall が入れ替わって見えるなら `axis` の取り違え（`cm.sum(0)`=列=予測側、`cm.sum(1)`=行=正解側）。
- 自作値とライブラリ値を必ず `assert np.isclose(a, b, atol=1e-6)` で突き合わせる。落ちたら「定義違い（AP のステップ和 vs 台形）」「同点スコアの扱い」「`average` の不一致」を順に疑う。
- 二値指標が `nan` になるなら、陽性 or 陰性が 0 件のバッチ（`n_pos=0` で 0 除算）を疑う。サブセット分割やしきい値掃引の端で起きやすい。
- torchmetrics で `RuntimeError`（device 不一致）が出たら、`preds.device` と `target.device` を print。CPU なら両方 `cpu` に揃える。
- 演習で FAIL が出たら、`exercises_solutions.py` を実行して**模範解答が PASS する採点条件**を確認し、自分の出力 shape / dtype / 0除算処理を見直す。

## 🚀 発展トピック・参考

この章の骨格（**混同行列 → PR 曲線 → 面積**）は、CV の評価のほぼ全てに展開できます。さらに深掘りするなら次の方向があります。

- **確率較正（calibration）と ECE** — AUC は順位の指標で、出力確率が「本当の確率」とは限りません。信頼度ビンごとに精度を比べる Reliability Diagram と ECE（Expected Calibration Error）、温度スケーリングによる較正は、しきい値設計の前提を整える重要トピックです。
- **多ラベル分類の評価** — 1 画像に複数ラベルが付く設定では、サンプル単位 / ラベル単位の平均（micro/macro/samples）や mAP（ラベルごとの AP 平均）を使います。本章の OvR 分解がそのまま土台になります。
- **コスト考慮・運用しきい値** — 誤検出と見逃しのコストが非対称なとき、期待コスト最小のしきい値や `TAR@FAR`（顔認証）など、運用 FAR を固定して閾値を決める指標を使います（`mini_project.py` のしきい値最適化の発展）。
- **統計的有意差** — ブートストラップ CI に加え、McNemar 検定（対応のある2分類器の差）や符号検定で「モデル差が偶然か」を検定できます。
- **物体検出 mAP への接続（第19回）** — 検出評価は「予測を confidence 降順に並べ、IoU マッチングで TP/FP を決め、PR 曲線→AP→クラス平均=mAP」という流れで、まさに本章の `02_roc_pr_auc.py` の構造の拡張です。`mAP@0.5` と `mAP@[.5:.95]` の違い、PASCAL 11点 / COCO 101点補間まで進みます。
- **参考ドキュメント** — scikit-learn [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) ／ torchmetrics [Classification](https://lightning.ai/docs/torchmetrics/stable/) ／ 検索・埋め込みの Recall@k は第17回、CLIP zero-shot の評価は第16回で扱います。

## 10. 動かし方

このモジュールは `numpy` / `scikit-learn` / `torch` / `torchmetrics` / `matplotlib` に依存します。画像モデルもネット接続も不要で、データは合成で自動生成されるため、依存さえ入っていればすぐ実行できます。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループを用意（初回のみ）。dl=torch, metrics=scikit-learn/torchmetrics
uv sync --group dl --group metrics

# 各スクリプトを実行（結果は outputs/14_eval_classification/ に保存される）
uv run python lectures/14_eval_classification/01_confusion_matrix_prf.py
uv run python lectures/14_eval_classification/02_roc_pr_auc.py
uv run python lectures/14_eval_classification/03_torchmetrics_vs_manual.py

# 章末ミニプロジェクト（評価レポートのダッシュボード＋JSON を生成）
uv run python lectures/14_eval_classification/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL。それでも exit 0 で落ちない）
uv run python lectures/14_eval_classification/exercises.py
# どうしても分からない時だけ、模範解答（全 PASS）を見る
uv run python lectures/14_eval_classification/exercises_solutions.py
# あるいは exercises 側の採点ロジックで模範解答を確認（採点共有）
SHOW_SOLUTION=1 uv run python lectures/14_eval_classification/exercises.py
```

実行後は `outputs/14_eval_classification/` に生成された画像と JSON を確認してください。`01_confusion_matrix.png`（対角が濃いほど良い）、`02_roc_curve.png` と `02_pr_curve.png`（不均衡 vs 均衡の曲線。PR にはベースラインの点線）、`03_multiclass_compare.png`（torchmetrics と sklearn の棒が重なる＝一致）、`mini_project_dashboard.png`（評価レポートの 6 パネル）を、本文の解説と照らし合わせると理解が定着します。各 JSON には自作とライブラリ双方の数値が記録されているので、値の一致を自分の目でも確かめられます。

## 11. よくある落とし穴（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。実装中に詰まったら、まずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| accuracy は高いのにモデルが使えない | クラス不均衡で多数派をなぞっているだけ | per-class の recall / F1、PR-AUC(AP) を併記する |
| precision と recall が逆に見える | 混同行列の `axis` を取り違え（FP は列・FN は行） | `cm.sum(0)` が列＝予測側、`cm.sum(1)` が行＝正解側 |
| 同じ「F1」なのに他者と値が違う | macro / micro / weighted の平均化が不一致 | どの `average` か必ず確認・明記する |
| `np.trapz` が無いと言われる | numpy 2.x で `trapz` 廃止 | `np.trapezoid(y, x)` を使う |
| 自作 AP が `average_precision_score` と合わない | 台形則で積んでいる（AP はステップ和） | `Σ(recall差)×precision` のステップ和で計算 |
| 不均衡なのに ROC-AUC が高くて安心してしまう | ROC-AUC は頻度に鈍感 | PR-AUC(AP) を主指標に。ベースライン=陽性割合と比較 |
| torchmetrics の値がエポックをまたいで膨らむ | `reset()` 忘れで状態が累積 | エポック頭で `metric.reset()` を呼ぶ |
| torchmetrics で `RuntimeError`（device 不一致） | `preds` と `target` の device がズレ | 両方を同じ device（CPUなら `cpu`）に揃える |

この8項目が、分類評価でつまずく原因のほぼ全てです。逆に、この8つを自分の言葉で説明でき・回避コードを書けるようになれば、この章のゴールに到達しています。

## 12. まとめ

この章では、すべての分類評価の最小単位である TP/FP/FN/TN と混同行列から出発し、precision/recall/F1 と3つの平均化、top-k accuracy、そして one-vs-rest 分解を経て、しきい値非依存の ROC-AUC と PR-AUC(=AP) までを、すべて「自分で式を書き、ライブラリと一致を検証する」レベルで扱いました。特に、不均衡で accuracy と ROC-AUC が楽観的になり、PR-AUC が実態を映すという非対称は、実務で評価設計を誤らないための核心です。

次の第15回以降は、ここで身につけた評価の目を持って、埋め込み・検索（Recall@k）や物体検出（mAP の自力実装）へ進みます。それらはどれも、本章の「混同行列 → PR 曲線 → 面積」という骨格の応用にすぎません。まずは演習を自力で全問 PASS させ、`assert` で自作とライブラリの一致を体感してから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点・CPU で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ scikit-learn 1.9.0 ／ torch 2.12.0+cpu ／ torchvision 0.27+cpu ／ torchmetrics 1.9.0 ／ transformers 5.11 ／ faiss-cpu ／ matplotlib 3.10.9。
> 本講座の評価トラックの想定スタック（2026-06 時点）は torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / faiss-cpu / scikit-learn 1.9 / torchmetrics 1.9 です（本回の計算は numpy・scikit-learn・torchmetrics で完結し、transformers・faiss は後続回で併用します）。