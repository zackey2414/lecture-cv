# 32_anomaly_iqa: 異常検知と画像品質評価 — PaDiM/PatchCore・IQA

> トラック: **異常検知・品質** ／ レベル: **上級** ／ 必要な依存: `numpy` `opencv(headless)`
> `torch/torchvision` `scikit-learn`（グループ `dl` / `metrics`）
> 前提モジュール: **15_image_embeddings_metric_learning**（埋め込みとコサイン/距離の基礎）
>
> この回は **衝突・重量級の依存を実行経路で使いません**。異常検知の定番 `anomalib`
> （PaDiM/PatchCore）や IQA の定番 `pyiqa`（BRISQUE/NIQE/LPIPS/MUSIQ）は Lightning や
> 大量の追加依存を引き込むため、**概念紹介＋任意導入**にとどめ、本体は **導入済みの
> torchvision ResNet18 と numpy/cv2/sklearn だけで PaDiM・PatchCore・各IQA指標を自前実装**
> します。合成画像のみで、**正常だけで学習 → 欠陥検出 → AUROC/AUPR/PRO 評価 → 品質ゲート**
> までを CPU で完走させます（ネットに出るのは初回の ResNet18 重み DL だけ。失敗時は
> ランダム初期化にフォールバックして必ず `exit 0`）。

---

## 🎯 この章のゴール

- **異常検知 (anomaly detection)** の基本姿勢「**正常だけを覚え、そこから外れたものを異常**」
  を説明でき、なぜ「異常を集めて分類する」のが現実的でないかを言える。
- **PaDiM** を自前で組める: 学習済み CNN の中間特徴を **位置別の埋め込み**にし、正常画像だけで
  **位置ごとの多変量ガウシアン**（平均・共分散）を推定、テスト画像を **マハラノビス距離**で採点。
- **PatchCore** を自前で組める: 正常パッチを **coreset（最遠点サンプリング）** で間引いた
  **メモリバンク**に記憶し、テストパッチの **最近傍距離**を異常スコアにする。PaDiM との違い
  （位置合わせ前提 vs 位置に寛容）を説明できる。
- 異常検知を **image-level / pixel-level の AUROC と AUPR、そして PRO** で評価でき、なぜ
  **AUROC だけでは不十分**で（特に画素は欠陥がごく少数の不均衡問題）、AUPR/PRO を併用するかを言える。
- しきい値を **正常スコアの分位点**からデータドリブンに決められる。
- **画像品質評価 (IQA)** で **参照あり (PSNR/SSIM)** と **無参照 (variance of Laplacian 等)** を
  自前計算でき、指標ごとの **lower_better（良し悪しの向き）** を取り違えない。生成(31)・超解像(29)・
  復元の評価にどう接続するかを説明できる。
- 完成形として「**撮影品質ゲート(無参照IQA) → 異常検知(PaDiM) → AUROC/AUPR/PRO 評価 →
  合否判定**」の外観検査ラインを CPU・合成データだけで動かせる。

---

## 1. 直感 — 異常検知は「正常だけを覚える片側の学習」

製造ラインの外観検査を思い浮かべてください。良品（正常）はいくらでも撮れますが、**欠陥
（キズ・異物・汚れ）は珍しく、種類も無限**です。「キズ」と一口に言っても太さ・向き・場所が
毎回違い、明日には見たこともない欠陥が出るかもしれません。だから「欠陥画像を集めて分類器を
学習する」という素直なアプローチは破綻します。代わりに異常検知が採るのは **片側の学習**——
**正常がどんな見た目かだけを覚え、その分布から外れたものを一律に『異常』**とする考え方です。
正常さえ十分あれば、未知の欠陥にもゼロショットで反応できます。

「正常を覚える」とはどういうことか。最も素朴には「正常画像の集合が作る確率分布を推定し、
新しい画像の尤度が低ければ異常」とします。しかし画像そのものは高次元すぎて分布推定が難しい。
そこで本章の主役 **PaDiM / PatchCore** は、**学習済み CNN の中間特徴**という「意味の詰まった
低次元表現」の上で分布を考えます。ImageNet で学んだ ResNet は、エッジ・テクスチャ・模様と
いった汎用的な視覚パターンを特徴に符号化しているので、正常テクスチャはまとまった塊になり、
欠陥はそこから外れた特徴を生みます。**特徴空間での外れ値検出**に持ち込むのが両手法の核です。

もう一つ重要なのは **「どこが異常か（pixel-level）」と「異常か否か（image-level）」は別の問い**
だということです。検査では「この製品は不良」（image）だけでなく「キズはここ」（pixel）まで
欲しい。PaDiM/PatchCore はどちらも **位置ごとにスコアを出す異常マップ**を作り、その最大値を
画像スコアにすることで両方に答えます。本章はこの異常マップを軸に、可視化と評価を組み立てます。

---

## 2. 特徴の土台 — 学習済み CNN の中間特徴を「位置別の埋め込み」にする

PaDiM も PatchCore も、出発点は同じ **「学習済み CNN の中間特徴マップ」** です。ResNet18 に
画像を通すと、`layer1`（解像度 H/4・64ch）、`layer2`（H/8・128ch）、`layer3`（H/16・256ch）…
と、浅いほど細かく・深いほど意味的な特徴が得られます。これらを **最も細かい layer1 の解像度に
そろえて（深い層は最近傍補間で拡大）チャネル方向に連結**すると、空間の各位置 (i, j) が
「64+128+256 = 448 次元のベクトル」を持つ、**位置別の埋め込み**になります。`anomaly_iqa_lab.py`
の `FeatureExtractor` は forward hook で各層の出力を捕まえ、これを `(N, Hc, Wc, C)` の numpy で
返します（128px 入力なら `Hc=Wc=32`）。

実装で大事なのは **`model.eval()` と `torch.inference_mode()`**、そして **前処理を学習時と
そろえる**ことです。ResNet は ImageNet の平均・標準偏差で正規化された入力を前提にしているので、
`(x/255 - mean) / std` を必ず通します（これを忘れると特徴が崩れて分布推定が無意味になります）。
本章は CPU 前提なので、特徴抽出は小バッチ（8枚）で回し、`torch.set_num_threads` でスレッドを
抑えて再現性を上げています。重みの DL に失敗しても **`weights=None`（ランダム初期化）に
フォールバック**して同じ経路で完走します——合成の整列済みテクスチャ＋局所欠陥という設定では、
ランダム初期化でも位置の一貫性が効いてそこそこ検出できます（品質は当然 pretrained が上）。

なぜ「中間層」で「複数層を連結」するのか。**深すぎる層（layer4 等）は意味が抽象的すぎて
位置情報が薄れ、浅すぎる層は意味が乏しい**。PaDiM の原論文は layer1〜3 の連結が良いバランスだと
示しました。多層を混ぜることで「細かいキズ（浅い層が反応）」も「広い色ムラ（深い層が反応）」も
拾えます。次節からは、この `(N, Hc, Wc, C)` 埋め込みを PaDiM と PatchCore がどう料理するかを見ます。

---

## 3. PaDiM — 位置別ガウシアン + マハラノビス距離（自前実装）

**PaDiM (Patch Distribution Modeling)** の発想はシンプルです。正常画像が **おおよそ位置合わせ
されている**（同じ製品を同じ構図で撮る）なら、**各位置 (i, j) には決まった見た目の特徴ベクトルが
来るはず**。そこで「位置 (i, j) の埋め込みベクトルは多変量ガウシアン分布に従う」と仮定し、正常
画像 N 枚から **各位置ごとに平均ベクトル μ_{ij}（C次元）と共分散行列 Σ_{ij}（C×C）** を推定します。
テスト画像の位置 (i, j) のベクトル x に対し、その位置のガウシアンからの **マハラノビス距離**
`d = √((x-μ)ᵀ Σ⁻¹ (x-μ))` を異常スコアにします。マハラノビス距離は「各方向のばらつきで割った
距離」なので、**正常がよく動く方向のズレは小さく、動かない方向のズレは大きく**評価されます。

正準的な実装の流れは `padim_fit` / `padim_score` の通りです。共分散は次元 C が大きいと
推定も逆行列も不安定なので、PaDiM は **チャネルをランダムに 100 本だけ間引いて**から学習します
（`select_dims`。ランダム選択でも経験的に十分という論文の知見）。さらに **共分散に小さな
`εI` を足して正則化**し、特異化を防いでから `np.linalg.inv` で逆行列を取ります。位置ごとに
1024 個（32×32）の `100×100` 逆行列が要りますが、numpy のバッチ `inv` と `einsum` でまとめて
計算でき、CPU でも数秒で終わります。

```python
# PaDiM の核（anomaly_iqa_lab.py 抜粋・概念）
mean = e.mean(axis=0)                                   # (P, C) 位置別平均
cov  = np.einsum("npi,npj->pij", e-mean, e-mean)/(N-1)  # (P, C, C) 位置別共分散
cov += eps * np.eye(C)                                  # ★ 正則化（特異化を防ぐ）
inv  = np.linalg.inv(cov)                               # (P, C, C) 一括逆行列
# スコア: maha^2 = diff^T inv diff を位置ごとに einsum で一括
left = np.einsum("npi,pij->npj", diff, inv)
maha = np.sqrt(np.clip(np.einsum("npj,npj->np", left, diff), 0, None))
```

PaDiM の **強みは軽さと解釈性**（位置ごとのガウシアンという明快なモデル、学習は平均・共分散の
推定だけ）。**弱みは「位置合わせ前提」**で、被写体が動く・回る・拡縮するデータには弱い（位置
(i, j) の意味がぶれるとガウシアンが緩み、感度が落ちる）。`01_padim_anomaly.py` を実行すると、
合成欠陥に対し image-AUROC≈1.0・pixel-AUROC≈0.99 が出て、異常マップが欠陥位置に綺麗に灯ります。
**欠陥を一切学習していないのに位置まで当てられる**——ここが異常検知の面白さです。

---

## 4. PatchCore — 正常パッチのメモリバンク + 最近傍距離（coreset）

**PatchCore** は同じ中間特徴を使いますが、分布を **パラメトリックに仮定しません**。代わりに
**正常画像から取れたパッチ埋め込みを全部「記憶」**しておき、テストパッチに対して **記憶の中の
最近傍までの距離**を異常スコアにします（=最近傍密度推定）。「正常パッチの近くにいれば正常、
どの正常パッチからも遠ければ異常」という、ノンパラメトリックで直感的な方式です。位置ごとの
ガウシアンを持たないので、**位置合わせを強く要求しない**のが PaDiM との大きな違いです。

素朴にやると正常パッチは膨大（20枚×32×32 = 約2万本）になり、最近傍探索が重い。そこで
PatchCore は **coreset サブサンプリング**でメモリバンクを代表点だけに圧縮します。本章は
その定番 **最遠点サンプリング (FPS, greedy)** を `coreset_subsample` で実装: 「すでに選んだ点
集合への最小距離が最大の点」を貪欲に選び続け、空間を**まんべんなく覆う**少数の点を残します
（ランダム間引きより端の正常パターンを取りこぼしにくい）。テストは `patchcore_score` で
`||q-b||² = ||q||² + ||b||² - 2 q·b` を使った行列演算で最近傍距離をブロック計算します。

```python
# PatchCore の核（概念）
bank = normal_patches              # (M0, C) 正常パッチを全部集める
idx  = coreset_subsample(bank, n_keep=800)   # ★ FPS で代表点だけに圧縮
mem  = bank[idx]                   # (M, C) メモリバンク
# スコア: 各テストパッチ → メモリバンク最近傍までの距離
d2   = (q**2).sum(1,keepdims=True) + (mem**2).sum(1) - 2*q@mem.T
score = np.sqrt(d2.min(axis=1))    # (位置ごと) 最近傍距離
```

`02_anomaly_eval.py` は PaDiM と PatchCore を同じデータで比べます。両者とも image-AUROC≈1.0、
pixel も高い値が出ますが、**実運用では「位置が安定するなら PaDiM が軽くて速い」「位置や姿勢が
ばらつく・正常の多様性が高いなら PatchCore が頑健」** という住み分けになります（PatchCore は
MVTec AD のような実データで PaDiM を上回ることが多い一方、メモリと最近傍コストが増えます）。
本章の PatchCore は教育目的の簡易版（厳密な近傍プーリングや re-weighting は省略）です。

---

## 5. 評価 — image/pixel の AUROC・AUPR・PRO の読み方と使い分け

異常検知の評価は **しきい値を 1 つに決めずに、全しきい値を見渡す曲線下面積**で行います。
**AUROC**（ROC 下面積）は「ランダムに選んだ異常が、ランダムに選んだ正常より高いスコアになる
確率」で、1.0 が完璧・0.5 が偶然。**image-level** は「画像が異常か」を、**pixel-level** は
「各画素が欠陥か」を、それぞれスコアと正解で測ります。`sklearn.metrics.roc_auc_score` に
スコアとラベルを渡すだけです（pixel は全画像の全画素を平らに並べる）。

しかし **AUROC には落とし穴**があります。**正例（欠陥）がごく少数の不均衡なデータでは、
AUROC は高く出やすい**。画素評価では欠陥画素は全体の 1% 程度しかなく（本章の合成でも
`pixel_defect_rate≈0.01`）、AUROC が 0.99 でも「正常画素を少し誤検出しただけで欠陥画素を
取りこぼしている」状態が隠れます。そこで併用するのが **AUPR**（Precision-Recall 曲線下面積＝
average precision, `average_precision_score`）。AUPR は **正例が希少なほど厳しく**なり、
偶然レベルは「base rate（正例率）」まで下がるので、不均衡下の実態を映します。本章でも
pixel-AUROC≈0.99 に対し **pixel-AUPR≈0.6 と大きく下がり**、「小欠陥の取りこぼし」を露わにします。

```bash
uv run python lectures/32_anomaly_iqa/02_anomaly_eval.py
# → [PaDiM]    img AUROC=1.000 AUPR=1.000 | px AUROC=0.994 AUPR=0.606 | PRO=0.872
#   [PatchCore]img AUROC=1.000 AUPR=1.000 | px AUROC=0.989 AUPR=0.555 | PRO=0.779
#   02_roc_pr.png（ROC と PR 曲線）, 02_summary.json
```

もう一つの定番が **PRO (Per-Region Overlap)**。pixel-AUROC は「画素単位の平均」なので **大きな
欠陥に引きずられ**、面積の小さい欠陥が無視されがちです。PRO は **欠陥領域（連結成分）ごとに
被覆率を計算し、領域の大きさに依らず平等に平均**するので、小さな欠陥も同じ重みで評価できます。
本章の `pro_auc` は複数しきい値での (FPR, 平均PRO) を作り、`FPR ≤ 0.3` の範囲で積分して
[0,1] に正規化した簡易版です。**「image-AUROC で見つけられるか、pixel-AUROC/AUPR/PRO でどこまで
正確に当てられるか」を別軸で読む**のが、異常検知評価の実務です。

---

## 6. しきい値の決め方 — 正常スコアの分位点でデータドリブンに

曲線で全体像を掴んだら、運用では **1 つのしきい値**を決めて合否を出します。素朴には「正常の
最大スコアと異常の最小スコアの中点」ですが、これは **異常ラベルが必要**で、運用時に未知の
欠陥には使えません。実務でよく使うのは **「正常品スコアの分位点」**——例えば正常スコアの
**99 パーセンタイル**をしきい値にすれば、「正常品の 1% だけを異常側に倒す（＝偽陽性率 1%
を許容する）」という運用要件をデータから直接決められます（`mini_project.py` の `choose_threshold`）。

しきい値の選択は **常に「2 種類の誤りのトレードオフ」**です。しきい値を下げれば欠陥を見逃さない
（recall ↑）が、正常品を不良と誤る（false positive ↑）。上げれば逆。**どちらの誤りが高コストか**
で動作点を決めます——「不良流出が致命的（医療・自動車部品）」なら recall を最優先してしきい値を
下げ、過検出は人手の再検査で吸収する。「過検査のコストが高い」なら precision を上げる。同じ
異常スコアでも、用途ごとに動作点を変えるのが運用の勘所です。

```bash
uv run python lectures/32_anomaly_iqa/01_padim_anomaly.py
# → image-level AUROC=1.000 / pixel-level AUROC=0.994
#   しきい値での image 判定 正解率=1.000（合成は分離が容易で満点になりやすい）
#   01_anomaly_maps.png（入力/GT/異常マップ/重ね）, 01_score_hist.png
```

合成データは分離が容易で満点になりがちですが、**実データでは正常/異常スコアが必ず重なり**、
その重なり領域が「どうしきい値を引いても避けられない誤り」になります。その重なりこそ改善の
余地（より良い特徴・より整った位置合わせ・PatchCore への切替）であり、AUROC/AUPR はその
重なりの大きさを定量化している、と理解してください。

---

## 7. 画像品質評価(IQA) 概論 — 参照あり / 無参照 と lower_better

ここから後半は **画像品質評価 (Image Quality Assessment, IQA)** です。生成(31)・超解像(29)・
復元（デノイズ/インペイント）を作ると、必ず「**この出力はどれだけ良いか**」を数値化したくなる。
IQA はその物差しで、大きく 2 系統に分かれます。**参照あり (full-reference)** は「正解のクリーン
画像」と比べて測る方式（PSNR・SSIM・LPIPS）。**無参照 (no-reference / blind)** は参照なしで
1 枚から品質を推定する方式（BRISQUE・NIQE・MUSIQ、本章では分散ラプラシアン等の鮮鋭度）。

どちらを使うかは **「正解画像が手に入るか」**で決まります。超解像の評価は高解像の正解があるので
参照あり（PSNR/SSIM）が使えます。一方、**実運用の撮影品質ゲート**（ボケた写真を弾く）や、
正解の存在しない生成画像の単体評価には参照がないので、無参照を使います。両者は補完関係で、
研究では参照あり、運用では無参照、という組み合わせがよくあります。

**最大の落とし穴は「指標の良し悪しの向き」を取り違えること**です。PSNR・SSIM・鮮鋭度は
**高いほど良い**が、LPIPS・BRISQUE・NIQE は **低いほど良い**（歪み/不自然さの量）。pyiqa では
`metric.lower_better` という属性でこれを管理します。本章は同じ思想で `METRIC_DIRECTION` という
辞書（`True=小さいほど良い`）を持ち、`03_iqa_metrics.py` がこれを表示します。**複数指標を
混ぜてランキングするときは、まず向きをそろえる**（lower_better のものは符号反転する）のが鉄則です。

---

## 8. 参照あり PSNR/SSIM の中身（自前実装）

**PSNR (Peak Signal-to-Noise Ratio)** は最も基本的な参照あり指標で、`PSNR = 10·log10(MAX² / MSE)`
（MAX は画素の最大値、MSE は参照との平均二乗誤差）。**画素誤差の対数**なので、誤差が小さいほど
大きな値（dB）になります。実装は数行（`psnr`）。直感的で計算も軽い反面、**人間の知覚と乖離**
しやすい——例えば全体を 1 画素ずらした画像は見た目ほぼ同じでも PSNR は大きく落ちます。「画素が
合っているか」しか見ていないからです。

**SSIM (Structural Similarity)** はこの弱点を補い、**局所の明るさ・コントラスト・構造の一致**を
測ります。窓（ガウシアン 11×11）ごとに参照と出力の平均 μ・分散 σ²・共分散 σ_xy を求め、
`SSIM = ((2μ_xμ_y+C1)(2σ_xy+C2)) / ((μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2))` を画像全体で平均します。
本章は **skimage を使わず cv2.GaussianBlur だけ**で μ・σ²・σ_xy を計算して実装しています
（`ssim`）。C1・C2 は分母がゼロに近いときの安定化項です。SSIM は 1.0 が完全一致で、PSNR より
**知覚に近い**評価になりますが、それでも人間の見え方とは完全には一致しません。

```bash
uv run python lectures/32_anomaly_iqa/03_iqa_metrics.py
#   variant           PSNR↑   SSIM↑   varLap↑   Teneng↑
#   clean             99.00   1.000    0.0065    0.2517
#   gaussian_noise    20.21   0.767    0.0952    0.3525   ← ノイズで鮮鋭度が“上がる”
#   blur              25.27   0.827    0.0004    0.0843   ← ボケで鮮鋭度が下がる
#   downscale         23.14   0.687    0.0233    0.2596
```

知覚により近い参照あり指標として **LPIPS**（学習済み CNN 特徴の距離、`lower_better=True`）が
あり、生成評価でよく使われます。LPIPS は重みが要るので本章では概念紹介（§10・任意）にとどめ、
PSNR/SSIM を自前で押さえます。**「PSNR は画素、SSIM は構造、LPIPS は知覚」**という守備範囲の
違いを覚えておくと、論文の評価表が読めるようになります。

---

## 9. 無参照 IQA と、生成/超解像の評価への接続

参照が無い状況では、**画像そのものの統計**から品質を推定します。本章が実装する最も簡単な
無参照指標は **variance of Laplacian（ラプラシアン応答の分散）** と **Tenengrad（Sobel 勾配
エネルギー）** で、どちらも **鮮鋭度（ボケていないか）** を測ります（高いほど鮮鋭）。ピントの
合った画像はエッジが強く、ラプラシアン/勾配の応答が大きくばらつくので分散が大きい。逆に
ボケると応答が平坦になり値が下がる——この性質で **参照なしにボケを検出**できます（撮影品質
ゲートの定番）。

ただし **無参照は万能ではありません**。`03` の表が示すように、**ガウシアンノイズを加えると
鮮鋭度（varLaplacian）はむしろ上がります**——ノイズが高周波成分を増やすからです。つまり
鮮鋭度指標は「ボケ」は検出できても「ノイズ」を品質低下と見なせない。だから本格的な無参照
IQA（BRISQUE・NIQE）は **自然画像の統計モデル**を学習し、ノイズ・圧縮・ボケなど多様な歪みを
まとめて「自然さからの逸脱」として捉えます（`lower_better=True`）。MUSIQ のような学習型は
逆に「高いほど良い」スコアを出します。**指標ごとに測れる歪みと向きが違う**ことが要点です。

この章が IQA を扱うのは、**31（生成）・29（超解像）・復元の評価に直結**するからです。超解像
なら高解像の正解があるので PSNR/SSIM（＋LPIPS）。生成画像の集合なら参照分布との距離を測る
**FID**（概念は 31 回）。実運用のゲートなら無参照の鮮鋭度。`mini_project.py` は最後の用途を
実演し、**無参照IQA を外観検査の前段ゲート**（ボケた撮影を異常検知に回す前に弾く）として
組み込みます。「評価指標は目的とデータ（参照の有無）で選ぶ」——これが IQA の実務の結論です。

---

## 10. anomalib / pyiqa — 概念と任意導入（実行経路では使わない）

実データで本気の異常検知をするなら **anomalib**（PaDiM/PatchCore/FastFlow/EfficientAD 等を
統一 API で提供）が定番です。`Engine(accelerator="cpu")` で CPU 推論でき、MVTec AD などの
標準データセットとベンチマークが揃っています。本章が実行経路に入れないのは、**PyTorch
Lightning や多数の追加依存を引き込み**、他グループと衝突・肥大化しやすいからです。試すときは
**専用グループに隔離**してください。本章の自前 PaDiM/PatchCore は、その中身を理解したうえで
anomalib に乗り換えるための土台です。

```bash
# 任意（衝突回避のため隔離グループ推奨）。CPU 推論は Engine(accelerator="cpu")。
uv add --group anomaly anomalib
```

```python
# 概念コード（本章の実行経路には含めない）。
from anomalib.models import Padim          # or Patchcore
from anomalib.engine import Engine
engine = Engine(accelerator="cpu")
engine.fit(model=Padim(), datamodule=dm)   # 正常のみで学習
engine.test(model=Padim(), datamodule=dm)  # image/pixel AUROC を自動算出
```

IQA の定番は **pyiqa**（`pyiqa.create_metric("lpips")` 等で 30+ 指標を統一 API、`metric.lower_better`
で向きを確認）。LPIPS・BRISQUE・NIQE・MUSIQ・TOPIQ などを一括で計算でき、復元/生成の評価を
一段引き上げます。本章は重み DL と依存を避けて PSNR/SSIM/鮮鋭度を自前実装しましたが、
**本番の品質評価では pyiqa を隔離グループで導入**するのが実用的です。

```bash
uv add --group iqa pyiqa     # 任意。LPIPS/BRISQUE/NIQE/MUSIQ などを一括計算
```

```python
import pyiqa
metric = pyiqa.create_metric("brisque")   # 無参照
print(metric.lower_better)                  # True（小さいほど良い）← 本章の METRIC_DIRECTION と同じ思想
```

---

## 🛠 章末ミニプロジェクト — 外観検査ライン（品質ゲート → 異常検知 → 評価 → 合否）

`mini_project.py` は本章の全工程を 1 本に統合した完成形です。現実の検査ラインを模します。

1. **撮影品質ゲート（無参照IQA）**: 正常品の鮮鋭度(varLaplacian)分布から下限
   `mean − 3·std` を決め、それ未満の画像を「ボケ＝要再撮影」として弾く（異常検知の前段）。
   デモのため評価セットにわざとボケ画像を 1 枚混ぜ、弾けることを確認する。
2. **異常検知（PaDiM・自前）**: 正常品だけで位置別ガウシアンを学習し、欠陥品とその位置を検出。
3. **評価**: image/pixel の AUROC・AUPR・PRO を一括算出。
4. **しきい値**: image スコアの **正常 99 パーセンタイル**をしきい値にして合否判定し、不良検出の
   **recall / precision** を出す。
5. **出力**: 合否ラベル付きの検査画像・異常マップ・全指標 JSON を保存。
6. **頑健性**: ResNet18 の重み DL に失敗してもランダム初期化にフォールバックして必ず `exit 0`。

```bash
uv run python lectures/32_anomaly_iqa/mini_project.py
# → 品質ゲート: 鮮鋭度下限=0.0040 → 要再撮影 1 枚（混入したボケ画像を検出）
#   評価: img AUROC=1.000/AUPR=1.000 | px AUROC=0.994/AUPR=0.606 | PRO=0.872
#   しきい値（正常99%点）→ 不良検出 recall=1.000 precision=0.933
#   mini_inspection.png / mini_summary.json
```

**発展課題**: ① PaDiM を §10 の anomalib(PatchCore/EfficientAD) に差し替えて実データ(MVTec AD)で
比べる。② `make_dataset` の欠陥を小さく/薄くして、pixel-AUROC は高いまま AUPR/PRO が落ちる様子を
観察する（不均衡の体感）。③ しきい値の分位点を 95/99/99.9% で振り、recall と precision の
トレードオフをプロットする。④ 品質ゲートを pyiqa の BRISQUE に替える。部品（特徴抽出 / PaDiM /
PatchCore / 評価 / IQA）が分かれているので差し替えが容易です。

---

## ✅ 到達チェックリスト

- [ ] 異常検知が「正常だけを覚える片側の学習」である理由を説明できる。
- [ ] 学習済み CNN の中間特徴を **位置別埋め込み**にする手順（層連結・解像度そろえ・正規化）を言える。
- [ ] PaDiM を **位置別ガウシアン + マハラノビス距離**として説明し、次元削減と共分散の正則化の
      役割を言える。
- [ ] PatchCore を **メモリバンク + 最近傍距離**として説明し、**coreset(最遠点サンプリング)** の
      目的を言える。PaDiM との使い分け（位置合わせ前提 vs 寛容）を説明できる。
- [ ] **image/pixel AUROC** を `roc_auc_score` で計算でき、**不均衡だと AUROC が高く出やすい**ことと
      **AUPR/PRO を併用**する理由を説明できる。
- [ ] **PRO** が「領域の大きさに依らず平等に被覆率を平均する」指標だと説明できる。
- [ ] しきい値を **正常スコアの分位点**から決められ、recall/precision のトレードオフを語れる。
- [ ] IQA の **参照あり/無参照** の違いと、PSNR/SSIM の中身を自分の言葉で説明できる。
- [ ] **lower_better（指標の向き）** を取り違えず、複数指標を混ぜるときに向きをそろえられる。
- [ ] 無参照鮮鋭度が **ボケは検出できるがノイズは品質低下と見なせない**ことを説明できる。
- [ ] 品質ゲート → 異常検知 → 評価 → 合否 のラインを最後まで動かし、結果を読める。

---

## ❓ 落とし穴・FAQ・デバッグ

- **PaDiM の AUROC が低い / 異常マップが灯らない**: ほぼ「位置合わせがずれている」か「前処理を
  間違えている」。PaDiM は正常画像が整列している前提。被写体が動くデータなら PatchCore を使う。
  ImageNet 正規化（mean/std）を通し忘れると特徴が崩れる。
- **`np.linalg.inv` が `LinAlgError`（特異行列）**: 共分散の正則化 `εI` が小さすぎ/無い。
  `padim_fit` の `eps` を上げる。次元 C をサンプル数 N より十分小さく（`select_dims` の n_dims を
  下げる）。
- **pixel-AUROC は高いのに AUPR が低い**: 仕様どおり。欠陥画素が希少（不均衡）なので AUROC は
  甘い。**AUPR/PRO で小欠陥の取りこぼしを点検**するのが正しい読み方。
- **`np.trapz` が AttributeError**: numpy 2.x で削除された。本章は `np.trapezoid` を使う（PRO の
  積分）。古い記事の `np.trapz` は `np.trapezoid` に読み替える。
- **IQA でノイズ画像の鮮鋭度が上がって戸惑う**: 正常な挙動。鮮鋭度(varLaplacian/Tenengrad)は
  高周波を測るのでノイズで上がる。ノイズも品質低下として測るなら BRISQUE/NIQE（pyiqa, 任意）。
- **PSNR が `inf` / 異常に大きい**: 参照と完全一致で MSE=0。本章は便宜上 99.0 でクリップしている。
- **指標の良し悪しを取り違える**: PSNR/SSIM/鮮鋭度は高いほど良い、LPIPS/BRISQUE/NIQE は低いほど
  良い。`METRIC_DIRECTION`（pyiqa の `lower_better`）で必ず確認。混ぜてランク付けする前に向きをそろえる。
- **ResNet18 の重み DL に失敗（オフライン）**: `FeatureExtractor` が `weights=None`（ランダム
  初期化）にフォールバックして完走する（`mode=random-init` と表示）。品質は落ちるが経路は同じ。
- **`cv2` の色順 (BGR/RGB)**: 本章の合成画像は内部で RGB を一貫使用し、cv2 への受け渡しでは
  グレースケール変換時に `COLOR_RGB2GRAY` を使っている。実画像を読むときは BGR→RGB に注意。
- **matplotlib の図で日本語が豆腐(□)になる**: 既定フォントに日本語が無いため。本章は図中の文字を
  英語にして回避している（コンソール出力は日本語のまま）。
- **CPU で遅い**: 入力 128px・layer1〜3・N=20 で数秒に収まる設計。大きくするなら `IMG_SIZE` を
  下げる、`select_dims` の次元を減らす、`patchcore` の `n_keep` を減らす。

---

## 🚀 発展トピック・参考

- **メモリバンク型の系譜**: PaDiM(Defard+ 2020) → PatchCore(Roth+ 2022, coreset+局所近傍プーリング)
  → EfficientAD（高速・高精度）。実データ(MVTec AD)では PatchCore/EfficientAD が定番。
- **正規化フロー型・再構成型**: FastFlow/CFLOW（特徴の正規化フローで尤度評価）、オートエンコーダ/
  拡散モデルで「正常を再構成 → 残差を異常」とする再構成ベースも一系統。
- **PRO と評価**: MVTec AD は image/pixel AUROC に加え **AUPRO**（FPR 上限で正規化した PRO）を標準
  採用。本章の `pro_auc` はその簡易版。
- **IQA の系譜**: 参照あり PSNR→SSIM→MS-SSIM→**LPIPS**（学習特徴）。無参照 BRISQUE/NIQE（統計）→
  **MUSIQ/TOPIQ/MANIQA**（Transformer）。生成評価は **FID/KID**（分布距離, 31 回）。
- **ライブラリ**: 異常検知は `anomalib`、IQA は `pyiqa`（`metric.lower_better`）が実用の入口（§10・任意）。
- 参考: PaDiM (Defard+ 2020) ICPR、PatchCore (Roth+ 2022) CVPR、SSIM (Wang+ 2004) IEEE TIP、
  LPIPS (Zhang+ 2018) CVPR、MVTec AD (Bergmann+ 2019) CVPR。

---

## 💡 実践ユースケース集

本章の「正常だけで覚える異常検知」と「IQA」は、そのまま現場の小ツールになります。
代表的な 3 つを挙げます（1 つ目は動く出発点 `use_case.py`）。

### ① 外観検品ツール（`use_case.py`・動く出発点）

- **何に使うか**: 「良品フォルダ」を渡すだけで覚え、検査フォルダの画像を 1 枚ずつ
  **OK/NG 判定 + 異常ヒートマップ + 欠陥位置（bbox）** にし、`inspection_report.csv/json`
  （1 行 = 1 製品）を吐く現場寄りの検品 CLI。製造ラインの外観検査・到着検品の叩き台。
- **mini_project.py との違い**: ミニプロジェクトは**ラベル付き評価セットで AUROC/AUPR/PRO を
  測る学習デモ**。`use_case.py` は**ラベル不要**で、未知の良品を取り置く **hold-out 較正**で
  しきい値を決め、評価指標ではなく「1 個ずつの判定と成果物（レポート/画像）」を出します。
- **作り方の要点**: 良品で `padim_fit` → 良品の一部を学習に使わず取り置き、その素点 ×
  マージンをしきい値に（不良サンプルが要らない）→ 検査画像を `padim_score` → 異常マップの
  最大値が境界を超えたら NG、高反応領域を bbox 化。

```bash
# 既定（引数なし）は合成データで完走。実画像は data/32_anomaly_iqa/ に置けば自動で使う
uv run python lectures/32_anomaly_iqa/use_case.py
# 自分のデータで検品（フォルダ指定・しきい値マージン調整）
uv run python lectures/32_anomaly_iqa/use_case.py \
    --train-dir data/32_anomaly_iqa/train --test-dir data/32_anomaly_iqa/test --margin 1.3
```

- **data/ の置き方**: `data/32_anomaly_iqa/train/` に**良品（正常）だけ**、
  `data/32_anomaly_iqa/test/` に**検査したい画像**（OK/NG 混在可）を置く（`.png/.jpg/.jpeg/.bmp/.webp`）。
  両方に画像が無ければ合成にフォールバックして必ず `exit 0`。出力は
  `outputs/32_anomaly_iqa/usecase_overlays/*.png`・`usecase_contact_sheet.png`・`inspection_report.csv/json`。
- **拡張アイデア**: ① しきい値較正を分位点 / 中央値+MAD のロバスト統計に変える ②
  PaDiM を PatchCore（`lab.patchcore_fit/score`）へ差し替え位置ずれに強くする ③ 入口に
  撮影品質ゲート（`lab.variance_of_laplacian`）を足しボケ画像を検査前に弾く ④ 欠陥 bbox を
  COCO/YOLO 形式で書き出し二次アノテーションの種にする。
- **注意**: PaDiM は**良品が位置合わせ済み**（同じ製品を同じ構図）を前提にする。被写体が
  動く/回るデータでは精度が落ちるので PatchCore へ。しきい値は良品の分布で決まるため、
  撮影条件が変わったら**再較正**が必要。合成は分離が容易で満点が出やすい点も忘れずに。

### ② 撮影品質ゲート（無参照 IQA で「検査に値する写真か」を弾く）

- **何に使うか**: 異常検知や OCR・検出の**前段**で、ボケ・手ブレ・露出不良の写真を自動で
  リジェクトし「撮り直し」に回す。ゴミ画像を下流に流さないことで誤検出を減らす。
- **作り方の要点**: 良品の鮮鋭度 `variance_of_laplacian`（または `tenengrad`）の分布から
  下限 `mean − k·std` を決め、それ未満を「要再撮影」に（`mini_project.py` の `quality_gate` が雛形）。
- **注意**: 鮮鋭度は**ボケは検出できてもノイズは品質低下と見なせない**（ノイズで値が上がる）。
  ノイズ/圧縮も弾くなら BRISQUE/NIQE（pyiqa・§10 の任意導入）に拡張する。

### ③ 生成・超解像パイプラインの自動品質チェック（参照あり/なし IQA）

- **何に使うか**: 超解像（29）や生成・復元（31）の出力を**人手レビュー前に自動採点**し、
  PSNR/SSIM が基準未満のものだけを目視に回す CI 的なゲート。バッチ生成の品質を定量監視。
- **作り方の要点**: 正解（高解像/クリーン）があれば `psnr`/`ssim`（参照あり）、無ければ
  鮮鋭度（無参照）でスコア化。`METRIC_DIRECTION`（pyiqa の `lower_better` 相当）で**向きを
  そろえて**から合否しきい値を引く。
- **注意**: 指標ごとに測れる歪みと**良し悪しの向き**が違う。PSNR/SSIM は知覚と乖離するため、
  生成評価には LPIPS（参照あり・低いほど良い）や分布距離 FID（31）を併用する。

---

## ▶ 動かし方

```bash
# 共有ヘルパの自己テスト（合成→特徴→PaDiM→評価→IQA が一通り動く）
uv run python lectures/32_anomaly_iqa/anomaly_iqa_lab.py
# 1) PaDiM 異常検知（正常だけで学習 → マハラノビス距離 → 異常マップ）
uv run python lectures/32_anomaly_iqa/01_padim_anomaly.py
# 2) 評価（image/pixel AUROC・AUPR・PRO / PaDiM vs PatchCore）
uv run python lectures/32_anomaly_iqa/02_anomaly_eval.py
# 3) 画像品質評価(IQA)（参照あり PSNR/SSIM・無参照 鮮鋭度・lower_better）
uv run python lectures/32_anomaly_iqa/03_iqa_metrics.py
# 章末ミニプロジェクト（品質ゲート → 異常検知 → 評価 → 合否）
uv run python lectures/32_anomaly_iqa/mini_project.py
# 実践ユースケース: PaDiM 外観検品ツール（良品で覚えて OK/NG + ヒートマップ + 検品レポート）
uv run python lectures/32_anomaly_iqa/use_case.py
# 演習（自己採点）と模範解答
uv run python lectures/32_anomaly_iqa/exercises.py
uv run python lectures/32_anomaly_iqa/exercises_solutions.py
```

出力（可視化・図・JSON）は `outputs/32_anomaly_iqa/` に保存されます。すべて **CPU・合成データ**で
完結し、ネットに出るのは **初回の ResNet18 重み DL のみ**（失敗時はランダム初期化へ自動フォール
バック）。実画像を `data/32_anomaly_iqa/` に置く拡張も想定した構造にしてあります。

---

> 参照ライブラリ（版）: torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 /
> diffusers 0.38（本章では未使用・関連は 31）/ opencv-python-headless 4.13 / scikit-learn 1.9 /
> numpy 2.x。異常検知の `anomalib`（PaDiM/PatchCore）と IQA の `pyiqa`（BRISQUE/NIQE/LPIPS/MUSIQ）は
> 依存衝突・肥大化を避けて **実行経路では使わず**、概念紹介＋任意導入（`uv add --group anomaly ...`
> / `--group iqa ...`）にとどめ、本体は torchvision ResNet18＋numpy/cv2/sklearn で自前実装しています。
> — 2026-06
