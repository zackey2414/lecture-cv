# 30_face_detection_recognition: 顔の検出・認識・人物クラスタリング

> トラック: **深度・姿勢・動き** ／ レベル: **中級** ／ 主要依存: `numpy` `opencv(headless)`
> `torch/torchvision` `scikit-learn`（グループ `dl` / `hf` / `metrics`）
> 前提モジュール: **15_image_embeddings_metric_learning**（埋め込みとコサイン類似の基礎）
>
> この回は **衝突依存を実行経路で使いません**。顔埋め込みの定番 `insightface(ArcFace)` や
> `mediapipe` は onnxruntime / numpy ピン等で衝突しがちなので **概念紹介＋任意導入** にとどめ、
> 本体は **導入済みの torchvision ResNet18 を「顔埋め込みの代用」** にして、検出 → 整列 →
> 埋め込み → 照合/識別 → クラスタリングまでを **合成顔だけで完走** させます。`FaceEncoder`
> を 1 つ差し替えれば、そのまま ArcFace 実装に乗り換えられる構造にしてあります。

---

## 🎯 この章のゴール

- 顔認識の三段構え **検出(detection) → 整列(alignment) → 埋め込み(embedding)** を説明でき、
  各段が何を担当し、どこで失敗するのかを区別できる。
- OpenCV **Haar カスケード**で顔を検出し、`detectMultiScale` の主要パラメータが「見逃し ↔
  誤検出」をどう動かすかを掌握する。合成 → 実写の **ドメインギャップ** を体感する。
- 顔埋め込みを **L2 正規化 → コサイン類似度**で比較し、**1:1 照合(verification)** と
  **1:N 識別(identification)** を実装できる。前者を **ROC / EER / TAR@FAR**、後者を
  **rank-1 / CMC** で評価できる。
- 顔埋め込みを **DBSCAN / AgglomerativeClustering** でクラスタリングし、ラベル無しで
  「同一人物ごとのアルバム」へ **自動仕分け** できる。品質を **purity / NMI / homogeneity /
  completeness** で測り、しきい値(eps)を掃引して最適点を定量的に選べる。
- 顔データ特有の **プライバシ・バイアス**の注意を一通り説明できる。
- 完成形として「**集合写真 → 検出 → 整列 → 埋め込み → クラスタリング → 人物アルバム ＋
  認識/クラスタ品質の一括指標**」を CPU・合成データだけで動かせる。

---

## 1. 直感 — 顔認識は「検出 → 整列 → 埋め込み」の三段構え

顔認識という言葉は、実は 3 つの別々の仕事の連なりです。まず **検出 (detection)**: 1 枚の
写真の「どこに顔があるか」を矩形で答える。次に **整列 (alignment)**: 見つけた顔を切り出し、
目・鼻・口がいつも同じ位置に来るように回転・拡縮して揃える。最後に **埋め込み (embedding,
＝認識の中核)**: 整列済みの顔を固定長のベクトルに変換し、**ベクトルの近さ＝同一人物らしさ**
として扱う。この「顔 → ベクトル」さえ手に入れば、後段はすべてベクトルの距離計算に帰着します。

ベクトルにしてしまえば、応用は一気に広がります。2 つのベクトルが近いか遠いかで「同じ人?」を
判定すれば **1:1 照合(verification)**（スマホのロック解除、本人確認）。登録済み N 人のベクトル
群と照らして「誰?」を当てれば **1:N 識別(identification)**（容疑者検索、入退室）。ラベル無しの
ベクトル群を**勝手に束ねれば** **クラスタリング**（写真アプリの「人物」タブ＝同じ人の写真が
自動でまとまる機能）。本章はこの 3 つの応用を、同じ 1 種類の埋め込みから順に組み立てます。

なぜベクトルの「近さ」で人物が分かるのか。鍵は **対照学習 (metric learning)** です。顔認識
モデル（ArcFace 等）は「同一人物のペアは近く、別人のペアは遠く」なるよう、角度マージン付きの
損失で訓練されます（前提モジュール 15 の三つ組損失・InfoNCE と同じ発想の顔特化版）。本講座は
依存衝突を避けて、ArcFace の代わりに **ImageNet 学習済み ResNet18 の 512 次元特徴**を顔埋め込みの
代用に使います。合成顔は人物ごとに見た目（肌・髪・目の色、眼鏡やひげ）をはっきり変えてあるので、
ImageNet 特徴でも「同一人物 → 近いベクトル」がきれいに成り立ち、照合もクラスタリングも意味を
持ちます。実写で本気を出すときは `FaceEncoder` を ArcFace に差し替えるだけ、という設計です。

---

## 2. 顔検出 — OpenCV Haar カスケードと `detectMultiScale`

**Haar カスケード**は 2001 年の Viola–Jones に始まる古典の顔検出器です。原理は「顔には共通の
明暗パターンがある（目の領域は頬より暗い、鼻筋は両脇より明るい…）」という観察。これを **Haar
特徴**（白黒の矩形フィルタの差分）で捉え、**積分画像**で高速に計算し、弱識別器を **カスケード**
（簡単な検査で大半の非顔を早期に棄却）に並べて実時間化します。学習済みの分類器は XML として
OpenCV に同梱されており、`cv2.data.haarcascades` のパスから即座に読み込めます。重みのダウン
ロードも要らず、CPU で軽快に動くのが古典手法の強みです。

API はシンプルです。`cascade = cv2.CascadeClassifier(path)` で読み込み、グレースケール化した
画像に `faces = cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize)` を呼ぶと、
顔の矩形 `[x, y, w, h]` の配列が返ります。重要なのは 3 つのパラメータの意味です。
**`scaleFactor`**（>1.0）は画像ピラミッドの縮小率で、1.05 なら細かいスケールを丁寧に探して
当たりやすいが遅く、1.3 なら粗く速い。**`minNeighbors`** は「近接する検出が最低この数集まったら
確定」で、大きいほど誤検出が減る代わりに本物も取りこぼす。**`minSize`** はこれより小さい顔を
無視する下限です。`detectMultiScale` の前に `cv2.equalizeHist` でヒストグラム平坦化して
明暗差を強調すると、検出が安定します。これらに**唯一の正解は無く**、用途（速度 vs 取りこぼし
許容度）でチューニングします。`01_face_detection.py` はパラメータを掃引し、検出数が滑らかに
変わる様子を表で見せます。

```bash
uv run python lectures/30_face_detection_recognition/01_face_detection.py
# → 24 顔の合成集合写真で sf=1.05,mN=5 のとき 22 件検出。
#   minNeighbors↑ で誤検出↓・取りこぼし↑、scaleFactor↓ で当たりやすく遅い、という表が出る。
```

ここで必ず触れておくべきが **ドメインギャップ (domain gap)** です。Haar は**実写の顔**で学習
されています。本講座の合成顔は明暗構造が顔っぽく作ってあるので集合写真ではそこそこ当たりますが、
誤検出や取りこぼしも出ますし、別の合成設定なら **0 件**になることもあります（同じ顔を単独で
拡大すると検出が外れる、なども起きる）。検出器の精度は突き詰めると「**学習データと入力ドメインの
一致**」が支配的で、合成 → 実写の乖離がそのまま精度差になります。だから本章のスクリプトは
検出数を一切アサートせず、**実写を `data/30_face_detection_recognition/` に置けばそちらでも
検出する**ようにしてあり、合成で 0 件でも必ず `exit 0` で完走します。より頑健な検出が要るなら、
**OpenCV DNN 顔検出（res10 SSD）**や MediaPipe Face Detector を使います。`01` は DNN を任意で
組み込めるようにしてあり、モデル重みが無ければ案内だけ出してスキップします（落とさない）。

---

## 3. 整列(alignment)と埋め込み — ArcFace の考え方と本講座の代用

**整列 (alignment)** は地味ですが認識精度を大きく左右します。検出枠の中で顔が傾いていたり、
中心がずれていたりすると、埋め込みが「顔の中身」ではなく「ポーズ」を拾ってしまい、同一人物
でもベクトルが離れます。そこで実運用の顔認識は、まず顔ランドマーク（両目・鼻・口角の 5 点等）
を検出し、それらが**テンプレート上の決まった座標**に来るようアフィン変換で正規化してから
埋め込みます。「目をいつも同じ高さ・同じ間隔に揃える」ことで、埋め込み器は見た目の差だけに
集中できます。本章の合成顔は最初から中央寄せで生成しているので、`mini_project.py` の整列は
「検出枠を切り出して `FACE_SIZE` 角にリサイズ」する簡易版で十分です（`crop_align`）。実写に
進むときは、ここを 5 点ランドマークのアフィン整列に置き換えます。

埋め込みの正準は **ArcFace**（Additive Angular Margin Loss）です。顔の特徴ベクトルを単位
球面上に置き、クラス（人物）間に**角度マージン**を強制することで、同一人物を密に・別人を
遠くに押し広げます。出力は典型的に 512 次元で、推論時は **必ず L2 正規化**してコサイン類似度
（＝角度）で比較します。`insightface` の `buffalo_l` などが定番ですが、`onnxruntime` 依存・
numpy ピンの衝突が起きやすいため、本講座では実行経路に入れません（§9 で任意導入を案内）。

本講座の代用は **torchvision ResNet18（ImageNet 学習済み）の fc 直前 512 次元特徴**です。
`FaceEncoder`（`face_lab.py`）は `resnet18(weights=...)` を読み込み、`model.fc =
torch.nn.Identity()` で分類ヘッドを外して Global Average Pooling 後の 512 次元ベクトルを
取り出します。前処理はモデル付属の `weights.transforms()` を使い、平均/分散/サイズのズレ事故を
防ぎます。**重みのダウンロードに失敗した（オフライン等）場合でも `exit 0` を保つ**ため、
`FaceEncoder` は cv2 だけで作る classical 特徴（HSV 色ヒストグラム＋グレースケールの 4×4 グリッド
平均）へ自動フォールバックします。どちらの経路でも「同一人物 → 近いベクトル」が成り立つよう
設計してありますが、品質は当然 `resnet18 > classical` です。

```python
# face_lab.FaceEncoder の核（抜粋・概念）
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()          # 512次元の埋め込みを取り出す
feat = model(batch)                      # (N, 512)
emb = fl.l2_normalize(feat.numpy())      # ★ 比較前に必ず L2 正規化（向きだけ見る）
```

---

## 4. 1:1 照合(verification) — しきい値・ROC・EER・TAR@FAR

**1:1 照合**は「この 2 枚は同一人物か?」を答えるイエス/ノー問題です。2 つの埋め込みを L2 正規化
してコサイン類似度 `s` を求め、**しきい値 `t` 以上なら同一人物 (accept)**、未満なら別人 (reject)
とします。評価するには、全ペアを **genuine（同一人物）** と **impostor（別人）** に分け、それぞれ
スコアの山を作ります（`02` は 48 枚 = 8 人 × 6 枚から genuine 120 ペア・impostor 1008 ペアを
生成）。良い埋め込みでは 2 つの山がきれいに離れ、「どこかに線 `t` を引けば分かれる」状態になります。

しきい値 `t` を動かすと 2 種類の誤りがトレードオフします。`t` を下げれば本人を取りこぼさない
（**TAR: True Accept Rate** が上がる）が、別人も通してしまう（**FAR: False Accept Rate** が
上がる）。`t` を上げれば逆。これを全 `t` について描いたのが **ROC 曲線**（横 FAR・縦 TAR）です。
顔認証の世界では「**FAR を業務要件以下に固定し、その下で TAR を最大化**」が基本姿勢で、その
読み方が **TAR@FAR**（例: FAR=1% のとき本人をどれだけ通せるか＝`TAR@FAR=1e-2`）。もう一つの
要約指標が **EER (Equal Error Rate)**: FAR と FRR(=1−TAR) が等しくなる動作点の誤り率で、しきい値に
依らず埋め込み品質をひと言で表します（低いほど良い）。

```bash
uv run python lectures/30_face_detection_recognition/02_face_embeddings_verification.py
# → EER=0.000 / TAR@FAR=1e-2=1.000 / rank-1=1.000（合成顔は分離が容易なので満点になりやすい）
#   02_score_dist.png（genuine と impostor の山）, 02_roc.png, 02_cmc.png を保存
```

実装は `face_lab` の小関数に分けてあります。`verification_scores` が全ペアのコサイン類似度と
genuine ラベルを返し、`roc_far_tar` がしきい値を高→低に掃引して `(thresholds, FAR, TAR)` を作り、
`compute_eer` が `|FAR−FRR|` 最小点で EER を、`tar_at_far` が `FAR<=target` を満たす最大 TAR を
返します。合成顔は分離が容易なので EER=0 になりがちですが、**実写ではここに必ず重なりが出て
EER>0 になる**——その重なり領域こそ「しきい値をどう選んでも避けられない誤り」だ、と理解するのが
このセクションの肝です。

---

## 5. 1:N 識別(identification) — gallery / probe・rank-1・CMC

**1:N 識別**は「このプローブ画像は、登録済み N 人のうち誰か?」を答える問題です。各人物を代表する
**ギャラリー (gallery, 登録テンプレート)** を用意し、照会したい **プローブ (probe)** をギャラリー
全員と照合して、最も似た人を本人候補とします。`02` は各人物の先頭 1 枚をギャラリーに登録し、
残りをプローブにして評価します。1:1 照合が「2 枚が同じか」のyes/noだったのに対し、1:N は
「N 個の候補から正解を引き当てる」ランキング問題である点が違います。

評価指標は **rank-1 精度**（最も似た 1 人が本人だった割合）と、それを緩めた **CMC (Cumulative
Match Characteristic) 曲線**（類似度上位 k 位以内に本人が入る割合を k について累積）です。CMC[0]
が rank-1、CMC が右肩上がりに 1 へ近づく速さが識別器の良さを表します。実務では「rank-1 だけ
でなく rank-5 まで見て候補提示する」(人手の最終確認を挟む容疑者検索など) 使い方も一般的です。

```bash
# 02 の出力（1:N 部分）
#   ギャラリー 8 人 / プローブ 40 枚 → rank-1 = 1.000, rank-5 = 1.000
```

ここで照合と識別の関係を整理しておきます。**1:N 識別は内部的に N 回の 1:1 照合**であり、同じ
埋め込み・同じコサイン類似度の上に乗っています。だから埋め込みの良さ（EER の低さ）が改善すれば
識別の rank-1 も上がる、という連動があります。違いは「閾値で yes/no を出す(照合)」か「最も近い
1 つを選ぶ(識別)」か。さらに実運用の 1:N には「**該当者なし (open-set)**」の扱い——最近傍でも
類似度が閾値未満なら『登録外』と答える——という難所がありますが、本章は閉集合 (closed-set) の
基本に絞ります。

---

## 6. 人物クラスタリング — DBSCAN / Agglomerative で「人物」タブを作る

ここまではラベル（誰が誰か）を使っていましたが、写真アプリの「人物」機能はラベルがありません。
大量の顔埋め込みを **教師なし**で束ね、「同じ人の写真」を自動でまとめる——これが **クラスタリング**
です。鍵は同じく **L2 正規化 + コサイン距離**（正規化済みなら `コサイン距離 = 1 − 内積`）。
本章は 2 つの代表手法を使います。

**DBSCAN** は密度ベースで、**クラスタ数を指定しません**。「半径 `eps` 以内に `min_samples` 件
以上の仲間がいる点」を芯にしてまとまりを広げ、どのまとまりにも届かない点を **ノイズ (-1)** に
します。人物数を事前に知らなくてよく、外れ値（変な角度・低画質の顔）をノイズに落とせるのが
写真アルバム向き。唯一にして最重要のダイヤルが `eps`: 大きすぎると別人が 1 つに融合し、小さ
すぎると同一人物が割れてノイズだらけになります。**Agglomerative（凝集型）** は近いものから順に
併合する木を作り、**距離しきい値 `distance_threshold`** で「ここまで近ければ同一人物」と切ります。
顔では `metric="cosine"` + `linkage="average"` が扱いやすく、`eps` と同じ役割のしきい値を 1 つ
決めるだけです。

```bash
uv run python lectures/30_face_detection_recognition/03_face_clustering.py
# → DBSCAN(eps=0.05) / Agglomerative(threshold=0.06) ともに 推定人数=8（真値8）, NMI=1.000
#   03_album_dbscan.png（行=自動仕分けされた人物）, 03_pca_scatter.png を保存
```

`03_face_clustering.py` は両手法の結果を「アルバム」画像（1 行 = 1 クラスタ = 1 人物）として
書き出し、PCA で 2 次元に落とした散布図（色 = 予測クラスタ、形 = 真の人物）で「同じ人がまとまり
別人が離れている」様子を可視化します。`min_samples=2` は「2 枚以上集まって初めて人物として認める」
の意味で、1 枚しか写っていない人をノイズに落とす実務的な設定です。**正規化を忘れて生の特徴量に
DBSCAN を当てるのは典型的な失敗**——ベクトルの長さに引きずられてクラスタが崩れます。必ず
`fl.l2_normalize` を通してから渡してください。

---

## 7. 評価指標の読み方 — 認識とクラスタリングを別軸で測る

`04_recognition_cluster_eval.py` は認識とクラスタリングの指標を一望し、しきい値をデータから
選びます。**認識側**は ROC-AUC / EER / TAR@FAR（1:1）と rank-1/5（1:N）。AUC は ROC 下の面積で
1.0 が完璧、EER は前述のとおり閾値非依存の総合誤り率です。**クラスタリング側**は 4 指標を相補的に
読みます。**homogeneity**（各クラスタが単一人物で占められているか＝混ざってないか）、
**completeness**（同一人物が 1 クラスタにまとまっているか＝割れてないか）、両者の調和が **NMI**
（情報量ベース、クラスタ数に左右されにくい総合指標）、そして直感的な **purity**（各クラスタを
多数派人物に割り当てた時の正解率）。

purity には**落とし穴**があります。クラスタを増やすほど上がる（極端には 1 点 1 クラスタで
purity=1.0）ので、**過分割に甘い**のです。だから purity 単独で判断せず、必ず NMI と併読します。
NMI は「全部 1 クラスタ（過併合）」でも「1 点 1 クラスタ（過分割）」でも下がり、ちょうど良い
分割で最大になる——この性質を使ってしきい値を選びます。`04` は DBSCAN の `eps` を 0.02〜0.15 で
掃引し、各点の NMI/purity/homogeneity と推定人数をプロットして、**NMI のピーク**を最適 `eps` と
します。実際にやってみると、`eps` が小さい側はノイズだらけ（過分割で NMI 低下）、大きい側は
全員融合（過併合で NMI=0）、その間に NMI=1 の「ちょうど良い谷間の山」が現れます。

```bash
uv run python lectures/30_face_detection_recognition/04_recognition_cluster_eval.py
# → 認識: ROC-AUC=1.0, EER=0.0, rank-1=1.0
#   クラスタ eps 掃引表 → NMI 最大の eps=0.04 で 推定人数=8（真値8）
#   04_eps_sweep.png（NMI/purity/homogeneity と #clusters の曲線）, 04_metrics.json を保存
```

```
   eps   #clu  noise  purity   NMI    homo
  0.020   12     5   0.938  0.856  0.933   ← 過分割（ノイズ多い）
  0.040    8     0   1.000  1.000  1.000   ← best（NMI ピーク・真値8に一致）
  0.080    5     0   0.625  0.800  0.667   ← 融合が始まる
  0.120    1     0   0.125  0.000  0.000   ← 全員 1 クラスタ（過併合）
```

最後に **しきい値は『社会的コスト』で決まる**ことを強調します。誤受入(FAR)が致命的な入退室
管理は FAR を極小に倒し（TAR を多少犠牲にしても）、写真アルバムの整理は多少の誤りより快適さ
(TAR/網羅性)を優先する。同じ埋め込みでも、用途ごとに動作点(しきい値)を変えるのが実務です。

---

## 8. プライバシ・バイアス — 顔を扱う者の責任

顔は**最も機微な個人情報**の一つです。教材・研究であっても、実在人物の顔を扱うなら **同意・
目的限定・保持期間・削除権**を前提にし、規制（GDPR の生体情報、各国の生体認証法）を確認します。
本講座が **合成顔だけ**を使うのはこのためでもあります（実在人物のデータを一切持ち込まない）。

顔認識には **バイアス**の問題が知られています。学習データの人口統計が偏ると、肌の色・年齢・
性別によって精度が大きく変わり、特定の集団で誤認が増えます（公的監査でも繰り返し報告されて
きました）。対策の第一歩は、**精度を 1 つの数字に丸めない**こと。全体の EER/TAR@FAR だけでなく、
**属性ごとに分解**して動作点の公平性を点検します。そして「技術的に可能か」と「やってよいか」は
別問題です。監視・追跡など濫用リスクの高い用途では、導入可否そのものを問う姿勢が求められます。

---

## 9. insightface(ArcFace) — 概念と任意導入（実行経路では使わない）

実写で本気の顔認識をするなら、埋め込みは **ArcFace** が定番です。`insightface` の
`FaceAnalysis`（`buffalo_l` パック）は検出＋5 点ランドマーク整列＋ArcFace 埋め込み＋年齢/性別
推定までを一括で提供し、`ctx_id=-1` で CPU 実行できます。各顔の `normed_embedding`（L2 正規化
済み 512 次元）をそのまま本章の照合・クラスタリングに流せます。本講座が実行経路に入れないのは、
`onnxruntime` 依存や numpy のバージョンピンが他のグループと **衝突**しやすいからです。試すときは
**専用の依存グループに隔離**して入れてください。

```bash
# 任意（衝突回避のため隔離グループ推奨）。CPU 実行は ctx_id=-1。
uv add --group face insightface onnxruntime
```

```python
# 概念コード（本講座の実行経路には含めない）。FaceEncoder をこれに差し替えるイメージ。
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)                 # CPU
faces = app.get(bgr_image)             # 検出＋整列＋埋め込みを一括
emb = faces[0].normed_embedding        # L2 正規化済み 512 次元 → 本章の照合/クラスタへ
```

同様に **MediaPipe Face Detector / Face Mesh** は CPU で実時間動作する高品質な検出・ランドマーク
ですが、numpy<2 ピンや protobuf 競合を起こしやすいので、使うなら別グループに隔離します。本章の
コードはこれらが未導入でも `import` を `try/except` でガードして案内だけ出し、必ず完走します。

---

## 🛠 章末ミニプロジェクト — 写真群を「人物アルバム」へ自動仕分け

`mini_project.py` は本章の全工程を 1 本に統合した完成形です。流れは次のとおり。

1. **合成集合写真**（6 人 × 4 枚 = 24 顔を 1 枚に配置）を、各顔の正解位置・人物 ID 付きで生成。
2. **Haar 検出**で顔矩形を得る（実写を `data/` に置けばそちらでも動く）。
3. 検出枠を正解枠と **IoU で突合せ**、検出品質を **recall / precision** で測る（FP・見逃しも計上）。
4. 検出枠を切り出して `FACE_SIZE` に **整列** → ResNet で **埋め込み** → L2 正規化。
5. `eps` を内部掃引した **DBSCAN** で **クラスタリング**（NMI 最大の動作点を採用）。
6. クラスタ = **人物アルバム**（行 = 自動仕分けされた人物）を画像化し、**認識(EER) と クラスタ
   品質(NMI 他)** を JSON にまとめる。
7. **頑健性**: 検出がドメインギャップでほぼ当たらない（recall<0.5）場合は、正解枠の切り出しに
   自動フォールバックして後段を必ず完走させる（`exit 0` を保証）。

```bash
uv run python lectures/30_face_detection_recognition/mini_project.py
# → 検出 recall=0.833 / precision=1.000（FP=0, 見逃し=4）
#   クラスタ best eps=0.05, 推定人数=6（真値6）, NMI=1.000, EER=0.000
#   mini_scene.png / mini_detected.png / mini_album.png / mini_summary.json
```

**発展課題**: ① `FaceEncoder` を §9 の insightface(ArcFace) に差し替えて、合成→実写で EER が
どう変わるか比べる。② 検出を OpenCV DNN(res10) や MediaPipe に替えて recall を上げる。
③ クラスタリングを Agglomerative に替え、しきい値を NMI で選ぶ。④ open-set 識別（最近傍でも
閾値未満なら『登録外』）を追加する。部品の境界（検出 / 整列 / 埋め込み / クラスタ / 評価）が
分かれているので差し替えが容易です。

---

## ✅ 到達チェックリスト

- [ ] 顔認識の三段（検出 → 整列 → 埋め込み）を説明でき、各段の失敗モードを区別できる。
- [ ] `cv2.CascadeClassifier` + `detectMultiScale` で顔検出でき、`scaleFactor`/`minNeighbors`/
      `minSize` の効き方とドメインギャップを説明できる。
- [ ] 埋め込みを **L2 正規化 → コサイン類似度**で比較する理由を説明できる。
- [ ] 1:1 照合を ROC / EER / TAR@FAR で、1:N 識別を rank-1 / CMC で評価できる。
- [ ] genuine/impostor ペアの作り方、FAR/TAR/FRR の定義、EER の意味を自分の言葉で言える。
- [ ] DBSCAN と Agglomerative で顔をクラスタリングし、`eps`/`distance_threshold` の役割を説明できる。
- [ ] purity / NMI / homogeneity / completeness の違いと、purity 単独の危うさを説明できる。
- [ ] `eps` 掃引で NMI ピークを選ぶ手順を実行できる。
- [ ] 顔データのプライバシ・バイアスの注意を 3 つ以上挙げられる。
- [ ] 検出 → 整列 → 埋め込み → クラスタリングのパイプラインを最後まで動かし、結果を読める。

---

## ❓ 落とし穴・FAQ・デバッグ

- **合成顔が検出されない / 0 件になる**: 仕様どおりの **ドメインギャップ**。Haar は実写で
  学習されている。`minSize` を下げる・`scaleFactor` を 1.05 にする・`equalizeHist` を入れると
  当たりやすくなるが、本質は実写を `data/` に置くこと。検出 0 でも `exit 0`、`mini_project` は
  正解枠にフォールバックして後段を完走する。
- **`cv2` の色順 (BGR/RGB) を取り違える**: `cv2` は BGR、torchvision/PIL/matplotlib は RGB。
  埋め込みやアルバム可視化で色が壊れたら `cv2.cvtColor` の向きを疑う。`face_lab` は PIL 化の
  直前に必ず `COLOR_BGR2RGB` している。
- **コサイン類似のつもりが正規化忘れ**: L2 正規化せずに内積を取ると「ベクトルの長さ」に
  引きずられて照合もクラスタも崩れる。比較の直前に必ず `fl.l2_normalize`。
- **EER が 0 で出来すぎに見える**: 合成顔は人物差を強くつけてあるので分離が容易＝満点に
  なりやすい。**実写では必ず重なりが出て EER>0**。指標の妥当性は実写で確かめる。
- **DBSCAN が 1 クラスタ / ノイズだらけになる**: `eps` が大きすぎる（全員融合）/ 小さすぎる
  （全部ノイズ）。`04` のように `eps` を掃引して **NMI ピーク**を選ぶ。距離は `metric="cosine"`、
  入力は L2 正規化済みであることを確認。
- **purity は高いのに NMI が低い**: 過分割（クラスタを増やしすぎ）。purity はクラスタ数に
  甘いので NMI/homogeneity と併読する。
- **`AgglomerativeClustering` で `metric` 引数エラー**: 新しめの scikit-learn は `metric=`、
  古い版は `affinity=`。本講座（sklearn 1.9）は `metric="cosine"` を使う。
- **ResNet 重みの DL に失敗（オフライン）**: `FaceEncoder` が classical 特徴へ自動フォール
  バックして完走する（`mode=classical` と表示）。品質は落ちるが指標計算は同じ流れで動く。
- **matplotlib の図で日本語が豆腐(□)になる**: 既定フォントに日本語が無いため。本講座は図中の
  文字を英語にしてこれを回避している（コンソール出力は日本語のまま）。
- **`detectMultiScale` が遅い**: 大きい画像でそのまま回している。`scaleFactor` を上げる
  （1.2 など）か、入力を縮小してから検出し、座標を戻す。

---

## 🚀 発展トピック・参考

- **ArcFace / CosFace / SphereFace**: 角度マージン系の顔埋め込み損失。ArcFace(Deng+ 2019,
  `arXiv:1801.07698`)が代表。`insightface` の `buffalo_l` が実用の入口（§9・任意）。
- **検出の系譜**: Viola–Jones(Haar, 2001) → DPM → SSD/RetinaFace(深層, ランドマーク同時推定)。
  RetinaFace は顔検出＋5 点ランドマークを一度に出し、整列まで一気通貫にできる。
- **整列(alignment)**: 5 点ランドマーク → 相似変換でテンプレート座標へ。`insightface` は内部で
  自動実行。精度に効くので実写では省略しない。
- **open-set 識別と品質管理**: 「該当者なし」を閾値で弾く open-set、顔品質(ぼけ・向き・遮蔽)で
  事前にフィルタする品質スコア、テンプレート更新(オンライン学習)など実運用の論点。
- **クラスタリング**: 大規模では近似最近傍(FAISS, モジュール 17/42) で類似グラフを作り、
  Chinese Whispers / Rank-Order / HDBSCAN で束ねるのが定番。本章 DBSCAN はその最小形。
- **公平性・規制**: NIST FRVT のデモグラフィック評価、GDPR の生体情報、各地の顔認識規制。
  技術の前に「使ってよいか」を問う。
- 参考: Viola–Jones (2001) CVPR、ArcFace (Deng+ 2019) CVPR、DBSCAN (Ester+ 1996) KDD、
  CMC/ROC は生体認証の標準評価（ISO/IEC 19795）。

---

## ▶ 動かし方

```bash
# 共有ヘルパの自己テスト（生成→埋め込み→各評価が一通り動く）
uv run python lectures/30_face_detection_recognition/face_lab.py
# 1) 顔検出（Haar・パラメータ掃引・ドメインギャップ）
uv run python lectures/30_face_detection_recognition/01_face_detection.py
# 2) 顔埋め込みと 1:1 照合 / 1:N 識別（ROC/EER/TAR@FAR, rank-1/CMC）
uv run python lectures/30_face_detection_recognition/02_face_embeddings_verification.py
# 3) 人物クラスタリング（DBSCAN / Agglomerative → アルバム）
uv run python lectures/30_face_detection_recognition/03_face_clustering.py
# 4) 認識・クラスタリングの一括評価（eps 掃引で NMI ピーク選択）
uv run python lectures/30_face_detection_recognition/04_recognition_cluster_eval.py
# 章末ミニプロジェクト（検出→整列→埋め込み→クラスタリングの統合）
uv run python lectures/30_face_detection_recognition/mini_project.py
# 演習（自己採点）と模範解答
uv run python lectures/30_face_detection_recognition/exercises.py
uv run python lectures/30_face_detection_recognition/exercises_solutions.py
```

出力（可視化・図・JSON）は `outputs/30_face_detection_recognition/` に保存されます。すべて
**CPU・合成顔**で完結し、ネットに出るのは **初回の ResNet18 重み DL のみ**（失敗時は classical
特徴へ自動フォールバック）。実写を `data/30_face_detection_recognition/` に置けば、検出と
パイプラインはそちらでも動きます。

---

> 参照ライブラリ（版）: torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 /
> faiss-cpu（本章では未使用・関連は 17/42）/ opencv-python-headless 4.13 / scikit-learn 1.9 /
> numpy 2.x。顔埋め込みの定番 `insightface(ArcFace)` と `mediapipe` は依存衝突を避けて
> **実行経路では使わず**、概念紹介＋任意導入（`uv add --group face ...`）にとどめています。
> なお `faiss-gpu` という pip 名は存在しません（GPU 版は `faiss-gpu-cuvs`・Linux+NVIDIA 限定）。
> — 2026-06