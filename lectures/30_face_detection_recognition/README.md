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

<figure class="lec-fig"><svg viewBox="0 0 660 210" role="img" aria-label="顔認識のパイプライン: 写真から検出で顔矩形、整列で正規化、埋め込みで512次元ベクトル" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">検出 → 整列 → 埋め込み の三段構え</text><rect x="18" y="50" width="118" height="120" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><circle cx="48" cy="85" r="12" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><circle cx="104" cy="80" r="12" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><circle cx="55" cy="128" r="12" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><circle cx="108" cy="132" r="12" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="77" y="190" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">集合写真</text><line x1="138" y1="110" x2="176" y2="110" stroke="#71717a" stroke-width="2"/><polygon points="184,110 174,105 174,115" fill="#71717a"/><text x="161" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">検出</text><rect x="182" y="50" width="118" height="120" rx="6" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="241" cy="104" r="26" fill="#ffedd5" stroke="#f97316" stroke-width="1.5"/><rect x="213" y="76" width="56" height="56" fill="none" stroke="#ea580c" stroke-width="2.5"/><text x="241" y="190" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">顔の矩形</text><line x1="302" y1="110" x2="340" y2="110" stroke="#71717a" stroke-width="2"/><polygon points="348,110 338,105 338,115" fill="#71717a"/><text x="325" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">整列</text><rect x="346" y="50" width="108" height="120" rx="6" fill="#fff7ed" stroke="#f97316" stroke-width="1.5"/><rect x="372" y="70" width="56" height="70" fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/><line x1="372" y1="96" x2="428" y2="96" stroke="#f97316" stroke-width="1" stroke-dasharray="3 2"/><circle cx="387" cy="96" r="3.5" fill="#18181b"/><circle cx="413" cy="96" r="3.5" fill="#18181b"/><line x1="391" y1="122" x2="409" y2="122" stroke="#18181b" stroke-width="2"/><text x="400" y="190" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">整列した顔</text><line x1="456" y1="110" x2="494" y2="110" stroke="#71717a" stroke-width="2"/><polygon points="502,110 492,105 492,115" fill="#71717a"/><text x="479" y="100" text-anchor="middle" font-size="11" font-weight="700" fill="#c2410c">埋め込み</text><rect x="504" y="50" width="138" height="120" rx="6" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="556" y="64" width="34" height="14" fill="#ffedd5" stroke="#c2410c" stroke-width="1"/><rect x="556" y="78" width="34" height="14" fill="#f97316" stroke="#c2410c" stroke-width="1"/><rect x="556" y="92" width="34" height="14" fill="#ea580c" stroke="#c2410c" stroke-width="1"/><rect x="556" y="106" width="34" height="14" fill="#ffedd5" stroke="#c2410c" stroke-width="1"/><rect x="556" y="120" width="34" height="14" fill="#c2410c" stroke="#c2410c" stroke-width="1"/><rect x="556" y="134" width="34" height="14" fill="#f97316" stroke="#c2410c" stroke-width="1"/><text x="573" y="190" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">512次元ベクトル</text></svg><figcaption>顔認識は<b>検出</b>(写真のどこに顔があるか)→<b>整列</b>(目・鼻・口を決まった位置へ正規化)→<b>埋め込み</b>(整列した顔を<b>512次元ベクトル</b>へ変換)の三段構えです。比較の前に必ず<b>L2正規化</b>し、以降の<b>照合・識別・クラスタリング</b>はすべてこのベクトル間の<b>距離計算</b>に帰着します。</figcaption></figure>

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

**Haar カスケード**は 2001 年の Viola–Jones に始まる古典の顔検出器です。その原理は、「顔には共通の
明暗パターンがある（目の領域は頬より暗い、鼻筋は両脇より明るい…）」という観察にあります。この明暗の差を
**Haar 特徴**（白黒の矩形フィルタの差分）で捉え、**積分画像**で高速に計算し、弱識別器を **カスケード**
（簡単な検査で大半の非顔を早期に棄却）に並べることで実時間化します。学習済みの分類器は XML として
OpenCV に同梱されているので、`cv2.data.haarcascades` のパスから即座に読み込めます。重みのダウン
ロードも要らず、CPU で軽快に動く——これが古典手法の強みです。

API はシンプルです。`cascade = cv2.CascadeClassifier(path)` で読み込み、グレースケール化した
画像に `faces = cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize)` を呼ぶと、
顔の矩形 `[x, y, w, h]` の配列が返ります。重要なのは 3 つのパラメータの意味です。
**`scaleFactor`**（>1.0）は画像ピラミッドの縮小率で、1.05 なら細かいスケールを丁寧に探して
当たりやすいが遅く、1.3 なら粗く速い。**`minNeighbors`** は「近接する検出が最低この数集まったら
確定」で、大きいほど誤検出が減る代わりに本物も取りこぼす。**`minSize`** はこれより小さい顔を
無視する下限です。`detectMultiScale` の前に `cv2.equalizeHist` でヒストグラム平坦化して
明暗差を強調すると、検出が安定します。これらに**唯一の正解は無く**、用途（速度 vs 取りこぼし
許容度）に応じてチューニングします。`01_face_detection.py` はこれらのパラメータを掃引し、検出数が
滑らかに変わる様子を表で見せます。

<figure class="lec-fig"><svg viewBox="0 0 620 240" role="img" aria-label="画像ピラミッド: 固定サイズの探索窓で、画像をscaleFactor倍ずつ縮小して全サイズの顔を検出する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="310" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像ピラミッド — 固定窓を全サイズの顔に当てる</text><rect x="34" y="54" width="150" height="150" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.6"/><circle cx="109" cy="129" r="56" fill="#dbeafe" stroke="#2563eb" stroke-width="1.4"/><rect x="46" y="66" width="42" height="42" fill="none" stroke="#ea580c" stroke-width="2.2"/><line x1="192" y1="128" x2="222" y2="128" stroke="#71717a" stroke-width="2"/><polygon points="230,128 220,123 220,133" fill="#71717a"/><text x="211" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#c2410c">×1/sf</text><rect x="234" y="78" width="110" height="110" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.6"/><circle cx="289" cy="135" r="28" fill="#dbeafe" stroke="#2563eb" stroke-width="1.4"/><rect x="267" y="113" width="44" height="44" fill="none" stroke="#16a34a" stroke-width="2.6"/><text x="322" y="98" text-anchor="middle" font-size="16" font-weight="700" fill="#16a34a">✓</text><line x1="352" y1="130" x2="382" y2="130" stroke="#71717a" stroke-width="2"/><polygon points="390,130 380,125 380,135" fill="#71717a"/><text x="371" y="120" text-anchor="middle" font-size="11" font-weight="700" fill="#c2410c">×1/sf</text><rect x="394" y="100" width="78" height="78" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.6"/><circle cx="433" cy="139" r="16" fill="#dbeafe" stroke="#2563eb" stroke-width="1.4"/><rect x="411" y="117" width="44" height="44" fill="none" stroke="#71717a" stroke-width="2" stroke-dasharray="4 3"/><text x="548" y="112" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">窓のサイズは1種類</text><text x="548" y="134" text-anchor="middle" font-size="12" fill="#3f3f46">画像を縮めて全サイズ探索</text><text x="548" y="164" text-anchor="middle" font-size="12" fill="#52525b">sf小=細かい・遅い</text><text x="548" y="184" text-anchor="middle" font-size="12" fill="#52525b">sf大=粗い・速い</text></svg><figcaption>Haar などの検出器は<b>探索窓が固定サイズ1種類</b>です。さまざまな大きさの顔を見つけるため、画像を <code>scaleFactor</code> 倍ずつ縮小した<b>画像ピラミッド</b>を作り、各段に同じ窓を滑らせて<b>窓と顔の大きさが一致した段</b>で検出します(中央)。<code>scaleFactor</code> が 1.05 に近いほど刻みが細かく当たりやすい代わりに遅く、1.3 だと粗く速くなります。</figcaption></figure>

```bash
uv run python lectures/30_face_detection_recognition/01_face_detection.py
# → 24 顔の合成集合写真で sf=1.05,mN=5 のとき 22 件検出。
#   minNeighbors↑ で誤検出↓・取りこぼし↑、scaleFactor↓ で当たりやすく遅い、という表が出る。
```

ここで必ず触れておくべきが **ドメインギャップ (domain gap)** です。というのも、Haar は**実写の顔**で
学習されているからです。本講座の合成顔は明暗構造を顔っぽく作ってあるので集合写真ではそこそこ当たりますが、
誤検出や取りこぼしも出ますし、別の合成設定なら **0 件**になることもあります（同じ顔でも単独で
拡大すると検出が外れる、といったことも起きます）。検出器の精度は、突き詰めれば「**学習データと入力ドメインの
一致**」が支配的であり、合成 → 実写の乖離がそのまま精度差として現れます。だからこそ本章のスクリプトは
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

<figure class="lec-fig"><svg viewBox="0 0 640 250" role="img" aria-label="同一人物と別人のコサイン類似度スコアの分布。しきい値tで分けると重なり領域にFARとFRRが残る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">コサイン類似度の分布としきい値 t</text><line x1="60" y1="200" x2="606" y2="200" stroke="#71717a" stroke-width="1.5"/><polygon points="614,200 604,195 604,205" fill="#71717a"/><text x="330" y="236" text-anchor="middle" font-size="12.5" fill="#3f3f46">コサイン類似度 s →</text><path d="M120,200 C170,200 185,100 220,100 C255,100 275,200 340,200 Z" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><path d="M280,200 C345,200 365,95 405,95 C445,95 460,200 520,200 Z" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="195" y="160" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">別人</text><text x="432" y="158" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">同一人物</text><line x1="310" y1="66" x2="310" y2="200" stroke="#18181b" stroke-width="2" stroke-dasharray="6 4"/><text x="310" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">しきい値 t</text><line x1="262" y1="150" x2="300" y2="193" stroke="#16a34a" stroke-width="1.2"/><text x="248" y="144" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">FRR</text><line x1="380" y1="150" x2="326" y2="193" stroke="#dc2626" stroke-width="1.2"/><text x="392" y="144" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">FAR</text></svg><figcaption>同一人物ペア(<b>genuine</b>)はスコアが高く、別人ペア(<b>impostor</b>)は低いので、2 つの山ができます。<b>しきい値 t</b> 以上を「同一人物」と判定します。2 つの山が<b>重なる領域</b>では、どこに t を引いても本人を弾く <b>FRR</b>(誤拒否)か別人を通す <b>FAR</b>(誤受入)のどちらかが必ず残ります。この重なりこそ埋め込み品質の限界で、<code>EER</code> はその総合的な誤り率です。</figcaption></figure>

しきい値 `t` を動かすと 2 種類の誤りがトレードオフします。`t` を下げれば本人を取りこぼさない
（**TAR: True Accept Rate** が上がる）が、別人も通してしまう（**FAR: False Accept Rate** が
上がる）。`t` を上げれば逆。これを全 `t` について描いたのが **ROC 曲線**（横 FAR・縦 TAR）です。
顔認証の世界では「**FAR を業務要件以下に固定し、その下で TAR を最大化**」が基本姿勢で、その
読み方が **TAR@FAR**（例: FAR=1% のとき本人をどれだけ通せるか＝`TAR@FAR=1e-2`）。もう一つの
要約指標が **EER (Equal Error Rate)**: FAR と FRR(=1−TAR) が等しくなる動作点の誤り率で、しきい値に
依らず埋め込み品質をひと言で表します（低いほど良い）。

<figure class="lec-fig"><svg viewBox="0 0 540 300" role="img" aria-label="ROC曲線。横軸FAR縦軸TAR、対角線がランダム、曲線が左上に近いほど高性能。EERとTAR@FARの動作点を示す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="270" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ROC 曲線 — しきい値を動かした軌跡</text><line x1="80" y1="250" x2="312" y2="250" stroke="#71717a" stroke-width="1.5"/><polygon points="320,250 310,245 310,255" fill="#71717a"/><line x1="80" y1="250" x2="80" y2="36" stroke="#71717a" stroke-width="1.5"/><polygon points="80,28 75,38 85,38" fill="#71717a"/><text x="200" y="283" text-anchor="middle" font-size="12.5" fill="#3f3f46">FAR 誤受入率 →</text><text x="42" y="143" text-anchor="middle" font-size="12.5" fill="#3f3f46" style="writing-mode:vertical-rl;text-orientation:upright">TAR 本人受入率</text><line x1="80" y1="250" x2="300" y2="36" stroke="#d4d4d8" stroke-width="1.5" stroke-dasharray="5 4"/><text x="210" y="162" text-anchor="middle" font-size="11" fill="#71717a">ランダム推測</text><path d="M80,250 C84,95 115,48 300,44" fill="none" stroke="#2563eb" stroke-width="2.6"/><circle cx="80" cy="44" r="4" fill="#16a34a"/><text x="140" y="42" text-anchor="middle" font-size="11" fill="#15803d">理想は左上</text><line x1="80" y1="81" x2="130" y2="81" stroke="#16a34a" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/><line x1="130" y1="250" x2="130" y2="81" stroke="#16a34a" stroke-width="1.4" stroke-dasharray="4 3"/><circle cx="130" cy="81" r="4" fill="#15803d"/><text x="200" y="92" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">TAR@FAR</text><circle cx="98" cy="120" r="5" fill="#dc2626"/><line x1="103" y1="119" x2="140" y2="110" stroke="#dc2626" stroke-width="1"/><text x="165" y="112" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">EER</text></svg><figcaption>しきい値 t を高→低に動かすと、別人を通す <b>FAR</b>(横軸)と本人を通す <b>TAR</b>(縦軸)が同時に増え、その軌跡が <b>ROC 曲線</b>です。点線の対角線が当てずっぽうで、曲線が<b>左上に張り付くほど高性能</b>。<b>EER</b> は FAR と FRR が等しくなる動作点の誤り率(低いほど良い)、<b>TAR@FAR</b> は「FAR を業務要件以下に固定したとき本人をどれだけ通せるか」を読む指標です。</figcaption></figure>

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
残りをプローブにして評価します。1:1 照合が「2 枚が同じか」のイエス/ノーだったのに対し、1:N は
「N 個の候補から正解を引き当てる」ランキング問題である点が異なります。

評価指標は **rank-1 精度**（最も似た 1 人が本人だった割合）と、それを緩めた **CMC (Cumulative
Match Characteristic) 曲線**（類似度上位 k 位以内に本人が入る割合を k について累積）です。CMC[0]
が rank-1、CMC が右肩上がりに 1 へ近づく速さが識別器の良さを表します。実務では、rank-1 だけ
でなく rank-5 まで見て候補を提示する使い方も一般的です（人手の最終確認を挟む容疑者検索など）。

```bash
# 02 の出力（1:N 部分）
#   ギャラリー 8 人 / プローブ 40 枚 → rank-1 = 1.000, rank-5 = 1.000
```

ここで照合と識別の関係を整理しておきます。**1:N 識別は内部的には N 回の 1:1 照合**であり、同じ
埋め込み・同じコサイン類似度の上に乗っています。したがって、埋め込みの良さ（EER の低さ）が改善すれば
識別の rank-1 も連動して上がります。両者の違いは、閾値で yes/no を出す（照合）か、最も近い
1 つを選ぶ（識別）か、という点にあります。さらに実運用の 1:N には、「**該当者なし (open-set)**」の扱い——最近傍でも
類似度が閾値未満なら『登録外』と答える——という難所がありますが、本章は閉集合 (closed-set) の
基本に絞ります。

---

## 6. 人物クラスタリング — DBSCAN / Agglomerative で「人物」タブを作る

ここまではラベル（誰が誰か）を使ってきましたが、写真アプリの「人物」機能にはラベルがありません。
大量の顔埋め込みを **教師なし**で束ね、「同じ人の写真」を自動でまとめる——これが **クラスタリング**
です。鍵となるのは、ここでも **L2 正規化 + コサイン距離** です（正規化済みなら `コサイン距離 = 1 − 内積`）。
本章では 2 つの代表手法を使います。

**DBSCAN** は密度ベースで、**クラスタ数を指定しません**。「半径 `eps` 以内に `min_samples` 件
以上の仲間がいる点」を芯にしてまとまりを広げ、どのまとまりにも届かない点を **ノイズ (-1)** に
します。人物数を事前に知らなくてよく、外れ値（変な角度・低画質の顔）をノイズに落とせるのが
写真アルバム向き。唯一にして最重要のダイヤルが `eps`: 大きすぎると別人が 1 つに融合し、小さ
すぎると同一人物が割れてノイズだらけになります。**Agglomerative（凝集型）** は近いものから順に
併合する木を作り、**距離しきい値 `distance_threshold`** で「ここまで近ければ同一人物」と切ります。
顔では `metric="cosine"` + `linkage="average"` が扱いやすく、`eps` と同じ役割のしきい値を 1 つ
決めるだけです。

<figure class="lec-fig"><svg viewBox="0 0 660 235" role="img" aria-label="DBSCANのeps効果。小さすぎると同一人物が割れノイズ化、適切だと人物ごとにまとまり、大きすぎると別人まで融合する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">DBSCAN の eps — 小さすぎ:分裂 / 大きすぎ:融合</text><rect x="18" y="52" width="196" height="150" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="232" y="52" width="196" height="150" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="446" y="52" width="196" height="150" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="116" y="46" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">eps 小さすぎ</text><text x="330" y="46" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">eps 適切</text><text x="544" y="46" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">eps 大きすぎ</text><circle cx="63" cy="92" r="5" fill="#2563eb"/><circle cx="80" cy="78" r="5" fill="#16a34a"/><circle cx="94" cy="100" r="5" fill="#ea580c"/><circle cx="110" cy="84" r="5" fill="#dc2626"/><line x1="131" y1="163" x2="141" y2="173" stroke="#71717a" stroke-width="2"/><line x1="131" y1="173" x2="141" y2="163" stroke="#71717a" stroke-width="2"/><line x1="149" y1="147" x2="159" y2="157" stroke="#71717a" stroke-width="2"/><line x1="149" y1="157" x2="159" y2="147" stroke="#71717a" stroke-width="2"/><line x1="163" y1="169" x2="173" y2="179" stroke="#71717a" stroke-width="2"/><line x1="163" y1="179" x2="173" y2="169" stroke="#71717a" stroke-width="2"/><line x1="127" y1="151" x2="137" y2="161" stroke="#71717a" stroke-width="2"/><line x1="127" y1="161" x2="137" y2="151" stroke="#71717a" stroke-width="2"/><text x="116" y="222" text-anchor="middle" font-size="11.5" fill="#52525b">同一人物が割れ、ノイズだらけ</text><circle cx="277" cy="92" r="5" fill="#2563eb"/><circle cx="294" cy="78" r="5" fill="#2563eb"/><circle cx="308" cy="100" r="5" fill="#2563eb"/><circle cx="324" cy="84" r="5" fill="#2563eb"/><circle cx="350" cy="168" r="5" fill="#ea580c"/><circle cx="368" cy="152" r="5" fill="#ea580c"/><circle cx="382" cy="174" r="5" fill="#ea580c"/><circle cx="346" cy="156" r="5" fill="#ea580c"/><circle cx="300" cy="89" r="30" fill="none" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="5 4"/><circle cx="364" cy="163" r="30" fill="none" stroke="#ea580c" stroke-width="1.4" stroke-dasharray="5 4"/><text x="330" y="222" text-anchor="middle" font-size="11.5" fill="#52525b">人物ごとにまとまる</text><circle cx="491" cy="92" r="5" fill="#ea580c"/><circle cx="508" cy="78" r="5" fill="#ea580c"/><circle cx="522" cy="100" r="5" fill="#ea580c"/><circle cx="538" cy="84" r="5" fill="#ea580c"/><circle cx="564" cy="168" r="5" fill="#ea580c"/><circle cx="582" cy="152" r="5" fill="#ea580c"/><circle cx="596" cy="174" r="5" fill="#ea580c"/><circle cx="560" cy="156" r="5" fill="#ea580c"/><circle cx="545" cy="130" r="72" fill="none" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="6 4"/><text x="544" y="222" text-anchor="middle" font-size="11.5" fill="#52525b">別人まで1クラスタに融合</text></svg><figcaption>DBSCAN は半径 <code>eps</code> 内に <code>min_samples</code> 件以上の仲間がいる点を芯にまとまりを広げ、どこにも届かない点を<b>ノイズ(−1)</b>(×印)にします。<b>eps が小さすぎる</b>と同一人物が割れてノイズだらけ、<b>大きすぎる</b>と別人まで 1 つに融合し、<b>ちょうど良い eps</b> でだけ人物ごとにまとまります。比較は必ず <b>L2 正規化</b>後のコサイン距離で行います。</figcaption></figure>

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

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="ミニプロジェクトの流れ: 合成集合写真の生成、Haar検出、IoU突合せ、整列と埋め込み、DBSCANクラスタリング、人物アルバム出力。検出が弱いときは正解枠にフォールバックして整列へ合流する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">写真群 → 人物アルバム自動仕分けの流れ</text><rect x="24" y="64" width="178" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="242" y="64" width="178" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="460" y="64" width="178" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="460" y="216" width="178" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="242" y="216" width="178" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="24" y="216" width="178" height="60" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="113" y="92" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">① 合成集合写真</text><text x="113" y="111" text-anchor="middle" font-size="11" fill="#71717a">6人×4枚=24顔</text><text x="331" y="92" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">② Haar 検出</text><text x="331" y="111" text-anchor="middle" font-size="11" fill="#71717a">detectMultiScale</text><text x="549" y="92" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">③ IoU 突合せ</text><text x="549" y="111" text-anchor="middle" font-size="11" fill="#71717a">recall・precision</text><text x="549" y="244" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">④ 整列 → 埋め込み</text><text x="549" y="263" text-anchor="middle" font-size="11" fill="#71717a">crop + ResNet・L2正規化</text><text x="331" y="244" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">⑤ DBSCAN</text><text x="331" y="263" text-anchor="middle" font-size="11" fill="#71717a">eps掃引・NMI最大</text><text x="113" y="244" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">⑥ 人物アルバム</text><text x="113" y="263" text-anchor="middle" font-size="11" fill="#71717a">mini_album + JSON</text><line x1="202" y1="94" x2="236" y2="94" stroke="#71717a" stroke-width="2"/><polygon points="242,94 232,89 232,99" fill="#71717a"/><line x1="420" y1="94" x2="454" y2="94" stroke="#71717a" stroke-width="2"/><polygon points="460,94 450,89 450,99" fill="#71717a"/><line x1="549" y1="124" x2="549" y2="210" stroke="#71717a" stroke-width="2"/><polygon points="549,216 544,206 554,206" fill="#71717a"/><line x1="460" y1="246" x2="426" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="420,246 430,241 430,251" fill="#71717a"/><line x1="242" y1="246" x2="208" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="202,246 212,241 212,251" fill="#71717a"/><line x1="331" y1="124" x2="469" y2="210" stroke="#c2410c" stroke-width="1.6" stroke-dasharray="6 4"/><polygon points="478,216 466.9,214.9 472.2,206.5" fill="#c2410c"/><text x="250" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">⑦ 検出が弱いとき</text><text x="250" y="181" text-anchor="middle" font-size="11.5" fill="#c2410c">正解枠にフォールバック</text></svg><figcaption><b>章末ミニプロジェクト</b>は、入口の合成集合写真から <b>① 生成 → ② Haar 検出 → ③ IoU 突合せ(recall/precision) → ④ 整列・埋め込み(L2正規化) → ⑤ DBSCAN → ⑥ 人物アルバム＋指標 JSON</b> の順に一気通貫で流れます。<b>⑦ 頑健性</b>として、検出がドメインギャップで弱い(recall が低い)ときは<b>正解枠の切り出し</b>に自動フォールバックして ④ に<b>合流</b>し、後段を必ず完走(<code>exit 0</code>)させます。</figcaption></figure>

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

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/30_face_detection_recognition/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. ベクトルを L2 ノルム 1 に正規化して返す。ゼロベクトルは 0 除算を避けてそのまま返す（`l2_normalize_vec` の TODO）。
2. 2 つのベクトルのコサイン類似度（内積 ÷ ノルムの積）を返す（`cosine_similarity` の TODO）。
3. コサイン距離（= 1 − コサイン類似度）を返す（`cosine_distance` の TODO）。
4. 人物 ID 配列から全ペア i<j を作り、同一人物か否かの genuine/impostor を付けた `(i, j, is_genuine)` のリストを返す（`make_pair_labels` の TODO）。
5. しきい値で `score>=thr` を「同一人物」と判定したときの FAR と TAR を返す（`far_tar_at_threshold` の TODO）。
6. ROC 曲線の FAR/TAR 配列から、FAR と FRR が最も近い点の EER を返す（`eer_from_curves` の TODO）。
7. ターゲット FAR 以下を満たす中で最大の TAR（TAR@FAR）を返す（`tar_at_far` の TODO）。
8. probe×gallery のコサイン類似度行列から、最も似たギャラリーが本人と一致した割合（rank-1 精度）を返す（`rank1_accuracy` の TODO）。
9. 各予測クラスタを多数派の真ラベルに割り当てたときの全体正解率（purity）を返す（`purity` の TODO）。
10. 2 つの矩形 `[x, y, w, h]` の IoU を返す。重ならなければ 0（`iou_xywh` の TODO）。

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

## 💡 実践ユースケース集

本章の「検出」は、**「誰が誰か」を当てる**だけでなく、その逆 ——**「誰か分からなくする」
匿名化**——にもそのまま使えます。`mini_project.py`（検出→埋め込み→クラスタリングで人物を
**特定**する統合課題）とは目的が真逆の、**そのまま製品になりうる小ツール**を挙げます。共通の
作り方は「**顔を検出 → その領域を読めなく潰す（ぼかし/モザイク/黒塗り）→ 自然に合成**」です。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="顔の匿名化パイプライン: 入力画像を顔検出してROIを得て、ぼかし・モザイク・黒塗りのいずれかで潰し、楕円アルファ合成で匿名化画像に戻す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">顔の匿名化パイプライン — 検出 → 潰す → 合成</text><rect x="16" y="134" width="104" height="62" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="176" y="134" width="98" height="62" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="326" y="50" width="128" height="44" rx="6" fill="#fff7ed" stroke="#f97316" stroke-width="1.8"/><rect x="326" y="143" width="128" height="44" rx="6" fill="#fff7ed" stroke="#f97316" stroke-width="1.8"/><rect x="326" y="236" width="128" height="44" rx="6" fill="#fff7ed" stroke="#f97316" stroke-width="1.8"/><rect x="506" y="134" width="126" height="62" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="68" y="163" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">入力画像</text><text x="68" y="181" text-anchor="middle" font-size="11" fill="#71717a">集合写真・実写</text><text x="225" y="163" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">顔を検出</text><text x="225" y="181" text-anchor="middle" font-size="11" fill="#71717a">Haar・枠を拡張</text><text x="390" y="77" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">ぼかし (blur)</text><text x="390" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">モザイク (pixelate)</text><text x="390" y="263" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">黒塗り (box)</text><text x="569" y="160" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">匿名化画像</text><text x="569" y="180" text-anchor="middle" font-size="11" fill="#71717a">楕円αで自然合成</text><line x1="120" y1="165" x2="170" y2="165" stroke="#71717a" stroke-width="2"/><polygon points="176,165 166,160 166,170" fill="#71717a"/><text x="148" y="156" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">検出</text><line x1="274" y1="165" x2="300" y2="165" stroke="#71717a" stroke-width="2"/><line x1="300" y1="72" x2="300" y2="258" stroke="#71717a" stroke-width="2"/><line x1="300" y1="72" x2="320" y2="72" stroke="#71717a" stroke-width="2"/><polygon points="326,72 316,67 316,77" fill="#71717a"/><line x1="300" y1="165" x2="320" y2="165" stroke="#71717a" stroke-width="2"/><polygon points="326,165 316,160 316,170" fill="#71717a"/><line x1="300" y1="258" x2="320" y2="258" stroke="#71717a" stroke-width="2"/><polygon points="326,258 316,253 316,263" fill="#71717a"/><line x1="454" y1="72" x2="480" y2="72" stroke="#71717a" stroke-width="2"/><line x1="454" y1="165" x2="480" y2="165" stroke="#71717a" stroke-width="2"/><line x1="454" y1="258" x2="480" y2="258" stroke="#71717a" stroke-width="2"/><line x1="480" y1="72" x2="480" y2="258" stroke="#71717a" stroke-width="2"/><line x1="480" y1="165" x2="500" y2="165" stroke="#71717a" stroke-width="2"/><polygon points="506,165 496,160 496,170" fill="#71717a"/><text x="390" y="306" text-anchor="middle" font-size="11.5" fill="#52525b">METHOD で 1 方式を選ぶ</text></svg><figcaption>匿名化ツールに共通の流れです。<b>入力画像</b>から <b>顔を検出</b>(Haar・枠を少し拡張)して顔 <b>ROI</b> を取り出し、その領域を <b>ぼかし / モザイク / 黒塗り</b> の <code>METHOD</code> で<b>潰し</b>、最後に<b>楕円の輪郭に沿った α 合成</b>で<b>匿名化画像</b>へ自然に戻します。検出が<b>取りこぼした顔はそのまま漏れる</b>ので、匿名化では<b>再現率(recall)を最優先</b>に検出器を選びます。</figcaption></figure>

### ① 顔の自動ぼかし匿名化ツール（`use_case.py`・動く出発点）

- **何に使うか**: SNS 投稿前の通行人の顔消し、街頭・イベント写真の公開用マスキング、ストリート
  ビュー的サービスの顔ぼかし、被写体の同意が取れていない人物の保護。
- **作り方の要点**: Haar で顔矩形を得て、各枠を少し広げて（額・あごまで覆う）ROI を切り出し、
  **ガウシアンぼかし**でその場を潰します。`USE_ELLIPSE=True` で**顔の楕円輪郭に沿って α 合成**
  するので、矩形の角に背景が残らず自然に仕上がります。方式は `METHOD` を `"blur"` /
  `"pixelate"`（モザイク）/ `"box"`（黒塗り）で切り替えられます。
- **注意**: Haar は**実写で学習**されているため、合成顔や横顔・小顔・逆光は**取りこぼし**ます。
  匿名化は「**取りこぼした顔は素通しで漏れる**」のが最大のリスクなので、**再現率(recall)を
  最優先**に検出器を選びます（DNN/MediaPipe へ差し替え推奨）。弱いモザイクは超解像で部分的に
  復元され得るため、**強度を「復元困難」な水準**に固定するのも実務上の勘所です。

```bash
uv run python lectures/30_face_detection_recognition/use_case.py
# → 合成 or data/30_face_detection_recognition/ の写真の顔を自動でぼかし、
#   Before/After モンタージュ・方式比較図・レポート JSON を outputs/30_.../ に保存
```

- **data 配置**: `data/30_face_detection_recognition/` に**画像**（`*.jpg` / `*.jpeg` / `*.png`
  / `*.bmp`）を置くと実入力で動きます（例: `data/30_face_detection_recognition/street.jpg`）。
  無ければ**合成集合写真**で必ず完走します。合成顔は検出が弱いことがあるので、その時だけ顔の
  **正解位置**にフォールバックして匿名化を実演します（実写は検出ドリブンで動きます）。
- **拡張アイデア**: 検出を **OpenCV DNN(res10)／MediaPipe／RetinaFace** に替えて取りこぼしを
  減らす、`cv2.VideoCapture` ＋ `VideoWriter` で**動画の顔マスキング**にする、本章の
  `FaceEncoder` で埋め込みを取って**特定の人物だけ残す/消す**（同意者は素通し等）、顔以外
  （ナンバープレート・名札）も検出して**総合匿名化**する、など。

### ② 動画・防犯カメラのリアルタイム顔マスキング

- **何に使うか**: 監視カメラ映像の公開・第三者共有、配信動画の通行人ぼかし、ドライブレコーダ
  映像の匿名化。録画の**保存前**に顔を潰しておけば、生の顔データを保持せずに済みます。
- **作り方の要点**: `use_case.py` の `anonymize_image` を **1 フレームごと**に呼び、`cv2.VideoCapture`
  で読んで `cv2.VideoWriter` で書き出すだけ。フレーム間で検出が途切れると一瞬素通しになるので、
  **追跡(モジュール 28)で顔 ID を保ち、検出が落ちたフレームは前回の枠を流用**して穴を塞ぎます。
- **注意**: CPU で実時間に追いつかない時は、`cv2.resize` で縮小してから検出し座標を戻す／N
  フレームに 1 回だけ検出して間を追跡で補間します。**1 フレームでも漏れたら匿名化失敗**なので、
  取りこぼし時は「画面全体を弱くぼかす」等の**フェイルセーフ**を入れると安全側に倒せます。

### ③ データセット公開時の被写体保護（プライバシー by デザイン）

- **何に使うか**: 自前で集めた画像データセットを論文・OSS で公開する際に、写り込んだ**第三者の
  顔を一括で匿名化**してから配布する（GDPR 等の生体情報規制への配慮）。
- **作り方の要点**: フォルダ内の全画像に検出→ぼかしをバッチ適用し、**元画像は残さず匿名化版
  だけを出力**します。「**どの顔も潰し漏れがないこと**」を担保するため、検出数を画像ごとに
  記録し（`use_case_report.json` のように）、0 件や極端に少ない画像を**人手レビュー**に回します。
- **注意**: 匿名化は**不可逆**であるべきです（元に戻せるとマスクの意味がない）。可逆な弱い処理は
  避け、メタデータ（EXIF の GPS・人物名タグ）も併せて除去します。匿名化の**精度自体に
  バイアス**（特定の肌色・年齢で取りこぼしが増える）が無いか、属性ごとに recall を点検します。

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
# 実践ユースケース: 顔の自動ぼかし（匿名化）ツール
uv run python lectures/30_face_detection_recognition/use_case.py
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