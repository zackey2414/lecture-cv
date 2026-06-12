# 第29回 動画理解・行動認識 — クリップサンプリング / r3d_18(3D CNN) / VideoMAE(Transformer) / top-k

> トラック: **動画・追跡** ／ レベル: **上級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers ほか）。CPU だけで完走します（初回のみ r3d_18 ~127MB / VideoMAE ~330MB の重みを自動ダウンロード、以降はキャッシュから即起動）。動画 I/O は **`cv2.VideoCapture` に統一**します（torchvision 0.26 で内蔵デコーダ `read_video` が廃止されたため）。`pipeline('video-classification')` が要求する **PyAV(`av`) は既定依存に含めず**、未導入なら「概念紹介＋手書き等価コード」に切り替えます（＝必ず exit 0）。入力動画はすべて**合成生成**（動く図形）で、ネットもデータセット DL も不要です。

## 🎯 この章のゴール

第27回（深度・姿勢・フロー）と第28回（多物体追跡）では、「1 枚の画像」や「フレーム間の対応」を扱ってきました。本章のテーマは、その先にある**「複数フレーム（＝クリップ）をひとまとまりに見て、そこで起きている『行動』を当てる」動画理解**です。`tossing coin`（コイン投げ）や `golf putting`（パター）が示すように、行動は**1 枚の静止画では決まらず、時間方向の動きによって決まります**。したがって入力は、画像の `(C,H,W)` ではなく、時間軸 `T` を足した **`(C,T,H,W)`**（バッチを入れると 5 次元 `(N,C,T,H,W)`）になります。この「時間が 1 次元増える」ことこそ、本章のすべての出発点です。

到達点は5つです。第一に、**動画モデルが受け取る入力テンソルの形 `(N,C,T,H,W)`** と、画像分類との違いを説明できること。第二に、**フレームサンプリング（clip_len＝何枚／frame_rate＝何フレームおき）と専用正規化**を正しく行い、`torchvision r3d_18`（3D CNN, Kinetics-400）と `transformers VideoMAE`（Transformer）の**両方**を CPU で動かせること。第三に、**`VideoMAEImageProcessor` の前処理**と **r3d_18 の手書き前処理**を、それぞれの正準作法で書けること。第四に、動画 I/O を **`cv2.VideoCapture`** で行い（`read_video` 廃止の代替）、mp4 からフレームを抽出して推論できること。第五に、**clip-level の top-1/top-5 accuracy** で評価し、**前処理（clip_len/frame_rate/正規化）を崩すとスコアが壊れる**ことを、自分の手で数値化できることです。

本章のスクリプトはすべて、ネットもデータセットも使わずに完走できるよう、入力を**合成クリップ**（cv2 で描いた「動く図形」）として生成します。ただし、合成クリップには**本物の Kinetics ラベルが存在しません**。そこで評価では、「**正しい前処理で得たモデルの top-1 を、各クリップの基準ラベル（pseudo-GT）とみなす**」という方針を採り、前処理を崩したときに pseudo-GT との一致率がどれだけ落ちるかを測ります（第8節で詳述）。これは「絶対精度」ではなく「**前処理感度（robustness）**」の定量化にあたりますが、本章の核心メッセージ――**「行動認識は前処理を間違えると無意味な出力になる」**――を再現可能な数値で体感するには、むしろ最適です。実測でも、正しい前処理を基準にすると、**正規化を外すだけで top-1 一致率が 1.00→0.50** へ半減し、**動きを消す（同一フレーム複製）と 0.08** まで崩れました。

---

## 1. 動画理解の地図 — 画像分類との決定的な違いは「時間軸 T」

画像分類（第13回）は `(N,C,H,W)` のテンソルを受け取り、1 枚の画像から 1 つのクラスを出しました。これに対し、**行動認識は時間軸 `T` を足した `(N,C,T,H,W)` を受け取り、クリップ全体から 1 つの行動クラスを出します**。なぜ時間が要るのか――それは、行動が「動き」で定義されるからです。たとえば「ドアの前に立つ人」の 1 枚からは、「ドアを開ける」のか「閉める」のか「ただ立っている」のか区別できません。`tossing coin` も同じで、コインが**上がって落ちる**という時間変化を見て初めて当てられます。つまり、静止画では原理的に区別できない行動を扱うことこそ、動画理解の本質なのです。

この「時間をどうモデルに入れるか」という設計の違いによって、アーキテクチャは大きく3系統に分かれます。**(1) 3D CNN**（本章の `r3d_18`）は、2D 畳み込みを時間方向にも広げた **3D 畳み込み**で、空間と時間を同時に畳み込みます。`(C,T,H,W)` をそのまま 3D カーネルで舐めるイメージで、実装が素直なうえ、小型なら CPU でも軽く動きます。**(2) Video Transformer**（本章の `VideoMAE`）は、各フレームをパッチ列に分解し、**時空間のトークン**として自己注意でまとめます。大規模事前学習（VideoMAE はマスク再構成による自己教師あり事前学習）と相性が良く精度が高い反面、CPU では重くなります。**(3) 2D CNN ＋ 時間集約**（TSN/SlowFast 系の発想）は、各フレームを 2D で処理し、後段で平均や別経路によって時間をまとめる折衷案です。本章では (1) と (2) を実際に動かし、最後に三者の使い分けを整理します。

3 系統のいずれにも共通する**最重要の作法が、「フレームサンプリングと専用正規化」**です。動画は何百フレームもありますが、モデルが受け取るのは固定枚数（VideoMAE・r3d_18 とも標準で `clip_len=16` 枚）だけです。そのため、**「全長から 16 枚をどう選ぶか（uniform か、何フレームおきか）」**と、**「モデル固有の mean/std でどう正規化するか」**を間違えると、どんなに良いモデルでも出力は無意味になってしまいます。本章の半分は、実はこの「前処理」に費やします。まずはサンプリングの理論から見ていきましょう。

## 2. フレームサンプリング — clip_len と frame_rate(stride) の理論

動画クリップから固定枚数を取り出す方法は、大きく2つあります。**(A) 等間隔（uniform）サンプリング**は、全長 `total` フレームを端から端まで均等に `clip_len` 枚へ間引く方法で、`np.linspace(0, total-1, clip_len)` を丸めて使います。「動画全体を広く薄く見たい」とき向きで、長さの違う動画を一律に扱えるのが利点です。本講座の `uniform_indices(32, 8)` は `[0,4,9,13,18,22,27,31]` を返します。一方の **(B) ストライド（clip_len + frame_rate）サンプリング**は、ある開始点から `frame_rate` フレームおきに `clip_len` 枚を連続的に取る方法で、`idx = start + arange(clip_len)*frame_rate` で表されます。「短い時間窓を密に・一定速度で」見たいとき向きで、Kinetics 系モデルの学習時サンプリングに近い手法です。`strided_indices(32, 8, frame_rate=4)` は中央寄せで `[1,5,9,13,17,21,25,29]` を返します。

ここで効くノブが、**`clip_len`（何枚）** と **`frame_rate`（何フレームおき＝ストライド）** です。`clip_len` を増やすほど時間解像度は上がりますが、その分だけ計算は重くなります。また `frame_rate` を上げる（粗く取る）ほど**長い時間範囲**をカバーできる反面、速い動きは取りこぼします。逆に `frame_rate=1`（密に取る）にすると**短い範囲**しか見えず、ゆっくりした行動の全体像を逃します。つまり「`clip_len × frame_rate ≒ 何秒ぶんを見るか（受容時間窓）」というトレードオフであり、ここでは**学習時と推論時のサンプリングを揃える**のが鉄則です（学習が `frame_rate=4` なら推論も合わせます）。なお、末尾を超えるインデックスは最終フレームへ**クランプ**し、短い動画でも処理が落ちないようにします。

サンプリングを間違えるとどうなるかは、本章のミニプロジェクトで定量化します。実測では、正しい前処理（uniform 16 枚）を基準にすると、**`clip_len` を 16→8→4→2 と減らすにつれ top-1 一致率が 1.00→0.17→0.17→0.00** と崩れ、**`frame_rate` を 1→2→4 と変えると 0.50→0.83→0.33** と山なりに動きました（`frame_rate=2` が最も基準に近い）。ここから、サンプリングは「些細な前処理」どころか、**予測を決定づける主役級のハイパーパラメータ**だと分かります。続いては、サンプリングと並ぶもう一つの主役、専用正規化を見ていきます。

## 3. 専用正規化 — なぜ ImageNet の mean/std ではダメなのか

画像分類では `[0.485,0.456,0.406]/[0.229,0.224,0.225]`（ImageNet 統計）で正規化したことを覚えているでしょう。**動画モデルは、これとは別の専用 mean/std を使います**。たとえば `r3d_18`（Kinetics-400 で学習）は `mean=[0.43216,0.394666,0.37645]`、`std=[0.22803,0.22145,0.216989]` で、入力解像度も **112×112** です。一方の VideoMAE は、ImageNet 統計に近い `mean=[0.485,0.456,0.406]/std=[0.229,0.224,0.225]` で **224×224** を使います。**学習時に使った統計と同じもので正規化しないと、モデルが見たことのない入力分布を渡すことになり、出力が崩れます**。「正規化なんてどれも同じ」ではなく、**モデルごとに正しい値が決まっている**――これを取り違えることこそ、動画認識で最も多い事故です。

正規化の手順自体は単純で、`[0,1]` にスケールした画素から `(x - mean) / std` を**チャンネル別**に引いて割るだけです。重要なのは、計算そのものよりも**「どの mean/std を使うか」と「正規化を忘れない／二重にかけない」**ことです。`r3d_18` に ImageNet 統計を使う、`[0,255]` のまま正規化する、正規化を丸ごと飛ばす――これらはどれも「壊れた前処理」です。本講座の `preprocess_for_r3d` は `normalize` フラグと `mean/std` 引数を持ち、こうしたわざと壊した前処理を再現できるようにしてあります。

この効果を実測で見てみましょう（第8節 `03` の結果）。正しい前処理を基準にすると、**正規化を丸ごと外したとき top-1 一致率は 1.00→0.50** へ、**ImageNet 統計を誤用したとき 0.67** へ落ちました（top-5 では持ち直す＝大まかには似た方向を向くものの、top-1 はズレる、ということです）。単一クリップで見ても、`02` のコイン投げクリップでは、「正規化あり」が `tossing coin 0.55 / golf putting 0.27 / frisbee 0.16` という鋭い分布になるのに対し、「正規化なし」では `tossing coin 0.33 / finger snapping 0.13 / juggling 0.11` と**確信度がのっぺり**してしまいます。つまり専用正規化は「精度を少し上げる調整」ではなく、**モデルを正しく動かすための前提条件**だと心に刻んでください。次節からは、実際のモデルを 1 つずつ動かしていきます。

## 4. torchvision r3d_18 — 3D CNN の正準API と (N,C,T,H,W)

最初に動かすのは **`torchvision.models.video.r3d_18`**（ResNet-3D 18 層、Kinetics-400 で学習）です。小型で CPU でも軽いため、本章の「動かして理解する」主役を担います。ロードは画像モデルと同じ流儀で、**`R3D_18_Weights.DEFAULT`** から重みとメタ情報（400 クラスの `categories`）を取り、`r3d_18(weights=weights).eval()` で評価モードにします。初回のみ ~127MB を `~/.cache/torch/hub` へダウンロードし、以降はキャッシュから即起動します。なお推論時は、必ず `torch.inference_mode()` で勾配を切りましょう。

3D CNN の肝は、**入力テンソルの形**です。`r3d_18` は **`(N, C, T, H, W)`** を受け取ります（画像の `(N,C,H,W)` に時間 `T` が挟まる形です）。本講座の手書き前処理 `preprocess_for_r3d` は、`(T,H,W,3)` の uint8 クリップを、(1) フレーム選択 → (2) 112×112 へリサイズ → (3) `[0,1]` スケール → (4) Kinetics 専用正規化（リサイズとスケールは線形なので順序は結果にほぼ影響しません） → (5) `(T,H,W,C)→(C,T,H,W)` へ並べ替え（`permute(3,0,1,2)`）→ (6) バッチ次元を足す、の順で `(1,3,16,112,112)` に整えます。torchvision には公式の前処理 `weights.transforms()` もあり、こちらは `(T,C,H,W)` uint8 を受け取って、内部で `resize(128×171)→center-crop(112)` を行い `(C,T,H,W)` を返します（リサイズの流儀が違うため手書きと画素は厳密には一致しませんが、最終形状と正規化統計は同じです）。`02_r3d18_action.py` では、この両者を並べて確認します。

```python
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights

weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights).eval()
categories = weights.meta["categories"]          # 400 個の行動名

x = preprocess_for_r3d(clip_rgb, uniform_indices(clip_rgb.shape[0], 16))  # (1,3,16,112,112)
with torch.inference_mode():
    logits = model(x)                            # (1, 400)
top5 = logits.softmax(-1)[0].topk(5)             # 上位5クラス
```

実測（合成「円が左→右」クリップ）では、正しい前処理のもとで top-1 が `tossing coin (0.55)` となり、続いて `golf putting (0.27)` / `catching or throwing frisbee (0.16)` と並びました。合成図形なので「正解」ではありませんが、モデルが**時間方向の並進運動に一貫して反応している**ことは読み取れます。同じクリップを mp4 に書き出し、`cv2.VideoCapture` で読み戻しても top-1 は `tossing coin` で一致しました（第6節の I/O 経路の検証にあたります）。続いては、より大きく高精度な Transformer 系、VideoMAE を見ていきます。

## 5. VideoMAE — Transformer 系の正準API と transformers v5 の落とし穴

**`VideoMAE`**（`MCG-NJU/videomae-base-finetuned-kinetics`）は、各フレームをパッチに刻み、**時空間トークン**として自己注意で処理する Video Transformer です。使い方は HuggingFace の他の画像モデルと同じく、**`VideoMAEImageProcessor`（前処理）＋ `VideoMAEForVideoClassification`（本体）** の二段構えです。processor は**「フレーム（HWC uint8）のリスト」**を受け取り、リサイズ・専用正規化・テンソル化までを一手に引き受け、**`pixel_values` を `(1, T, C, H, W)`**（r3d_18 の `(1,C,T,H,W)` とは T と C の順が違う点に注意）で返します。`clip_len` はモデル設定 `model.config.num_frames`（通常 16）に合わせ、こちらで 16 枚をサンプリングして渡します。

```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

processor = VideoMAEImageProcessor.from_pretrained(name)
model = VideoMAEForVideoClassification.from_pretrained(name).eval()

idx = uniform_indices(clip_rgb.shape[0], model.config.num_frames)   # 16 枚
frames = [clip_rgb[i] for i in idx]                                  # list of (H,W,3) uint8
inputs = processor(frames, return_tensors="pt")                     # pixel_values: (1,16,3,224,224)
with torch.inference_mode():
    logits = model(**inputs).logits                                 # (1, 400)
```

ここで、本章最大の**落とし穴**を紹介します。**transformers v5 で VideoMAE をロードすると、`q_bias`/`v_bias` という特殊なバイアスが取りこぼされ、`query.bias`/`value.bias` が 0 のまま**になることがあります（ロード時に `UNEXPECTED: q_bias/v_bias` と `MISSING: query.bias/...` という報告が出ます）。これは、VideoMAE の自己注意が「`q_bias` と `v_bias` だけを学習し、`k_bias` は 0 に固定する」という特殊構造を持つためで、v5 が attention を `query/key/value` の Linear に分解した際に、この対応付けが正しく移らないことから起きます。バイアスが 0 でもそれらしく動いてしまうため見逃しがちですが、この状態では**本来の精度は出ません**。そこで本講座の `load_videomae` は、元のチェックポイント（safetensors）から `q_bias`/`v_bias` を直接読み、`query.bias`/`value.bias` へ**手当て**します（`01` の出力で「手当てしたレイヤ数=12」「query.bias の絶対値和=260.74」と確認できます）。「ロード報告は読む」「バイアスが 0 になっていないか疑う」――この一手間こそ、Transformer 系を正しく使う作法です。実測では、手当て後の top-1 は `golf putting (0.26)` でした。

VideoMAE は r3d_18 より大きく（~330MB、CPU では 1 クリップあたり数秒）、精度も高い一方で動作は重くなります。そこで本講座では、**VideoMAE を「概念＋実演」、r3d_18 を「多数クリップを回す主力」**と役割分担させます。`01_videomae_action.py` は、モデルが落とせない/重い環境でも `try/except` によって必ず exit 0 になるよう設計してあります。続いては、両モデルに共通する動画 I/O と高レベル API を見ていきます。

## 6. 動画 I/O — cv2.VideoCapture（read_video 廃止）と pipeline の位置づけ

動画ファイルからフレームを取り出す方法としては、かつて `torchvision.io.read_video` が定番でしたが、**torchvision 0.26 で内蔵デコーダが廃止**され、現在は使えません。そこで本講座は、動画 I/O を **`cv2.VideoCapture` に統一**します。`cap = cv2.VideoCapture(path)` で開き、`cap.read()` を `False` が返るまで回してフレームを集め、`cap.release()` で閉じる――手順はこれだけです。注意点は、**cv2 が BGR 順**である点です。torch/transformers/matplotlib はいずれも RGB 前提なので、読み出したフレームは `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` で RGB に直します（書き出す `cv2.VideoWriter` 側は、逆に RGB→BGR とします）。本講座の `write_clip_mp4`/`read_clip_mp4` は、この変換込みのラウンドトリップを提供し、`mp4v` コーデックによって headless 環境でも動きます。

```python
import cv2
cap = cv2.VideoCapture(path)                     # mp4 を開く（read_video の代替）
frames = []
while True:
    ok, frame = cap.read()                       # BGR で 1 枚ずつ
    if not ok:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # RGB へ
cap.release()
```

高レベル API の **`pipeline('video-classification')`** にも触れておきます。これは「mp4 のパス → top-k ラベル」を 1 行で返す便利な窓口ですが、**内部の動画デコードに PyAV（`av`）を必要**とします。本講座は、依存衝突（numpy2 系との相性など）と巨大化を避けるため **`av` を既定依存に含めない**方針です。そのため `01` では `try: import av` で存在を確認し、**未導入なら「概念紹介＋手書き等価コード（processor＋model）」に切り替え**ます。pipeline がやっているのは、結局「デコード→サンプリング→processor→model→softmax→top-k」であり、本章で手書きした経路と等価です。使いたい場合は `uv add --group video av` で追加できますが、**学習目的では、中身が見える手書き経路を通すこと**を勧めます（pipeline はブラックボックスになりがちだからです）。それでは、いよいよ評価に進みます。

## 7. 評価 — clip-level の top-1/top-5 と「前処理を壊す」実験

行動認識の標準評価は、**clip-level の top-1 / top-5 accuracy** です。top-1 は「最尤クラスが正解か」を、top-5 は「上位 5 クラスに正解が入るか」を見ます。Kinetics-400 のように**意味が近いクラスが多い**（`juggling balls` と `contact juggling` など）データでは、top-1 が厳しすぎることもあるため、**top-5 を併記**するのが慣例です。実装は素朴で、`np.argsort(-logits, axis=1)[:, :k]` で各サンプルの上位 k クラスを取り、そこに `gt` が含まれる割合を数えるだけです（`topk_accuracy`）。さらに**混同行列**（行=正解、列=予測）を併記すると、「どのクラスがどのクラスと取り違えられるか」まで見えてきます。

ここで、本章特有の工夫が必要になります。**合成クリップには本物の Kinetics ラベルが無い**ため、素直には top-1 accuracy を測れないからです。そこで `03_action_topk_eval.py` は、**「正しい前処理（uniform 16 枚＋Kinetics 正規化）でのモデルの top-1 を、各クリップの基準ラベル（pseudo-GT）とみなす」**方針を採ります。そのうえで前処理をさまざまに崩し、**pseudo-GT に対する top-1/top-5 一致率**を測ります。正しい前処理なら定義上 1.00 となり、崩すほど一致率が落ちる――これは「絶対精度」ではなく「**前処理感度（robustness）**」の定量化ですが、本章の主張「前処理を誤ると出力が壊れる」を**再現可能な数値**で示すには最適です。混同行列は、pseudo-GT に出現するクラスだけを行に、列にそれら＋`other`（出現外のクラス）を置いた小さな行列とし、崩れた予測が**どれだけ別物（other）へ散るか**を可視化します。

実測（6 種の動き×2 seed＝12 クリップ）の結果が下表です。正しい前処理 `correct` は、当然ながら 1.00/1.00 です。これに対し、**正規化を外す `no_normalize` では top-1 が 0.50** へ半減し、**ImageNet 統計を誤用する `imagenet_norm` では 0.67**、そして**動きを消す `static_frame`（同一フレームを 16 枚複製）では 0.08** まで崩壊しました。とりわけ最後の結果は決定的で、**「行動認識は『動き』を消すと完全に無力化する」**――時間情報こそがこのタスクの本体だと、数字が物語っています。

| 前処理バリアント | 内容 | top-1（vs pseudo-GT） | top-5 |
| --- | --- | --- | --- |
| `correct` | uniform 16 枚 ＋ Kinetics 正規化（基準） | **1.000** | **1.000** |
| `no_normalize` | 正規化を丸ごと省略 | 0.500 | 0.833 |
| `imagenet_norm` | 画像用 ImageNet の mean/std を誤用 | 0.667 | 1.000 |
| `static_frame` | 中央フレームを 16 枚複製（動きゼロ） | **0.083** | 0.500 |
| `front_window` | 先頭だけを連続サンプリング（動きの一部のみ） | 0.500 | 1.000 |

```python
def topk_accuracy(logits, gt_indices, k=1):
    topk = np.argsort(-logits, axis=1)[:, :k]          # 各行の上位 k クラス
    return float(np.mean([gt in row for gt, row in zip(gt_indices, topk)]))

correct_logits = run(model, clips, preprocess="correct")
pseudo_gt = correct_logits.argmax(1)                   # 正しい前処理の top-1 を基準ラベルに
broken = run(model, clips, preprocess="no_normalize")
print(topk_accuracy(broken, pseudo_gt, k=1))           # → 0.50（崩れた）
```

この「pseudo-GT を基準に前処理感度を測る」枠組みは、実データが手元にあれば**そのまま本物の top-1/top-5 accuracy に置き換わります**（pseudo-GT を本物の GT に差し替えるだけです）。評価コードの骨格は変わらないので、合成データで仕組みを固めておけば、そのまま実運用へ移せます。次は、ここまでの要素を統合するミニプロジェクトです。

## 8. 実務の使い分け — 3D CNN vs Video Transformer vs (2D＋時間)

最後に、本章で扱った 2 系統（＋折衷案）を実務目線で整理します。まず **3D CNN（r3d_18, X3D, SlowFast）**は、空間と時間を 3D 畳み込みで同時に扱い、**小型なら CPU でも軽い**のが強みです。エッジ・リアルタイム寄りの用途や、まず動かして当たりを付けたいときの第一候補になります。弱みは、超長時間の文脈を取りにくいことです。次に **Video Transformer（VideoMAE, TimeSformer, ViViT）**は、自己注意で**長距離の時空間依存**を捉え、大規模事前学習と相性が良く**精度が高い**反面、**CPU では重く**なります。クラウド GPU で精度を追うときの主力です。最後に **2D CNN＋時間集約（TSN）**は、各フレームを 2D で安く処理し後段で平均する折衷案で、**実装が軽く速い**ものの、フレーム平均では捉えにくい「順序が効く行動」には弱いのが難点です。

選択の初期方針としては、**「速度・コスト優先かつ CPU なら 3D CNN（小）」「精度優先で GPU が使えるなら Video Transformer」「とにかく軽く大量に回すなら 2D＋時間集約」**と大づかみに捉えると良いでしょう。そして、どの系統でも共通して効くのが、本章で繰り返してきた**「学習時と同じサンプリング（clip_len/frame_rate）と専用正規化を、推論でも厳守する」**ことです。さらに精度を一段上げる定番が **multi-clip / multi-crop の TTA（test-time augmentation）**で、1 本の動画から複数クリップ（時間方向に複数窓、空間方向に複数クロップ）を取り、logits を平均します。本章は単一クリップ推論に絞りましたが、評価の枠組み（top-1/top-5・混同行列）は、そのまま multi-clip にも乗せられます。

実装面の注意も一つ挙げておきます。本章の `r3d_18` 例では `clip_len` を可変（2〜16）にして掃引できましたが、これは 3D CNN が**時間方向を畳み込みで吸収する**ため、時間長が変わっても動くからです。一方で **VideoMAE は `num_frames=16` を前提に位置埋め込みを持つ**ので、原則として 16 枚固定で渡します（枚数を変えると位置埋め込みと整合しなくなります）。このように「モデルが時間長の変化を許すか」はアーキテクチャ依存であり――ここもまた、「前処理はモデルごとに正しい形が決まっている」という本章の通奏低音の一例です。

## 🛠 章末ミニプロジェクト — 「mp4 → フレーム抽出 → 行動推定 → 評価」一気通貫

`mini_project.py` は、本章の要素（cv2 動画 I/O ／ サンプリング ／ 専用正規化 ／ r3d_18 推論 ／ top-1/top-5 評価 ／ 混同行列）を**1 本のパイプライン**に統合します。**ステージA【動画 I/O】**で合成行動クリップを mp4 に書き出し `cv2.VideoCapture` で読み戻す（`read_video` 廃止の代替経路をそのまま実演）、**ステージB【基準作り】**で正しい前処理の top-1 を pseudo-GT に、**ステージC【サンプリング掃引】**で `clip_len`（2/4/8/16）と `frame_rate`（1/2/4）を変えながら top-1/top-5 一致率を測り、**ステージD【別モデル（任意）】**で VideoMAE が使えれば数クリップで r3d_18 と top-1 を突き合わせ、**ステージE【レポート】**でサンプリング掃引のサマリ図と総合 json を出力します。

実測では、ステージC の `clip_len` 掃引が **2→0.00, 4→0.17, 8→0.17, 16→1.00**（短いほど崩れる）、`frame_rate` 掃引が **1→0.50, 2→0.83, 4→0.33**（基準＝uniform に近い `rate=2` が最良で、粗すぎる `rate=4` では悪化）となり、**サンプリングが予測を決定づける**ことを鮮明に示しました。`clip_len=2` の混同行列を見ると、予測が `other` 列に大きく散っており、「2 枚では動きが読めない」様子が一目で分かります。ステージD では、r3d_18 と VideoMAE が別クラスを出すこともありますが（例: r3d_18 `tossing coin` vs VideoMAE `juggling balls`）、これは**合成データに唯一の正解が無い**ことの裏返しです。なお、**どのモデルが落ちてもパイプラインは止めず、できた範囲でレポートを出す**設計なので、必ず exit 0 になります。

```bash
uv run python lectures/29_video_action_recognition/mini_project.py
# → outputs/29_video_action_recognition/mini_summary.png（サンプリング掃引）,
#    mini_confusion_cliplen2.png（粗サンプリングの崩れ）, mini_report.json, mini_clip_XX.mp4
```

この統合課題を自分の手で動かし、**「動画 I/O → サンプリング → 正規化 → 推論 → 評価」の全工程を、前処理を変えながら top-k で検証できる**ようになることが、本章のゴールです。余力があれば、`frame_rate` の範囲を広げる、multi-clip 平均を実装して単一クリップと比べる、VideoMAE でも掃引する、といった拡張に挑戦してみてください。

## ✅ 到達チェックリスト

次の項目を「人に説明でき／コードで再現できる」かで、定着を自己確認してください。

- [ ] 動画モデルの入力が **`(N,C,T,H,W)`**（画像の `(N,C,H,W)` に時間 `T` が増える）であることを説明でき、行動が「動き」で決まる理由を言える
- [ ] **uniform サンプリング**（`np.linspace`）と **clip_len+frame_rate(stride) サンプリング** の違い・使い分けを説明でき、末尾クランプを実装できる
- [ ] **r3d_18 の専用正規化**（Kinetics mean/std・112×112）が ImageNet 統計と別物である理由を言える
- [ ] `R3D_18_Weights.DEFAULT` から `r3d_18` をロードし、`(T,H,W,C)→(1,C,T,H,W)` に整えて推論できる（手書き前処理と `weights.transforms()` の両方）
- [ ] `VideoMAEImageProcessor`（`pixel_values` は `(1,T,C,H,W)`）＋ `VideoMAEForVideoClassification` で行動認識を実行できる
- [ ] transformers v5 の **VideoMAE q_bias/v_bias 取りこぼし**を説明でき、ロード報告から異常を見抜き手当てできる
- [ ] 動画 I/O を **`cv2.VideoCapture`**（`read_video` 廃止の代替）で行い、**BGR↔RGB 変換**を忘れない
- [ ] `pipeline('video-classification')` が **PyAV(`av`) を要する**ことと、手書き（processor＋model）等価経路を説明できる
- [ ] **clip-level top-1 / top-5 accuracy** を `np.argsort` から自前実装でき、混同行列を併記できる
- [ ] **前処理（clip_len/frame_rate/正規化）を崩すと top-k が壊れる**ことを、pseudo-GT を基準に数値で示せる

## ❓ よくある落とし穴・FAQ・デバッグ

本章で詰まりやすい点を「症状 → 原因 → 対処」の形でまとめます。動画特有・transformers v5 特有の罠が多いため、エラーが出たら、まずはここを確認してください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `RuntimeError: ... expected 5D ... got 4D` | r3d_18 に `(C,T,H,W)` を渡しバッチ次元が無い | `.unsqueeze(0)` で `(1,C,T,H,W)` にする |
| 出力がのっぺり／毎回違うクラス | 専用正規化を忘れた／`[0,255]` のまま渡した | `[0,1]` スケール後に**モデル指定の mean/std**で正規化 |
| r3d_18 の精度が出ない | ImageNet 統計を流用した | Kinetics 用 `mean=[0.43216,...]/std=[0.22803,...]`、112×112 |
| 色がおかしい（青っぽい等） | cv2 は BGR、torch/HF は RGB | `cv2.cvtColor(..., COLOR_BGR2RGB)`（書込は RGB→BGR） |
| `module torchvision.io has no read_video` | torchvision 0.26 で内蔵デコーダ廃止 | `cv2.VideoCapture` で読む（本講座の `read_clip_mp4`） |
| `pipeline('video-classification')` が `requires av` | PyAV 未導入 | 手書き（processor＋model）経路を使う or `uv add --group video av` |
| VideoMAE ロードで `UNEXPECTED q_bias/v_bias` | v5 の attention 分解でバイアス未移行（query.bias=0） | 元 checkpoint から `q_bias→query.bias`,`v_bias→value.bias` を手当て（`load_videomae`） |
| VideoMAE が遅い／固まる | base でも CPU は重い、長クリップ | `clip_len=16` 固定・`inference_mode`・スレッド数調整、まず r3d_18 で試す |
| `pixel_values` の形が合わない | r3d_18 は `(N,C,T,H,W)`、VideoMAE は `(N,T,C,H,W)` | モデルごとの軸順を確認（C と T の位置が逆） |
| 短い動画でサンプリングが落ちる | インデックスが範囲外 | `np.clip(idx, 0, total-1)` でクランプ（`uniform/strided_indices`） |
| top-1 が毎回 0 で評価が変 | 学習時と違うサンプリング/正規化で推論 | **学習時と推論時の前処理を必ず揃える** |
| CPU で `float16` が遅い/エラー | CPU は half 非効率・未対応 op | CPU は `float32`、half/bf16 は GPU のときだけ |
| 毎回モデルを再 DL（Docker） | キャッシュ未マウント | `~/.cache/huggingface`・`~/.cache/torch` をボリューム化、`HF_HUB_OFFLINE=1` |

とくに上位3つ――**バッチ次元の付け忘れ（5D 要求）**、**専用正規化の取り違え**、**BGR↔RGB**――は、本章の「あるある」です。そして**VideoMAE の q_bias/v_bias 取りこぼし**は v5 特有の見えにくい罠なので、「ロード報告を読む」「バイアスが 0 でないか確かめる」という癖をつけておきましょう。

## 🚀 発展トピック・参考

- **multi-clip / multi-crop TTA**: 1 本の動画から時間方向の複数窓・空間方向の複数クロップを取り、logits を平均して精度を上げる定番。評価枠組み（top-k・混同行列）はそのまま使える。
- **他の動画モデル**: 3D CNN 系の **X3D / SlowFast**（torchvision/`pytorchvideo`）、Transformer 系の **TimeSformer / ViViT / VideoMAEv2**。CPU で軽いのは小型 3D CNN（r3d_18, mc3_18, r2plus1d_18）。
- **(2+1)D 畳み込み**: `r2plus1d_18` は 3D 畳み込みを「空間 2D ＋ 時間 1D」に分解し、表現力と計算量のバランスを取る。3D CNN の発展形として比較すると面白い。
- **時間方向のデータ拡張**: フレーム間引き率の変動、時間反転、temporal jitter など。サンプリングと表裏一体で、学習時に多様なサンプリングを見せると推論時の頑健性が上がる。
- **姿勢ベースの行動認識（概念）**: 第27回の人体キーポイント列を入力に **ST-GCN** 等で行動を当てる系統もある（外見に依存せず軽い）。MediaPipe Pose で骨格列を作る発想は概念として接続できる（本講座では `mediapipe` は導入衝突回避のため任意・ガードのみ）。
- **公式ドキュメント**: [torchvision video models](https://docs.pytorch.org/vision/stable/models.html#video-classification) / [R3D_18_Weights](https://docs.pytorch.org/vision/stable/models/video_resnet.html) / [VideoMAE (transformers)](https://huggingface.co/docs/transformers/model_doc/videomae) / [video-classification pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) / [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset)。

## 💡 実践ユースケース集

本章で身につけた「クリップ → 行動クラス → top-k」を、現実の道具に落とすと何ができるかを 3 つ挙げます。最後の 1 つは実際に動く `use_case.py` として同梱しています。

- **動画ライブラリの自動タグ付け（=同梱の `use_case.py`）**: 何に使うか＝溜まった動画フォルダを横断し、各動画に行動ラベル(上位 N)を自動で振って、あとからタグで検索・集計できる**カタログ**にする（アセット管理・ざっくり分類・"この棚はどんな行動が多いか" の俯瞰）。作り方の要点＝フォルダを舐めて 1 本ずつ `cv2.VideoCapture` でデコード → 等間隔サンプリング → 専用正規化 → `r3d_18` → `top-k` を JSON/CSV に書き出す。注意＝タグは「クリップ全体に 1 つの行動」という前提なので、複数行動が混ざる長尺は時間窓に区切ってから掛ける。
- **アップロード動画の事前モデレーション／ルーティング**: 何に使うか＝投稿動画に行動タグを付け、特定カテゴリ（例: 危険行為・スポーツ）を**自動振り分け**や人手レビューのキューイングに回す。作り方の要点＝`top-1` 確率にしきい値を設け、「自信あり=自動処理 / 低信頼=人へ」の二段構え（human-in-the-loop）にする。注意＝Kinetics-400 のクラス語彙に無い行動は当然出ないので、自前カテゴリには転移学習か後段のルール対応が要る。
- **長尺動画のハイライト／チャプタ抽出**: 何に使うか＝1 本の長い動画をスライディング窓で区切って各窓を行動認識し、行動が切り替わる境目を**チャプタ**に、特定行動の窓を**ハイライト**として切り出す。作り方の要点＝`strided_indices` で窓ごとにサンプリング → 窓系列の `top-1` を時間方向に並べ、連続区間をまとめる。注意＝窓長(clip_len)と窓ずらし幅で粒度と計算量が変わるため、第2節のサンプリング理論がそのまま効いてくる。

### 同梱 `use_case.py` の使い方

```bash
# 動画フォルダ → 上位Nタグのカタログ(JSON/CSV)＋サムネ一覧＋タグ頻度図 を生成
uv run python lectures/29_video_action_recognition/use_case.py
```

- **実データの置き方**: `data/29_video_action_recognition/` に手持ちの動画（`.mp4` / `.avi` / `.mov` / `.mkv` / `.webm`）を置くと、それを優先してタグ付けします。1 本も無ければ合成「行動クリップ」を数本 mp4 に書き出し、実動画と同じ `cv2.VideoCapture` 経路でデコードしてタグ付けするので、ネットもデータも無しで `exit 0`（合成クリップに本物の Kinetics ラベルは無いため、付くタグは「例示」です。実写を置くとタグが意味を持ちます）。
- **出力**（`outputs/29_video_action_recognition/`）: `use_case_tags.json`（動画→上位Nタグのカタログ）／ `use_case_tags.csv`（1 行=1 動画の検索しやすい表）／ `use_case_gallery.png`（代表フレーム＋主タグのサムネ一覧）／ `use_case_tagfreq.png`（ライブラリ全体の主タグ頻度）。実行末尾では、最頻タグの語でカタログを横断検索するデモも走ります（タグ付けの実利＝「あとから探せる」を体感）。
- **`mini_project.py` との違い**: ミニプロジェクトは合成データセットで `clip_len`/`frame_rate` を掃引し「前処理を崩すと予測がどれだけ壊れるか」を pseudo-GT 基準で**定量化する学習用**パイプライン。`use_case.py` は学習の話を脇に置き、**現実のフォルダ**を入力に「検索できる成果物（カタログ）」を作る**実アプリの出発点**です。
- **拡張アイデア**: (1) `top-1` 確率にしきい値を設けて低信頼タグを捨てる、(2) 1 動画から時間窓を複数取って logits を平均する multi-clip TTA で頑健化、(3) `H.load_videomae()` に差し替えて精度を上げる（重い）、(4) `use_case_tags.csv` を pandas/SQLite に読み込んでタグ検索・集計 UI を作る、(5) 主タグごとに代表フレームを切り出してプレビューを量産。

## ▶ 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers ほか）に依存します。r3d_18・VideoMAE は、いずれも CPU だけで完走します（初回のみ重みをダウンロードし、以降はキャッシュから即起動します）。`pipeline('video-classification')` 用の `av`（PyAV）は**任意**です（未導入でも全スクリプトが exit 0 になります）。準備ができたら、プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf

# 道具箱の自己点検（モデル不要・純計算）＋合成クリップ/サンプリングの確認
uv run python lectures/29_video_action_recognition/action_helpers.py

# 各スクリプト（結果は outputs/29_video_action_recognition/ に保存）
uv run python lectures/29_video_action_recognition/01_videomae_action.py    # VideoMAE(Transformer)＋v5バイアス手当て
uv run python lectures/29_video_action_recognition/02_r3d18_action.py       # r3d_18(3D CNN)＋手書き/公式前処理＋cv2 I/O
uv run python lectures/29_video_action_recognition/03_action_topk_eval.py   # top-1/top-5＋前処理を壊す実験＋混同行列
uv run python lectures/29_video_action_recognition/mini_project.py          # 章末: mp4→抽出→推論→評価 の統合
uv run python lectures/29_video_action_recognition/use_case.py              # 実践: 動画フォルダ→上位Nタグの自動カタログ(JSON/CSV)

# 演習: まず TODO を自分で埋める（最初は全 FAIL だが exit 0）
uv run python lectures/29_video_action_recognition/exercises.py
# 模範解答（実行すると全 PASS）
uv run python lectures/29_video_action_recognition/exercises_solutions.py

# （任意）pipeline('video-classification') も動かす: uv add --group video av
# （任意）別モデルに差し替え: 01 の MODEL_NAME を編集（例: TimeSformer/ViViT 系）
```

実行後は、`outputs/29_video_action_recognition/` の図と json を、解説と照合してください。とくに `02_top5.png`（r3d_18 の top-5）、`03_topk_by_variant.png`（前処理を壊したときの top-1/top-5 の崩れ）、`03_confusion_no_normalize.png`（正規化を外したときの予測の散り）、`mini_summary.png`（clip_len/frame_rate 掃引）の4枚を見ると、本章の要点――**サンプリングと正規化が予測を支配する**――が視覚的に腑に落ちるはずです。なお、図中の文字は CJK フォントの豆腐（□）を避けるため、ASCII にしてあります。

## このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で構成されており、上から順に読むと「Transformer で動かす → 3D CNN で動かす → 評価で締める → 統合する」という流れで理解が積み上がります。device 判定・合成クリップ生成・mp4 I/O・サンプリング・専用正規化・モデルロード・top-k/混同行列・可視化といった共通処理は `action_helpers.py` に集約してあり、各スクリプトはそれを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `action_helpers.py` | device 判定・合成クリップ・mp4 書込/読戻し・サンプリング・r3d 前処理・r3d/VideoMAE ロード（v5 バイアス手当て）・top-k/混同行列・可視化。**道具箱** |
| `01_videomae_action.py` | VideoMAE(Transformer) で行動認識。processor の `(1,T,C,H,W)`、v5 の q_bias/v_bias 手当て、pipeline の概念紹介 |
| `02_r3d18_action.py` | r3d_18(3D CNN) で行動認識。手書き前処理 vs `weights.transforms()`、正規化あり/なし比較、cv2.VideoCapture I/O |
| `03_action_topk_eval.py` | clip-level top-1/top-5 を pseudo-GT 基準で測り、前処理(正規化/サンプリング)を壊すと崩れることを定量化＋混同行列 |
| `mini_project.py` | 章末統合: mp4→cv2抽出→サンプリング掃引(clip_len/frame_rate)→r3d_18→top-k→レポート。VideoMAE 突き合わせ（任意） |
| `use_case.py` | 実践ユースケース: 動画フォルダを舐めて r3d_18 で上位 N タグを自動付与し、検索できるタグ・カタログ(JSON/CSV)＋サムネ一覧を出力する小ツール |
| `exercises.py` | TODO 形式の演習8問（自己採点ランナー付き・純計算・モデル DL 不要） |
| `exercises_solutions.py` | 演習の模範解答（実行すると全 PASS） |

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub（transformers 同梱）／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0
> 使用モデル: `torchvision r3d_18`（`R3D_18_Weights.KINETICS400_V1`・3D CNN・400 クラス）／ `MCG-NJU/videomae-base-finetuned-kinetics`（VideoMAE・Transformer・400 クラス）。いずれも初回のみ重みを取得しキャッシュします。動画 I/O は `cv2.VideoCapture`（torchvision 0.26 で `read_video` 廃止）。`pipeline('video-classification')` 用の `av`（PyAV）は任意（未導入でも全スクリプト exit 0）。transformers v5 準拠（`VideoMAEImageProcessor`/`VideoMAEForVideoClassification`、VideoMAE の q_bias/v_bias はロード後に手当て）。
