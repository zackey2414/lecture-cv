# 第29回 動画理解・行動認識 — クリップサンプリング / r3d_18(3D CNN) / VideoMAE(Transformer) / top-k

> トラック: **動画・追跡** ／ レベル: **上級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers ほか）。CPU だけで完走します（初回のみ r3d_18 ~127MB / VideoMAE ~330MB の重みを自動ダウンロード、以降はキャッシュから即起動）。動画 I/O は **`cv2.VideoCapture` に統一**します（torchvision 0.26 で内蔵デコーダ `read_video` が廃止されたため）。`pipeline('video-classification')` が要求する **PyAV(`av`) は既定依存に含めず**、未導入なら「概念紹介＋手書き等価コード」に切り替えます（＝必ず exit 0）。入力動画はすべて**合成生成**（動く図形）で、ネットもデータセット DL も不要です。

## 🎯 この章のゴール

第27回（深度・姿勢・フロー）・第28回（多物体追跡）では「1 枚の画像」や「フレーム間の対応」を扱いました。本章のテーマはその先――**「複数フレーム（＝クリップ）をひとまとまりに見て、そこで起きている『行動』を当てる」動画理解**です。`tossing coin`（コイン投げ）や `golf putting`（パター）のように、行動は**1 枚の静止画では決まらず、時間方向の動きで決まる**。だから入力は画像の `(C,H,W)` ではなく、時間軸 `T` を足した **`(C,T,H,W)`**（バッチを入れると 5 次元 `(N,C,T,H,W)`）になります。この「時間が 1 次元増える」ことが、本章のすべての出発点です。

到達点は5つです。第一に、**動画モデルが受け取る入力テンソルの形 `(N,C,T,H,W)`** と、画像分類との違いを説明できること。第二に、**フレームサンプリング（clip_len＝何枚／frame_rate＝何フレームおき）と専用正規化**を正しく行い、`torchvision r3d_18`（3D CNN, Kinetics-400）と `transformers VideoMAE`（Transformer）の**両方**を CPU で動かせること。第三に、**`VideoMAEImageProcessor` の前処理**と **r3d_18 の手書き前処理**を、それぞれの正準作法で書けること。第四に、動画 I/O を **`cv2.VideoCapture`** で行い（`read_video` 廃止の代替）、mp4 からフレームを抽出して推論できること。第五に、**clip-level の top-1/top-5 accuracy** で評価し、**前処理（clip_len/frame_rate/正規化）を崩すとスコアが壊れる**ことを、自分の手で数値化できることです。

本章のスクリプトはすべて、ネットもデータセットも無しで完走するよう、入力を**合成クリップ**（cv2 で描いた「動く図形」）として生成します。ただし合成クリップには**本物の Kinetics ラベルが存在しない**。そこで評価では、「**正しい前処理で得たモデルの top-1 を、各クリップの基準ラベル（pseudo-GT）とみなす**」という方針を採り、前処理を崩したときに pseudo-GT との一致率がどれだけ落ちるかを測ります（第8節で詳述）。これは「絶対精度」ではなく「**前処理感度（robustness）**」の定量化ですが、本章の核心メッセージ――**「行動認識は前処理を間違えると無意味な出力になる」**――を、再現可能な数値で体感するには最適です。実測では、正しい前処理を基準に**正規化を外すと top-1 一致率が 1.00→0.50**、**動きを消す（同一フレーム複製）と 0.08**まで崩れました。

---

## 1. 動画理解の地図 — 画像分類との決定的な違いは「時間軸 T」

画像分類（第13回）は `(N,C,H,W)` のテンソルを受け取り、1 枚から 1 つのクラスを出しました。**行動認識は、これに時間軸 `T` を足した `(N,C,T,H,W)` を受け取り、クリップ全体から 1 つの行動クラスを出します**。なぜ時間が要るのか――それは、行動が「動き」で定義されるからです。たとえば「ドアの前に立つ人」の 1 枚からは「ドアを開ける」のか「閉める」のか「ただ立っている」のか分かりません。`tossing coin` も、コインが**上がって落ちる**という時間変化を見て初めて当たります。静止画では原理的に区別できない行動を扱うのが、動画理解の本質です。

この「時間をどうモデルに入れるか」で、アーキテクチャは大きく3系統に分かれます。**(1) 3D CNN**（本章の `r3d_18`）は、2D 畳み込みを時間方向にも広げた **3D 畳み込み**で、空間と時間を同時に畳み込みます。`(C,T,H,W)` をそのまま 3D カーネルで舐めるイメージで、実装が素直で小型なら CPU でも軽い。**(2) Video Transformer**（本章の `VideoMAE`）は、各フレームをパッチ列に分解し、**時空間のトークン**として自己注意でまとめます。大規模事前学習（VideoMAE はマスク再構成で自己教師あり事前学習）と相性が良く、精度が高い反面 CPU では重い。**(3) 2D CNN ＋ 時間集約**（TSN/SlowFast 系の発想）は、各フレームを 2D で処理し、後段で平均や別経路で時間をまとめる折衷案です。本章は (1)(2) を実際に動かし、最後に使い分けを整理します。

3 系統に共通する**最重要の作法が「フレームサンプリングと専用正規化」**です。動画は何百フレームもありますが、モデルが食うのは固定枚数（VideoMAE/r3d_18 とも標準 `clip_len=16` 枚）だけ。**「全長から 16 枚をどう選ぶか（uniform か、何フレームおきか）」**と、**「モデル固有の mean/std でどう正規化するか」**を間違えると、どんなに良いモデルでも出力は無意味になります。この章の半分は、実はこの「前処理」に費やします。まずはサンプリングの理論から見ましょう。

## 2. フレームサンプリング — clip_len と frame_rate(stride) の理論

動画クリップから固定枚数を取り出す方法は、大きく2つあります。**(A) 等間隔（uniform）サンプリング**は、全長 `total` フレームを端から端まで均等に `clip_len` 枚に間引く方法で、`np.linspace(0, total-1, clip_len)` を丸めて使います。「動画全体を広く薄く見たい」とき向きで、長さの違う動画を一律に扱えるのが利点。本講座の `uniform_indices(32, 8)` は `[0,4,9,13,18,22,27,31]` を返します。**(B) ストライド（clip_len + frame_rate）サンプリング**は、ある開始点から `frame_rate` フレームおきに `clip_len` 枚を連続的に取る方法で、`idx = start + arange(clip_len)*frame_rate` です。「短い時間窓を密に・一定速度で」見たいとき向きで、Kinetics 系モデルの学習時サンプリングに近い。`strided_indices(32, 8, frame_rate=4)` は中央寄せで `[1,5,9,13,17,21,25,29]` を返します。

ここで効くノブが **`clip_len`（何枚）** と **`frame_rate`（何フレームおき＝ストライド）** です。`clip_len` を増やすほど時間解像度は上がりますが計算は重くなる。`frame_rate` を上げる（粗く取る）ほど**長い時間範囲**をカバーできますが、速い動きは取りこぼす。逆に `frame_rate=1`（密に取る）だと**短い範囲**しか見えず、ゆっくりした行動の全体像を逃します。つまり「`clip_len × frame_rate ≒ 何秒ぶんを見るか（受容時間窓）」というトレードオフで、**学習時と推論時のサンプリングを揃える**のが鉄則です（学習が `frame_rate=4` なら推論も合わせる）。末尾を超えるインデックスは最終フレームに**クランプ**して、短い動画でも落ちないようにします。

サンプリングを間違えるとどうなるかは、本章のミニプロジェクトで定量化します。実測では、正しい前処理（uniform 16 枚）を基準に、**`clip_len` を 16→8→4→2 と減らすと top-1 一致率が 1.00→0.17→0.17→0.00** と崩れ、**`frame_rate` を 1→2→4 と変えると 0.50→0.83→0.33** と山なりに動きました（`frame_rate=2` が最も基準に近い）。サンプリングは「些細な前処理」ではなく、**予測を決定づける主役級のハイパーパラメータ**だと分かります。次は、サンプリングと並ぶもう一つの主役、専用正規化です。

## 3. 専用正規化 — なぜ ImageNet の mean/std ではダメなのか

画像分類で `[0.485,0.456,0.406]/[0.229,0.224,0.225]`（ImageNet 統計）を使ったのを覚えているでしょう。**動画モデルは、これとは別の専用 mean/std を使います**。`r3d_18`（Kinetics-400 学習）は `mean=[0.43216,0.394666,0.37645]`、`std=[0.22803,0.22145,0.216989]` で、入力解像度も **112×112**。VideoMAE は ImageNet 統計に近い `mean=[0.485,0.456,0.406]/std=[0.229,0.224,0.225]` で **224×224**。**学習時に使った統計と同じもので正規化しないと、モデルが見たことのない入力分布を渡すことになり、出力が崩れます**。「正規化なんてどれも同じ」ではなく、**モデルごとに正しい値が決まっている**――これを取り違えるのが、動画認識で最も多い事故です。

正規化の手順自体は単純で、`[0,1]` にスケールした画素から `(x - mean) / std` を**チャンネル別**に引いて割るだけ。重要なのは**「どの mean/std を使うか」と「正規化を忘れない／二重にかけない」**こと。`r3d_18` に ImageNet 統計を使う、`[0,255]` のまま正規化する、正規化を丸ごと飛ばす――どれも「壊れた前処理」です。本講座の `preprocess_for_r3d` は `normalize` フラグと `mean/std` 引数を持ち、わざと壊した前処理を再現できるようにしてあります。

実測でこの効果を見ましょう（第8節 `03` の結果）。正しい前処理を基準に、**正規化を丸ごと外すと top-1 一致率 1.00→0.50**、**ImageNet 統計を誤用すると 0.67** に落ちました（top-5 では持ち直す＝大まかには似た方向を向くが top-1 はズレる）。単一クリップでも、`02` のコイン投げクリップで「正規化あり」は `tossing coin 0.55 / golf putting 0.27 / frisbee 0.16` という鋭い分布なのに、「正規化なし」では `tossing coin 0.33 / finger snapping 0.13 / juggling 0.11` と**確信度がのっぺり**します。専用正規化は「精度を少し上げる調整」ではなく、**モデルを正しく動かす前提条件**だと心に刻んでください。次節から、実際のモデルを 1 つずつ動かします。

## 4. torchvision r3d_18 — 3D CNN の正準API と (N,C,T,H,W)

最初に動かすのは **`torchvision.models.video.r3d_18`**（ResNet-3D 18 層、Kinetics-400 で学習）です。小型で CPU でも軽く、本章の「動かして理解する」主役。ロードは画像モデルと同じ流儀で、**`R3D_18_Weights.DEFAULT`** から重みとメタ情報（400 クラスの `categories`）を取り、`r3d_18(weights=weights).eval()` で評価モードにします。初回のみ ~127MB を `~/.cache/torch/hub` にダウンロードし、以降はキャッシュから即起動します。推論は必ず `torch.inference_mode()` で勾配を切ること。

3D CNN の肝は**入力テンソルの形**です。`r3d_18` は **`(N, C, T, H, W)`** を受け取ります（画像の `(N,C,H,W)` に時間 `T` が挟まる）。本講座の手書き前処理 `preprocess_for_r3d` は、`(T,H,W,3)` の uint8 クリップを、(1) フレーム選択 → (2) `[0,1]` スケール → (3) 112×112 へリサイズ → (4) Kinetics 専用正規化 → (5) `(T,H,W,C)→(C,T,H,W)` へ並べ替え（`permute(3,0,1,2)`）→ (6) バッチ次元を足す、の順で `(1,3,16,112,112)` に整えます。torchvision には公式の前処理 `weights.transforms()` もあり、こちらは `(T,C,H,W)` uint8 を受け取って内部で `resize(128×171)→center-crop(112)` を行い `(C,T,H,W)` を返します（リサイズ流儀が違うので手書きと画素は厳密一致しませんが、最終形状と正規化統計は同じ）。`02_r3d18_action.py` はこの両者を並べて確認します。

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

実測（合成「円が左→右」クリップ）では、正しい前処理で top-1 が `tossing coin (0.55)`、続いて `golf putting (0.27)` / `catching or throwing frisbee (0.16)`。合成図形なので「正解」ではありませんが、モデルが**時間方向の並進運動に一貫して反応している**ことが分かります。同じクリップを mp4 に書き出して `cv2.VideoCapture` で読み戻しても top-1 は `tossing coin` で一致しました（第6節の I/O 経路の検証）。次は、より大きく高精度な Transformer 系、VideoMAE を見ます。

## 5. VideoMAE — Transformer 系の正準API と transformers v5 の落とし穴

**`VideoMAE`**（`MCG-NJU/videomae-base-finetuned-kinetics`）は、各フレームをパッチに刻んで**時空間トークン**として自己注意で処理する Video Transformer です。使い方は HuggingFace の他の画像モデルと同じく **`VideoMAEImageProcessor`（前処理）＋ `VideoMAEForVideoClassification`（本体）** の二段。processor は**「フレーム（HWC uint8）のリスト」**を受け取り、リサイズ・専用正規化・テンソル化までを一手に行い、**`pixel_values` を `(1, T, C, H, W)`**（r3d_18 の `(1,C,T,H,W)` とは T と C の順が違う点に注意）で返します。`clip_len` はモデル設定 `model.config.num_frames`（通常 16）に合わせて、こちらで 16 枚をサンプリングして渡します。

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

ここで本章最大の**落とし穴**を紹介します。**transformers v5 で VideoMAE をロードすると、`q_bias`/`v_bias` という特殊なバイアスが取りこぼされ、`query.bias`/`value.bias` が 0 のまま**になることがあります（ロード時に `UNEXPECTED: q_bias/v_bias` と `MISSING: query.bias/...` という報告が出る）。VideoMAE の自己注意は「`q_bias` と `v_bias` だけを学習し `k_bias` は 0 に固定する」特殊構造で、v5 が attention を `query/key/value` の Linear に分解した際、この対応付けが正しく移らないために起きます。バイアスが 0 でもそれらしく動くため見逃しがちですが、**本来の精度は出ません**。本講座の `load_videomae` は、元のチェックポイント（safetensors）から `q_bias`/`v_bias` を直接読み、`query.bias`/`value.bias` へ**手当て**します（`01` の出力で「手当てしたレイヤ数=12」「query.bias の絶対値和=260.74」と確認できます）。「ロード報告は読む」「バイアスが 0 になっていないか疑う」――この一手間が、Transformer 系を正しく使う作法です。実測では手当て後の top-1 は `golf putting (0.26)` でした。

VideoMAE は r3d_18 より大きく（~330MB、CPU では 1 クリップ数秒）、精度も高い一方で重い。本講座では **VideoMAE を「概念＋実演」、r3d_18 を「多数クリップを回す主力」**と役割分担します。`01_videomae_action.py` はモデルが落とせない/重い環境でも `try/except` で必ず exit 0 になるよう設計してあります。次は、両モデルに共通する動画 I/O と高レベル API を見ます。

## 6. 動画 I/O — cv2.VideoCapture（read_video 廃止）と pipeline の位置づけ

動画ファイルからフレームを取り出す方法として、かつては `torchvision.io.read_video` が定番でしたが、**torchvision 0.26 で内蔵デコーダが廃止**され、現在は使えません。本講座は動画 I/O を **`cv2.VideoCapture` に統一**します。`cap = cv2.VideoCapture(path)` で開き、`cap.read()` を `False` が返るまで回してフレームを集め、`cap.release()` で閉じる――これだけ。注意点は **cv2 が BGR 順**であること。torch/transformers/matplotlib はすべて RGB 前提なので、読み出したフレームは `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` で RGB に直します（書き出す `cv2.VideoWriter` 側は逆に RGB→BGR）。本講座の `write_clip_mp4`/`read_clip_mp4` はこの変換込みのラウンドトリップを提供し、`mp4v` コーデックで headless 環境でも動きます。

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

高レベル API の **`pipeline('video-classification')`** も触れておきます。これは「mp4 のパス → top-k ラベル」を 1 行で返す便利な窓口ですが、**内部で動画デコードに PyAV（`av`）を必要**とします。本講座は依存衝突（numpy2 系との相性など）と巨大化を避けるため **`av` を既定依存に含めない**方針なので、`01` では `try: import av` で存在を確認し、**未導入なら「概念紹介＋手書き等価コード（processor＋model）」に切り替え**ます。pipeline がやっているのは結局「デコード→サンプリング→processor→model→softmax→top-k」であり、本章で手書きした経路と等価です。使いたい場合は `uv add --group video av` で追加できますが、**学習目的では中身が見える手書き経路を通すこと**を勧めます（pipeline はブラックボックスになりがち）。次は、いよいよ評価です。

## 7. 評価 — clip-level の top-1/top-5 と「前処理を壊す」実験

行動認識の標準評価は **clip-level の top-1 / top-5 accuracy** です。top-1 は「最尤クラスが正解か」、top-5 は「上位 5 クラスに正解が入るか」。Kinetics-400 のように**意味が近いクラスが多い**（`juggling balls` と `contact juggling` など）データでは、top-1 は厳しすぎることがあり、**top-5 を併記**するのが慣例です。実装は素朴で、`np.argsort(-logits, axis=1)[:, :k]` で各サンプルの上位 k クラスを取り、`gt` が含まれる割合を数えるだけ（`topk_accuracy`）。さらに**混同行列**（行=正解、列=予測）を併記すると、「どのクラスがどのクラスと取り違えられるか」が見えます。

ここで本章特有の工夫が要ります。**合成クリップには本物の Kinetics ラベルが無い**ので、素直には top-1 accuracy を測れません。そこで `03_action_topk_eval.py` は、**「正しい前処理（uniform 16 枚＋Kinetics 正規化）でのモデルの top-1 を、各クリップの基準ラベル（pseudo-GT）とみなす」**方針を採ります。そのうえで前処理を色々に崩し、**pseudo-GT に対する top-1/top-5 一致率**を測る。正しい前処理なら定義上 1.00、崩すほど一致率が落ちる――これは「絶対精度」ではなく「**前処理感度（robustness）**」の定量化ですが、本章の主張「前処理を誤ると出力が壊れる」を**再現可能な数値**で示すのに最適です。混同行列は、pseudo-GT に出現するクラスだけを行に、列にそれら＋`other`（出現外のクラス）を置く小さな行列にして、崩れた予測が**どれだけ別物（other）へ散るか**を可視化します。

実測（6 種の動き×2 seed＝12 クリップ）が下表です。正しい前処理 `correct` は当然 1.00/1.00。**正規化を外す `no_normalize` で top-1 が 0.50** に半減、**ImageNet 統計を誤用する `imagenet_norm` で 0.67**、そして**動きを消す `static_frame`（同一フレームを 16 枚複製）で 0.08**まで崩壊しました。最後の結果は決定的で、**「行動認識は『動き』を消すと完全に無力化する」**――時間情報こそがこのタスクの本体だと、数字が語っています。

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

この「pseudo-GT を基準に前処理感度を測る」枠組みは、実データが手元にあれば**そのまま本物の top-1/top-5 accuracy に置き換わります**（pseudo-GT を本物の GT にするだけ）。評価コードの骨格は同じなので、合成データで仕組みを固めておけば、実運用にそのまま移せます。次は、ここまでを統合するミニプロジェクトです。

## 8. 実務の使い分け — 3D CNN vs Video Transformer vs (2D＋時間)

最後に、本章で扱った 2 系統（＋折衷案）を実務目線で整理します。**3D CNN（r3d_18, X3D, SlowFast）**は、空間と時間を 3D 畳み込みで同時に扱い、**小型なら CPU でも軽い**のが強み。エッジ・リアルタイム寄りの用途や、まず動かして当たりを付けたいときの第一候補です。弱みは超長時間の文脈を取りにくいこと。**Video Transformer（VideoMAE, TimeSformer, ViViT）**は、自己注意で**長距離の時空間依存**を捉え、大規模事前学習と相性が良く**精度が高い**反面、**CPU では重い**。クラウド GPU で精度を追うときの主力です。**2D CNN＋時間集約（TSN）**は、各フレームを 2D で安く処理し後段で平均する折衷で、**実装が軽く速い**が、フレーム平均では捉えにくい「順序が効く行動」に弱い。

選択の初期方針は、**「速度・コスト優先かつ CPU なら 3D CNN（小）」「精度優先で GPU が使えるなら Video Transformer」「とにかく軽く大量に回すなら 2D＋時間集約」**が大づかみです。どの系統でも共通して効くのが、本章で繰り返した**「学習時と同じサンプリング（clip_len/frame_rate）と専用正規化を推論でも厳守する」**こと。さらに精度を一段上げる定番が **multi-clip / multi-crop の TTA（test-time augmentation）**で、1 本の動画から複数クリップ（時間方向に複数窓、空間方向に複数クロップ）を取って logits を平均します。本章は単一クリップ推論に絞りましたが、評価の枠組み（top-1/top-5・混同行列）はそのまま multi-clip にも乗ります。

実装面の注意も一つ。本章の `r3d_18` 例では `clip_len` を可変（2〜16）にして掃引できましたが、これは 3D CNN が**時間方向を畳み込みで吸収する**ため時間長が変わっても動くからです。一方 **VideoMAE は `num_frames=16` を前提に位置埋め込みを持つ**ので、原則 16 枚固定で渡します（枚数を変えると位置埋め込みと整合しません）。「モデルが時間長の変化を許すか」はアーキテクチャ依存――ここも「前処理はモデルごとに正しい形が決まっている」という本章の通奏低音の一例です。

## 🛠 章末ミニプロジェクト — 「mp4 → フレーム抽出 → 行動推定 → 評価」一気通貫

`mini_project.py` は、本章の要素（cv2 動画 I/O ／ サンプリング ／ 専用正規化 ／ r3d_18 推論 ／ top-1/top-5 評価 ／ 混同行列）を**1 本のパイプライン**に統合します。**ステージA【動画 I/O】**で合成行動クリップを mp4 に書き出し `cv2.VideoCapture` で読み戻す（`read_video` 廃止の代替経路をそのまま実演）、**ステージB【基準作り】**で正しい前処理の top-1 を pseudo-GT に、**ステージC【サンプリング掃引】**で `clip_len`（2/4/8/16）と `frame_rate`（1/2/4）を変えながら top-1/top-5 一致率を測り、**ステージD【別モデル（任意）】**で VideoMAE が使えれば数クリップで r3d_18 と top-1 を突き合わせ、**ステージE【レポート】**でサンプリング掃引のサマリ図と総合 json を出力します。

実測では、ステージC の `clip_len` 掃引が **2→0.00, 4→0.17, 8→0.17, 16→1.00**（短いほど崩れる）、`frame_rate` 掃引が **1→0.50, 2→0.83, 4→0.33**（基準＝uniform に近い `rate=2` が最良、粗すぎる `rate=4` で悪化）と、**サンプリングが予測を決定づける**ことを鮮明に示しました。`clip_len=2` の混同行列を見ると、予測が `other` 列に大きく散り、「2 枚では動きが読めない」様子が一目で分かります。ステージD では r3d_18 と VideoMAE が別クラスを出すこともあり（例: r3d_18 `tossing coin` vs VideoMAE `juggling balls`）、これは**合成データに唯一の正解が無い**ことの裏返しです。**どのモデルが落ちてもパイプラインは止めず、できた範囲でレポートを出す**設計なので、必ず exit 0 になります。

```bash
uv run python lectures/29_video_action_recognition/mini_project.py
# → outputs/29_video_action_recognition/mini_summary.png（サンプリング掃引）,
#    mini_confusion_cliplen2.png（粗サンプリングの崩れ）, mini_report.json, mini_clip_XX.mp4
```

この統合課題を自分の手で動かし、**「動画 I/O → サンプリング → 正規化 → 推論 → 評価」の全工程を、前処理を変えながら top-k で検証できる**ことが、本章のゴールです。余力があれば、`frame_rate` の範囲を広げる、multi-clip 平均を実装して単一クリップと比べる、VideoMAE でも掃引する、といった拡張に挑戦してください。

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

本章で詰まりやすい点を「症状 → 原因 → 対処」でまとめます。動画特有・transformers v5 特有の罠が多いので、エラーが出たらまずここを確認してください。

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

とくに上位3つ――**バッチ次元の付け忘れ（5D 要求）**、**専用正規化の取り違え**、**BGR↔RGB**――は本章の「あるある」です。そして**VideoMAE の q_bias/v_bias 取りこぼし**は v5 特有の見えにくい罠なので、「ロード報告を読む」「バイアスが 0 でないか確かめる」癖をつけてください。

## 🚀 発展トピック・参考

- **multi-clip / multi-crop TTA**: 1 本の動画から時間方向の複数窓・空間方向の複数クロップを取り、logits を平均して精度を上げる定番。評価枠組み（top-k・混同行列）はそのまま使える。
- **他の動画モデル**: 3D CNN 系の **X3D / SlowFast**（torchvision/`pytorchvideo`）、Transformer 系の **TimeSformer / ViViT / VideoMAEv2**。CPU で軽いのは小型 3D CNN（r3d_18, mc3_18, r2plus1d_18）。
- **(2+1)D 畳み込み**: `r2plus1d_18` は 3D 畳み込みを「空間 2D ＋ 時間 1D」に分解し、表現力と計算量のバランスを取る。3D CNN の発展形として比較すると面白い。
- **時間方向のデータ拡張**: フレーム間引き率の変動、時間反転、temporal jitter など。サンプリングと表裏一体で、学習時に多様なサンプリングを見せると推論時の頑健性が上がる。
- **姿勢ベースの行動認識（概念）**: 第27回の人体キーポイント列を入力に **ST-GCN** 等で行動を当てる系統もある（外見に依存せず軽い）。MediaPipe Pose で骨格列を作る発想は概念として接続できる（本講座では `mediapipe` は導入衝突回避のため任意・ガードのみ）。
- **公式ドキュメント**: [torchvision video models](https://docs.pytorch.org/vision/stable/models.html#video-classification) / [R3D_18_Weights](https://docs.pytorch.org/vision/stable/models/video_resnet.html) / [VideoMAE (transformers)](https://huggingface.co/docs/transformers/model_doc/videomae) / [video-classification pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) / [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset)。

## ▶ 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers ほか）に依存します。r3d_18・VideoMAE はいずれも CPU だけで完走します（初回のみ重みをダウンロード、以降はキャッシュから即起動）。`pipeline('video-classification')` 用の `av`（PyAV）は**任意**（未導入でも全スクリプトが exit 0）です。プロジェクトルートで以下を順に実行してください。

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

# 演習: まず TODO を自分で埋める（最初は全 FAIL だが exit 0）
uv run python lectures/29_video_action_recognition/exercises.py
# 模範解答（実行すると全 PASS）
uv run python lectures/29_video_action_recognition/exercises_solutions.py

# （任意）pipeline('video-classification') も動かす: uv add --group video av
# （任意）別モデルに差し替え: 01 の MODEL_NAME を編集（例: TimeSformer/ViViT 系）
```

実行後は `outputs/29_video_action_recognition/` の図と json を解説と照合してください。とくに `02_top5.png`（r3d_18 の top-5）、`03_topk_by_variant.png`（前処理を壊したときの top-1/top-5 の崩れ）、`03_confusion_no_normalize.png`（正規化を外したときの予測の散り）、`mini_summary.png`（clip_len/frame_rate 掃引）の4枚を見ると、本章の要点――**サンプリングと正規化が予測を支配する**――が視覚的に腑に落ちます。図中の文字は CJK フォントの豆腐（□）を避けるため ASCII にしてあります。

## このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から読むと「Transformer で動かす → 3D CNN で動かす → 評価で締める → 統合する」と理解が積み上がります。device 判定・合成クリップ生成・mp4 I/O・サンプリング・専用正規化・モデルロード・top-k/混同行列・可視化といった共通処理は `action_helpers.py` に集約し、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `action_helpers.py` | device 判定・合成クリップ・mp4 書込/読戻し・サンプリング・r3d 前処理・r3d/VideoMAE ロード（v5 バイアス手当て）・top-k/混同行列・可視化。**道具箱** |
| `01_videomae_action.py` | VideoMAE(Transformer) で行動認識。processor の `(1,T,C,H,W)`、v5 の q_bias/v_bias 手当て、pipeline の概念紹介 |
| `02_r3d18_action.py` | r3d_18(3D CNN) で行動認識。手書き前処理 vs `weights.transforms()`、正規化あり/なし比較、cv2.VideoCapture I/O |
| `03_action_topk_eval.py` | clip-level top-1/top-5 を pseudo-GT 基準で測り、前処理(正規化/サンプリング)を壊すと崩れることを定量化＋混同行列 |
| `mini_project.py` | 章末統合: mp4→cv2抽出→サンプリング掃引(clip_len/frame_rate)→r3d_18→top-k→レポート。VideoMAE 突き合わせ（任意） |
| `exercises.py` | TODO 形式の演習8問（自己採点ランナー付き・純計算・モデル DL 不要） |
| `exercises_solutions.py` | 演習の模範解答（実行すると全 PASS） |

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub（transformers 同梱）／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0
> 使用モデル: `torchvision r3d_18`（`R3D_18_Weights.KINETICS400_V1`・3D CNN・400 クラス）／ `MCG-NJU/videomae-base-finetuned-kinetics`（VideoMAE・Transformer・400 クラス）。いずれも初回のみ重みを取得しキャッシュします。動画 I/O は `cv2.VideoCapture`（torchvision 0.26 で `read_video` 廃止）。`pipeline('video-classification')` 用の `av`（PyAV）は任意（未導入でも全スクリプト exit 0）。transformers v5 準拠（`VideoMAEImageProcessor`/`VideoMAEForVideoClassification`、VideoMAE の q_bias/v_bias はロード後に手当て）。
