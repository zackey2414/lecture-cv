# 第2回 画像・動画処理ライブラリの地図 — OpenCV / Pillow / scikit-image / albumentations / kornia ほか

> トラック: 画像の基礎 ／ レベル: 入門 ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（比較対象ライブラリは未導入でもスキップして必ず動きます）

## 🎯 この章のゴール

第1回では「画像とは結局 `(H, W, 3)` の `uint8` numpy 配列にすぎない」という土台を作りました。この章では一歩引いて、その配列を実際に処理する**ライブラリの全体像（地図）**を頭に入れます。画像・動画処理の世界には OpenCV・Pillow・scikit-image・imageio・PyAV・torchvision・albumentations・kornia と多数のライブラリがあり、機能が重なり合っています。初学者がつまずくのは「どれを使えばいいのか分からない」点で、ここを整理しないと、検索で出てきたコードをつぎはぎして一貫性のないプログラムを書き続けることになります。

この章を終えたとき、あなたは**課題を見た瞬間に「まずこのライブラリ」と当たりを付けられる**ようになります。たとえば「動画を手早く読みたい」なら OpenCV の `VideoCapture`、「検出用に bbox ごと画像を拡張したい」なら albumentations、「学習ループの中で GPU・微分可能に拡張したい」なら kornia、というふうに、役割分担を地図として説明できる状態を目指します。アルゴリズムの暗記ではなく、**ライブラリ選択の判断基準**を身につけるのがこの「概念回」の主眼です。

到達点を一言でいえば、**同じ処理（読込・リサイズ・ぼかし・回転）を複数ライブラリで書き比べられ、それぞれの色順・データ表現・引数の癖を説明でき、データ拡張が学習データの分布をどう広げるかを自分の目で確認できる**ことです。実行コードは main 依存（cv2 / PIL / numpy / matplotlib）だけで完走し、比較対象の任意ライブラリは「入っていれば実演、無ければ導入コマンドを案内してスキップ」する設計になっているので、手元に何が入っていても全スクリプトが動きます。

---

## 1. なぜ「地図」が必要か — 機能が重なるライブラリ群

画像処理ライブラリの厄介なところは、**できることが大きく重複している**点です。「画像を半分に縮小する」一つを取っても、OpenCV・Pillow・scikit-image のどれでも書けます。だからこそ初学者は「結局どれが正解なのか」で迷い、記事ごとに違うライブラリが出てくるたびに混乱します。重要なのは「どれか一つが正しい」のではなく、**それぞれに得意分野と前提（データ型・色順・GPU対応）があり、場面で使い分ける**という発想です。

使い分けの軸は主に4つあります。第一に**速度と機能の網羅性**（OpenCV が頭一つ抜けている）、第二に**扱うデータ表現**（OpenCV / scikit-image は numpy 配列、Pillow は `PIL.Image`、torchvision / kornia は PyTorch の Tensor）、第三に**微分可能性・GPU対応**（学習ループに組み込めるのは torchvision / kornia）、第四に**エコシステムとの親和性**（PyTorch 学習なら torchvision、検出/セグメの拡張なら albumentations）です。この4軸で各ライブラリを位置づけられれば、選択はほぼ自動的に決まります。

本章のスクリプト `01_library_map.py` は、この地図をコードと図で表現します。狙いは「主要ライブラリを一覧表（早見表）にまとめ、いま自分の環境に何が入っているかを `import` で点検し、役割を2軸の散布図にプロットする」こと。確認ポイントは、**早見表で各ライブラリの色順・データ表現・GPU対応を一目で見比べられること**と、**未導入のライブラリには導入コマンドが案内されること**の2点です。次節からこの早見表の中身を読み解いていきます。

## 2. 主要ライブラリ早見表

まずは全体像を一枚の表で押さえます。下の表は `01_library_map.py` が生成する早見表（`outputs/02_cv_libraries_overview/01_library_cheatsheet.png`）と同じ内容です。「色順」「主なデータ表現」「GPU/微分可能」「主な役割」「こういう時に選ぶ」を並べてあります。眺める前に一つだけ予告しておくと、**色順の列で OpenCV だけが BGR**で、ほかは全部 RGB です。これが第1回でも強調した最重要の非対称で、ライブラリをまたぐたびに効いてきます。

| ライブラリ | 色順 | 主なデータ表現 | GPU/微分 | 主な役割・こういう時に選ぶ |
| --- | --- | --- | --- | --- |
| NumPy | - | `ndarray` | CPU | 全画像の実体。画素を直接いじる・自作処理 |
| OpenCV | **BGR** | `ndarray(BGR)` | CPU | 古典CVの総合商社。前処理/検出/動画。まず第一候補 |
| Pillow | RGB | `PIL.Image` | CPU | 直感的な画像編集・フォント描画・EXIF・読み書き |
| scikit-image | RGB | `ndarray(float[0,1])` | CPU | アルゴリズム豊富。論文再現・計測・科学計算寄り |
| imageio | RGB | `ndarray` | CPU | 画像/動画/GIFを統一APIで手軽に入出力 |
| PyAV (`av`) | RGB等 | `ndarray`/`Frame` | CPU | FFmpeg同梱。精密な動画デコード/エンコード |
| torchvision v2 | RGB | `Tensor` | GPU/微分可 | PyTorch学習の前処理・データ拡張（transforms.v2） |
| albumentations | RGB | `ndarray` | CPU | 高速・bbox/mask/keypoint同時変換。検出/セグメ拡張 |
| kornia | RGB | `Tensor` | GPU/微分可 | GPUバッチ拡張・勾配が流れる。学習ループ内で拡張 |
| matplotlib | RGB | `ndarray` | CPU | headlessでの結果確認・図/比較/ヒストグラム保存 |

この表で最初に注目してほしいのは**データ表現の列**です。`ndarray`（numpy）・`PIL.Image`・`Tensor`（PyTorch）の3系統があり、ライブラリをまたぐときはこの3つの間で変換が必要になります。OpenCV と scikit-image と albumentations は同じ numpy 配列を共有するので相互運用が楽ですが、scikit-image だけは値域が `float[0,1]` を好む点が落とし穴です（後述）。torchvision と kornia は Tensor 世界なので、numpy との橋渡しが要ります。

次に**GPU/微分可能の列**です。CPU と書いてあるライブラリは「前処理は CPU で完結」という前提で、学習の勾配は流れません。対して torchvision v2 と kornia は Tensor 上で動き、GPU・微分可能なので、**拡張処理を学習ループの一部として組み込める**のが決定的な違いです。本講座は CPU 前提なので実演は CPU で行いますが、「この2つだけは学習と一体化できる」という性質は地図上で覚えておいてください。`01_library_map.py` を実行すると、この表が「導入済み=緑・未導入=赤」で色分けされて保存されるので、自分の環境の状態が一目で分かります。

## 3. 2軸の地図で「守備範囲」を掴む

表は網羅的ですが、関係性は見えにくいものです。そこで `01_library_map.py` は同じ情報を**2軸の散布図**（`01_library_quadrant_map.png`）にもプロットします。横軸は「低レベル（配列を直接いじる）↔ 高レベル（学習向け・抽象度が高い）」、縦軸は「CPUのみ ↔ GPU・微分可能」です。地図上で近い位置にあるライブラリは守備範囲が似ていて競合しやすく、遠いものは役割がはっきり分かれている、と読めます。

この地図を眺めると、いくつかの塊が見えてきます。左下には NumPy・OpenCV・Pillow・scikit-image・imageio/PyAV といった**CPU の古典CV・I/O**勢が集まります。ここは「前処理・読み書き・古典アルゴリズム」の領域で、本講座の画像基礎トラックの主役です。右上に向かうにつれて albumentations → torchvision v2 → kornia と並び、**学習に近い・GPU 寄り**の拡張ライブラリへ移っていきます。albumentations が中間（高レベルだが CPU）、kornia が右上隅（高レベルかつ GPU/微分可能）という配置が、それぞれの立ち位置を端的に表しています。

なぜこの配置になるのかを一言で言えば、**「画素をどれだけ生で触るか」と「学習パイプラインにどれだけ食い込むか」が独立した2つの軸だから**です。OpenCV は機能が多くても基本は CPU・配列処理なので左下、kornia は機能の見た目こそ拡張ですが Tensor・GPU・微分可能なので右上、というわけです。この地図が頭に入ると、新しいライブラリに出会ったときも「どのあたりの仲間か」で性質を推測できるようになります。確認ポイントは、**自分が今やりたいことが地図のどの象限に当たるかを言えること**です。

## 4. 相互運用 — `ndarray` ⇄ `PIL` ⇄ `Tensor` と「色順・軸順」

地図で見た通り、実務ではライブラリをまたいで処理を繋ぎます。そのとき必ず必要になるのが**データ表現の相互変換**です。第1回で `cv2(BGR) → PIL(RGB) → numpy → cv2(BGR)` のラウンドトリップを練習しましたが、ここに torchvision/kornia の Tensor が加わると、変換の組み合わせが増えます。コツは「境界をまたぐたびに**色順**と**軸順**の2つを意識する」ことに尽きます。

色順は繰り返しになりますが OpenCV だけ BGR、ほかは RGB です。軸順はもう一つの罠で、**numpy/OpenCV の `shape` は `(H, W, C)`、PIL の `size` は `(W, H)`、PyTorch の Tensor は `(C, H, W)`** と、それぞれ並びが違います。下のコードは3系統の典型的な変換です。読む前に要点を言うと、PIL へ渡すときは BGR→RGB、Tensor へ渡すときはさらに軸を `(H,W,C)→(C,H,W)` に並べ替える、という2段階を踏みます。

```python
import cv2, numpy as np
from PIL import Image

bgr = cv2.imread("img.png")                  # OpenCV: (H,W,3) BGR
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)    # 境界をまたぐ前に RGB へ

pil = Image.fromarray(rgb)                    # numpy(RGB) -> PIL   size=(W,H)
arr = np.asarray(pil)                         # PIL -> numpy(RGB)   shape=(H,W,3)

# PyTorch Tensor へ（torchvision がある場合）: (H,W,C) -> (C,H,W)
# import torch; chw = torch.from_numpy(rgb).permute(2, 0, 1)
```

上のコードで `size=(W,H)` と `shape=(H,W,3)` が逆順である点を必ず意識してください。リサイズで幅と高さを取り違える事故の大半はここから生まれます。`cv2.resize` の `dsize` も `(W, H)` 順、PIL の `resize` も `(W, H)` 順ですが、scikit-image の `transform.resize` だけは `(H, W)` 順——という「逆の逆」が次節で出てきます。**「いま触っているライブラリは幅・高さどちらが先か」を毎回確認する**のが、軸順事故を防ぐ唯一の確実な方法です。

## 5. 同一処理の書き比べ — resize / blur / rotate

地図と相互運用を押さえたら、いよいよ手を動かします。`02_same_op_across_libs.py` は、**まったく同じ3処理（リサイズ・ガウシアンぼかし・30度回転）を OpenCV・Pillow・scikit-image で書き比べ**、結果を1枚の比較図（`02_same_op_compare.png`）にまとめます。狙いは「同じことをするのに API がどれだけ違うか」を体で覚えること。確認ポイントは、**各ライブラリの引数の順序・データ型・既定の補間方法の違い**が表とコードで腑に落ちることです。

下の表は3ライブラリの「同じ処理」の書き方の違いをまとめたものです。表の前に結論を言うと、OpenCV は `uint8` の配列をそのまま速く処理し、Pillow は `PIL.Image` を介し、scikit-image は内部で `float[0,1]` に正規化してから処理して最後に `uint8` へ戻す——という流儀の差があります。とくに scikit-image の「値域が 0〜1 になる」点と「サイズ指定が `(H, W)` 順」の点が、ほかの2つと逆なので最大の注意所です。

| 処理 | OpenCV | Pillow | scikit-image |
| --- | --- | --- | --- |
| リサイズ | `cv2.resize(img, (W,H), INTER_AREA)` | `pil.resize((W,H), LANCZOS)` | `transform.resize(f, (H,W))` ※順序逆 |
| ぼかし | `cv2.GaussianBlur(img,(0,0),sigmaX=3)` | `pil.filter(ImageFilter.GaussianBlur(3))` | `filters.gaussian(f, sigma=3, channel_axis=-1)` |
| 回転 | `getRotationMatrix2D`+`warpAffine` | `pil.rotate(30, expand=False)` | `transform.rotate(f, 30)` |
| データ型 | `ndarray uint8` (BGR) | `PIL.Image` (RGB) | `ndarray float[0,1]` (RGB) |

この表で「リサイズ」の行を横に見ると、OpenCV と Pillow は `(W, H)` 順なのに scikit-image だけ `(H, W)` 順、という非対称が見えます。これは scikit-image が「画像も普通の numpy 配列の一種」として `shape` の並び（行=高さが先）に忠実だからです。理屈は分かっても手は間違えるので、**scikit-image を使うときは『リサイズは高さ・幅の順』と声に出して確認する**くらいでちょうどいいです。`02_same_op_across_libs.py` は scikit-image が未導入でも OpenCV と Pillow の2列だけで必ず動き、導入済みなら3列目として scikit-image を加えます。

もう一つ面白い学びがあります。`02_same_op_across_libs.py` は OpenCV と Pillow の結果がどれだけ**数値的に違うか**（平均絶対差）も表示します。実行すると、同じ「リサイズ」でも平均で数階調ぶんズレることが分かります。これは補間アルゴリズム（OpenCV の `INTER_AREA` と Pillow の `LANCZOS`）が違うからで、**「見た目はほぼ同じでもピクセル値は完全一致しない」**という事実を数値で体感できます。これを知っておくと、「ライブラリを変えたら学習結果が微妙に変わった」といった現象に冷静に対処できます。

## 6. データ拡張ライブラリの住み分け — albumentations / torchvision v2 / kornia

この章のもう一つの主役が**データ拡張（augmentation）**です。データ拡張とは「1枚の画像から、ラベルを変えずに見た目の違う複数枚を作り出す」技術で、学習データの多様性を水増しして過学習を抑え、汎化性能を上げるのが目的です。`03_augmentation_albumentations.py` は、まず main 依存（cv2/numpy）だけで小さな拡張パイプラインを自作し、その後で定番ライブラリと対比します。狙いは「拡張の仕組みを自分の手で理解してから、ライブラリの便利さを知る」こと。

拡張ライブラリの住み分けは、第3節の地図そのものです。**albumentations** は CPU・numpy ベースで非常に高速、そして検出やセグメンテーション用に**画像と一緒に bounding box / mask / keypoint も同じ幾何変換で動かせる**のが最大の武器です。**torchvision transforms v2** は PyTorch の Tensor 上で動き、学習パイプラインと一体化しているのが強み。**kornia** は同じく Tensor ですが、GPU でバッチ処理でき**勾配が流れる（微分可能）**ため、拡張を学習ループの内側に置けます。下のコードは albumentations の正準形です。

```python
import albumentations as A

# Compose にずらりと並べた変換を、確率的に適用するパイプラインを作る
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(limit=15, p=0.8),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.4),
])
augmented = transform(image=rgb)["image"]   # 入力は RGB の ndarray、戻りも ndarray
```

このコードのポイントは、`A.Compose([...])` に変換を並べ、`transform(image=rgb)["image"]` で適用する——という「辞書で入出力する」スタイルです。なぜ辞書かというと、`transform(image=rgb, bboxes=..., masks=...)` のように**複数の入力を同時に同じ幾何変換へ通す**ためで、これが検出/セグメで albumentations が選ばれる理由です。`03_augmentation_albumentations.py` は albumentations が入っていれば本物のパイプラインで拡張グリッドを作り、入っていなければ「役割の説明＋導入コマンド」を表示してスキップします（torchvision v2 / kornia も同様にガードしてあります）。だから、どのライブラリが入っていなくてもスクリプトは必ず最後まで動きます。

## 7. 評価ポイント — 拡張は「分布」をどう広げるか

この回は数値スコアでの評価はありませんが、**拡張がデータの分布に与える影響を可視化で確認する**のが評価の代わりです。`03_augmentation_albumentations.py` は、自作パイプラインを1枚の画像に300回かけ、その都度の「平均明るさ」をヒストグラムにします（`03_aug_distribution.png`）。元画像は1点（縦線）だったのに対し、拡張をかけた集合は明るさが幅広く散らばった分布になる——これが「データの多様性が増えた」ことの数値的な証拠です。

なぜこの可視化が大事かというと、拡張は「やればやるほど良い」わけではなく、**やりすぎると元のラベルが意味を失うほど画像が崩れる**からです。明るさを極端に振れば真っ黒・真っ白なサンプルが混じり、回転を大きくすれば物体が画面外に出ます。分布を眺めることで「拡張が現実的な範囲に収まっているか」を判断できます。実務では、検証用画像で拡張後のサンプルを必ず目で見て、過激すぎないかを確認するのが鉄則です。

拡張グリッド（`03_manual_aug_grid.png`）も併せて確認してください。これは原画像＋8枚の拡張サンプルを3×3で並べたもので、左右反転・明るさ/コントラスト変化・回転・色相揺らし・ノイズが組み合わさって、1枚から多様な見た目が生まれる様子が一目で分かります。確認ポイントは、**ヒストグラムの「広がり」とグリッドの「見た目の多様さ」が対応している**ことです。自作パイプラインの各関数（`aug_hflip`/`aug_brightness_contrast`/`aug_rotate`/`aug_hsv_jitter`/`aug_gaussian_noise`）を読めば、albumentations が内部でやっていることの正体が分かります。

## 8. 動画I/O の地図 — OpenCV VideoCapture / imageio / PyAV

画像だけでなく**動画I/O**にも地図があります。これは第9回以降の動画トラックの予告でもあります。動画読み込みの選択肢は主に3つで、手軽さと制御の細かさがトレードオフになっています。結論から言えば、**まずは OpenCV の `VideoCapture` で十分**で、精密なタイムスタンプやコーデック制御が必要になったら PyAV へ、というのが基本方針です。

下の表は3つの動画I/Oの位置づけです。表の前に要点を言うと、OpenCV はカメラもファイルも同じ API で読めて手軽（ただしフレームは BGR）、imageio は多形式を統一 API で扱えて RGB が返り手軽、PyAV は FFmpeg を同梱し最も低レベルで正確、という住み分けです。

| ライブラリ | 手軽さ | 色順 | 強み | こういう時 |
| --- | --- | --- | --- | --- |
| OpenCV `VideoCapture` | ◎ | BGR | カメラ/ファイル/RTSPを同一API。CPUで実時間 | まず最初に使う動画I/O |
| imageio (`imageio[ffmpeg]`) | ◎ | RGB | 画像/動画/GIFを統一API・RGBで返る | 多形式を手早く読み書き |
| PyAV (`av`) | △ | RGB等 | FFmpeg同梱・正確なtimestamp/コーデック制御 | フレーム単位の精密な制御 |

この表で覚えておきたいのは、**OpenCV の動画フレームも画像と同じく BGR で返る**こと、そして PyPI の opencv wheel は CPU ビルドなので `cv2.cuda`（GPU デコード）は使えないことです。本講座は CPU 前提なので、動画も `VideoCapture` で問題なく扱えます。PyAV と imageio-ffmpeg は wheel に FFmpeg バイナリを同梱しているため、システムに `ffmpeg` を入れなくても動くのが利点です。これらの実演は動画トラック（第9回以降）で行うので、ここでは「動画I/O にも地図がある」ことだけ頭に置いてください。

## 9. 選び方の判断基準（意思決定ガイド）

最後に、この章のエッセンスを「課題 → まず試すライブラリ」の早見にまとめます。`01_library_map.py` はこれと同じガイドをコンソールにも出力します。迷ったらこの順で考える、というチェックリストとして使ってください。完璧な正解を一つ選ぶより、**「まず試す候補」を即決して手を動かす**ほうが学習も実務も速く進みます。

下のガイドの前に大原則を一つ。**画像処理で迷ったら、まず OpenCV を試す。** OpenCV は機能が最も網羅的で速く、動画やカメラまでカバーするからです。そのうえで「直感的に編集したい→Pillow」「学習の前処理→torchvision v2」「検出/セグメの本格的拡張→albumentations」と、目的に応じて分岐していきます。

| やりたいこと | まず選ぶ |
| --- | --- |
| 画素を直接いじる・自作の処理を書く | NumPy（+ OpenCV） |
| 読込/保存/リサイズ/ぼかしなど定番処理を速く | OpenCV |
| 直感的に画像を編集・フォント描画・EXIF対応 | Pillow |
| 論文のアルゴリズム再現・計測・科学計算寄り | scikit-image |
| 検出/セグメ用に bbox/mask ごと拡張 | albumentations |
| PyTorch 学習の前処理・データ拡張 | torchvision transforms v2 |
| 拡張を GPU・学習ループ内で微分可能に回す | kornia |
| 動画/カメラを手早く読む | OpenCV `VideoCapture` |
| 動画を精密に（timestamp/コーデック）扱う | PyAV(`av`) / imageio-ffmpeg |
| headless で結果を見る・図にする | matplotlib(Agg) / ファイル保存 |

この表は暗記する必要はありませんが、**「自分が今やりたいことが左列のどれに当たるか」を言語化する習慣**をつけてください。課題を言葉にできれば、右列のライブラリは自然と決まります。そして一度選んでも、行き詰まったら別のライブラリへ乗り換えればよい——地図を持っていれば、その乗り換えも怖くありません。

## 10. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば理解が積み上がるように並べています。すべて `outputs/02_cv_libraries_overview/` に結果を保存し、画面表示には依存しません。共通処理（合成画像生成・出力先管理・任意ライブラリの導入判定 `probe`）は `cv_helpers.py` にまとめ、各スクリプトはそれを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `cv_helpers.py` | 合成画像生成・出力先・`probe`（任意ライブラリの導入判定）・`to_rgb`。各スクリプトが import する道具箱 |
| `01_library_map.py` | 主要ライブラリの早見表＋2軸の地図を生成。導入状況を点検し、未導入には導入コマンドを案内 |
| `02_same_op_across_libs.py` | resize/blur/rotate を OpenCV/Pillow/scikit-image で書き比べ、比較図と数値差を出す |
| `03_augmentation_albumentations.py` | 自作拡張パイプライン＋分布の可視化＋albumentations/torchvision v2/kornia の実演（任意） |
| `mini_project.py` | 章末ミニプロジェクト。書き比べ・相互変換・拡張・選定を統合し JSON＋図でレポート |
| `exercises.py` | TODO形式の演習10問（易→難・自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 全演習の模範解答（実行すると全 PASS。答え合わせ用） |

表の通り、`cv_helpers.py` だけは「読み物」ではなく「再利用する道具」です。とくに `probe()` は「入っていれば使う・無ければ案内してスキップ」を一手に引き受ける関数で、本講座が main 依存だけで完走できる仕掛けの中心です。最初に `cv_helpers.py` を一読してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。

## 🛠 章末ミニプロジェクト — ライブラリ選定レポート生成器（`mini_project.py`）

この回の学び（地図・相互運用・書き比べ・拡張）を 1 本に束ねる総合課題です。`mini_project.py` は合成画像 1 枚を入力に、**「どのライブラリをどう選び・どう橋渡しするか」を実測してレポート化**します。具体的には次の 4 つを統合します。

1. **同一処理の書き比べ（速度＋数値差）** — まったく同じ「正準リサイズ（半分に縮小）」を OpenCV（`INTER_AREA`）・Pillow（`LANCZOS`）・scikit-image（任意）で実行し、**1 回あたりの処理時間**と、OpenCV を基準にした**平均絶対差（MAD）**を測ります。第5節「同一処理の書き比べ」を、速度という新しい軸まで広げた発展版です。
2. **相互変換の無損失チェック（色順・軸順の自動検証）** — `cv2(BGR) → PIL(RGB) → numpy → cv2(BGR)` の往復、および torch があれば `cv2 → Tensor(C,H,W) → cv2` の往復が**画素まで完全一致するか**を検証します。第4節で学んだ「境界をまたぐたびに色順と軸順を意識する」を、合格/不合格で機械的に確かめます。
3. **拡張が分布を広げる度合いの定量化** — 自作の小さな拡張パイプライン（反転・明るさ/コントラスト・回転）を 300 回かけ、平均明るさの **std / range** を計算します。第7節の「拡張は分布をどう広げるか」を、ヒストグラムに加えて数値（広がりの大きさ）でも押さえます。
4. **意思決定表の自己整合チェック** — 「課題 → まず選ぶライブラリ」の対応表が一貫して引けること、未知の課題が既定の OpenCV に落ちることを点検します。

実行すると、結果は機械可読な `outputs/02_cv_libraries_overview/mini_project_report.json`（時間・MAD・往復可否・分布統計・環境のライブラリ版）と、`mini_project_summary.png`（左＝ライブラリ別リサイズ時間の棒グラフに MAD を注記／右＝拡張後の明るさ分布ヒストグラム）に出力されます。OpenCV が最速・MAD=0（基準）で、Pillow は数階調ぶん値がズレる——という「速度も画素値もライブラリで違う」事実を、図と JSON の両方で確認してください。`uv run python lectures/02_cv_libraries_overview/mini_project.py` で実行できます（CPU 完結・ネット不要・任意ライブラリは未導入でも動く）。

実務では、このレポートがそのまま**ライブラリ選定の判断材料**になります。「速度が効くホットパスは OpenCV」「学習と一体化したいなら torchvision/kornia」「往復で色が壊れていないか CI で機械チェック」——といった意思決定を、地図（定性）と実測（定量）の両面から下せるようになるのがゴールです。

## ✅ 到達チェックリスト

このモジュールを終えたら、次が「できる／説明できる」状態かを確認してください。

- [ ] 主要ライブラリ（OpenCV/Pillow/NumPy/scikit-image/imageio/PyAV/torchvision v2/albumentations/kornia/matplotlib）の**役割分担**を一言で言える。
- [ ] 各ライブラリの**色順（OpenCV だけ BGR）・データ表現（ndarray/PIL.Image/Tensor）・GPU/微分可能性**を早見表で見分けられる。
- [ ] 「低レベル⇔高レベル」「CPU⇔GPU/微分可能」の 2 軸で、任意のライブラリが地図のどの象限に来るかを説明できる。
- [ ] `cv2(BGR) ⇄ PIL(RGB) ⇄ numpy ⇄ Tensor(C,H,W)` の相互変換を、**色順と軸順**を正しく入れて書ける。
- [ ] `shape=(H,W)` / PIL `size=(W,H)` / `dsize=(W,H)` / skimage の `(H,W)` 順 / Tensor の `(C,H,W)` の**軸順の違い**を区別できる。
- [ ] 同じ resize/blur/rotate を OpenCV/Pillow/scikit-image で書き分けられ、**補間や値域（float[0,1]）の違い**を説明できる。
- [ ] uint8 の**オーバーフロー（numpy `+`）と飽和（`cv2.add`／clip）**の違いを説明し、意図どおりに書ける。
- [ ] データ拡張の目的（多様性の水増し→過学習抑制）と、albumentations/torchvision v2/kornia の**住み分け**を言える。
- [ ] 課題を見て「まず試すライブラリ」を即答でき、行き詰まったら地図上で隣のライブラリへ乗り換えられる。
- [ ] `mini_project.py` を動かし、JSON とサマリ図から**速度・数値差・往復可否・分布の広がり**を読み取れる。

## ❓ よくある落とし穴・FAQ・デバッグ

第12節の「症状→原因→対処」表に加えて、ライブラリをまたぐときに実際に詰まりやすい点を Q&A 形式で補足します。

- **Q. ライブラリを変えたら画像の色が変わった（赤青反転）。** → A. ほぼ確実に BGR/RGB の取り違えです。**OpenCV だけ BGR**、Pillow・matplotlib・scikit-image・torchvision はすべて RGB。境界をまたぐ前に `cv2.cvtColor(x, cv2.COLOR_BGR2RGB)`（戻すときは `RGB2BGR`）を入れます。`mini_project.py` の往復チェックが FAIL なら、まずここを疑います。
- **Q. リサイズで縦横が入れ替わった／scikit-image だけ結果が変。** → A. 軸順です。`cv2.resize` と `PIL.resize` の引数は **(幅, 高さ)**、`skimage.transform.resize` は **(高さ, 幅)**。`shape` は `(H,W)`、`size` は `(W,H)`。「いま触っているのは幅・高さどちらが先か」を毎回声に出して確認します。
- **Q. scikit-image の結果が真っ白／真っ黒になる。** → A. skimage は内部で `float[0,1]` を使います。`img_as_float` で 0〜1 に正規化してから処理し、保存・表示前に `img_as_ubyte` で 0〜255 の uint8 へ戻します。clip を忘れると範囲外で破綻します。
- **Q. 明るさを足したら暗い点が混じる（一部が真っ黒）。** → A. uint8 のまま `img + 50` するとオーバーフローして 255 超が 0 へ回り込みます。`cv2.add` を使うか、`int16`/`float32` に広げて加算 → `np.clip(0,255)` → `uint8` に戻します（演習7・ex7 参照）。
- **Q. `ModuleNotFoundError: albumentations`（または skimage/kornia）。** → A. これらは**任意グループ**です。`uv add --group aug albumentations scikit-image` 等で導入するか、案内に従ってスキップしてください。本講座の実行コードは未導入でも必ず最後まで動きます（`cv_helpers.probe()` がガード）。
- **Q. Tensor に変換したら形が合わない（チャンネルが画像に化ける）。** → A. numpy/OpenCV は `(H,W,C)`、PyTorch Tensor は `(C,H,W)`。`torch.from_numpy(rgb).permute(2,0,1)` で軸を入れ替え、戻すときは `permute(1,2,0)`（演習6・ex6 参照）。
- **Q. `cv2.imshow` が Docker/SSH で固まる・エラーになる。** → A. headless 環境では GUI が無く、`opencv-python-headless` には `imshow` 自体がありません。`plt.imshow`（Agg）か `cv2.imwrite`/`savefig` でファイルに保存して確認します。本モジュールは全スクリプトがファイル保存方式です。
- **Q. 拡張したら画像が崩れすぎて学習が悪化した。** → A. 拡張は「やるほど良い」わけではありません。`03_aug_distribution.png` や `mini_project.py` の std/range で**分布の広がり**を見て、真っ黒/真っ白や画面外への流出が混じらない強度に調整します。
- **Q. `opencv-python` と `opencv-python-headless`（や `opencv-contrib-python`）を一緒に入れたら挙動が変。** → A. これらは同じ `cv2` 名前空間を共有するため**同時インストール禁止**。どれか一つに統一します（本講座は headless 一本）。

## 🚀 発展トピック・参考

- **scikit-image を地図に加える**: セグメンテーション（`segmentation.slic`/`felzenszwalb`）、計測（`measure.regionprops`）、復元（`restoration`）など、OpenCV にない研究寄りアルゴリズムが豊富。論文再現や定量計測では第一候補になります（第8回以降の古典CVで再登場）。
- **torchvision transforms v2 の真価**: v2 は画像だけでなく **bounding box / mask / keypoint を同じ変換で同時に動かせる**よう設計されています（検出・セグメの学習で必須）。`tv_tensors` でラベル種別を保ったまま `Compose` に通すのが正準形。第12回「データパイプラインと拡張」で本格的に扱います。
- **albumentations のターゲット連動**: `A.Compose([...], bbox_params=...)` や `mask`/`keypoints` を渡すと、幾何変換が画像とアノテーションに**一貫して**適用されます。検出/セグメの拡張で albumentations が標準になる理由がここにあります。
- **kornia と微分可能 CV**: kornia は拡張だけでなく、特徴点・ホモグラフィ・エッジ等を**微分可能・GPU バッチ**で提供します。拡張を学習ループの内側に置く（テスト時拡張 TTA や敵対的訓練）用途で効きます。
- **動画 I/O の地図（第9回以降）**: まず `cv2.VideoCapture`、精密なタイムスタンプ/コーデック制御が要れば PyAV(`av`)、多形式を手軽になら imageio。torchvision 0.26+ は内蔵デコーダを廃止したため、動画読込は `VideoCapture` を基本にします。
- **公式ドキュメント**: OpenCV `https://docs.opencv.org/4.x/` ／ Pillow `https://pillow.readthedocs.io/` ／ scikit-image `https://scikit-image.org/` ／ torchvision transforms v2 `https://docs.pytorch.org/vision/stable/transforms.html` ／ albumentations `https://albumentations.ai/docs/` ／ kornia `https://kornia.readthedocs.io/`。
- **発展課題**: `mini_project.py` の書き比べに **blur と rotate** の列を足して 3 処理 × 速度/MAD を比較する／scikit-image・torch を導入して往復チェックと skimage 列が増える様子を確認する／自分の写真を `data/sample.jpg` に置いて実画像で同じレポートを出す、など。

## 11. 動かし方

このモジュールの実行コードは `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` だけに依存し、GPU もネット接続も不要です。サンプル画像が無くても合成画像が自動生成されるので、いきなり実行できます（`data/sample.jpg` を置けば、そちらが優先して使われます）。比較対象のライブラリ（scikit-image / albumentations / torchvision / kornia）は未導入でもスキップして必ず最後まで動きます。プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ）
uv sync

# 各スクリプトを実行（結果は outputs/02_cv_libraries_overview/ に保存される）
uv run python lectures/02_cv_libraries_overview/01_library_map.py
uv run python lectures/02_cv_libraries_overview/02_same_op_across_libs.py
uv run python lectures/02_cv_libraries_overview/03_augmentation_albumentations.py

# 章末ミニプロジェクト（書き比べ・相互変換・拡張・選定を統合し JSON＋図でレポート）
uv run python lectures/02_cv_libraries_overview/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL・10問・易→難）
uv run python lectures/02_cv_libraries_overview/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る（全 PASS を確認）
SHOW_SOLUTION=1 uv run python lectures/02_cv_libraries_overview/exercises.py
uv run python lectures/02_cv_libraries_overview/exercises_solutions.py

# （任意）比較対象ライブラリを入れると 02/03 の出力が増える
uv add --group aug scikit-image albumentations   # 02 に skimage 列 / 03 に albumentations グリッド
```

実行後は `outputs/02_cv_libraries_overview/` に生成された PNG を画像ビューアで開いてください。`01_library_cheatsheet.png`（早見表・緑=導入済み/赤=未導入）と `01_library_quadrant_map.png`（2軸の地図）で全体像を、`02_same_op_compare.png` で同一処理の書き比べを、`03_aug_distribution.png`（分布の広がり）と `03_manual_aug_grid.png`（拡張サンプル）で拡張の効果を、それぞれ目で確認します。コンソールにも早見表・選択ガイド・数値差が出るので、図と合わせて読んでください。

## 12. よくある落とし穴（チェックリスト）

最後に、この章で扱った内容を「症状 → 原因 → 対処」で一覧にします。多くは第1回からの地続きですが、ライブラリをまたぐ場面で改めて顔を出します。

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| ライブラリ間で渡したら赤青が反転 | OpenCV だけ BGR、ほかは RGB | 境界をまたぐ前に `cv2.cvtColor(BGR2RGB)` |
| リサイズで幅と高さが入れ替わる | `shape=(H,W)` と PIL `size=(W,H)`、`dsize=(W,H)` の混同 | 「いま幅・高さどちらが先か」を毎回確認 |
| scikit-image の結果が真っ白/真っ黒 | 出力が `float[0,1]` なのに `uint8` 扱いした | `img_as_ubyte` で 0〜255 へ戻す。`(H,W)` 順にも注意 |
| `ModuleNotFoundError: albumentations` | 任意グループ未導入 | `uv add --group aug albumentations`。または案内に従いスキップ |
| Tensor へ変換すると形が合わない | `(H,W,C)` と Tensor の `(C,H,W)` の違い | `permute(2,0,1)` で軸を並べ替える |
| 拡張したら画像が崩れすぎ | 拡張の強度が過激 | `03_aug_distribution.png` で分布を見て強度を調整 |

この表の6項目が、ライブラリをまたぐときの典型的なつまずきです。逆に言えば、**「色順」と「軸順」と「データ型（値域）」の3点さえ毎回確認すれば、どのライブラリの組み合わせでも事故は防げます**。この章のゴールは、まさにこの3点を反射的にチェックできるようになることです。

## 13. まとめ

この章では、画像・動画処理ライブラリの乱立を「地図」として整理しました。早見表で各ライブラリの色順・データ表現・GPU対応・役割を見比べ、2軸の散布図で守備範囲を掴み、同一処理（resize/blur/rotate）の書き比べで API の癖を体感し、データ拡張ライブラリ（albumentations / torchvision v2 / kornia）の住み分けと、拡張が分布を広げる様子を可視化で確認しました。アルゴリズムを覚えるより、**課題に応じてライブラリを選ぶ判断基準**を身につけたことが、この概念回の成果です。

次回以降は、この地図の左下（CPU の古典CV）から本格的に手を動かしていきます。幾何変換・フィルタ・エッジ・モルフォロジー・特徴点と、OpenCV を中心に積み上げる中で、「ここは Pillow のほうが楽」「ここは scikit-image の関数が便利」といった使い分けの感覚も育っていきます。まずは演習を自力で全問 PASS させ、相互変換（色順・軸順）とライブラリ選択の判断を体に入れてから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4 ／ opencv-python-headless 4.13（`cv2` 4.13.0）／ Pillow 12.2 ／ matplotlib 3.10。
> 本文で言及した任意ライブラリ（実行コードは未導入でもスキップ）: scikit-image ／ imageio 2.37 ／ PyAV(`av`) 17.1 ／ torch 2.12+cpu ／ torchvision 0.27（transforms v2）／ albumentations 1.4+ ／ kornia。