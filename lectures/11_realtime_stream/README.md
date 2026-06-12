# 第11回 リアルタイム・ストリーム処理 — 背景差分・CPU最適化・スレッド/プロセス分離・RTSP/再接続

> トラック: **動画・ストリーム** ／ レベル: **中級** ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません）。`video` グループは任意（yt-dlp 等は実ライブ接続を試すときだけ）

## 🎯 この章のゴール

第9回で動画の入出力（VideoCapture/VideoWriter）を、第10回で古典的な動き解析（背景差分・オプティカルフロー）を学びました。本章のテーマは、それらを「**止まらずに流し続ける**」こと——つまり**リアルタイム/ストリーム処理**です。録画済みファイルを後からじっくり処理するのと違い、ライブ映像は待ってくれません。1フレームの処理が映像の到着間隔より長ければ、未処理フレームが溜まって表示が現実から何秒も遅れていきます。この章を終えたとき、あなたは「**GPU が無くても、CPU だけで実時間に追いつかせる**」ための定石を、原理から実装まで自分の手で書けるようになります。

具体的な到達点は4つです。第一に、**背景差分（MOG2/KNN）＋モルフォロジー**で動体を検出し、warm-up・影・ノイズという実務上の癖を理解すること。第二に、**早期縮小・フレームスキップ・grab/retrieve**という3つのCPU最適化を、処理FPS・ストリーム消化レートの実測で比較できること。第三に、**取得と処理をスレッド/プロセスに分離**し、`queue.Queue(maxsize=1)` ＋ `put_nowait` の**フレームドロップ**で「レイテンシの雪だるま」を断ち切れること。第四に、**RTSP/ライブ配信の低遅延設定と再接続ループ**を書け、ネットワークが切れても落ちずに復旧できること。

本章のスクリプトはすべて、Webカメラもネット接続も無い環境で完走するよう、**合成フレームをその場で生成**します（静止背景の上を円が動く映像）。Webカメラ・RTSP・ライブ配信といった実入力は、`CV_CAM=1` / `CV_RTSP=<url>` / `CV_YOUTUBE=<url>` を明示したときだけ有効化する設計なので、**引数なしなら必ず `exit 0`** になります。リアルタイム処理は「環境に依存して動いたり動かなかったり」しがちな分野ですが、本章では合成ソースを土台に、原理そのものを確実に体得します。

---

## 1. リアルタイム処理の核心 — 「実時間に追いつく」とは何か

リアルタイム処理を一言でいえば「**映像が届く速さ以上の速さで処理し続ける**」ことです。ここで2つのFPSを区別するのが出発点になります。**ソースFPS**は映像そのもののフレームレート（30fpsで撮られた、など素材の属性）で、`cap.get(cv2.CAP_PROP_FPS)` で読めます。一方の**処理FPS**は、あなたのプログラムが実際に1秒あたり何枚さばけるかの実測値です。**処理FPS ≥ ソースFPS** なら間に合い、逆なら遅延がどんどん溜まります。この大小関係こそが、リアルタイム処理で最初に意識すべき不等式です。

では間に合わないとき、何が起きるのでしょうか。鍵は「**レイテンシ（遅延）の蓄積**」です。処理が追いつかないと、未処理フレームがバッファに溜まり、いま処理しているフレームは「数秒前の世界」になります。防犯カメラで人が映ってから検知まで5秒遅れたら、もはやリアルタイムとは呼べません。そこで本章が繰り返し強調する発想が、「**遅れて全部処理するくらいなら、古いフレームは捨てて最新に追いつく**」というフレームドロップです。直感に反するようですが、リアルタイムでは「最新の状態を低遅延で知る」ことが「全フレームを漏れなく処理する」ことより重要な場面が多いのです。

この章では、追いつかせるための手段を段階的に積み上げます。まず**処理そのものを軽くする**（解像度を落とす・フレームを間引く・無駄なデコードを省く＝第3節）。それでも足りなければ**処理を別スレッド/プロセスへ逃がして並行化し、溢れたフレームは落とす**（第4・5節）。そして**ネットワーク起因の遅延と切断に備える**（第6節）。どれも派手なGPUを使わず、CPUだけで「間に合わせる」ための現場の知恵です。順番に見ていきましょう。

## 2. 背景差分による動体検出（MOG2 / KNN）

固定カメラの映像では、背景はほとんど動きません。ならば「動かない背景」を統計モデルとして学習し、そこから外れた画素を**前景＝動いているもの**とみなせます。これが**背景差分（background subtraction）**で、OpenCV には `cv2.createBackgroundSubtractorMOG2`（混合ガウス分布）と `cv2.createBackgroundSubtractorKNN`（k近傍）が本体同梱で用意されています（contrib 不要）。学習を一切使わずCPUだけで軽快に動くため、ストリーム処理の最初の実例として最適です。第10回で学んだ単純なフレーム差分（前フレームとの引き算）と違い、背景差分は**過去数百フレームから背景を学習し続ける**ので、ゆっくりした照明変化に強く、止まっている物体を背景に溶け込ませることもできます。

使い方の中心は `apply()` の一行です。フレームを1枚渡すと、その画素を背景モデルと比較した**前景マスク**（前景=255）が返り、同時にモデルも内部で更新されます。ここで初学者が必ず戸惑うのが2点。ひとつは **warm-up（立ち上がり）**で、最初の数フレームは背景がまだ学習されておらず、画面の大部分が前景判定されます。もうひとつは**影**で、`detectShadows=True` のとき影と判断された画素は前景(255)ではなく**127**という中間値で返ります。下のコードのように、影(127)を捨てて2値化し、モルフォロジーでノイズを掃除するのが定石です。

```python
sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=True)
raw = sub.apply(frame)                                  # 前景マスク（影は127で返る）
_, binary = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)   # 影127を捨て、255だけ残す
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)   # ぽつぽつ除去
clean  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)  # 穴埋め
```

上の `clean` から `cv2.findContours` で輪郭を取り、小さすぎる領域を捨てて `cv2.boundingRect` で外接矩形にすれば、動体のバウンディングボックスが得られます（`findContours` は OpenCV 4 系で `(contours, hierarchy)` の**2つ返し**である点に注意）。`01_background_subtraction.py` はこのパイプラインを `[入力 | 生マスク | 掃除後マスク | 検出結果]` の4枚パネルとして保存し、warm-up の様子（前景率の推移）もグラフ化します。MOG2 と KNN の使い分けは次の通りで、迷ったら速くて定番の MOG2 から始めるのが無難です。

| 手法 | 内部モデル | 特徴 | 向いている場面 |
| --- | --- | --- | --- |
| `MOG2` | 混合ガウス分布 | 速い・定番。`varThreshold` で感度調整 | まず試す既定。一般的な動体検出 |
| `KNN` | k近傍 | 前景が小さい/まばらな場面で頑健なことがある | 細かい/低密度の動体、MOG2 で取りこぼす時 |

背景差分はあくまで「動いた画素」を取るだけで、**それが何か（人・車）は分かりません**。物体の種類が必要なら第18回以降の深層検出が、物体間の対応付け（追跡）が必要なら第28回が担当します。背景差分は「軽く・速く・動きの当たりを付ける」前段として、検出器を毎フレーム走らせる代わりに「動いた領域だけ重い処理に回す」といった最適化（第3節のスキップと相性が良い）に使うのが実務での王道です。

## 3. CPUで実時間化する3つの定石（縮小・スキップ・grab/retrieve）

処理が間に合わないとき、最初に効くのは「**処理する画素を減らす**」ことです。多くの画像処理は画素数に比例して重くなるので、640×480 を 320×240 に落とすだけで画素数は1/4、計算量もおおむね1/4になります。鉄則は**重い処理の「前」に縮小する**こと、そして縮小の補間は **`cv2.INTER_AREA`**（モアレが出にくい縮小の定石）を使うことです。「640×480 程度に落とせばCPUのみでも実時間処理が可能」というのが現場の感覚で、入力をいきなり原寸で重い処理に渡すのは最ももったいないパターンです。

次に効くのが**フレームスキップ**で、毎フレームではなく N フレームに1回だけ重い処理を走らせ、間のフレームは前回の結果を使い回します。検出結果は数フレーム前のものでも実用上問題ないことが多く、フレームレートを稼ぐ最も手軽な手段です。さらに OpenCV 特有の最適化が **grab/retrieve** です。`cap.read()` は内部的に「デコード＋numpy変換＋コピー」を毎回行いますが、`cap.grab()` は**内部バッファを進めるだけ（安価）**で、本当に必要なフレームだけ `cap.retrieve()` でデコードします。カメラ/RTSP では「溜まったフレームを `grab()` で安価に読み飛ばして最新に追いつく」用途でも使います。

```python
while True:
    grabbed = cap.grab()              # 安価: デコードしないでバッファを1つ進める
    if not grabbed:
        break
    seen += 1
    if seen % 3 != 0:                 # 3枚に1枚だけ
        continue                      # retrieve() を呼ばない＝デコード/コピーを省く
    ok, frame = cap.retrieve()        # ここで初めてデコードして numpy 化
    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # 早期縮小
    heavy_process(small)
```

`02_frameskip_grab_retrieve.py` は合成動画を mp4 に書き出してから読み戻し、4シナリオを実測します。評価の主指標は「**ストリーム消化レート**＝入力フレームを1秒に何枚さばけたか（= 見たフレーム数 / 総時間）」で、これが**ソースFPSを上回っていれば実時間に追いつける**という解釈ができます。手元のCPUでの実測例（環境で変動します）が下表です。縮小（B）・スキップ（C）・grab併用（D）と積むほどレートが上がり、特に D は不要フレームのデコード自体を省くため最速になります。

| シナリオ | 見た枚数 | 処理枚数 | 消化レート[frames/s] | 効いている最適化 |
| --- | --- | --- | --- | --- |
| A: 原寸・毎フレーム | 120 | 120 | 約1150 | （ベースライン） |
| B: 縮小・毎フレーム | 120 | 120 | 約1420 | 早期縮小で処理が軽い |
| C: 縮小＋3枚に1回 | 120 | 40 | 約2150 | ＋フレームスキップ |
| D: grab/retrieve＋縮小 | 120 | 40 | 約3170 | ＋不要フレームのデコード省略 |

数値の絶対値より「**A < B < C < D の順にレートが上がる**」という関係を押さえてください（いずれもソースFPS=30をはるかに上回っており、CPUだけで余裕で間に合うことが分かります）。注意点として、フレームスキップは「動きの速い物体を見逃しやすくなる」副作用があり、grab で読み飛ばすのも「捨てたフレームは戻ってこない」ので、**精度と速度のトレードオフ**であることは忘れないでください。どこまで落としてよいかはタスク要件で決まります。

## 4. 取得と処理のスレッド分離（producer/consumer）

縮小やスキップで処理を軽くしても足りないなら、次は**並行化**です。ここで OpenCV 特有の好都合があります。`cap.read()` はディスク/ネットワークI/Oの待ち時間が長く、その間 Python は **GIL（グローバルインタプリタロック）を解放**します。つまり「取得専用スレッド」と「処理側」を分ければ、読み込み待ちの隙に処理を進められます。これが **producer/consumer 構成**で、producer がひたすらフレームを読んでキューに入れ、consumer（処理側）がキューから取り出して処理します。

このとき決定的に重要なのが**キューのサイズとドロップ戦略**です。無制限キューにすると、consumer が追いつかない分だけ未処理フレームが溜まり、取り出した頃には「何秒も前のフレーム」になってレイテンシが雪だるま式に増えます。これを防ぐ定石が **`queue.Queue(maxsize=1)` ＋ `put_nowait`** で、キューが満杯なら新フレームを捨て（`queue.Full` を握りつぶす）、常に「最新1枚」だけを保ちます。下が producer 側の核心です。`time.sleep` は入力到着間隔の模擬で、実際にはカメラ/ネットワークがこの間隔を決めます。

```python
def producer(source, q, stop_event):
    while not stop_event.is_set():
        ret, frame = source.read()
        if not ret:
            break
        try:
            q.put_nowait((time.perf_counter(), frame))  # 生成時刻を添えて入れる
        except queue.Full:
            pass                                        # 満杯 → 新フレームを捨てる（遅延蓄積を防ぐ）
        time.sleep(1 / produce_fps)
    stop_event.set()
```

`03_threaded_capture.py` は「無制限キュー（捨てない）」と「maxsize=1＋ドロップ」を同じ入力で走らせ、各処理フレームの**齢（age＝生成から処理までの遅れ）**を比較します。手元の実測では、無制限キューは平均齢が**約700ms以上に増え続ける**のに対し、ドロップ版は**約60msで横ばい**、その代わり約3割のフレームを捨てていました。`03_latency_vs_drop.png` の折れ線（無制限＝右肩上がり、ドロップ＝横ばい）を見れば、「**ドロップ率と引き換えに遅延の蓄積を断ち切る**」というリアルタイム設計の本質が一目で腑に落ちます。

なお、ドロップには「新しい方を捨てる（`put_nowait` で満杯なら諦める）」と「古い方を捨てる（一度 `get` してから `put`）」の2流派があります。最新への追従を最優先するなら後者が低遅延ですが、実装が単純で `put_nowait`/`queue.Full` という標準APIにそのまま乗る前者を本章では採用しています。daemon スレッドにしておくと本体終了を妨げない、`threading.Event` で安全に停止フラグを渡す、といった作法もコードで確認してください。

## 5. multiprocessing でCPUバウンドを分離（GIL回避）

スレッド分離はI/O待ち（`read()` のデコード待ちなど）には有効ですが、**純粋なCPU計算には効きません**。GILがあるため、Python のスレッドは同時に1つしかバイトコードを実行できず、重い推論や画像処理を別スレッドに置いても**並列化されない**のです。この限界を超える手段が **`multiprocessing`** で、別プロセスは独立した Python インタプリタ（独立したGIL）を持つので、CPUバウンドな処理を本当に並列実行できます。

書き方はスレッド版とよく似ていて、`multiprocessing.Process` で取得プロセスを起こし、`multiprocessing.Queue(maxsize=1)` でフレームを受け渡し、満杯ならドロップする——という構図は同じです。違いは、プロセス間ではメモリを共有しないため、フレーム（numpy配列）が**pickle 化されてコピーされる**点です。大きな配列を高頻度で渡すとこのシリアライズ自体がボトルネックになり得るので、**渡す前に縮小する**、共有メモリを使う、といった配慮が要ります。本番のストリームパイプライン（参照: Cluster-CLIP の `stream/capture.py` が `put_nowait` で即ドロップ）はこのプロセス分離型が定石です。

```python
ctx = mp.get_context("spawn")          # fork 依存を避け移植性を上げる（mac/Windows でも動く）
q = ctx.Queue(maxsize=1)
p = ctx.Process(target=producer, args=(q,), daemon=True)
p.start()
# ... 本体プロセスが q.get() で取り出して重い処理（GIL を共有しないので真に並列）...
p.join(timeout=2.0)
if p.is_alive():
    p.terminate()                      # 念のため後始末（ハング防止）
```

`03_threaded_capture.py` は、この multiprocessing デモを **`CV_MP=1` のときだけ**実行します（既定はスレッド版のみ）。使い分けの指針はシンプルで、**ボトルネックがI/O待ちならスレッドで十分、CPU計算ならプロセス分離**。そして「`spawn` コンテキストを明示してプラットフォーム差を吸収する」「`daemon=True` ＋ `terminate()` でハングを防ぐ」のが、安全に運用するための定番作法です。

## 6. ネットワークストリーム（RTSP / ライブ配信）と再接続

最後はネットワーク越しの映像です。防犯カメラの **RTSP** やライブ配信は、ローカルファイルと違って「**途中で切れるのが普通**」です。だから本番のストリームアプリは、`read()` の失敗を即クラッシュにせず、「**開き直して続ける**」再接続ループとして書きます。あわせて低遅延化の設定も入れます。RTSP では UDP のパケロス対策に環境変数 `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`（**VideoCapture を作る「前」に**設定する点が肝）、バックエンドに `cv2.CAP_FFMPEG` を明示、`CAP_PROP_BUFFERSIZE=1` で内部バッファを最小化して「古いフレームの溜め込み＝遅延」を抑えます。

```python
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"   # 作る前に設定
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # 溜めない（最新に追従して低遅延化）
```

再接続ループの骨格は「`read()` が失敗したら `release()` → **指数バックオフ**で少し待つ → `open()` し直す → 成功したらバックオフをリセットして処理継続」です。バックオフ（待ち時間を失敗のたびに倍にする・上限つき）で再接続の連打を避けます。そして**ライブ入力には「終端」が無い**ので、`CAP_PROP_FRAME_COUNT` で総数を当てにしたループは禁物（ライブでは 0/不正値になり得る）。終了条件は「目標枚数さばけたら正常終了」または「**連続失敗が一定回数を超えたらストリーム断と判断して諦める**（無限リトライ回避）」とします。

```python
while good < target_good:
    ret, frame = source.read()
    if not ret:
        source.release(); reconnects += 1; consecutive += 1
        if consecutive >= give_up_after:
            break                          # 連続失敗が続いたら諦める
        time.sleep(backoff); backoff = min(backoff * 2, 0.2)   # 指数バックオフ
        source.open(); continue            # 本物なら VideoCapture を作り直す
    consecutive = 0; backoff = 0.01        # 復旧したらリセット
    process(frame); good += 1
```

`04_rtsp_youtube_stream.py` は、ネット/カメラ無しでもこの再接続を学べるよう、**「途中2回わざと切断する合成ソース（FlakySource）」**で既定デモを動かします（切れても落ちずに復旧して処理を継続できることを確認）。実入力は、`CV_RTSP=<url>` で RTSP、`CV_YOUTUBE=<url>` で **yt-dlp** によるライブURL解決（`yt-dlp -g` で直URLを取って VideoCapture へ）、`CV_CAM=1` で Webカメラ——を明示したときだけ有効化します。なお **yt-dlp はサイト変更で壊れやすく定期更新が前提**で、Docker でホストのカメラを使うには `--device=/dev/video0` が必要、という運用上の注意も押さえておきましょう。

## 7. 性能プロファイルの読み方（p50/p99・EMA・ドロップ率）

リアルタイム処理は「速くなった気がする」では評価になりません。**数値で律速段（ボトルネック）を特定**するのが本章の評価方針です。まずスループットは**処理FPS**ですが、瞬間値は跳ねやすいので**EMA（指数移動平均）**でならして「いまおおよそ何FPS出ているか」を安定表示します（`stream_helpers.FpsMeter`）。次にレイテンシは平均だけ見ると「たまの詰まり」を見落とすので、**p50（中央値）と p99（上位1%の遅さ）**の両方を見ます。p99 が要件を超えていれば、平均は良くても「時々カクつく」パイプラインだと分かります。

そして本章ならではの指標が**フレームドロップ率**です。リアルタイムでは「全フレーム処理したか」より「最新に追従できているか（齢が低いか）」が重要なので、ドロップ率と平均齢（レイテンシ）をセットで見ます。第3〜4節で見たように、**ドロップ率が上がる代わりに齢が下がる**——この交換レートを把握して、タスクが許す範囲でドロップを許容するのが設計判断です。`time.perf_counter`（壁時計より安定した高分解能タイマ）で各ステージを区切って測り、どの段が一番時間を食っているかを見れば、縮小・スキップ・スレッド分離のどれを足すべきかが決まります。

これらの計測は、最適化の**前後で同じ指標を取って比較**してはじめて意味を持ちます。「縮小したら消化レートが1150→1420に上がった」「ドロップを入れたら平均齢が700ms→60msに下がった」のように、必ず数値で効果を確認する癖をつけてください。当て推量で最適化を積むと、効かない場所に労力を割いたり、精度を無駄に犠牲にしたりします。**測ってから直す**——これはリアルタイムに限らず性能改善の鉄則です（より体系的なプロファイリングは第34回で深掘りします）。

## 8. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「軽くする → 並行化する → ネットワークに備える」と理解が積み上がるように並べています。すべて `outputs/11_realtime_stream/` に結果を保存し、画面表示（`cv2.imshow`）には一切依存しません。共通処理（合成フレーム生成・合成ソース・FPS/レイテンシ計測・重い処理の模擬）は `stream_helpers.py` にまとめ、各スクリプトはそれを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `stream_helpers.py` | 合成フレーム生成・`SyntheticCapture`・`open_source`(CV_CAM対応)・`FpsMeter`/`percentiles`/`simulate_heavy_work`。道具箱 |
| `01_background_subtraction.py` | MOG2/KNN で前景マスク、影127除去＋モルフォロジー、輪郭→外接矩形、warm-up 可視化 |
| `02_frameskip_grab_retrieve.py` | 早期縮小(INTER_AREA)・フレームスキップ・grab/retrieve を実測比較（消化レート/p50/p99） |
| `03_threaded_capture.py` | producer/consumer スレッド分離、maxsize=1＋put_nowait ドロップ、齢の比較、(任意)multiprocessing |
| `04_rtsp_youtube_stream.py` | 再接続ループ＋指数バックオフ、RTSP低遅延設定、yt-dlp 解決、CV_CAM/CV_RTSP/CV_YOUTUBE ガード |
| `mini_project.py` | **章末ミニプロジェクト**: 背景差分→CPU最適化→スレッド分離+ドロップ→プロファイルを1本に統合したリアルタイム動体検出ストリーム |
| `exercises.py` | TODO 形式の演習10問（易→難。自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 演習の完全な模範解答（実行すると全10問 PASS。採点ロジックは `exercises.py` を再利用） |

表の通り、`stream_helpers.py` だけは「読み物」ではなく「再利用する道具」です。中身も厚くコメントしてあるので、最初に一読してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。とりわけ `synthetic_frames`（静止背景＋動く円＋ノイズ）が、背景差分・最適化・スレッド・再接続のすべての練習台になっている点に注目してください。

## 9. 動かし方

このモジュールは `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` だけに依存し、GPUもネット接続もWebカメラも不要です。合成フレームが自動生成されるので、いきなり実行できます。プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ。00〜11 は uv sync だけで完走）
uv sync

# 各スクリプトを実行（結果は outputs/11_realtime_stream/ に保存される）
uv run python lectures/11_realtime_stream/01_background_subtraction.py
uv run python lectures/11_realtime_stream/02_frameskip_grab_retrieve.py
uv run python lectures/11_realtime_stream/03_threaded_capture.py
uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py

# 章末ミニプロジェクト: この回の学びを統合したリアルタイム動体検出ストリーム
uv run python lectures/11_realtime_stream/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL）。全10問・易→難
uv run python lectures/11_realtime_stream/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/11_realtime_stream/exercises.py
# 完全な模範解答（実行すると全10問 PASS）
uv run python lectures/11_realtime_stream/exercises_solutions.py

# （任意）プロセス分離(GIL回避)のデモも試す
CV_MP=1 uv run python lectures/11_realtime_stream/03_threaded_capture.py

# （任意・要環境）実カメラ/RTSP/ライブ配信を入力にする
CV_CAM=1                uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py
CV_RTSP=rtsp://...      uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py
CV_YOUTUBE=https://...  uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py   # 要 yt-dlp
```

実行後は `outputs/11_realtime_stream/` の画像を開いて解説と照らし合わせてください。特に `03_latency_vs_drop.png`（無制限キューは右肩上がり、ドロップ版は横ばい）と `02_throughput_compare.png`（縮小・スキップ・grab で消化レートが上がる）を見比べると、本章の2大テーマ（最適化とドロップ）が視覚的に腑に落ちます。`cv2.imshow` はheadless環境で固まる/落ちるため本章では使わず、結果はすべてファイル保存です。どうしてもローカルGUIで見たい場合のみ、`opencv-python-headless` を `opencv-python`（GUI版）に差し替えてください（両者は排他）。

## 10. よくあるエラーと対処（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。リアルタイム処理は環境依存の罠が多いので、詰まったらまずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| 表示が現実より数秒遅れる | 処理FPS < ソースFPS でレイテンシが蓄積 | 縮小・スキップで軽量化。`maxsize=1`＋`put_nowait` で最新だけ保つ |
| スレッド分離したのに速くならない | CPUバウンド処理は GIL で並列化しない | `multiprocessing` でプロセス分離する |
| `cv2.imshow` でフリーズ/プロセスごと落ちる | headless 環境にGUIバックエンドが無い | `imwrite`/動画保存で確認。headless では imshow を呼ばない |
| ライブ入力で `for` ループが終わらない/壊れる | `CAP_PROP_FRAME_COUNT` がライブで不正値 | 総数に頼らず `ret` 判定だけで回す |
| RTSP がカクつく/壊れる | UDPパケロス・バッファ溜め込み | `rtsp_transport;tcp`（作る前に設定）＋`BUFFERSIZE=1` |
| 再接続が無限ループする | 切断と「ストリーム終端」を区別していない | 連続失敗が閾値を超えたら諦める（give-up） |
| matplotlib の図で日本語が豆腐(□)になる | 既定フォントにCJKグリフが無い | 図中の文字はASCIIにする（本章はそうしている） |
| 色が変（赤青が逆）に見える | BGRのまま matplotlib/PIL に渡した | `cv2.cvtColor(..., COLOR_BGR2RGB)` を挟む |

この表の8項目が、本章で遭遇しがちな不具合のほぼ全てです。とりわけ上4つ（レイテンシ蓄積・GIL・headless表示・ライブの総数）はリアルタイム処理の「あるある」なので、症状を見たら原因を即座に言い当てられるようにしておきましょう。

## 11. まとめ

本章では、リアルタイム/ストリーム処理を「**CPUだけで実時間に追いつかせる**」という一点に絞って、背景差分による動体検出、早期縮小・フレームスキップ・grab/retrieve による軽量化、producer/consumer のスレッド/プロセス分離とフレームドロップ、RTSP低遅延設定と再接続ループまでを、すべて合成フレームの上で「自分で再現し、数値で確認できる」レベルで扱いました。通底するのは「**測ってから直す**」「**遅れて全部処理するより、捨てて最新に追いつく**」という2つの発想です。

ここで身につけた「取得と処理を分け、キューでつなぎ、満杯なら落とす」という骨格は、第28回の追跡や、最終章（第40・41回）の Cluster-CLIP ストリームパイプラインまでそのまま効いてきます。まずは演習を全問 PASS させ、`03_latency_vs_drop.png` の2本の折れ線が意味することを自分の言葉で説明できるようにしてから、次へ進んでください。

---

## 🛠 章末ミニプロジェクト — CPUだけで成立させるリアルタイム動体検出ストリーム

ここまでの部品（背景差分・CPU最適化・スレッド分離＋ドロップ・性能プロファイル）を **1 本のストリームアプリ**に統合する総合課題です。`mini_project.py` を実行すると、合成フレーム（静止背景の上を円が動く）を入力に、次の 4 ステージが順に走ります。

1. **STAGE 1 — 動体検出パイプライン**: 早期 `resize(INTER_AREA)` → `MOG2` 背景差分 → モルフォロジ掃除（影127除去＋open/close）→ 輪郭→外接矩形、という検出の一連を回し、十分に学習済みのフレームを `[入力 | 掃除後マスク | 検出枠]` の 3 枚パネルで保存する。
2. **STAGE 2 — CPU 最適化（同期・決定的）**: 「原寸・毎フレーム」と「早期縮小＋3枚に1回」を同じ入力で回し、**ストリーム消化レート [frames/s]** と検出段レイテンシ（p50/p99）を比較する。縮小＋スキップで不要な重い検出を省けるぶん、消化レートが上がることを数値で確認する。
3. **STAGE 3 — スレッド分離＋フレームドロップ**: 一定間隔でフレームを供給する producer スレッドに対し、「**無制限キュー（捨てない）**」と「**`maxsize=1` ＋ `put_nowait`（最新だけ）**」の 2 つの consumer 戦略を走らせ、各フレームの**齢（age＝生成→処理の遅れ）**とドロップ率を比較する。consumer は実際に背景差分検出を行い、下流の重い推論コストも模擬する。
4. **STAGE 4 — プロファイル出力**: p50/p99 レイテンシ・処理FPS(EMA)・ドロップ率を JSON とまとめ図に書き出す。

この課題は「固定/移動カメラの映像から動体を低遅延で検出し続ける」という、防犯・監視・入退室カウントなどの最小核です。入力を Webカメラ・動画ファイル・RTSP・ライブ配信に差し替えれば、同じ骨格（取得と処理を分け、キューでつなぎ、満杯なら落とす）がそのまま実運用で効きます。これは最終章（第40・41回）の Cluster-CLIP ストリームパイプラインへ直結する部品でもあります。

**到達の目安**: STAGE2 で「縮小＋スキップ」の消化レートが「原寸・毎フレーム」を上回ること。STAGE3 で**無制限キューの齢が右肩上がりに増える**一方、**ドロップ版の齢はほぼ一定に保たれる**（その代わり一定割合のフレームを捨てる）こと。出力は `outputs/11_realtime_stream/` に以下が保存されます。

| 生成物 | 内容 |
| --- | --- |
| `mini_project_detection.png` | 背景差分の検出サンプル（入力／掃除後マスク／検出枠 の3枚パネル） |
| `mini_project_cpu_opt.png` | 原寸毎フレーム vs 縮小＋スキップ の消化レート棒グラフ |
| `mini_project_latency.png` | 無制限キュー（右肩上がり）vs ドロップ（横ばい）の齢の折れ線 |
| `mini_project_summary.png` | 上記＋数値サマリを 1 枚に並べたまとめ図 |
| `mini_project_metrics.json` | 検出数・消化レート・p50/p99・平均齢・ドロップ率・処理FPS(EMA) の数値ログ |

```bash
uv run python lectures/11_realtime_stream/mini_project.py
cat outputs/11_realtime_stream/mini_project_metrics.json
```

## ✅ 到達チェックリスト

この章を終えたら、次が**できる／説明できる**ことを確認してください。

- [ ] **ソースFPS と処理FPS** の違いを説明し、`処理FPS ≥ ソースFPS` でないとレイテンシが蓄積する、という不等式を言える。
- [ ] `cv2.createBackgroundSubtractorMOG2/KNN` の `apply()` で前景マスクを作り、**warm-up（序盤は前景だらけ）と影(127)** という2つの癖を説明できる。
- [ ] 生マスクを **`threshold(>200)`→`MORPH_OPEN`→`MORPH_CLOSE`** で掃除し、`findContours`（4系は**2つ返し**）＋`contourArea`＋`boundingRect` で動体の外接矩形を取れる。
- [ ] **早期縮小（`INTER_AREA`）・フレームスキップ・grab/retrieve** の3定石を、それぞれ「何を省くのか」とともに説明し、消化レートの変化を実測で示せる。
- [ ] `cap.grab()` が**安価（デコードしない）**で `cap.retrieve()` が**デコードする**こと、両者の使い分け（古いフレームの安価な読み飛ばし）を説明できる。
- [ ] producer/consumer のスレッド分離が効くのは **`read()` が I/O 待ちで GIL を解放するから**だと説明できる。
- [ ] `queue.Queue(maxsize=1)` ＋ `put_nowait`／`queue.Full` で**満杯時に新フレームを捨てる**ドロップを自力で書き、「齢を一定に保つ代わりにドロップ率が上がる」交換を数値で示せる。
- [ ] **CPUバウンド処理は GIL のためスレッドでは並列化しない**こと、その場合は `multiprocessing`（`spawn` 明示・配列は pickle コピー）で分離する、と判断できる。
- [ ] RTSP の低遅延設定（**`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` は作る前に設定**・`CAP_FFMPEG`・`CAP_PROP_BUFFERSIZE=1`）を書ける。
- [ ] **再接続ループ＋指数バックオフ**を書き、「ライブ入力に総数は当てにできない」「連続失敗が閾値を超えたら諦める」を実装できる。
- [ ] **p50/p99・EMA・ドロップ率**で律速段を特定し、「測ってから直す」を実践できる。ミニプロジェクトを実行し、2つの比較図を自分の言葉で説明できる。

## ❓ よくある落とし穴・FAQ・デバッグ

実装中に詰まったら、まずここを見てください（第10節の症状別チェックリストと併せて参照）。多くの不具合は次の数個の原因に集約されます。

- **Q. スレッド分離したのに全く速くならない。** A. ボトルネックが **CPUバウンドな計算**なら、GIL のためスレッドでは並列化しません。スレッドが効くのは `read()`/デコードのような **I/O 待ち**だけ。計算が重いなら `multiprocessing` でプロセス分離します（ただし配列の pickle コピー代に注意し、渡す前に縮小する）。
- **Q. `put_nowait` でドロップしているのに表示がどんどん遅れる。** A. キューの `maxsize` が大きすぎる（実質ドロップしていない）か、**ドロップしているのは入口だけで内部バッファ（`CAP_PROP_BUFFERSIZE`）が溜めている**可能性。カメラ/RTSP では `BUFFERSIZE=1`、または `grab()` で溜まったフレームを安価に読み飛ばしてから `retrieve()` します。
- **Q. 背景差分の序盤が画面まっ白（全部前景）。** A. **warm-up** です。最初の数十フレームは背景が未学習なので前景だらけになります。`history` ぶんのフレームを流して学習が進んでから結果を使う、もしくは序盤を捨てます。
- **Q. 前景マスクに灰色(127)が混じる／影が物体として検出される。** A. `detectShadows=True` のとき影は **127** で返ります。`threshold(raw, 200, 255, THRESH_BINARY)` で 255 だけ残して 2 値化してから後段に渡します。
- **Q. `findContours` が `ValueError: too many values to unpack` で落ちる。** A. OpenCV **3 系向けの 3 つ返し**サンプルです。4 系は `contours, hierarchy = cv2.findContours(...)` の**2つ返し**です。
- **Q. `cv2.VideoWriter` で動画を書いたのに 0 バイト／壊れている。** A. **出力サイズ (W,H) が write するフレームと 1px でも違う**と、エラーも出ずに壊れます。`write` 前に `resize` でサイズを合わせ、`writer.isOpened()` を必ず確認。移植性重視なら `mp4v`(.mp4) か `XVID`(.avi)。
- **Q. `cv2.imshow` を呼んだらフリーズ／プロセスごと落ちた。** A. headless 環境（Docker/CI/`opencv-python-headless`）には GUI バックエンドがありません。本章のように `imwrite`／`VideoWriter`／`imencode`(MJPEG配信) で確認します。ローカルで見たいときだけ `opencv-python`(GUI版) に差し替え（両者は排他）。
- **Q. RTSP 接続がカクつく／フレームが壊れる。** A. UDP のパケロスです。`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` を **`VideoCapture` を作る前に**設定し、`cv2.CAP_FFMPEG` を明示、`CAP_PROP_BUFFERSIZE=1` で溜め込みを防ぎます。
- **Q. ライブ入力で `for i in range(total)` が終わらない／途中で壊れる。** A. `CAP_PROP_FRAME_COUNT` はライブで **0/不正値**になり得ます。総数に頼らず `ret` 判定だけで回し、終了は「目標枚数」か「連続失敗の give-up」で決めます。
- **Q. 再接続ループが無限に回り続ける。** A. **切断とストリーム終端を区別していない**からです。連続失敗回数が閾値を超えたら「ストリーム断」と判断して諦め（give-up）、指数バックオフ（上限つき）で再接続の連打も避けます。
- **Q. `multiprocessing` が mac/Windows で `RuntimeError`／子プロセスが暴走。** A. `mp.get_context("spawn")` を明示し、エントリを `if __name__ == "__main__":` で守ります。`daemon=True` ＋ 後始末の `terminate()` でハングも防ぎます。
- **Q. matplotlib の図で日本語が豆腐(□)になる／色が反転する。** A. 図中の文字は **ASCII** にします（本章はそうしている）。色は cv2 が **BGR**、matplotlib/PIL は RGB なので `cv2.cvtColor(img, COLOR_BGR2RGB)` を挟みます。

## 🚀 発展トピック・参考

- **「古い方を捨てる」ドロップ**: 本章は `put_nowait` で**新しい方を捨てる**実装ですが、最新追従を最優先するなら「一度 `get` してから `put`」で**古い方を捨てる**流派もあります。`queue.Queue` ではなく `collections.deque(maxlen=1)` を使うと、満杯時に自動で最古要素が押し出されるので簡潔に書けます。
- **共有メモリでコピーを省く**: `multiprocessing` で大きな numpy 配列を高頻度に渡すと pickle 化がボトルネックになります。`multiprocessing.shared_memory.SharedMemory` やリングバッファでゼロコピー受け渡しにすると効きます（本番ストリームの定石）。
- **PyAV / imageio で精密なデコード**: OpenCV で扱いにくいタイムスタンプ・ピクセルフォーマット制御は `av`（FFmpeg 同梱・system 不要）の `av.open → container.decode → frame.to_ndarray` が向きます。`imageio[ffmpeg]` は RGB フレームを手軽にイテレートできます。
- **ハードウェアデコード（任意・上級）**: PyPI の `opencv-python(-headless)` は **CUDA 無効ビルド**で `cv2.cuda.*` は使えません。HWデコード（Mac=VideoToolbox / NVIDIA=NVDEC）は FFmpeg/PyAV 経由で使えますが、本章のソフトウェアデコード経路は全環境で動きます。実時間性はGPUより「**解像度を下げる・スキップ・スレッド/プロセス分離**」で決まります。
- **vidgear / GStreamer**: `vidgear`（0.3.5）は OpenCV/FFmpeg 上にスレッド化キャプチャや配信を被せたラッパで、再接続・低遅延の定型を肩代わりしてくれます。より低レベルに詰めるなら GStreamer パイプライン（`appsink`）を `cv2.VideoCapture` の backend に使う構成もあります。
- **yt-dlp の運用**: ライブ配信URLの解決（`yt-dlp -g`）はサイト仕様変更で**頻繁に壊れる**ため、バージョン固定のまま放置せず定期更新が前提です。Docker でホストのWebカメラを使うには `--device=/dev/video0` が要ります。
- **本番パイプラインの実例**: Cluster-CLIP の `stream/capture.py`（`put_nowait` による即ドロップ）と `profiler.py`（ステージ別 p50/p99）は、本章の骨格をそのまま実運用に拡張したものです。より体系的なプロファイリングは第34回で深掘りします。
- 参考ドキュメント: OpenCV `VideoCapture` https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html ／ 背景差分チュートリアル https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html ／ PyAV https://pyav.org/docs/stable/ ／ Python `queue`・`multiprocessing` 標準ライブラリ。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ opencv-python-headless 4.13.0.92（`cv2` 4.13.0）／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／（任意）av 17.1.0・yt-dlp は実ライブ接続時のみ