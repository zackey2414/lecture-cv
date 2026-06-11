# 第9回 動画I/Oの基礎 — VideoCapture/VideoWriter・メタデータ・FPS

> トラック: **動画・ストリーム** ／ レベル: **入門** ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません・追加依存グループなし）

## 🎯 この章のゴール

ここまでの章では1枚の静止画を相手にしてきました。本章から扱う「動画」は、その静止画（**フレーム**）を時間方向に大量に並べただけのもの——という単純な事実をまず腹落ちさせます。動画ファイルやWebカメラを **`cv2.VideoCapture`** で開き、`while` ループで1フレームずつ `read()` し、成功フラグ `ret` でループの終わりを判定し、最後に必ず `release()` する。この「開く→ループで読む→ret判定→解放」という**正準パターン**を、何も見ずに書けるようになるのが第一の到達点です。

第二に、動画を扱ううえで欠かせない**メタデータ**と**書き出し**を身につけます。`cap.get(cv2.CAP_PROP_*)` で FPS・総フレーム数・幅・高さ・コーデック（FOURCC）を読み、`cap.set(cv2.CAP_PROP_POS_FRAMES, i)` で任意フレームへシークする。逆に、処理した結果を **`cv2.VideoWriter`** で動画として書き出す際の FOURCC 指定・出力サイズの整合・`isOpened()` 検証という定石を押さえます。そして「動画本来の**ソースFPS**」と「自分のパイプラインが1秒あたり何枚さばけるかの**処理FPS**」がまったくの別物であることを、`time.perf_counter` と `collections.deque` の移動平均で実測して理解します。

到達点を一言でいえば、**サンプル動画もWebカメラもGPUも無い環境で、合成したフレーム列を動画に書き出し、それを読み戻してメタデータを確認し、1フレームずつ処理しながら処理FPSを表示して結果を再書き出しするパイプラインを、最初から最後まで自分の手で書ける**ことです。本章のスクリプトはすべて、サンプル動画すら `numpy`/`cv2` で**その場で合成生成**するので、ネットにもカメラにも依存せずに動きます。

---

## 1. 動画とは何か — 連続するフレーム

動画は魔法ではなく、**等間隔の時刻に撮られた静止画（フレーム）の列**です。1秒間に何枚並ぶかが **FPS（frames per second）**で、30fps の動画なら1秒に30枚の画像が入っています。OpenCV で動画を扱うとは、結局「この画像列を順番に1枚ずつ取り出して、好きに処理し、必要なら別の画像列として書き戻す」ことに尽きます。静止画の知識（BGR配列・cvtColor・resize）がそのまま動画に効くのは、フレームが結局ただの画像だからです。

OpenCV はこの「画像列の入口」を `cv2.VideoCapture`、「出口」を `cv2.VideoWriter` という2つのクラスに集約しています。入口は**動画ファイル・画像シーケンス・Webカメラ・ネットワークストリーム**のどれでも同じ API（`read()`/`release()`）で扱え、出口は FOURCC（コーデック）と FPS とサイズを決めれば `write()` でフレームを足していけます。本章はこの入口と出口、そして両者の間でやり取りされるフレームの正体を、最小のコードで一通り体験します。

本講座はWebカメラもサンプル動画も前提にしません。その代わり、**「左から右へ動く円」と「フレーム番号テキスト」を描いた合成フレーム列**を `cv_helpers.make_demo_video()` で作り、それを動画ファイルに書き出してから読み戻します。フレームに番号を描いておくのがコツで、あとでシーク（指定フレームへ飛ぶ）した結果が正しいかを目で確認できます。まずはこの「合成→書き出し→読み戻し」が一周することを、`cv_helpers.py` 単体実行のスモークテストで確かめてください。

## 2. VideoCapture の正準ループ — isOpened / ret / release

動画読み込みの骨格は、どんな処理でも次の形に収まります。`VideoCapture` で開き、`isOpened()` で開けたか確認し、`while` で `read()` を繰り返し、`ret` が `False` になったら抜け、最後に `release()` する——たったこれだけです。`read()` は `(ret, frame)` のタプルを返し、`ret` は「次のフレームが取れたか」の真偽値、`frame` がその画像（取れなければ `None`）です。下が本章 `01_videocapture_loop.py` の中心で、**この形を丸暗記する価値があります**。

```python
cap = cv2.VideoCapture(source)
if not cap.isOpened():                 # 開けたかを必ず確認（失敗で read し続けると危険）
    raise RuntimeError(f"動画を開けませんでした: {source}")
while True:
    ret, frame = cap.read()
    if not ret:                        # ★ ret 判定がループ終了条件
        break
    # ... ここで frame を処理する ...
cap.release()                          # ★ 使い終わったら必ず解放する
```

このコードで初学者が必ず守るべき点が3つあります。第一に **`isOpened()` チェック**。パスが間違っていたりコーデックが無い場合、`VideoCapture` は例外を投げずに「開けていない」状態になります——`cv2.imread` が失敗時に `None` を返すのと同じ静かな罠で、確認を怠ると無限ループや謎のエラーになります。第二に、**ループ終了は `ret` で判定する**こと。「総フレーム数だけ `for` で回す」書き方は、後述するライブ入力で総数が当てにならないため避け、`ret == False` を唯一の終了条件にします。第三に **`release()` を必ず呼ぶ**こと。これを忘れるとファイルハンドルやカメラデバイスが掴まれたままになります。

## 3. フレームの正体 — BGR numpy 配列と基本操作

`read()` が返す `frame` は、これまでの静止画とまったく同じ **`(高さ, 幅, 3)` の `uint8` BGR numpy 配列**です。つまり静止画でやった操作はすべてフレームにそのまま使えます。本章では前処理の土台として、**グレースケール化（`cvtColor`）・縮小（`resize`）・ROI 切り出し（numpyスライス）**の3点を確認します。特にここで軸順の混乱を一度で片付けておきます——`resize` の `dsize` は **`(幅, 高さ)`** の順、numpy の `shape` と ROI スライスは **`(高さ, 幅)`** の順で、両者は逆並びです。

```python
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # (H,W,3) → (H,W)：チャンネル軸が消える
small = cv2.resize(frame, (w // 2, h // 2),             # dsize=(幅,高さ)。縮小は INTER_AREA が定石
                   interpolation=cv2.INTER_AREA)
roi   = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]   # スライスは [y0:y1, x0:x1]
```

もうひとつ動画でも繰り返しハマるのが **BGR と RGB の取り違え**です。OpenCV のフレームはチャンネル順が **BGR**ですが、matplotlib や Pillow は **RGB** を前提とします。BGR のまま `plt.imshow` に渡すと赤と青が入れ替わり、本章のオレンジの円が青っぽく表示されてしまいます。`01_videocapture_loop.py` はこの違いを `01_bgr_vs_rgb.png` に「変換なし（崩れる）」と「`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`（正しい）」を並べて保存します。matplotlib にフレームを渡すときは **必ず BGR→RGB を挟む**、という癖をここで付けておきましょう。なお `cvtColor(..., COLOR_BGR2GRAY)` の出力は `(H, W)` の2次元になり**チャンネル軸が消える**点も、後段で色を指定する処理のエラー要因になりやすいので覚えておきます。

## 4. メタデータ取得 — CAP_PROP と FOURCC のデコード

開いた `cap` からは `cap.get(プロパティID)` で各種メタデータが読めます。よく使うのは `CAP_PROP_FPS`（FPS）・`CAP_PROP_FRAME_COUNT`（総フレーム数）・`CAP_PROP_FRAME_WIDTH`/`HEIGHT`（幅・高さ）・`CAP_PROP_FOURCC`（コーデック）です。注意点として **`get()` は常に `float` を返す**ので、フレーム数や幅など整数で欲しいものは `int()` で丸めます。FPS と総フレーム数が分かれば「動画の長さ（秒）＝総数 ÷ FPS」も計算できます。下が `02_capprops_seek.py` の取得部です。

```python
fps        = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))         # コーデックは整数で返る
```

`CAP_PROP_FOURCC` だけは少し特殊で、4文字のコーデック名（例: `"mp4v"`）を **4バイトに詰めた32bit整数**として返ってきます。人が読める文字列に戻すには、下位8bitから1バイトずつ取り出して文字へ変換します（本章の `fourcc_to_str` がこれを1関数にしています）。

```python
def fourcc_to_str(code: int) -> str:
    return "".join(chr((int(code) >> (8 * i)) & 0xFF) for i in range(4))
```

ここで面白い実務知識がひとつあります。**要求したFOURCCと、実際に記録されるFOURCCは一致しないことがある**のです。本章では `"mp4v"` で書き出していますが、読み戻して `CAP_PROP_FOURCC` をデコードすると環境によっては `"FMP4"` のように別名が返ります。これはコンテナ/FFmpeg がコーデックタグを正規化するためで、異常ではありません。`02_capprops_seek.py` を実行し、FPS=24.0・総数=60・320×240 といったメタデータと、記録名が要求名と違い得ることを確認してください。

## 5. シーク — POS_FRAMES とライブでの注意

**シーク**とは「次に読むフレームの位置を任意の場所へ飛ばす」操作で、`cap.set(cv2.CAP_PROP_POS_FRAMES, i)` の後に `read()` すると `i` 番目のフレームが取れます。動画の途中だけ処理したい、サムネイルを等間隔で抜きたい、といった場面で必須です。本章はフレームに番号を描いてあるので、`0`・中ほど・終端付近へ飛んで取り出した画像の「円の位置」と「`frame NNN` の文字」が指定インデックスと一致することを `02_seek_grid.png` で目視確認できます。

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, idx)   # 次に読む位置を idx に
ret, frame = cap.read()                 # → idx 番目のフレームが返る（読むと位置は idx+1 へ進む）
```

ただし**シークが効くのは「ファイル」だけ**だと肝に銘じてください。Webカメラやネットワークストリーム（ライブ入力）は「過去のフレーム」を持っていないので、`POS_FRAMES` での巻き戻しは原理的にできません。同じ理由で、ライブでは **`CAP_PROP_FRAME_COUNT` が `0` や負値・巨大値といった当てにならない値になり得ます**。だから「総フレーム数だけ `for` で回す」設計は禁物で、第2節で強調したとおり**終了は `ret` 判定だけに任せる**のが安全です。`02_capprops_seek.py` の `[3]` は、総数を信用せず `ret` だけで回す「ライブ安全」なループを実演します（合成動画なので実際には自然に終わりますが、ライブはこの書き方でないと止まらない、という発想を体に入れます）。

この「ファイルとライブで前提が変わる」という感覚は、次回以降のリアルタイム/ストリーム処理（第11回）でさらに重要になります。本章のうちに、**シークと総フレーム数はファイルの特権であって、ライブでは ret 判定に頼る**という線引きをはっきりさせておきましょう。

## 6. VideoWriter で書き出す — FOURCC・サイズ整合・isOpened 検証

処理した結果を動画として残すのが `cv2.VideoWriter` です。作るときに **出力パス・FOURCC（コーデック）・FPS・出力サイズ `(幅, 高さ)`** の4つを渡します。FOURCC は `cv2.VideoWriter_fourcc(*"mp4v")` のように4文字から作り、**コンテナ拡張子と整合させる**のが鉄則です。headless 版でも安定して使える組み合わせは **`"mp4v"`＋`.mp4`** と **`"MJPG"`＋`.avi`** で、本章はこの2つを順に試します。下が `03_videowriter.py` の書き出し器です。

```python
for ext, cc in ((".mp4", "mp4v"), (".avi", "MJPG")):
    target = path.with_suffix(ext)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*cc), fps, size)
    if writer.isOpened():               # ★ 開けたかを必ず確認（開けないと無言で空ファイル）
        return writer
    writer.release()
return None                             # どの組み合わせもダメなら連番PNGへフォールバック
```

VideoWriter で最も多い失敗が **`isOpened()` を確認しないこと**です。コーデックが環境に無い・拡張子と不整合といった場合、`VideoWriter` は例外を投げずに「開けていない」状態になり、`write()` しても**サイズ0の壊れたファイル**ができるだけです。必ず `isOpened()` で確認し、開けなければ別コーデック、それも無理なら**連番PNG保存にフォールバック**する——本章はこの三段構えで、どんな環境でも必ず結果が残るようにしています（`mp4v`/`MJPG` が両方ダメでも `cv2.imwrite` の連番PNGなら確実に書けるため）。

もうひとつの定番の罠が **書き込むフレームのサイズと VideoWriter の出力サイズの不一致**です。`VideoWriter` は最初に決めた `(幅, 高さ)` 以外のフレームを渡されると、黙って書き込みに失敗します（その1枚が欠落する）。本章のパイプラインは「最初のフレームで出力サイズを確定し、以降は同じサイズにそろえる」ことでこれを防いでいます。`size` が `(幅, 高さ)` 順である点（`frame.shape` の `(高さ, 幅)` と逆）にも注意してください。

## 7. ソースFPS と 処理FPS は別物 — perf_counter + deque

初学者が混同しがちなのが「**ソースFPS**」と「**処理FPS**」です。ソースFPSは `cap.get(cv2.CAP_PROP_FPS)` で得られる**動画本来のフレームレート**（30fpsで撮られた、など素材の属性）。一方の処理FPSは、**自分のプログラムが実際に1秒あたり何フレームさばけているか**の実測値で、CPUの速さや処理の重さで決まります。両者の関係はシンプルで、**処理FPS ≥ ソースFPS なら実時間に追いつける（間に合う）**、逆なら遅延がどんどん溜まる、ということです。

処理FPSは `time.perf_counter()` で各フレームの処理時間 `dt` を測って `1/dt` で求めますが、1枚ごとの値はブレるので **`collections.deque(maxlen=N)` で直近 N 枚の移動平均**をとって滑らかにします。下が `03_videowriter.py` の計測部の核心です。

```python
recent = deque(maxlen=20)          # 直近20枚分の処理時間をためる
prev = time.perf_counter()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    proc = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # 処理
    now = time.perf_counter()
    recent.append(now - prev); prev = now
    avg_dt = sum(recent) / len(recent)
    proc_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0       # ← 移動平均した処理FPS
```

`03_videowriter.py` はこの処理FPSをフレームごとに記録し、ソースFPS（24fps）の水平線と重ねて `03_fps_plot.png` に描きます。合成フレームへの縮小程度なら処理FPSはソースFPSを大きく上回る（このCPUでは数千fps級）ので、青線が赤い破線のはるか上にあり「実時間処理に余裕がある」ことが一目で分かります。重い処理（深層モデルなど）を入れると処理FPSが落ち、いずれソースFPSを下回る——そのとき初めて「縮小する・Nフレームに1回だけ処理する」といった軽量化（第11回の主題）が必要になる、という流れを掴んでください。

そしてこの `03_videowriter.py` こそが**本章の完成物**です。「1フレームずつ読み → 縮小処理し → 処理FPSを移動平均で表示し → Nフレーム間引きで縮小サムネを連番保存しつつ → 結果を VideoWriter で動画へ再書き出し → 最後に読み戻して検証」という一連を、ここまでの全要素（正準ループ・基本操作・メタデータ・FOURCC・FPS計測）を結集して1本にしています。

## 8. headless / Docker での確認 — imshow を使わない

ローカルのデスクトップでは `cv2.imshow(...)` + `cv2.waitKey(1)` で動画をウィンドウ再生するのが手軽です。しかし本講座が使う **opencv-python-headless には `imshow`/`waitKey` がそもそも存在せず**、呼ぶと `cv2.error` になります。Docker・SSH・CI といった GUI の無い環境でも同様で、画面表示に頼った確認方法は使えません。だから本章のスクリプトは**一切 `imshow` を呼ばず**、結果はすべて `outputs/09_video_io_basics/` にファイルとして残します。

headless での確認手段は主に3つです。**(1) `cv2.imwrite` で代表フレームをPNG保存**して目視する、**(2) `cv2.VideoWriter` で結果を動画にまとめて**後で再生する、**(3) matplotlib(Agg) で複数フレームを並べた図**を保存する。本章はこの3つを全部使い分けています（`01` のフレーム保存・`03` の動画再書き出し・`02`/`03` の matplotlib モンタージュ）。matplotlib を使うときは必ず冒頭で `import matplotlib; matplotlib.use("Agg")` とバックエンドを固定し、フレームは BGR→RGB に直してから渡す、という第3節の作法を守ります。

なお opencv-python（GUIあり）と opencv-python-headless は **同じ `cv2` 名前空間を共有するため、同一環境に混在させてはいけません**。ローカルで `imshow` したいなら full 版、Docker/サーバ配布なら headless 版、と**どちらか一方に統一**します。本講座は配布と CI を見据えて headless に統一し、表示の代わりにファイル保存で確認する、という現代的な作法で一貫させています。

## 9. よくあるエラーと対処（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」で一覧にします。動画I/Oの不具合の多くは、この数個の原因に集約されます。実装中に詰まったら、まずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `read()` が最初から `ret=False` | パス間違い・コーデック無しで開けていない | `cap.isOpened()` を必ず確認。パス/拡張子を見直す |
| ループが終わらない / クラッシュ | 総フレーム数で `for` 回した、`ret` を見ていない | 終了は `if not ret: break` だけに任せる |
| 書き出した動画がサイズ0/壊れる | `VideoWriter` が開けていない（FOURCC不整合） | `writer.isOpened()` を確認。`mp4v/.mp4` か `MJPG/.avi` に。ダメなら連番PNG |
| 動画にフレームが入らない | 書くフレームのサイズが出力サイズと違う | 出力 `(W,H)` を固定し、全フレームを同サイズに `resize` |
| matplotlib で色が変（赤青反転） | BGR のまま渡した | `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` してから `imshow` |
| `cv2.error` で imshow が落ちる | headless 版に `imshow` は無い | `imwrite`/`VideoWriter`/matplotlib(Agg) で保存して確認 |
| 総フレーム数が 0 や変な値 | 入力がライブ（カメラ/RTSP）でファイルでない | 総数を信用せず `ret` 判定で回す。シークも諦める |
| FOURCC が要求と違う文字で返る | コンテナがコーデックタグを正規化した | 異常ではない。`mp4v`→`FMP4` 等はそのまま受け入れる |
| `resize` の結果が縦横おかしい | `dsize` を `(高さ, 幅)` で渡した | `dsize` は `(幅, 高さ)`。`shape` の逆順に注意 |
| `cap.get` の値が小数で扱いにくい | `get()` は常に `float` を返す | 整数が要るものは `int(...)` で丸める |

この表の項目が、本章で時間を取られる原因のほぼ全てです。逆にこれらを自分で説明でき・回避コードを書けるようになれば、この章のゴールに到達しています。

## 動かし方

すべて CPU・ネット非依存・カメラ不要・追加依存なしで動きます（サンプル動画は各スクリプトが `numpy`/`cv2` で合成生成します）。リポジトリのルートで以下を順に実行してください。結果はすべて `outputs/09_video_io_basics/` に画像・動画として保存され、画面表示はしません（headless 安全）。

```bash
# 1) VideoCapture の正準ループ（isOpened/read/ret/release）とフレーム基本操作
uv run python lectures/09_video_io_basics/01_videocapture_loop.py

# 2) メタデータ取得（CAP_PROP・FOURCC デコード）と POS_FRAMES シーク
uv run python lectures/09_video_io_basics/02_capprops_seek.py

# 3) VideoWriter 書き出し + 処理FPS計測（この回の完成物）
uv run python lectures/09_video_io_basics/03_videowriter.py

# 演習（TODO を実装 → 自己採点。未実装でも FAIL 表示で正常終了する）
uv run python lectures/09_video_io_basics/exercises.py
# 行き詰まったら模範解答で挙動を確認（まずは自力で！）
SHOW_SOLUTION=1 uv run python lectures/09_video_io_basics/exercises.py
```

実行後は `outputs/09_video_io_basics/` の成果物を順に開いて、本文の確認ポイントと照らし合わせてください。特に `01_bgr_vs_rgb.png`（BGR/RGB の崩れ）、`02_seek_grid.png`（シーク先の円の位置とフレーム番号の一致）、`03_fps_plot.png`（処理FPS が ソースFPS を上回る様子）、`03_processed.mp4`（再書き出しした縮小動画）を見ると、各節の内容が一気に腑に落ちます。`cv_helpers.py` を単体で実行すると、合成動画の「書き出し→読み戻し」が一周するスモークテストになります。

## まとめ

この章では、動画＝連続フレームという捉え方を起点に、`cv2.VideoCapture` で開いて `while`＋`ret` 判定でループし `release()` する正準パターン、`read()` が返す BGR numpy フレームへの基本操作（cvtColor/resize/ROI と BGR→RGB 変換）、`cap.get(CAP_PROP_*)` でのメタデータ取得と FOURCC のデコード、`POS_FRAMES` でのシーク（とライブでの注意）、`cv2.VideoWriter` での書き出し（FOURCC・サイズ整合・`isOpened()` 検証・連番PNGフォールバック）、そして `time.perf_counter`＋`deque` で測る処理FPS と ソースFPS の違いまでを、すべて「自分で再現し説明できる」レベルで一通り組み立てました。

ここで身につけた「開く→retでループ→処理→書き出す→解放」という流れと、「ファイルとライブで前提が変わる／処理FPSとソースFPSは別物」という勘所は、次回（第10回 古典的動画処理：背景差分・オプティカルフロー）と第11回（リアルタイム/ストリーム処理）の土台にそのまま効いてきます。まずは演習を自力で全問 PASS させ、`isOpened()` チェック・`ret` 判定ループ・FOURCC デコード・処理FPS の移動平均という定石を手に馴染ませてから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06-11 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ opencv-python-headless 4.13.0.92（`cv2` 4.13.0、VideoCapture/VideoWriter は FFmpeg 同梱・本体機能で contrib 不要）／ Pillow 12.2.0 ／ matplotlib 3.10.x（Agg バックエンドで画面非依存に保存）