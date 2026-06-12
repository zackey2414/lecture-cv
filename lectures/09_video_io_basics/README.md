# 第9回 動画I/Oの基礎 — VideoCapture/VideoWriter・メタデータ・FPS

> トラック: **動画・ストリーム** ／ レベル: **入門** ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません・追加依存グループなし）

## 🎯 この章のゴール

ここまでの章では、1枚の静止画を相手にしてきました。これに対して本章から扱う「動画」は、その静止画（**フレーム**）を時間方向に大量に並べただけのもの——という単純な事実を、まず腹落ちさせることから始めます。具体的には、動画ファイルやWebカメラを **`cv2.VideoCapture`** で開き、`while` ループで1フレームずつ `read()` し、成功フラグ `ret` でループの終わりを判定し、最後に必ず `release()` する。この「開く→ループで読む→ret判定→解放」という**正準パターン**を、何も見ずに書けるようになることが、第一の到達点です。

第二に、動画を扱ううえで欠かせない**メタデータ**と**書き出し**を身につけます。まず読み取り側では、`cap.get(cv2.CAP_PROP_*)` で FPS・総フレーム数・幅・高さ・コーデック（FOURCC）を読み、`cap.set(cv2.CAP_PROP_POS_FRAMES, i)` で任意フレームへシークします。一方の書き出し側では、処理した結果を **`cv2.VideoWriter`** で動画にする際の FOURCC 指定・出力サイズの整合・`isOpened()` 検証という定石を押さえます。さらに、「動画本来の**ソースFPS**」と「自分のパイプラインが1秒あたり何枚さばけるかの**処理FPS**」がまったくの別物であることを、`time.perf_counter` と `collections.deque` の移動平均で実測しながら理解します。

これらの到達点を一言でまとめれば、**サンプル動画もWebカメラもGPUも無い環境で、合成したフレーム列を動画に書き出し、それを読み戻してメタデータを確認し、1フレームずつ処理しながら処理FPSを表示して結果を再書き出しするパイプラインを、最初から最後まで自分の手で書ける**ことです。なお本章のスクリプトはすべて、サンプル動画すら `numpy`/`cv2` で**その場で合成生成**するため、ネットにもカメラにも依存せずに動きます。

---

## 1. 動画とは何か — 連続するフレーム

動画は魔法ではなく、**等間隔の時刻に撮られた静止画（フレーム）の列**にすぎません。その1秒間に何枚並ぶかを表すのが **FPS（frames per second）**で、たとえば 30fps の動画なら1秒に30枚の画像が入っています。したがって、OpenCV で動画を扱うとは、結局「この画像列を順番に1枚ずつ取り出して、好きに処理し、必要なら別の画像列として書き戻す」ことに尽きます。静止画の知識（BGR配列・cvtColor・resize）がそのまま動画にも効くのは、フレームが結局はただの画像だからです。

<figure class="lec-fig"><svg viewBox="0 0 660 220" role="img" aria-label="動画は静止画フレームを時間方向に並べたもの。1秒間にFPS枚が並び、各フレームはHW3のBGR配列" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">1 秒間に FPS 枚 並ぶ（例: 30fps → 30 枚）</text><rect x="60" y="46" width="88" height="90" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="178" y="46" width="88" height="90" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="296" y="46" width="88" height="90" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="414" y="46" width="88" height="90" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="532" y="46" width="88" height="90" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><text x="104" y="66" text-anchor="middle" font-size="12" fill="#71717a">frame 0</text><text x="222" y="66" text-anchor="middle" font-size="12" fill="#71717a">1</text><text x="340" y="66" text-anchor="middle" font-size="12" fill="#71717a">2</text><text x="458" y="66" text-anchor="middle" font-size="12" fill="#71717a">3</text><text x="576" y="66" text-anchor="middle" font-size="12" fill="#71717a">…</text><circle cx="78" cy="110" r="9" fill="#f97316" stroke="#c2410c" stroke-width="1.5"/><circle cx="209" cy="110" r="9" fill="#f97316" stroke="#c2410c" stroke-width="1.5"/><circle cx="340" cy="110" r="9" fill="#f97316" stroke="#c2410c" stroke-width="1.5"/><circle cx="471" cy="110" r="9" fill="#f97316" stroke="#c2410c" stroke-width="1.5"/><circle cx="602" cy="110" r="9" fill="#f97316" stroke="#c2410c" stroke-width="1.5"/><line x1="50" y1="168" x2="600" y2="168" stroke="#71717a" stroke-width="2"/><polygon points="612,168 600,162 600,174" fill="#71717a"/><text x="330" y="196" text-anchor="middle" font-size="14" fill="#3f3f46">時間 t（フレーム番号）→</text></svg><figcaption>動画は<b>静止画（フレーム）を時間方向に並べたもの</b>です。1 秒間に並ぶ枚数が <b>FPS</b>（frames per second）で、30fps なら 1 秒 ＝ 30 枚。各フレームの正体は静止画と同じ <code>(H, W, 3)</code> の <b>uint8 BGR</b> 配列なので、静止画の操作（cvtColor / resize / スライス）がそのまま使えます。</figcaption></figure>

OpenCV は、この「画像列の入口」を `cv2.VideoCapture`、「出口」を `cv2.VideoWriter` という2つのクラスに集約しています。入口のほうは、**動画ファイル・画像シーケンス・Webカメラ・ネットワークストリーム**のどれであっても同じ API（`read()`/`release()`）で扱えます。出口のほうは、FOURCC（コーデック）と FPS とサイズさえ決めれば、あとは `write()` でフレームを足していくだけです。本章では、この入口と出口、そして両者の間でやり取りされるフレームの正体を、最小のコードで一通り体験します。

なお本講座は、Webカメラもサンプル動画も前提にしません。その代わりに、**「左から右へ動く円」と「フレーム番号テキスト」を描いた合成フレーム列**を `cv_helpers.make_demo_video()` で作り、それを動画ファイルに書き出してから読み戻します。ここでフレームに番号を描いておくのがコツで、こうしておけば、あとでシーク（指定フレームへ飛ぶ）した結果が正しいかを目で確認できます。まずはこの「合成→書き出し→読み戻し」が一周することを、`cv_helpers.py` 単体実行のスモークテストで確かめてください。

## 2. VideoCapture の正準ループ — isOpened / ret / release

動画読み込みの骨格は、どんな処理であっても次の形に収まります。すなわち、`VideoCapture` で開き、`isOpened()` で開けたかを確認し、`while` で `read()` を繰り返し、`ret` が `False` になったら抜け、最後に `release()` する——たったこれだけです。この `read()` は `(ret, frame)` のタプルを返し、`ret` は「次のフレームが取れたか」を表す真偽値、`frame` がその画像（取れなければ `None`）になります。下が本章 `01_videocapture_loop.py` の中心で、**この形は丸暗記する価値があります**。

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

このコードには、初学者が必ず守るべき点が3つあります。第一に、**`isOpened()` チェック**です。パスが間違っていたりコーデックが無かったりする場合、`VideoCapture` は例外を投げずに、ただ「開けていない」状態になります——これは `cv2.imread` が失敗時に `None` を返すのと同じ静かな罠で、確認を怠ると無限ループや謎のエラーにつながります。第二に、**ループ終了は `ret` で判定する**ことです。「総フレーム数だけ `for` で回す」書き方は、後述するライブ入力では総数が当てにならないため避け、`ret == False` を唯一の終了条件にします。第三に、**`release()` を必ず呼ぶ**ことです。これを忘れると、ファイルハンドルやカメラデバイスが掴まれたままになってしまいます。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="VideoCaptureの正準ループ。openしisOpenedを確認しwhileでreadしret判定で処理かreleaseへ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="14" y="80" width="120" height="46" rx="6" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/><text x="74" y="108" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">VideoCapture()</text><rect x="156" y="80" width="104" height="46" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><text x="208" y="108" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">isOpened() ?</text><rect x="282" y="80" width="114" height="46" rx="6" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/><text x="339" y="108" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">read()</text><rect x="414" y="80" width="92" height="46" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><text x="460" y="108" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">not ret ?</text><rect x="520" y="80" width="104" height="46" rx="6" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/><text x="572" y="108" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">frame を処理</text><line x1="134" y1="103" x2="150" y2="103" stroke="#71717a" stroke-width="2"/><polygon points="156,103 146,98 146,108" fill="#71717a"/><line x1="260" y1="103" x2="276" y2="103" stroke="#71717a" stroke-width="2"/><polygon points="282,103 272,98 272,108" fill="#71717a"/><line x1="396" y1="103" x2="408" y2="103" stroke="#71717a" stroke-width="2"/><polygon points="414,103 404,98 404,108" fill="#71717a"/><line x1="506" y1="103" x2="514" y2="103" stroke="#71717a" stroke-width="2"/><polygon points="520,103 510,98 510,108" fill="#71717a"/><polyline points="572,80 572,42 339,42 339,82" fill="none" stroke="#16a34a" stroke-width="2"/><polygon points="339,82 333,72 345,72" fill="#16a34a"/><text x="455" y="36" text-anchor="middle" font-size="12" fill="#15803d">ret=True：read() へ戻る</text><line x1="208" y1="126" x2="208" y2="174" stroke="#71717a" stroke-width="2"/><polygon points="208,178 202,168 214,168" fill="#71717a"/><text x="226" y="152" font-size="11" fill="#dc2626">False</text><rect x="138" y="180" width="144" height="44" rx="6" fill="#dc2626"/><text x="210" y="200" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">RuntimeError</text><text x="210" y="216" text-anchor="middle" font-size="10" fill="#ffffff">（開けない）</text><line x1="460" y1="126" x2="460" y2="174" stroke="#71717a" stroke-width="2"/><polygon points="460,178 454,168 466,168" fill="#71717a"/><text x="476" y="152" font-size="11" fill="#3f3f46">ret=False</text><rect x="394" y="180" width="158" height="44" rx="6" fill="#f4f4f5" stroke="#16a34a" stroke-width="1.5"/><text x="473" y="207" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">release() で解放</text></svg><figcaption><b>VideoCapture の正準ループ</b>です。<code>VideoCapture()</code> で開いたら必ず <b>isOpened()</b> を確認し（開けなければ即 <code>raise</code>）、<code>while</code> で <code>read()</code> を回し <b>ret が True の間だけ</b>フレームを処理します。<b>ret が False になったら抜けて release()</b>。終了条件を総フレーム数ではなく <b>ret</b> にするのが、ライブ入力でも安全な定石です。</figcaption></figure>

## 3. フレームの正体 — BGR numpy 配列と基本操作

`read()` が返す `frame` は、これまでの静止画とまったく同じ **`(高さ, 幅, 3)` の `uint8` BGR numpy 配列**です。つまり、静止画でやった操作はすべて、そのままフレームにも使えます。そこで本章では、前処理の土台として **グレースケール化（`cvtColor`）・縮小（`resize`）・ROI 切り出し（numpyスライス）**の3点を確認します。あわせてここで、軸順の混乱も一度で片付けておきましょう——`resize` の `dsize` は **`(幅, 高さ)`** の順なのに対し、numpy の `shape` と ROI スライスは **`(高さ, 幅)`** の順で、両者はちょうど逆並びです。

```python
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # (H,W,3) → (H,W)：チャンネル軸が消える
small = cv2.resize(frame, (w // 2, h // 2),             # dsize=(幅,高さ)。縮小は INTER_AREA が定石
                   interpolation=cv2.INTER_AREA)
roi   = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]   # スライスは [y0:y1, x0:x1]
```

動画でもうひとつ繰り返しハマるのが、**BGR と RGB の取り違え**です。OpenCV のフレームはチャンネル順が **BGR** であるのに対し、matplotlib や Pillow は **RGB** を前提とします。そのため、BGR のまま `plt.imshow` に渡すと赤と青が入れ替わり、本章のオレンジの円が青っぽく表示されてしまいます。`01_videocapture_loop.py` はこの違いを、`01_bgr_vs_rgb.png` に「変換なし（崩れる）」と「`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`（正しい）」を並べて保存します。matplotlib にフレームを渡すときは **必ず BGR→RGB を挟む**、という癖をここで付けておきましょう。なお、`cvtColor(..., COLOR_BGR2GRAY)` の出力は `(H, W)` の2次元になり**チャンネル軸が消える**点も、後段で色を指定する処理のエラー要因になりやすいので、あわせて覚えておきます。

## 4. メタデータ取得 — CAP_PROP と FOURCC のデコード

開いた `cap` からは、`cap.get(プロパティID)` で各種メタデータが読めます。よく使うのは `CAP_PROP_FPS`（FPS）・`CAP_PROP_FRAME_COUNT`（総フレーム数）・`CAP_PROP_FRAME_WIDTH`/`HEIGHT`（幅・高さ）・`CAP_PROP_FOURCC`（コーデック）です。ただし注意点として、**`get()` は常に `float` を返す**ため、フレーム数や幅など整数で欲しいものは `int()` で丸めます。なお、FPS と総フレーム数さえ分かれば「動画の長さ（秒）＝総数 ÷ FPS」も計算できます。下が `02_capprops_seek.py` の取得部です。

```python
fps        = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))         # コーデックは整数で返る
```

このうち `CAP_PROP_FOURCC` だけは少し特殊で、4文字のコーデック名（例: `"mp4v"`）を **4バイトに詰めた32bit整数**として返してきます。これを人が読める文字列に戻すには、下位8bitから1バイトずつ取り出して文字へ変換します（本章の `fourcc_to_str` が、この処理を1関数にまとめています）。

```python
def fourcc_to_str(code: int) -> str:
    return "".join(chr((int(code) >> (8 * i)) & 0xFF) for i in range(4))
```

<figure class="lec-fig"><svg viewBox="0 0 640 250" role="img" aria-label="FOURCCの32bit整数を下位バイトから1バイトずつ文字に直すとmp4vになる。バイト並びは反転して読む" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#3f3f46">32bit 整数 ＝ 4 バイト（上位 → 下位）</text><rect x="139" y="50" width="86" height="46" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><rect x="231" y="50" width="86" height="46" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><rect x="323" y="50" width="86" height="46" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><rect x="415" y="50" width="86" height="46" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="182" y="82" text-anchor="middle" font-size="22" font-weight="700" fill="#1d4ed8">v</text><text x="274" y="82" text-anchor="middle" font-size="22" font-weight="700" fill="#1d4ed8">4</text><text x="366" y="82" text-anchor="middle" font-size="22" font-weight="700" fill="#1d4ed8">p</text><text x="458" y="82" text-anchor="middle" font-size="22" font-weight="700" fill="#1d4ed8">m</text><text x="458" y="114" text-anchor="middle" font-size="10.5" fill="#c2410c">下位 (i=0)</text><line x1="458" y1="96" x2="182" y2="160" stroke="#ea580c" stroke-width="2"/><polygon points="182,162 177,152 187,152" fill="#ea580c"/><line x1="366" y1="96" x2="274" y2="160" stroke="#71717a" stroke-width="1.6"/><polygon points="274,162 269,152 279,152" fill="#71717a"/><line x1="274" y1="96" x2="366" y2="160" stroke="#71717a" stroke-width="1.6"/><polygon points="366,162 361,152 371,152" fill="#71717a"/><line x1="182" y1="96" x2="458" y2="160" stroke="#71717a" stroke-width="1.6"/><polygon points="458,162 453,152 463,152" fill="#71717a"/><rect x="139" y="164" width="86" height="46" rx="4" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/><rect x="231" y="164" width="86" height="46" rx="4" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/><rect x="323" y="164" width="86" height="46" rx="4" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/><rect x="415" y="164" width="86" height="46" rx="4" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/><text x="182" y="196" text-anchor="middle" font-size="22" font-weight="700" fill="#c2410c">m</text><text x="274" y="196" text-anchor="middle" font-size="22" font-weight="700" fill="#c2410c">p</text><text x="366" y="196" text-anchor="middle" font-size="22" font-weight="700" fill="#c2410c">4</text><text x="458" y="196" text-anchor="middle" font-size="22" font-weight="700" fill="#c2410c">v</text><text x="320" y="238" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">fourcc_to_str() ＝ "mp4v"</text></svg><figcaption>FOURCC（コーデック名）は4文字を<b>32bit 整数に詰めた</b>値として <code>CAP_PROP_FOURCC</code> から返ります。<code>fourcc_to_str</code> は <code>(code &gt;&gt; 8·i) &amp; 0xFF</code> で<b>下位バイト（i=0）から1バイトずつ</b>取り出して文字に直すため、整数のバイト並び（上位→下位）を<b>反転した順</b>に読むと <code>"mp4v"</code> が復元されます。要求名と記録名がずれる（<code>mp4v</code>→<code>FMP4</code>）こともあります。</figcaption></figure>

ここで、面白い実務知識をひとつ挙げておきます。実は、**要求したFOURCCと、実際に記録されるFOURCCは一致しないことがある**のです。たとえば本章では `"mp4v"` で書き出していますが、読み戻して `CAP_PROP_FOURCC` をデコードすると、環境によっては `"FMP4"` のように別名が返ります。これはコンテナ/FFmpeg がコーデックタグを正規化するためであり、異常ではありません。`02_capprops_seek.py` を実行して、FPS=24.0・総数=60・320×240 といったメタデータと、記録名が要求名と違い得ることを確認してください。

## 5. シーク — POS_FRAMES とライブでの注意

**シーク**とは、「次に読むフレームの位置を任意の場所へ飛ばす」操作です。具体的には、`cap.set(cv2.CAP_PROP_POS_FRAMES, i)` の後に `read()` すると `i` 番目のフレームが取れます。これは、動画の途中だけ処理したい、サムネイルを等間隔で抜きたい、といった場面で必須になります。本章ではフレームに番号を描いてあるので、`0`・中ほど・終端付近へ飛んで取り出した画像の「円の位置」と「`frame NNN` の文字」が指定インデックスと一致することを、`02_seek_grid.png` で目視確認できます。

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, idx)   # 次に読む位置を idx に
ret, frame = cap.read()                 # → idx 番目のフレームが返る（読むと位置は idx+1 へ進む）
```

ただし、**シークが効くのは「ファイル」だけ**だと肝に銘じてください。Webカメラやネットワークストリーム（ライブ入力）は「過去のフレーム」を持っていないため、`POS_FRAMES` での巻き戻しは原理的にできません。そして同じ理由で、ライブでは **`CAP_PROP_FRAME_COUNT` が `0` や負値・巨大値といった当てにならない値になり得ます**。したがって「総フレーム数だけ `for` で回す」設計は禁物であり、第2節で強調したとおり、**終了は `ret` 判定だけに任せる**のが安全です。`02_capprops_seek.py` の `[3]` は、総数を信用せず `ret` だけで回す「ライブ安全」なループを実演します（合成動画なので実際には自然に終わりますが、ライブはこの書き方でないと止まらない、という発想を体に入れておきます）。

この「ファイルとライブで前提が変わる」という感覚は、次回以降のリアルタイム/ストリーム処理（第11回）でさらに重要になります。だからこそ本章のうちに、**シークと総フレーム数はファイルの特権であって、ライブでは ret 判定に頼る**という線引きを、はっきりさせておきましょう。

<figure class="lec-fig"><svg viewBox="0 0 640 240" role="img" aria-label="シークはファイルの特権。ファイルは任意位置へ前後に飛べ総数も正確、ライブは前進のみで巻き戻し不可" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="14" y="46" width="84" height="44" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><text x="56" y="73" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">ファイル</text><rect x="120" y="50" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="171" y="50" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="222" y="50" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="273" y="50" width="46" height="36" fill="#f97316" stroke="#c2410c" stroke-width="1.5"/><rect x="324" y="50" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="375" y="50" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="426" y="50" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><text x="296" y="72" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">idx</text><polyline points="143,50 143,30 449,30 449,50" fill="none" stroke="#c2410c" stroke-width="1.8"/><polygon points="143,50 137,40 149,40" fill="#c2410c"/><polygon points="449,50 443,40 455,40" fill="#c2410c"/><text x="296" y="24" text-anchor="middle" font-size="12" fill="#c2410c">POS_FRAMES で任意位置へ（前後どちらも）</text><text x="296" y="104" text-anchor="middle" font-size="11.5" fill="#71717a">総フレーム数も正確（0 … N）</text><rect x="14" y="150" width="84" height="44" rx="6" fill="#fafafa" stroke="#dc2626" stroke-width="1.5"/><text x="56" y="170" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">ライブ</text><text x="56" y="186" text-anchor="middle" font-size="9.5" fill="#71717a">カメラ / RTSP</text><text x="120" y="180" text-anchor="middle" font-size="22" font-weight="700" fill="#dc2626">✕</text><text x="120" y="201" text-anchor="middle" font-size="10.5" fill="#dc2626">戻れない</text><rect x="158" y="156" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="209" y="156" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="260" y="156" width="46" height="36" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.2"/><rect x="311" y="156" width="46" height="36" fill="#2563eb" stroke="#1d4ed8" stroke-width="1.5"/><text x="334" y="179" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">今</text><line x1="365" y1="174" x2="470" y2="174" stroke="#2563eb" stroke-width="2.5"/><polygon points="482,174 470,167 470,181" fill="#2563eb"/><text x="424" y="150" text-anchor="middle" font-size="11" fill="#2563eb">前進のみ →（read）</text><text x="300" y="224" text-anchor="middle" font-size="12" fill="#71717a">総数は 0 / 不定 → ret 判定で終了</text></svg><figcaption><b>シークはファイルの特権</b>です。ファイルは全フレームが手元にあるので <code>cap.set(CAP_PROP_POS_FRAMES, idx)</code> で<b>前にも後ろにも任意位置へ</b>飛べ、総フレーム数も正確に分かります。一方 <b>ライブ（カメラ / RTSP）</b>は過去フレームを持たず <b>前進のみ・巻き戻し不可</b>で、総数も <code>0</code> や不定値になり得ます。だからライブでは総数に頼らず <b>ret 判定だけで終了</b>します。</figcaption></figure>

## 6. VideoWriter で書き出す — FOURCC・サイズ整合・isOpened 検証

処理した結果を動画として残す役割を担うのが、`cv2.VideoWriter` です。作るときには、**出力パス・FOURCC（コーデック）・FPS・出力サイズ `(幅, 高さ)`** の4つを渡します。このうち FOURCC は `cv2.VideoWriter_fourcc(*"mp4v")` のように4文字から作り、**コンテナ拡張子と整合させる**のが鉄則です。headless 版でも安定して使える組み合わせは **`"mp4v"`＋`.mp4`** と **`"MJPG"`＋`.avi`** で、本章ではこの2つを順に試します。下が `03_videowriter.py` の書き出し器です。

```python
for ext, cc in ((".mp4", "mp4v"), (".avi", "MJPG")):
    target = path.with_suffix(ext)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*cc), fps, size)
    if writer.isOpened():               # ★ 開けたかを必ず確認（開けないと無言で空ファイル）
        return writer
    writer.release()
return None                             # どの組み合わせもダメなら連番PNGへフォールバック
```

VideoWriter で最も多い失敗は、**`isOpened()` を確認しないこと**です。コーデックが環境に無い、あるいは拡張子と不整合といった場合、`VideoWriter` は例外を投げずに「開けていない」状態になり、そのまま `write()` しても**サイズ0の壊れたファイル**ができるだけです。そこで必ず `isOpened()` で確認し、開けなければ別コーデック、それも無理なら**連番PNG保存にフォールバック**する——本章はこの三段構えによって、どんな環境でも必ず結果が残るようにしています（`mp4v`/`MJPG` が両方ダメでも、`cv2.imwrite` の連番PNGなら確実に書けるためです）。

もうひとつの定番の罠は、**書き込むフレームのサイズと VideoWriter の出力サイズの不一致**です。`VideoWriter` は、最初に決めた `(幅, 高さ)` 以外のフレームを渡されると、黙って書き込みに失敗します（その1枚が欠落します）。本章のパイプラインは、「最初のフレームで出力サイズを確定し、以降は同じサイズにそろえる」ことで、これを防いでいます。なお、`size` が `(幅, 高さ)` 順である点（`frame.shape` の `(高さ, 幅)` とは逆）にも注意してください。

## 7. ソースFPS と 処理FPS は別物 — perf_counter + deque

初学者が混同しがちなのが、「**ソースFPS**」と「**処理FPS**」の違いです。ソースFPSは、`cap.get(cv2.CAP_PROP_FPS)` で得られる**動画本来のフレームレート**（30fpsで撮られた、などの素材の属性）です。一方の処理FPSは、**自分のプログラムが実際に1秒あたり何フレームさばけているか**の実測値であり、CPUの速さや処理の重さで決まります。両者の関係はシンプルで、**処理FPS ≥ ソースFPS なら実時間に追いつける（間に合う）**、逆なら遅延がどんどん溜まる、ということです。

<figure class="lec-fig"><svg viewBox="0 0 640 230" role="img" aria-label="ソースFPSと処理FPSの違い。1フレームの予算は1割るソースFPSで、予算内なら間に合い超過なら遅延が溜まる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="410" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">1フレームの予算 ＝ 1 / ソースFPS</text><rect x="80" y="70" width="260" height="34" rx="4" fill="#16a34a"/><text x="210" y="92" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">速い処理（1枚）</text><rect x="80" y="112" width="470" height="34" rx="4" fill="#dc2626"/><text x="230" y="134" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">重い処理（1枚）</text><line x1="410" y1="44" x2="410" y2="152" stroke="#c2410c" stroke-width="2" stroke-dasharray="5 4"/><text x="240" y="60" text-anchor="middle" font-size="11" fill="#2563eb">予算内</text><text x="510" y="60" text-anchor="middle" font-size="11" fill="#c2410c">予算超過</text><text x="80" y="180" font-size="12.5" fill="#15803d">速い → 処理FPS ≥ ソースFPS → 間に合う</text><text x="80" y="206" font-size="12.5" fill="#dc2626">重い → 処理FPS ＜ ソースFPS → 遅延が溜まる</text></svg><figcaption><b>ソースFPS</b>（素材が 1 秒に何枚か）と <b>処理FPS</b>（自分のプログラムが 1 秒に何枚さばけるか）は別物です。1 フレームに使える時間の<b>予算は 1 / ソースFPS</b>。処理がこの予算内に収まれば <b>処理FPS ≥ ソースFPS</b> で実時間に追いつき（間に合う）、予算を超えると <b>処理FPS &lt; ソースFPS</b> となって遅延が溜まります。重い処理を入れて初めて、縮小や N フレーム間引きが要るかを見極めます。</figcaption></figure>

処理FPSは、`time.perf_counter()` で各フレームの処理時間 `dt` を測り、`1/dt` で求めます。ただし1枚ごとの値はブレるので、**`collections.deque(maxlen=N)` で直近 N 枚の移動平均**をとって滑らかにします。下が `03_videowriter.py` の計測部の核心です。

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

`03_videowriter.py` は、この処理FPSをフレームごとに記録し、ソースFPS（24fps）の水平線と重ねて `03_fps_plot.png` に描きます。合成フレームへの縮小程度であれば処理FPSはソースFPSを大きく上回る（このCPUでは数千fps級）ので、青線が赤い破線のはるか上にあり、「実時間処理に余裕がある」ことが一目で分かります。逆に、重い処理（深層モデルなど）を入れると処理FPSが落ち、いずれソースFPSを下回ります——そのとき初めて「縮小する・Nフレームに1回だけ処理する」といった軽量化（第11回の主題）が必要になる、という流れを掴んでください。

そして、この `03_videowriter.py` こそが**本章の完成物**です。「1フレームずつ読み → 縮小処理し → 処理FPSを移動平均で表示し → Nフレーム間引きで縮小サムネを連番保存しつつ → 結果を VideoWriter で動画へ再書き出し → 最後に読み戻して検証」という一連の流れを、ここまでの全要素（正準ループ・基本操作・メタデータ・FOURCC・FPS計測）を結集して、1本にまとめています。

## 8. headless / Docker での確認 — imshow を使わない

ローカルのデスクトップであれば、`cv2.imshow(...)` + `cv2.waitKey(1)` で動画をウィンドウ再生するのが手軽です。しかし、本講座が使う **opencv-python-headless には `imshow`/`waitKey` がそもそも存在せず**、呼ぶと `cv2.error` になります。これは Docker・SSH・CI といった GUI の無い環境でも同様で、画面表示に頼った確認方法は使えません。だからこそ本章のスクリプトは**一切 `imshow` を呼ばず**、結果はすべて `outputs/09_video_io_basics/` にファイルとして残します。

headless での確認手段は、主に3つあります。**(1) `cv2.imwrite` で代表フレームをPNG保存**して目視する、**(2) `cv2.VideoWriter` で結果を動画にまとめて**後で再生する、**(3) matplotlib(Agg) で複数フレームを並べた図**を保存する、の3つです。本章では、この3つを全部使い分けています（`01` のフレーム保存・`03` の動画再書き出し・`02`/`03` の matplotlib モンタージュ）。なお matplotlib を使うときは、必ず冒頭で `import matplotlib; matplotlib.use("Agg")` とバックエンドを固定し、フレームは BGR→RGB に直してから渡す、という第3節の作法を守ります。

なお、opencv-python（GUIあり）と opencv-python-headless は、**同じ `cv2` 名前空間を共有するため、同一環境に混在させてはいけません**。したがって、ローカルで `imshow` したいなら full 版、Docker/サーバ配布なら headless 版、と**どちらか一方に統一**します。本講座は配布と CI を見据えて headless に統一し、表示の代わりにファイル保存で確認する、という現代的な作法で一貫させています。

## 9. よくあるエラーと対処（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」の形で一覧にまとめます。動画I/Oの不具合の多くは、ここに挙げた数個の原因に集約されます。実装中に詰まったら、まずはここを見てください。

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

この表の項目が、本章で時間を取られる原因のほぼ全てです。逆にいえば、これらを自分で説明でき、回避コードまで書けるようになれば、この章のゴールに到達したことになります。

---

## 🛠 章末ミニプロジェクト — 「動体ハイライト」動画パイプライン

ここまでの全要素を **1本のパイプライン**に束ねる総合課題が、`mini_project.py` です。そのテーマは、「合成動画を読み込み、**連続フレーム差分**で動いた画素を見つけて赤くハイライトし、結果を動画に再書き出しして、計測値をレポートにまとめる」ことです。いわば、第10回（背景差分・オプティカルフロー）の最も素朴な前段——**フレーム差分による動体検出**——を、本章の I/O 技能だけで自力実装します。なお `mini_project.py` は、他のスクリプト（`01_`〜`03_`）を一切 import せず、**1ファイルで自己完結**しています（出力先の解決だけ `cv_helpers.output_dir` を借用します）。

このミニプロジェクトが踏む流れには、本章の各節がそのまま順番に効いてきます。

1. **合成（素材づくり）**: 「左→右に動くオレンジの円」と「右→左に動く緑の四角」＋フレーム番号テキストを `numpy`/`cv2` で 72 枚生成し、`VideoWriter`（`mp4v`→`MJPG`→連番PNG のフォールバック）でソース動画に書き出す（第1・6節）。
2. **メタデータ確認**: 書いた動画を開き直し、`cap.get(CAP_PROP_*)` で FPS・総数・幅高さ・FOURCC を取り、FOURCC を4文字へデコードして「長さ（秒）＝総数÷FPS」まで出す（第4節）。要求 `mp4v` がコンテナ側で `FMP4` 等に正規化される実例も観察できます。
3. **本体処理**: 正準ループ（`isOpened`→`while read`→`ret` 判定→`release`）で1フレームずつ読み、`cvtColor` でグレー化、**1つ前のフレームとの `cv2.absdiff` が閾値超え**の画素を動体マスクとして赤く塗り、`VideoWriter` で再書き出し。同時に `perf_counter`＋`deque` で**処理FPSの移動平均**を、フレームごとの**動体画素数**を集計する（第2・3・7節）。
4. **シーク確認**: `POS_FRAMES` で先頭・1/4・中央・3/4・終端へ飛び、原フレームのモンタージュを保存（第5節）。
5. **レポート化**: 「処理FPS vs ソースFPS」と「動体画素数の推移」の2段グラフ（`mini_project_report.png`）と、メタデータ・処理統計・出力一覧を収めた `mini_project_report.json` を書き出す。

実行（`outputs/09_video_io_basics/` に成果物が出ます）:

```bash
uv run python lectures/09_video_io_basics/mini_project.py
```

主な出力は、`mini_project_source.mp4`（合成ソース）・`mini_project_processed.mp4`（動体ハイライト結果）・`mini_project_seek_montage.png`（シーク確認）・`mini_project_report.png`（FPSと動体のグラフ）・`mini_project_report.json`（数値レポート）・`mini_project_thumb_***.png`（間引きサムネ）です。**処理FPSがソースFPS(24)をはるか上回り、動く2物体のところだけ赤く染まる**ことを、目と数値の両方で確かめてください。これが、第10回以降の動体解析の出発点になります。

## 📜 スクリプト一覧

| ファイル | 役割 | 主な出力 |
| --- | --- | --- |
| `cv_helpers.py` | 共通ヘルパ（出力先・合成フレーム生成・動画書き出し・FOURCCデコード）。単体実行でスモークテスト | `helper_smoke.*` |
| `01_videocapture_loop.py` | VideoCapture 正準ループ＋フレーム基本操作＋BGR/RGB | `01_first_frame.png` / `01_gray.png` / `01_resized_half.png` / `01_roi_center.png` / `01_bgr_vs_rgb.png` |
| `02_capprops_seek.py` | メタデータ取得（CAP_PROP・FOURCCデコード）と POS_FRAMES シーク | `02_seek_grid.png` |
| `03_videowriter.py` | VideoWriter 書き出し＋処理FPS計測（基本の完成物） | `03_thumb_***.png` / `03_fps_plot.png` / `03_processed.mp4` |
| `mini_project.py` | 章末ミニプロジェクト（動体ハイライト統合パイプライン＋JSONレポート） | `mini_project_*.{mp4,png,json}` |
| `exercises.py` | 演習9問（TODO＋自己採点。未実装でも exit 0） | 採点用 `ex_grade.*` |
| `exercises_solutions.py` | 演習の模範解答（実行で全PASS。採点ロジックは exercises を再利用） | — |

## ✅ 到達チェックリスト

次のすべてを「何も見ずに書ける／理由を説明できる」ようになっていれば、本章は合格です。

- [ ] `cv2.VideoCapture` を開き、`isOpened()` を確認し、`while`＋`ret` 判定で読み、`release()` する**正準ループ**を空で書ける。
- [ ] ループの終了条件を**総フレーム数ではなく `ret`** にする理由（ライブでは総数が当てにならない）を説明できる。
- [ ] `read()` が返す `frame` が `(H,W,3)` `uint8` の **BGR** 配列だと分かり、`cvtColor`/`resize`（`dsize=(W,H)`）/ROI スライス（`[y0:y1,x0:x1]`）を正しく使える。
- [ ] matplotlib/Pillow に渡す前に **`BGR→RGB`** を挟む必要性を説明でき、忘れると赤青が反転することを再現できる。
- [ ] `cap.get(CAP_PROP_*)` が **float** を返すこと、整数が欲しい値は `int()` で丸めることを理解している。
- [ ] **FOURCC の 32bit 整数を4文字へデコード**でき、要求名と記録名が違い得る（`mp4v`→`FMP4`）ことを知っている。
- [ ] `POS_FRAMES` でのシークが**ファイルの特権**で、ライブでは使えないことを説明できる。
- [ ] `cv2.VideoWriter` を `(FOURCC, FPS, (W,H))` で作り、**`isOpened()` 検証**と**フレームサイズ整合**、ダメなら**連番PNGフォールバック**まで書ける。
- [ ] **ソースFPS と 処理FPS は別物**だと説明でき、`perf_counter`＋`deque` で処理FPSの移動平均を計測できる。
- [ ] `imshow`/`waitKey` を使わず、**`imwrite`/`VideoWriter`/matplotlib(Agg)** で headless 安全に結果を残せる。
- [ ] `mini_project.py` を実行し、出力（ハイライト動画・FPSグラフ・JSON）の意味を自分の言葉で説明できる。
- [ ] `exercises.py` を**全問 PASS**させた（`exercises_solutions.py` で答え合わせ済み）。

## ❓ よくある落とし穴・FAQ・デバッグ

第9節の「症状→原因→対処」表に加えて、ここでは実装中に効くデバッグの勘所をまとめます。

- **Q. `read()` が最初から `ret=False`。動画は確かにあるのに。** A. まず `cap.isOpened()` を print。`False` ならパス/コーデックの問題、`True` なのに読めないならコーデック未対応の可能性。`int(cap.get(cv2.CAP_PROP_FRAME_COUNT))` と `fourcc` も print して、そもそも何が開いているか可視化する。非ASCIIパスは `np.fromfile`＋`cv2.imdecode` を検討。
- **Q. 書き出した動画が 0 バイト／再生できない。** A. ほぼ `VideoWriter.isOpened()` が `False`。①FOURCCとコンテナ拡張子の整合（`mp4v`↔`.mp4`、`MJPG`↔`.avi`）、②`write()` するフレームの `(W,H)` が出力サイズと一致しているか、を疑う。`frame.shape[:2][::-1]` で `(W,H)` を出力サイズと比べてデバッグ。最終手段は連番PNG。
- **Q. 動画にフレームが入っているのに `CAP_PROP_FRAME_COUNT` がズレる（±1）。** A. コンテナ/コーデックの都合でメタ上の総数は近似値になり得る。だから採点も `±1` を許容しており、**正確に数えたいなら `ret` ループで実カウント**する（`ex1`）。
- **Q. 処理FPS が「数千fps」と出るが本当？** A. 合成フレームへの縮小/差分は非常に軽いので妥当。重い処理（深層モデル等）を挟むと一気に下がる。処理FPS は「**この処理の重さ**」の指標であって、ソースFPS（素材の属性）とは無関係です。
- **Q. matplotlib で保存した図の色がおかしい。** A. `BGR` のまま `imshow` に渡している。`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` を挟む。グレー画像は `cmap="gray"` を付ける。
- **Q. `cv2.error: ... function 'imshow'` で落ちる。** A. headless 版に `imshow`/`waitKey` は存在しない。表示は諦めて `imwrite`/`VideoWriter`/Agg 保存に統一する（本章は全スクリプトがこの方針）。
- **デバッグの定石**: 動画I/Oの不具合は「①開けたか（`isOpened`）②読めたか（`ret`）③サイズは合っているか（`shape` と `(W,H)`）④色順は合っているか（BGR/RGB）」の4点を順に print すれば、ほぼ切り分けられます。外部ツールが使える環境なら `ffprobe <file>` でコンテナ/コーデック/解像度/総数を一次情報として確認するのも早道です。

## 🚀 発展トピック・参考

本章で扱うのは、「ファイルとして合成動画を読み書きする」ところまでです。ここから先は、次回以降で深掘りしていきます。

- **第10回 古典的動画処理**: 本章のフレーム差分を、`createBackgroundSubtractorMOG2`/`KNN` による背景差分や、Lucas-Kanade / Farneback の**オプティカルフロー**へ発展させる。ミニプロジェクトの「動体マスク」がそのまま入口になります。
- **第11回 リアルタイム/ストリーム処理**: 実時間に追いつかせる定石（`cv2.resize(INTER_AREA)` での早期縮小・**Nフレームに1回だけ重い処理**・`cap.grab()`/`cap.retrieve()` での読み飛ばし）、`threading`＋`queue.Queue(maxsize=1)` の **producer/consumer でフレームをドロップ**して遅延蓄積を防ぐ構成、CPUバウンドは `multiprocessing` で分離、という現場の作法へ。
- **Webカメラ/RTSP**: `cv2.VideoCapture(0)`（OSごとのバックエンド: Linux=V4L2 / macOS=AVFOUNDATION / Windows=DSHOW・MSMF）、RTSP は `cv2.CAP_FFMPEG`＋環境変数 `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`、低遅延化に `CAP_PROP_BUFFERSIZE=1`。ライブ配信URLの解決には `yt-dlp`。
- **精密なコーデック/タイムスタンプ制御**: OpenCV で扱いにくい領域は **PyAV（`av.open`→`container.decode`→`frame.to_ndarray`、wheel に FFmpeg 同梱で system 不要）**や `imageio[ffmpeg]`、`subprocess` での `ffmpeg`/`ffprobe` 連携へ。`H264('avc1')` はライセンス次第で使えないことがあるため、移植性重視なら `mp4v`/`XVID` を既定にする。
- **公式ドキュメント**: [VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) ／ [VideoWriter](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html) ／ [opencv-python の配布形態](https://github.com/opencv/opencv-python)（full と headless の違い）。

---

## 動かし方

これらのスクリプトはすべて CPUのみ・ネット非依存・カメラ不要・追加依存なしで動きます（サンプル動画は各スクリプトが `numpy`/`cv2` で合成生成します）。リポジトリのルートで、以下を順に実行してください。結果はすべて `outputs/09_video_io_basics/` に画像・動画として保存され、画面表示はしません（headless 安全）。

```bash
# 1) VideoCapture の正準ループ（isOpened/read/ret/release）とフレーム基本操作
uv run python lectures/09_video_io_basics/01_videocapture_loop.py

# 2) メタデータ取得（CAP_PROP・FOURCC デコード）と POS_FRAMES シーク
uv run python lectures/09_video_io_basics/02_capprops_seek.py

# 3) VideoWriter 書き出し + 処理FPS計測（基本の完成物）
uv run python lectures/09_video_io_basics/03_videowriter.py

# 4) 章末ミニプロジェクト（動体ハイライト統合パイプライン＋JSONレポート）
uv run python lectures/09_video_io_basics/mini_project.py

# 演習（TODO を実装 → 自己採点。未実装でも FAIL 表示で正常終了する。全9問）
uv run python lectures/09_video_io_basics/exercises.py
# 行き詰まったら模範解答で挙動を確認（まずは自力で！）
SHOW_SOLUTION=1 uv run python lectures/09_video_io_basics/exercises.py
# 模範解答だけを直接実行して全PASSを確認することもできる
uv run python lectures/09_video_io_basics/exercises_solutions.py
```

実行後は、`outputs/09_video_io_basics/` の成果物を順に開いて、本文の確認ポイントと照らし合わせてください。特に `01_bgr_vs_rgb.png`（BGR/RGB の崩れ）、`02_seek_grid.png`（シーク先の円の位置とフレーム番号の一致）、`03_fps_plot.png`（処理FPS が ソースFPS を上回る様子）、`03_processed.mp4`（再書き出しした縮小動画）を見ると、各節の内容が一気に腑に落ちるはずです。なお `cv_helpers.py` を単体で実行すると、合成動画の「書き出し→読み戻し」が一周するスモークテストになります。

## まとめ

この章では、動画＝連続フレームという捉え方を起点に、`cv2.VideoCapture` で開いて `while`＋`ret` 判定でループし `release()` する正準パターン、`read()` が返す BGR numpy フレームへの基本操作（cvtColor/resize/ROI と BGR→RGB 変換）、`cap.get(CAP_PROP_*)` でのメタデータ取得と FOURCC のデコード、`POS_FRAMES` でのシーク（とライブでの注意）、`cv2.VideoWriter` での書き出し（FOURCC・サイズ整合・`isOpened()` 検証・連番PNGフォールバック）、そして `time.perf_counter`＋`deque` で測る処理FPS と ソースFPS の違いまでを、すべて「自分で再現し説明できる」レベルで一通り組み立てました。

ここで身につけた「開く→retでループ→処理→書き出す→解放」という流れと、「ファイルとライブで前提が変わる／処理FPSとソースFPSは別物」という勘所は、次回（第10回 古典的動画処理：背景差分・オプティカルフロー）と第11回（リアルタイム/ストリーム処理）の土台にそのまま効いてきます。まずは演習を自力で全問 PASS させ、`isOpened()` チェック・`ret` 判定ループ・FOURCC デコード・処理FPS の移動平均という定石を手に馴染ませてから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4 ／ opencv-python-headless 4.13（`cv2` 4.13.0、VideoCapture/VideoWriter は FFmpeg 同梱・本体機能で contrib 不要）／ Pillow 12.2 ／ matplotlib 3.10（Agg バックエンドで画面非依存に保存）
