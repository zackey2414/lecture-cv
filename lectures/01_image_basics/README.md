# 第1回 画像の基礎 — ndarray表現・BGR/RGB・OpenCV/Pillow I/O・headless表示

> トラック: 画像の基礎 ／ レベル: 入門 ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません）

## 🎯 この章のゴール

この章を終えたとき、あなたは「画像とは結局 `(H, W, 3)` の `uint8` の numpy 配列にすぎない」という感覚を体に覚え込ませた状態になります。つまり、画像処理の関数を前にしても「これは配列のどの軸を、どんな値域で触っているのか」を自分の言葉で説明でき、ライブラリの戻り値の `shape` と `dtype` を見ただけで何が起きたかを推測できるようになります。この感覚は今後の全モジュールの土台であり、ここが曖昧なままだと、後段のフィルタ・幾何変換・深層学習の前処理で必ずつまずくことになります。

同時に、現場で初学者が必ず踏む3つの地雷 ——「OpenCV は BGR、Pillow / matplotlib / PyTorch は RGB」という色順の食い違い、「`cv2.imread` は失敗しても例外ではなく `None` を返す」という静かな罠、「`cv2.imshow` は画面の無いサーバや Docker でプロセスごと固まる/落ちる」という表示の罠 —— を、知識としてではなく「自分で再現し、自分で回避コードを書ける」レベルで潰します。

到達点を一言でいえば、**サンプル画像が無くても、GPU が無くても、ディスプレイが無くても、自分一人で画像を確実に読み・書き・確認できる**こと。AI 補助なしで I/O のヘルパをそらで書け、色が化けたら原因を即座に言い当てられる。それがこの章の合格ラインです。

---

## 1. 画像は `(H, W, 3)` の `uint8` numpy 配列である

最初に頭に刻むべきことは、デジタル画像が特別な「画像オブジェクト」ではなく、ただの数値の格子（numpy 配列）だという事実です。カラー画像は縦 `H` ピクセル・横 `W` ピクセルの各点に、色を表す数値が並んでいるだけです。OpenCV で画像を読み込むと、その正体は形状 `(H, W, 3)`・データ型 `uint8` の `numpy.ndarray` です。`3` は色チャンネル数、`uint8` は 0〜255 の整数（8ビット符号なし整数）で、これが「真っ暗（0）から最大の明るさ（255）まで」を表します。

<figure class="lec-fig"><svg viewBox="0 0 660 256" role="img" aria-label="カラー画像は形状(H,W,3)・uint8のnumpy配列。各画素は[B,G,R]の0〜255の整数3つを持つ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="176" y="32" text-anchor="middle" font-size="15" fill="#3f3f46">列数 W（幅・第1軸） →</text><line x1="86" y1="46" x2="260" y2="46" stroke="#71717a" stroke-width="2"/><polygon points="266,46 256,41 256,51" fill="#71717a"/><text x="44" y="124" text-anchor="middle" font-size="15" fill="#3f3f46" style="writing-mode:vertical-rl;text-orientation:upright">行数 H・高さ</text><line x1="70" y1="64" x2="70" y2="182" stroke="#71717a" stroke-width="2"/><polygon points="70,188 65,178 75,178" fill="#71717a"/><rect x="86" y="64" width="180" height="120" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><line x1="116" y1="64" x2="116" y2="184" stroke="#e4e4e7"/><line x1="146" y1="64" x2="146" y2="184" stroke="#e4e4e7"/><line x1="176" y1="64" x2="176" y2="184" stroke="#e4e4e7"/><line x1="206" y1="64" x2="206" y2="184" stroke="#e4e4e7"/><line x1="236" y1="64" x2="236" y2="184" stroke="#e4e4e7"/><line x1="86" y1="94" x2="266" y2="94" stroke="#e4e4e7"/><line x1="86" y1="124" x2="266" y2="124" stroke="#e4e4e7"/><line x1="86" y1="154" x2="266" y2="154" stroke="#e4e4e7"/><rect x="176" y="94" width="30" height="30" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><line x1="206" y1="94" x2="440" y2="72" stroke="#c2410c" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.5"/><line x1="206" y1="124" x2="440" y2="172" stroke="#c2410c" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.5"/><text x="500" y="60" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">1 画素 = [B, G, R]</text><rect x="440" y="72" width="120" height="32" fill="#2563eb"/><text x="500" y="93" text-anchor="middle" font-size="14" font-weight="600" fill="#ffffff">B = 0</text><rect x="440" y="106" width="120" height="32" fill="#16a34a"/><text x="500" y="127" text-anchor="middle" font-size="14" font-weight="600" fill="#ffffff">G = 0</text><rect x="440" y="140" width="120" height="32" fill="#dc2626"/><text x="500" y="161" text-anchor="middle" font-size="14" font-weight="600" fill="#ffffff">R = 255</text><rect x="440" y="200" width="120" height="26" rx="5" fill="#dc2626"/><text x="500" y="217" text-anchor="middle" font-size="12.5" fill="#ffffff">= 真っ赤</text><text x="86" y="210" font-size="15" font-weight="700" fill="#18181b" font-family="'JetBrains Mono', monospace">shape = (H, W, 3)</text><text x="86" y="232" font-size="12.5" fill="#52525b">dtype = uint8（各値 0〜255 の整数）</text></svg><figcaption>カラー画像の正体は、形状 <b>(H, W, 3)</b>・データ型 <b>uint8</b> の numpy 配列です。形状は <b>(行数 H, 列数 W, チャンネル数 3)</b> の順で、縦に H 行・横に W 列の画素が並び、各画素は <b>[B, G, R]</b> の3つの整数（各 0〜255）を持ちます。図の画素 <code>[0, 0, 255]</code>（B=0, G=0, R=255）は「真っ赤」を表します。</figcaption></figure>

なぜ `uint8` なのかというと、人間の目が区別できる明るさの段階がだいたい 256 段階に収まり、1ピクセル1チャンネルを1バイトで表現するのがメモリ効率・互換性ともに最良だからです。ここで重要なのは値域が固定された整数型だという点で、後述する「255 を超えたらどうなるか」という飽和/オーバーフロー問題は、この `uint8` という型の性質から直接生まれます。型を意識せずに足し算すると痛い目を見るのは、まさにこの型の性質ゆえなのです。

デバッグの第一歩は「いま手元の配列の素性を見ること」です。だからこそスクリプト `01_imread_imwrite.py` も、読み込んだ画像について必ず次の3点を表示します。現場でも私たちは、まずこの3行を打つところから始めます。

```python
print(img.shape)   # (240, 320, 3)  ← (高さH, 幅W, チャンネル3)
print(img.dtype)   # uint8           ← 0〜255 の整数
print(img.min(), img.max())  # 0 255 ← 値域の確認
```

上の出力で `shape` が `(H, W, 3)` になっていることを必ず確認してください。numpy の慣習では、**第0軸（縦）の大きさが行数 H＝高さ、第1軸（横）の大きさが列数 W＝幅**です。`shape` は `(行数 H, 列数 W, チャンネル数)` の順に並び、これは日常感覚の「横×縦」とは順序が逆です。ここを取り違えると幅と高さを永遠に間違え続けます。

<figure class="lec-fig"><svg viewBox="0 0 640 230" role="img" aria-label="numpyの第0軸が縦(行y・高さH)、第1軸が横(列x・幅W)。img[y,x]は行が先・列が後で、日常の横×縦とは逆" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="255" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#2563eb">第1軸：列数 W（横）</text><line x1="80" y1="55" x2="414" y2="55" stroke="#2563eb" stroke-width="2.5"/><polygon points="426,55 414,49 414,61" fill="#2563eb"/><text x="42" y="123" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c" style="writing-mode:vertical-rl;text-orientation:upright">第0軸：行数 H</text><line x1="80" y1="55" x2="80" y2="200" stroke="#c2410c" stroke-width="2.5"/><polygon points="80,212 74,200 86,200" fill="#c2410c"/><rect x="80" y="55" width="170" height="136" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><line x1="114" y1="55" x2="114" y2="191" stroke="#e4e4e7"/><line x1="148" y1="55" x2="148" y2="191" stroke="#e4e4e7"/><line x1="182" y1="55" x2="182" y2="191" stroke="#e4e4e7"/><line x1="216" y1="55" x2="216" y2="191" stroke="#e4e4e7"/><line x1="80" y1="89" x2="250" y2="89" stroke="#e4e4e7"/><line x1="80" y1="123" x2="250" y2="123" stroke="#e4e4e7"/><line x1="80" y1="157" x2="250" y2="157" stroke="#e4e4e7"/><rect x="182" y="89" width="34" height="34" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><line x1="216" y1="106" x2="296" y2="100" stroke="#c2410c" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.45"/><text x="300" y="106" font-size="17" font-weight="700" fill="#18181b" font-family="'JetBrains Mono', monospace">img[1, 3]</text><text x="300" y="130" font-size="13" fill="#52525b">＝ img[行 1, 列 3]</text><text x="300" y="154" font-size="13" font-weight="600" fill="#c2410c">先に行（何行目）→ 列（何列目）</text><text x="300" y="178" font-size="12.5" fill="#b45309">日常の「横 × 縦」とは順序が逆</text></svg><figcaption>numpy 配列の軸の向きです。<b>第0軸（縦）の大きさが行数 H＝高さ</b>、<b>第1軸（横）の大きさが列数 W＝幅</b>です。画素アクセス <code>img[y, x]</code> は<b>先に行（何行目 y）、次に列（何列目 x）</b>で、日常の「横×縦」とは順序が逆になる点に注意してください。</figcaption></figure>

`dtype` が `uint8` でないとき（例えば `float64`）は、どこかで型が変わった証拠で、保存や表示で色が壊れる前兆だと考えてください。

## 2. 画素アクセス・ROI・チャンネルは「BGR」順

個々のピクセルには `img[y, x]` でアクセスします。指定は**先に行（何行目 y）、次に列（何列目 x）**で、`img[x, y]` ではありません。返ってくるのは長さ3の配列ですが、その中身の並びが OpenCV 最大の癖です。OpenCV は歴史的経緯から色を **B, G, R（青・緑・赤）の順**で格納します。したがって「真っ赤」な画素は `[0, 0, 255]`、「真っ青」は `[255, 0, 0]` になります。世間一般の RGB とは赤と青が入れ替わっている、と最初に強く意識してください。

画像の一部分だけを切り出したいときは numpy のスライスをそのまま使います。`roi = img[y0:y1, x0:x1]` で矩形領域（ROI: Region Of Interest）を取り出せ、`img[:, :, 2]` と書けば R チャンネルだけを2次元配列として取り出せます。ここが「画像 = 配列」であることの最大の利点で、専用 API を覚えなくても numpy の知識がそのまま画像操作に転用できます。スライスは（多くの場合）コピーではなくビューなので、`roi[:] = 0` のように書き換えると元画像にも反映される点だけ覚えておきましょう。

```python
y, x = 10, 20
print(img[y, x])      # [0, 0, 255] のように [B, G, R]
roi = img[0:120, 0:160]   # 左上の矩形を取り出す（ビュー）
red_plane = img[:, :, 2]  # R チャンネルだけ（2次元 (H, W)）
```

このコードで `img[y, x]` が3要素の `[B, G, R]` を返すこと、`red_plane` が2次元になることを押さえてください。「色を1つ取り出すとチャンネル軸が消えて次元が減る」という挙動は、後で出てくるグレースケール画像の `(H, W)` とも直結します。配列の次元が増えたり減ったりする感覚に慣れることが、この章の隠れた目標です。

## 3. OpenCV の入出力: `cv2.imread` / `cv2.imwrite` と `IMREAD_*` フラグ

ファイルから画像を読むのが `cv2.imread(path, flags)`、書き出すのが `cv2.imwrite(path, img)` です。`imread` は **BGR** の配列を返し、`imwrite` も **BGR** の配列を受け取る前提です。つまり OpenCV の世界だけで完結している限り色順は一貫していて、変換は要りません。問題が起きるのは、この BGR 配列を Pillow や matplotlib（RGB の世界）へ渡したときだけ —— という構図を最初に理解しておくと、後の混乱が激減します。

`imread` の第2引数 `flags` は、何チャンネルで読み込むかを決めます。代表的な3つは次の通りで、フラグによって戻り値の `shape` が変わる点が肝心です。`IMREAD_GRAYSCALE` を選ぶと2次元の `(H, W)` になり、チャンネル軸が消えます。

| フラグ | 読み込まれるチャンネル | 戻り値の shape |
| --- | --- | --- |
| `cv2.IMREAD_COLOR`（既定） | 常に3ch（BGR、アルファは捨てる） | `(H, W, 3)` |
| `cv2.IMREAD_GRAYSCALE` | 1ch（輝度のみ） | `(H, W)` |
| `cv2.IMREAD_UNCHANGED` | 元のまま（アルファがあれば4ch） | `(H, W, 3)` or `(H, W, 4)` |

この表のポイントは「同じファイルでも、どのフラグで読むかで次元数が変わる」ことです。`IMREAD_COLOR` はたとえ元がグレースケールでも必ず3チャンネルに膨らませ、逆に `IMREAD_UNCHANGED` は元の構造（アルファ含む）をそのまま保ちます。後段の処理が「3チャンネル前提」なのか「アルファも欲しい」のかで、ここを意識的に選び分けることになります。

保存形式は拡張子で決まります。`.png` は可逆圧縮で画質が劣化しない代わりにサイズが大きく、`.jpg` は非可逆で小さい代わりに劣化します。JPEG の画質は `[cv2.IMWRITE_JPEG_QUALITY, 0..100]` で、PNG の圧縮率は `[cv2.IMWRITE_PNG_COMPRESSION, 0..9]` で指定します。`01_imread_imwrite.py` を実行すると、同じ画像を品質95と品質30の JPEG で保存し、ファイルサイズが目に見えて変わることを確認できます。

```python
cv2.imwrite("out.png", img)                                   # 可逆・劣化なし
cv2.imwrite("q95.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])    # 高品質
cv2.imwrite("q30.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])    # 低品質・小サイズ
```

実務では「中間生成物は PNG（劣化させたくない）、最終的な軽い配布物は JPEG」のように使い分けます。JPEG を繰り返し開いて保存し直すと劣化が蓄積するので、加工途中のデータを JPEG で持ち回らないのが鉄則です。

## 4. 最大の静かな罠 — `imread` は失敗しても `None` を返す

`cv2.imread` は、ファイルが存在しない・壊れている・対応外の形式といった失敗時に、**例外を投げず `None` を返します**。これが現場で最も多い事故の温床です。`None` のまま次の処理に渡すと、その場では落ちず、ずっと後の `img.shape` などで `AttributeError: 'NoneType' object has no attribute 'shape'` となって初めて気づく —— しかもエラー箇所が本当の原因（読み込み失敗）から遠いので、デバッグに無駄な時間を取られます。

なぜ例外でなく `None` なのかというと、OpenCV が C++ ライブラリで、戻り値で成否を返す設計思想だからです。Python 的な「失敗したら例外」に慣れていると見落としますが、これは仕様です。だからこそ、**読み込んだ直後に必ず `None` チェックを書く**のが定石になります。本講座のヘルパ `load_bgr_checked` は、この `None` を握りつぶさず、原因がすぐ分かる `FileNotFoundError` に変換して早期に投げます。

```python
img = cv2.imread(path)
if img is None:                       # ← 読み込み直後に必ず確認
    raise FileNotFoundError(f"読めません: {path}")
```

ポイントは「早期 return / 早期 raise」で問題を発生源の近くで潰すことです。`None` を返す API は OpenCV に限らず多いので、「戻り値で失敗を表す関数は、呼んだ直後にチェックする」という習慣そのものを身につけてください。これは単なる画像処理の話ではなく、堅いコードを書くための一般原則です。

## 5. 日本語（非ASCII）パスの罠と `imdecode` / `imencode`

`cv2.imread` / `cv2.imwrite` は内部で C の標準ファイル API を使うため、環境（特に Windows）によっては**日本語やマルチバイト文字を含むパスを開けない**ことがあります。Linux/macOS では UTF-8 でうまくいくことも多いのですが、「環境によって動いたり動かなかったり」するコードは実務では地雷です。クロスプラットフォームで確実に動かすための定石を、最初から手癖にしておきましょう。

その定石が、ファイル入出力を「バイト列の読み書き」と「メモリ上でのデコード/エンコード」に分解する方法です。読み込みは `np.fromfile`（Python の I/O 経由なので非ASCIIパスに強い）でバイト列を読み、`cv2.imdecode` でメモリ上の画像に変換します。書き込みは逆で、`cv2.imencode` でメモリ上にエンコードしてから `ndarray.tofile` で書き出します。本講座のヘルパ `imread_unicode` / `imwrite_unicode` がまさにこの実装です。

<figure class="lec-fig"><svg viewBox="0 0 660 320" role="img" aria-label="日本語など非ASCIIパス対応のIOパイプライン。読み込みはファイルをnp.fromfileでバイト列にしcv2.imdecodeで画像へ、書き込みはcv2.imencodeでバイト列にしtofileでファイルへ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="355" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">日本語(非ASCII)パス対応 I/O のパイプライン</text><text x="38" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c" style="writing-mode:vertical-rl;text-orientation:upright">読み込み</text><text x="38" y="230" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c" style="writing-mode:vertical-rl;text-orientation:upright">書き込み</text><rect x="75" y="72" width="140" height="56" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><rect x="290" y="72" width="140" height="56" rx="6" fill="#ffedd5" stroke="#f97316" stroke-width="2"/><rect x="505" y="72" width="130" height="56" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><rect x="75" y="202" width="140" height="56" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><rect x="290" y="202" width="140" height="56" rx="6" fill="#ffedd5" stroke="#f97316" stroke-width="2"/><rect x="505" y="202" width="130" height="56" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="145" y="96" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像ファイル</text><text x="145" y="114" text-anchor="middle" font-size="11" fill="#52525b">(日本語パス)</text><text x="360" y="96" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">バイト列</text><text x="360" y="114" text-anchor="middle" font-size="11" fill="#52525b">uint8 1次元</text><text x="570" y="96" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">画像 ndarray</text><text x="570" y="114" text-anchor="middle" font-size="11" fill="#52525b">(H, W, 3)</text><text x="145" y="226" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">画像 ndarray</text><text x="145" y="244" text-anchor="middle" font-size="11" fill="#52525b">(H, W, 3)</text><text x="360" y="226" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">バイト列</text><text x="360" y="244" text-anchor="middle" font-size="11" fill="#52525b">エンコード済</text><text x="570" y="226" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像ファイル</text><text x="570" y="244" text-anchor="middle" font-size="11" fill="#52525b">(日本語パス)</text><line x1="217" y1="100" x2="284" y2="100" stroke="#71717a" stroke-width="2"/><polygon points="290,100 280,95 280,105" fill="#71717a"/><text x="250" y="90" text-anchor="middle" font-size="11.5" fill="#3f3f46">np.fromfile</text><line x1="432" y1="100" x2="499" y2="100" stroke="#71717a" stroke-width="2"/><polygon points="505,100 495,95 495,105" fill="#71717a"/><text x="467" y="90" text-anchor="middle" font-size="11.5" fill="#3f3f46">cv2.imdecode</text><line x1="217" y1="230" x2="284" y2="230" stroke="#71717a" stroke-width="2"/><polygon points="290,230 280,225 280,235" fill="#71717a"/><text x="250" y="220" text-anchor="middle" font-size="11.5" fill="#3f3f46">cv2.imencode</text><line x1="432" y1="230" x2="499" y2="230" stroke="#71717a" stroke-width="2"/><polygon points="505,230 495,225 495,235" fill="#71717a"/><text x="467" y="220" text-anchor="middle" font-size="11.5" fill="#3f3f46">.tofile</text></svg><figcaption>非ASCIIパス（日本語など）でも確実に動かす定石です。<b>読み込み</b>はファイルを <code>np.fromfile</code> で<b>バイト列</b>として読み、<code>cv2.imdecode</code> で<b>画像 ndarray</b> に変換します。<b>書き込み</b>はその逆で、<code>cv2.imencode</code> でバイト列にしてから <code>ndarray.tofile</code> で書き出します。グレーがファイル、橙がバイト列、青が画像配列です。</figcaption></figure>

```python
# 読み込み（日本語パスでも安全）
buf = np.fromfile("画像/サンプル.png", dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

# 書き込み（日本語パスでも安全）
ok, buf = cv2.imencode(".png", img)   # 拡張子で形式が決まる
buf.tofile("出力/結果_日本語.png")
```

このパターンの嬉しい副作用として、`imdecode` は「ネットワークから受け取ったバイト列」や「ZIP の中の画像」など、ファイルパスを介さないデータにもそのまま使えます。`01_imread_imwrite.py` は実際に `サンプル_日本語パス.png` という名前で保存して読み戻し、欠損なく往復できることを確認します。「非ASCIIパスは fromfile/imdecode で」と覚えておけば、Windows ユーザに渡したコードが突然動かない、という事故を未然に防げます。

## 6. 最重要前提 — BGR と RGB の食い違い

この章で唯一「絶対に忘れてはいけない」ことを挙げるなら、それは**OpenCV は BGR、それ以外（Pillow・matplotlib・PyTorch・一般的な画像の常識）は RGB** という非対称です。OpenCV で読んだ配列をそのまま Pillow や matplotlib に渡すと、赤と青が入れ替わった奇妙な色になります。バグというより「翻訳し忘れ」で、原因を知らないと延々と悩みます。逆に知っていれば一発で直せます。

<figure class="lec-fig"><svg viewBox="0 0 640 240" role="img" aria-label="同じ3バイト[0,0,255]を、OpenCVはBGR順で赤、PILやmatplotlibはRGB順で青と解釈する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="28" text-anchor="middle" font-size="14" fill="#3f3f46">ファイル上の同じ 3 バイト列</text><rect x="245" y="40" width="50" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><rect x="295" y="40" width="50" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><rect x="345" y="40" width="50" height="40" fill="#ffedd5" stroke="#c2410c" stroke-width="2.2"/><text x="270" y="66" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">0</text><text x="320" y="66" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">0</text><text x="370" y="66" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">255</text><line x1="305" y1="84" x2="197" y2="125" stroke="#71717a" stroke-width="1.8"/><polygon points="195,126 203,118 206,127" fill="#71717a"/><line x1="335" y1="84" x2="463" y2="125" stroke="#71717a" stroke-width="1.8"/><polygon points="465,126 454,128 457,118" fill="#71717a"/><text x="160" y="150" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">OpenCV ＝ BGR 順</text><text x="160" y="172" text-anchor="middle" font-size="12.5" fill="#52525b">3番目の 255 は R（赤）</text><rect x="100" y="182" width="120" height="44" rx="5" fill="#dc2626"/><text x="160" y="210" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">＝ 赤</text><text x="480" y="150" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">PIL / plt ＝ RGB 順</text><text x="480" y="172" text-anchor="middle" font-size="12.5" fill="#52525b">3番目の 255 は B（青）</text><rect x="420" y="182" width="120" height="44" rx="5" fill="#2563eb"/><text x="480" y="210" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">＝ 青</text></svg><figcaption>ファイルから読んだ<b>同じ 3 バイト [0, 0, 255]</b> でも、解釈する色順で見える色が変わります。<b>OpenCV は BGR 順</b>なので 3 番目の 255 は R となり<b>赤</b>、<b>PIL / matplotlib は RGB 順</b>なので 3 番目の 255 は B となり<b>青</b>に見えます。<code>cv2.cvtColor(bgr, COLOR_BGR2RGB)</code> を挟み忘れると、この赤青の入れ替わりが起きます。</figcaption></figure>

変換は `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` の一行です。OpenCV 内で完結するなら変換は不要、**OpenCV の外（PIL / matplotlib / 学習フレームワーク）に出す瞬間に RGB へ変換する**、と覚えてください。`02_bgr_rgb_pitfall.py` は、わざと変換し忘れた「色が入れ替わった画像」と、正しく変換した画像の両方を保存し、さらに matplotlib で左右に並べた比較図を作ります。左上を赤にした合成画像が、変換を忘れると青くなる様子を目で見て確認してください。

```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # 外の世界へ渡す前に変換
Image.fromarray(rgb).save("correct.png")     # PIL は RGB 前提なのでこれで正しい
Image.fromarray(bgr).save("wrong.png")       # BGR を渡すと赤青が入れ替わる
```

| ライブラリ | 既定の色順 | 渡すときに必要なこと |
| --- | --- | --- |
| OpenCV (`cv2`) | **BGR** | OpenCV 内なら変換不要 |
| Pillow (`PIL`) | RGB | `cv2.cvtColor(BGR2RGB)` してから `Image.fromarray` |
| matplotlib (`plt.imshow`) | RGB | 同上。忘れると色が化ける |
| PyTorch（後の章） | RGB | 同上。前処理で必ず変換する |

この表を「OpenCV だけが仲間外れ」と捉えると記憶が定着します。あわせて、実務でのチェック法も覚えておきましょう。表示・保存した画像で「空が赤い」「肌が青い」など**赤と青が入れ替わって見えたら、ほぼ確実に BGR/RGB の変換忘れ**です。原因の当たりがつくだけでデバッグ速度が段違いになります。

## 7. グレースケールは `(H, W)`、次元が1つ違う

カラー画像が `(H, W, 3)` なのに対し、グレースケール（白黒）画像は **`(H, W)` の2次元**で、チャンネル軸がありません。`cv2.imread(path, cv2.IMREAD_GRAYSCALE)` や `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` の戻り値はこの2次元配列です。`img[y, x]` がカラーでは3要素配列を返したのに対し、グレーでは1個の輝度値（スカラー）を返します。「色を落とすと次元が1つ減る」という対応関係を、ここではっきり結びつけてください。

<figure class="lec-fig"><svg viewBox="0 0 620 250" role="img" aria-label="カラー(H,W,3)の3チャンネルをBGR2GRAYで輝度1枚にまとめると(H,W)の2次元になり次元が1つ減る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="104" y="56" width="104" height="104" fill="#ffffff" stroke="#dc2626" stroke-width="2"/><rect x="90" y="70" width="104" height="104" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><rect x="76" y="84" width="104" height="104" fill="#ffffff" stroke="#2563eb" stroke-width="2"/><text x="200" y="72" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">R</text><text x="186" y="86" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">G</text><text x="128" y="142" text-anchor="middle" font-size="14" font-weight="700" fill="#2563eb">B</text><text x="140" y="218" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">(H, W, 3)・3次元</text><line x1="228" y1="122" x2="392" y2="122" stroke="#c2410c" stroke-width="2.5"/><polygon points="400,122 388,116 388,128" fill="#c2410c"/><text x="312" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">cvtColor(BGR2GRAY)</text><rect x="412" y="72" width="116" height="116" fill="#d4d4d8" stroke="#52525b" stroke-width="2"/><text x="470" y="218" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">(H, W)・2次元</text><text x="310" y="242" text-anchor="middle" font-size="13" fill="#52525b">チャンネル軸が消え、次元が 3 → 2 に減る</text></svg><figcaption>カラー画像は B・G・R の<b>3 枚の面</b>を重ねた <b>(H, W, 3)</b> の 3 次元配列です。<code>cv2.cvtColor(bgr, COLOR_BGR2GRAY)</code> で輝度 1 枚にまとめると <b>(H, W)</b> の 2 次元になり、<b>チャンネル軸が消えて次元が 1 つ減ります</b>。3ch 前提の処理に渡すときは <code>GRAY2BGR</code> で見た目そのまま 3ch に戻します。</figcaption></figure>

この次元の違いは、関数に画像を渡すときに地味に効いてきます。例えば「3チャンネル前提」で書かれた処理にグレースケール画像を渡すと形が合わずに落ちますし、逆もまた然りです。グレースケール画像を後で他の3チャンネル画像と横に並べたい（連結したい）ときは、`cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)` で**見た目は白黒のまま3チャンネルに戻す**必要があります。`04_display_headless.py` のコンタクトシート生成では、まさにこの「グレーを3chに戻してから並べる」処理を行っています。

```python
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)   # (H, W) 2次元になる
print(gray.shape)                               # (240, 320)
gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 並べる/重ねる用に3chへ戻す
```

要するに「チャンネル軸の有無」を常に意識することが大切です。エラーメッセージに `(240, 320)` と `(240, 320, 3)` が混在していたら、それはグレーとカラーの取り違えを疑うサインです。`shape` を一目見て次元数を数える癖をつけましょう。

## 8. `uint8` オーバーフロー — numpy の `+` と `cv2.add` は別物

画像の明るさを上げたくて「全画素に 100 を足す」とき、書き方によって結果が**まったく異なります**。これは `uint8` が 0〜255 しか表せない型だからです。numpy の `+` 演算子は 256 を法とした剰余演算（巻き戻り）を行うため、`200 + 100 = 300` は `300 - 256 = 44` になります。明るくしたはずの部分がかえって暗くなり、ノイズのような模様が出ます。これがオーバーフローです。

一方 `cv2.add` は**飽和演算（saturating）**で、255 を超えたら 255 で頭打ちにします。`200 + 100` は `255` に丸められ、「これ以上明るくならない」という人間の直感に一致します。明るさ・コントラスト調整のような画像処理では、こちらの飽和が正しい振る舞いです。`02_bgr_rgb_pitfall.py` は両方を計算して保存し、白い画素（255）に 100 を足した結果が、numpy では `99` に巻き戻り、`cv2.add` では `255` に留まることを数値で示します。

<figure class="lec-fig"><svg viewBox="0 0 640 250" role="img" aria-label="uint8で200+100=300は範囲外。numpyの+は300−256=44に巻き戻り、cv2.addは255で飽和する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">200 + 100 = 300（uint8 の範囲は 0〜255）</text><rect x="70" y="62" width="434" height="24" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><rect x="504" y="62" width="76" height="24" fill="#ffedd5" stroke="#f97316" stroke-width="1.5"/><line x1="70" y1="86" x2="70" y2="96" stroke="#71717a" stroke-width="1.5"/><line x1="504" y1="86" x2="504" y2="96" stroke="#71717a" stroke-width="1.5"/><line x1="580" y1="86" x2="580" y2="96" stroke="#71717a" stroke-width="1.5"/><text x="70" y="112" text-anchor="middle" font-size="12.5" fill="#52525b">0</text><text x="504" y="112" text-anchor="middle" font-size="12.5" fill="#52525b">255</text><text x="580" y="112" text-anchor="middle" font-size="12.5" fill="#c2410c">300</text><text x="410" y="52" text-anchor="middle" font-size="12.5" font-weight="700" fill="#2563eb">200</text><circle cx="410" cy="74" r="6" fill="#2563eb"/><line x1="418" y1="74" x2="572" y2="74" stroke="#71717a" stroke-width="1.8" stroke-dasharray="5 3"/><polygon points="580,74 570,69 570,79" fill="#71717a"/><text x="498" y="52" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3f3f46">+100</text><circle cx="145" cy="74" r="6" fill="#dc2626"/><circle cx="504" cy="74" r="6" fill="#16a34a"/><line x1="145" y1="80" x2="185" y2="148" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5 3"/><line x1="504" y1="80" x2="465" y2="148" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="5 3"/><rect x="70" y="150" width="230" height="72" rx="6" fill="#fff7ed" stroke="#dc2626" stroke-width="2"/><text x="185" y="182" text-anchor="middle" font-size="15" font-weight="700" fill="#dc2626">numpy の + → 44</text><text x="185" y="206" text-anchor="middle" font-size="12.5" fill="#52525b">300 − 256（巻き戻り）</text><rect x="350" y="150" width="230" height="72" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="465" y="182" text-anchor="middle" font-size="15" font-weight="700" fill="#15803d">cv2.add → 255</text><text x="465" y="206" text-anchor="middle" font-size="12.5" fill="#52525b">255 で頭打ち（飽和）</text></svg><figcaption><code>uint8</code> は <b>0〜255</b> しか表せないため、<code>200 + 100 = 300</code> は範囲外です。<b>numpy の <code>+</code></b> は 256 で割った余りに巻き戻り <code>300 − 256 = 44</code> となって逆に暗くなります。<b><code>cv2.add</code></b> は<b>飽和</b>して <code>255</code> で頭打ちになります。明るさ調整では飽和する <code>cv2.add</code> が直感どおりの結果です。</figcaption></figure>

```python
add = np.full_like(bgr, 100)
naive     = bgr + add          # 255 を超えると巻き戻る（オーバーフロー）
saturated = cv2.add(bgr, add)  # 255 で頭打ち（飽和）
```

このポイントの本質は「`uint8` の演算では値域を意識せよ」ということです。安全策として、複雑な計算をするときは一度 `astype(np.int16)` や `astype(np.float32)` で広い型に変換し、計算し終えてから `np.clip(x, 0, 255).astype(np.uint8)` で戻す、というパターンもよく使います（演習9でこれを手作りします）。「画像の足し算・掛け算は型のことを考える」——これを忘れると、原因不明の汚い画像に悩まされます。

## 9. Pillow 入門と PIL ↔ numpy ↔ cv2 の相互変換

Pillow（`PIL`）はもう一つの定番ライブラリで、`Image.open` / `Image.save` による I/O、`resize` / `crop` / `rotate` などの加工、`ImageDraw` での描画、各種フィルタを、読みやすい API で提供します。OpenCV が「配列をゴリゴリ計算する」のに向くのに対し、Pillow は「画像を直感的に編集する」のに向き、実務では両者を行き来します。だからこそ**相互変換を淀みなく書けること**が重要になります。

Pillow で最も事故りやすいのが**サイズの順序**です。`Image.size` は `(幅W, 高さH)` を返しますが、numpy の `shape` は `(高さH, 幅W, チャンネル)` で**順序が逆**です。`resize((160, 120))` も `(幅, 高さ)` で指定します。さらに `mode`（`"RGB"` / `"L"`（グレー）/ `"RGBA"`（アルファ付き））が、numpy にしたときのチャンネル数を決めます。下の表のように、PIL の `mode` と numpy の `shape` は対応しています。

| PIL `mode` | 意味 | `np.asarray` 後の shape |
| --- | --- | --- |
| `"L"` | グレースケール（1ch） | `(H, W)` |
| `"RGB"` | カラー（3ch） | `(H, W, 3)` |
| `"RGBA"` | カラー+アルファ（4ch） | `(H, W, 4)` |

この表が示す通り、`mode` を見れば numpy にしたときの次元が予測できます。変換は `np.asarray(pil_img)` で PIL → numpy、`Image.fromarray(arr)` で numpy → PIL です。ただし `np.asarray` は多くの場合**コピーを作らず読み取り専用のビュー**を返すので、書き換えたいときは `np.array(pil_img)`（コピー）を使うか `.copy()` してください。`03_pillow_numpy_interop.py` はこの writeable フラグまで表示して確認します。

```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # cv2(BGR) → RGB
pil = Image.fromarray(rgb)                    # RGB → PIL
arr = np.asarray(pil)                         # PIL → numpy（読み取り専用ビュー）
back = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)   # numpy → cv2(BGR) へ戻す
assert np.array_equal(bgr, back)              # 1周して一致 = 変換が正しく閉じた証拠
```

この「cv2 → PIL → numpy → cv2」のラウンドトリップが一致することを確認できれば、3者の橋渡しを完全に理解した証拠です。鍵は常に同じで、**PIL/matplotlib は RGB、cv2 は BGR。境界をまたぐたびに `cvtColor` する**。この一点さえ守れば、どんな組み合わせでも色は崩れません。`03_pillow_numpy_interop.py` は加えて `resize` / `crop` / `rotate` / `ImageDraw` / `GaussianBlur` も一通り触るので、Pillow の基本操作はここで体験できます。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="色順のラウンドトリップ。cv2のBGRをcvtColorでPILのRGBへ、np.asarrayでnumpyのRGBへ、cvtColorで再びcv2のBGRへ戻すと画素が完全一致する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="36" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">色順のラウンドトリップ (cv2 → PIL → numpy → cv2)</text><rect x="14" y="80" width="116" height="70" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="186" y="80" width="116" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="358" y="80" width="116" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="530" y="80" width="116" height="70" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="72" y="108" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">cv2 画像</text><text x="72" y="132" text-anchor="middle" font-size="15" font-weight="700" fill="#dc2626">BGR</text><text x="244" y="108" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">PIL Image</text><text x="244" y="132" text-anchor="middle" font-size="15" font-weight="700" fill="#2563eb">RGB</text><text x="416" y="108" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">numpy 配列</text><text x="416" y="132" text-anchor="middle" font-size="15" font-weight="700" fill="#2563eb">RGB</text><text x="588" y="108" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">cv2 画像</text><text x="588" y="132" text-anchor="middle" font-size="15" font-weight="700" fill="#dc2626">BGR</text><line x1="132" y1="110" x2="180" y2="110" stroke="#71717a" stroke-width="2"/><polygon points="186,110 176,105 176,115" fill="#71717a"/><line x1="304" y1="110" x2="352" y2="110" stroke="#71717a" stroke-width="2"/><polygon points="358,110 348,105 348,115" fill="#71717a"/><line x1="476" y1="110" x2="524" y2="110" stroke="#71717a" stroke-width="2"/><polygon points="530,110 520,105 520,115" fill="#71717a"/><text x="159" y="96" text-anchor="middle" font-size="11.5" fill="#3f3f46">cvtColor</text><text x="330" y="96" text-anchor="middle" font-size="11.5" fill="#3f3f46">np.asarray</text><text x="502" y="96" text-anchor="middle" font-size="11.5" fill="#3f3f46">cvtColor</text><text x="159" y="172" text-anchor="middle" font-size="11" fill="#c2410c">BGR → RGB</text><text x="330" y="172" text-anchor="middle" font-size="11" fill="#71717a">そのまま</text><text x="502" y="172" text-anchor="middle" font-size="11" fill="#c2410c">RGB → BGR</text><polyline points="588,150 588,272 72,272 72,156" fill="none" stroke="#16a34a" stroke-width="2"/><polygon points="72,150 67,160 77,160" fill="#16a34a"/><text x="330" y="262" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">1 周して画素が完全一致（変換が正しく閉じた）</text></svg><figcaption><b>cv2(BGR) → PIL(RGB) → numpy(RGB) → cv2(BGR)</b> と橋渡ししても、境界ごとに <code>cvtColor</code> を入れれば色は崩れません。<b>cv2 は BGR</b>（橙）、<b>PIL / numpy は RGB</b>（青）で、外へ出す/戻すたびに <code>BGR2RGB</code> / <code>RGB2BGR</code> を挟みます。最後に <code>np.array_equal(bgr, back)</code> が <b>True</b> なら、3 者の変換が正しく閉じた証拠です。</figcaption></figure>

## 10. headless 表示 — `cv2.imshow` の罠と安全な代替

最後は「結果をどう見るか」です。チュートリアルでよく出てくる `cv2.imshow(name, img)` + `cv2.waitKey(0)` は、**GUI バックエンド（GTK や Qt）が動く環境でしか使えません**。Docker コンテナ、SSH 越しのサーバ、CI、そして多くの「ディスプレイの無い」環境では、ウィンドウを開けずにエラーになるか、最悪の場合 **Qt が `try/except` でも捕まえられない強制終了（abort）を起こしてプロセスごと落ちます**。本講座が CPU・headless 前提である以上、`imshow` を既定の表示手段にしてはいけません（そもそも `opencv-python-headless` には `imshow` 自体が含まれません）。

そこで本講座の方針は明快です。**結果は「画面に出す」のではなく「`outputs/` に保存して後で見る」**。保存手段は2つあり、`cv2.imwrite`（BGR のまま渡せる・最も手軽）と、matplotlib です。matplotlib を使うときは、**import の前に** `matplotlib.use("Agg")` で画面不要の Agg バックエンドに固定するのが肝心で、これで `DISPLAY` が無くても `savefig` が確実に動きます。matplotlib は RGB 前提なので、`imshow` に渡す前に `BGR2RGB` を忘れないでください。

```python
import matplotlib
matplotlib.use("Agg")          # ← import pyplot より前に！画面不要にする
import matplotlib.pyplot as plt

cv2.imwrite("out.png", bgr)    # 方法A: そのまま保存（BGR）
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
plt.imsave("out2.png", rgb)    # 方法B: matplotlib で保存（RGB に変換してから）
```

どうしてもローカル GUI で見たいときのために、`04_display_headless.py` は `cv2.imshow` を**二重のガード**で囲っています ——「環境変数 `CV_SHOW=1` を明示したか」と「実際に `DISPLAY` / `WAYLAND_DISPLAY` があるか」。後者を**呼ぶ前に**確認するのが決定的に重要で、abort は例外として捕まえられない以上、「危ない環境ではそもそも呼ばない」しか安全策が無いからです。この設計のおかげで、headless で誤って `CV_SHOW=1` を付けてもプロセスは落ちず、安全にスキップされます。

## 11. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば理解が積み上がるように並べています。すべて `outputs/01_image_basics/` に結果を保存し、画面表示には依存しません。共通処理（合成画像生成・日本語パス対応 I/O・出力先管理）は `cv_helpers.py` にまとめ、各スクリプトはそれを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `cv_helpers.py` | 合成画像生成・`imread_unicode`/`imwrite_unicode`・`None`チェック・出力先。各スクリプトが import する道具箱 |
| `01_imread_imwrite.py` | ndarray の素性、`IMREAD_*` フラグ、`None` 戻り値、日本語パス、保存形式と品質、アルファPNG |
| `02_bgr_rgb_pitfall.py` | BGR↔RGB の食い違いを目で確認、`cvtColor`、`uint8` オーバーフロー vs `cv2.add` 飽和 |
| `03_pillow_numpy_interop.py` | Pillow の `size`/`mode`、PIL↔numpy↔cv2 ラウンドトリップ、`resize`/`crop`/`rotate`/`ImageDraw`/filter |
| `04_display_headless.py` | `imshow` の罠、matplotlib(Agg) と `imwrite` での保存、`CV_SHOW`/`DISPLAY` 二重ガード、コンタクトシート |
| `mini_project.py` | **章末ミニプロジェクト**。この回の全要素を統合した「画像 I/O ＆ 色順サニティ・ツールキット」 |
| `exercises.py` | TODO 形式の演習10問（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 演習の完全な模範解答（実行すると全10問 PASS） |

表の通り、`cv_helpers.py` だけは「読み物」ではなく「再利用する道具」です。中身も豊富にコメントしてあるので、最初に一読してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。

---

## 🛠 章末ミニプロジェクト — 画像 I/O ＆ 色順サニティ・ツールキット（`mini_project.py`）

この回で学んだ要素を1本に統合する総合課題です。「1枚の画像を入口に、自分一人で・画面が無くても・確実に I/O と色順を検証できる小さな健康診断ツール」を完成させます。`mini_project.py` を実行すると、`get_sample_bgr()` で得た画像（`data/sample.jpg` があればそれを優先）に対して、次の検証を一気通貫で行います。

1. **素性の観察** — `shape` / `dtype` / 値域 / カラーかグレーかを JSON 化（`inspect_image`）。
2. **3経路の正しい保存** — `cv2.imwrite`（BGR のまま）/ Pillow（RGB へ変換）/ matplotlib（RGB へ変換）で保存し、わざと変換し忘れた失敗例（`mini_pil_wrong.png`）も並べる（`save_three_libraries`）。
3. **ラウンドトリップ検証** — `cv2(BGR)→PIL(RGB)→numpy→cv2(BGR)` が完全一致するか（`verify_roundtrip`）。
4. **日本語パス検証** — `ミニ_日本語パス.png` で保存→読み戻しが画素一致で往復できるか（`verify_unicode_io`）。
5. **オーバーフロー実演** — 明るさ `+120` を numpy `+`（巻き戻り）と `cv2.add`（飽和）で比較し、数値と画像で示す（`overflow_demo`）。
6. **比較パネル** — 上の要点を2行×3列の1枚（`mini_panel.png`）にまとめ、JSON とテキストのレポートも出力する。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="章末ミニプロジェクトの6ステップ。素性の観察→3経路で保存→往復検証→日本語パスIO→あふれ実演→比較パネル出力の順に一気通貫で流れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — ① から ⑥ までを一気通貫で検証</text><rect x="24" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="214" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="214" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="24" y="214" width="180" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="114" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 素性の観察</text><text x="114" y="112" text-anchor="middle" font-size="11" fill="#71717a">inspect_image</text><text x="330" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② 3 経路で保存</text><text x="330" y="112" text-anchor="middle" font-size="11" fill="#71717a">save_three_libraries</text><text x="546" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 往復検証</text><text x="546" y="112" text-anchor="middle" font-size="11" fill="#71717a">verify_roundtrip</text><text x="546" y="242" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">④ 日本語パス I/O</text><text x="546" y="262" text-anchor="middle" font-size="11" fill="#71717a">verify_unicode_io</text><text x="330" y="242" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">⑤ あふれ実演</text><text x="330" y="262" text-anchor="middle" font-size="11" fill="#71717a">overflow_demo</text><text x="114" y="242" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">⑥ 比較パネル出力</text><text x="114" y="262" text-anchor="middle" font-size="11" fill="#71717a">mini_panel.png</text><line x1="206" y1="96" x2="234" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="240,96 230,91 230,101" fill="#71717a"/><line x1="422" y1="96" x2="450" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="456,96 446,91 446,101" fill="#71717a"/><line x1="546" y1="130" x2="546" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="546,214 541,204 551,204" fill="#71717a"/><line x1="454" y1="246" x2="426" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="420,246 430,241 430,251" fill="#71717a"/><line x1="238" y1="246" x2="210" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="204,246 214,241 214,251" fill="#71717a"/></svg><figcaption><b>章末ミニプロジェクト</b>の検証は、入口の 1 枚から <b>① 素性の観察 → ② 3 経路で保存 → ③ 往復検証 → ④ 日本語パス I/O → ⑤ オーバーフロー実演 → ⑥ 比較パネル出力</b> の順に一気通貫で流れます。① から ⑤ までが個別の検証で、最後の <b>⑥</b> がそれらを 1 枚の <code>mini_panel.png</code> と JSON/テキストのレポートへ束ねる出力ステップです。</figcaption></figure>

```
=== 画像 I/O ＆ 色順サニティ・ツールキット ===
  画像: shape=[240, 320, 3] dtype=uint8 min/max=0/255 color=True
  検証:
    [OK] roundtrip_equal
    [OK] pil_color_preserved
    [OK] unicode_io_equal
  uint8 +120: numpy+=[119, 119, 119] (巻き戻り) / cv2.add=[255, 255, 255] (飽和)
  全検証パス: True
```

ねらいは「**この章の地雷（色順・次元・型）をすべて自動で踏み抜いて、踏まずに済む書き方を1つのツールに固める**」こと。出力は `outputs/01_image_basics/mini_panel.png` ／ `mini_report.json` ／ `mini_report.txt` に保存されます。`mini_panel.png` を開いて、「正しい色 / 赤青が入れ替わった失敗例 / グレー / オーバーフロー / 飽和 / R チャンネル」を目で見比べてください。

---

## ✅ 到達チェックリスト

自分の言葉で説明でき、AI 補助なしでコードを書ける状態を目指します。

- [ ] 画像が `(H, W, 3)` の `uint8` numpy 配列であることを、`shape` / `dtype` / 値域の観点で説明できる。
- [ ] `img[y, x]` が「行（何行目 y）→列（何列目 x）」の順であり、`img[x, y]` ではないと説明できる。
- [ ] OpenCV の画素並びが **BGR**（R は index 2）であることを言え、`img[:, :, 2]` が R だと分かる。
- [ ] `cv2.imread` の `IMREAD_COLOR` / `GRAYSCALE` / `UNCHANGED` で戻り値の次元が変わることを説明できる。
- [ ] `cv2.imread` が失敗時に **例外でなく `None`** を返すことを知り、読み込み直後の `None` チェックを書ける。
- [ ] 日本語(非ASCII)パスを `np.fromfile`+`imdecode` / `imencode`+`tofile` で安全に読み書きできる。
- [ ] **BGR↔RGB** の食い違いを説明でき、PIL / matplotlib へ渡す前に `cv2.cvtColor(BGR2RGB)` を入れられる。
- [ ] グレースケールが `(H, W)` の2次元であること、`GRAY2BGR` で3chへ戻せることを使い分けられる。
- [ ] `uint8` の `numpy +`（巻き戻り）と `cv2.add`（飽和）の違いを説明し、用途で選べる。
- [ ] 広い型 + `np.clip` で飽和を手作りでき、`cv2.add` と一致させられる。
- [ ] `cv2(BGR)→PIL(RGB)→numpy→cv2(BGR)` のラウンドトリップが一致することを確認できる。
- [ ] `cv2.imshow` が headless で危険な理由を説明し、`imwrite` / matplotlib(Agg) 保存に切り替えられる。

---

## ❓ よくある落とし穴・FAQ・デバッグ

まず「症状 → 原因 → 対処」の早見表です。実装中に詰まったら最初にここを見てください。第1回でつまずく原因は、ほぼこの6つに集約されます。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `AttributeError: 'NoneType' object has no attribute 'shape'` | `imread` が `None` を返している（パス間違い・ファイル破損） | 読み込み直後に `None` チェック。`load_bgr_checked` を使う |
| 赤と青が入れ替わって見える | BGR の配列を RGB として渡した | 外部へ渡す前に `cv2.cvtColor(BGR2RGB)` |
| 明るくしたら一部が暗くなった/ノイズが出た | `uint8` オーバーフロー（numpy の `+`） | `cv2.add` を使う、または広い型で計算して `clip` |
| `(240, 320)` と `(240, 320, 3)` で形が合わず落ちる | グレー(2次元)とカラー(3次元)の取り違え | `cvtColor(GRAY2BGR)` で揃える。`shape` を確認 |
| 日本語パスで読み書きが失敗する | `imread`/`imwrite` が非ASCIIパス非対応 | `np.fromfile`+`imdecode` / `imencode`+`tofile` |
| `imshow` でフリーズ/プロセスごと落ちる | headless 環境で GUI バックエンドが無い | `imwrite`/matplotlib(Agg) で保存。`DISPLAY` を確認 |

**Q. `module 'cv2' has no attribute 'imshow'` と言われた**
A. `opencv-python-headless` を入れているからです。これは仕様（GUI 機能を除いた軽量版）で、本講座は headless 前提です。表示は `cv2.imwrite` か matplotlib(Agg) の保存に切り替えてください。`opencv-python`（full）と headless を**同時にインストールしてはいけません**（`cv2` 名前空間が衝突します）。

**Q. `cv2.add(img, 100)` が思った色にならない**
A. スカラー `100` は**第0チャンネル（B）だけ**に足されます。全チャンネルへ一様に足したいときは `cv2.add(img, np.full_like(img, 100))` のように配列で渡すか、`(100, 100, 100, 0)` のように4要素のスカラータプルを渡してください。

**Q. `Image.fromarray(arr)` で `TypeError` / 色がおかしい**
A. (1) `arr` の `dtype` が `uint8` でない（`float64` のまま）と失敗します。`arr.astype(np.uint8)`（必要なら `clip` 後）にしてください。(2) `dtype` が合っていても色が変なら、BGR を渡している可能性大。`cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` してから渡します。

**Q. `np.asarray(pil)` を書き換えようとしたら `ValueError: assignment destination is read-only`**
A. `np.asarray` は多くの場合コピーを作らず読み取り専用ビューを返します。書き換えたいときは `np.array(pil)`（コピー）を使うか `.copy()` してください。

**Q. `cv2.resize` のサイズ指定で縦横が逆になった**
A. `cv2.resize(img, (W, H))` の `dsize` は **(幅, 高さ)** の順で、numpy の `shape=(H, W)` とは逆です。第3回で詳しく扱いますが、「PIL の `size` と cv2 の `dsize` は (W, H)、numpy の `shape` は (H, W)」と覚えてください。

**Q. matplotlib で `savefig` しても画像が出ない / エラーになる**
A. `import matplotlib.pyplot` より**前に** `matplotlib.use("Agg")` を呼べているか確認してください。順序が逆だと、DISPLAY の無い環境で別のバックエンドが選ばれて失敗します。

**Q. JPEG を編集して保存し直すたびに画質が落ちる**
A. JPEG は非可逆圧縮なので、開く→保存を繰り返すと劣化が蓄積します。加工途中は PNG（可逆）で持ち回り、最後だけ JPEG にするのが鉄則です。

---

## 🚀 発展トピック・参考

- **値域の正規化**: 深層学習の前処理では `uint8`(0–255) を `float32`(0.0–1.0、さらに mean/std で標準化) へ変換します。`img.astype(np.float32) / 255.0` が基本形。逆に保存時は `(x*255).clip(0,255).astype(np.uint8)` で戻します（第12回で本格的に扱います）。
- **チャンネルの軸位置（HWC vs CHW）**: OpenCV/PIL は `(H, W, C)` ですが、PyTorch のテンソルは `(C, H, W)`。`np.transpose(img, (2,0,1))` や `torch.from_numpy(img).permute(2,0,1)` で並べ替えます。色順(BGR/RGB)と軸順(HWC/CHW)は**別の問題**なので、両方を区別して意識してください。
- **EXIF Orientation**: スマホ写真は画素を回さず「向き情報」だけを EXIF に持つことがあり、無視すると横倒しで読まれます。`PIL.ImageOps.exif_transpose` で正規化します（第3回で扱います）。
- **16bit / float 画像**: 医療・HDR・深度などでは `uint16` や `float32` の画像も登場します。`uint8` 前提の表示・保存コードはそのままだと破綻するので、`dtype` を必ず確認する癖を。
- **ICC プロファイル・色管理**: 厳密な色再現が要る用途では sRGB 等のカラープロファイルも絡みますが、本講座の範囲では「BGR/RGB の入れ替え」に集中すれば十分です。
- 公式ドキュメント:
  - OpenCV-Python チュートリアル: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
  - Pillow（PIL）ハンドブック: https://pillow.readthedocs.io/en/stable/
  - NumPy 配列の基礎: https://numpy.org/doc/stable/
  - Matplotlib（画像表示）: https://matplotlib.org/stable/

---

## 12. 動かし方

このモジュールは `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` だけに依存し、GPU もネット接続も不要です。サンプル画像が無くても合成画像が自動生成されるので、いきなり実行できます（`data/sample.jpg` を置けば、そちらが優先して使われます）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ）
uv sync

# 各スクリプトを実行（結果は outputs/01_image_basics/ に保存される）
uv run python lectures/01_image_basics/01_imread_imwrite.py
uv run python lectures/01_image_basics/02_bgr_rgb_pitfall.py
uv run python lectures/01_image_basics/03_pillow_numpy_interop.py
uv run python lectures/01_image_basics/04_display_headless.py

# 章末ミニプロジェクト（統合課題。図/JSON/テキストを outputs/ に出す）
uv run python lectures/01_image_basics/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/01_image_basics/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る（全 PASS）
uv run python lectures/01_image_basics/exercises_solutions.py
#   または: SHOW_SOLUTION=1 uv run python lectures/01_image_basics/exercises.py

# （任意）ローカルにGUIがある場合のみ、ウィンドウ表示を試す
CV_SHOW=1 uv run python lectures/01_image_basics/04_display_headless.py
```

実行後は `outputs/01_image_basics/` に生成された PNG/JPEG を画像ビューアで開き、解説と照らし合わせてください。特に `02_pil_wrong_swapped.png`（赤青が入れ替わった失敗例）と `02_pil_correct.png`（正しい色）、`02_matplotlib_compare.png`（左右比較）を見比べると、BGR/RGB の食い違いが一目で腑に落ちます。`02_overflow_numpy_plus.png` と `02_overflow_cv2_add.png` の違いからは、オーバーフローと飽和の差を視覚的に確認できます。仕上げに `mini_panel.png` を開けば、この章の要点が1枚に集約されています。

## 13. まとめ

この章では、画像が `(H, W, 3)` の `uint8` numpy 配列であるという根本から出発し、OpenCV/Pillow の入出力、`IMREAD_*` フラグ、`None` 戻り値、日本語パス、そして最重要の BGR/RGB の食い違い、グレースケールの次元、`uint8` オーバーフロー、PIL↔numpy↔cv2 の相互変換、headless での安全な表示までを、すべて「自分で再現し回避できる」レベルで扱いました。派手さはありませんが、ここが今後すべての画像処理の地盤になります。

次回以降は、この地盤の上に色空間変換・閾値処理・フィルタ・幾何変換などを積み上げていきます。本章の `cv_helpers.py` のような「自分の手に馴染んだ I/O ヘルパ」を一つ持っておくと、以降の学習でいちいち定型処理に煩わされずに済みます。まずは演習を自力で全問 PASS させ、章末ミニプロジェクトの全検証を緑にしてから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4（2.4.6）／ opencv-python-headless 4.13（`cv2` 4.13.0）／ Pillow 12.2（12.2.0）／ matplotlib 3.10（3.10.9）。
> 本章は torch を使いません（torch を使う回は 2.12+cpu を前提）。すべて CPUのみ・合成データ・ネット不要で動作し、結果は `outputs/01_image_basics/` に保存します（画面表示には依存しません）。
