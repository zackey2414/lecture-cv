# 第4回 フィルタ・エッジ・閾値・モルフォロジー・輪郭・ワーピング

> トラック: **画像の基礎** ／ レベル: **中級** ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch / faiss は使いません）

## 🎯 この章のゴール

この章のゴールは、**「二値化 → モルフォロジー → 輪郭 → ワーピング」という古典CVの前処理連鎖を、AI 補助なしで自分の手で組めるようになる**ことです。深層学習がどれだけ強力になっても、その入口に置く前処理 —— 平滑化でノイズを均し、エッジや閾値で対象を浮き上がらせ、モルフォロジーで形を整え、輪郭から形状を測り、幾何変換でまっすぐに起こす —— は今も古典CVの独壇場です。ここを「ライブラリの関数を順に呼ぶだけ」ではなく「なぜその順番か」「なぜそのフィルタか」を説明できる状態にするのが目標です。

具体的な到達点は2つです。1つ目は、**フィルタ・エッジ・閾値の効果を並べて比較できる**こと。同じノイズでもガウスとごま塩で効くフィルタが違うこと、Sobel の出力に `CV_64F` を使う理由、固定しきい値が照明ムラで破綻し適応的閾値が救うこと —— これらを数値と画像の両方で確認します。2つ目は、このモジュールの完成物である、**傾いて撮影された書類を正面のスキャン画像へ自動補正するスクリプト**を書き上げることです。

そしてこの章には、初学者が必ず一度は踏む地雷が埋まっています。最大のものが **`cv2.findContours` は OpenCV 4 系で返り値が `(contours, hierarchy)` の「2つ」**である点（3系の「3つ返し」サンプルをコピペすると `ValueError` になる）。ほかにも `Sobel` の `CV_8U` で負の勾配が消える罠、OpenCV の HSV スケールの独自性などを、知識ではなく「自分で再現して回避できる」レベルで潰します。すべて CPU のみ・ネット非依存で、サンプル画像が無くても合成画像が自動生成されるので、いきなり動かせます。

---

## 1. なぜ「前処理連鎖」を自力で書くのか

画像処理のパイプラインは、ほとんどの場合いくつかの単純な操作の連鎖でできています。たとえば「紙の書類を撮った写真からテキスト領域を切り出す」なら、グレースケール化 → 平滑化 → 二値化 → モルフォロジーで整形 → 輪郭抽出 → 透視変換、という流れになります。一つひとつは数行ですが、**どの操作を、どんなパラメータで、どの順に置くか**で結果がまるで変わります。この章は、その「組み立ての勘所」を体に入れるためのものです。

重要なのは、各操作が「画像という `(H, W)` あるいは `(H, W, 3)` の `uint8` 配列」に対する変換にすぎない、という第1回からの視点を保つことです。フィルタは近傍の重み付き平均、エッジは微分、二値化はしきい値での 0/255 振り分け、モルフォロジーは集合演算、ワーピングは座標の写像 —— どれも配列を別の配列に移すだけです。この視点があると、関数の戻り値の `shape` と `dtype` を見ただけで「何が起きたか」を推測でき、デバッグが一気に速くなります。

本章のスクリプトはすべて、結果を画面に出さず `outputs/04_filtering_edges_morphology/` に保存します。`cv2.imshow` は GUI バックエンドを必要とし、Docker・SSH・CI などディスプレイの無い環境ではプロセスごと落ちることがあるため、headless 前提の本講座では使いません。比較は1枚のグリッド画像にまとめて保存するので、実行後にそれを開いて解説と照らし合わせてください。

## 2. 平滑化フィルタ — どのノイズにどれが効くか（`01_smoothing.py`）

平滑化（スムージング）は、近傍画素の重み付き平均で画素のばらつきを均す処理です。`cv2.blur` は単純な箱型平均、`cv2.GaussianBlur` は中心ほど重みが大きいガウス分布の重み、`cv2.medianBlur` は近傍の「中央値」を取り、`cv2.bilateralFilter` は「空間的な近さ」と「輝度の近さ」の両方で重みを決めます。`cv2.filter2D` はこれらの一般形で、自前のカーネルを渡せば平均化も先鋭化（シャープ化）も同じ枠組みで書けます。

肝心なのは「万能のフィルタは無い」という点です。`01_smoothing.py` は、同じ合成画像に**ガウスノイズ**（センサノイズ相当の細かいゆらぎ）と**ごま塩ノイズ**（画素が突発的に 0 か 255 に化ける）の2種類を加え、各フィルタの結果を、クリーン画像との平均二乗誤差（MSE、小さいほど元に近い）で数値比較します。実行すると、ガウスノイズには `bilateralFilter`（エッジを保ちつつ除去）や `GaussianBlur` が、ごま塩ノイズには `medianBlur` が圧倒的に強い、という結果がはっきり出ます。

```python
g_gauss  = cv2.GaussianBlur(noisy, (5, 5), sigmaX=0)   # ksize は奇数。sigmaX=0 なら ksize から自動
g_median = cv2.medianBlur(noisy, 5)                    # ごま塩に強い（中央値が外れ値を無視）
g_bilat  = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)  # エッジ保持・やや重い
sharp    = cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32))  # 先鋭化
```

上のコードで `GaussianBlur` / `medianBlur` の `ksize` が**奇数**でなければならない点に注意してください（中心画素を定めるため）。`bilateralFilter` は `sigmaColor` を小さくするとエッジ近傍を残し、大きくすると普通のぼかしに近づきます。`filter2D` のカーネルは総和を 1 にしておくと明るさが変わりません。出力の `01_smoothing_gaussian_noise.png` と `01_smoothing_salt_pepper.png` を見比べ、ごま塩で `medianBlur` だけが粒をきれいに消していることを目で確認してください。

## 3. エッジ検出 — Sobel / Laplacian / Canny と `CV_64F` の理由（`02_edges_canny.py`）

エッジは「輝度が急に変わる場所」で、数学的には微分（勾配）の大きいところです。`cv2.Sobel` は1次微分で、`dx=1, dy=0` なら横方向の勾配（縦エッジ）、`dx=0, dy=1` なら縦方向の勾配（横エッジ）を返します。`cv2.Laplacian` は2次微分で、エッジを符号反転点として捉えます。そして `cv2.Canny` は、平滑化 → 勾配計算 → 非最大抑制（細線化）→ ヒステリシス閾値、という多段アルゴリズムで、最も実用的な「きれいな1画素幅のエッジ」を出します。

ここでの最重要ポイントが**出力の `dtype`** です。勾配には符号があり、黒→白のエッジは正、白→黒のエッジは負の値になります。`Sobel` の `ddepth` に `cv2.CV_8U`（0〜255）を指定すると、**負の勾配がすべて 0 にクリップされ、エッジの半分が消えます**。だから `cv2.CV_64F`（符号付き浮動小数）で計算してから、`cv2.convertScaleAbs`（絶対値を取り 0〜255 にスケール）で可視化するのが定石です。`02_edges_canny.py` は「左黒・中白・右黒」の帯にこれを適用し、`CV_8U` では片側（240画素）しか出ず `CV_64F` では両側（480画素）出ることを数値で示します。

```python
sob_8u  = cv2.Sobel(step, cv2.CV_8U,  1, 0, ksize=3)              # 負の勾配が消える（片側だけ）
sob_64f = cv2.Sobel(step, cv2.CV_64F, 1, 0, ksize=3)             # 符号付きで両方向を保持
edge    = cv2.convertScaleAbs(sob_64f)                            # |値| を 8bit に → 可視化
canny   = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)  # 前段ぼかし→2しきい値
```

`Canny` の2つのしきい値は、「高い方を超えたら確実にエッジ」「低い方を下回ったら確実に非エッジ」「間は強いエッジに繋がっていれば採用」というヒステリシスを表します。`02_canny_thresholds.png` の出力では、**前段に `GaussianBlur` を入れるとノイズ由来の偽エッジが激減**すること（ぼかし無し約32000画素 → 有り約2000画素）、しきい値を下げると細かいエッジが増え、上げると強いエッジだけ残ることが確認できます。「Canny の前にぼかす」は実務の鉄則です。

## 4. 閾値処理 — 固定 / Otsu / 適応的（`03_threshold_morphology.py` 前半）

二値化は「しきい値より明るいか暗いか」で画素を 0 / 255 に振り分ける操作で、後段のモルフォロジーや輪郭抽出の入口です。`cv2.threshold` に固定値（例 127）を渡すのが最も単純ですが、照明ムラのある実画像ではすぐ破綻します。`THRESH_OTSU` を併用すると、ヒストグラムから前景と背景を最もよく分けるしきい値を**自動で1つ**選んでくれます。ただし Otsu も「画像全体で1つの値」を使う大域的な手法なので、片側が暗いような照明ムラには弱いままです。

それを救うのが `cv2.adaptiveThreshold` です。これは画素ごとに、その近傍（`blockSize` 四方）の平均や加重平均からローカルにしきい値を決めるため、左が明るく右が暗いような画像でも、文字だけをきれいに拾えます。`03_threshold_morphology.py` は、右へ行くほど暗くなる照明ムラ付きの「文書」を合成し、固定・Otsu・適応的の3手法を並べます。実行すると、固定と Otsu では右側の暗い紙まで「インク」と誤判定して白く潰れるのに対し、適応的だけが文字を正しく分離することが一目で分かります。

```python
_, fixed = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY_INV)                 # 固定（照明ムラに弱い）
t, otsu  = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # 自動・ただし大域的
adapt    = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=31, C=10)     # 局所・照明ムラに強い
```

`THRESH_BINARY_INV` は「暗い画素（インク）を 255 にする」反転版で、文書処理では前景を白にしておくと後段が楽になります。`adaptiveThreshold` の `blockSize` は**奇数**で、大きいほど広い範囲を見て緩やかに、小さいほど細かく反応します。`C` は計算したしきい値からさらに引く補正値です。出力 `03_threshold_compare.png` で、適応的閾値の頑健さを確認してください。

## 5. モルフォロジー演算 — 形を整える集合演算（`03_threshold_morphology.py` 後半）

二値化した直後の画像は、たいてい汚れています。背景に白い粒ノイズが残ったり、前景の中に黒い小穴が空いたりします。これを整えるのがモルフォロジー演算です。基本は2つ —— **収縮（erode）** は白を痩せさせ（細い橋や小さな粒を消す）、**膨張（dilate）** は白を太らせます（小穴を埋める）。この2つを組み合わせたのが、**オープニング（open = 収縮→膨張）** で白い粒ノイズを除去し、**クロージング（close = 膨張→収縮）** で黒い穴を埋めます。ほかに輪郭線を出す勾配（gradient）や、細部を抽出するトップハット（tophat）があります。

どの近傍を「近い」とみなすかは、`cv2.getStructuringElement` で作る**構造要素（カーネル）**が決めます。同じ 5×5 でも、矩形（`MORPH_RECT`）は全マス、楕円（`MORPH_ELLIPSE`）は丸く、十字（`MORPH_CROSS`）は縦横だけ、と「見る形」が変わります。物体の形に合わせて選ぶのがコツで、丸い対象には楕円がよく馴染みます。`03_threshold_morphology.py` は、粒ノイズと小穴を含む二値画像にこれらを適用し、効果を**白の連結成分数**で定量化します。

```python
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))      # 楕円の構造要素
opened  = cv2.morphologyEx(blobs, cv2.MORPH_OPEN,  kernel)          # 白い粒ノイズを除去
closed  = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, kernel)          # 黒い小穴を埋める
grad    = cv2.morphologyEx(blobs, cv2.MORPH_GRADIENT, kernel)       # dilate - erode = 輪郭線
```

実行ログでは、オープニング前に **877個**もあった白の連結成分（粒ノイズが大量にカウントされている）が、オープニング後には本来の **3個**（円2つ＋矩形1つ）まで減ります。この「数が正しくなる」感覚が、後段の輪郭抽出や物体カウントの精度に直結します。出力 `03_morphology.png` で、open が粒を消し close が穴を埋める様子を見比べてください。

## 6. 輪郭抽出と形状解析 — findContours は4系で「2返し」（`04_contours_warp.py` 前半）

整った二値画像から物体の境界線を取り出すのが `cv2.findContours` です。ここで**この章で最も多い事故**が起きます。OpenCV 4 系の `findContours` は返り値が **`(contours, hierarchy)` の2つ**ですが、ネット上には3系時代の「`image, contours, hierarchy` の3つ返し」のサンプルが大量に残っており、それをそのまま使うと `ValueError: not enough values to unpack` で落ちます。本講座は 4 系前提なので、必ず2つで受けてください。

得られた輪郭（点列）からは、さまざまな形状量が測れます。`cv2.contourArea`（面積）、`cv2.arcLength`（周囲長）、`cv2.boundingRect`（軸並行の外接矩形）、`cv2.convexHull`（凸包）、そして `cv2.approxPolyDP`（輪郭を少ない頂点の多角形に近似）です。とくに `approxPolyDP` は、書類のような四角い対象を「4つの角」に要約するのに使え、次節の透視変換の前段になります。許容誤差 `epsilon` を周囲長に比例させる（`0.02 * arcLength` など）のが定石です。

```python
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # ★2つ！
largest = max(contours, key=cv2.contourArea)            # 面積最大の輪郭を選ぶ
peri    = cv2.arcLength(largest, True)                  # True = 閉じた輪郭
approx  = cv2.approxPolyDP(largest, 0.02 * peri, True)  # 4点に近似できれば len(approx)==4
x, y, w, h = cv2.boundingRect(largest)                  # 外接矩形
```

`RETR_EXTERNAL` は最も外側の輪郭だけを取り（内部の穴や文字を無視する）、`CHAIN_APPROX_SIMPLE` は直線部分の中間点を省いて頂点だけ残すモードです。`04_contours_warp.py` を実行すると、合成した書類について面積・周囲長・外接矩形・凸包の点数・近似頂点数が表示され、`approxPolyDP` がきっちり **4頂点**を返すことが確認できます。

## 7. 透視変換で書類をまっすぐにする（`04_contours_warp.py` 後半・完成物）

いよいよこのモジュールの完成物です。傾いて撮影された書類を正面のスキャン画像へ起こすには、**透視変換（射影変換）** を使います。アフィン変換（回転・拡縮・せん断・平行移動）が平行線を平行のまま保つのに対し、透視変換は「奥が狭く手前が広い」遠近を表現でき、台形を長方形に補正できます。`cv2.getPerspectiveTransform` に「変換元の4点」と「変換先の4点」を渡すと 3×3 の変換行列が得られ、`cv2.warpPerspective` でそれを画像に適用します。

連鎖はこうです。グレースケール化 → 平滑化 → Otsu 二値化（明るい紙を白に）→ `findContours` で最大輪郭 → `approxPolyDP` で4隅 → 4隅を「左上・右上・右下・左下」に並べ替え → 出力長方形の4隅へ `getPerspectiveTransform` → `warpPerspective`。4隅の並べ替えは、座標の**和 `x+y`（最小=左上・最大=右下）と差 `x-y`（最大=右上・最小=左下）**で判定するのが定番のテクニックです。順序を間違えると、まっすぐ化どころか上下や鏡像が反転します。

```python
def order_corners(pts):                      # 4点を TL, TR, BR, BL の順へ
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.float32([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])

dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])  # 正面の長方形
M = cv2.getPerspectiveTransform(quad, dst)   # 元の4隅 → 正面の4隅
front = cv2.warpPerspective(photo, M, (W, H))  # 傾いた写真を正面化
```

出力サイズ `(W, H)` は、検出した4隅の対辺の長さの大きい方から決めると、つぶれずに自然な比率で起こせます。`04_contours_warp.py` を実行すると、`04_document_pipeline.png` に「傾いた入力 → 二値化 → 輪郭と4隅 → 正面化 → （比較用に）アフィン回転」が一列で並びます。とくに `04_doc_straightened.png` が、斜めから撮った "INVOICE" が正面のスキャン画像へ起こされた最終成果物です。同じ枠組みで `cv2.getRotationMatrix2D` + `cv2.warpAffine` のアフィン回転も並置しているので、両者の性質の違いも体感できます。

## 8. ヒストグラム平坦化と CLAHE — コントラストを救う（`05_histogram_clahe.py`）

最後はコントラスト補正です。暗かったり霞んだりした低コントラスト画像は、ヒストグラム（各輝度値の画素数の分布）が狭い範囲に固まっています。これを広げて見やすくするのが平坦化です。`cv2.equalizeHist` は分布を画像全体で一様に引き伸ばす**大域的**な手法で、手軽ですが、明るい部分と暗い部分が混在する画像では一方が飽和したり不自然になりがちです。`cv2.calcHist` で補正前後の分布を可視化すると、狭かった山が横に広がる様子が分かります。

そこで実務でよく使うのが **CLAHE（Contrast Limited Adaptive Histogram Equalization）** です。`cv2.createCLAHE` で作り、画像を小さなタイルに分けてタイルごとに平坦化し、`clipLimit` でコントラストの増幅しすぎ（＝ノイズの強調）を抑えます。これにより、照明ムラのある画像でも、明るい所も暗い所も**局所的に**自然な見え方へ補正できます。`05_histogram_clahe.py` は、低コントラスト＋片側が暗い合成画像に対し、`normalize`（単純な min-max ストレッチ）・`equalizeHist`・CLAHE を並べ、輝度の標準偏差で広がり方を比較します。

```python
eq    = cv2.equalizeHist(gray)                                  # 大域的に最大限引き伸ばす
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))     # タイルごと・過増幅を clipLimit で抑制
cl    = clahe.apply(gray)
# カラーは BGR 各 ch を別々に平坦化すると色が崩れる → HSV の V（明度）だけに適用するのが正解
hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
h_, s_, v_ = cv2.split(hsv)
out = cv2.cvtColor(cv2.merge([h_, s_, clahe.apply(v_)]), cv2.COLOR_HSV2BGR)
```

カラー画像で気をつけたいのが、**B/G/R チャンネルを別々に平坦化すると色相がずれて色が崩れる**ことです。正しくは、いったん HSV に変換し、明度 `V` だけを平坦化して色相 `H`・彩度 `S` を保ちます。なお OpenCV の HSV は `H` が 0〜179（0〜360 ではない）・`S`/`V` が 0〜255 という独自スケールなので、他ツールの色相角をそのまま入れない点も覚えておきましょう。出力 `05_color_value_channel.png` で、per-BGR 平坦化の不自然な色変化と、HSV-V 平坦化の自然さを見比べてください。

## 9. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「フィルタ → エッジ → 閾値・モルフォロジー → 輪郭・ワーピング → コントラスト補正」と理解が積み上がるよう並べています。すべて結果を `outputs/04_filtering_edges_morphology/` に保存し、画面表示には依存しません。サンプル画像は各スクリプト内で `numpy`/`cv2` により合成生成するため、`data/` に何も無くても動きます（外部モジュールからの import もありません＝自己完結）。

| ファイル | 役割（単一責務） |
| --- | --- |
| `01_smoothing.py` | 平滑化の比較。`blur`/`GaussianBlur`/`medianBlur`/`bilateralFilter`/`filter2D` を、ガウス・ごま塩ノイズに対し MSE で評価 |
| `02_edges_canny.py` | `Sobel`/`Laplacian`/`Canny`。`CV_8U` で負の勾配が消える罠、`CV_64F`+`convertScaleAbs`、Canny の前段ぼかしと2しきい値 |
| `03_threshold_morphology.py` | 固定/Otsu/適応的の二値化、`getStructuringElement`、`erode`/`dilate`/`morphologyEx`（open/close/gradient/tophat）|
| `04_contours_warp.py` | **基礎完成物**。`findContours`（4系2返し）・形状解析・`approxPolyDP`→`getPerspectiveTransform`/`warpPerspective` で書類まっすぐ化 |
| `05_histogram_clahe.py` | `calcHist`/`equalizeHist`/`createCLAHE`、HSV-V チャンネルでの色を壊さないコントラスト補正 |
| `mini_project.py` | **章末ミニプロジェクト**。劣化したスマホ写真 → ノイズ除去 → 四隅検出 → 透視変換 → CLAHE → 適応的閾値で「きれいなスキャン」を作る統合課題（PNG/JSON 出力）|
| `exercises.py` | TODO 形式の演習**10問**（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答）|
| `exercises_solutions.py` | 全演習の模範解答（実行で全 PASS を確認できる）|

表の通り、`04_contours_warp.py` が deliverable の中核（書類補正）で、`01`〜`03` と `05` が「フィルタ/エッジ/閾値の効果を並べて保存する比較ツール」群に対応します。まず 01 から順に動かし、各 `outputs/04_*.png` を開きながら本文を読み返すと理解が定着します。そして全要素を1本に束ねたのが `mini_project.py` です。

---

## 🛠 章末ミニプロジェクト — スマホ写真からきれいなスキャンを作る（`mini_project.py`）

この章のすべての要素を **1本のパイプラインに統合**する総合課題です。題材は実務頻出の「**スマホで斜めに撮った領収書を、正面のきれいな二値スキャンに変換する**」。`04_contours_warp.py` が「傾きを正す」ところまでだったのに対し、ミニプロジェクトでは入力をさらに劣化させ（照明ムラ＋ガウス＋ごま塩ノイズ＋低コントラスト）、出口を「読みやすい二値スキャン＋定量レポート」まで広げます。つまり **平滑化・エッジ/閾値・モルフォロジー・輪郭・透視変換・CLAHE のすべてが、欠けると破綻する形で1つの目的に奉仕している**ことを体感するのが狙いです。

パイプラインは次の7工程です。各工程が「なぜそこに居るのか」を、入力の劣化要因と対応づけて理解してください。

1. **入力合成** — 理想の書類を透視で傾け、暗い背景に置き、照明ムラ・ガウス/ごま塩ノイズ・低コントラストを乗せて「劣化したスマホ写真」を作る。
2. **ノイズ除去**（`medianBlur` → `bilateralFilter`）— ごま塩の粒は中央値で確実に潰し、残りはエッジを保つバイラテラルで均す。**ここを飛ばすと後段の二値化と輪郭が荒れる**。
3. **四隅検出**（Otsu二値化 → クロージング → `findContours` → `approxPolyDP`）— 紙を白に二値化し、文字や朱印で空く穴をクロージングで埋めてから最大輪郭を4点に近似。**4系の2返し**を体に刻む。
4. **可視化** — 検出した輪郭と4隅（番号付き）を元写真に重ねて、並べ替えが正しいか目視確認。
5. **透視変換**（`getPerspectiveTransform` → `warpPerspective`）— 4隅を正面の長方形へ起こす。出力サイズは対辺の長い方から決めて潰れを防ぐ。
6. **コントラスト補正**（`createCLAHE`）— 正面化した紙面の局所コントラストを引き上げ、薄い文字を立たせる。
7. **二値スキャン化**（`adaptiveThreshold` → `morphologyEx(OPEN)`）— 残った照明ムラに強い適応的閾値で「黒文字・白地」にし、オープニングで地の粒ノイズを掃除して完成。

実行すると `outputs/04_filtering_edges_morphology/` に4つの成果物が出ます。**`mini_project_pipeline.png`**（7工程を並べたグリッド）、**`mini_project_scan.png`**（最終スキャン単体）、**`mini_project_hist.png`**（CLAHE 前後の輝度ヒストグラム）、**`mini_project_report.json`**（四隅座標・最大輪郭面積・正面化サイズ・CLAHE 前後のコントラスト std・インク比率などの定量指標）です。レポートJSONは「検出した四隅が入力の傾きと一致しているか」「CLAHE でコントラスト std が広がったか」を数値で振り返るのに使えます。

```bash
uv run python lectures/04_filtering_edges_morphology/mini_project.py
# → mini_project_pipeline.png / mini_project_scan.png / mini_project_hist.png / mini_project_report.json
```

**腕試し（任意の発展）**: ① `make_phone_photo` の `dst`（傾き）やノイズ量を変えても四隅検出が崩れないか試す。② 工程2のノイズ除去をコメントアウトして、二値化・輪郭がどれだけ荒れるか観察する。③ 工程6の CLAHE を `equalizeHist` に置き換えて、照明ムラが残る画像での差を見る。④ 最終スキャンの読みやすさを `adaptiveThreshold` の `blockSize`/`C` で詰める。どれも「1つの工程の必然性」を逆説的に確かめる良い実験です。

## ✅ 到達チェックリスト

この章を終えたら、次のことが**できる**／**説明できる**状態になっているか確認してください。半分以上に詰まるなら、該当スクリプトをもう一度動かしながら本文を読み返すのがおすすめです。

**手を動かしてできる**

- [ ] ガウスノイズとごま塩ノイズを合成し、`GaussianBlur`/`bilateralFilter`/`medianBlur` の効きの違いを MSE で比較できる。
- [ ] `Sobel` を `CV_64F` で計算し、`magnitude` → `convertScaleAbs` で勾配強度画像を作れる。
- [ ] `Canny` の前に `GaussianBlur` を入れ、2つのしきい値を調整して狙ったエッジ量にできる。
- [ ] 固定・Otsu・適応的の3手法を書き分け、照明ムラの画像で `adaptiveThreshold` を選べる。
- [ ] `getStructuringElement` でカーネルを作り、`open`/`close` で粒ノイズ除去・穴埋めができる。
- [ ] `findContours`（**2返し**）→ 最大輪郭 → `approxPolyDP` で四角形を4点に要約できる。
- [ ] 4隅を `order_corners` で並べ替え、`getPerspectiveTransform`/`warpPerspective` で正面化できる。
- [ ] カラー画像のコントラスト補正を、HSV の **V チャンネルだけ** CLAHE して色を壊さず行える。
- [ ] `mini_project.py` を読み、各工程を取り除くと何が破綻するか実験で示せる。
- [ ] `exercises.py` の**10問を自力で全 PASS** できる。

**言葉で説明できる**

- [ ] なぜごま塩には中央値、ガウスにはガウス/バイラテラルが向くのか。
- [ ] なぜ `Sobel` を `CV_8U` で計算するとエッジの片側が消えるのか。
- [ ] Otsu が大域的で、`adaptiveThreshold` が局所的とはどういう意味か。
- [ ] `open` と `close` が「収縮と膨張のどちらを先にやるか」でなぜ効果が逆になるのか。
- [ ] OpenCV 4 系で `findContours` の返り値が2つである（3系は3つ）こと。
- [ ] アフィン変換と透視変換の違い（平行線が保たれるか／台形を長方形にできるか）。
- [ ] カラーで BGR 各チャンネルを個別に平坦化すると色が崩れる理由。

## ❓ よくある落とし穴・FAQ・デバッグ

実装中に詰まったら、まず症状から原因を引けるようにしておきましょう。下の表は「症状 → ほぼ確実な原因 → 対処」の早見表です（とくに上の2つは、この回で必ず一度は遭遇します）。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `ValueError: not enough values to unpack` | `findContours` を3つで受けた（3系の古いサンプル） | OpenCV 4 系は `contours, hierarchy = cv2.findContours(...)` の **2つ**で受ける |
| Sobel でエッジの片側だけしか出ない | `ddepth=cv2.CV_8U` で負の勾配が 0 に潰れた | `cv2.CV_64F` で計算し `cv2.convertScaleAbs` で 8bit 化 |
| 照明ムラのある文書で二値化が片側だけ真っ黒/真っ白 | 固定値や Otsu の**大域的**しきい値 | `cv2.adaptiveThreshold`（局所的）を使う |
| `medianBlur`/`GaussianBlur` で `error` | `ksize` が偶数または非対応 | `ksize` は**奇数**（3,5,7…）。median は int、Gaussian は `(w,h)` のタプル |
| まっすぐ化したら上下/左右が反転した | 4隅の並べ替え順が違う | `order_corners` で和(x+y)・差(x-y)から TL,TR,BR,BL に揃える |
| カラーで CLAHE/equalize したら色が変わった | BGR 各 ch を別々に平坦化した | HSV に変換し **V チャンネルだけ**に適用して戻す |
| matplotlib 保存でエラー/フリーズ | バックエンド未設定（DISPLAY 無し） | `pyplot` を import する前に `matplotlib.use("Agg")` |

さらに、つまずきやすいポイントを Q&A 形式で補足します。

- **Q. `approxPolyDP` が4点にならず、5点や3点になる。** A. `epsilon`（許容誤差）が画像/輪郭ごとに最適値が違うためです。`mini_project.py` のように `epsilon = factor * arcLength` の `factor` を複数（0.02, 0.03, …）試し、`len(approx)==4` になったものを採用するのが堅牢。どうしても駄目なら `cv2.minAreaRect` → `boxPoints` の最小外接矩形で代用します。

- **Q. 書類より大きい外枠（背景）が最大輪郭として拾われる。** A. 二値化の前景/背景が逆になっている可能性大。`THRESH_BINARY` と `THRESH_BINARY_INV`、`THRESH_OTSU` の組み合わせを見直し、「紙が白・背景が黒」になっているか `cv2.imwrite` で途中の二値画像を保存して確認します。

- **Q. デバッグの基本手順は？** A. パイプラインは**途中結果を全部ファイルに保存して目で追う**のが最短です。本章のスクリプトが工程ごとにパネルを並べて保存しているのはこのためです。`print` では `arr.shape` と `arr.dtype`、二値画像なら `np.unique(arr)`（0/255 になっているか）、輪郭なら `len(contours)` を必ず出します。

- **Q. グレースケール画像に色付きで描画しようとしたらエラー/色がつかない。** A. グレーは `(H, W)` の2次元で3チャンネルが無いためです。描画や合成の前に `cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)` で3ch化します（本章の `panel()` がこれをやっています）。

- **Q. `uint8` の足し算で色がおかしくなる。** A. numpy の素の `+` はオーバーフローでラップアラウンド（255+5=4）します。飽和（255で頭打ち）させたいときは `cv2.add` を使います。

## 🚀 発展トピック・参考

この章の古典前処理は、次のトラック以降の土台になります。さらに深めたい人向けの発展テーマと参照先を挙げます。

- **輪郭の階層（hierarchy）と `RETR_*` モード**: `RETR_EXTERNAL` は外枠だけですが、`RETR_TREE`/`RETR_CCOMP` を使うと「穴の中の物体」まで親子関係付きで取れます。ドーナツ状の対象や入れ子図形を数えるときに必要。→ OpenCV [Contours Hierarchy](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)
- **接触した物体の分離（Watershed / 距離変換）**: `open`/`close` では離れない「くっついた円」を、`distanceTransform` + マーカ制御 `watershed` で分けるのが古典の定番。第7回以降のセグメンテーションの前段。
- **直線・円のパラメトリック検出（Hough 変換）**: Canny エッジ → `HoughLinesP`/`HoughCircles` で「直線・円」を式として取り出す。書類の罫線検出やコイン計数に。次回（第5回）で扱います。
- **適応的閾値の発展（Sauvola / Niblack）**: 文書二値化では `adaptiveThreshold` より進んだ Sauvola 法が定番。`scikit-image` の `threshold_sauvola` で試せます（任意ライブラリ）。
- **モルフォロジーの応用（tophat/blackhat による不均一照明除去）**: 大きな構造要素のトップハットで「ゆるい照明ムラ」を引き算する古典テクニック。CLAHE と併せて文書/顕微鏡画像の前処理に。
- **射影幾何の理解**: `getPerspectiveTransform` が解いているのは 3×3 ホモグラフィです。4点未満では解けない理由、`findHomography`+RANSAC で多数点から頑健に推定する話は、第5回（特徴点マッチング）／パノラマ合成へ直結します。
- **公式チュートリアル**: OpenCV の [Image Processing in OpenCV](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html) は本章の全 API（smoothing/gradients/canny/thresholding/morphology/contours/geometric transforms/histograms）を網羅しており、関数の引数を1つずつ確認するのに最適です。

## 10. 動かし方

このモジュールは `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` だけに依存し、GPU もネット接続も不要です。サンプル画像が無くても合成画像が自動生成されるので、`uv sync` 後すぐ実行できます。プロジェクトルートで以下を順に実行してください（カレントはリポジトリルート前提。`outputs/` は各スクリプトが自動で作ります）。

```bash
# 依存をインストール（初回のみ）
uv sync

# 各スクリプトを実行（結果は outputs/04_filtering_edges_morphology/ に保存される）
uv run python lectures/04_filtering_edges_morphology/01_smoothing.py
uv run python lectures/04_filtering_edges_morphology/02_edges_canny.py
uv run python lectures/04_filtering_edges_morphology/03_threshold_morphology.py
uv run python lectures/04_filtering_edges_morphology/04_contours_warp.py
uv run python lectures/04_filtering_edges_morphology/05_histogram_clahe.py

# 章末ミニプロジェクト（全工程を統合した書類スキャナ）
uv run python lectures/04_filtering_edges_morphology/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL）
uv run python lectures/04_filtering_edges_morphology/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る（2通り）
SHOW_SOLUTION=1 uv run python lectures/04_filtering_edges_morphology/exercises.py
uv run python lectures/04_filtering_edges_morphology/exercises_solutions.py
```

実行後は `outputs/04_filtering_edges_morphology/` に生成された PNG を画像ビューアで開いてください。とくに `mini_project_pipeline.png`（書類スキャナの全工程）、`04_document_pipeline.png`（書類補正の全工程）、`02_sobel_dtype_pitfall.png`（`CV_8U` で片側のエッジが消える様子）、`03_threshold_compare.png`（適応的閾値の頑健さ）、`03_morphology.png`（open/close の効果）を解説と照らし合わせると、各操作の役割が腑に落ちます。

## 11. よくあるエラーと対処（クイック表）

「❓ よくある落とし穴・FAQ・デバッグ」の早見表を再掲します。実装中に詰まったら、まずこの表を見てください。この章の不具合の大半は、ここに挙げた数個の原因に集約されます。とくに上の2つは、この回で必ず一度は遭遇します。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `ValueError: not enough values to unpack` | `findContours` を3つで受けた（3系の古いサンプル） | OpenCV 4 系は `contours, hierarchy = cv2.findContours(...)` の **2つ**で受ける |
| Sobel でエッジの片側だけしか出ない | `ddepth=cv2.CV_8U` で負の勾配が 0 に潰れた | `cv2.CV_64F` で計算し `cv2.convertScaleAbs` で 8bit 化 |
| 照明ムラのある文書で二値化が片側だけ真っ黒/真っ白 | 固定値や Otsu の**大域的**しきい値 | `cv2.adaptiveThreshold`（局所的）を使う |
| `medianBlur`/`GaussianBlur` で `error` | `ksize` が偶数または非対応 | `ksize` は**奇数**（3,5,7…）。median は int、Gaussian は `(w,h)` のタプル |
| まっすぐ化したら上下/左右が反転した | 4隅の並べ替え順が違う | `order_corners` で和(x+y)・差(x-y)から TL,TR,BR,BL に揃える |
| カラーで CLAHE/equalize したら色が変わった | BGR 各 ch を別々に平坦化した | HSV に変換し **V チャンネルだけ**に適用して戻す |
| matplotlib 保存でエラー/フリーズ | バックエンド未設定（DISPLAY 無し） | `pyplot` を import する前に `matplotlib.use("Agg")` |

この7項目を「症状を見たら原因が言える」状態にできれば、この章のゴールに到達しています。逆に言えば、つまずいたらまずこの7つを疑えば、たいていの問題は数分で解けます。

## 12. まとめ

この章では、平滑化（ノイズ別の使い分け）→ エッジ（`CV_64F` の理由・Canny）→ 閾値（固定/Otsu/適応的）→ モルフォロジー（open/close で整形）→ 輪郭抽出と形状解析（4系の2返し）→ 透視変換（書類まっすぐ化）→ ヒストグラム/CLAHE（色を壊さないコントラスト補正）、という古典CVの前処理連鎖を、すべて「自分で組んで・なぜそうするか説明できる」レベルで扱いました。とくに `findContours` の2返しと `Sobel` の `CV_64F` は、知っているだけで無駄なデバッグ時間を確実に減らせます。そして `mini_project.py` で、これらが1つの目的（劣化写真→きれいなスキャン）のために連鎖する様子を統合的に確認しました。

次のトラックでは、ここで身につけたエッジ・輪郭・幾何変換の感覚を土台に、特徴点検出とマッチング（ORB/SIFT）やホモグラフィ推定（パノラマ合成）へ進みます。それらも結局は「画像という配列を、勾配や対応点を頼りに別の配列へ写す」操作の延長です。まずは演習10問を自力で全問 PASS させ、二値化からワーピングまでの連鎖を手に馴染ませてから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4（2.4.6）／ opencv-python-headless 4.13（`cv2` 4.13.0）／ Pillow 12.2（12.2.0）／ matplotlib 3.10（3.10.9）。本章は torch を使いませんが、講座全体の深層パートでは torch 2.12+cpu を前提とします。
> すべて CPU のみ・ネット非依存で動作します（`cv2.imshow` は使わず、結果は `outputs/04_filtering_edges_morphology/` に保存）。
