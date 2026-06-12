# 第8回 古典セグメンテーションと復元 — Watershed・GrabCut・古典inpaint

> トラック: **古典CV** ／ レベル: **初級** ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch / faiss は使いません）

## 🎯 この章のゴール

この章のゴールは、**「人手の事前知識（マーカ・矩形・マスク）を手がかりに、物体を切り出し・復元する古典セグメンテーションを、自分の手で組めるようになる」**ことです。深層の SAM（第20・22回）や LaMa（第29回）は「何も教えなくても」前景や欠損を埋めてくれますが、その内部で何が起きているのか、なぜ初期化が結果を左右するのかを腹に落とすには、まず人間が距離変換でマーカを置き、矩形で大まかな範囲を示し、マスクで「ここは欠損」と教える古典手法を通るのが近道です。古典を知っていると、深層の出力を「正しいか」検証する目も養えます。

具体的な到達点は3つです。1つ目は、**接触してくっついた物体を Watershed で1つずつに分離できる**こと。単純な二値化＋連結成分では融合してしまう（このモジュールでは6枚の硬貨が4個に数えられてしまう）touch した物体を、距離変換のピークをマーカにすることで6個へ正しく分離します。2つ目は、このモジュールの完成物である、**矩形を指定するだけで前景を切り出す GrabCut ツールと、その精度を IoU / Dice で測る評価コード**を書き上げること。3つ目は、**`cv2.inpaint` で傷消し・物体除去を行い、PSNR / SSIM で「どこまで直せたか」を数値化**できることです。

そしてこの章の隠れたテーマが「**古典手法は初期化とパラメータに敏感だ**」という体感です。GrabCut は矩形の置き方ひとつで IoU が 1.00 から 0.75 まで落ち、Watershed は距離変換のしきい値係数を間違えると物体が消えたり融合したりします。この「敏感さ」を知識ではなく数値で体験することが、後段の深層手法の「頑健さ」のありがたみを理解する土台になります。すべて CPU のみ・ネット非依存で、サンプル画像が無くても合成画像が自動生成されるので、いきなり動かせます。

---

## 1. なぜ古典セグメンテーションを今学ぶのか

セグメンテーション（領域分割）とは、画像を「どの画素がどの物体に属するか」で塗り分ける作業です。現在の主役は深層学習（SAM のように1クリックで何でも切り出すモデル）ですが、その手前に **distanceTransform / Watershed / GrabCut / inpaint** という古典の系譜が確固としてあります。これらは「学習済みモデルが要らない」「CPU で軽快」「人間の直感（ここが芯、ここが前景、ここが欠損）を直接コードに落とせる」という強みを持ち、今でも前処理・後処理・ラベル作成・小規模タスクで現役です。

古典と深層の最大の違いは、**「事前知識をどこから持ってくるか」**にあります。深層モデルは大量の学習データから「前景らしさ」を獲得しますが、古典手法は人間が**その場で**与えます。Watershed なら「物体の芯はここ（マーカ）」、GrabCut なら「だいたいこの矩形の中」、inpaint なら「この領域が欠損（マスク）」という具合です。つまり古典セグメンテーションを学ぶことは、「セグメンテーションという問題に、どんな事前知識を、どう与えれば解けるのか」という問題の骨格そのものを学ぶことに等しいのです。

本章のスクリプトはすべて、結果を画面に出さず `outputs/08_classical_segmentation/` に保存します。`cv2.imshow` は GUI バックエンドを必要とし、Docker・SSH・CI などディスプレイの無い環境ではプロセスごと落ちることがあるため、headless 前提の本講座では使いません。各スクリプトは工程を1枚のグリッド画像にまとめて保存するので、実行後にそれを開いて解説と照らし合わせてください。連続値の可視化（距離マップ・スコアの棒グラフ）だけは matplotlib（Agg バックエンド）を併用します。

## 2. 距離変換とマーカ制御 Watershed で接触物体を分離する（`01_watershed.py`）

Watershed（分水嶺）法は、グレースケール画像を「標高マップ」とみなし、低い谷から水を注いでいって、別々の谷から来た水がぶつかる線を境界とするアルゴリズムです。素朴に適用すると画像中の細かな濃淡すべてに反応して過剰分割（over-segmentation）してしまうので、実務では**マーカ制御 Watershed**を使います。これは「ここは確実に物体A」「ここは確実に物体B」「ここは確実に背景」という種（マーカ）を人間（やヒューリスティック）が置き、その種からだけ水を流す方式です。

接触した物体の分離で鍵になるのが **`cv2.distanceTransform`** です。これは二値画像の各前景画素について「最も近い背景画素までの距離」を計算します。1つの物体の中では中心（芯）ほど距離が大きく、ピークになります。2枚の硬貨がくっついていても、芯は2つあるので距離マップには山が2つでき、その間には必ず低い谷ができます。だから距離マップを「最大値の半分」あたりでしきい値処理すると、ピーク付近だけが残り、**触れ合っていた物体が別々の塊（＝マーカの種）に分かれる**のです。これが「単純な連結成分では融合するのに Watershed では分離できる」種明かしです。

```python
# 1) Otsu 二値化 → 2) オープニングでノイズ除去
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
sure_bg = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)   # 確実な背景
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)                    # 各前景画素→最寄り背景の距離
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)  # ピーク付近=確実な前景
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg)                                 # どちらか不明な帯
n_seeds, markers = cv2.connectedComponents(sure_fg)                      # 種にラベル
markers = markers + 1            # 背景を 1 にずらす（0 は unknown 用に空ける）
markers[unknown == 255] = 0      # unknown は 0（未確定）にする
cv2.watershed(img, markers)      # ★ markers を in-place 更新。境界画素に -1 が入る
```

上のコードで注意したいのが **マーカ作成の作法**です。`connectedComponents` は背景に 0、各種に 1,2,… を割り当てますが、Watershed では 0 を「未確定（unknown）」の意味に使うため、全体に `+1` して背景を 1 にずらし、空いた 0 を unknown 専用にします。ここを忘れて背景と unknown が同じ 0 のままだと正しく流れません。また `cv2.watershed` は**カラー（3チャンネル）画像**を要求し、引数の `markers` を破壊的に書き換え、境界画素に `-1` を入れます。実行すると、単純な `connectedComponents` が **4個**（接触ペアが融合）と数えるのに対し、Watershed は **6個**へ正しく分離します。出力 `01_watershed_pipeline.png` の6工程と、`01_distance_transform.png` の距離ヒートマップ（芯にピークが立つ様子）を見比べてください。

## 3. GrabCut による矩形指定の半自動前景抽出（`02_grabcut.py`・完成物）

GrabCut は、**矩形を1つ指定するだけ**で前景を切り出す対話的セグメンテーションです。ユーザは「この矩形の外側は確実に背景、内側は前景かもしれない」という非常に弱い事前知識だけを与えます。アルゴリズムは内部で、前景・背景それぞれの色をガウス混合モデル（GMM）で表現し、各画素が前景か背景かをグラフカット（隣り合う画素は同じラベルになりやすいという滑らかさを考慮した最適化）で決め、それを反復して GMM を更新する、という流れを数回回します。色がそこそこ分離していれば、矩形だけで驚くほどきれいに切れます。

実装で覚えるべき作法は3つです。第1に、出力の `mask` は **4値**（`GC_BGD=0`／`GC_FGD=1`／`GC_PR_BGD=2`／`GC_PR_FGD=3`）で返るので、前景マスクは「`FGD` または `PR_FGD`」を 1 にして作ります。第2に、`bgdModel` と `fgdModel` は **必ず `np.zeros((1, 65), np.float64)`** で渡します（GMM の内部作業領域。形・dtype を間違えるとエラー）。第3に、矩形は**物体全体を囲む**必要があります。矩形が物体を切ってしまうと、はみ出した部分は「確実に背景」と確定してしまい、二度と前景になれません（永続的な取りこぼし＝FN）。

```python
def run_grabcut_rect(img, rect, iters=5):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)   # 背景 GMM の作業領域（中身は触らない）
    fgd = np.zeros((1, 65), np.float64)   # 前景 GMM の作業領域
    cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
```

このスクリプトでは、同じ画像に対して矩形を3通り置いて結果を比べます。**good**（物体をちょうど囲む）は IoU≒1.00、**loose**（画面いっぱい。同じ色の「おとり」も含む）は同色の物体まで拾って IoU≒0.92（余計な前景＝FP）、**clip**（小さすぎて物体を切る）は IoU≒0.75（取りこぼし＝FN）になります。さらに loose の失敗は、後述の「なすり書き」で IoU≒0.94 まで救えます。`02_grabcut_pipeline.png` で各矩形の結果を、`02_grabcut_cutout.png` で good 矩形の切り出し（背景を黒にした成果物）を確認してください。

## 4. なすり書き（GC_INIT_WITH_MASK）で対話的に直す

矩形だけでうまくいかないとき、GrabCut は**マスクによる追加指示**で結果を磨けます。これが画像編集ソフトの「前景ブラシ／背景ブラシ」の正体です。やり方は、1回目の結果マスクを初期値にし、その上に「ここは**確実に**前景（`GC_FGD`）」「ここは**確実に**背景（`GC_BGD`）」という線や点を数本描き足して、`mode=cv2.GC_INIT_WITH_MASK` で再実行するだけです（このとき矩形引数は `None` で構いません）。`GC_FGD`／`GC_BGD` は「確定」、`GC_PR_FGD`／`GC_PR_BGD` は「たぶん」で、確定の指示はアルゴリズムが覆せません。

```python
refined_mask = np.where(fg_loose > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
cv2.circle(refined_mask, (190, 150), 6, cv2.GC_FGD, -1)    # 物体中心は「確実に前景」
cv2.circle(refined_mask, (350, 250), 10, cv2.GC_BGD, -1)   # おとりは「確実に背景」
cv2.grabCut(img, refined_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
```

上のように「物体の中心＝前景」「おとり＝背景」をたった2筆教えるだけで、loose 矩形が拾ってしまった同色のおとりを取り除き、IoU が 0.92→0.94 へ改善します。少ない人手で結果を詰めていけるのが GrabCut の真価で、これは深層 SAM の「クリックで足し引きする」操作の古典版にあたります。なすり書きでどこまで直せるか、`02_grabcut_scores.png` の棒グラフ（good／loose／clip／loose+scribble の IoU・Dice）で俯瞰してください。

## 5. マスク評価 — 混同行列から IoU と Dice を出す

セグメンテーションの良し悪しは「目で見て良さそう」では不十分で、**正解マスク（ground truth）との重なり**を数値化します。本章は合成画像なので前景の真の領域が分かり、教師あり評価ができます。基本は画素単位の**混同行列**です。予測前景かつ真前景を **TP**、予測前景だが真背景を **FP**（余計に拾った）、予測背景だが真前景を **FN**（取りこぼした）と数えます。ここから2つの代表指標が出ます。

**IoU（Intersection over Union）** は `TP / (TP + FP + FN)`、すなわち「予測と正解の重なり ÷ 和集合」です。**Dice 係数**は `2*TP / (2*TP + FP + FN)` で、これは F1 スコアと同値です。両者は `Dice = 2*IoU / (1 + IoU)` の関係にあり、つねに Dice ≧ IoU、つまり Dice の方がやや甘く出ます（小さな領域や境界のズレに少し優しい）。検出・セグメンテーションでは IoU が、医用画像では Dice がよく使われる、という住み分けも覚えておくと良いです。

```python
def confusion(pred, truth):
    p, t = pred > 0, truth > 0
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t)); fn = int(np.sum(~p & t)); tn = int(np.sum(~p & ~t))
    return tp, fp, fn, tn

def iou_dice(pred, truth):
    tp, fp, fn, _ = confusion(pred, truth)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return iou, dice
```

この関数を `02_grabcut.py` が呼び、good 矩形では `TP=18293, FP=0, FN=0`（IoU=Dice=1.000）、clip 矩形では FN が増えて IoU=0.747／Dice=0.855 という具合に、**同じ結果でも Dice の方が高く出る**ことが確認できます。評価指標を自作の混同行列から計算できるようになると、どんなマスク同士でも（深層モデルの出力でも）一貫した物差しで比較でき、これは第14回の評価トラックや検出の mAP へそのまま地続きでつながります。

## 6. 古典 inpaint で傷消し・物体除去（`03_inpaint_classic.py`）

最後はセグメンテーションの裏返しとも言える**復元（inpainting）**です。`cv2.inpaint(src, mask, radius, flags)` は、`mask>0` で示した欠損領域を、その**周囲の画素**から滑らかに埋め戻します。傷や引っかき、日付の透かし文字、不要な小物体の除去などに使えます。アルゴリズムは2種類選べます。**`INPAINT_TELEA`**（Fast Marching 法）は欠損の境界に近い画素から距離の近い順に内側へ埋めていく方式、**`INPAINT_NS`**（Navier-Stokes 法）は流体力学の発想で等照度線（明るさが等しい線）を欠損の中へ伸ばすように埋める方式です。細い欠損ではどちらも良好で、差は大きくありません。

肝心なのは**古典 inpaint の得意・不得意**を体で知ることです。`03_inpaint_classic.py` は2つのケースを比べます。ケースA（細い傷＋透かし文字）は、周囲から色を引いて来れば十分なので、PSNR が破損時 16.5dB から **約39dB** へ大きく回復し、SSIM も 0.996 とほぼ完全です。一方ケースB（大きな四角い穴）は、周囲の色を引き伸ばすだけなので内部がのっぺりボケ、PSNR は **約17dB** までしか戻りません。これが「構造のある大穴は古典では埋まらない」という限界で、深層 LaMa（第29回）が必要になる動機そのものです。

```python
broken = clean.copy(); broken[mask > 0] = (255, 255, 255)          # 欠損を白で表現
telea = cv2.inpaint(broken, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
ns    = cv2.inpaint(broken, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
# 評価: 元画像との PSNR(大きいほど良い) と SSIM(1.0 で完全一致) を自前実装で計算
```

PSNR と SSIM は `numpy`/`cv2` だけで自作しています（PSNR は平均二乗誤差から、SSIM はガウス窓で局所の平均・分散・共分散を取る Wang らの定義を `cv2.GaussianBlur` で実装）。外部の評価ライブラリに頼らずとも、これらの定番指標は数式どおりに書けば再現できる、という点も持ち帰ってください。出力 `03_inpaint_pipeline.png`（A/B の工程）と `03_inpaint_psnr.png`（破損→TELEA→NS の PSNR 棒グラフ）で、細い傷の大回復と大穴の小回復の差を目と数値の両方で確認しましょう。

## 7. パラメータ・初期化への敏感性（体系化）

この章の3手法に共通する性格が「**初期化とパラメータに敏感**」です。Watershed は距離変換のしきい値係数（`0.5 * dist.max()`）が要で、高すぎると小さい物体の種が消えて検出漏れ、低すぎると谷が埋まって接触物体が融合します。GrabCut は矩形の置き方が IoU を 1.00↔0.75 で揺らし、加えて内部で `kmeans` の乱数を使うため、同じ入力でも実行ごとに結果がわずかに変動します（だから本章の演習採点は厳密一致でなく IoU しきい値で判定します）。inpaint は `inpaintRadius` と欠損の大きさで品質が決まります。

この敏感さに振り回されないコツは、**前処理込みで一連の流れとして設計する**ことです。Watershed なら「Otsu 二値化 → オープニングでノイズ除去 → 距離変換 → しきい値で種 → マーカ整形」までを1セットとして調整し、途中の中間結果（二値画像・距離マップ・sure_fg）を必ず保存して目で確かめます。GrabCut なら「矩形を物体に合わせて置く → 結果を見る → なすり書きで詰める」を反復前提で組みます。パラメータを1つずつ動かして中間出力の変化を観察する、という地道なループが、古典手法を使いこなす唯一の近道です。

本章のスクリプトはどれも、入力・中間・最終を1枚のグリッドにまとめて保存するように作ってあります。これは「どの工程で結果が崩れたか」を一目で切り分けるためで、古典CVのデバッグの定石です。まずは付属のパラメータで動かし、次に距離変換のしきい値係数や GrabCut の矩形を自分で変えて、出力画像と IoU/PSNR がどう動くかを観察してみてください。敏感さを「自分で再現できる」ことが、この節のゴールです。

## 8. 深層手法（SAM / LaMa）への橋渡し

ここで身につけた感覚は、後段の深層手法を理解する足場になります。GrabCut の「矩形やなすり書きで前景を指示する」操作は、**SAM（Segment Anything Model、第20・22回）**の「ボックスプロンプト・点プロンプトでマスクを得る」操作の古典版です。SAM は学習済みなので事前知識が少なくても切れますが、「プロンプトで領域を絞り、足し引きで詰める」という対話の骨格は GrabCut と同じです。古典で手を動かしておくと、SAM の出力が「なぜそう切れるのか」「どこを直せばよいか」が直感的に分かります。

復元側も同様です。`cv2.inpaint` の「周囲から色を引いて埋める」発想は大きな構造のある穴では破綻しますが、**LaMa（第29回）**は Fourier 畳み込みで広域の文脈を捉え、大穴や物体除去でも自然に補完します。本章で「古典 inpaint は細い傷には強いが大穴には弱い」と PSNR で定量化した経験があると、LaMa との比較が「なんとなく綺麗」ではなく「ここがこれだけ改善した」という数値の議論になります。古典は深層の対照群（ベースライン）として今も価値があるのです。

つまりこの章は、単体で完結する古典の技術であると同時に、講座後半の深層セグメンテーション・復元への「予習」でもあります。距離変換・マーカ・矩形・マスクという事前知識の与え方、IoU/Dice・PSNR/SSIM という評価の物差し——この2つを手に入れておけば、深層モデルが出てきても「何を入力し、出力をどう測るか」で迷いません。次章以降の動画トラック（第9〜11回）でも、背景差分のマスクや動体の領域抽出という形でセグメンテーションの考え方が再登場します。

## 9. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「接触物体の分離 → 矩形指定の前景抽出と評価 → 復元」と理解が積み上がるよう並べています。すべて結果を `outputs/08_classical_segmentation/` に保存し、画面表示には依存しません。サンプル画像は各スクリプト内で `numpy`/`cv2` により合成生成するため、`data/` に何も無くても動きます（外部モジュールからの import もありません＝自己完結）。

| ファイル | 役割（単一責務） |
| --- | --- |
| `01_watershed.py` | `distanceTransform`→`connectedComponents`→`watershed` で接触硬貨を分離。単純連結成分（4個）と Watershed（6個）を対比 |
| `02_grabcut.py` | **完成物**。`grabCut(GC_INIT_WITH_RECT)` で前景抽出、矩形3通り（good/loose/clip）の IoU/Dice 比較、`GC_INIT_WITH_MASK` でなすり書き改善 |
| `03_inpaint_classic.py` | `cv2.inpaint`（TELEA/NS）で傷消し・大穴埋め。元画像との PSNR/SSIM を自前実装で計算し古典の限界を定量化 |
| `mini_project.py` | **章末ミニプロジェクト**。1 つの合成シーンに対し「前景抽出(GrabCut+IoU/Dice)→計数(Watershed)→清掃(inpaint+PSNR/SSIM)」を統合し、工程図・まとめ図・指標 JSON を出力 |
| `exercises.py` | TODO 形式の演習9問（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答に差し替え）|
| `exercises_solutions.py` | 演習9問の完全な模範解答（実行すると全 PASS を assert で保証。採点ロジックは exercises 側を再利用）|

表の通り、`02_grabcut.py` が deliverable の中核（矩形指定の切り出しツール＋IoU/Dice 評価）で、`01` が接触物体分離、`03` が復元と評価に対応します。そして `mini_project.py` がこの 3 つを 1 本のシーンへ束ねた統合課題です。まず 01 から順に動かし、各 `outputs/08_*.png` を開きながら本文を読み返すと理解が定着します。

## 10. 動かし方

このモジュールは `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` だけに依存し、GPU もネット接続も不要です。サンプル画像が無くても合成画像が自動生成されるので、`uv sync` 後すぐ実行できます。プロジェクトルートで以下を順に実行してください（カレントはリポジトリルート前提。`outputs/` は各スクリプトが自動で作ります）。

```bash
# 依存をインストール（初回のみ）
uv sync

# 各スクリプトを実行（結果は outputs/08_classical_segmentation/ に保存される）
uv run python lectures/08_classical_segmentation/01_watershed.py
uv run python lectures/08_classical_segmentation/02_grabcut.py
uv run python lectures/08_classical_segmentation/03_inpaint_classic.py

# 章末ミニプロジェクト（統合課題。工程図・まとめ図・指標 JSON を出力）
uv run python lectures/08_classical_segmentation/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL）
uv run python lectures/08_classical_segmentation/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/08_classical_segmentation/exercises.py
# 完全な模範解答（全 9 問 PASS を確認）
uv run python lectures/08_classical_segmentation/exercises_solutions.py
```

実行後は `outputs/08_classical_segmentation/` に生成された PNG を画像ビューアで開いてください。とくに `01_watershed_pipeline.png`（接触物体の分離6工程）、`01_distance_transform.png`（距離マップの芯ピーク）、`02_grabcut_pipeline.png`（矩形3通りの結果）、`02_grabcut_scores.png`（IoU/Dice 比較）、`03_inpaint_pipeline.png`（傷消しと大穴）、`03_inpaint_psnr.png`（PSNR 比較）を解説と照らし合わせると、各手法の役割と限界が腑に落ちます。

## 11. よくあるエラーと対処（チェックリスト）

実装中に詰まったら、まずこの表を見てください。この章の不具合の大半は、ここに挙げた数個の原因に集約されます。とくに Watershed のマーカ作法と GrabCut のモデル引数は、初回に必ず一度はつまずきます。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `cv2.watershed` で `error`（チャンネル数） | グレースケール（2次元）を渡した | watershed は **3チャンネルのカラー画像**が必要。`cvtColor(GRAY2BGR)` で3chに |
| Watershed で背景まで1物体に巻き込まれる | マーカの `+1` を忘れ背景と unknown が同じ 0 | `markers += 1` で背景を 1 にし、unknown だけ 0 にする |
| 接触物体が分離されず融合したまま | 距離しきい値が低すぎ（谷が埋まる）/オープニング不足 | `0.5 * dist.max()` 付近に上げる。前段で `MORPH_OPEN` |
| 小さい物体が Watershed で消える | 距離しきい値が高すぎ（小物体の芯が残らない） | しきい値係数を下げる。`dist.max()` が大物体に支配される点に注意 |
| `cv2.grabCut` で `error`（モデル） | `bgdModel`/`fgdModel` の形・dtype 違い | 必ず `np.zeros((1, 65), np.float64)` で渡す |
| GrabCut で物体の一部が必ず欠ける | 矩形が物体を切っている（外は確定背景） | 矩形は**物体全体を囲む**。少し大きめに取る |
| GrabCut の結果が実行ごとに変わる | 内部 `kmeans` の乱数（仕様） | 厳密一致でなく IoU 等のしきい値で評価する。`iterCount` を増やすと安定寄り |
| inpaint で `error` | mask が3chだったり float | mask は **単一チャンネル uint8（0/255）**。`>0` が修復対象 |
| 大穴の inpaint がボケる | 古典 inpaint は周囲色の引き伸ばし（仕様の限界） | 構造復元が要るなら深層 LaMa（第29回）へ |
| matplotlib 保存でエラー/フリーズ | バックエンド未設定（DISPLAY 無し） | `pyplot` を import する前に `matplotlib.use("Agg")`。カラーを `plt.imshow` する時は `BGR→RGB` |

この表を「症状を見たら原因が言える」状態にできれば、この章のゴールに到達しています。逆に言えば、つまずいたらまずこの数項目を疑えば、たいていの問題は数分で解けます。

## 12. まとめ

この章では、距離変換とマーカ制御 Watershed（接触物体の分離）→ GrabCut（矩形指定の半自動前景抽出となすり書き改善）→ IoU/Dice 評価（混同行列から自作）→ 古典 inpaint（傷消し・大穴と PSNR/SSIM 評価）、という古典セグメンテーション・復元の一連を、すべて「自分で組んで・なぜそうするか説明できる・数値で測れる」レベルで扱いました。とくに「Watershed はマーカの `+1` とカラー画像」「GrabCut は矩形が物体を囲む・モデルは `(1,65) float64`」「古典 inpaint は細い傷に強く大穴に弱い」の3点は、知っているだけで無駄なデバッグを確実に減らせます。

そして本章の通奏低音は「古典手法は初期化とパラメータに敏感で、人手の事前知識が結果を左右する」という体感でした。この敏感さを数値で味わったことが、後半の SAM（第20・22回）や LaMa（第29回）の「頑健さ」を正しく評価する目を養います。まずは演習9問を自力で全問 PASS させ、距離変換のマーカ作り・GrabCut の前景抽出・IoU/Dice・古典 inpaint を手に馴染ませてから次のトラック（第9回 動画I/O）へ進んでください。

---

## 🛠 章末ミニプロジェクト — シーンを切り出し・数え・清掃する統合パイプライン

ここまでの 3 手法（Watershed / GrabCut / 古典 inpaint）と 2 系統の評価指標（IoU・Dice と PSNR・SSIM）を、**1 つの合成シーンに対する 1 本のワークフロー**へ統合する総合課題です。`mini_project.py` を実行すると、「机の上に触れ合った硬貨の山と不要なシミが写ったシーン」を題材に、次の 3 段が一気に走ります。

1. **① 前景抽出（GrabCut）**: 硬貨群を囲む矩形を与えて `grabCut(GC_INIT_WITH_RECT)` で前景を切り出し、硬貨領域の真マスクとの **IoU / Dice**（混同行列から自作）で精度を測る。
2. **② 計数（Watershed）**: Otsu 二値化 → `distanceTransform` → マーカ制御 `watershed` で接触した硬貨を 1 枚ずつに分離して数え、**単純連結成分（融合して少なく出る）と対比**する。
3. **③ 清掃（古典 inpaint）**: シミのマスクで `cv2.inpaint`（TELEA / NS）を当てて汚れを消し、シミの無いキレイな参照画像との **PSNR / SSIM** で「どこまで直せたか」を数値化する。

この課題は「撮影画像から対象を切り出し・個数を数え・汚れを修復する」という、製造ラインの部品検査や顕微鏡画像の細胞計数の最小核です。硬貨を実物の部品/細胞に、シミを実写の汚れに差し替えれば、そのまま検査・修復の雛形になります。`digit` 始まりの 01〜03 は import できないため、PSNR/SSIM やパネル合成などはミニプロジェクト内に**自己完結**で書いてあります（外部データ・ネット・GPU 不要）。

**到達の目安**: 付属パラメータでは GrabCut の IoU≒1.00（背景が硬貨と色分離しているので隙間を拾わない）、Watershed が真値どおり **6 枚**へ分離（単純連結成分は 4）、inpaint がシミ除去で PSNR を大きく回復し、総合判定 `all_ok=True` になります。出力は `outputs/08_classical_segmentation/` に保存されます。

| 生成物 | 内容 |
| --- | --- |
| `mini_project_grabcut.png` | 入力＋矩形・真マスク・GrabCut 前景（IoU）・切り出しの 4 枚 |
| `mini_project_watershed.png` | 入力・Watershed 境界（赤）・色分け分離結果（枚数）の 3 枚 |
| `mini_project_inpaint.png` | 破損（シミ）・シミマスク・TELEA 復元（PSNR）・キレイな参照の 4 枚 |
| `mini_project_summary.png` | 切り出し・計数・修復・参照・IoU/Dice 棒グラフを 1 枚に並べたまとめ図 |
| `mini_project_metrics.json` | IoU/Dice・混同行列・naive/watershed 枚数・PSNR/SSIM・総合判定の数値ログ |

```bash
uv run python lectures/08_classical_segmentation/mini_project.py
cat outputs/08_classical_segmentation/mini_project_metrics.json
```

発展課題として、(a) シミを硬貨の上に重ねると inpaint の PSNR がどう落ちるか、(b) GrabCut の矩形を画面いっぱいに広げると IoU がどう崩れるか、(c) 距離変換のしきい値係数（既定 0.5）を 0.3／0.7 に変えると計数がどう変わるか、を自分で試して数値の動きを観察してみてください。「敏感さ」を自分の手で再現できることが、この課題の真のゴールです。

## ✅ 到達チェックリスト

この章を終えたら、次が**できる／説明できる**ことを確認してください。

- [ ] `cv2.distanceTransform` が「各前景画素から最寄り背景までの距離」であり、その**ピークが物体の芯**になることを説明できる。
- [ ] 接触物体を、距離変換のしきい値（`0.5 * dist.max()` など）で `sure_fg` を作って**別々の種**に分離できる。
- [ ] Watershed のマーカ作法（`connectedComponents` の結果に **`+1`** で背景を 1 にずらし、`unknown` を 0 にする）を理由とともに書ける。
- [ ] `cv2.watershed` が**3 チャンネルのカラー画像**を要求し、`markers` を **in-place** で書き換え境界に **-1** を入れることを説明できる。
- [ ] 単純連結成分が接触物体を**融合**してしまうのに対し、Watershed が**正しく分離**できる理由を距離マップで説明できる。
- [ ] `grabCut(GC_INIT_WITH_RECT)` を、`bgdModel`/`fgdModel` を `np.zeros((1,65), np.float64)` で渡して実行し、出力 4 値（`GC_BGD`/`GC_FGD`/`GC_PR_BGD`/`GC_PR_FGD`）から前景マスクを作れる。
- [ ] 矩形の置き方（good/loose/clip）で **IoU が大きく変わる**こと、矩形が物体を切ると **FN が永続化**することを説明できる。
- [ ] `GC_INIT_WITH_MASK` ＋ なすり書き（`GC_FGD`/`GC_BGD`）で loose の失敗を**対話的に修正**できる。
- [ ] 混同行列（TP/FP/FN）から **IoU = TP/(TP+FP+FN)** と **Dice = 2TP/(2TP+FP+FN)** を自作し、つねに **Dice ≧ IoU** になる理由を言える。
- [ ] `cv2.inpaint`（TELEA/NS）が**細い傷には強く大穴には弱い**ことを、PSNR/SSIM の数値で示せる。
- [ ] PSNR（`10*log10(255^2/MSE)`）を自前で実装でき、引き算前に **float へキャスト**してオーバーフローを避ける理由を説明できる。
- [ ] ミニプロジェクトを実行し、前景抽出→計数→清掃の各段を**それぞれの指標**で評価できる。

## ❓ よくある落とし穴・FAQ・デバッグ

実装中に詰まったら、まずここを見てください（第11節の症状別チェックリストと併せて参照）。多くの不具合はこの数個の原因に集約されます。

- **Q. `cv2.watershed` が `error`（チャンネル数）で落ちる。** A. グレースケール（2 次元）を渡しています。watershed は **3 チャンネルのカラー画像**が必須です。`cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)` で 3ch にしてから渡します。
- **Q. Watershed で背景まで 1 物体に巻き込まれる／全部つながる。** A. マーカの **`+1` を忘れ**、背景と `unknown` がどちらも 0 になっています。`markers = markers + 1` で背景を 1 にずらし、`markers[unknown==255] = 0` で未確定だけを 0 にします。
- **Q. 接触物体が分離されず融合したまま。** A. 距離しきい値が**低すぎ**て谷（物体の境目）が埋まっています。`0.5 * dist.max()` 付近まで上げ、前段で `MORPH_OPEN` をかけます。逆に**高すぎる**と小さい物体の芯が消えて検出漏れになります（`dist.max()` は大きい物体に支配される点に注意）。
- **Q. `cv2.grabCut` が `error`（モデル）で落ちる。** A. `bgdModel`/`fgdModel` の形・dtype 違いです。必ず `np.zeros((1, 65), np.float64)` で渡します（中身は触らない）。
- **Q. GrabCut で前景に余計な背景が大量に混ざる（FP が多い）。** A. 矩形が**広すぎ**て、矩形内の背景が前景化しています。矩形は物体にタイトに合わせるか、背景の色分布を一様に近づける／`GC_INIT_WITH_MASK` で「ここは確実に背景」を教えます。本章ミニプロジェクトが背景をほぼ一様にしているのはこのためです。
- **Q. GrabCut で物体の一部が必ず欠ける（FN）。** A. 矩形が物体を**切って**います。矩形の外は「確実に背景」と確定し二度と前景になれません。矩形は**物体全体を少し大きめに**囲みます。
- **Q. GrabCut の結果が実行ごとに変わる。** A. 内部 `kmeans` の乱数による仕様です。厳密一致でなく **IoU 等のしきい値**で評価し、安定させたいなら `iterCount` を増やします（本章の演習採点も IoU しきい値で判定）。
- **Q. `cv2.inpaint` が `error` になる。** A. mask が 3ch だったり float です。mask は**単一チャンネル uint8（0/255）**で、`>0` が修復対象です。
- **Q. PSNR が異常な値（負やゼロ割れ）になる。** A. 引き算前に **`astype(np.float64)`** していない＝uint8 のまま引いて桁あふれしています。`mse==0`（完全一致）は `inf` を返す分岐も入れます。
- **Q. 大穴の inpaint がのっぺりボケる。** A. 古典 inpaint は周囲色の引き伸ばしなので**構造のある大穴は苦手**（仕様の限界）。構造復元が要るなら深層 LaMa（第29回）へ進みます。
- **Q. matplotlib 保存でエラー/フリーズ、または色が反転する。** A. `pyplot` を import する**前**に `matplotlib.use("Agg")` を呼びます。カラーを `plt.imshow` する時は **BGR→RGB**（`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`）を忘れずに。

## 🚀 発展トピック・参考

- **`connectedComponentsWithStats`**: 連結成分ごとの面積・外接矩形・重心を一度に得られる拡張版。Watershed の前後で「小さすぎる種を面積で間引く」「各硬貨の中心に番号を振る」といった後処理に便利です。
- **マーカの作り方いろいろ**: 本章は距離変換ピークでマーカを作りましたが、`cv2.cornerHarris` の極大点、ユーザのクリック、`peak_local_max` 相当の局所最大など、マーカ源を変えると過剰分割／不足分割の度合いが変わります。マーカ品質＝Watershed 品質です。
- **対話的 GrabCut**: 実運用ではユーザが前景/背景ブラシでなすり書きを足す GUI を作ります。本章の `GC_INIT_WITH_MASK` がその心臓部で、`GC_FGD`/`GC_BGD`（確定）と `GC_PR_FGD`/`GC_PR_BGD`（推定）の使い分けがそのまま「ブラシの強さ」になります。
- **しきい値・前処理の自動化**: 距離変換の係数や Otsu の閾値を画像ごとに自動調整する（例: 種の個数が想定範囲に入る最小係数を探索）と、敏感さをある程度吸収できます。古典手法を「使える道具」にする実務テクです。
- **深層への橋渡し**: GrabCut の「矩形・なすり書きで指示」は **SAM（第20・22回）** のボックス/点プロンプトの古典版、`cv2.inpaint` の限界は **LaMa（第29回）** の動機です。本章で古典のベースライン値（IoU・PSNR）を取っておくと、深層との比較が「数値の議論」になります。
- **評価指標の発展**: IoU/Dice は領域の重なりだけを見ますが、境界の正確さを測る **Boundary IoU** や **Hausdorff 距離**、多クラスの **mIoU**（第21回）へ拡張できます。PSNR/SSIM の先には知覚指標 **LPIPS**（第32回 IQA）があります。
- 参考ドキュメント: OpenCV 公式チュートリアル「Image Segmentation with Watershed Algorithm」 https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html ／「Interactive Foreground Extraction using GrabCut」 https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html ／「Image Inpainting」 https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html 。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4（2.4.6）／ opencv-python-headless 4.13（`cv2` 4.13.0.92）／ Pillow 12.2（12.2.0）／ matplotlib 3.10（3.10.9）。
> すべて CPU のみ・ネット非依存で動作します（`cv2.imshow` は使わず、結果は `outputs/08_classical_segmentation/` に保存）。
> 版表記: opencv-python-headless 4.13 / Pillow 12.2 / numpy 2.4 / matplotlib 3.10（2026-06）。
