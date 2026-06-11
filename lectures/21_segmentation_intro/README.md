# 第21回 セマンティックセグメンテーション 入門 — DeepLab/FCN/LR-ASPP・SegFormer・mIoU/Dice

> トラック: **セグメンテーション** ／ レベル: **入門** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers/timm ほか）・`metrics`（torchmetrics）。CPU だけで完走します（初回のみモデル重みを HuggingFace / torch hub からダウンロード）。

## 🎯 この章のゴール

第13回〜第17回で「画像1枚に1ラベル」を返す分類や、画像を1本のベクトルにする埋め込みを学びました。第18回〜第20回では「物体を四角（bbox）で囲む」検出へ進みました。本章のテーマは、その粒度をさらに細かくして、**画像の“すべての画素”にクラスを割り当てる**セマンティックセグメンテーションです。bbox が「だいたいこの辺に車がある」と言うのに対し、セグメンテーションは「この画素は車、隣の画素は道路」と輪郭まで塗り分けます。だから自動運転の走行可能領域、医用画像の病変領域、衛星画像の土地被覆のように、**面積や形が意味を持つ**タスクで主役になります。

この章を終えると、4つのことが自分の手でできるようになります。第一に、torchvision の `deeplabv3 / fcn / lraspp` を Weights API でロードし、**出力 dict の `out['out']` を `argmax` してクラスマップにする**正準フロー（dict をそのまま argmax しない落とし穴つき）。第二に、低解像度のロジットを `nn.functional.interpolate` で**元の解像度に戻し**、クラスID→パレット色で可視化すること。第三に、HuggingFace の `pipeline('image-segmentation')` で **SegFormer** を即推論し、`[{label, mask}]` という返り値の読み方と、手動 API（`post_process_semantic_segmentation`、`target_sizes` が `(高さ,幅)` 順という最頻バグ）を押さえること。第四に、評価指標 **pixel accuracy / per-class IoU / mIoU / Dice / FWIoU** を**画素混同行列から自作**し、`torchmetrics` と**数値が一致する**ことを確認することです。

本章のスクリプトは、ネット接続もデータセットDLも無しで完走するよう、入力画像を**その場で合成**します。ただし正直に書いておくと、torchvision/SegFormer が学習した実在クラス（人・車など）は、幾何学的な合成画像では**うまく検出されないことがあります**（第10節）。それでもスクリプトは必ず `exit 0` で終わり、`data/21_segmentation_intro/` に実写を1枚置けば自動でそちらを使って実用的な結果になります。一方、評価指標を扱う `03` は**モデルに依存しない決定的な GT/予測ペア**を使うので、数値が手計算で検算でき、torchmetrics との一致確認が毎回再現します。ダウンロードが走るのは初回のモデル重み取得だけです。

---

## 1. 分類・検出・セグメンテーション — 粒度の違いと種類

まず「セグメンテーション」と一括りにされがちな3つの種類を整理します。**セマンティック**（本章）は、画素ごとに**クラス**だけを割り当てます。「人」が2人写っていても、両方とも同じ「人」クラスの領域として塗られ、個体は区別しません。**インスタンス**（第22回）は同じクラスでも個体を分けます（人1・人2…）。**パノプティック**（第22回）はその両方を統合し、「物（人・車＝個体あり）」と「背景（空・道路＝個体なし）」を1枚に矛盾なく塗り分けます。本章はもっとも基礎的なセマンティックに集中し、評価指標もここで足腰を作ります。

なぜ画素単位なのか、という問いには「**出力の形が違う**」と答えるのが一番すっきりします。分類器の出力は `(クラス数,)` のベクトルでした。検出は物体ごとに `(box, label, score)` を返しました。セマンティックセグメンテーションの出力は **`(クラス数, 高さ, 幅)` のロジット**で、各画素の位置に「その画素が各クラスである度合い」が並びます。だから画素ごとに `argmax` を取れば `(高さ, 幅)` の**クラスマップ**（各画素にクラスIDが入った2次元配列）になります。分類が「画像→1ラベル」なら、セグメンテーションは「画像→ラベルの画像」だと捉えると、後段の処理（可視化も評価も）が一本の線でつながります。

実務での使い分けはこうです。「画面のどこに何があるか」を**位置とクラスだけ**素早く知りたいなら検出（bbox）で十分で軽量です。「領域の**面積・形・境界**が意思決定に効く」（走行可能領域、腫瘍の体積、農地の区画）ならセグメンテーションが要ります。逆にセグメンテーションは画素ごとに正解（マスク）を用意するアノテーションが高コストで、推論も重くなりがち、というトレードオフがあります。次節から、まず torchvision で「動かす」体験を作ります。

## 2. torchvision で最短セマンティックセグメンテーション

torchvision のセグメモデルは、分類モデルとほぼ同じ Weights API で扱えます。`weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT` のように重みを選ぶと、その重みに紐づく**前処理 `weights.transforms()`** と**クラス名 `weights.meta['categories']`** がセットで手に入ります。これらのモデルは Pascal VOC の **21クラス**（index 0 は `__background__`）で学習されています。`01_torchvision_semseg.py` の中核は次のとおりで、分類と決定的に違うのは**出力が `OrderedDict`** である点です。

```python
weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
model = lraspp_mobilenet_v3_large(weights=weights).to(device).eval()

x = weights.transforms()(image).unsqueeze(0).to(device)  # 前処理（リサイズ＋正規化）込み
with torch.inference_mode():                              # 推論は勾配を切る
    out = model(x)            # ★ out は dict。out['out'] が (1, 21, H, W) のロジット
logits = out["out"]           # ← ここを忘れて out.argmax(...) としない
pred = logits.argmax(dim=1)   # (1, H, W) クラスマップ
```

ここでの一番の落とし穴は**「`out` をそのまま `argmax` しない」**ことです。返り値は `{'out': ロジット, ('aux': ...)}` という dict で、ロジット本体は `out['out']` に入っています（一部モデルは補助出力 `out['aux']` も持ちます）。dict を直接 `argmax` しようとすると意味のないエラーになります。もう一つ、入力は `weights.transforms()` に通すこと。これが ImageNet 統計での正規化と所定サイズへのリサイズを行うので、**自前で二重に正規化しない**のが鉄則です。`model.eval()` と `torch.inference_mode()` を必ずセットにするのも、これまでの回と同じ作法です。動かす土台ができたので、次は「出てきたクラスマップを正しい解像度で見る」話に進みます。

## 3. 解像度を戻す — interpolate とパレット可視化

`weights.transforms()` は入力を内部で **520px 程度へリサイズ**します。そのため `out['out']` のロジットも 520×520 で出てきて、**元画像の解像度とはズレています**。可視化や評価で元画像に重ねるには、解像度を戻す必要があります。ここで重要なのが「**ラベルを戻す**のではなく**ロジットを戻してから argmax する**」という順番です。ラベルマップ（整数）を最近傍補間で拡大すると境界がガタつきますが、ロジット（連続値）を `bilinear` で補間してから `argmax` すると境界が素直に決まります。`01` では次のようにしています。

```python
import torch.nn.functional as F
w0, h0 = image.size                       # PIL は (幅, 高さ) 順
logits_full = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)
pred = logits_full.argmax(dim=1)[0]       # 元解像度 (H, W) のクラスマップ
```

得られたクラスマップは、**クラスID→色のパレット**で塗ると一目で読めます。本講座では Pascal VOC の標準カラーマップ（クラスIDのビットを R/G/B に振り分ける古典手法、`seg_helpers.voc_colormap`）を使い、`palette[class_map]` という索引一発で `(H,W,3)` のカラー画像にします。さらに元画像と `alpha` ブレンドして重ねると、「どの領域が何クラスか」が直感的に見えます。色を扱うときの定番の注意として、**matplotlib に渡す画像は RGB** であること（cv2 の `imread/imwrite` を経由すると BGR が混ざるので、本章は最初から RGB で合成して取り違えを避けています）。

なお `01` を合成シーンで実行すると、`lraspp` も `deeplabv3` も**ほぼ全画素が `__background__`** になります。これは合成の人物シルエットが VOC の「person」として認識されにくいためで、バグではありません（第10節で対処を述べます）。それでもクラスマップの取り出し・解像度復元・可視化という**パイプラインの骨格**は完全に同じです。次は、別のクラス体系・別の実装である HuggingFace の SegFormer を見て、視野を広げます。

## 4. HuggingFace SegFormer — pipeline と手動の2通り

torchvision が VOC の21クラスだったのに対し、HuggingFace の **SegFormer-b0**（`nvidia/segformer-b0-finetuned-ade-512-512`）は **ADE20K の150クラス**で学習されています。だから同じ画像でも「壁・建物・空・草・柱…」のような、より細かい屋内外シーンの語彙でラベル付けされます。`02_segformer_pipeline.py` では2通りの呼び方を体験します。まず最短路の `pipeline('image-segmentation')` は、前処理から後処理までを丸ごと引き受け、**`[{'score', 'label', 'mask'}, ...]` のリスト**を返します。セマンティックでは **`score` は `None`**（インスタンス/パノプティックだとスコアが付く）で、`mask` は前景255の PIL の L 画像です。

```python
seg = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", device=-1)
outputs = seg(image)                       # [{'score': None, 'label': 'sky', 'mask': <PIL L>}, ...]
# 各セグメントのマスクを1枚のクラスマップに畳み込む（マスクは互いに重ならない）
class_map = np.zeros((h, w), dtype=np.int64)
for o in outputs:
    class_map[np.asarray(o["mask"]) > 127] = seg.model.config.label2id[o["label"]]
```

もう一つの**手動 API** は、中身を理解するために重要です。`AutoImageProcessor` ＋ `AutoModelForSemanticSegmentation` で読み込み、`image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(高さ, 幅)])` で原寸のクラスマップを得ます。SegFormer のロジットは**入力の1/4解像度**で出るので、この後処理が内部で `interpolate`＋`argmax` まで行ってくれます。ここでの最頻バグが **`target_sizes` の順番**で、これは `(高さ, 幅)` 順です。PIL の `image.size` は `(幅, 高さ)` なので、**`image.size[::-1]` で渡さないと縦横が入れ替わってマスクが歪みます**。

```python
proc = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForSemanticSegmentation.from_pretrained(model_id).eval()
inputs = proc(images=image, return_tensors="pt")
with torch.inference_mode():
    out = model(**inputs)                  # out.logits: (1, 150, H/4, W/4)
seg_map = proc.post_process_semantic_segmentation(out, target_sizes=[image.size[::-1]])[0]  # (H, W)
```

`(A) pipeline` と `(B) 手動` は同じモデル・同じ後処理なので、得られるクラスマップは一致するはずです。`02` ではこれを画素一致率で確認し、合成シーンに対して **`wall / building / sky / grass / column / signboard / tower`** といったラベルが付き、**一致率 = 1.0000** が出ます。同じ合成画像でも torchvision（全部 background）と SegFormer（多彩なラベル）で結果がまるで違うのは、**学習データセットのクラス体系が違う**からです。次節で、この2系統をどう使い分けるかを整理します。

## 5. torchvision と HuggingFace / モデルの使い分け

セマンティックセグメンテーションのモデルは、大きく **CNN系（torchvision）** と **Transformer系（HF の SegFormer/Mask2Former など）** に分かれます。下の表は本章で扱う代表モデルの性格をまとめたものです。実務では「クラス体系が目的に合うか」「CPU/GPU どちらで回すか」「精度と速度のどちらを取るか」で選びます。表の後に、選定の指針を述べます。

| モデル | 実装 | バックボーン | 学習データ(クラス数) | 速度/重さ | 向く場面 |
| --- | --- | --- | --- | --- | --- |
| **LR-ASPP** `lraspp_mobilenet_v3_large` | torchvision | MobileNetV3 | VOC(21) | 最軽量・約12MB | CPU/エッジ、まず動かす |
| **FCN** `fcn_resnet50` | torchvision | ResNet50 | VOC(21) | 中・約135MB | 素朴な全層畳み込みの基準 |
| **DeepLabV3** `deeplabv3_resnet50` | torchvision | ResNet50 | VOC(21) | 重い・約160MB | 高精度寄り（ASPPで広域文脈） |
| **SegFormer-b0** `segformer-b0-...ade` | HF | MiT-b0(Transformer) | ADE20K(150) | 軽量 Transformer | 多クラス屋内外シーン |

選定の指針はこうです。**まず CPU で軽く動かす**なら LR-ASPP が一番（MobileNet バックボーンで本章でも既定）。**広い文脈を見て精度を上げたい**なら DeepLabV3 の ASPP（空洞畳み込みで受容野を広げ、遠くの手がかりも使う）が効きますが、ResNet50 ぶん重くなります。FCN は「全結合を全部畳み込みに置き換えた」セグメンテーションの原点で、比較の基準として価値があります。**屋内外シーンを150クラスで細かく**塗りたいなら SegFormer。さらに高精度を狙うなら（本章では扱いませんが）`SegFormer-b1/b2` や `Mask2Former` が候補で、第22回のインスタンス/パノプティックにもつながります。

CPU 前提の実務知識も押さえておきましょう。**半精度（fp16）は CPU では遅い/未対応**が多いので `float32` のままにします。スレッド数は `torch.set_num_threads(物理コア数)` で安定します。HF 系は `device='cpu'`（または `-1`）を明示し、`device_map='auto'` は `accelerate` 前提なので CPU のみでは使いません。SegFormer など HF の画像モデルは内部で `timm` を要することがあるため、依存に `timm` を入れておきます（本講座は `hf` グループに含めています）。ここまでで「推論して可視化する」側は一通りです。ここからが本章の主眼、**評価**に入ります。

## 6. 評価指標の定義 — pixel acc / IoU / mIoU / Dice / FWIoU

セグメンテーションの精度は、**予測クラスマップと正解クラスマップを画素単位で突き合わせた混同行列**から、すべて計算できます。`cm[g, p]` を「正解が g・予測が p の画素数」とすると、各クラス c について `TP=cm[c,c]`、`FP=（列cの合計）-TP`、`FN=（行cの合計）-TP` が読み取れます。これを使った5つの指標を定義します。下の表の後で、それぞれの“気持ち”と使い分けを述べます。

| 指標 | 定義 | 何を見るか |
| --- | --- | --- |
| **pixel accuracy** | ΣTP / 全画素 | 正しく塗れた画素の割合。直感的 |
| **per-class IoU** | TP / (TP + FP + FN) | クラスごとの「重なり具合」（Jaccard） |
| **mIoU** | per-class IoU のクラス平均 | セグメンテーションの**主指標** |
| **Dice (=F1)** | 2TP / (2TP + FP + FN) | 重なりを重視。医用で頻出 |
| **FWIoU** | Σ(クラスのGT画素割合 × IoU) | 頻度で重み付けした IoU |

それぞれの使い分けはこうです。**pixel accuracy** は分かりやすい反面、クラス不均衡に弱い指標です。画像の8割が「空」なら、空さえ当てれば accuracy は高く出てしまい、小さな物体の取りこぼしが見えません。そこで主指標は **mIoU** になります。クラスごとに IoU（重なり面積÷和集合面積）を出し、**クラスを平等に平均**するので、小さなクラスの失敗もきちんと効きます。**Dice** は IoU と単調に対応しますが TP の重みが2倍で、小領域に少し甘く（高めに）出るため、病変のように「重なってさえいれば良い」医用で好まれます（Dice は二値の F1 と同じ式です）。**FWIoU** は逆に「面積の大きいクラスを重視」したいときに使う、頻度重み付き版です。

IoU と Dice の関係を一言で押さえると、**両者は `Dice = 2·IoU / (1+IoU)` で結ばれる**ので、片方が上がれば必ずもう片方も上がります。にもかかわらず両方を報告するのは、コミュニティの慣習（セグメンテーション一般は mIoU、医用は Dice）に合わせるためと、「重なりの厳しさ」を2つの目盛りで見たいためです。定義が分かったら、これを**自分で実装して**、ライブラリと数値が合うかを確かめます。

## 7. 自作（画素混同行列）→ torchmetrics と一致確認

`03_miou_dice_eval.py` は、上の定義を `numpy` で素直に実装し、`torchmetrics` と**1e-6 の精度で一致**することを `assert` で確認します。混同行列は二重ループでも作れますが、`np.bincount` を使うと一発です。指標側は、各クラスの `TP/FP/FN` を出して定義どおり割り算するだけです。要点は**未出現クラスの扱い**で、GT にも予測にも一度も出ないクラスは分母が 0 になるので **`np.nan`** にして、平均では `np.nanmean` で除外します（理由は次節）。

```python
def pixel_confusion_matrix(gt, pred, num_classes):
    idx = gt.ravel() * num_classes + pred.ravel()        # (g,p) を一意な整数に符号化
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)

tp = np.diag(cm).astype(float)
fp = cm.sum(0) - tp        # 列和 - TP
fn = cm.sum(1) - tp        # 行和 - TP
present = (tp + fp + fn) > 0
iou = np.full(num_classes, np.nan); iou[present] = tp[present] / (tp + fp + fn)[present]
miou = np.nanmean(iou)     # 未出現クラス(NaN)は除外して平均
```

`torchmetrics` 側で**注意すべきは集計の単位**です。`MeanIoU` / `DiceScore` は既定で「**画像ごとに計算して平均（samplewise）**」しますが、論文の mIoU は「**全画素を1つの混同行列に貯めてから IoU**（global / dataset-level）」です。両者は一般に一致しません（第9節）。自作の混同行列は global なので、torchmetrics 側も**全画素を1枚に平坦化して1サンプルとして渡す**ことで global に揃えます。また `MeanIoU` は未出現クラスを **`-1.0` のセンチネル**で、`DiceScore` は **`NaN`** で返すので、比較前に `-1` を `NaN` に正規化します。

```python
P = torch.from_numpy(pred.ravel()).long().unsqueeze(0)   # (1, N) ← 全画素を1サンプルに
G = torch.from_numpy(gt.ravel()).long().unsqueeze(0)
m = MeanIoU(num_classes=K, per_class=True, input_format="index"); m.update(P, G)
iou_tm = m.compute().numpy(); iou_tm[iou_tm < 0] = np.nan  # -1 センチネル → NaN
assert np.allclose(iou_manual, iou_tm, atol=1e-6, equal_nan=True)
```

決定的な GT/予測ペア（`seg_helpers.make_toy_gt_pred`、6クラスのおもちゃのシーン）での実測が下表です。自作と torchmetrics が**完全に一致**し、`person` クラスは GT にも予測にも出ないため **NaN（mIoU から除外）** になっています。`tree` や `car` の IoU が中程度なのは、予測でわざと境界をずらし・領域を取り違えてあるためで、指標が「重なりの甘辛」をきちんと拾っていることが読めます。

| クラス | per-class IoU（自作=tm） | per-class Dice |
| --- | --- | --- |
| background | 0.807 | 0.893 |
| sky | 0.941 | 0.969 |
| tree | 0.560 | 0.718 |
| road | 0.765 | 0.867 |
| car | 0.505 | 0.671 |
| person | **NaN（未出現→除外）** | NaN |
| **集計** | **mIoU = 0.7157** | **mean Dice = 0.8238** ／ pixel acc = 0.8917 ／ FWIoU = 0.8078 |

## 8. 未出現クラスと ignore_index — NaN 扱いの作法

なぜ未出現クラスを `0` ではなく `NaN`（除外）にするのか。仮に「ある画像に犬が1匹も写っていない」とき、犬クラスの IoU を `0` と数えると、**写っていないクラスのせいで mIoU が不当に下がり**ます。これはモデルの良し悪しと無関係なので、ベンチマーク慣行では**そのクラスを平均から外す**（NaN にして `nanmean`）のが正解です。torchmetrics もこの思想で、`MeanIoU(per_class=False)` の集計値は `-1` センチネルを除いた平均になっており、`03` でも自作の `nanmean` と一致します。「未出現＝0」と素朴に書くと、ライブラリと値がズレてデバッグに苦しむ典型ポイントです。

もう一つ実務で必須なのが **`ignore_index`** です。データセットによっては「アノテーションが曖昧な境界」「評価対象外の領域」を特別なラベル（VOC なら255が定番）で塗ってあり、これらの画素は**評価から完全に除外**しなければなりません。`03` では `pixel_confusion_matrix(..., ignore_index=255)` のように、**混同行列を作る前に該当画素を捨てる**ことで実現します。捨てられた画素は分母から消えるので、その領域で何を予測しても指標は動きません。

```python
gt_ign = gt.copy(); gt_ign[:, :12] = 255          # 左端を「評価対象外(255)」にする
cm = pixel_confusion_matrix(gt_ign, pred, K, ignore_index=255)  # 255 の画素は集計しない
```

`03` の実測では、左端12列を `ignore` にすると pixel acc が 0.8917→**0.8963**、mIoU が 0.7157→**0.7251** に変わります（無視した領域の誤りが消えたぶん上がった）。ここで注意したいのは、**クラス範囲（0..K-1）の外のラベル（255）を `ignore_index` 指定せずに混同行列へ渡すと範囲外で壊れる**ことです。`ignore` ラベルは必ず明示的に除外する、と覚えてください。次は、第7節で予告した「集計単位」の話を、数字で見ます。

## 9. mIoU の集計のクセ — samplewise と global

第7節で「torchmetrics 既定は画像ごと平均（samplewise）、論文は全画素まとめ（global）」と述べました。これは**同じ予測でも mIoU の値が変わる**、地味だが重要な落とし穴です。`03` では同じ画素集合を上下2枚の“画像”に割って `MeanIoU`（既定の samplewise）に渡し、global 定義と比べます。結果は **samplewise = 0.6239 / global = 0.7157** と、はっきり食い違いました。理由は、画像ごとに mIoU を出して平均すると、**小さな画像で1クラスでも外すと一気に下がる**などの偏りが入り、全画素をまとめて数える global とは別の量になるからです。

```python
# 同じ画素を2枚に割って samplewise で平均すると…
P = torch.stack([pred_top, pred_bottom]); G = torch.stack([gt_top, gt_bottom])
m = MeanIoU(num_classes=K, per_class=False, input_format="index"); m.update(P, G)
samplewise = float(m.compute())     # 0.6239（画像ごと平均）
# 全画素を1枚に平坦化すれば global（ベンチマーク慣行）になる
global_miou = miou_from_global_confusion_matrix(gt, pred)  # 0.7157
```

どちらが「正しい」というより、**報告するときに集計単位を明記する**のが肝心です。Cityscapes・ADE20K・Pascal VOC の公式 mIoU はいずれも **global（データセット全体で intersection と union を貯めてから比を取る）**なので、論文と比較するなら global を使います。torchmetrics をそのまま学習ループに差し込んで `MeanIoU().update(batch)` を回すと **samplewise** になり、論文値と微妙にズレる——これを知らないと「実装は合ってるのに数字が合わない」と悩むことになります。本章の自作実装は global なので、ベンチマークの定義と素直に一致します。

## 10. 合成データの限界と実画像への導線

正直に書いておくべき点です。`01`（torchvision, VOC）を合成シーンで実行すると、**ほぼ全画素が `__background__`** になります。これは合成の人物シルエットが「person」として認識されにくいためで、より精細な人物画像を描いても結果は変わりませんでした（実験済み）。一方 `02`（SegFormer, ADE20K）は同じ合成画像から `wall/building/sky/grass/...` と多彩に塗り分けます。つまり**合成画像での見栄えはモデルの学習データに強く依存**します。これは欠陥ではなく、本講座の方針「ネット非依存で必ず `exit 0`、実写を置けば実用的」に沿った設計です。

実写で試すのは簡単で、**`data/21_segmentation_intro/` に `.png/.jpg` を1枚置くだけ**です。`seg_helpers.load_user_or_synthetic_image()` がそれを自動で拾い（無ければ合成にフォールバック）、`01`/`02` がそのまま実画像をセグメンテーションします。人や車、屋内外シーンが写った写真を入れれば、VOC モデルは人・車・椅子などを、SegFormer は壁・床・家具などを塗り分け、本章のパイプラインが実用的に機能するのを確認できます。評価（`03`）はモデル非依存の決定的データを使うので、画像を置いても置かなくても**毎回同じ数値**で再現します。

## 11. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から「動かす → 別実装で広げる → 評価する」と理解が積み上がります。すべて `outputs/21_segmentation_intro/` に図と json を保存し、画面表示には依存しません。合成画像・パレット・モデルロード・決定的 GT/予測といった共通処理は `seg_helpers.py` にまとめてあります。

| ファイル | 役割（単一責務） |
| --- | --- |
| `seg_helpers.py` | device 判定・合成シーン生成・VOC/乱数パレット・モデルロード・図保存・決定的 GT/予測。道具箱 |
| `01_torchvision_semseg.py` | torchvision(lraspp/deeplabv3) で推論。`out['out']` → interpolate → argmax → パレット可視化 |
| `02_segformer_pipeline.py` | HF SegFormer を pipeline と手動の2通りで実行。`[{label,mask}]`・`target_sizes=(H,W)`・一致率 |
| `03_miou_dice_eval.py` | pixel acc/IoU/mIoU/Dice/FWIoU を**自作**し torchmetrics と一致確認。NaN・ignore_index・集計単位 |
| `exercises.py` | TODO 形式の演習（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |

`seg_helpers.py` だけは「読み物」ではなく「再利用する道具」です。とくに `make_toy_gt_pred`（決定的な6クラスのGT/予測）と `colorize`/`voc_colormap`（クラスID→色）が全スクリプトの土台になります。まず helper を一読してから 01 へ進むと、各スクリプトが何を import しているか腑に落ちます。

## 12. 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers/timm ほか）・`metrics`（torchmetrics）グループに依存します。CPU だけで完走し、初回のみ各モデルの重みをダウンロードします（以降はキャッシュから即起動）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf --group metrics

# 各スクリプトを実行（結果は outputs/21_segmentation_intro/ に保存される）
uv run python lectures/21_segmentation_intro/seg_helpers.py          # 道具箱のスモークテスト
uv run python lectures/21_segmentation_intro/01_torchvision_semseg.py
uv run python lectures/21_segmentation_intro/02_segformer_pipeline.py
uv run python lectures/21_segmentation_intro/03_miou_dice_eval.py     # 自作 vs torchmetrics 一致確認

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/21_segmentation_intro/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/21_segmentation_intro/exercises.py

# （任意）実画像で試す: data/21_segmentation_intro/ に .png/.jpg を置くと自動で使われる
```

実行後は `outputs/21_segmentation_intro/` の図を解説と照らし合わせてください。とくに `02_segformer_pipeline.png`（ADE20K の多彩なラベル）と `03_confusion_iou.png`（混同行列と per-class IoU、灰色のバーが NaN＝除外クラス）を見ると、本章の2大テーマ（推論パイプライン・評価指標）が視覚的に腑に落ちます。`deeplabv3` の重み（約160MB）の初回DLに時間がかかる場合は、`01` の `MODEL_NAMES` を `["lraspp"]` に絞ると軽くなります。

## 13. よくあるエラーと対処（チェックリスト）

最後に、本章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。セグメンテーション特有の罠が多いので、詰まったらまずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `argmax` で意味不明なエラー | torchvision 出力は dict | `out['out']` を取り出してから `argmax(1)` |
| マスクが元画像とサイズ違い | transforms が 520px へリサイズ | `F.interpolate(logits, size=(h0,w0))` で戻す |
| SegFormer のマスクが縦横逆 | `target_sizes` が `(高さ,幅)` 順 | `image.size[::-1]` を渡す（PIL は (幅,高さ)） |
| mIoU が論文値と微妙にズレる | torchmetrics 既定は samplewise | global（全画素を1枚に平坦化/混同行列を貯める）で計算 |
| 未出現クラスで mIoU が下がる | 未出現を 0 と数えている | `NaN` にして `nanmean` で除外 |
| per-class IoU 比較で値が合わない | `MeanIoU` は未出現を `-1` で返す | `-1` を `NaN` に直してから比較（`equal_nan=True`） |
| 混同行列が壊れる/巨大化 | 範囲外ラベル(255等)を渡した | `ignore_index` で事前に除外する |
| 色が反転して見える | cv2 経由で BGR が混入 | 画像は RGB のまま扱う（`imread/imwrite` を避ける） |
| `torchmetrics.classification.Dice` が無い | v1.9 で削除 | `torchmetrics.segmentation.DiceScore` を使う |
| SegFormer のロードでエラー | `timm` 未導入 | `hf` グループ（timm 同梱）を入れる |
| CPU 推論が極端に遅い | fp16/half を CPU で使用 | CPU は `float32`＋`inference_mode()` |

この表の項目が、本章で遭遇しがちな不具合のほぼ全てです。とくに上の3つ（`out['out']`・`interpolate`・`target_sizes`）は torchvision/HF セグメの「あるある」なので、症状を見たら原因を即座に言い当てられるようにしておきましょう。

## 14. まとめ

本章では、**画素ごとにクラスを割り当てる**セマンティックセグメンテーションを、torchvision（VOC・`out['out']`→argmax）と HuggingFace SegFormer（ADE20K・`[{label,mask}]`・手動 `post_process`）の両系統で動かし、低解像度ロジットの `interpolate` 復元とパレット可視化までを通しました。そして本章の主眼である評価では、**画素混同行列だけ**から pixel acc / per-class IoU / mIoU / Dice / FWIoU を自作し、`torchmetrics` と**数値が完全一致**することを確認しました。通底するのは「**評価はすべて混同行列に還元できる**」「**未出現クラスは NaN で除外**」「**mIoU は集計単位（global/samplewise）を明記**」という3つの勘所です。

ここで身につけた「クラスマップ → 混同行列 → 指標」という骨格は、次の第22回（インスタンス/パノプティック、SAM）の **mask AP・PQ=SQ×RQ** や、第23回（テキストプロンプトセグメンテーション）の評価へとそのまま発展します。まずは演習を全問 PASS させ、`03` の「**samplewise 0.624 と global 0.716 が食い違う**」結果と「**person が NaN で mIoU から除外される**」理由を、自分の言葉で説明できるようにしてから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06-11 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ timm 1.0.27 ／ torchmetrics 1.9.0 ／ pycocotools 2.0.11（第19・22回で使用）／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成画像の描画）
> 使用モデル: `lraspp_mobilenet_v3_large` / `deeplabv3_resnet50`（torchvision, Pascal VOC 21クラス）／ `nvidia/segformer-b0-finetuned-ade-512-512`（HF, ADE20K 150クラス）。初回のみ重みを取得しキャッシュします。
> 注意: `torchmetrics.classification.Dice` は v1.9 で削除済みのため、本章は `torchmetrics.segmentation.DiceScore` / `MeanIoU` を使用しています。