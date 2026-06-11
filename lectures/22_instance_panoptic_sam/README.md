# 第22回 インスタンス/パノプティックセグメンテーションと SAM — mask AP・PQ

> トラック: **セグメンテーション** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers）・`metrics`（pycocotools/torchmetrics）。モデル重みの初回 DL 以外はネット不要で、入力画像は合成生成します。

## 🎯 この章のゴール

この章を終えたとき、あなたは「セグメンテーション」と一括りに呼ばれてきたものが、実は**3つの別タスク**——セマンティック（画素ごとのクラス）、インスタンス（個体を区別したマスク）、パノプティック（things と stuff を全画素にもれなく統合）——に分かれること、そして**プロンプト型の SAM** がそのどれとも違う第4の軸（クラスを当てず、指した“もの”を切り出す）にあることを、出力フォーマットのレベルで説明できるようになります。Mask R-CNN の `masks` がなぜ `(N, 1, H, W)` の**確率**で返るのか、SAM がなぜ1プロンプトに**3枚**のマスクを返すのか、といった「最初の関門」を自分の手で通り抜けます。

さらに、これらのモデルを**どう評価するか**を、式と実装の両面で身につけます。インスタンスは物体検出の mAP（第19回）の IoU を box から mask に置き換えた **mask AP**（`COCOeval(iouType="segm")` と RLE）で測り、パノプティックは検出とは別系統の指標 **PQ = SQ × RQ** で測る。とくに PQ は、`numpy` で一から組んだ自作値が `torchmetrics.detection.PanopticQuality` と一致することを `assert` で確認し、「ライブラリのブラックボックス」を「中で何をしているか分かる道具」に変えます。

到達点を一言でいえば、**Mask R-CNN / Mask2Former / SAM の出力を正しく後処理して可視化でき、mask AP と PQ を“式が分かる状態”で計算できる**こと。合成図形では検出が乏しい（0件のこともある）という現実も含めて、「動かす・読む・測る」を一通り回せるようになるのが合格ラインです。

---

## 1. 3つのセグメンテーションと SAM の位置づけ

まず地図を持ちましょう。**セマンティックセグメンテーション**（第21回）は各画素にクラスだけを振るので、隣り合う2匹の犬は同じ「犬」に塗られ、個体を区別しません。**インスタンスセグメンテーション**は逆に「犬1 / 犬2」を別物として、それぞれにマスクを付けます（ただし背景=stuff は扱わない）。**パノプティックセグメンテーション**はこの2つを統合し、things（数えられる前景: 人・車）のインスタンスと stuff（数えられない背景: 空・道路・芝）の領域を、**全画素にもれなく・重なりなく**1枚へまとめます。各画素はちょうど1つの `(category, instance)` を持ちます。

この3つはいずれも「学習済みのクラスを当てる」枠組みですが、**SAM（Segment Anything）**だけは設計思想が違います。SAM はクラスを一切当てず、点や箱の**プロンプトで指した領域の輪郭マスク**を返す、クラス非依存（class-agnostic）のセグメンタです。「これは犬だ」とは言わないが「ここにある“もの”の形はこれだ」と答える。だから未知の対象でも、合成図形でも、指しさえすれば切り出せます。本章の4スクリプトは、この地図の各点を1つずつ歩きます。

| タスク | 個体の区別 | 背景(stuff) | 代表モデル | 主な評価指標 |
| --- | --- | --- | --- | --- |
| セマンティック | しない | 扱う | DeepLab/SegFormer | mIoU / Dice（第21回） |
| インスタンス | する | 扱わない | **Mask R-CNN** | **mask AP** |
| パノプティック | する | 扱う | **Mask2Former** | **PQ = SQ×RQ** |
| プロンプト型 | プロンプト次第 | プロンプト次第 | **SAM** | マスクと GT の IoU |

表の通り、評価指標もタスクごとに別物です。「セグメンテーションの精度」と言われたら、まず「どのタスクか」を確認するのが第一歩。以降は Mask R-CNN（§2）→ Mask2Former（§3）→ SAM（§4）→ 評価（§5・§6）の順に進みます。

## 2. Mask R-CNN でインスタンスセグメンテーション（`01_maskrcnn_instance.py`）

torchvision の `maskrcnn_resnet50_fpn_v2` は、物体検出（Faster R-CNN）にマスク予測の枝を足した2段階モデルです。ロードは Weights enum から重み・前処理・クラス名を一括取得する torchvision の正準パターンに従います。検出モデルの入力は「`[0,1]` の float CHW テンソルのリスト」で、ImageNet 正規化はモデル内部が行う点が分類と違います（`weights.transforms()` に任せ、二重に正規化しない）。推論は `model.eval()` と `torch.inference_mode()` をセットで使うのが必須作法です。

出力は1枚あたり `{boxes, labels, scores, masks}` の dict ですが、ここに**この回最大の関門**があります。`masks` は `(N, 1, H, W)` の**確率マップ**（0〜1）であって、bool のマスクではありません。可視化や評価に使うには、チャンネル次元を潰して閾値で二値化する必要があります。これを忘れて `(N,1,H,W)` のまま `draw_segmentation_masks`（uint8 画像 + **bool** マスクを要求）へ渡すのが典型バグです。

```python
keep = pred["scores"] >= 0.5                  # スコア閾値でフィルタ
masks_prob = pred["masks"][keep]              # (M, 1, H, W) ★確率
masks_bool = (masks_prob.squeeze(1) > 0.5)    # (M, H, W) bool に二値化
canvas = draw_segmentation_masks(img_uint8, masks_bool, alpha=0.6)  # uint8 画像 + bool
```

このスニペットの `squeeze(1)` と `> 0.5` が後処理の核心です。なお、本講座の合成図形は COCO のクラス（人・車・犬…）に一致しないため、Mask R-CNN は自信を持って検出できず、**検出0件**になることがあります。これはモデルの故障ではなく想定内で、スクリプトは0件でも `draw` に空テンソルを渡さないよう早期分岐し、必ず `exit 0` で完了します。実写を `data/22_instance_panoptic_sam/` に置けば、人や車が検出され実用的なオーバーレイになります。

## 3. Mask2Former でパノプティックセグメンテーション（`02_mask2former_panoptic.py`）

パノプティックの代表が **Mask2Former** です。これは「N 個の固定クエリが、それぞれ1枚のマスク＋1個のクラスを予測する」Transformer 型の統一アーキテクチャで、**同じ重みのまま後処理を変えるだけ**で instance / semantic / panoptic を出し分けられるのが最大の特徴です。HuggingFace では `post_process_panoptic_segmentation` がパノプティック専用の後処理で、`segmentation`（各画素=segment id の `(H,W)` マップ）と `segments_info`（各セグメントの `id / label_id / score`）を返します。CPU 前提では軽量な `swin-tiny` 版を使い、`swin-large` や ViT-Huge は避けます。

ここでも座標系の落とし穴があります。`target_sizes` は `(height, width)` 順で渡しますが、PIL の `image.size` は `(width, height)` なので、`image.size[::-1]` と反転させる必要があります（検出の `post_process_object_detection` と同じ約束）。

```python
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
result = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]], threshold=0.5)[0]  # (W,H) を反転して (H,W) に
segmentation = result["segmentation"]      # (H, W) 各画素=segment id（-1=未割当）
segments_info = result["segments_info"]    # [{id, label_id, score, was_fused}]
```

合成図形では確信のある領域が見つからず `segments_info` が空（あるいはごく少数）になることがあります。その場合でも本スクリプトは、**各クエリが提案している生のマスク**（上位クエリを `masks_queries_logits` から sigmoid して可視化）を描き、「Mask2Former＝クエリ→マスク＋クラス」という仕組みそのものを見せます。PQ の数値計算はここでは行わず、GT を完全に制御できる §6（`04_*.py`）に回します——合成シーンの曖昧な検出結果で PQ を出しても学びが薄いからです。

## 4. SAM でプロンプト型セグメンテーション（`03_sam_prompt_seg.py`）

SAM は本章で唯一、**合成画像でも“絵になる”**デモです。クラスを当てないので、点や箱で指した対象をそのまま高精度に切り出せます。HF では `pip install segment-anything` ではなく `transformers` の `SamModel` / `SamProcessor`（`facebook/sam-vit-base`）を使います。CPU でさらに軽くしたいときは `MODEL_ID` を `Zigeng/SlimSAM-uniform-77` に変えるだけです。

正準フローで2つ注意があります。第一に**点ラベルは 1=前景 / 0=背景**で、取り違えるとマスクが反転します。第二に、SAM の `pred_masks` は **256×256 の低解像**で返るので、`processor.post_process_masks(...)` を通して原寸へ戻さないと位置が合いません。さらに SAM は曖昧性を考慮して**1プロンプトにつき3枚**のマスクを返すので、`iou_scores`（予測した品質）が最大の1枚を採ります。

```python
inputs = processor(image, input_points=[[[x, y]]], input_labels=[[1]], return_tensors="pt")
outputs = model(**inputs)                              # pred_masks:(B,1,3,256,256)
masks = processor.post_process_masks(                   # 低解像→原寸（必須）
    outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][0]  # (3,H,W)
best = masks[outputs.iou_scores[0, 0].argmax()]         # 3枚から品質最大を選ぶ
```

ここで強調したいのは、`iou_scores` は**モデルが自己申告した品質スコア**であって、GT との真の IoU ではない（1.0 を超えることすらある）点です。スクリプトは合成 GT がある場合に**真の IoU**も併記し、両者を区別します。実際に合成シーンで動かすと、点プロンプト・箱プロンプトともに **GT との IoU が 0.98 前後**という鮮やかな結果になり、「クラスを知らなくても形は正確に切れる」という SAM の本質が体感できます。

## 5. mask AP と RLE（`04_maskap_pq_eval.py` 前半）

インスタンスの評価は、第19回の物体検出 mAP を**そのまま**マスクへ拡張します。やることは1つだけ——マッチングに使う IoU を、**box IoU から mask IoU（交差画素 / 和集合画素）に置き換える**。あとは予測を confidence 降順に並べ、未マッチの GT に IoU≥閾値で貪欲対応し、TP/FP を累積して PR 曲線→AP を出す、という流れは検出と同一です。これが **mask AP** で、`COCOeval(iouType="segm")` が公式実装です。

公式実装にマスクを渡すには **RLE（Run-Length Encoding）** という圧縮形式を使います。`pycocotools.mask.encode` が bool マスク（Fortran 連続=列優先で渡すのが約束）を RLE 辞書 `{"size", "counts"}` に変換し、`area` や `bbox` もこの RLE から計算できます。GT は COCO 形式の dict（`images` には `height/width` が必須）、予測は `[{image_id, category_id, segmentation(RLE), score}]` のリストで渡します。

```python
rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))  # bool → RLE（counts は bytes）
# GT: images に height/width、annotations に segmentation=RLE, area, bbox, iscrowd
ev = COCOeval(coco_gt, coco_dt, iouType="segm")  # box ではなく segm
ev.evaluate(); ev.accumulate(); ev.summarize()
mAP, AP50, AP75 = ev.stats[0], ev.stats[1], ev.stats[2]
```

スクリプトは、`numpy` で組んだ素朴な **AP@0.5**（PR 全点積分）と、`COCOeval` の公式 segm AP を並べて表示します。これにより「自分の手で書いた AP の考え方」と「論文比較で正準とされる実装」が地続きであることを確認できます。`COCOeval` の `summarize()` は `AP_S/M/L`（面積別）や `AR`（maxDets 別）も出すので、`areaRng` と `maxDets` の既定値が結果を左右することも目で追えます。

## 6. パノプティック PQ = SQ × RQ（`04_maskap_pq_eval.py` 後半）

パノプティックは things と stuff を全画素に割り当てるため、AP とは別の指標 **PQ（Panoptic Quality）** で測ります。PQ はカテゴリごとに次の3つを計算し、全カテゴリで平均します。まず IoU > 0.5 を満たす GT-予測セグメントを**一意マッチ**させます（0.5 超なので相手は高々1つに定まる、というのが PQ の設計の妙）。マッチを TP、余った予測を FP、取りこぼした GT を FN とすると、

- **SQ（Segmentation Quality）= マッチした組の平均 IoU** … 当てた領域の「形の良さ」
- **RQ（Recognition Quality）= TP / (TP + 0.5 FP + 0.5 FN)** … 検出の F1（過不足の少なさ）
- **PQ = SQ × RQ** … 形の良さと検出の確かさの積

この分解が PQ の読み方を明快にします。SQ が高く RQ が低ければ「当てた所は綺麗だが取りこぼし/誤検出が多い」、逆なら「数は合うが輪郭が雑」。スクリプトでは、わざと**取りこぼし（FN）と誤検出（FP）を1つずつ仕込んだ**合成パノプティックを使い、RQ が 1 未満（この設定で約 0.83）になる様子を見せます。

```python
# preds/target は (H, W, 2): 最後の次元が (category_id, instance_id)
metric = PanopticQuality(things={1, 2}, stuffs={10, 11}, return_sq_and_rq=True)
pq, sq, rq = metric(torch.tensor(pred)[None], torch.tensor(gt)[None])
# 自作(numpy)のカテゴリ別集計→平均が、上の torchmetrics と一致する
assert np.isclose(pq_manual, float(pq), atol=1e-3)
```

`torchmetrics.detection.PanopticQuality` の入力は `(B, H, W, 2)` の整数テンソルで、最後の次元が `(category_id, instance_id)` の組です。`things` は数えられるカテゴリ、`stuffs` は背景カテゴリの ID 集合で、どちらにも属さない ID は void として無視されます。スクリプトは `numpy` でカテゴリ別に SQ/RQ/PQ を集計して平均する自作実装を書き、それが torchmetrics と**小数第3位まで一致**することを `assert` で保証します。この「自作＝公式」の一致体験こそ、PQ を式で理解できた証拠です。

## 7. なぜ合成画像か／実写で試すには

本モジュールが入力を合成生成する理由は2つあります。第一に、本講座は**ネットへ出るのをモデル重みの DL だけ**に限定したいので、入力画像はローカルで決定的に作ります。第二に、評価指標（mask AP / PQ）は「予測と GT の形さえ分かっていれば」計算でき、**GT を完全に制御できる合成データの方がむしろ学びやすい**からです。実際 §6 の PQ は、FN/FP を意図的に1つずつ仕込んで RQ の挙動を狙い通り再現しています。

一方で、Mask R-CNN や Mask2Former は COCO の実物体を覚えているので、抽象的な合成図形には自信を持てず**検出が乏しく（0件のことも）**なります。これは正常な挙動で、各スクリプトは0件でも例外を出さず `exit 0` します。実写で本来の力を見たいときは、`data/22_instance_panoptic_sam/` に `.png` / `.jpg` を1枚置いてください。各スクリプトの `load_user_or_synthetic()` が自動でそれを優先し、人・車・空・地面などに対する本物のインスタンス/パノプティック結果が得られます（その場合 GT マスクは無いので、SAM の真 IoU 等は表示されません）。

## 8. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「インスタンス → パノプティック → プロンプト型 → 評価」と理解が積み上がります。すべて `outputs/22_instance_panoptic_sam/` に図と JSON を保存し、画面表示（`cv2.imshow`）には依存しません。合成シーンと GT マスクの生成、device 判定、可視化の小道具は `seg_helpers.py` にまとめ、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `seg_helpers.py` | 合成シーン＋GT インスタンスの生成、device、ラベルマップのカラー化、図の保存。各スクリプトが import する道具箱 |
| `01_maskrcnn_instance.py` | Mask R-CNN でインスタンス。`masks (N,1,H,W)` 確率→`>0.5` で bool 化、`draw_segmentation_masks` で可視化 |
| `02_mask2former_panoptic.py` | Mask2Former でパノプティック。`post_process_panoptic_segmentation` の `segmentation`/`segments_info`、クエリ可視化 |
| `03_sam_prompt_seg.py` | SAM の点/箱プロンプト。`post_process_masks` で原寸化、3マスクから `iou_scores` 最大を選択、GT との真 IoU |
| `04_maskap_pq_eval.py` | mask AP（`COCOeval` segm + RLE）と PQ=SQ×RQ（自作 numpy ＝ torchmetrics を assert で照合） |
| `exercises.py` | TODO 形式の演習（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答に差し替え） |

表の通り `seg_helpers.py` だけは「読み物」ではなく「再利用する道具」です。中身も厚くコメントしてあるので、最初に一読してから 01 へ進むと、各スクリプトが何を題材に実験しているかが腑に落ちます。

## 9. 動かし方

このモジュールは `torch` / `torchvision`（`dl`）、`transformers`（`hf`）、`pycocotools` / `torchmetrics`（`metrics`）に依存します。初回実行時のみ Mask R-CNN・Mask2Former(swin-tiny)・SAM(vit-base) の重みが自動 DL されます（数百 MB 規模。以後はキャッシュされ高速）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループを用意（初回のみ）
uv sync --group dl --group hf --group metrics

# 各スクリプトを実行（結果は outputs/22_instance_panoptic_sam/ に保存される）
uv run python lectures/22_instance_panoptic_sam/seg_helpers.py             # 道具箱のスモークテスト
uv run python lectures/22_instance_panoptic_sam/01_maskrcnn_instance.py    # インスタンス
uv run python lectures/22_instance_panoptic_sam/02_mask2former_panoptic.py # パノプティック
uv run python lectures/22_instance_panoptic_sam/03_sam_prompt_seg.py       # SAM（点/箱）
uv run python lectures/22_instance_panoptic_sam/04_maskap_pq_eval.py       # mask AP・PQ

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL。それでも exit 0 で落ちない）
uv run python lectures/22_instance_panoptic_sam/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/22_instance_panoptic_sam/exercises.py
```

実行後は `outputs/22_instance_panoptic_sam/` の画像と JSON を確認してください。`03_sam_prompt_seg.png`（点/箱で指した領域が綺麗に切れている）と `04_eval_metrics.json`（自作 PQ と torchmetrics が一致）を、本文の解説と照らし合わせると理解が定着します。実写で試したい場合は §7 の通り `data/22_instance_panoptic_sam/` に画像を置いてから再実行します。

> 補足: `needs_groups` には概念紹介として **Ultralytics SAM**（`SAM('mobile_sam.pt')` / `SAM('sam2.1_t.pt')`）も挙げられますが、`ultralytics` は `opencv-python`（full 版）を引き込み、本講座既定の `opencv-python-headless` と**衝突**します。本スクリプトは衝突を避けるため HF SAM のみを実行経路に使い、Ultralytics 版は「軽量・1行で動く別実装」として概念に留めます（試すなら別環境で `uv add ultralytics`）。

## 10. よくある落とし穴（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。実装中に詰まったら、まずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `draw_segmentation_masks` が例外/真っ黒 | masks が `(N,1,H,W)` の float 確率のまま | `squeeze(1) > 0.5` で `(N,H,W)` の **bool** にする |
| マスクや bbox が画像とズレる | `target_sizes` に `(W,H)` を渡した | `(height, width)` 順＝`image.size[::-1]` を渡す |
| `draw_*` で「uint8 を要求」と怒られる | float の画像テンソルを渡した | 画像を **uint8** の `(3,H,W)` にしてから描画 |
| SAM のマスクが小さい/ずれる | 256×256 の低解像のまま使った | `processor.post_process_masks(...)` で原寸へ戻す |
| SAM のマスクが反転する | 点ラベルの前景/背景を取り違え | `input_labels` は **1=前景 / 0=背景** |
| `iou_scores` を真の IoU と誤解 | SAM の自己申告品質を IoU と混同 | 評価は GT との実 IoU を別に計算（1超もある） |
| `COCOeval` が `KeyError: height` | `images` に `height/width` が無い | 画像メタに必ず `height` と `width` を入れる |
| RLE 化で値がおかしい | C 連続のまま `mask.encode` した | `np.asfortranarray(...)`（列優先）で渡す |
| 合成画像で検出 0 件 | COCO クラスに無い抽象図形 | 想定内。実写を `data/` に置くか、SAM/評価で学ぶ |
| 自作 PQ が torchmetrics と合わない | カテゴリ別→平均でなく全体で集計した | カテゴリごとに SQ/RQ を出し**最後に平均** |

この10項目が、インスタンス/パノプティック/SAM でつまずく原因のほぼ全てです。逆に、これらを自分の言葉で説明でき・回避コードを書けるようになれば、この章のゴールに到達しています。

## 11. まとめ

この章では、セグメンテーションが**セマンティック/インスタンス/パノプティック**の3タスクに分かれること、そして**SAM** がクラス非依存のプロンプト型として独立した軸にあることを、出力フォーマットと後処理のレベルで押さえました。Mask R-CNN の `(N,1,H,W)` 確率マスク、Mask2Former のクエリ→`segments_info`、SAM の3マスク＋`post_process_masks` という「最初の関門」を一つずつ通り抜け、評価では mask AP（`COCOeval` segm + RLE）と PQ=SQ×RQ（自作＝torchmetrics）を式で理解しました。

次回（第23回）は、ここで学んだ SAM を **テキストプロンプト**で動かす方向へ進みます。CLIPSeg で「文で指定した領域」を直接マスク化し、さらに Grounding DINO の検出 box を SAM の `input_boxes` に渡す **Grounded-SAM** の2段構成へ。本章の「プロンプト型セグメンテーション」と「IoU/Dice 評価」が、そのまま下地になります。まずは演習を自力で全問 PASS させ、`assert` で自作と公式実装の一致を体感してから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06-11 時点・CPU で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ opencv-python-headless 4.13 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ pycocotools 2.0.11 ／ torchmetrics 1.9.0 ／ matplotlib 3.10.9。
> 使用モデル: `maskrcnn_resnet50_fpn_v2`（torchvision Weights API）／ `facebook/mask2former-swin-tiny-coco-panoptic` ／ `facebook/sam-vit-base`（軽量化は `Zigeng/SlimSAM-uniform-77`）。本講座セグメ/評価トラックの想定スタック（2026-06 時点）は torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / pycocotools 2.0.11 / torchmetrics 1.9 です。