# 第12回 PyTorch画像テンソルとデータ拡張 — transforms v2 / albumentations / DataLoader

> トラック: **深層CV(分類)** ／ レベル: **初級** ／ 依存グループ: `dl`（torch・torchvision）＋ `aug`（albumentations）。完全CPUで完走します。

## 🎯 この章のゴール

第11回までで、私たちは画像を numpy 配列（OpenCV/Pillow の世界）として自在に扱ってきました。本章からは、いよいよ**深層学習トラック**に入ります。その最初の一歩となるのが、見慣れた画像（高さ×幅×チャンネル, 0〜255 の整数）を、**ニューラルネットが食べられる形**——チャンネルが先頭に来た `(C, H, W)` の float テンソルで、しかも `[0,1]` にスケールし、平均と標準偏差で正規化したもの——へ正しく変換することです。この工程は地味ですが、深層CVで最初に必ずつまずく関門でもあります。なぜなら、並び替えを間違えれば形が合わずエラーになり、スケーリングを忘れれば学習がまったく進まないからです。だからこそ本章を終えたとき、あなたは**この前処理パイプラインを、ライブラリの魔法に頼らず一段ずつ自分の手で書け、各段で dtype・shape・値域がどう変わるかを言葉で説明できる**ようになります。

到達点は4つです。第一に、**HWC↔CHW の並び替えと `ToImage`/`ToDtype(scale=True)` による 0〜1 スケーリング**を理解し、`Normalize` で **ImageNet 統計と CLIP 専用統計が別物**であることを使い分けられること。第二に、**自作の `Dataset` と `DataLoader`** で画像フォルダをミニバッチに積み、`num_workers` の意味を説明できること。第三に、**torchvision transforms v2 と albumentations** の両方でデータ拡張を書け、とりわけ albumentations が**画像・bbox・mask を同時に変換できる**強みを理解すること。第四に、**「拡張は学習時だけ・推論/評価は決定論」**という鉄則を、同じ画像を2回読んで結果を比べることで体得することです。なお、本章のスクリプトはネット接続もデータセットのDLも要らず、**合成画像（円・四角・三角の小画像）をその場で生成**して完走します（実画像で試したい人向けの導線も用意しています）。

---

## 1. なぜ「並べ替えて・割って・正規化」するのか — HWC↔CHW と ToImage/ToDtype

画像を深層モデルに渡す前処理は、煎じ詰めると3つの操作です。**(1) 軸の並び替え（HWC→CHW）**、**(2) 値域のスケーリング（0〜255 → 0〜1）**、**(3) 正規化（平均0・分散1に近づける）**。というのも、OpenCV や Pillow が返す画像は `(高さ, 幅, チャンネル)` の `uint8`（人間が見慣れた並び）ですが、PyTorch の畳み込み層は**チャンネルを先頭にした `(C, H, W)`** を期待するからです。さらにモデルは小さな浮動小数点（おおむね `[-2.6, +2.7]` くらいに収まる正規化済みの値）で学習されているため、0〜255 の整数のまま入れると桁が合わず、学習が破綻します。だからこの3操作は「お作法」ではなく、**モデルが学習されたときの入力分布に合わせる**という必然なのです。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="画像をテンソルへ：(H,W,3)のHWCをToImageで(3,H,W)のCHWへ並び替え、R/G/Bが別々の面に分かれる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="60" y="100" width="130" height="130" fill="#eff6ff" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="125" cy="165" r="33" fill="#f97316"/><text x="125" y="256" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">(H, W, 3) ・ HWC</text><line x1="200" y1="150" x2="356" y2="150" stroke="#71717a" stroke-width="2.5"/><polygon points="364,150 351,144 351,156" fill="#71717a"/><text x="280" y="138" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">ToImage()</text><text x="280" y="172" text-anchor="middle" font-size="12" fill="#71717a">軸を並び替え</text><rect x="420" y="76" width="116" height="116" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="400" y="98" width="116" height="116" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><rect x="380" y="120" width="116" height="116" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="478" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#2563eb">B</text><text x="458" y="115" text-anchor="middle" font-size="15" font-weight="700" fill="#16a34a">G</text><text x="438" y="192" text-anchor="middle" font-size="30" font-weight="700" fill="#dc2626">R</text><text x="458" y="266" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">(3, H, W) ・ CHW</text></svg><figcaption>OpenCV/Pillow が返す画像は <b>(H, W, 3)</b>＝各画素に R,G,B が混ざって並ぶ <b>HWC</b> 形式です。PyTorch の畳み込み層は<b>チャンネルを先頭</b>にした <b>(3, H, W)</b>＝<b>CHW</b>（R/G/B が別々の面に分かれた形）を要求します。<code>v2.ToImage()</code> はこの<b>軸の並び替えだけ</b>を行い、dtype は uint8 のまま据え置きます。</figcaption></figure>

torchvision の **transforms v2**（`torchvision.transforms.v2`、現行の正準API）は、この3操作を**小さな部品に分解**して提供します。正準的な並びは次の3段で、`01_tensor_layout_normalize.py` はこれを1段ずつ適用して、各段で dtype・shape・値域がどう変わるかを表示します。ポイントは、`ToImage()` が**並び替え（HWC→CHW）だけ**を行い dtype は `uint8` のまま据え置くこと、`ToDtype(..., scale=True)` の **`scale=True` こそが「/255」の本体**であることです。

```python
from torchvision.transforms import v2
import torch

transform = v2.Compose([
    v2.ToImage(),                                   # PIL/ndarray → Tensor、HWC→CHW（uint8 のまま）
    v2.ToDtype(torch.float32, scale=True),          # uint8(0..255) → float32(0..1)。scale=True が /255
    v2.Normalize(mean=(0.485, 0.456, 0.406),        # チャンネルごとに (x-mean)/std
                 std=(0.229, 0.224, 0.225)),
])
```

実行すると、`ToImage()` の出力が `uint8, (3,96,96), [75,182]`、`ToDtype(scale=True)` で `float32, [0.29,0.71]`、`Normalize` で `float32, [-0.57,1.15]`（負値を含む）と段階的に変わる様子が出力されます。なお **transformers v5 / torchvision 0.27 の世代では、旧来の `transforms.ToTensor()` は非推奨**になりました（`ToTensor` は「CHW化＋/255」を一気にやる融合版で、警告つきで残ってはいます）。現行の推奨は上のように `ToImage()` ＋ `ToDtype(scale=True)` の2部品に分けて書くことです。古いブログの `ToTensor()` をそのまま写経すると非推奨警告が出るので、本講座では最初から v2 の分解形で統一します。

<figure class="lec-fig"><svg viewBox="0 0 480 350" role="img" aria-label="前処理3段：ToImage→ToDtype(scale=True)→Normalize で dtype・並び・値域が段階的に変わる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="35" y="18" width="190" height="56" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="130" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">uint8 ・ HWC</text><text x="130" y="63" text-anchor="middle" font-size="12" fill="#52525b">(H, W, 3)  [0, 255]</text><line x1="130" y1="74" x2="130" y2="97" stroke="#71717a" stroke-width="2.5"/><polygon points="130,104 123,96 137,96" fill="#71717a"/><text x="210" y="93" font-size="12.5" font-weight="600" fill="#c2410c">ToImage()：HWC→CHW</text><rect x="35" y="104" width="190" height="56" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="130" y="130" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">uint8 ・ CHW</text><text x="130" y="149" text-anchor="middle" font-size="12" fill="#52525b">(3, H, W)  [0, 255]</text><line x1="130" y1="160" x2="130" y2="183" stroke="#71717a" stroke-width="2.5"/><polygon points="130,190 123,182 137,182" fill="#71717a"/><text x="210" y="179" font-size="12.5" font-weight="600" fill="#c2410c">ToDtype(scale=True)：÷255</text><rect x="35" y="190" width="190" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="130" y="216" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">float32</text><text x="130" y="235" text-anchor="middle" font-size="12" fill="#2563eb">(3, H, W)  [0, 1]</text><line x1="130" y1="246" x2="130" y2="269" stroke="#71717a" stroke-width="2.5"/><polygon points="130,276 123,268 137,268" fill="#71717a"/><text x="210" y="265" font-size="12.5" font-weight="600" fill="#c2410c">Normalize：(x−μ)/σ</text><rect x="35" y="276" width="190" height="56" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="130" y="302" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">float32</text><text x="130" y="321" text-anchor="middle" font-size="12" fill="#ea580c">(3, H, W)  [-0.6, 1.2]</text></svg><figcaption>transforms v2 の前処理は<b>3つの部品</b>に分かれます。<code>ToImage()</code> が <b>HWC→CHW</b> の並び替え（dtype は uint8 のまま）、<code>ToDtype(..., scale=True)</code> の <b>scale=True が「÷255」の本体</b>で <b>[0,1]</b> の float32 へ、<code>Normalize</code> が <b>(x−μ)/σ</b> で標準化して<b>負値を含む</b>値域にします。各段で <b>dtype・shape・値域</b>がどう変わるかを追えることが本章の要です。</figcaption></figure>

## 2. Normalize — ImageNet統計とCLIP統計はなぜ違うのか／二重スケーリングの罠

`Normalize(mean, std)` は各チャンネルに `(x - mean) / std` を施す操作で、**どの統計（mean/std）を使うかはモデルが学習されたときに決まっています**。ImageNet で事前学習された CNN/ViT（torchvision の ResNet など）は `mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)` を使うのが定番ですが、**OpenAI CLIP は専用の統計**（`mean=(0.481,0.458,0.408), std=(0.269,0.261,0.276)`）を使います。ここを取り違えて CLIP に ImageNet 統計を当ててしまうと、入力分布が学習時とズレてゼロショット精度が落ちます。「正規化の値はモデルに付いてくる仕様であって、自分で適当に決めるものではない」——これが本節の一番大事な感覚です。実務では `AutoImageProcessor` や `ResNet18_Weights.DEFAULT.transforms()` が**正しい統計を自動で持ってきてくれる**ので、手打ちでズレる事故を避けられます。

`01_tensor_layout_normalize.py` は同じ画像を ImageNet 統計と CLIP 統計の両方で正規化し、結果の平均値が変わること（例: ImageNet=−0.193 / CLIP=−0.163）を示します。あわせて**最も多い前処理バグ「スケーリング忘れ」**も実演します。下の表のように、`scale=True`（/255 する）を忘れて 0〜255 のまま `Normalize` に入れると、値が桁外れに膨らんで（最大 800 超）モデルが学習不能になります。「Loss が NaN になる」「まったく収束しない」ときの容疑者筆頭がこれです。

| ケース | パイプライン | Normalize 後の値域 | 判定 |
| --- | --- | --- | --- |
| (A) 正しい | `ToDtype(scale=True)` → Normalize | おおむね `[-0.57, 1.15]` | 妥当 |
| (B) スケール忘れ | `ToDtype(scale=False)` → Normalize | `[331, 810]` のような桁外れ | バグ |

ここで「二重スケーリング」という言葉の意味も整理しておきます。`scale=True` は **uint8→float のときだけ /255** する仕様で、すでに float の値にもう一度かけても何も起きません（v2 の `ToDtype` は float→float の scale を no-op にしている）。本当に危ないのは、**v1 の `ToTensor()`（中で /255 する）と自前の `img/255.0` を二重にかける**、あるいは v1 と v2 の部品を**混在**させて「どこで /255 したか」を見失うパターンです。だから本講座のルールはシンプルで、**「ひとつのパイプラインの中ではスケーリングを1回だけ」「v1 と v2 を混ぜない（v2 で統一）」**。これさえ守れば二重スケーリングは起きません。

## 3. Dataset と DataLoader — 「1枚を返す」と「バッチに積む」

深層学習は1枚ずつではなく**ミニバッチ（複数枚を1つのテンソルにまとめたもの）**で回します。その橋渡しを担うのが `torch.utils.data` の2つの役者です。**`Dataset`** は「i番目の `(画像テンソル, ラベル)` を返す係」で、実装すべきメソッドは突き詰めると2つだけ——`__len__`（総数）と `__getitem__(i)`（i番目を返す）です。一方の **`DataLoader`** は「`Dataset` から複数枚を取り出して自動で1つのバッチテンソルに積むまとめ役」で、`batch_size`・`shuffle`・`num_workers`・`drop_last` といった**回し方**を担当します。役割分担を一言でいえば、**Dataset=データの定義、DataLoader=データの供給**です。

<figure class="lec-fig"><svg viewBox="0 0 640 320" role="img" aria-label="Datasetは1枚ずつ(3,H,W)とラベルを返し、DataLoaderがbatch_size枚を積んで(B,3,H,W)のバッチにする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="93" y="36" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Dataset（定義）</text><rect x="58" y="52" width="70" height="40" rx="4" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="93" cy="72" r="13" fill="#f97316"/><rect x="58" y="100" width="70" height="40" rx="4" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="80" y="110" width="26" height="20" fill="#2563eb"/><rect x="58" y="148" width="70" height="40" rx="4" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><polygon points="93,154 106,176 80,176" fill="#16a34a"/><rect x="58" y="196" width="70" height="40" rx="4" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="93" cy="216" r="13" fill="#f97316"/><text x="93" y="260" text-anchor="middle" font-size="12" fill="#52525b">各 (3, H, W) と ラベル</text><line x1="145" y1="150" x2="350" y2="150" stroke="#c2410c" stroke-width="3"/><polygon points="362,150 348,141 348,159" fill="#c2410c"/><text x="255" y="118" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">DataLoader</text><text x="255" y="180" text-anchor="middle" font-size="12" fill="#71717a">batch_size・shuffle</text><polygon points="430,104 462,78 570,78 538,104" fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/><polygon points="538,104 570,78 570,190 538,216" fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/><rect x="430" y="104" width="108" height="112" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/><line x1="430" y1="132" x2="538" y2="132" stroke="#ea580c" stroke-width="1"/><line x1="430" y1="160" x2="538" y2="160" stroke="#ea580c" stroke-width="1"/><line x1="430" y1="188" x2="538" y2="188" stroke="#ea580c" stroke-width="1"/><text x="498" y="256" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">(8, 3, H, W)</text><text x="498" y="278" text-anchor="middle" font-size="11.5" fill="#52525b">B=8 枚を1つに ＋ ラベル(8,)</text></svg><figcaption><b>Dataset</b> は <code>__getitem__(i)</code> で <b>i 番目の1枚</b>＝<b>(3, H, W)</b> テンソルとラベルを返す「データの定義」係です。<b>DataLoader</b> はそれを <b>batch_size</b> 枚集めて1つの <b>(B, 3, H, W)</b> バッチテンソル（ラベルは長さ B のベクトル）に積む「供給」係で、<code>shuffle</code> や <code>num_workers</code> といった<b>回し方</b>を担当します。</figcaption></figure>

`02_dataset_dataloader.py` は、`class_name/xxx.png` という**「フォルダ＝ラベル」レイアウト**（torchvision の `ImageFolder` と同じ定番）を読む `ShapeFolderDataset` を自作します。下がその核です。まずコンストラクタでサブフォルダ名を**ソートして**クラスIDを振り（再現性のため必ずソート）、`(パス, ラベル)` の一覧を作っておきます。そして `__getitem__` で**そのとき初めて画像を開く（遅延読み込み）**のがポイントです。全画像を最初にメモリへ載せるとデータが大きいとき破綻するので、必要な1枚だけ都度読むのが定石です。

```python
class ShapeFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.classes = sorted(p.name for p in root.iterdir() if p.is_dir())  # 必ずソート
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = [(p, self.class_to_idx[name])                          # (パス, ラベル)
                        for name in self.classes
                        for p in sorted((root / name).glob("*.png"))]

    def __len__(self):  return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")        # ここで初めて読む（遅延）
        return (self.transform(img) if self.transform else img), label
```

この `Dataset` を `DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)` に渡すと、`for xb, yb in loader:` で `xb` が `(8, 3, 64, 64)` のバッチテンソル、`yb` が長さ8のラベルになって出てきます。学習回（第13回〜）では、この `xb` を `xb.to(device)` でモデルと同じデバイスに載せてから forward します。`drop_last=True` は端数のバッチ（最後に8枚そろわない分）を捨てる指定で、バッチサイズを揃えたい学習時に使います。`shuffle=True` は**学習時はエポックごとに順番を混ぜる**ため（学習の偏りを防ぐ）、評価時は `False`（順番固定）にするのが普通です。

## 4. num_workers — 並列読み込みの勘所

`DataLoader(num_workers=N)` は、**データの読み込み・前処理を N 個の別プロセスで並列に行う**指定です。`num_workers=0`（既定）はメインプロセスが1枚ずつ読むので、GPU 学習では「GPU が次のバッチを待ってアイドルする」ボトルネックになりがちです。`num_workers>0` にすると、GPU が現在のバッチを計算している間に、ワーカーが次のバッチを先回りで用意できます。目安は**CPUコア数前後**ですが、多すぎると**プロセス起動やデータ受け渡し（pickle 化）のオーバーヘッド**で逆に遅くなることもあるので、実測して決めます。

`02_dataset_dataloader.py` は `num_workers=0` と `=2` で1エポックの時間を比べます。本章は画像が小さく枚数も少ないので差はわずか（むしろワーカー起動コストで `=2` の方が遅く出ることすらあります）ですが、**「別プロセスで先読みする」概念**を体感するのが狙いです。実データ・GPU 学習では効きが大きく変わります。なお `num_workers>0` を使うときは、**スクリプトを `if __name__ == "__main__":` ガードの中で動かす**のが必須の作法です（ワーカープロセスがモジュールを再 import するため、ガードが無いと無限にプロセスが増えたり落ちたりします）。本章のスクリプトはすべてこのガードを守っています。

| 設定 | 読み込み方 | 向いている場面 | 注意 |
| --- | --- | --- | --- |
| `num_workers=0` | メインプロセスが逐次読む | 小規模・デバッグ・本章のような軽い前処理 | GPU 学習だと供給待ちになりやすい |
| `num_workers>0` | 別プロセスで並列先読み | 実データ・GPU 学習で供給を間に合わせたい | `__main__` ガード必須。多すぎると逆効果 |

表の通り、`num_workers` は「GPU を遊ばせないための供給力チューニング」だと捉えると腑に落ちます。本章では概念の確認に留め、実際に効果が大きくなる転移学習は次章（第13回）で扱います。

## 5. データ拡張① — torchvision transforms v2（RandomResizedCrop / Flip / ColorJitter）

**データ拡張（augmentation）**は、学習画像にランダムな変形（反転・拡大切り抜き・色ゆらぎなど）をかけて**実質的にデータを水増し**し、過学習を抑えてモデルの汎化を上げる技術です。たとえば「猫は左右反転しても色味が少し変わっても猫」——この当たり前の不変性をモデルに教え込むのが拡張の役割です。torchvision transforms v2 は分類で最も手軽な選択肢で、`RandomResizedCrop`（ランダムな位置・スケールで切り抜いてリサイズ）、`RandomHorizontalFlip`（確率 p で左右反転）、`ColorJitter`（明るさ・コントラスト・彩度・色相をゆらす）などを `Compose` で連ねます。

`03_augment_v2_albumentations.py` は、同じ1枚の画像に拡張を**8回**かけて格子状に並べます（`outputs/.../03_aug_v2.png`）。下が拡張パイプラインで、**ランダム変換を含むので呼ぶたびに違う結果**になる——これが「水増し」の正体です。可視化のため `Normalize` は付けず `ToDtype(scale=True)` までで `[0,1]` に留めています（正規化すると負値が出て、戻さずに表示すると色が壊れるため）。

```python
aug = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(96, 96), scale=(0.6, 1.0), antialias=True),  # ランダム拡大切り抜き
    v2.RandomHorizontalFlip(p=0.5),                                         # 確率0.5で左右反転
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 色ゆらぎ
    v2.ToDtype(torch.float32, scale=True),
])
```

ここで `antialias=True`（リサイズ時のアンチエイリアス）を付けるのは、v2 のリサイズ系で**ジャギーやモアレを避ける**ための定石です。拡張は**強すぎると逆効果**（猫が猫に見えなくなるほど歪めると学習を阻害する）なので、`scale` の下限や `ColorJitter` の振れ幅はタスクに応じて加減します。生成された `03_aug_v2.png` を見ると、同じ画像から切り抜き位置・反転・色味の異なる8枚が得られていることが一目で分かります。これが「1枚を何通りにも見せて学習データを実質増やす」ということです。

## 6. データ拡張② — albumentations と bbox/mask の同時変換

もうひとつの主役が **albumentations** です。OpenCV ベースで高速、拡張の種類が非常に豊富で、入出力は **numpy（HWC, uint8）**——PIL でも Tensor でもない点に注意します（torchvision v2 が PIL/Tensor を扱うのと対照的）。分類だけなら v2 でも albumentations でも好みですが、albumentations が本当に光るのは**検出やセグメンテーション**です。画像を反転・回転・切り抜きしたとき、**バウンディングボックス（bbox）やセグメンテーションマスク（mask）も同じ幾何変換で一緒に動かさないと、教師データの「画像と正解の対応」が崩れてしまう**——この同時変換を albumentations は標準で面倒見てくれます。

`03_augment_v2_albumentations.py` の後半は、暗い背景に明るい四角の物体を置いたシーンで、**画像・bbox・mask を同時に左右反転＋リサイズ**します。`A.Compose([...], bbox_params=...)` の `bbox_params` に **`format="pascal_voc"`（=xyxy）** を指定し、ラベルは `label_fields` で渡すのが作法です。下のように `image=`・`mask=`・`bboxes=` を一緒に渡すと、返り値の `bboxes`・`mask` が画像と整合した状態で返ってきます。

```python
joint = A.Compose(
    [A.HorizontalFlip(p=1.0), A.Resize(128, 128)],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)
res = joint(image=scene, mask=mask, bboxes=[bbox], labels=[0])
# 反転前 bbox(xyxy) = [24, 36, 84, 96] → 反転後 = [44, 36, 104, 96]（x座標が左右反転して追従）
```

生成される `03_bbox_mask_joint.png` は、左に「変換前（赤枠の bbox ＋ 黄色の mask）」、右に「左右反転後」を並べます。物体が画像の左寄りから右寄りへ移り、**bbox と mask がぴたりと追従している**のが見て取れます。もし画像だけ反転して bbox/mask を放置すれば、枠と物体がズレた壊れた教師データになります。使い分けの結論はシンプルです——**分類だけなら torchvision v2 で十分、検出/セグメで bbox/mask を扱うなら albumentations**。本講座では検出（第18回〜）・セグメ（第21回〜）でこの同時変換が効いてきます。

<figure class="lec-fig"><svg viewBox="0 0 640 320" role="img" aria-label="水平反転で物体が左から右へ移り、bbox(赤)とmask(橙)が同じ幾何変換で追従する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="130" y="58" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">変換前</text><rect x="40" y="72" width="180" height="180" fill="#18181b" stroke="#3f3f46" stroke-width="1.5"/><circle cx="116" cy="164" r="37" fill="#f97316"/><rect x="74" y="122" width="84" height="85" fill="none" stroke="#dc2626" stroke-width="3"/><text x="130" y="274" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">bbox = [24, 36, 84, 96]</text><line x1="232" y1="162" x2="406" y2="162" stroke="#c2410c" stroke-width="3"/><polygon points="418,162 404,153 404,171" fill="#c2410c"/><text x="320" y="118" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">HorizontalFlip</text><text x="320" y="140" text-anchor="middle" font-size="12" fill="#52525b">bbox・mask も追従</text><text x="320" y="202" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">x′ = W − x（W=128）</text><text x="510" y="58" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">左右反転後</text><rect x="420" y="72" width="180" height="180" fill="#18181b" stroke="#3f3f46" stroke-width="1.5"/><circle cx="524" cy="164" r="37" fill="#f97316"/><rect x="482" y="122" width="84" height="85" fill="none" stroke="#dc2626" stroke-width="3"/><text x="510" y="274" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">bbox = [44, 36, 104, 96]</text></svg><figcaption>画像を水平反転すると、物体は<b>左から右へ</b>移ります。このとき <b>bbox（赤枠）と mask（橙）も同じ幾何変換で動かさない</b>と、教師データの「画像と正解の対応」が壊れます。albumentations に <code>image=</code>／<code>mask=</code>／<code>bboxes=</code> を同時に渡せば、座標は <b>x′ = W − x</b>（W=128）の鏡映で自動追従し、<code>[24,36,84,96] → [44,36,104,96]</code> のように更新されます。</figcaption></figure>

## 7. 学習時拡張・推論時決定論の原則

データ拡張で**最も間違えやすい運用ルール**が「**拡張は学習時だけ、推論/評価は決定論**」です。学習時はランダムな拡張で汎化を上げたい一方、**推論や評価のときに毎回ランダムに変形してしまうと、同じ画像でも結果がブレて再現性が失われます**。だから評価用の前処理は、ランダム要素を一切含まない**決定論的なパイプライン**——たとえば「固定サイズにリサイズ → 中央を切り抜く（CenterCrop）→ 正規化」——にします。`RandomResizedCrop` ではなく `Resize`＋`CenterCrop`、`RandomHorizontalFlip` は入れない、という具合です。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="同じ1枚の画像を学習用と評価用の2経路に通す。学習はランダム拡張で毎回不一致、評価は決定論前処理で毎回一致" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="262" y="44" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">学習時（拡張あり・ランダム）</text><text x="262" y="210" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">推論・評価時（決定論）</text><rect x="18" y="131" width="92" height="58" rx="6" fill="#f4f4f5" stroke="#52525b" stroke-width="1.8"/><text x="64" y="157" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">入力画像</text><text x="64" y="175" text-anchor="middle" font-size="10.5" fill="#52525b">同じ1枚</text><rect x="150" y="54" width="224" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="262" y="76" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">ランダム拡張</text><text x="262" y="92" text-anchor="middle" font-size="9.5" fill="#ea580c">RandomResizedCrop /</text><text x="262" y="105" text-anchor="middle" font-size="9.5" fill="#ea580c">HFlip / ColorJitter</text><rect x="406" y="54" width="96" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="454" y="88" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">Normalize</text><rect x="534" y="58" width="108" height="48" rx="6" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="588" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">× 不一致</text><text x="588" y="97" text-anchor="middle" font-size="9.5" fill="#52525b">毎回ランダム</text><rect x="150" y="220" width="224" height="58" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="262" y="246" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">決定論前処理</text><text x="262" y="264" text-anchor="middle" font-size="10" fill="#2563eb">Resize / CenterCrop</text><rect x="406" y="220" width="96" height="58" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="454" y="254" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">Normalize</text><rect x="534" y="224" width="108" height="48" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><text x="588" y="246" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">○ 一致</text><text x="588" y="263" text-anchor="middle" font-size="9.5" fill="#52525b">毎回同じ</text><line x1="110" y1="160" x2="145" y2="92" stroke="#71717a" stroke-width="1.8"/><polygon points="150,83 149.82,94.17 140.96,89.57" fill="#71717a"/><line x1="110" y1="160" x2="146" y2="240" stroke="#71717a" stroke-width="1.8"/><polygon points="150,249 141.34,241.93 150.46,237.83" fill="#71717a"/><line x1="374" y1="83" x2="393" y2="83" stroke="#c2410c" stroke-width="2.5"/><polygon points="406,83 393,77 393,89" fill="#c2410c"/><line x1="502" y1="83" x2="521" y2="83" stroke="#c2410c" stroke-width="2.5"/><polygon points="534,83 521,77 521,89" fill="#c2410c"/><line x1="374" y1="249" x2="393" y2="249" stroke="#c2410c" stroke-width="2.5"/><polygon points="406,249 393,243 393,255" fill="#c2410c"/><line x1="502" y1="249" x2="521" y2="249" stroke="#c2410c" stroke-width="2.5"/><polygon points="534,249 521,243 521,255" fill="#c2410c"/></svg><figcaption>データ拡張の鉄則は <b>『拡張は学習時だけ・推論/評価は決定論』</b>です。<b>同じ 1 枚</b>の画像を 2 つの経路に通すと、学習用は <code>RandomResizedCrop</code>／<code>HFlip</code>／<code>ColorJitter</code> など<b>ランダム変換</b>を含むため<b>毎回結果が変わり（不一致）</b>、評価用は <code>Resize</code>＋<code>CenterCrop</code> の<b>決定論前処理</b>なので<b>毎回同じ（一致）</b>になります。<code>Normalize</code> は両方に共通です。『同じ画像を 2 回読んで一致するか』は、評価系にランダム拡張が紛れていないかの<b>健康診断</b>になります。</figcaption></figure>

`02_dataset_dataloader.py` はこの原則を**実測で確かめます**。学習用 transform（拡張あり）と評価用 transform（決定論）で、**同じ画像を2回読んで一致するか**を比べると、学習用は2回が一致せず（毎回ランダム＝期待通り）、評価用は完全一致します（毎回同じ＝期待通り）。下の出力がその証拠です。

```text
[決定論チェック] 同じ画像を2回読む
  train(拡張あり): 2回が一致? False  ← False が正しい（毎回ランダム）
  eval (決定論)  : 2回が一致? True   ← True  が正しい（毎回同じ）
```

この「2回読んで一致するか」というチェックは、自分の評価パイプラインに**うっかりランダム拡張が混入していないか**を確かめる簡単な健康診断になります。あわせて、推論時には `model.eval()`（BatchNorm/Dropout を推論モードに切替）と `torch.inference_mode()`（勾配計算を切ってメモリ・速度を節約）を併用するのが定石です——これは前処理の決定論とは別軸ですが、「**学習モードと推論モードを混同しない**」という同じ精神の話で、モデルを扱う次章以降で繰り返し出てきます。本章では「前処理の決定論」を体に染み込ませてください。

## 8. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「テンソル変換 → データ供給 → 拡張」と理解が積み上がるように並べています。01〜03 とミニプロジェクトは結果（図と JSON）を `outputs/12_data_pipeline_augmentation/` に保存し、演習スクリプト（`exercises.py`／`exercises_solutions.py`）は採点結果を画面に表示するだけです。共通の道具（合成データセット生成・正規化統計・逆正規化・格子保存）は `pipeline_helpers.py` にまとめ、各スクリプトはそれを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `pipeline_helpers.py` | 出力先・合成画像フォルダ生成・ImageNet/CLIP 統計・`denormalize`・`chw_to_hwc_uint8`・格子保存。道具箱 |
| `01_tensor_layout_normalize.py` | HWC↔CHW・`ToImage`/`ToDtype(scale=True)`/`Normalize` を1段ずつ観察。ImageNet/CLIP 統計の違い、スケール忘れの罠 |
| `02_dataset_dataloader.py` | 自作 `ShapeFolderDataset`、`DataLoader` でバッチ化、学習/推論 transform の切替、決定論チェック、`num_workers` 計測 |
| `03_augment_v2_albumentations.py` | transforms v2 と albumentations の拡張を可視化、albumentations で bbox/mask 同時変換 |
| `mini_project.py` | **章末ミニプロジェクト**。データ→自作 Dataset→拡張→DataLoader→バッチ統計→自前正規化統計→bbox/mask 同時変換を1本に統合し、図と JSON を出力 |
| `exercises.py` | TODO 形式の演習10問（易→難・自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 演習の模範解答（実行すると全10問 PASS。答え合わせ・教材検証用） |

`pipeline_helpers.py` だけは「読み物」ではなく「再利用する道具」です。合成データセットは `outputs/.../synthetic_dataset/<class>/*.png` に作られ、**`data/synth_shapes/` に自分の画像フォルダを置けばそちらが自動で優先**されます（実画像で試す導線）。まず helper に目を通してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。

## 9. 動かし方

このモジュールは深層学習トラックの入口なので、`dl`（torch・torchvision）と `aug`（albumentations）の依存グループが必要です。ネット接続もデータセットのDLも不要で、合成画像が自動生成されるのでいきなり実行できます。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループを追加（初回のみ）。torch/torchvision は CPU ホイールが入る
uv sync --group dl --group aug

# 各スクリプトを実行（結果は outputs/12_data_pipeline_augmentation/ に保存される）
uv run python lectures/12_data_pipeline_augmentation/01_tensor_layout_normalize.py
uv run python lectures/12_data_pipeline_augmentation/02_dataset_dataloader.py
uv run python lectures/12_data_pipeline_augmentation/03_augment_v2_albumentations.py

# 章末ミニプロジェクト: この回の要素を統合した総合課題（図＋JSON を出力）
uv run python lectures/12_data_pipeline_augmentation/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/12_data_pipeline_augmentation/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/12_data_pipeline_augmentation/exercises.py
# 模範解答そのもの（実行すると全10問 PASS）
uv run python lectures/12_data_pipeline_augmentation/exercises_solutions.py
```

実行後は `outputs/12_data_pipeline_augmentation/` の図を開いて解説と照らし合わせてください。とくに `01_layout_normalize.png`（並び替え・スケール・正規化の各段を逆正規化して表示）、`03_aug_v2.png` / `03_aug_albumentations.png`（同じ画像から生まれる8通りの拡張）、`03_bbox_mask_joint.png`（反転に bbox/mask が追従）を見比べると、本章の要点が視覚的に腑に落ちます。**自分の画像で試したい**場合は、`data/synth_shapes/<クラス名>/*.png` のようにクラスごとのフォルダを作って画像を置けば、合成画像の代わりにそちらが読まれます。

## 10. よくあるエラーと対処（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。前処理のバグは「エラーは出ないのに学習が進まない」形で表れることが多く、原因を知らないと延々ハマります。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| Loss が NaN / 全く収束しない | `scale=True` を忘れ 0〜255 のまま Normalize | `ToImage()`→`ToDtype(scale=True)`→`Normalize` の順を守る |
| 値が 1/255 のように極端に小さい | 自前 `/255` と `ToTensor` で二重スケーリング | スケーリングは1パイプラインに1回だけ。v1/v2 を混ぜない |
| CLIP のゼロショット精度が低い | ImageNet 統計を CLIP に流用 | CLIP 専用 mean/std を使う（processor に任せる） |
| 形が合わずエラー（channels が3でない等） | HWC のまま渡した／CHW を忘れた | `ToImage()` か `.permute(2,0,1)` で CHW にする |
| 評価結果が毎回ブレる | 推論時にランダム拡張が混入 | 評価は決定論 transform（Resize＋CenterCrop）に分ける |
| `num_workers>0` でプロセスが暴走/落ちる | `__main__` ガードが無い | 実行コードを `if __name__ == "__main__":` の中へ |
| 正規化後の画像を表示すると色が壊れる | 負値を含むテンソルをそのまま imshow | `denormalize`（x*std+mean）で `[0,1]` に戻してから表示 |
| albumentations に PIL/Tensor を渡してエラー | 入出力が numpy(HWC,uint8) である | `np.asarray(pil_img)` で numpy にしてから渡す |
| 反転後に枠と物体がズレる | 画像だけ変換し bbox/mask を放置 | `bbox_params` を指定し `image/mask/bboxes` を同時に渡す |

この9項目が本章で遭遇しがちな不具合のほぼ全てです。とくに上3つ（スケール忘れ・二重スケーリング・統計の取り違え）は「**エラーは出ないのに結果がおかしい**」沈黙のバグなので、症状を見たら原因を即座に言い当てられるようにしておきましょう。

## 11. まとめ

本章では、深層CVの土台となる**データパイプライン**を、HWC↔CHW の並び替え・`ToImage`/`ToDtype(scale=True)`/`Normalize` の3段、ImageNet と CLIP の統計の違い、自作 `Dataset` と `DataLoader` によるバッチ化、`num_workers` の勘所、transforms v2 と albumentations による拡張、そして「学習時拡張・推論時決定論」の原則まで、すべて合成画像の上で「自分で再現し、数値と図で確認できる」レベルで扱いました。通底するのは「**前処理はモデルが学習された入力分布に合わせる必然**」「**ランダム性は学習時だけ**」という2つの発想です。

ここで身につけた「画像をテンソルに整え、Dataset/DataLoader で供給し、拡張で水増しする」骨格は、次の**第13回（ResNet/ViT 分類・転移学習）**でそのまま学習ループの入力側になります。まずは演習を全問 PASS させ、`01_layout_normalize.png` の各段が何をしているか、`03_bbox_mask_joint.png` でなぜ bbox/mask を同時に動かす必要があるかを、自分の言葉で説明できるようにしてから次へ進んでください。

---

## 🛠 章末ミニプロジェクト — データパイプラインを「データ→Dataset→拡張→バッチ→統計」で一気通貫に組む

ここまで、テンソル変換・自作 Dataset・拡張・DataLoader をバラバラに学んできました。最後にそれらを**1 本のミニ学習前処理パイプライン**へ束ね、本章の技能が「単独で使える」だけでなく「つながって動く」ことを体感します。これは第13回以降の学習ループに**そのまま接続する入力側の雛形**です。実装は `mini_project.py` にあり、実行すると図と総合レポート（JSON）が `outputs/12_data_pipeline_augmentation/` に出ます。

パイプラインは本章の核を順に踏む7段です。**(1) Dataset** ——合成データセット（円/四角/三角）を `ShapeFolderDataset` で『フォルダ＝ラベル』として読む（遅延読み込み）。**(2) 3系統の transform** ——学習用（ランダム拡張あり）・推論用（決定論 Resize+CenterCrop）・統計用（Normalize を外し `[0,1]` で止める）を組み分ける。**(3) DataLoader でバッチ化** ——`(B,C,H,W)` の1バッチに積み、per-channel 平均・標準偏差を確認する。**(4) 拡張の多様性** ——同じ1枚に学習拡張を8回かけ、逆正規化して並べ「水増し」を可視化。**(5) 決定論チェック** ——`train=毎回変わる(False)` / `eval=毎回同じ(True)` を実測。**(6) 自前の正規化統計** ——データセット全体の per-channel mean/std を計算し、ImageNet 統計と**別物**であること（＝自前データには自前統計を使う筋）を確認。**(7) bbox/mask 同時変換** ——albumentations で画像・bbox・mask を一緒に水平反転し、教師データの対応が崩れないことを確認。

<figure class="lec-fig"><svg viewBox="0 0 650 330" role="img" aria-label="ミニプロジェクトの7段パイプライン。Dataset、transform3系統、DataLoaderで構築し、拡張8通り、決定論チェック、自前統計、bbox/mask同時変換で検証する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="325" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">7 段で組むデータパイプライン</text><rect x="20" y="62" width="130" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="85" y="87" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">1. Dataset</text><text x="85" y="106" text-anchor="middle" font-size="10.5" fill="#2563eb">フォルダ＝ラベル</text><rect x="180" y="62" width="130" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="245" y="87" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">2. transform</text><text x="245" y="106" text-anchor="middle" font-size="10.5" fill="#2563eb">学習/推論/統計</text><rect x="340" y="62" width="130" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="405" y="87" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">3. DataLoader</text><text x="405" y="106" text-anchor="middle" font-size="10.5" fill="#2563eb">バッチに積む</text><rect x="500" y="62" width="130" height="56" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="565" y="87" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">4. 拡張 ×8</text><text x="565" y="106" text-anchor="middle" font-size="10.5" fill="#ea580c">多様性を可視化</text><rect x="500" y="212" width="130" height="56" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="565" y="237" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">5. 決定論チェック</text><text x="565" y="256" text-anchor="middle" font-size="10.5" fill="#ea580c">再現性を点検</text><rect x="340" y="212" width="130" height="56" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="405" y="237" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">6. 自前統計</text><text x="405" y="256" text-anchor="middle" font-size="10.5" fill="#ea580c">mean/std を算出</text><rect x="180" y="212" width="130" height="56" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="245" y="237" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">7. bbox/mask</text><text x="245" y="256" text-anchor="middle" font-size="10.5" fill="#ea580c">同時変換</text><line x1="150" y1="90" x2="167" y2="90" stroke="#c2410c" stroke-width="2.5"/><polygon points="180,90 167,84 167,96" fill="#c2410c"/><line x1="310" y1="90" x2="327" y2="90" stroke="#c2410c" stroke-width="2.5"/><polygon points="340,90 327,84 327,96" fill="#c2410c"/><line x1="470" y1="90" x2="487" y2="90" stroke="#c2410c" stroke-width="2.5"/><polygon points="500,90 487,84 487,96" fill="#c2410c"/><line x1="565" y1="118" x2="565" y2="199" stroke="#c2410c" stroke-width="2.5"/><polygon points="565,212 559,199 571,199" fill="#c2410c"/><line x1="500" y1="240" x2="483" y2="240" stroke="#c2410c" stroke-width="2.5"/><polygon points="470,240 483,234 483,246" fill="#c2410c"/><line x1="340" y1="240" x2="323" y2="240" stroke="#c2410c" stroke-width="2.5"/><polygon points="310,240 323,234 323,246" fill="#c2410c"/></svg><figcaption>章末ミニプロジェクトは本章の要素を <b>7 段のひと続きのパイプライン</b>に束ねます。<b>1.Dataset</b>（フォルダ＝ラベルで遅延読み込み）→ <b>2.transform を 3 系統</b>（学習/推論/統計）→ <b>3.DataLoader</b> でバッチに積む、までが<b>パイプラインの構築</b>（青）。続く <b>4.拡張×8</b>・<b>5.決定論チェック</b>（train は不一致・eval は一致）・<b>6.自前の正規化統計</b>・<b>7.bbox/mask の同時変換</b>が<b>検証と分析</b>（橙）で、<code>mini_project.py</code> がこの流れを実行して図と JSON を出力します。</figcaption></figure>

```bash
uv run python lectures/12_data_pipeline_augmentation/mini_project.py
# → mini_pipeline_overview.png（原画像→決定論eval→学習拡張の概観）、mini_aug_grid.png（同一画像からの拡張8通り）、
#    mini_joint_transform.png（bbox/mask が反転に追従）、mini_report.json（機械可読の総合レポート）
```

`mini_report.json` には、バッチ形状と正規化後の per-channel 統計、自前 mean/std と ImageNet 統計の比較、決定論チェックの結果、bbox の反転前後座標とマスク面積が機械可読でまとまります。**発展課題**として、(a) `build_transforms` の `size` や `ColorJitter` の振れ幅を変えると拡張多様性と統計がどう動くか、(b) `data/synth_shapes/<クラス名>/*.png` に自分の画像を置いて実画像で完走させ自前統計を作ってみる、(c) 自前統計を `Normalize` に挿し替えてバッチの per-channel 平均が `0` 付近・標準偏差が `1` 付近に寄ることを確認する、を試してみてください。

## ✅ 到達チェックリスト

この章を「できた」と言える基準です。手を動かして、できる／説明できるの両方を確認してください。

- [ ] **できる**: `v2.ToImage()` → `v2.ToDtype(torch.float32, scale=True)` → `v2.Normalize(mean, std)` の3段を組み、各段で dtype・shape・値域がどう変わるかを言える。
- [ ] **できる**: ImageNet 統計と CLIP 専用統計を使い分け、取り違えると入力分布がズレることを説明できる。
- [ ] **できる**: スケール忘れ（`scale=False` のまま Normalize）で値が桁外れに膨らむ「沈黙のバグ」を再現し、原因を即答できる。
- [ ] **できる**: `__len__` と `__getitem__` だけのカスタム `Dataset` を書き、`DataLoader` で `(B,C,H,W)` のバッチに積める。
- [ ] **できる**: 学習用（ランダム拡張）と推論用（決定論 Resize+CenterCrop）の transform を組み分けられる。
- [ ] **できる**: 同じ画像を2回変換して「学習=不一致(False)・推論=一致(True)」を確認し、評価にランダム拡張が紛れていないか健康診断できる。
- [ ] **できる**: albumentations で画像・bbox・mask を**同時に**変換し、`bbox_params`／`label_fields` の作法を使える。
- [ ] **できる**: データセットの per-channel mean/std を `dim=(0,2,3)` の集約で計算し、自前の正規化統計を作れる。
- [ ] **説明できる**: なぜ HWC→CHW・/255・正規化が「お作法」ではなく「モデルが学習された入力分布に合わせる必然」なのか。
- [ ] **説明できる**: `num_workers>0` が効く場面と、`if __name__ == "__main__":` ガードが必須な理由。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/12_data_pipeline_augmentation/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. **HWC・uint8(0〜255) の numpy 画像を CHW・float32(0〜1) テンソルへ変換する**（`ex1_to_chw_float` の TODO）。`from_numpy` → `permute(2,0,1)` で並び替え、`float()` して 255 で割る。
2. **CHW float[0,1] テンソルをチャンネルごとに `(x-mean)/std` で正規化する**（`ex2_normalize` の TODO）。mean/std を `(C,1,1)` に整形してブロードキャストする。
3. **正規化の逆 `x*std+mean` で [0,1] スケールに戻す**（`ex3_denormalize` の TODO）。可視化のために正規化を打ち消す逆変換。
4. **推論/評価用の決定論的な `v2.Compose` を組む**（`ex4_build_eval_transform` の TODO）。`ToImage`→`Resize`→`ToDtype(scale=True)`→`Normalize` をランダム要素なしで並べる。
5. **`root/<class>/*.png` を走査して `(画像パス, ラベル)` 一覧と `class_to_idx` を作る**（`ex5_scan_folder` の TODO）。クラス名とファイル名を sorted で並べ、クラス順→ファイル名順に samples を作る。
6. **CHW・float[0,1] テンソルを HWC・uint8 の numpy 配列へ逆変換する**（`ex6_chw_to_hwc_uint8` の TODO）。`clamp(0,1)`→`*255`→`round`→uint8→`permute(1,2,0)` で表示用に戻す。
7. **`[(画像CHW, ラベル), ...]` のリストを `(B,C,H,W)` バッチとラベルテンソルに積む**（`ex7_collate_batch` の TODO）。`torch.stack` で画像を積み、ラベルを long テンソルにする collate の最小版。
8. **学習用の“ランダム拡張あり”な `v2.Compose` を組む**（`ex8_build_train_transform` の TODO）。`RandomResizedCrop`/`RandomHorizontalFlip`/`ColorJitter` を含め、同じ入力でも毎回変わる構成にする。
9. **pascal_voc 形式 bbox `[x0,y0,x1,y1]` を画像幅で水平反転する**（`ex9_hflip_bbox` の TODO）。`new_x0 = width - x1`, `new_x1 = width - x0` で x を鏡映し、y は不変・x0<x1 の順を保つ。
10. **`(N,C,H,W)` float テンソルの per-channel mean/std を計算する**（`ex10_dataset_mean_std` の TODO）。`dim=(0,2,3)` で集約し、自前の正規化統計（どちらも shape=(C,)）を作る。

## ❓ よくある落とし穴・FAQ・デバッグ

実装中に詰まったら、まずここを見てください。この章のバグはほぼ「軸順」「スケール」「ランダム性の混入」のどれかに集約されます。

- **Q. Loss が NaN になる／まったく収束しない。** → `scale=True`（/255）を忘れて 0〜255 のまま `Normalize` に入れていませんか。`ToImage()` → `ToDtype(scale=True)` → `Normalize` の順を守ると値域は妥当（おおむね `[-2.6, 2.7]`）に収まります。`01` の (B) ケースが桁外れ（最大 800 超）になる様子を再現して原因を体に入れてください。
- **Q. 値が `1/255` のように極端に小さい。** → 自前の `img/255.0` と v1 の `ToTensor()`（中で /255 する）を二重にかけています。スケーリングは1パイプラインに1回だけ・v1 と v2 を混ぜない、を徹底します。
- **Q. CLIP のゼロショット精度が低い。** → ImageNet 統計を CLIP に流用しています。CLIP 専用 `mean/std` を使う（実務では `AutoImageProcessor` に任せて手打ちのズレを避ける）。
- **Q. 形が合わずエラー（channels が 3 でない等）。** → HWC のまま渡しています。`ToImage()` か `.permute(2,0,1)` で CHW にしてから渡します。
- **Q. 評価結果が毎回ブレる。** → 推論時にランダム拡張（`RandomResizedCrop`/`RandomHorizontalFlip`）が混入しています。評価は決定論 transform（`Resize`＋`CenterCrop`）に分け、「同じ画像を2回読んで一致するか」で点検します。
- **Q. `num_workers>0` でプロセスが暴走/落ちる。** → 実行コードが `if __name__ == "__main__":` ガードの外にあります。ワーカーがモジュールを再 import するため、ガードが無いと無限にプロセスが増えます。
- **Q. 正規化後の画像を表示すると色が壊れる。** → 負値を含むテンソルをそのまま `imshow` しています。`denormalize`（`x*std+mean`）で `[0,1]` に戻し、`clamp` してから表示します（`chw_to_hwc_uint8` がこれを内側で行います）。
- **Q. albumentations に PIL/Tensor を渡してエラー。** → albumentations の入出力は **numpy(HWC, uint8)** です。`np.asarray(pil_img)` で numpy にしてから渡します（torchvision v2 が PIL/Tensor を扱うのと対照的）。
- **Q. 反転後に枠と物体がズレる。** → 画像だけ変換して bbox/mask を放置しています。`bbox_params` を指定し `image=`／`mask=`／`bboxes=` を**同時に**渡せば、座標は自動追従します。`ex9_hflip_bbox` で「`new_x0 = W - x1`, `new_x1 = W - x0`」という鏡映計算を手で書いて中身を理解してください。
- **デバッグの定石**: 学習が進まない・値が変なときは、まず前処理の最終出力に `print(t.shape, t.dtype, t.min(), t.max())` を挟む。形・dtype・値域の3つを見れば、軸順・スケール・統計のどれが崩れているか一目で切り分けられます。

## 🚀 発展トピック・参考

この章の先に広がるテーマです。興味のある方向へ掘り進めてください。

- **強い拡張（MixUp / CutMix / RandAugment / TrivialAugment）**: 1枚単位の幾何/色変換を超え、画像とラベルを混ぜる・自動で拡張方策を選ぶ手法。torchvision v2 は `v2.MixUp`／`v2.CutMix`／`v2.RandAugment`／`v2.TrivialAugmentWide` を提供します（過学習が強いときの定番）。
- **正規化統計を自分のデータで作る**: 本章のミニプロジェクトで触れた per-channel mean/std の計算は、ドメインが ImageNet と大きく違う（医用・衛星・赤外）データで効きます。全データを1パスして集計するのが定石です。
- **`DataLoader` のチューニング**: `pin_memory=True`（GPU 転送を速く）、`persistent_workers=True`（エポック間でワーカーを使い回す）、`prefetch_factor`（先読み量）など。GPU 学習で供給待ちを潰す実戦的なダイヤルです（第13回〜で実測）。
- **`tv_tensors` と幾何変換の同時適用**: torchvision v2 は `tv_tensors.BoundingBoxes`／`tv_tensors.Mask` を使うと、albumentations のように画像・bbox・mask を v2 だけで同時変換できます（検出/セグメの章で対比）。
- **WebDataset / FFCV など高速データ供給**: 大規模学習では画像を tar/専用フォーマットにまとめてシーケンシャル I/O を最大化します。本章の `Dataset`/`DataLoader` の発展形です。
- 公式ドキュメント: [torchvision transforms v2](https://pytorch.org/vision/stable/transforms.html) ／ [torch.utils.data（Dataset/DataLoader）](https://pytorch.org/docs/stable/data.html) ／ [albumentations](https://albumentations.ai/docs/)

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ albumentations 2.0.8 ／ numpy 2.4.6 ／ Pillow 12.2.0 ／ opencv-python-headless 4.13.0（`cv2` 4.13.0）／ matplotlib 3.10.9。
> 前処理 API は現行の正準形 `torchvision.transforms.v2`（旧 `transforms.ToTensor` は非推奨）に統一。CLIP の正規化統計は transformers v5 系の `CLIPImageProcessor` 既定値に準拠しています。