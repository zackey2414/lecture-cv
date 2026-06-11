"""02: カスタム Dataset と DataLoader — 画像フォルダを“自力で”バッチ化する。

深層学習は「1枚ずつ」ではなく「ミニバッチ（複数枚をまとめたテンソル）」で回します。
その橋渡しが torch.utils.data の2役:
  - Dataset: __len__（総数）と __getitem__（i番目の (画像テンソル, ラベル)）を実装する“1枚を返す係”
  - DataLoader: Dataset から複数枚を取り出し、自動で1つのバッチテンソルに積む“まとめ役”
    （shuffle・batch_size・num_workers・drop_last などを担う）

このスクリプトは『フォルダ＝ラベル』レイアウト（class/xxx.png）を読む ShapeFolderDataset を
書き、学習用（拡張あり）と推論用（決定論）の2系統の transform を切り替えられるようにします。
最後に num_workers の効果を軽く計測します。

実行:
  uv run python lectures/12_data_pipeline_augmentation/02_dataset_dataloader.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline_helpers import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    chw_to_hwc_uint8,
    denormalize,
    find_or_make_dataset,
    output_dir,
    pick_device,
    save_image_grid,
)
from PIL import Image  # noqa: E402


class ShapeFolderDataset(Dataset):
    """class_name/xxx.png レイアウトの画像フォルダを読むカスタム Dataset。

    Dataset がやることは突き詰めると2つだけ:
      __len__   : サンプル総数を返す
      __getitem__(i): i番目の (画像テンソル, ラベル整数) を返す
    transform を差し替えるだけで、学習用（拡張あり）と推論用（決定論）を切り替えられる。
    """

    def __init__(self, root: pathlib.Path, transform=None) -> None:
        self.root = pathlib.Path(root)
        self.transform = transform
        # サブフォルダ名をソートしてクラスIDを割り当てる（再現性のため必ずソート）。
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        # (画像パス, ラベル) の一覧を作る。これが「i番目」の実体。
        self.samples: list[tuple[pathlib.Path, int]] = []
        for name in self.classes:
            for path in sorted((self.root / name).glob("*.png")):
                self.samples.append((path, self.class_to_idx[name]))
        if not self.samples:
            raise RuntimeError(f"画像が見つかりません: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        # 画像読み込みは __getitem__ の中で行う（遅延読み込み＝メモリに優しい）。
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms() -> tuple[v2.Compose, v2.Compose]:
    """学習用（拡張あり）と推論用（決定論）の2系統の transform を作る。

    原則: 拡張（ランダム性）は『学習時だけ』。推論/評価は毎回同じ結果になる決定論的な
    前処理（リサイズ＋中央切り抜き＋正規化）にする。ここを混同すると評価がブレる。
    """
    train_tf = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(64, 64), antialias=True),  # ランダムな拡大切り抜き
        v2.RandomHorizontalFlip(p=0.5),                       # 左右反転
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 色ゆらぎ
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(72, 72), antialias=True),  # 決定論: リサイズして
        v2.CenterCrop(size=(64, 64)),               # 中央を切り抜く（ランダム性なし）
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def time_loader(dataset: Dataset, num_workers: int, epochs: int = 3) -> float:
    """指定 num_workers で全データを epochs 周し、1周あたりの平均秒数を返す。"""
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=num_workers)
    start = time.perf_counter()
    for _ in range(epochs):
        for xb, yb in loader:
            _ = xb.shape  # 取り出すだけ（前処理コストを測りたい）
    return (time.perf_counter() - start) / epochs


def main() -> None:
    out = output_dir()
    torch.manual_seed(0)
    device = pick_device()
    print(f"device = {device}（本章はデータパイプラインが主役なので CPU で完結）")

    ds_root = find_or_make_dataset(per_class=8)
    train_tf, eval_tf = build_transforms()

    # --- 1. Dataset 単体の挙動を確認 ----------------------------------------
    train_ds = ShapeFolderDataset(ds_root, transform=train_tf)
    eval_ds = ShapeFolderDataset(ds_root, transform=eval_tf)
    print(f"\nDataset: {len(train_ds)} 枚 / クラス {train_ds.classes} "
          f"(class_to_idx={train_ds.class_to_idx})")
    x0, y0 = train_ds[0]
    print(f"  train_ds[0] -> tensor shape={tuple(x0.shape)} dtype={x0.dtype}, label={y0}")

    # --- 2. 学習=拡張ありは毎回変わる / 推論=決定論は毎回同じ ----------------
    # 同じインデックスを2回読んで比べる。これが『学習時のみ拡張』原則の核心。
    a1, _ = train_ds[0]
    a2, _ = train_ds[0]
    e1, _ = eval_ds[0]
    e2, _ = eval_ds[0]
    train_same = torch.allclose(a1, a2)
    eval_same = torch.allclose(e1, e2)
    print("\n[決定論チェック] 同じ画像を2回読む")
    print(f"  train(拡張あり): 2回が一致? {train_same}  ← False が正しい（毎回ランダム）")
    print(f"  eval (決定論)  : 2回が一致? {eval_same}  ← True  が正しい（毎回同じ）")

    # --- 3. DataLoader でミニバッチに積む -----------------------------------
    # num_workers=0 は『メインプロセスで読む』。>0 で別プロセス並列読み込み（後で計測）。
    # drop_last=True は端数バッチを捨てる（バッチサイズを揃えたい学習時に使う）。
    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True
    )
    print("\n[DataLoader] batch_size=8, shuffle=True, drop_last=True")
    for bi, (xb, yb) in enumerate(train_loader):
        # ここで初めて (B, C, H, W) のバッチテンソルになる。学習回では xb.to(device)。
        xb = xb.to(device)
        print(f"  batch {bi}: images={tuple(xb.shape)} labels={yb.tolist()}")
        if bi == 0:
            first_batch = xb.detach().cpu()
        if bi >= 2:
            break

    # --- 4. バッチを可視化（逆正規化して並べる）-----------------------------
    panel = [
        chw_to_hwc_uint8(denormalize(first_batch[i], IMAGENET_MEAN, IMAGENET_STD))
        for i in range(first_batch.shape[0])
    ]
    grid_path = out / "02_train_batch.png"
    save_image_grid(panel, grid_path, ncols=4,
                    suptitle="one training batch (augmented, denormalized)")
    print(f"\nバッチを可視化: {grid_path}")

    # --- 5. num_workers の効果をざっくり計測 --------------------------------
    # データが少ないので差は小さいが、『別プロセス並列読み込み』の概念を体感する。
    # 注意: num_workers>0 は __main__ ガードの中で動かすこと（このスクリプトはそうしている）。
    print("\n[num_workers の効果（1エポックの平均秒。値は環境で変動）]")
    timings = {}
    for nw in (0, 2):
        sec = time_loader(train_ds, num_workers=nw)
        timings[nw] = round(sec, 4)
        print(f"  num_workers={nw}: {sec * 1000:.1f} ms/epoch")
    print("  （CPUコア数に応じて調整。多すぎると起動オーバーヘッドで逆に遅くなることもある）")

    # --- 6. 結果を JSON 保存 -------------------------------------------------
    summary = {
        "dataset_root": str(ds_root),
        "num_samples": len(train_ds),
        "classes": train_ds.classes,
        "train_aug_deterministic": bool(train_same),
        "eval_deterministic": bool(eval_same),
        "batch_shape": list(first_batch.shape),
        "worker_timings_sec": timings,
    }
    (out / "02_dataset_dataloader.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n数値を保存: {out / '02_dataset_dataloader.json'}")
    print("要点: Dataset は『1枚を返す』、DataLoader は『まとめてバッチにする』。"
          "拡張は学習時のみ・推論は決定論。")


if __name__ == "__main__":
    main()
