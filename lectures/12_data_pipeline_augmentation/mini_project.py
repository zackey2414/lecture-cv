"""章末ミニプロジェクト: データパイプラインを「データ→Dataset→拡張→バッチ→統計」で一気通貫に組む。

この回でバラバラに学んだ部品——テンソル変換(HWC→CHW/scale/Normalize)、自作 Dataset、
学習用(拡張あり)と推論用(決定論)の transform、DataLoader でのバッチ化、albumentations の
bbox/mask 同時変換——を、1本の「ミニ学習前処理パイプライン」へ束ねる総合課題です。
第13回以降の学習ループに、ここで組んだ入力側がそのまま接続します。

統合する要素（この回の核を順に踏む）:
  (1) 合成データセット(円/四角/三角)を用意し、自作 Dataset で『フォルダ＝ラベル』を読む
  (2) 学習用(ランダム拡張)・推論用(決定論)・統計用(正規化なし)の3系統 transform を組む
  (3) DataLoader で1バッチ (B,C,H,W) に積み、バッチの形と per-channel 統計を確認する
  (4) 同じ画像に学習拡張を何度もかけて『水増し』の多様性を可視化する
  (5) 「学習=毎回変わる / 推論=毎回同じ」の決定論チェックを実測する
  (6) データセット自前統計(per-channel mean/std)を計算し ImageNet 統計と比べる
  (7) albumentations で画像+bbox+mask を同時に反転し、教師データの対応が崩れないことを確認

設計方針: 各関数は単一責務・早期 return。digit 始まりの 01〜03 は import できないので、
必要な処理(Dataset 等)は自己完結で書き、共通の道具だけ pipeline_helpers から借りる。
合成データで完走し、CPU のみ・ネット不要。どこかで例外が出ても exit 0 を守る。

実行: uv run python lectures/12_data_pipeline_augmentation/mini_project.py
出力: outputs/12_data_pipeline_augmentation/ に mini_*.png と mini_report.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import albumentations as A
import cv2
import numpy as np
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


# ---------------------------------------------------------------------------
# (1) 自作 Dataset（02 と同じ責務だが import 不可なので自己完結で再掲）
# ---------------------------------------------------------------------------
class ShapeFolderDataset(Dataset):
    """class_name/xxx.png レイアウトを読む最小の Dataset。

    Dataset の責務は2つだけ: __len__(総数) と __getitem__(i番目の (画像, ラベル))。
    画像はコンストラクタで全読みせず __getitem__ で都度開く(遅延読み込み)。
    """

    def __init__(self, root: pathlib.Path, transform=None) -> None:
        self.root = pathlib.Path(root)
        self.transform = transform
        # サブフォルダ名をソートしてクラスIDを割り当てる(再現性のため必ずソート)。
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
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
        img = Image.open(path).convert("RGB")  # ここで初めて読む(遅延)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# (2) 3系統の transform — 学習(拡張) / 推論(決定論) / 統計(正規化なし)
# ---------------------------------------------------------------------------
def build_transforms(size: int = 64) -> tuple[v2.Compose, v2.Compose, v2.Compose]:
    """学習用・推論用・統計用の3つの v2.Compose を返す。

    - train_tf : ランダム拡張あり(毎回変わる)。汎化を上げるための水増し。
    - eval_tf  : 決定論(Resize+CenterCrop)。推論/評価は毎回同じ結果にする。
    - stat_tf  : Normalize を外し [0,1] で止める。データ自前統計の計算に使う。
    """
    train_tf = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(size, size), scale=(0.6, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(size + 8, size + 8), antialias=True),
        v2.CenterCrop(size=(size, size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    # 統計用は正規化しない([0,1] のまま)。ここで Normalize すると mean≈0 に寄って意味が無い。
    stat_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_tf, eval_tf, stat_tf


# ---------------------------------------------------------------------------
# (3) DataLoader で1バッチ積み、形と per-channel 統計を見る
# ---------------------------------------------------------------------------
def first_batch(dataset: Dataset, batch_size: int = 8) -> torch.Tensor:
    """DataLoader から最初の1バッチ (B,C,H,W) を取り出して返す。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    xb, _ = next(iter(loader))
    return xb


def channel_stats(batch: torch.Tensor) -> tuple[list[float], list[float]]:
    """(B,C,H,W) バッチの per-channel 平均・標準偏差を (mean, std) のリストで返す。

    N・H・W 方向に集約してチャンネルごとの統計にする(dim=(0,2,3))。
    """
    mean = batch.mean(dim=(0, 2, 3))
    std = batch.std(dim=(0, 2, 3))
    return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std]


# ---------------------------------------------------------------------------
# (6) データセット自前統計(per-channel mean/std)を計算する
# ---------------------------------------------------------------------------
def dataset_mean_std(dataset: Dataset) -> tuple[list[float], list[float]]:
    """データセット全体を [0,1] スケールで読み、per-channel の mean/std を計算する。

    自分のデータで正規化統計を作るときの定石。全画像を (N,C,H,W) に積んで
    N・H・W 方向に集約する。本章は小さいので一括スタックで十分。
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    xb, _ = next(iter(loader))  # 全件を1バッチに(枚数が少ないので可能)
    mean = xb.mean(dim=(0, 2, 3))
    std = xb.std(dim=(0, 2, 3))
    return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std]


# ---------------------------------------------------------------------------
# (4) 学習拡張の多様性を可視化する
# ---------------------------------------------------------------------------
def augmentation_grid(
    dataset: Dataset, train_tf: v2.Compose, index: int = 0, n: int = 8
) -> list[np.ndarray]:
    """同じ1枚に学習拡張を n 回かけ、逆正規化した HWC uint8 のリストを返す。

    学習拡張はランダムなので、同じ入力でも毎回違う絵になる(=水増しの正体)。
    Normalize 済みのテンソルは負値を含むので denormalize で [0,1] に戻してから表示する。
    """
    path, _ = dataset.samples[index]  # type: ignore[attr-defined]
    pil = Image.open(path).convert("RGB")
    out: list[np.ndarray] = []
    for _ in range(n):
        t = train_tf(pil)  # (C,H,W) 正規化済み
        out.append(chw_to_hwc_uint8(denormalize(t, IMAGENET_MEAN, IMAGENET_STD)))
    return out


# ---------------------------------------------------------------------------
# (5) 決定論チェック — 学習は毎回変わる / 推論は毎回同じ
# ---------------------------------------------------------------------------
def determinism_check(
    dataset_root: pathlib.Path, train_tf: v2.Compose, eval_tf: v2.Compose
) -> tuple[bool, bool]:
    """同じ画像を2回変換し、(train_同一?, eval_同一?) を返す。

    期待値は train=False(毎回ランダム)・eval=True(毎回同じ)。
    自分の評価パイプラインにランダム拡張が紛れていないかの健康診断になる。
    """
    sample = sorted((dataset_root / "circle").glob("*.png"))[0]
    pil = Image.open(sample).convert("RGB")
    train_same = bool(torch.allclose(train_tf(pil), train_tf(pil)))
    eval_same = bool(torch.allclose(eval_tf(pil), eval_tf(pil)))
    return train_same, eval_same


# ---------------------------------------------------------------------------
# (7) albumentations で画像+bbox+mask を同時変換する
# ---------------------------------------------------------------------------
def make_object_scene(size: int = 128) -> tuple[np.ndarray, list[float], np.ndarray]:
    """1物体のシーンを作る: (image HWC uint8, bbox[pascal_voc], mask H×W)。"""
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    x0, y0, x1, y1 = 24, 36, 84, 96  # 左寄りに置く(反転で位置が動くのが分かる)
    img[y0:y1, x0:x1] = (60, 200, 120)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return img, [float(x0), float(y0), float(x1), float(y1)], mask


def joint_hflip(
    scene: np.ndarray, bbox: list[float], mask: np.ndarray
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """画像・bbox・mask を同じ水平反転で動かし、変換後の (image, bbox, mask) を返す。"""
    joint = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    res = joint(image=scene, mask=mask, bboxes=[bbox], labels=[0])
    return res["image"], list(res["bboxes"][0]), res["mask"]


def draw_box(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    """RGB 画像に bbox(pascal_voc) を赤枠で描いて返す(非破壊)。"""
    out = img.copy()
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return out


def overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB 画像にマスク領域を黄色半透明で重ねて返す(可視化用)。"""
    out = img.copy().astype(np.float32)
    sel = mask.astype(bool)
    out[sel] = 0.5 * out[sel] + 0.5 * np.array([255, 255, 0], dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 可視化（パイプライン全体像を1枚に）
# ---------------------------------------------------------------------------
def save_overview(
    dataset_root: pathlib.Path,
    eval_tf: v2.Compose,
    aug_samples: list[np.ndarray],
    out_path: pathlib.Path,
) -> None:
    """原画像→決定論(eval)→学習拡張サンプル2枚 を1列に並べた概観図を保存する。"""
    sample = sorted((dataset_root / "square").glob("*.png"))[0]
    pil = Image.open(sample).convert("RGB")
    eval_t = eval_tf(pil)
    panel = [
        np.asarray(pil),  # 原画像(HWC uint8)
        chw_to_hwc_uint8(denormalize(eval_t, IMAGENET_MEAN, IMAGENET_STD)),  # 推論用(決定論)
        aug_samples[0],  # 学習拡張1
        aug_samples[1],  # 学習拡張2
    ]
    titles = ["original", "eval (deterministic)", "train aug #1", "train aug #2"]
    save_image_grid(panel, out_path, titles=titles, ncols=4,
                    suptitle="data pipeline: original -> eval(deterministic) / train(augmented)")


# ---------------------------------------------------------------------------
# パイプライン本体
# ---------------------------------------------------------------------------
def main() -> None:
    out = output_dir()
    torch.manual_seed(0)
    np.random.seed(0)
    device = pick_device()
    print("=== 第12回 章末ミニプロジェクト: データパイプライン一気通貫 ===")
    print(f"device = {device}（本章は前処理が主役なので CPU で完結）\n")

    # (1) 合成データセット＋自作 Dataset
    ds_root = find_or_make_dataset(per_class=8)
    train_tf, eval_tf, stat_tf = build_transforms(size=64)
    train_ds = ShapeFolderDataset(ds_root, transform=train_tf)
    stat_ds = ShapeFolderDataset(ds_root, transform=stat_tf)
    print(f"[Dataset] {len(train_ds)} 枚 / クラス {train_ds.classes} "
          f"(class_to_idx={train_ds.class_to_idx})")

    # (3) DataLoader で1バッチ → 形と per-channel 統計
    batch = first_batch(train_ds, batch_size=8)
    b_mean, b_std = channel_stats(batch)
    print(f"\n[バッチ] shape={tuple(batch.shape)} (B,C,H,W)")
    print(f"  per-channel mean(正規化後)={b_mean}  std={b_std}")

    # (4) 学習拡張の多様性を可視化
    aug_samples = augmentation_grid(train_ds, train_tf, index=0, n=8)
    save_image_grid(aug_samples, out / "mini_aug_grid.png", ncols=4,
                    suptitle="train augmentation diversity (same image, 8 random draws)")
    print(f"\n[拡張] 同じ画像から 8 通りの拡張を生成 -> {out / 'mini_aug_grid.png'}")

    # (5) 決定論チェック
    train_same, eval_same = determinism_check(ds_root, train_tf, eval_tf)
    print("\n[決定論チェック] 同じ画像を2回変換")
    print(f"  train(拡張あり): 2回一致? {train_same}  ← False が正しい")
    print(f"  eval (決定論)  : 2回一致? {eval_same}  ← True  が正しい")

    # (6) データセット自前統計 vs ImageNet 統計
    ds_mean, ds_std = dataset_mean_std(stat_ds)
    print("\n[自前統計] このデータセットの per-channel mean/std を計算")
    print(f"  dataset  mean={ds_mean} std={ds_std}")
    print(f"  ImageNet mean={[round(v, 4) for v in IMAGENET_MEAN]} "
          f"std={[round(v, 4) for v in IMAGENET_STD]}")
    print("  → 合成データの分布は ImageNet と別物。自前データには自前統計を使うのが筋。")

    # (7) albumentations で bbox/mask 同時変換
    scene, bbox, mask = make_object_scene()
    new_img, new_bbox, new_mask = joint_hflip(scene, bbox, mask)
    print("\n[bbox/mask 同時変換]")
    print(f"  bbox 反転前(xyxy)={[round(v, 1) for v in bbox]}")
    print(f"  bbox 反転後(xyxy)={[round(float(v), 1) for v in new_bbox]}  ← x が左右反転")
    panel = [
        overlay_mask(draw_box(scene, bbox), mask),
        overlay_mask(draw_box(new_img, new_bbox), new_mask),
    ]
    save_image_grid(panel, out / "mini_joint_transform.png",
                    titles=["before (bbox + mask)", "after HFlip (follow)"], ncols=2,
                    suptitle="albumentations: image + bbox + mask transformed together")
    print(f"  保存: {out / 'mini_joint_transform.png'}")

    # 概観図
    save_overview(ds_root, eval_tf, aug_samples, out / "mini_pipeline_overview.png")
    print(f"\n[概観] {out / 'mini_pipeline_overview.png'}")

    # 総合レポート(機械可読)
    report = {
        "module": "12_data_pipeline_augmentation",
        "dataset_root": str(ds_root),
        "num_samples": len(train_ds),
        "classes": train_ds.classes,
        "batch_shape": list(batch.shape),
        "batch_channel_mean_after_norm": b_mean,
        "batch_channel_std_after_norm": b_std,
        "dataset_mean": ds_mean,
        "dataset_std": ds_std,
        "imagenet_mean": [round(v, 4) for v in IMAGENET_MEAN],
        "imagenet_std": [round(v, 4) for v in IMAGENET_STD],
        "train_deterministic": train_same,  # 期待 False
        "eval_deterministic": eval_same,    # 期待 True
        "bbox_before": [round(v, 2) for v in bbox],
        "bbox_after_hflip": [round(float(v), 2) for v in new_bbox],
        "mask_area_before": int(mask.sum()),
        "mask_area_after": int(new_mask.sum()),
    }
    (out / "mini_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(f"\n完了。outputs/12_data_pipeline_augmentation/ を確認してください。")
    print("  - mini_pipeline_overview.png : 原画像→決定論eval→学習拡張の概観")
    print("  - mini_aug_grid.png          : 同一画像からの拡張8通り(水増し)")
    print("  - mini_joint_transform.png   : bbox/mask が反転に追従")
    print("  - mini_report.json           : 総合レポート(機械可読)")


if __name__ == "__main__":
    # ミニプロジェクトは「途中で落ちても exit 0」。レポートまで辿り着くことを優先する。
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 例外が発生しましたが終了コードは 0 にします: {type(e).__name__}: {e}")
