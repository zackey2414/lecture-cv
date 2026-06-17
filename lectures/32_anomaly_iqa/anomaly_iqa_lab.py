"""anomaly_iqa_lab.py — 32_anomaly_iqa の共有ヘルパ（合成データ・PaDiM/PatchCore・IQA・評価）。

この回の全スクリプトはこのモジュールを `import anomaly_iqa_lab as lab` して使います。
方針（教材の制約）:
  - ネットへ出てよいのは **torchvision の resnet18 重み DL のみ**。入力画像はすべて **合成生成**。
  - 重い/衝突する依存（anomalib・pyiqa）は **実行経路では使わず**、PaDiM/PatchCore・各IQA指標を
    **自前実装**する（概念と任意導入は README §9 を参照）。
  - CPU 前提・`model.eval()` + `torch.inference_mode()`・小サイズ（128px）で数十秒以内に完走。
  - resnet18 の重み DL に失敗（オフライン）しても **ランダム初期化にフォールバック**して exit 0。

提供するもの:
  [合成データ]   make_dataset() … 正常(規則テクスチャ) と 異常(キズ/異物) + 画素GTマスク
  [特徴抽出]     FeatureExtractor … resnet18 中間特徴(layer1..3)を位置ごとの埋め込みにする
  [PaDiM]        padim_fit / padim_score … 位置別ガウシアン → マハラノビス距離で異常スコア
  [PatchCore]    coreset_subsample / patchcore_score … 正常パッチのメモリバンク + 最近傍距離
  [IQA]          psnr / ssim / variance_of_laplacian / tenengrad と METRIC_DIRECTION
  [評価]         image_auroc / pixel_auroc / aupr / pro_auc … sklearn の AUROC/AUPR と PRO 概念
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# --- 共通設定 -------------------------------------------------------------------
MODULE_ID = "32_anomaly_iqa"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = os.path.join("data", MODULE_ID)  # 実画像を置けば make_dataset が拾う（任意）
IMG_SIZE = 128  # 入力解像度。128 なら resnet18 の layer1=32x32 / layer2=16 / layer3=8。

# ImageNet 正規化（resnet18 は ImageNet で学習。ランダム初期化時も同じで問題ない）。
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CPU を前提に過剰なスレッドを避けて再現性を上げる。
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def ensure_out_dir() -> str:
    """lectures/32_anomaly_iqa/outputs/ を作って返す。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


# ============================================================================
# 1) 合成データ生成
#   PaDiM/PatchCore は「正常画像が（おおよそ）位置合わせ済み」を前提にする。
#   そこで全画像で共通の規則テクスチャ（織物風の格子）を作り、正常はわずかな
#   個体差（位相ゆらぎ・明るさ）だけ、異常はそこへ局所的な欠陥（キズ/異物）を載せる。
# ============================================================================
def make_normal_image(seed: int) -> np.ndarray:
    """規則的な「正常」テクスチャ（RGB uint8, IMG_SIZE 角）を 1 枚作る。

    位相と明るさだけを個体ごとにわずかに変え、構造は共有する（=位置合わせ済み相当）。
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    freq = 2.0 * np.pi / 16.0  # 16px 周期の格子
    phase_x = rng.uniform(-0.3, 0.3)
    phase_y = rng.uniform(-0.3, 0.3)
    weave = 0.5 + 0.22 * np.sin(freq * xx + phase_x) * np.sin(freq * yy + phase_y)
    weave += 0.10 * np.sin(freq * 0.5 * (xx + yy))  # 斜めの縞を少し重ねる
    weave += rng.normal(0.0, 0.015, size=weave.shape)  # 微小ノイズ（センサノイズ相当）
    brightness = rng.uniform(0.92, 1.08)
    base = np.clip(weave * brightness, 0.0, 1.0)
    # 3 チャネルに弱い色味を付ける（チャネルごとに少し違う倍率）。
    tint = np.array([1.0, 0.96, 0.90], dtype=np.float32)
    img = np.clip(base[..., None] * tint[None, None, :], 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def add_defect(img: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """正常画像に局所欠陥を 1 つ載せ、(欠陥画像, 画素GTマスク bool) を返す。

    欠陥は scratch(キズ=細い線) / blob(異物=塗り潰し円) / contamination(汚れ=多点) の 3 種。
    """
    rng = np.random.RandomState(seed)
    out = img.copy()
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    kind = rng.choice(["scratch", "blob", "contamination"])

    if kind == "scratch":  # 細い線状のキズ（暗い）
        p1 = (int(rng.uniform(20, IMG_SIZE - 20)), int(rng.uniform(20, IMG_SIZE - 20)))
        ang = rng.uniform(0, np.pi)
        length = int(rng.uniform(30, 55))
        p2 = (int(p1[0] + length * np.cos(ang)), int(p1[1] + length * np.sin(ang)))
        color = (int(rng.uniform(10, 50)),) * 3
        cv2.line(out, p1, p2, color, thickness=2)
        cv2.line(mask, p1, p2, 255, thickness=3)
    elif kind == "blob":  # 異物（明るい色付きの塊）
        c = (int(rng.uniform(25, IMG_SIZE - 25)), int(rng.uniform(25, IMG_SIZE - 25)))
        r = int(rng.uniform(8, 16))
        color = tuple(int(v) for v in rng.randint(150, 256, size=3))
        cv2.circle(out, c, r, color, thickness=-1)
        cv2.circle(mask, c, r, 255, thickness=-1)
    else:  # contamination: 小さな汚れを数点
        for _ in range(int(rng.uniform(4, 8))):
            c = (int(rng.uniform(15, IMG_SIZE - 15)), int(rng.uniform(15, IMG_SIZE - 15)))
            r = int(rng.uniform(3, 6))
            shade = int(rng.uniform(10, 60))
            cv2.circle(out, c, r, (shade, shade, shade), thickness=-1)
            cv2.circle(mask, c, r, 255, thickness=-1)
    return out, (mask > 0)


def make_dataset(
    n_train: int = 20, n_test_normal: int = 10, n_test_anom: int = 14, seed: int = 0
) -> dict:
    """異常検知用のデータセットを作る。

    返り値 dict:
      train      … 正常のみ (n_train,)  ← 学習に使う（正常だけで分布を覚える）
      test       … 正常+異常 を連結した評価用画像 (M,)
      labels     … test の image-level ラベル (0=正常, 1=異常)
      masks      … test の画素GTマスク bool (M, H, W)（正常画像は全 False）
    すべて RGB uint8 (N, H, W, 3)。data/32_anomaly_iqa/ に画像があれば将来差し替え可。
    """
    train = np.stack([make_normal_image(seed + i) for i in range(n_train)])

    test_imgs: list[np.ndarray] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []
    # 正常テスト（学習に使っていない seed 帯から）
    for i in range(n_test_normal):
        img = make_normal_image(seed + 1000 + i)
        test_imgs.append(img)
        labels.append(0)
        masks.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool))
    # 異常テスト（正常テクスチャ + 欠陥）
    for i in range(n_test_anom):
        base = make_normal_image(seed + 2000 + i)
        dimg, dmask = add_defect(base, seed + 3000 + i)
        test_imgs.append(dimg)
        labels.append(1)
        masks.append(dmask)

    return {
        "train": train,
        "test": np.stack(test_imgs),
        "labels": np.array(labels, dtype=np.int64),
        "masks": np.stack(masks),
    }


# ============================================================================
# 2) 特徴抽出（resnet18 の中間特徴 → 位置ごとの埋め込み）
#   PaDiM/PatchCore はどちらも「学習済み CNN の中間特徴」を共通の土台にする。
#   layer1..3 の特徴を最も粗くない解像度（layer1）に揃えて連結し、各空間位置を
#   1 本のベクトルとみなす。PaDiM はチャネルをランダムに間引いて次元を抑える。
# ============================================================================
class FeatureExtractor:
    """resnet18 の layer1/layer2/layer3 をフックして位置別埋め込みを作る。"""

    def __init__(self, layers: tuple[str, ...] = ("layer1", "layer2", "layer3"), seed: int = 0):
        from torchvision.models import ResNet18_Weights, resnet18

        torch.manual_seed(seed)  # ランダム初期化時の再現性
        try:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.mode = "pretrained"  # 通常はこちら（初回のみ重みを DL）
        except Exception:  # noqa: BLE001 — オフライン等で DL 失敗 → ランダム初期化で完走
            self.model = resnet18(weights=None)
            self.mode = "random-init"
        self.model.eval()
        self.layers = layers
        self._feats: dict[str, torch.Tensor] = {}
        for name in layers:
            getattr(self.model, name).register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_m, _i, out):
            self._feats[name] = out.detach()

        return hook

    def _preprocess(self, imgs: np.ndarray) -> torch.Tensor:
        """RGB uint8 (N,H,W,3) → ImageNet 正規化済み tensor (N,3,H,W)。"""
        x = imgs.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (0, 3, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(x))

    @torch.inference_mode()
    def embed(self, imgs: np.ndarray, batch: int = 8) -> np.ndarray:
        """画像群 → 位置別埋め込み (N, P, C, ) ではなく (N, Hc, Wc, C) を numpy で返す。

        layer2/3 は layer1 の解像度へ最近傍補間してから連結する（PaDiM/PatchCore 共通）。
        """
        x = self._preprocess(imgs)
        outs: list[np.ndarray] = []
        for s in range(0, len(x), batch):
            self._feats.clear()
            self.model(x[s : s + batch])
            ref = self._feats[self.layers[0]]  # 最も細かい解像度に揃える
            target_hw = ref.shape[-2:]
            parts = []
            for name in self.layers:
                f = self._feats[name]
                if f.shape[-2:] != target_hw:
                    f = F.interpolate(f, size=target_hw, mode="nearest")
                parts.append(f)
            emb = torch.cat(parts, dim=1)  # (b, C, Hc, Wc)
            emb = emb.permute(0, 2, 3, 1).contiguous()  # (b, Hc, Wc, C)
            outs.append(emb.numpy())
        return np.concatenate(outs, axis=0)


def select_dims(total_c: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """チャネルからランダムに n_dims 本だけ選ぶインデックス（PaDiM の次元削減）。"""
    rng = np.random.RandomState(seed)
    n = min(n_dims, total_c)
    return np.sort(rng.choice(total_c, size=n, replace=False))


# ============================================================================
# 3) PaDiM — 位置別ガウシアン + マハラノビス距離
#   正常画像だけを使い、各空間位置 (i,j) ごとに埋め込みベクトルの平均と共分散を推定。
#   テスト画像の各位置を、その位置のガウシアンからのマハラノビス距離で採点する。
# ============================================================================
def padim_fit(emb_normal: np.ndarray, eps: float = 0.01) -> dict:
    """正常埋め込み (N, Hc, Wc, C) から位置別ガウシアンを推定する。

    返り値: mean (P,C), inv_cov (P,C,C), shape (Hc,Wc)。P=Hc*Wc。
    共分散は正則化のため eps*I を足してから逆行列をとる（特異化を防ぐ）。
    """
    n, hc, wc, c = emb_normal.shape
    p = hc * wc
    e = emb_normal.reshape(n, p, c)  # (N, P, C)
    mean = e.mean(axis=0)  # (P, C)
    centered = e - mean  # (N, P, C)
    # 位置ごとの共分散（不偏推定 N-1）。einsum で一括計算。
    cov = np.einsum("npi,npj->pij", centered, centered) / max(1, n - 1)  # (P, C, C)
    cov += eps * np.eye(c, dtype=cov.dtype)[None, :, :]
    inv_cov = np.linalg.inv(cov)  # (P, C, C) を一括で逆行列
    return {"mean": mean, "inv_cov": inv_cov, "shape": (hc, wc)}


def padim_score(emb: np.ndarray, model: dict) -> np.ndarray:
    """テスト埋め込み (N, Hc, Wc, C) → 位置別マハラノビス距離マップ (N, Hc, Wc)。"""
    n, hc, wc, c = emb.shape
    p = hc * wc
    e = emb.reshape(n, p, c)
    diff = e - model["mean"]  # (N, P, C)
    # maha^2 = diff^T inv_cov diff を位置ごとに一括計算。
    left = np.einsum("npi,pij->npj", diff, model["inv_cov"])  # (N, P, C)
    m2 = np.einsum("npj,npj->np", left, diff)  # (N, P)
    maha = np.sqrt(np.clip(m2, 0.0, None))
    return maha.reshape(n, hc, wc)


def upsample_maps(maps: np.ndarray, size: int = IMG_SIZE, smooth_sigma: float = 4.0) -> np.ndarray:
    """異常マップ (N,Hc,Wc) を画像サイズへ拡大しガウシアン平滑化する（画素評価/可視化用）。"""
    out = np.empty((len(maps), size, size), dtype=np.float32)
    for i, m in enumerate(maps):
        up = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        if smooth_sigma > 0:
            up = cv2.GaussianBlur(up, (0, 0), sigmaX=smooth_sigma)
        out[i] = up
    return out


def image_scores(maps: np.ndarray) -> np.ndarray:
    """画素マップ (N,H,W) → image-level スコア。各画像の最大値（最も異常な点）を採用。"""
    return maps.reshape(len(maps), -1).max(axis=1)


# ============================================================================
# 4) PatchCore（簡易版）— 正常パッチのメモリバンク + 最近傍距離
#   PaDiM が「位置ごとにガウシアン」を持つのに対し、PatchCore は正常パッチ埋め込みを
#   そのまま記憶し（coreset で間引く）、テストパッチの最近傍距離を異常スコアにする。
#   位置に縛られない＝多少の位置ずれに強いのが利点。
# ============================================================================
def coreset_subsample(bank: np.ndarray, n_keep: int, seed: int = 0) -> np.ndarray:
    """貪欲な最遠点サンプリング(FPS)で代表点を n_keep 個選び、その行インデックスを返す。

    すでに選んだ点集合への最小距離が最大の点を都度追加する（カバレッジ重視の coreset）。
    """
    m = len(bank)
    n_keep = min(n_keep, m)
    rng = np.random.RandomState(seed)
    first = rng.randint(m)
    selected = [first]
    min_d = np.linalg.norm(bank - bank[first], axis=1)  # 各点→選択集合の最小距離
    for _ in range(1, n_keep):
        idx = int(np.argmax(min_d))
        selected.append(idx)
        d = np.linalg.norm(bank - bank[idx], axis=1)
        min_d = np.minimum(min_d, d)
    return np.array(selected, dtype=np.int64)


def patchcore_fit(emb_normal: np.ndarray, n_keep: int = 800, seed: int = 0) -> dict:
    """正常埋め込み (N,Hc,Wc,C) → coreset 圧縮済みメモリバンク (M,C) を作る。"""
    n, hc, wc, c = emb_normal.shape
    bank = emb_normal.reshape(n * hc * wc, c)
    idx = coreset_subsample(bank, n_keep, seed=seed)
    return {"bank": bank[idx], "shape": (hc, wc)}


def patchcore_score(emb: np.ndarray, model: dict, block: int = 2048) -> np.ndarray:
    """テスト埋め込み → 各位置の「メモリバンク最近傍までの距離」マップ (N,Hc,Wc)。"""
    n, hc, wc, c = emb.shape
    q = emb.reshape(n * hc * wc, c)
    bank = model["bank"]
    out = np.empty(len(q), dtype=np.float32)
    bank_sq = (bank**2).sum(axis=1)  # (M,)
    for s in range(0, len(q), block):  # メモリ節約のためブロック処理
        qb = q[s : s + block]
        # ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q·b を距離行列にして最小を取る
        d2 = (qb**2).sum(axis=1, keepdims=True) + bank_sq[None, :] - 2.0 * qb @ bank.T
        out[s : s + block] = np.sqrt(np.clip(d2.min(axis=1), 0.0, None))
    return out.reshape(n, hc, wc)


# ============================================================================
# 5) 画像品質評価（IQA）
#   参照あり: PSNR / SSIM（劣化前のクリーン画像が必要）。
#   無参照  : variance of Laplacian / Tenengrad（参照なしで鮮鋭度を測る簡易指標）。
#   指標ごとに「高い方が良い/低い方が良い」が違うので METRIC_DIRECTION で管理する。
# ============================================================================
def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """RGB uint8 → グレースケール float[0,1]。"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return gray.astype(np.float32) / 255.0


def psnr(ref: np.ndarray, test: np.ndarray, max_val: float = 1.0) -> float:
    """Peak SNR（参照あり, 高いほど良い）。入力は同形 float[0,1] か uint8。"""
    a = ref.astype(np.float32)
    b = test.astype(np.float32)
    if a.max() > 1.5:  # uint8 を渡された場合は [0,1] に寄せる
        a, b, max_val = a / 255.0, b / 255.0, 1.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0  # 完全一致は便宜上の上限
    return float(10.0 * np.log10((max_val**2) / mse))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """構造的類似度 SSIM（参照あり, 高いほど良い）。ガウシアン窓 11x11 / sigma=1.5。"""
    a = _to_gray_float(ref)
    b = _to_gray_float(test)
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    k, sig = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(a, k, sig)
    mu_b = cv2.GaussianBlur(b, k, sig)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, sig) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, sig) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, sig) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return float(ssim_map.mean())


def variance_of_laplacian(img: np.ndarray) -> float:
    """無参照の鮮鋭度指標（高いほど鮮鋭＝ボケていない）。ラプラシアン応答の分散。"""
    gray = _to_gray_float(img)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def tenengrad(img: np.ndarray) -> float:
    """無参照の鮮鋭度指標（高いほど鮮鋭）。Sobel 勾配エネルギーの平均。"""
    gray = _to_gray_float(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


# pyiqa の metric.lower_better に相当する「良し悪しの向き」表。
# True = 小さいほど良い（誤差/歪み）, False = 大きいほど良い（類似/鮮鋭）。
METRIC_DIRECTION = {
    "psnr": False,
    "ssim": False,
    "variance_of_laplacian": False,
    "tenengrad": False,
    # 参考（pyiqa 任意導入時）: いずれも lower_better=True
    "lpips": True,
    "brisque": True,
    "niqe": True,
    # musiq は大きいほど良い（lower_better=False）
    "musiq": False,
}


# ============================================================================
# 6) 評価指標（異常検知）
#   image-level / pixel-level AUROC と AUPR を sklearn で計算。PRO は概念実装。
# ============================================================================
def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """image-level AUROC（異常=正例 1）。"""
    return float(roc_auc_score(labels, scores))


def image_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """image-level AUPR（average precision, 不均衡で実態を映す）。"""
    return float(average_precision_score(labels, scores))


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """pixel-level AUROC。全画像の全画素を平らに並べて算出。"""
    y = masks.reshape(-1).astype(np.int64)
    s = maps.reshape(-1).astype(np.float64)
    if y.min() == y.max():  # 片側しか無いと AUROC は未定義
        return float("nan")
    return float(roc_auc_score(y, s))


def pixel_aupr(maps: np.ndarray, masks: np.ndarray) -> float:
    """pixel-level AUPR。欠陥画素は全体のごく一部なので AUROC より厳しく実態を映す。"""
    y = masks.reshape(-1).astype(np.int64)
    s = maps.reshape(-1).astype(np.float64)
    if y.max() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def pro_auc(maps: np.ndarray, masks: np.ndarray, n_th: int = 30, max_fpr: float = 0.3) -> float:
    """PRO(Per-Region Overlap) を FPR<=max_fpr で積分し正規化した簡易スコア。

    PRO は「欠陥領域(連結成分)ごとの被覆率を、領域の大きさに依らず平等に平均」する指標。
    大きな欠陥に引きずられず、小さな欠陥も同じ重みで評価できるのが pixel-AUROC との違い。
    ここでは複数しきい値での (FPR, 平均PRO) を作り、FPR<=max_fpr の台形面積を返す。
    """
    # 欠陥の連結成分を画像横断で集める
    regions: list[tuple[int, np.ndarray]] = []  # (画像index, bool領域)
    for i, m in enumerate(masks):
        if not m.any():
            continue
        num, lab = cv2.connectedComponents(m.astype(np.uint8))
        for r in range(1, num):
            regions.append((i, lab == r))
    if not regions:
        return float("nan")

    flat_scores = maps.reshape(len(maps), -1)
    normal_pixels = ~masks  # 正常画素（FPR の分母）
    ths = np.linspace(flat_scores.min(), flat_scores.max(), n_th)

    fprs, pros = [], []
    for t in ths:
        pred = maps >= t
        # FPR: 正常画素のうち誤って異常と判定した割合
        fp = np.logical_and(pred, normal_pixels).sum()
        fpr = fp / max(1, normal_pixels.sum())
        # PRO: 各欠陥領域の被覆率（領域内で異常判定された割合）の平均
        covs = [pred[i][reg].mean() for i, reg in regions]
        fprs.append(fpr)
        pros.append(float(np.mean(covs)))
    fprs, pros = np.array(fprs), np.array(pros)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    keep = fprs <= max_fpr
    if keep.sum() < 2:
        return float("nan")
    area = float(np.trapezoid(pros[keep], fprs[keep]))  # numpy 2.x は trapezoid
    return area / max_fpr  # [0,1] に正規化


# ============================================================================
# 7) 自己テスト（このファイルを直接実行すると一通り動かして exit 0 を確認）
# ============================================================================
def _self_test() -> None:
    print("=" * 64)
    print("anomaly_iqa_lab 自己テスト（mode は resnet18 の重み取得状況で変わる）")
    ds = make_dataset(n_train=8, n_test_normal=4, n_test_anom=6, seed=0)
    print(f"  train={ds['train'].shape}  test={ds['test'].shape}  異常率={ds['labels'].mean():.2f}")

    ext = FeatureExtractor(seed=0)
    print(f"  FeatureExtractor.mode = {ext.mode}")
    emb_tr = ext.embed(ds["train"])
    emb_te = ext.embed(ds["test"])
    print(f"  埋め込み: train={emb_tr.shape} test={emb_te.shape}（最後の次元=連結チャネル数）")

    # PaDiM はチャネルを間引いてから学習
    idx = select_dims(emb_tr.shape[-1], n_dims=100, seed=0)
    model = padim_fit(emb_tr[..., idx])
    maps = upsample_maps(padim_score(emb_te[..., idx], model))
    img_s = image_scores(maps)
    print(f"  PaDiM image-AUROC = {image_auroc(img_s, ds['labels']):.3f}")
    print(f"  PaDiM pixel-AUROC = {pixel_auroc(maps, ds['masks']):.3f}")

    # IQA を 1 組だけ確認
    clean = ds["test"][0]
    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=2.0)
    print(f"  PSNR(clean,blur)={psnr(clean, blurred):.2f}  SSIM={ssim(clean, blurred):.3f}")
    print(
        f"  varLaplacian clean={variance_of_laplacian(clean):.4f} blur={variance_of_laplacian(blurred):.4f}"
    )
    print("  OK: 合成→特徴→PaDiM→評価→IQA が一通り動作。")


if __name__ == "__main__":
    _self_test()
