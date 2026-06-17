"""第35回 共通ヘルパ — 量子化・枝刈りを「同じ題材」で公平に比較するための部品。

このモジュールが提供するもの:
  - 再現可能な合成分類タスク（小さな 1ch 16x16 画像、6クラス）
  - 2種類の小モデル
      * TinyMLP    … Linear 主体（動的量子化が効く題材）
      * TinyConvNet … Conv 主体 + QuantStub/DeQuantStub（静的PTQ / QAT の題材）
  - 軽い学習ループ（CPU で 1〜2 秒、決定的）
  - 三角評価の道具: accuracy / latency(p50/p99・スループット) / モデルサイズ(MB)
  - 量子化ヘルパ（動的・静的・QAT）と枝刈りヘルパ（非構造化 / 構造化リビルド）

設計方針:
  - 量子化は CPU 向け機能なので device は CPU 固定。`torch.backends.quantized.engine`
    を環境に合わせて明示する（x86/fbgemm or qnnpack）。
  - 「重い学習」はしない。合成タスクは数エポックで高精度に達するよう作ってある。
  - torch.ao.quantization は将来 torchao へ移行予定の API（DeprecationWarning が出る）。
    本講座では CPU で確実に動く正準として torch.ao を使い、torchao は 05 で概念紹介する。
"""

from __future__ import annotations

import copy
import io
import statistics
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    fuse_modules,
    fuse_modules_qat,
    get_default_qat_qconfig,
    get_default_qconfig,
    prepare,
    prepare_qat,
    quantize_dynamic,
)
import torch.nn.utils.prune as prune

# === 定数 =================================================================
MODULE_ID = "35_quantization_pruning"
IMG_SIZE = 16
NUM_CLASSES = 6
IN_DIM = IMG_SIZE * IMG_SIZE  # MLP の入力次元 = 256


# === 環境設定 =============================================================
def quiet_warnings() -> None:
    """学習・量子化の繰り返し警告（torch.ao 非推奨告知など）を出力から抑える。

    教育上の注意点（torch.ao は torchao へ移行予定）は README と 05 で扱うので、
    ここではログを読みやすくするために抑制するだけにする。
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")
    warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")


def configure_backend(num_threads: int = 4) -> str:
    """量子化バックエンド（x86/fbgemm/qnnpack）を選び、計測を安定させる。

    - x86 CPU では fbgemm/x86、Apple Silicon/ARM では qnnpack を使う。
    - スレッド数を絞ると小モデルのベンチが安定する（大量スレッドは dispatch 過多）。
    返り値は選んだエンジン名。
    """
    supported = torch.backends.quantized.supported_engines
    for cand in ("x86", "fbgemm", "qnnpack"):
        if cand in supported:
            torch.backends.quantized.engine = cand
            break
    torch.set_num_threads(num_threads)
    return torch.backends.quantized.engine


def ensure_output_dir() -> Path:
    """結果(図・JSON)の保存先 lectures/35_quantization_pruning/outputs/ を作って返す。"""
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int = 0) -> None:
    """乱数を固定（再現性のため）。"""
    torch.manual_seed(seed)


# === 合成データセット =====================================================
def _class_prototypes() -> torch.Tensor:
    """クラスごとの「お手本パターン」(NUM_CLASSES,1,16,16) を決定的に作る。

    低周波化（平均プーリング）して空間構造を持たせ、CNN でも MLP でも学べる素材にする。
    """
    g = torch.Generator().manual_seed(123)
    protos = torch.randn(NUM_CLASSES, 1, IMG_SIZE, IMG_SIZE, generator=g)
    return F.avg_pool2d(protos, kernel_size=3, stride=1, padding=1)


def make_dataset(n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """お手本 + ノイズ で (X, y) を生成する決定的データ。

    X: (n_per_class*NUM_CLASSES, 1, 16, 16) float32 / y: (N,) long
    """
    protos = _class_prototypes()
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        noise = 0.5 * torch.randn(n_per_class, 1, IMG_SIZE, IMG_SIZE, generator=g)
        xs.append(protos[c] + noise)
        ys += [c] * n_per_class
    x = torch.cat(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(len(y), generator=g)
    return x[perm], y[perm]


def get_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """学習用 / 評価用の固定データを返す (Xtr, ytr, Xte, yte)。"""
    x_tr, y_tr = make_dataset(n_per_class=120, seed=1)
    x_te, y_te = make_dataset(n_per_class=40, seed=2)
    return x_tr, y_tr, x_te, y_te


# === モデル ===============================================================
class TinyMLP(nn.Module):
    """Linear 主体の小分類器。動的量子化が「効く」題材（重みの大半が Linear）。

    構造を引数化してあるので、構造化プルーニング後に「小さい密モデル」へ作り直せる。
    """

    def __init__(self, in_dim: int = IN_DIM, h1: int = 512, h2: int = 256,
                 num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TinyConvNet(nn.Module):
    """Conv 主体の小分類器。静的PTQ / QAT の題材。

    QuantStub/DeQuantStub で「ここから int8 / ここで fp32 に戻す」境界を明示する。
    重みの約 95% が Conv に集中する設計なので、Linear だけ量子化する『動的量子化では
    ほとんど縮まない』ことが観察できる（縮めるには Conv も量子化する静的PTQ が要る）。
    分類ヘッドはあえて小さな Linear 1 枚にして Conv を主役にしている。
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.gpool = nn.AdaptiveAvgPool2d(2)  # 空間情報を少し残す（1x1 だと信号が消える）
        self.fc = nn.Linear(32 * 2 * 2, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = torch.flatten(self.gpool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    @staticmethod
    def fuse_groups() -> list[list[str]]:
        """fuse 対象のレイヤ名グループ（3 つの conv-bn-relu）。"""
        return [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["conv3", "bn3", "relu3"]]


# === 学習 =================================================================
def train_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epochs: int, lr: float = 2e-3, batch: int = 32) -> nn.Module:
    """ミニバッチでさっと学習する（CPU・数エポック・決定的）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for i in range(0, len(x), batch):
            xb, yb = x[i:i + batch], y[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def get_trained_mlp(seed: int = 0, epochs: int = 8) -> nn.Module:
    """学習済み TinyMLP を返す（毎回同じ重みになるよう seed 固定）。"""
    set_seed(seed)
    x_tr, y_tr, _, _ = get_data()
    return train_model(TinyMLP(), x_tr, y_tr, epochs=epochs)


def get_trained_cnn(seed: int = 0, epochs: int = 16) -> nn.Module:
    """学習済み TinyConvNet を返す（毎回同じ重み）。"""
    set_seed(seed)
    x_tr, y_tr, _, _ = get_data()
    return train_model(TinyConvNet(), x_tr, y_tr, epochs=epochs)


# === 三角評価の道具（accuracy / size / latency） =========================
@torch.inference_mode()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """テスト精度（0〜1）。必ず eval + inference_mode で測る。"""
    model.eval()
    pred = model(x).argmax(dim=1)
    return (pred == y).float().mean().item()


def state_dict_size_mb(model: nn.Module) -> float:
    """state_dict をシリアライズした実バイト数(MB)。int8 化で実際に縮むかを測る指標。"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1e6


def benchmark(model: nn.Module, sample: torch.Tensor,
              warmup: int = 10, iters: int = 100) -> dict[str, float]:
    """正しい手順のレイテンシ計測（ウォームアップ→多数回→分位点）。

    返り値: p50/p90/p99(ms)・mean(ms)・throughput(samples/sec)。
    CPU なので GPU の同期(torch.cuda.synchronize)は不要。
    """
    model.eval()
    batch = sample.shape[0]
    with torch.inference_mode():
        for _ in range(warmup):  # JIT/キャッシュを温めて初回コストを計測から除く
            model(sample)
        times_ms: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(sample)
            times_ms.append((time.perf_counter() - t0) * 1e3)
    times_ms.sort()

    def pct(q: float) -> float:
        return times_ms[min(len(times_ms) - 1, int(q * len(times_ms)))]

    p50 = statistics.median(times_ms)
    return {
        "p50": p50,
        "p90": pct(0.90),
        "p99": pct(0.99),
        "mean": statistics.fmean(times_ms),
        "throughput": batch * 1000.0 / p50,
    }


def count_params(model: nn.Module) -> int:
    """総パラメータ数。"""
    return sum(p.numel() for p in model.parameters())


# === 量子化ヘルパ =========================================================
def dynamic_quantize(model: nn.Module) -> nn.Module:
    """動的量子化: Linear/LSTM の重みを int8 化（activation は実行時に量子化）。

    最も手軽で CPU 向き。Conv はほぼ対象外なので CNN ではあまり縮まない。
    """
    return quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)


def static_quantize(model: TinyConvNet, calib_x: torch.Tensor,
                    backend: str = "x86", batch: int = 32) -> nn.Module:
    """静的PTQ: fuse → qconfig → prepare(observer挿入) → キャリブレーション → convert。

    activation の範囲を代表データで事前推定して固定するので、Conv 主体でも int8 で動く。
    """
    q = copy.deepcopy(model).eval()
    fuse_modules(q, q.fuse_groups(), inplace=True)  # conv-bn-relu を1演算に畳む
    q.qconfig = get_default_qconfig(backend)
    prepare(q, inplace=True)  # 各所に observer を挿入
    with torch.inference_mode():  # ★キャリブレーション: activation の min/max を観測
        for i in range(0, len(calib_x), batch):
            q(calib_x[i:i + batch])
    convert(q, inplace=True)  # observer の統計から scale/zero_point を確定し int8 化
    return q


def qat_quantize(model: TinyConvNet, x: torch.Tensor, y: torch.Tensor,
                 backend: str = "x86", epochs: int = 3, lr: float = 5e-4,
                 batch: int = 32) -> nn.Module:
    """QAT(量子化を意識した学習): fake-quant を挟んで微調整 → convert。

    学習で int8 の丸めを模擬し、精度劣化を回復させる。静的PTQ で精度が落ちる時の次の一手。
    """
    q = copy.deepcopy(model).train()
    fuse_modules_qat(q, q.fuse_groups(), inplace=True)  # QAT 用の fuser（train モード対応）
    q.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(q, inplace=True)  # fake-quant(観測+丸め)を挿入
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    for _ in range(epochs):
        for i in range(0, len(x), batch):
            opt.zero_grad()
            loss = F.cross_entropy(q(x[i:i + batch]), y[i:i + batch])
            loss.backward()
            opt.step()
    q.eval()
    convert(q, inplace=True)
    return q


# === 量子化の数値（教材用） ==============================================
def quantize_tensor(x: torch.Tensor, num_bits: int = 8, symmetric: bool = False
                    ) -> tuple[torch.Tensor, float, int]:
    """テンソルを int8 に量子化して (q, scale, zero_point) を返す（手計算版）。

    - 非対称(affine): q = round(x/scale) + zero_point、範囲は [qmin, qmax]
    - 対称: zero_point=0、|x|max を使って scale を決める
    実 API の中身を理解するための最小実装。
    """
    qmin, qmax = (-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1) if symmetric else (0, 2 ** num_bits - 1)
    if symmetric:
        max_abs = float(x.abs().max())
        scale = max_abs / qmax if max_abs > 0 else 1.0
        zero_point = 0
    else:
        x_min, x_max = float(x.min()), float(x.max())
        scale = (x_max - x_min) / (qmax - qmin) if x_max > x_min else 1.0
        zero_point = int(round(qmin - x_min / scale))
        zero_point = max(qmin, min(qmax, zero_point))
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return q, scale, zero_point


def dequantize_tensor(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """量子化テンソルを実数へ戻す: x ≈ scale*(q - zero_point)。"""
    return scale * (q - zero_point)


# === 枝刈りヘルパ =========================================================
def prunable_modules(model: nn.Module) -> list[nn.Module]:
    """枝刈り対象（Linear/Conv2d）の一覧。"""
    return [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def global_unstructured_prune(model: nn.Module, amount: float) -> nn.Module:
    """全層横断のグローバル非構造化 L1 枝刈り（小さい重みから amount 割合を 0 に）。

    ※マスクを掛けるだけ。テンソル形状・dtype は密のまま（実速度・サイズは縮まない）。
    """
    params = [(m, "weight") for m in prunable_modules(model)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    return model


def remove_prune_reparam(model: nn.Module) -> nn.Module:
    """prune.remove でマスクを重みへ「焼き付け」、再パラメータ化を外す。

    便利だが、これでも密テンソルのまま。0 が増えるだけで疎フォーマットにはならない。
    """
    for m in prunable_modules(model):
        if hasattr(m, "weight_mask"):
            prune.remove(m, "weight")
    return model


def weight_sparsity(model: nn.Module) -> float:
    """重み全体のスパース率（0 の割合）。"""
    zeros, total = 0, 0
    for m in prunable_modules(model):
        w = m.weight
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros / total if total else 0.0


def structured_compress_mlp(model: TinyMLP, keep_frac: float) -> TinyMLP:
    """構造化枝刈り → 小さい『密』MLP に作り直す（本当に縮む圧縮）。

    手順:
      1. fc1 の出力ニューロン(=fc2 の入力)を L2 ノルムで採点し、上位 keep_frac を残す。
      2. fc2 の出力ニューロン(=fc3 の入力)も同様に残す。
      3. 残したニューロンだけの小さい TinyMLP を作り、対応する重みをコピーする。
    非構造化と違い、行列そのものが小さくなるので実速度もサイズも縮む。
    """
    w1, b1 = model.fc1.weight.data, model.fc1.bias.data  # (h1, in), (h1,)
    w2, b2 = model.fc2.weight.data, model.fc2.bias.data  # (h2, h1), (h2,)
    w3, b3 = model.fc3.weight.data, model.fc3.bias.data  # (out, h2), (out,)

    keep1 = _topk_rows(w1, keep_frac)  # 残す fc1 出力ニューロン
    keep2 = _topk_rows(w2, keep_frac)  # 残す fc2 出力ニューロン

    small = TinyMLP(in_dim=w1.shape[1], h1=len(keep1), h2=len(keep2),
                    num_classes=w3.shape[0])
    # fc1: 行(出力)を間引く
    small.fc1.weight.data = w1[keep1].clone()
    small.fc1.bias.data = b1[keep1].clone()
    # fc2: 行(出力=keep2) と 列(入力=keep1) の両方を間引く
    small.fc2.weight.data = w2[keep2][:, keep1].clone()
    small.fc2.bias.data = b2[keep2].clone()
    # fc3: 列(入力=keep2) を間引く
    small.fc3.weight.data = w3[:, keep2].clone()
    small.fc3.bias.data = b3.clone()
    small.eval()
    return small


def _topk_rows(weight: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """行(出力ニューロン)を L2 ノルムで採点し、上位 keep_frac の行インデックスを昇順で返す。"""
    n = weight.shape[0]
    keep = max(1, int(round(n * keep_frac)))
    scores = weight.flatten(1).norm(dim=1)  # 各出力ニューロンの大きさ
    idx = torch.topk(scores, keep).indices
    return torch.sort(idx).values


if __name__ == "__main__":
    # 自己テスト: 部品が一通り動くことを確認（uv run python ... quant_lab.py）
    quiet_warnings()
    eng = configure_backend()
    print(f"[quant_lab] backend={eng}, threads={torch.get_num_threads()}")
    x_tr, y_tr, x_te, y_te = get_data()
    print(f"[quant_lab] data: train={tuple(x_tr.shape)} test={tuple(x_te.shape)}")
    mlp = get_trained_mlp()
    cnn = get_trained_cnn()
    print(f"[quant_lab] MLP acc={accuracy(mlp, x_te, y_te):.3f} "
          f"size={state_dict_size_mb(mlp):.3f}MB")
    print(f"[quant_lab] CNN acc={accuracy(cnn, x_te, y_te):.3f} "
          f"size={state_dict_size_mb(cnn):.3f}MB")
    print("[quant_lab] OK")
