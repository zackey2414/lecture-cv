"""第36回 共通ヘルパ — ONNX エクスポート / onnxruntime 実行 / ベンチ / 合成データ。

この回のスクリプト（01〜05・mini_project・演習）から import して再利用する土台。
「単一責務・早期 return・日本語コメント厚め」を守り、各関数は 1 つの仕事だけする。

設計メモ（重要）:
  - torch 2.9+ では ``torch.onnx.export`` の既定が ``dynamo=True``（torch.export 基盤の
    新エクスポータ）に変わった。だが dynamo 経路は ``onnxscript`` パッケージを必須とする。
    本講座の標準環境には onnxscript を入れていないため、ここでは旧来の TorchScript ベース
    エクスポータ（``dynamo=False``）を正準として使う。両者の違いは README とコメントで解説する。
  - 推論は必ず ``model.eval()`` + ``torch.inference_mode()``。CPU 前提（GPU 同期は不要）。
  - 図は matplotlib(Agg)、画像生成は OpenCV headless（imshow は呼ばない）。
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

MODULE_ID = "36_onnx_runtime"
OPSET = 18  # onnxruntime 1.26 が安定に読める opset。上げ過ぎ/下げ過ぎは未対応エラーの元。


# =====================================================================
# 出力先・乱数・スレッド
# =====================================================================

def ensure_output_dir() -> pathlib.Path:
    """lectures/36_onnx_runtime/outputs/ を作って返す（無ければ作成）。"""
    out = pathlib.Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int = 0) -> None:
    """再現性のため torch / numpy の乱数を固定する。"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def pin_threads(n: int = 1) -> None:
    """CPU スレッド数を固定して計測のブレを抑える（公平なベンチの前提）。

    torch 側（intra-op）を固定する。onnxruntime 側は SessionOptions.intra_op_num_threads で
    別途指定する（make_session 参照）。スレッド数を揃えないと比較が不公平になる。
    """
    torch.set_num_threads(n)


# =====================================================================
# 合成データ（色付き図形）と小型モデル
# =====================================================================

# クラス: 0=circle, 1=square, 2=triangle（CPU で一瞬で学習できる易しいタスク）
CLASS_NAMES = ("circle", "square", "triangle")
IMG_SIZE = 32


def make_shape_image(cls: int, rng: np.random.Generator) -> np.ndarray:
    """1 枚の合成図形画像 (H,W,3) uint8 を作る。位置・大きさ・色をランダムに振る。

    OpenCV headless で描画（ウィンドウは開かない）。背景は淡色、図形は濃色にして判別を容易にする。
    """
    import cv2

    h = w = IMG_SIZE
    bg = rng.integers(200, 256, size=3).tolist()
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    color = rng.integers(0, 120, size=3).tolist()
    cx, cy = int(rng.integers(12, 20)), int(rng.integers(12, 20))
    r = int(rng.integers(7, 11))

    if cls == 0:  # circle
        cv2.circle(img, (cx, cy), r, color, -1)
    elif cls == 1:  # square
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    else:  # triangle
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)

    # 軽いノイズ（量子化の精度劣化を観察できる程度の現実味を足す）
    noise = rng.normal(0, 6, size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (3,H,W) float32 [0,1] のテンソルへ。学習・推論共通の前処理。"""
    x = img_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def make_dataset(n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """各クラス n_per_class 枚の合成データ (N,3,H,W), ラベル (N,) を作る。"""
    rng = np.random.default_rng(seed)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for cls in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            xs.append(to_tensor(make_shape_image(cls, rng)))
            ys.append(cls)
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(len(y))  # クラス順をシャッフル
    return x[perm], y[perm]


class TinyCNN(nn.Module):
    """CPU で一瞬に学習できる極小 CNN 分類器（3 クラス）。

    conv 主体の特徴抽出 + Linear のヘッドを持つ。これにより:
      - グラフ最適化（Conv+ReLU 融合など）の効果を観察でき、
      - 動的量子化（Linear/MatMul を int8 化）の効きも体験できる。
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        # 空間情報（=図形の形）を保つため大域平均プールはせず 8x8 特徴を平坦化する。
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_tiny_cnn(epochs: int = 3, seed: int = 0, verbose: bool = True) -> TinyCNN:
    """合成データで TinyCNN を数エポックだけ学習する（数秒で終わる軽量学習）。

    重い学習はしない方針。易しいタスクなので 3 epoch で十分高精度になる。
    """
    set_seed(seed)
    model = TinyCNN()
    x, y = make_dataset(n_per_class=120, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    batch = 32
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"    epoch {ep + 1}/{epochs}  loss={total / len(y):.4f}")
    model.eval()
    return model


# =====================================================================
# ONNX エクスポート / 検証
# =====================================================================

def export_onnx(
    model: nn.Module,
    example: torch.Tensor,
    path: pathlib.Path | str,
    *,
    dynamic_batch: bool = True,
    opset: int = OPSET,
    input_name: str = "input",
    output_name: str = "logits",
) -> pathlib.Path:
    """model を ONNX に書き出す（TorchScript ベース exporter = dynamo=False）。

    - dynamic_batch=True なら 0 次元（バッチ）を動的軸にし、任意バッチで推論できるようにする。
    - 量子化済みモデルや MPS 上のモデルは失敗しやすい。必ず eval() の fp32 モデルから出す。
    """
    model.eval()
    path = pathlib.Path(path)
    dynamic_axes = (
        {input_name: {0: "batch"}, output_name: {0: "batch"}} if dynamic_batch else None
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(path),
            dynamo=False,  # ★ onnxscript 不要の旧経路を明示。既定(True)は onnxscript 必須。
            opset_version=opset,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )
    return path


def check_onnx(path: pathlib.Path | str) -> None:
    """onnx.checker でモデルの整合性（型・shape・opset 等）を検証する。壊れていれば例外。"""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def make_session(
    path: pathlib.Path | str,
    *,
    optimize: bool = True,
    intra_threads: int = 1,
):
    """CPUExecutionProvider の InferenceSession を作る。

    - optimize=True で全グラフ最適化（定数畳み込み・演算子融合など）を有効化。
    - intra_threads で演算内並列のスレッド数を固定（torch 側と揃えて公平に比較する）。
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    so.intra_op_num_threads = intra_threads
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def run_session(sess, x: np.ndarray) -> np.ndarray:
    """セッションの第 1 入力に x を流して第 1 出力を返す小ヘルパ。"""
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x.astype(np.float32)})[0]


def numerical_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """2 つの出力配列の一致度を返す（最大絶対誤差・最大相対誤差）。

    ONNX 化が正しいかは「torch と onnxruntime の出力が atol/rtol 内で一致するか」で必ず検証する。
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    max_abs = float(np.max(np.abs(a - b)))
    max_rel = float(np.max(np.abs(a - b) / (np.abs(b) + 1e-8)))
    return {"max_abs": max_abs, "max_rel": max_rel}


# =====================================================================
# ベンチマーク / サイズ / 精度
# =====================================================================

def benchmark(fn: Callable[[], object], *, batch_size: int = 1, warmup: int = 10, iters: int = 50) -> dict:
    """正しい手順のベンチ: ウォームアップ → 多数回反復 → p50/p90/p99 と スループット。

    CPU なので同期は不要（GPU なら torch.cuda.synchronize が必須）。ウォームアップで初回の
    JIT/最適化/キャッシュ miss を計測から除外するのが肝。
    """
    for _ in range(warmup):
        fn()
    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(samples_ms)
    p50 = float(np.percentile(arr, 50))
    mean_s = float(arr.mean()) / 1000.0
    return {
        "p50_ms": p50,
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "throughput": batch_size / mean_s if mean_s > 0 else 0.0,
    }


def file_size_mb(path: pathlib.Path | str) -> float:
    """ファイルサイズを MB で返す。"""
    return pathlib.Path(path).stat().st_size / (1024 * 1024)


def accuracy_torch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """torch モデルの top-1 精度。"""
    model.eval()
    with torch.inference_mode():
        pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean())


def accuracy_session(sess, x: torch.Tensor, y: torch.Tensor) -> float:
    """onnxruntime セッションの top-1 精度（fp32 / int8 量子化済みどちらでも）。"""
    logits = run_session(sess, x.numpy())
    pred = logits.argmax(axis=1)
    return float((pred == y.numpy()).mean())


if __name__ == "__main__":
    # ヘルパ単体のスモークテスト（壊れていないかの最小確認）。
    set_seed(0)
    pin_threads(1)
    out = ensure_output_dir()
    print("[helper] output dir:", out)
    model = train_tiny_cnn(epochs=2, verbose=False)
    x, y = make_dataset(n_per_class=20, seed=123)
    print("[helper] torch acc:", round(accuracy_torch(model, x, y), 3))
    p = export_onnx(model, x[:1], out / "_smoke.onnx")
    check_onnx(p)
    sess = make_session(p)
    print("[helper] onnx size MB:", round(file_size_mb(p), 3))
    print("[helper] onnx acc:", round(accuracy_session(sess, x, y), 3))
    print("[helper] OK")
