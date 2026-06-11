"""第38回 共通ヘルパ — 知識蒸留(KD)を「同じ題材」で公平に比較するための部品。

このモジュールが提供するもの:
  - cv2 で合成する図形分類タスク（1ch 32x32・6クラス）。決定的で再現可能。
      * 学習集合は「教師用フル」と「生徒用サブセット(少量)」に分け、
        “生徒はデータが少ない → 教師の暗黙知(soft target)が効く” という KD の王道を再現する。
  - 3種類のモデル
      * TeacherCNN … やや大きめの CNN（torchvision の resnet18 を使う代わりに、CPU で
                      数秒で学習できる中規模 CNN にしてある。役割は resnet18 と同じ）
      * StudentCNN … 小さい CNN（蒸留の主役。素の学習 vs 蒸留学習を比べる）
      * StudentMLP … さらに小さい MLP（圧縮率を体感する比較用）
  - 蒸留の心臓部（自前実装）
      * soften_logits / kd_kl_loss(T^2 スケール) / response_kd_loss(soft+hard の alpha 結合)
  - 特徴量蒸留（FitNets 系）: forward hook で中間特徴を取り、射影層 + 正規化 + MSE で寄せる
  - 評価の道具: accuracy / パラメータ数(圧縮率) / レイテンシ(p50/p99)
  - 図/JSON の保存先 outputs/38_knowledge_distillation/

設計方針:
  - すべて CPU 前提・小データ・少エポックで「数秒〜数十秒」に収める。
  - teacher は必ず eval() + inference_mode() で擬似ラベルを作り、勾配も optimizer も渡さない
    （KD 最大の落とし穴。freeze_module で requires_grad=False にする）。
  - KL は「入力=log_softmax(student/T)・ターゲット=softmax(teacher/T)」「reduction='batchmean'」
    「ソフト損失に T**2」を厳守する（数式と勾配スケールを正しく保つため）。
"""

from __future__ import annotations

import statistics
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 定数 =================================================================
MODULE_ID = "38_knowledge_distillation"
IMG_SIZE = 32
CLASS_NAMES = ["circle", "square", "triangle", "ellipse", "cross", "line"]
NUM_CLASSES = len(CLASS_NAMES)


# === 環境設定 =============================================================
def quiet_warnings() -> None:
    """学習の繰り返しで出る非本質的な警告を抑え、ログを読みやすくする。"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.set_num_threads(4)  # 小モデルのベンチは少スレッドの方が安定する


def set_seed(seed: int = 0) -> None:
    """乱数を固定（再現性のため）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_output_dir() -> Path:
    """結果(図・JSON)の保存先 outputs/38_knowledge_distillation/ を作って返す。"""
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


# === 合成データセット（cv2 で図形を描く） ================================
def _draw_shape(class_idx: int, rng: np.random.Generator) -> np.ndarray:
    """クラス class_idx の図形を 1ch 32x32 の uint8 画像として描く（位置/大きさはランダム）。

    クラス: 0=円, 1=四角, 2=三角, 3=楕円, 4=十字, 5=線分。
    背景はうっすらノイズ、図形は明るい線。形が似ているクラス（円↔楕円など）を混ぜて、
    「クラス間の類似構造」を持たせる（教師の soft target が効く土壌になる）。
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cx = int(rng.integers(11, 21))
    cy = int(rng.integers(11, 21))
    r = int(rng.integers(7, 11))
    color = int(rng.integers(180, 256))
    thick = int(rng.integers(1, 3))

    if class_idx == 0:  # 円
        cv2.circle(img, (cx, cy), r, color, thick)
    elif class_idx == 1:  # 四角
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thick)
    elif class_idx == 2:  # 三角
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thick)
    elif class_idx == 3:  # 楕円（円と紛らわしい）
        cv2.ellipse(img, (cx, cy), (r, max(3, r // 2)), 0, 0, 360, color, thick)
    elif class_idx == 4:  # 十字
        cv2.line(img, (cx - r, cy), (cx + r, cy), color, thick)
        cv2.line(img, (cx, cy - r), (cx, cy + r), color, thick)
    else:  # 線分（十字と紛らわしい）
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), color, thick)

    noise = rng.normal(0.0, 14.0, size=img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def make_dataset(n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, y) を生成する決定的データ。X:(N,1,32,32) float32[0,1] / y:(N,) long。"""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        for _ in range(n_per_class):
            xs.append(_draw_shape(c, rng))
            ys.append(c)
    x = torch.from_numpy(np.stack(xs)).float().unsqueeze(1) / 255.0
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    return x[perm], y[perm]


NOISE_FRAC = 0.4  # 生徒サブセットのハードラベルを汚す割合（KD の regularization 効果を見せる）


def make_noisy_labels(y: torch.Tensor, frac: float, seed: int) -> torch.Tensor:
    """ラベルの frac 割合をランダムな別クラスへ書き換えた“ノイジーラベル”を返す。

    現実のデータは少量かつラベルが汚い。クリーンなフルデータで育った teacher の
    soft target は正しい構造を保つので、KD はノイズへの過学習を抑える正則化として効く。
    """
    g = torch.Generator().manual_seed(seed)
    y_noisy = y.clone()
    mask = torch.rand(len(y), generator=g) < frac
    y_noisy[mask] = torch.randint(0, NUM_CLASSES, (int(mask.sum()),), generator=g)
    return y_noisy


def get_data() -> dict[str, torch.Tensor]:
    """KD 実験用に「教師フル学習 / 生徒サブセット(クリーン+ノイジー) / テスト」を返す。

    生徒は少量データ(student subset)で、しかもラベルが一部汚れている。teacher は
    クリーンなフルデータで育つので、その soft target を真似る KD はノイズに惑わされにくい。
    """
    x_tr, y_tr = make_dataset(n_per_class=110, seed=1)  # 教師用フル(クリーン): 660 枚
    x_te, y_te = make_dataset(n_per_class=40, seed=7)   # テスト: 240 枚
    sub = torch.arange(0, len(y_tr), 5)                 # 生徒用サブセット: 1/5 = 132 枚
    x_sub, y_sub = x_tr[sub], y_tr[sub]
    return {
        "x_tr": x_tr, "y_tr": y_tr,
        "x_sub": x_sub, "y_sub": y_sub,
        "y_sub_noisy": make_noisy_labels(y_sub, NOISE_FRAC, seed=42),
        "x_te": x_te, "y_te": y_te,
    }


# === モデル ===============================================================
class TeacherCNN(nn.Module):
    """やや大きめの CNN（teacher 役）。resnet18 の代わりに CPU で速い中規模 CNN。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)            # (N,64,8,8) 中間特徴（特徴量蒸留で使う）
        z = self.pool(feat).flatten(1)
        return self.fc(z)


class StudentCNN(nn.Module):
    """小さい CNN（student 役・蒸留の主役）。teacher より大幅に小さい。"""

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 32 -> 16
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)            # (N, width*2, 8, 8) 中間特徴
        z = self.pool(feat).flatten(1)
        return self.fc(z)


class StudentMLP(nn.Module):
    """小さな MLP（圧縮率の比較用）。

    画像にそのまま MLP を当てると最初の Linear(1024, hidden) が大きく、conv の StudentCNN より
    ずっとパラメータが多い。「蒸留の student には conv 系の小モデルが効率的」という対比に使う。
    """

    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE * IMG_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualHeadStudent(nn.Module):
    """小 CNN バックボーン + 2ヘッド（class ヘッド / distill ヘッド）。

    DeiT の class token / distillation token を CPU で回る小モデルで模した構造（05 で詳説）。
      - 学習時 forward は (cls_logits, dist_logits) を返す。
      - 推論時 forward は両ヘッドの平均ロジットを返す（DeiT と同じ）。
    """

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 8) -> None:
        super().__init__()
        backbone = StudentCNN(num_classes, width)
        self.features = backbone.features
        self.pool = backbone.pool
        self.cls_head = nn.Linear(width * 2, num_classes)     # class token 相当
        self.dist_head = nn.Linear(width * 2, num_classes)    # distillation token 相当

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor):
        z = self.embed(x)
        cls_logits = self.cls_head(z)
        dist_logits = self.dist_head(z)
        if self.training:
            return cls_logits, dist_logits          # 学習時は2つ別々に損失を取る
        return (cls_logits + dist_logits) / 2.0      # 推論時は平均（DeiT 流）


# === 蒸留の心臓部（自前実装） ============================================
def soften_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """温度 T でロジットを軟化した確率分布 softmax(logits/T) を返す。

    T を上げるほど分布がなだらかになり、「2 番手以降の確率（暗黙知）」が露わになる。
    """
    return F.softmax(logits / temperature, dim=1)


def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               temperature: float) -> torch.Tensor:
    """温度付き KL ダイバージェンス（Hinton 蒸留のソフト損失）。

    正準形（順序・log・reduction・T^2 を厳守）:
        soft = KL( softmax(teacher/T) || softmax(student/T) )
             = F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction='batchmean')
        return soft * T**2      # T で割った勾配スケールを元に戻す補正
    入力に log_softmax、ターゲットに softmax を渡すこと（逆にすると学習が崩れる）。
    """
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return soft * (temperature ** 2)


def response_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     targets: torch.Tensor, temperature: float,
                     alpha: float) -> torch.Tensor:
    """レスポンス蒸留の合成損失 = alpha*ソフト(KD) + (1-alpha)*ハード(CE)。

    alpha=0 で素の教師あり学習、alpha=1 でソフトのみ。実務は 0.5〜0.9 が多い。
    """
    soft = kd_kl_loss(student_logits, teacher_logits, temperature)
    hard = F.cross_entropy(student_logits, targets)
    return alpha * soft + (1.0 - alpha) * hard


# === teacher 凍結 =========================================================
def freeze_module(model: nn.Module) -> nn.Module:
    """teacher を凍結する: eval() にし、全パラメータの requires_grad=False にする。

    これを忘れると BN/dropout が動いて擬似ラベルが揺れ、optimizer に teacher の
    パラメータを渡すと誤って更新される（KD 最大の落とし穴）。
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# === 学習・評価ループ =====================================================
def _iterate_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int):
    """ミニバッチを毎エポック別シャッフルで回すジェネレータ。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    for i in range(0, len(y), batch_size):
        idx = perm[i:i + batch_size]
        yield x[idx], y[idx]


# 学習設定（ロック済み・全スクリプトで共有して再現性を担保する）。
TEACHER_EPOCHS = 10
TEACHER_LR = 3e-3
STUDENT_EPOCHS = 40   # ノイジーラベルへ baseline が過学習しきる程度に長めに回す
STUDENT_LR = 5e-3
TEMPERATURE = 4.0
ALPHA = 0.5  # ソフト(KD)とハード(CE)を半々で混ぜる。最適値は T と一緒にスイープで探す


def get_trained_teacher(seed: int = 0) -> "TeacherCNN":
    """クリーンなフルデータで teacher を学習して返す（凍結はしない）。"""
    set_seed(seed)
    teacher = TeacherCNN()
    d = get_data()
    train_supervised(teacher, d["x_tr"], d["y_tr"], epochs=TEACHER_EPOCHS, lr=TEACHER_LR)
    return teacher


def train_supervised(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     epochs: int = 8, lr: float = 2e-3, batch_size: int = 64,
                     seed: int = 0) -> nn.Module:
    """素の教師あり学習（ハードラベル CE のみ）。学習ループを明示的に書く。"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        for xb, yb in _iterate_batches(x, y, batch_size, seed + ep):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)   # forward → loss
            loss.backward()                          # backward
            opt.step()                               # step
    return model


def train_response_kd(student: nn.Module, teacher: nn.Module, x: torch.Tensor,
                      y: torch.Tensor, temperature: float = 4.0, alpha: float = 0.7,
                      epochs: int = 8, lr: float = 2e-3, batch_size: int = 64,
                      seed: int = 0) -> nn.Module:
    """レスポンス蒸留で student を学習する（teacher は凍結して soft target を供給）。"""
    freeze_module(teacher)
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    for ep in range(epochs):
        for xb, yb in _iterate_batches(x, y, batch_size, seed + ep):
            with torch.inference_mode():                 # teacher は勾配不要
                t_logits = teacher(xb)
            opt.zero_grad()
            s_logits = student(xb)
            loss = response_kd_loss(s_logits, t_logits.clone(), yb, temperature, alpha)
            loss.backward()
            opt.step()
    return student


@torch.inference_mode()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """テスト accuracy（0〜1）。eval()+inference_mode() で評価する。"""
    model.eval()
    pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean())


# === 圧縮率・速度 =========================================================
def count_params(model: nn.Module) -> int:
    """学習可能パラメータ総数。"""
    return sum(p.numel() for p in model.parameters())


def compression_ratio(teacher: nn.Module, student: nn.Module) -> float:
    """圧縮率 = teacher のパラメータ数 / student のパラメータ数。"""
    return count_params(teacher) / max(count_params(student), 1)


@torch.inference_mode()
def benchmark(model: nn.Module, sample: torch.Tensor, warmup: int = 3,
              iters: int = 30) -> dict[str, float]:
    """推論レイテンシ(ms) を p50/p99 で計測する（ウォームアップ込み）。"""
    model.eval()
    for _ in range(warmup):
        model(sample)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(sample)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "p50": statistics.median(times),
        "p99": times[min(len(times) - 1, int(0.99 * len(times)))],
    }


# === 特徴量蒸留（FitNets 系） ============================================
class FeatureDistiller:
    """forward hook で teacher/student の中間特徴を取り、射影+正規化+MSE で寄せる。

    使い方:
        fd = FeatureDistiller(teacher.features, student.features,
                              teacher_ch=64, student_ch=16)
        ... 学習ループ内で teacher(xb); student(xb) の後に fd.loss() を足す ...
        fd.remove()   # 学習後にフックを外す

    肝（落とし穴）:
      - student と teacher の特徴次元(チャネル数)は一致しないので射影層(1x1 Conv)が必須。
      - チャネルごとのスケール差を消すため空間方向に正規化してから MSE を取る。
    """

    def __init__(self, teacher_layer: nn.Module, student_layer: nn.Module,
                 teacher_ch: int, student_ch: int) -> None:
        # student の特徴を teacher の次元へ持ち上げる射影層（学習対象）。
        self.projector = nn.Conv2d(student_ch, teacher_ch, kernel_size=1)
        self._t_feat: torch.Tensor | None = None
        self._s_feat: torch.Tensor | None = None
        self._h_t = teacher_layer.register_forward_hook(self._save_teacher)
        self._h_s = student_layer.register_forward_hook(self._save_student)

    def _save_teacher(self, _m, _i, out: torch.Tensor) -> None:
        self._t_feat = out

    def _save_student(self, _m, _i, out: torch.Tensor) -> None:
        self._s_feat = out

    def parameters(self):
        """射影層も optimizer に渡せるようにパラメータを公開する。"""
        return self.projector.parameters()

    def loss(self) -> torch.Tensor:
        """直近 forward の中間特徴で特徴量蒸留 MSE を計算する。"""
        assert self._t_feat is not None and self._s_feat is not None
        proj = self.projector(self._s_feat)                 # 次元合わせ
        t = F.normalize(self._t_feat.flatten(1), dim=1)     # 正規化してスケール差を消す
        s = F.normalize(proj.flatten(1), dim=1)
        return F.mse_loss(s, t)

    def remove(self) -> None:
        """フックを外す（付けっぱなしは事故の元）。"""
        self._h_t.remove()
        self._h_s.remove()


def train_feature_kd(student: nn.Module, teacher: nn.Module, x: torch.Tensor,
                     y: torch.Tensor, teacher_ch: int = 64, student_ch: int = 16,
                     temperature: float = 4.0, alpha: float = 0.7, beta: float = 100.0,
                     epochs: int = 8, lr: float = 2e-3, batch_size: int = 64,
                     seed: int = 0) -> nn.Module:
    """レスポンス蒸留 + 特徴量蒸留(FitNets) を同時に行う学習。

    合成損失 = response_kd + beta * 特徴MSE。beta は特徴項の重み（正規化済みなので大きめ）。
    student.features / teacher.features の末尾出力をフックで取る前提。
    """
    freeze_module(teacher)
    fd = FeatureDistiller(teacher.features, student.features, teacher_ch, student_ch)
    student.train()
    params = list(student.parameters()) + list(fd.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    for ep in range(epochs):
        for xb, yb in _iterate_batches(x, y, batch_size, seed + ep):
            with torch.no_grad():
                t_logits = teacher(xb)        # フックが teacher 特徴を保存
            opt.zero_grad()
            s_logits = student(xb)            # フックが student 特徴を保存
            resp = response_kd_loss(s_logits, t_logits, yb, temperature, alpha)
            loss = resp + beta * fd.loss()
            loss.backward()
            opt.step()
    fd.remove()
    return student


def train_deit_style(student: "DualHeadStudent", teacher: nn.Module, x: torch.Tensor,
                     y_noisy: torch.Tensor, epochs: int = STUDENT_EPOCHS, lr: float = STUDENT_LR,
                     batch_size: int = 64, seed: int = 0) -> "DualHeadStudent":
    """DeiT 流 hard distillation: class ヘッド←正解ラベル / distill ヘッド←teacher の argmax。

    DeiT は soft(KL) ではなく hard distillation（teacher の argmax を CE で学ぶ）を既定にする。
    推論時は student.forward が2ヘッドを平均する（DualHeadStudent 参照）。
    """
    freeze_module(teacher)
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    for ep in range(epochs):
        for xb, yb in _iterate_batches(x, y_noisy, batch_size, seed + ep):
            with torch.inference_mode():
                teacher_label = teacher(xb).argmax(dim=1)   # hard distillation の擬似ラベル
            opt.zero_grad()
            cls_logits, dist_logits = student(xb)
            loss_cls = F.cross_entropy(cls_logits, yb)                    # 正解(ノイジー)ラベル
            loss_dist = F.cross_entropy(dist_logits, teacher_label.clone())  # teacher の argmax
            loss = 0.5 * loss_cls + 0.5 * loss_dist
            loss.backward()
            opt.step()
    return student
