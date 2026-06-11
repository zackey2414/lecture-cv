"""共有ヘルパ: 合成顔データセット / 顔埋め込み(代用) / 認識・クラスタリングの評価。

このモジュール内のスクリプトは `import face_lab as fl` で利用する。
`uv run python lectures/30_face_detection_recognition/01_xxx.py` のように直接実行すれば、
スクリプトのあるディレクトリが sys.path に入るので、そのまま import できる。

設計方針（なぜ「代用」なのか）:
  - 顔埋め込みの定番は insightface(ArcFace) だが、依存衝突（onnxruntime / numpy ピン等）を
    避けるため **本講座の実行経路では使わない**。代わりに導入済みの torchvision/HF バックボーン
    （ResNet18 の最終 512 次元特徴）を「顔埋め込みの代用」として用いる。
  - 入力は **合成顔**（N 人 × M 枚、同一人物は似た見た目になるよう生成）。実写を
    data/30_face_detection_recognition/ に置けば検出スクリプトはそちらも使う。
  - これで「検出 → 整列 → 埋め込み → 1:1照合 / 1:N識別 → クラスタリング」という顔認識の
    パイプライン全体を、CPU・合成データだけで一気通貫に学べる。ArcFace に差し替えるのは
    `FaceEncoder` を 1 つ入れ替えるだけ、という構造にしてある。

ここに置くのは「モデルに依存しない汎用部品」と「顔埋め込みの抽出器」:
  - device 判定
  - 合成顔データセット（外部データ無しで毎回同じものを再現）
  - FaceEncoder（ResNet18 代用埋め込み。重みDL不可でも classical 特徴へ自動フォールバック）
  - L2 正規化 / コサイン類似
  - 認識評価: ROC / EER / TAR@FAR / 1:N 識別(rank-1, CMC)
  - クラスタリング評価: purity / NMI / homogeneity / completeness
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image

MODULE_ID = "30_face_detection_recognition"
OUT_DIR = os.path.join("outputs", MODULE_ID)
DATA_DIR = os.path.join("data", MODULE_ID)  # 実写を置けば検出スクリプトが拾う（任意）

# 顔のキャンバスサイズ。ResNet の前処理で 224 にリサイズされるので 112 で十分。
FACE_SIZE = 112


# ===========================================================================
# device 判定（全スクリプト共通の定石）
# ===========================================================================
def get_device() -> torch.device:
    """cuda → mps → cpu の優先順で利用可能な device を返す（本講座は CPU 前提）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# 合成顔データセット（同一人物は似た見た目／別人は別の見た目）
# ===========================================================================
# 各「人物(identity)」は固定の DNA（肌色・輪郭の縦横・目の色/間隔・眉/口/髪…）を持つ。
# 同じ人物の M 枚は DNA を共有し、照明・わずかな回転/位置/ノイズだけを揺らす。
# こうすると ImageNet 学習済みバックボーンでも「同一人物 → 近いベクトル」に写りやすく、
# 1:1 照合やクラスタリングが意味を持つ。実写の ArcFace ほど分離しないのは想定どおりで、
# それ自体が「埋め込みの良し悪しを評価指標で測る」題材になる。


# 人物ごとに「はっきり違う」見た目になるよう、肌・髪・目の色を十分に離して用意する。
# 実写の顔ほど微妙ではないが、こうしておくと埋め込み空間で同一人物がきれいに固まり、
# 「クラスタリングで人物ごとに自動仕分け」という本章の主題を再現できる。BGR で持つ。
_SKIN_PALETTE = [
    (150, 180, 225),
    (120, 160, 235),
    (170, 150, 200),
    (110, 190, 210),
    (200, 170, 180),
    (130, 200, 240),
    (160, 140, 170),
    (140, 210, 225),
]
_HAIR_PALETTE = [
    (30, 30, 40),
    (40, 60, 150),
    (70, 120, 200),
    (90, 90, 90),
    (20, 20, 20),
    (180, 120, 40),
    (110, 70, 30),
    (160, 160, 170),
]
_EYE_PALETTE = [
    (60, 40, 30),
    (150, 90, 40),
    (40, 140, 60),
    (30, 30, 160),
    (120, 70, 30),
    (110, 110, 110),
    (50, 50, 50),
    (80, 40, 20),
]


def _identity_dna(person_id: int) -> dict:
    """人物 ID から決定的に DNA（顔パーツの固定パラメータ）を作る。

    乱数シードを person_id で固定するので、同じ人物は常に同じ DNA を持つ。色は離散
    パレットから選び、人物間で十分に違う見た目（肌/髪/目の色＋眼鏡/ひげ/髪型）にする。
    """
    rng = np.random.default_rng(1000 + person_id)
    k = len(_SKIN_PALETTE)
    return {
        "skin": _SKIN_PALETTE[person_id % k],
        "hair": _HAIR_PALETTE[person_id % k],
        "eye": _EYE_PALETTE[person_id % k],
        # 輪郭（楕円の半径）。顔の縦長/丸顔の差。
        "face_rx": int(rng.integers(32, 44)),
        "face_ry": int(rng.integers(40, 52)),
        "eye_dx": int(rng.integers(14, 22)),  # 目の間隔（左右中心からのずれ）
        "eye_y": int(rng.integers(-7, 3)),  # 目の高さ
        "eye_r": int(rng.integers(5, 9)),  # 目の大きさ
        "brow": int(rng.integers(2, 5)),  # 眉の太さ
        "mouth_w": int(rng.integers(11, 21)),  # 口の幅
        "smile": int(rng.integers(-6, 11)),  # 口角（+で笑顔）
        "nose_len": int(rng.integers(8, 17)),
        # 人物を強く特徴づけるアクセサリ（同一人物では常に同じ）
        "glasses": person_id % 2 == 0,
        "beard": person_id % 3 == 0,
        "hairfull": person_id % 2 == 1,
    }


def _draw_face(dna: dict, rng: np.random.Generator) -> np.ndarray:
    """DNA + 1 枚ぶんの揺らぎから顔画像（FACE_SIZE 角・BGR uint8）を描く。"""
    s = FACE_SIZE
    cx, cy = s // 2, s // 2 + 4
    # 背景（部屋の壁のつもり。うっすらノイズ）
    bg = int(rng.integers(70, 130))
    img = np.full((s, s, 3), bg, dtype=np.uint8)

    # --- 髪（髪型を 2 通り。顔より一回り大きい楕円）---
    if dna["hairfull"]:
        cv2.ellipse(
            img,
            (cx, cy - 4),
            (dna["face_rx"] + 8, dna["face_ry"] + 8),
            0,
            160,
            380,
            dna["hair"],
            -1,
            cv2.LINE_AA,
        )
    else:
        cv2.ellipse(
            img,
            (cx, cy - 8),
            (dna["face_rx"] + 6, dna["face_ry"] + 4),
            0,
            180,
            360,
            dna["hair"],
            -1,
            cv2.LINE_AA,
        )

    # --- 顔の輪郭（肌色の楕円）---
    cv2.ellipse(
        img, (cx, cy), (dna["face_rx"], dna["face_ry"]), 0, 0, 360, dna["skin"], -1, cv2.LINE_AA
    )

    # --- 目（白目 + 虹彩）と眉 ---
    for sign in (-1, +1):
        ex = cx + sign * dna["eye_dx"]
        ey = cy + dna["eye_y"]
        cv2.circle(img, (ex, ey), dna["eye_r"] + 2, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, (ex, ey), dna["eye_r"], dna["eye"], -1, cv2.LINE_AA)
        cv2.circle(img, (ex, ey), max(1, dna["eye_r"] // 2), (20, 20, 20), -1, cv2.LINE_AA)
        cv2.line(
            img,
            (ex - dna["eye_r"], ey - dna["eye_r"] - 5),
            (ex + dna["eye_r"], ey - dna["eye_r"] - 6),
            (40, 40, 60),
            dna["brow"],
            cv2.LINE_AA,
        )

    # --- 眼鏡（強い個人マーカ。偶数 ID）---
    if dna["glasses"]:
        ey = cy + dna["eye_y"]
        for sign in (-1, +1):
            cv2.circle(
                img, (cx + sign * dna["eye_dx"], ey), dna["eye_r"] + 4, (20, 20, 20), 2, cv2.LINE_AA
            )
        cv2.line(
            img,
            (cx - dna["eye_dx"] + dna["eye_r"] + 4, ey),
            (cx + dna["eye_dx"] - dna["eye_r"] - 4, ey),
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    # --- 鼻 ---
    cv2.line(
        img,
        (cx, cy - 2),
        (cx, cy + dna["nose_len"]),
        tuple(int(c * 0.8) for c in dna["skin"]),
        2,
        cv2.LINE_AA,
    )

    # --- ひげ（強い個人マーカ。3 の倍数 ID）と口（笑顔は下に膨らむ円弧）---
    my = cy + dna["nose_len"] + 8
    if dna["beard"]:
        cv2.ellipse(
            img, (cx, my + 4), (dna["face_rx"] - 6, 14), 0, 0, 180, dna["hair"], -1, cv2.LINE_AA
        )
    cv2.ellipse(
        img,
        (cx, my),
        (dna["mouth_w"], 6 + dna["smile"] // 2),
        0,
        0,
        180,
        (60, 70, 160),
        2,
        cv2.LINE_AA,
    )

    # --- 1 枚ごとの揺らぎ: 明るさ・回転・スケール・ノイズ（同一人物内のばらつき）---
    bright = int(rng.integers(-18, 18))
    img = np.clip(img.astype(np.int16) + bright, 0, 255).astype(np.uint8)
    img = cv2.add(img, rng.integers(0, 10, (s, s, 3), dtype=np.uint8))
    ang = float(rng.uniform(-9, 9))
    rot = cv2.getRotationMatrix2D((s / 2, s / 2), ang, float(rng.uniform(0.95, 1.05)))
    img = cv2.warpAffine(img, rot, (s, s), borderValue=(bg, bg, bg))
    return img


def make_face_dataset(
    n_ids: int = 8, n_per_id: int = 6, seed: int = 0
) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    """合成顔データセットを作る。返り値は (PIL画像リスト, ラベル配列, 人物名リスト)。

    同一人物(ラベル)は DNA を共有し、揺らぎだけが違う M 枚。PIL(RGB) で返すのは HF/torchvision
    の前処理がそのまま受け取れる形だから。cv2 は BGR なので PIL 化の直前に RGB へ変換する。
    """
    images: list[Image.Image] = []
    labels: list[int] = []
    names = [f"person_{i:02d}" for i in range(n_ids)]
    for pid in range(n_ids):
        dna = _identity_dna(pid)
        # 人物ごと・枚数ごとに別シード（だが決定的）にして再現性を持たせる
        for j in range(n_per_id):
            rng = np.random.default_rng(seed * 10_000 + pid * 100 + j)
            bgr = _draw_face(dna, rng)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # ★ BGR→RGB を忘れない
            images.append(Image.fromarray(rgb))
            labels.append(pid)
    return images, np.asarray(labels, dtype=np.int64), names


def compose_group_photo(
    images: list[Image.Image], cols: int = 4, pad: int = 18, return_boxes: bool = False
):
    """顔画像を格子状に並べた「集合写真」BGR canvas を作る（検出デモ用）。

    Haar は合成顔をうまく検出できないことがある（=domain gap）。それでも『大きな 1 枚から
    顔を探す』という検出タスクの形を体験するために使う。実写があれば data/ 側を優先する。
    return_boxes=True なら各顔の正解矩形 (N,4)=[x,y,w,h] も返す（検出評価/識別の突合せ用）。
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fs = FACE_SIZE
    H = rows * (fs + pad) + pad
    W = cols * (fs + pad) + pad
    canvas = np.full((H, W, 3), 200, dtype=np.uint8)  # 明るい壁
    boxes = []
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y = pad + r * (fs + pad)
        x = pad + c * (fs + pad)
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        canvas[y : y + fs, x : x + fs] = bgr
        boxes.append([x, y, fs, fs])
    if return_boxes:
        return canvas, np.asarray(boxes, dtype=np.int64)
    return canvas


def iou_xywh(a, b) -> float:
    """[x,y,w,h] 同士の IoU（検出と正解の突合せに使う）。重ならなければ 0。"""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


# ===========================================================================
# 顔埋め込み（代用）: ResNet18 の 512 次元特徴。重みDL不可なら classical へ自動退避。
# ===========================================================================
class FaceEncoder:
    """顔画像 → 埋め込みベクトルの抽出器。

    primary: torchvision resnet18(ImageNet) の fc 直前 512 次元特徴。初回のみ重みをDL。
    fallback: 重みDLに失敗した（オフライン等）場合でも exit 0 を保つため、cv2 だけの
              classical 特徴（HSV 色ヒストグラム + グレースケールの空間グリッド平均）に退避。
    どちらも『同一人物 → 近いベクトル』になるよう設計してあるが、品質は当然 resnet > classical。
    実運用ではこの 1 クラスを insightface(ArcFace) に差し替えるだけでよい。
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or get_device()
        self.mode = "classical"
        self.model = None
        self.transform = None
        self._try_load_resnet()

    def _try_load_resnet(self) -> None:
        try:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()  # fc を外し、512 次元の GAP 特徴を埋め込みにする
            self.model = model.to(self.device).eval()
            self.transform = weights.transforms()  # モデル固有の前処理（リサイズ/正規化）
            self.mode = "resnet18"
        except Exception as e:  # noqa: BLE001  # 重みDL不可/オフライン等はここで握り潰す
            print(
                f"  [FaceEncoder] resnet18 を使えませんでした（{type(e).__name__}）。"
                "classical 特徴へフォールバックします。"
            )
            self.mode = "classical"

    @torch.inference_mode()
    def _encode_resnet(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        feat = self.model(batch)  # (N, 512)
        return feat.cpu().numpy().astype(np.float32)

    def _encode_classical(self, images: list[Image.Image]) -> np.ndarray:
        """cv2 のみで作る決定的な顔特徴（フォールバック）。

        HSV のヒストグラム（肌色/髪色/目色の差を拾う）と、4x4 グリッドのグレー平均
        （顔パーツの空間配置を粗く拾う）を連結する。L2 正規化前提の素朴な記述子。
        """
        feats = []
        for im in images:
            bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            grid = cv2.resize(gray, (4, 4)).flatten().astype(np.float32) / 255.0
            feats.append(np.concatenate([hist, grid]).astype(np.float32))
        return np.stack(feats)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """画像リスト → 埋め込み行列 (N, D)。L2 正規化は呼び出し側で行う。"""
        if self.mode == "resnet18":
            return self._encode_resnet(images)
        return self._encode_classical(images)


# ===========================================================================
# ベクトル演算（L2 正規化・コサイン）
# ===========================================================================
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """各行を L2 ノルム 1 に正規化。正規化後は「内積 = コサイン類似度」になる。

    顔認識では埋め込みを必ず L2 正規化してからコサイン類似で比較するのが定石。
    ベクトルの長さ（明るさ等で変動）に振り回されず、向き（顔の特徴）だけを見るため。
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a(行ごと) と b(行ごと) のコサイン類似度行列 (len(a), len(b))。"""
    return l2_normalize(a) @ l2_normalize(b).T


# ===========================================================================
# 1:1 照合(verification) の評価: ペア生成 / ROC / EER / TAR@FAR
# ===========================================================================
def build_pairs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全ペア (i<j) を作り、(pairs[K,2], is_genuine[K]) を返す。

    is_genuine=True は「同一人物ペア(genuine)」、False は「別人ペア(impostor)」。
    照合は『2 枚が同一人物か?』の二値判定なので、この genuine/impostor の山を作って評価する。
    """
    n = len(labels)
    iu, ju = np.triu_indices(n, k=1)  # 上三角（自分自身と重複を除く）
    pairs = np.stack([iu, ju], axis=1)
    is_genuine = labels[iu] == labels[ju]
    return pairs, is_genuine


def verification_scores(emb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全ペアのコサイン類似度スコアと genuine ラベルを返す（emb は未正規化でよい）。"""
    nrm = l2_normalize(emb)
    pairs, is_genuine = build_pairs(labels)
    scores = np.sum(nrm[pairs[:, 0]] * nrm[pairs[:, 1]], axis=1)  # 各ペアのコサイン類似
    return scores.astype(np.float32), is_genuine


def roc_far_tar(
    scores: np.ndarray, is_genuine: np.ndarray, n_thr: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """しきい値を掃引して (thresholds, FAR, TAR) を返す。score>=thr を「同一人物」と判定。

    TAR(True Accept Rate)  = genuine のうち accept した割合（高いほど良い）。
    FAR(False Accept Rate) = impostor のうち誤って accept した割合（低いほど良い）。
    顔認証は「FAR を業務要件以下に固定し、その下で TAR を最大化」する世界。
    """
    gen = scores[is_genuine]
    imp = scores[~is_genuine]
    lo, hi = float(scores.min()), float(scores.max())
    thrs = np.linspace(hi + 1e-6, lo - 1e-6, n_thr)  # 高→低（FAR 昇順になる）
    tar = np.array([(gen >= t).mean() for t in thrs])
    far = np.array([(imp >= t).mean() for t in thrs])
    return thrs, far, tar


def compute_eer(far: np.ndarray, tar: np.ndarray, thrs: np.ndarray) -> tuple[float, float]:
    """EER(Equal Error Rate) と、その近傍のしきい値を返す。

    EER は FAR == FRR(=1-TAR) となる動作点の誤り率。閾値非依存で『2 つの誤りが釣り合う点』
    として埋め込み品質をひと言で表す指標。低いほど良い。
    """
    frr = 1.0 - tar
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(thrs[idx])


def tar_at_far(
    far: np.ndarray, tar: np.ndarray, thrs: np.ndarray, target_far: float
) -> tuple[float, float]:
    """FAR <= target_far を満たす中で最大の TAR と、その閾値を返す。

    実務での読み方:「FAR=1% のとき本人をどれだけ通せるか(TAR@FAR=1e-2)」。
    target_far を満たす点が無ければ TAR=0 を返す（=その厳しさでは誰も通せない）。
    """
    ok = far <= target_far
    if not ok.any():
        return 0.0, float(thrs[0])
    j = int(np.argmax(tar * ok))  # ok の範囲で TAR 最大
    return float(tar[j]), float(thrs[j])


# ===========================================================================
# 1:N 識別(identification) の評価: rank-1 / CMC
# ===========================================================================
def identification_cmc(
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    probe: np.ndarray,
    probe_labels: np.ndarray,
    max_rank: int = 5,
) -> np.ndarray:
    """CMC 曲線（rank 1..max_rank の累積識別率）を返す。CMC[0] が rank-1 精度。

    1:N 識別 = 「このプローブは登録済み N 人の誰か?」。各プローブをギャラリー全員と照合し、
    類似度上位に正解人物が rank-k 以内で出れば成功、を k ごとに累積する。
    """
    sim = cosine_sim_matrix(probe, gallery)  # (Np, Ng) 大きいほど近い
    order = np.argsort(-sim, axis=1)  # 各プローブで類似度降順のギャラリー添字
    ranked_labels = gallery_labels[order]  # (Np, Ng)
    correct = ranked_labels == probe_labels[:, None]
    # rank-k 以内に正解が「初めて」出たか（=上位 k のどれかが正解）
    cmc = np.array([correct[:, : k + 1].any(axis=1).mean() for k in range(max_rank)])
    return cmc.astype(np.float32)


# ===========================================================================
# クラスタリング評価: purity / NMI / homogeneity / completeness
# ===========================================================================
def purity_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """purity = 各クラスタを『多数派の真ラベル』に割り当てた時の正解率。

    ノイズ点(-1, DBSCAN の外れ値)も 1 つのクラスタ扱いで素直に数える。
    直感的だが、クラスタを増やすほど上がる（極端には 1 点 1 クラスタで purity=1）ので、
    NMI など他指標と併読する。
    """
    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)
    total = len(labels_true)
    correct = 0
    for c in np.unique(labels_pred):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        # このクラスタ内で最も多い真ラベルの個数を正解とみなす
        correct += np.bincount(labels_true[mask]).max()
    return float(correct / total) if total else 0.0


def clustering_report(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    """purity / NMI / homogeneity / completeness / 推定人数 をまとめて返す。"""
    from sklearn.metrics import (
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )

    n_clusters = len(set(labels_pred) - {-1})
    n_noise = int(np.sum(np.asarray(labels_pred) == -1))
    return {
        "purity": purity_score(labels_true, labels_pred),
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
        "completeness": float(completeness_score(labels_true, labels_pred)),
        "n_clusters_found": n_clusters,
        "n_noise": n_noise,
    }


# ===========================================================================
# 自己テスト（このファイル単体で実行すると、生成→埋め込み→各評価が一通り動く）
# ===========================================================================
def _selftest() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("face_lab セルフテスト")
    images, labels, names = make_face_dataset(n_ids=4, n_per_id=4, seed=0)
    print(f"  生成: {len(images)} 枚 / {len(names)} 人")
    enc = FaceEncoder()
    emb = enc.encode(images)
    print(f"  埋め込み: mode={enc.mode}, shape={emb.shape}")

    scores, gen = verification_scores(emb, labels)
    thrs, far, tar = roc_far_tar(scores, gen)
    eer, _ = compute_eer(far, tar, thrs)
    t1, _ = tar_at_far(far, tar, thrs, 0.01)
    print(f"  照合: EER={eer:.3f}  TAR@FAR=1e-2={t1:.3f}")

    cmc = identification_cmc(emb, labels, emb, labels)
    print(f"  識別: rank-1={cmc[0]:.3f}")

    from sklearn.cluster import DBSCAN

    pred = DBSCAN(eps=0.05, min_samples=2, metric="cosine").fit_predict(l2_normalize(emb))
    rep = clustering_report(labels, pred)
    print(f"  クラスタリング: {rep}")
    print("OK")


if __name__ == "__main__":
    _selftest()
