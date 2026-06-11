"""gen_lab.py — 31_generation_editing の共有ヘルパ（合成データ・劣化・指標・モデルローダ）。

この回のスクリプト（01〜04・mini_project）はすべてこのモジュールを `import gen_lab as gl`
で共有します。狙いは「各スクリプトを単一責務に保ち、合成データ生成・画像劣化・PSNR/SSIM・
CLIP/拡散/超解像モデルのロード（失敗してもガードして exit 0）を 1 箇所に集約する」こと。

設計方針:
  - ネットへ出てよいのは **モデル重みの初回ダウンロードのみ**。入力画像はすべて **合成生成**
    （data/31_generation_editing/ に画像を置けばそちらを優先して読む）。
  - CPU 前提。torch は float32・`model.eval()`＋`torch.inference_mode()`、拡散は
    `enable_attention_slicing()`。重いモデルのロードはすべて try/except で包み、失敗時は
    None を返して呼び出し側がフォールバックできるようにする（＝オフラインでも必ず exit 0）。
  - PSNR / SSIM は skimage を使わず numpy / cv2 で自前実装する（学習目的）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# パス定数（このモジュールの出力は outputs/31_generation_editing/ に集約）
# ----------------------------------------------------------------------------
MODULE_ID = "31_generation_editing"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT_DIR = os.path.join(_ROOT, "outputs", MODULE_ID)
DATA_DIR = os.path.join(_ROOT, "data", MODULE_ID)


def ensure_dirs() -> None:
    """出力ディレクトリを作る（無ければ）。"""
    os.makedirs(OUT_DIR, exist_ok=True)


def set_threads() -> None:
    """CPU スレッド数を控えめ（物理コアの半分程度）に設定。過剰並列は逆に遅くなる。"""
    try:
        import torch

        n = max(1, (os.cpu_count() or 4) // 2)
        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001  torch が無い経路でも落とさない
        pass


# ----------------------------------------------------------------------------
# 画像 ⇔ 配列の変換（OpenCV は BGR、PIL/torch/matplotlib は RGB という違いに注意）
# ----------------------------------------------------------------------------
def to_pil(rgb: np.ndarray):
    """RGB uint8 配列 → PIL.Image。"""
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)))


def to_np(pil_img) -> np.ndarray:
    """PIL.Image → RGB uint8 配列。"""
    return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)


def save_rgb(rgb: np.ndarray, name: str) -> str:
    """RGB 配列を outputs に PNG 保存（cv2 は BGR なので変換してから書く）。パスを返す。"""
    ensure_dirs()
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return path


def hstack(images: list[np.ndarray], pad: int = 6, bg: int = 32) -> np.ndarray:
    """同じ高さの RGB 画像を横に並べて 1 枚にする（before/after の比較表示に使う）。"""
    h = max(im.shape[0] for im in images)
    out = []
    for im in images:
        if im.shape[0] != h:  # 高さを揃える
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        out.append(im)
        out.append(np.full((h, pad, 3), bg, np.uint8))
    return np.hstack(out[:-1])


# ----------------------------------------------------------------------------
# 合成データ生成（「実画像」の代わり。決定的・seed 固定で再現可能）
# ----------------------------------------------------------------------------
def make_scene(size: int = 256, seed: int = 0) -> np.ndarray:
    """簡単な風景（空のグラデ＋太陽＋丘＋家）を描いた RGB uint8 画像を返す。

    超解像・インペイントの「劣化前のきれいな参照画像（ground truth）」として使う。
    細部（窓・煙）を入れて、ダウンスケール→復元で差が出るようにしている。
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), np.uint8)
    # 空: 上(水色)→下(白)への縦グラデーション
    for y in range(size):
        t = y / size
        img[y, :] = (int(135 + 80 * t), int(180 + 50 * t), int(235))
    # 太陽
    cx, cy = int(size * 0.75), int(size * 0.25)
    cv2.circle(img, (cx, cy), size // 10, (90, 220, 255), -1)
    # 丘（緑の半楕円を 2 つ）
    cv2.ellipse(img, (size // 3, size), (size // 2, size // 4), 0, 180, 360, (70, 160, 90), -1)
    cv2.ellipse(
        img, (size, int(size * 0.9)), (size // 2, size // 5), 0, 180, 360, (60, 140, 80), -1
    )
    # 家（壁・屋根・ドア・窓）
    hx, hy, hw, hh = int(size * 0.15), int(size * 0.55), int(size * 0.28), int(size * 0.28)
    cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), (200, 220, 235), -1)
    roof = np.array([[hx - 6, hy], [hx + hw + 6, hy], [hx + hw // 2, hy - hh // 2]])
    cv2.fillPoly(img, [roof], (60, 90, 200))
    cv2.rectangle(
        img, (hx + hw // 2 - 8, hy + hh - 30), (hx + hw // 2 + 8, hy + hh), (40, 60, 120), -1
    )
    cv2.rectangle(img, (hx + 8, hy + 10), (hx + 26, hy + 28), (250, 240, 150), -1)
    # 軽いノイズで質感を足す（毎回同じになるよう seed 固定の rng を使う）
    noise = rng.randint(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def make_texture(size: int = 256, seed: int = 1) -> np.ndarray:
    """規則的な市松＋円のテクスチャ。背景除去や img2img の入力バリエーション用。"""
    rng = np.random.RandomState(seed)
    img = np.full((size, size, 3), 230, np.uint8)
    step = size // 8
    for i in range(0, size, step):
        for j in range(0, size, step):
            if ((i // step) + (j // step)) % 2 == 0:
                img[i : i + step, j : j + step] = (180, 200, 215)
    for _ in range(6):
        c = (rng.randint(0, size), rng.randint(0, size))
        col = tuple(int(v) for v in rng.randint(60, 220, 3))
        cv2.circle(img, c, rng.randint(size // 12, size // 6), col, -1)
    return img


def load_or_make_scene(size: int = 256, seed: int = 0) -> tuple[np.ndarray, str]:
    """data/ に画像があれば最初の 1 枚を読み（リサイズして返す）、無ければ合成シーンを返す。

    返り値は (RGB配列, 由来ラベル)。由来ラベルは "real:<name>" か "synthetic"。
    """
    if os.path.isdir(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                bgr = cv2.imread(os.path.join(DATA_DIR, f))
                if bgr is not None:
                    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
                    return rgb, f"real:{f}"
    return make_scene(size, seed), "synthetic"


# ----------------------------------------------------------------------------
# 画像劣化（参照あり評価のための「壊し方」）
# ----------------------------------------------------------------------------
def downscale(hr: np.ndarray, scale: int = 2) -> np.ndarray:
    """高解像→低解像（面積平均 INTER_AREA。エイリアスを抑え超解像の入力に向く）。"""
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)


def upscale_bicubic(lr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """低解像→指定サイズへバイキュービック拡大（超解像の古典ベースライン）。size=(w,h)。"""
    return cv2.resize(lr, size, interpolation=cv2.INTER_CUBIC)


def upscale_lanczos(lr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos 拡大（バイキュービックよりわずかに鮮鋭。古典ベースライン比較用）。size=(w,h)。"""
    return cv2.resize(lr, size, interpolation=cv2.INTER_LANCZOS4)


def add_gaussian_noise(img: np.ndarray, sigma: float = 12.0, seed: int = 0) -> np.ndarray:
    """ガウシアンノイズを加える（復元タスクの劣化要因）。"""
    rng = np.random.RandomState(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_scratches(img: np.ndarray, n: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """白い傷（線・斑点）を描き、(傷あり画像, マスク) を返す。インペイントの題材。

    マスクは uint8（傷=255, それ以外=0）。cv2.inpaint や拡散インペイントにそのまま渡せる。
    """
    rng = np.random.RandomState(seed)
    dmg = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    h, w = img.shape[:2]
    for _ in range(n):
        p1 = (rng.randint(0, w), rng.randint(0, h))
        p2 = (p1[0] + rng.randint(-w // 3, w // 3), p1[1] + rng.randint(-h // 6, h // 6))
        th = rng.randint(2, 5)
        cv2.line(dmg, p1, p2, (255, 255, 255), th)
        cv2.line(mask, p1, p2, 255, th)
    for _ in range(n):
        c = (rng.randint(0, w), rng.randint(0, h))
        r = rng.randint(2, 6)
        cv2.circle(dmg, c, r, (255, 255, 255), -1)
        cv2.circle(mask, c, r, 255, -1)
    return dmg, mask


def make_box_mask(size: int, rect: tuple[int, int, int, int]) -> np.ndarray:
    """矩形 (x, y, w, h) を 255 で塗ったマスク（インペイントの「消したい領域」）。"""
    m = np.zeros((size, size), np.uint8)
    x, y, w, h = rect
    m[y : y + h, x : x + w] = 255
    return m


# ----------------------------------------------------------------------------
# 参照あり指標（PSNR / SSIM）— skimage を使わず自前実装
# ----------------------------------------------------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    """平均二乗誤差。a, b は同形状（uint8 でも float でも可）。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """PSNR = 10*log10(MAX^2 / MSE)。完全一致なら inf。値は大きいほど良い。"""
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_val**2) / m))


def _ssim_one_channel(x: np.ndarray, y: np.ndarray, max_val: float) -> float:
    """1 チャネルの SSIM（Wang ら 2004 のガウシアン窓 11x11, sigma=1.5）。

    平均・分散・共分散をガウシアン重みで局所推定し、輝度・コントラスト・構造の積を
    画素ごとに計算→平均する。cv2.GaussianBlur を局所重み付き平均として流用する。
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    win, sigma = (11, 11), 1.5
    mu_x = cv2.GaussianBlur(x, win, sigma)
    mu_y = cv2.GaussianBlur(y, win, sigma)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = cv2.GaussianBlur(x * x, win, sigma) - mu_x2
    sigma_y = cv2.GaussianBlur(y * y, win, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, win, sigma) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(num / den))


def ssim(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """SSIM（カラーはチャネル平均）。1 に近いほど良い。範囲は概ね [-1, 1]。

    注: 境界処理（cv2 は反射、skimage/torchmetrics はクロップ）の違いで他実装と小数第 3 位
    程度ずれることがある。指標の「向き（大きいほど良い）」と相対比較が目的なので問題ない。
    """
    if a.ndim == 2:
        return _ssim_one_channel(a, b, max_val)
    return float(
        np.mean([_ssim_one_channel(a[..., c], b[..., c], max_val) for c in range(a.shape[2])])
    )


# ----------------------------------------------------------------------------
# モデルローダ（すべてガード付き。失敗時は None を返してフォールバックさせる）
# ----------------------------------------------------------------------------
def load_t2i_pipeline(model_id: str = "stabilityai/sd-turbo"):
    """SD-Turbo の text-to-image パイプラインを返す（失敗時 None）。

    SD-Turbo は蒸留済みで **num_inference_steps=1〜2 / guidance_scale=0.0** が前提。
    CPU 向けに float32・attention slicing・進捗バー無効化を設定する。
    """
    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe.to("cpu")
        pipe.enable_attention_slicing()  # メモリ削減（CPU/小VRAM で有効）
        _quiet_pipe(pipe)
        return pipe
    except Exception as e:  # noqa: BLE001
        print(
            f"  [warn] SD-Turbo(t2i) をロードできません（{type(e).__name__}）。合成画像で代替します。"
        )
        return None


def load_img2img_pipeline(model_id: str = "stabilityai/sd-turbo"):
    """SD-Turbo の image-to-image パイプラインを返す（失敗時 None）。"""
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image

        pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        _quiet_pipe(pipe)
        return pipe
    except Exception as e:  # noqa: BLE001
        print(
            f"  [warn] SD-Turbo(img2img) をロードできません（{type(e).__name__}）。スキップします。"
        )
        return None


def load_swin2sr(model_id: str = "caidas/swin2SR-classical-sr-x2-64"):
    """Swin2SR（×2 超解像）の (processor, model) を返す（失敗時 None）。"""
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

        proc = AutoImageProcessor.from_pretrained(model_id)
        model = Swin2SRForImageSuperResolution.from_pretrained(model_id).eval()
        return proc, model
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] Swin2SR をロードできません（{type(e).__name__}）。古典補間で代替します。")
        return None


def swin2sr_upscale(proc, model, lr_rgb: np.ndarray) -> np.ndarray:
    """Swin2SR で ×2 超解像し、ちょうど 2 倍サイズに切り詰めた RGB uint8 を返す。

    注意: Swin2SR はウィンドウサイズの倍数に内部パディングするため、出力は厳密に 2×H,2×W
    とは限らない。必要サイズにクロップして揃える。
    """
    import torch

    inp = proc(to_pil(lr_rgb), return_tensors="pt")
    with torch.inference_mode():
        out = model(**inp)
    sr = out.reconstruction.squeeze().clamp(0, 1).permute(1, 2, 0).numpy()
    sr = (sr * 255.0).round().astype(np.uint8)
    th, tw = lr_rgb.shape[0] * 2, lr_rgb.shape[1] * 2
    return sr[:th, :tw]


def load_clip_scorer(model_id: str = "openai/clip-vit-base-patch32"):
    """CLIPScore 計算用のクロージャ score(rgb, text)->float を返す（失敗時 None）。

    CLIPScore = w * max(cos(画像埋め込み, テキスト埋め込み), 0)（慣例 w=2.5）。
    プロンプトと生成画像の「意味的な整合度」を測る。値が大きいほど整合が良い。
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_id).eval()
        proc = CLIPProcessor.from_pretrained(model_id)

        def score(rgb: np.ndarray, text: str, w: float = 2.5) -> float:
            inputs = proc(text=[text], images=to_pil(rgb), return_tensors="pt", padding=True)
            with torch.inference_mode():
                out = model(**inputs)
            # transformers v5 では forward 出力の image_embeds / text_embeds が
            # 射影後の埋め込み。コサイン類似度のため明示的に L2 正規化する（再正規化は無害）。
            img_emb = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_emb = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
            cos = float((img_emb * txt_emb).sum(dim=-1).item())
            return w * max(cos, 0.0)

        return score
    except Exception as e:  # noqa: BLE001
        print(
            f"  [warn] CLIP をロードできません（{type(e).__name__}）。CLIPScore をスキップします。"
        )
        return None


def _quiet_pipe(pipe) -> None:
    """diffusers パイプラインの進捗バーを黙らせる（ログを読みやすく）。"""
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:  # noqa: BLE001
        pass


# import 時に CPU スレッドを整える
set_threads()
