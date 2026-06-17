"""実践ユースケース: ワンクリック背景除去ツール（GrabCut で前景を切り出し、背景を透明化／差し替え）。

この章で学んだ `cv2.grabCut(GC_INIT_WITH_RECT)` を、評価ベンチではなく
**現実の小ツール**（いわゆる "remove.bg" 風の背景除去）に落とし込んだ動く出発点です。
EC の商品写真やプロフィール画像から被写体だけを切り抜き、

    1) 入力画像を読み込み（実写優先・無ければ合成シーン）、
    2) 被写体を囲む矩形を決め（既定は自動の内側矩形 / 任意で Tkinter ドラッグ選択）、
    3) GrabCut で前景マスクを推定、最大連結成分＋穴埋めで掃除し、境界を羽化（feather）、
    4) 背景を「透明（RGBA）」「白スタジオ」「新しい背景画像」「ぼかし（ポートレート風）」に
       差し替えて書き出す。

mini_project.py との違い:
    mini_project.py は GrabCut→Watershed→inpaint を IoU/Dice・PSNR/SSIM で測る
    「評価ベンチマーク（数値レポート）」でした。こちらは指標を出さず、マスクを“編集に使う”
    背景除去ツールです。透過 PNG の書き出し・アルファ羽化・背景合成・最大連結成分での
    後処理など、実務の切り抜きで効くテクニックに寄せてあります。

実データ優先・合成フォールバック:
    - 入力画像:   data/08_classical_segmentation/ に .png/.jpg を置くと、その最初の1枚を使います
                  （無ければ「スタジオ背景に置いた赤いマグカップ」の合成シーンにフォールバック）。
    - 差し替え背景: data/08_classical_segmentation/background/ に .png/.jpg を置くと、それを
                  「新しい背景」に使います（無ければ合成のグラデ背景を自前生成）。
    被写体が中央にある写真ほど自動矩形でうまく切れます。中央にない・複数被写体のときは
    USE_CASE_INTERACTIVE=1 を付けて起動し、Tkinter のドラッグで矩形を指定してください。

頑健性（必ず exit 0）:
    画面表示（cv2.imshow）は使いません。Tkinter 選択は USE_CASE_INTERACTIVE=1 のときだけ試み、
    ヘッドレス／失敗時は自動の内側矩形に必ずフォールバックします。GrabCut は内部 kmeans の
    乱数で結果が毎回わずかに変わる点に注意（厳密一致は期待しない）。ネット接続は不要です。

拡張して練習するアイデア:
    - 自動矩形ではなく、04 章の Canny + 輪郭で「一番大きな物体の外接矩形」を自動検出して
      初期化する（被写体が中央でなくても切れるようにする）。
    - GC_INIT_WITH_MASK のなすり書き（前景/背景ブラシ）を足して、取りこぼし/拾いすぎを対話修正。
    - 透過 PNG にドロップシャドウを合成して、EC 出品用のキレイな影付き切り抜きにする。
    - data/ のフォルダを総なめしてバッチ処理する CLI（出力をファイル名ごとに保存）に拡張する。
    - 深層 SAM（第20・22回）の矩形/点プロンプトに置き換え、GrabCut と切り抜き品質を比べる。

実行:
    uv run python lectures/08_classical_segmentation/use_case.py
    USE_CASE_INTERACTIVE=1 uv run python lectures/08_classical_segmentation/use_case.py   # 矩形をドラッグ選択
出力:
    lectures/08_classical_segmentation/outputs/use_case_cutout.png      … 背景透過の切り抜き（RGBA）
    lectures/08_classical_segmentation/outputs/use_case_panel.png       … 入力/アルファ/白背景/新背景の一覧
    lectures/08_classical_segmentation/outputs/use_case_bokeh.png       … 背景ぼかし（ポートレート風）
    lectures/08_classical_segmentation/outputs/use_case_bg_removal.json … 設定と前景占有率などのログ
"""

from __future__ import annotations

import json
import os
import pathlib

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# 設定（数値をいじって挙動を観察する練習用のツマミ）
# ---------------------------------------------------------------------
MODULE_ID = "08_classical_segmentation"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = pathlib.Path("data") / MODULE_ID
BG_DIR = DATA_DIR / "background"            # 差し替え用の「新しい背景」を置くフォルダ
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

MAX_SIDE = 800           # 実写が巨大なときの縮小上限（GrabCut を CPU で軽く保つ）
INSET_FRAC = 0.08        # 自動矩形が画像の縁から何割内側に入るか（縁＝確実な背景）
GRABCUT_ITERS = 5        # GrabCut の反復回数（多いほど安定寄りだが遅い）
FEATHER_SIGMA = 2.0      # アルファ境界の羽化幅[px]（大きいほど継ぎ目がなだらか）
STUDIO_WHITE = (245, 245, 245)   # 白スタジオ背景の色（BGR）


# =====================================================================
# 小道具（番号付きスクリプトは import 不可なので、必要な道具はここに自己完結で持つ）
# =====================================================================
def panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 270)) -> np.ndarray:
    """1 枚を固定サイズ＋上部の黒帯タイトル付きパネルにする（タイトルは ASCII）。"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w, h = size
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    """パネル群を cols 列のグリッドに並べる（端数は黒で埋める）。"""
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def checkerboard(h: int, w: int, sq: int = 16) -> np.ndarray:
    """透過を可視化するための市松模様（透明部分がここに透けて見える）。"""
    board = np.full((h, w, 3), 235, np.uint8)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            if ((x // sq) + (y // sq)) % 2 == 0:
                board[y : y + sq, x : x + sq] = 200
    return board


# =====================================================================
# 1) 入力画像を用意する（実写優先 / 無ければ合成）
# =====================================================================
def _list_images(folder: pathlib.Path) -> list[pathlib.Path]:
    if not folder.is_dir():
        return []
    # background/ サブフォルダは入力候補から除外する
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def make_synthetic_scene() -> np.ndarray:
    """合成シーン: スタジオ背景の中央に「赤いマグカップ」を置いた商品写真風 BGR 画像。

    被写体が中央にあり背景と色が分離しているので、自動の内側矩形でも GrabCut が
    きれいに切り出せる（＝ツールの動作確認に向く）。実写を data/ に置けば不要。
    """
    h, w = 360, 480
    rng = np.random.default_rng(7)
    # 背景: 斜めグラデーションの淡いグレー（撮影ブースのホリゾント）
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.zeros((h, w, 3), np.float32)
    base = 180 + (xx / w) * 40 + (yy / h) * 20
    bg[..., 0] = base + 8      # B やや高め（寒色のグレー）
    bg[..., 1] = base
    bg[..., 2] = base - 6
    bg = np.clip(bg + rng.normal(0, 4, bg.shape), 0, 255).astype(np.uint8)

    img = bg.copy()
    # 被写体: 丸みのある赤いマグカップ（胴＋持ち手）。内部に軽いテクスチャ。
    body = np.zeros((h, w), np.uint8)
    cv2.ellipse(body, (w // 2 - 20, h // 2 + 10), (95, 110), 0, 0, 360, 255, -1)  # 胴
    cv2.ellipse(body, (w // 2 + 78, h // 2 + 10), (40, 58), 0, 0, 360, 255, 18)   # 持ち手(リング)
    cv2.ellipse(body, (w // 2 - 20, h // 2 - 90), (95, 26), 0, 0, 360, 255, -1)   # 飲み口の楕円
    mug_color = np.array([55, 45, 210], np.float32)  # BGR の赤
    tex = np.clip(mug_color + rng.normal(0, 14, (h, w, 3)), 0, 255).astype(np.uint8)
    img[body == 255] = tex[body == 255]
    # 飲み口の内側を暗くして立体感（飲み口の上部だけ）
    inner = np.zeros((h, w), np.uint8)
    cv2.ellipse(inner, (w // 2 - 20, h // 2 - 90), (78, 18), 0, 0, 360, 255, -1)
    img[inner == 255] = (40, 30, 120)
    return img


def load_input_image() -> tuple[np.ndarray, str]:
    """入力 BGR 画像と出所ラベルを返す。実写優先、無ければ合成シーン。"""
    files = _list_images(DATA_DIR)
    if files:
        # 非 ASCII パスにも強い PIL で開き、BGR へ変換（cv2 は BGR 前提）
        pil = Image.open(files[0]).convert("RGB")
        rgb = np.asarray(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = _shrink_to_max_side(bgr)
        return bgr, f"real:{files[0].name}"
    return make_synthetic_scene(), "synthetic:red_mug"


def _shrink_to_max_side(bgr: np.ndarray) -> np.ndarray:
    """長辺が MAX_SIDE を超えたら縮小（GrabCut を CPU で軽く保つ）。"""
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


# =====================================================================
# 2) 初期矩形を決める（自動の内側矩形 / 任意で Tkinter 選択）
# =====================================================================
def auto_inset_rect(shape: tuple[int, int], frac: float = INSET_FRAC) -> tuple[int, int, int, int]:
    """画像の縁から frac だけ内側を被写体候補とする矩形 (x,y,w,h)。

    縁の帯は GrabCut にとって「確実に背景」になる。中央被写体ならこれだけで十分切れる。
    """
    h, w = shape
    mx, my = int(w * frac), int(h * frac)
    return (mx, my, w - 2 * mx, h - 2 * my)


def clamp_rect(rect: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    """矩形を画像内に収め、幅・高さが正になるよう補正する。"""
    x, y, rw, rh = rect
    x = max(0, min(int(x), w - 2))
    y = max(0, min(int(y), h - 2))
    rw = max(1, min(int(rw), w - x))
    rh = max(1, min(int(rh), h - y))
    return (x, y, rw, rh)


def select_rect_interactive(bgr: np.ndarray,
                            default_rect: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], str]:
    """Tkinter のドラッグで被写体を囲む矩形を選ぶ。ヘッドレス/失敗時は default_rect。

    cv2.imshow は headless で固まるため使わない。USE_CASE_INTERACTIVE=1 のときだけ
    Tkinter を試し、ウィンドウを出せない CI/SSH では即座に自動矩形へフォールバックする。
    """
    if os.environ.get("USE_CASE_INTERACTIVE", "0") != "1":
        return default_rect, "auto_inset"
    try:
        import tkinter as tk

        from PIL import ImageTk

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        root = tk.Tk()
        root.title("Drag a box around the subject, then close the window")
        tkimg = ImageTk.PhotoImage(pil)
        canvas = tk.Canvas(root, width=pil.width, height=pil.height)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=tkimg)
        st = {"x0": 0, "y0": 0, "rect_id": None, "sel": None}

        def on_press(e):
            st["x0"], st["y0"] = e.x, e.y

        def on_drag(e):
            if st["rect_id"] is not None:
                canvas.delete(st["rect_id"])
            st["rect_id"] = canvas.create_rectangle(
                st["x0"], st["y0"], e.x, e.y, outline="#00ff00", width=2)

        def on_release(e):
            x0, y0, x1, y1 = st["x0"], st["y0"], e.x, e.y
            st["sel"] = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.mainloop()

        sel = st["sel"]
        if sel is not None and sel[2] > 10 and sel[3] > 10:
            return sel, "tkinter_roi"
        return default_rect, "auto_inset(no_selection)"
    except Exception as e:  # noqa: BLE001  headless/Tk無し等でも exit 0 を守る
        return default_rect, f"auto_inset(headless:{type(e).__name__})"


# =====================================================================
# 3) GrabCut → マスク掃除 → アルファ羽化
# =====================================================================
def run_grabcut(bgr: np.ndarray, rect: tuple[int, int, int, int], iters: int = GRABCUT_ITERS) -> np.ndarray:
    """矩形初期化の GrabCut を実行し、前景マスク(0/255)を返す。"""
    mask = np.zeros(bgr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)   # 背景 GMM の作業領域（形・dtype 固定。中身は触らない）
    fgd = np.zeros((1, 65), np.float64)   # 前景 GMM の作業領域
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    # FGD(=1) と PR_FGD(=3) を前景とする
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def keep_largest(mask: np.ndarray) -> np.ndarray:
    """最大連結成分だけ残す（背景に散った小さな FP のかけらを除去）。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask  # 前景が無い
    areas = stats[1:, cv2.CC_STAT_AREA]      # 0 は背景なので除く
    idx = 1 + int(np.argmax(areas))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """被写体内部の穴（取りこぼした暗部など）を埋める。

    角(0,0)から背景を flood fill し、到達できなかった画素＝内部の穴として前景に足す。
    GrabCut＋内側矩形なら角は背景なので安全。
    """
    h, w = mask.shape
    if mask[0, 0] != 0:
        return mask  # 角が前景なら前提が崩れるので何もしない
    ff = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 255)     # 背景を 255 で塗る
    holes = cv2.bitwise_not(ff)                    # 塗り残し＝内部の穴
    return cv2.bitwise_or(mask, holes)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """GrabCut 生マスクを掃除: 軽いオープン→最大連結成分→穴埋め→軽いクローズ。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)   # 孤立ノイズを落とす
    m = keep_largest(m)                             # 主被写体だけ残す
    m = fill_holes(m)                               # 内部の穴を埋める
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)     # 縁の小欠けを埋める
    return m


def feather_alpha(mask: np.ndarray, sigma: float = FEATHER_SIGMA) -> np.ndarray:
    """0/255 マスクを [0,1] のソフトなアルファに（境界をぼかして合成の継ぎ目を消す）。"""
    a = (mask > 0).astype(np.float32)
    a = cv2.GaussianBlur(a, ksize=(0, 0), sigmaX=sigma)
    return np.clip(a, 0.0, 1.0)


# =====================================================================
# 4) 背景差し替え（透明 / 白 / 新背景画像 / ぼかし）
# =====================================================================
def make_studio_backdrop(h: int, w: int) -> np.ndarray:
    """差し替え用の合成背景（青→白の縦グラデ）。data/ に背景画像が無いときの既定。"""
    top = np.array([150, 110, 70], np.float32)     # 上: 落ち着いた青(BGR)
    bottom = np.array([245, 240, 235], np.float32)  # 下: ほぼ白
    ys = np.linspace(0.0, 1.0, h, np.float32)[:, None, None]
    grad = (1.0 - ys) * top + ys * bottom
    grad = np.repeat(grad, w, axis=1)
    return np.clip(grad, 0, 255).astype(np.uint8)


def load_background(h: int, w: int) -> tuple[np.ndarray, str]:
    """data/<id>/background/ の画像を入力サイズに合わせて返す。無ければ合成背景。"""
    files = _list_images(BG_DIR)
    if files:
        pil = Image.open(files[0]).convert("RGB").resize((w, h))
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return bgr, f"real:{files[0].name}"
    return make_studio_backdrop(h, w), "synthetic:studio_backdrop"


def composite(fg_bgr: np.ndarray, bg_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """alpha で前景 fg を背景 bg に重ねる（alpha=1→前景, 0→背景, 中間→なめらかに混合）。"""
    a = alpha[..., None]  # (H,W,1) にして 3ch へブロードキャスト
    out = a * fg_bgr.astype(np.float32) + (1.0 - a) * bg_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_cutout_rgba(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """背景を透明にした切り抜き RGBA 画像 (H,W,4) を作る（α=前景アルファ×255）。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    a = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return np.dstack([rgb, a])


# =====================================================================
# メイン
# =====================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 入力（実写優先 / 合成）
    bgr, src = load_input_image()
    h, w = bgr.shape[:2]

    # 2) 初期矩形（自動の内側矩形が既定。USE_CASE_INTERACTIVE=1 で Tkinter 選択）
    default_rect = auto_inset_rect((h, w))
    rect, rect_method = select_rect_interactive(bgr, default_rect)
    rect = clamp_rect(rect, w, h)

    # 3) GrabCut → 掃除 → 羽化
    raw = run_grabcut(bgr, rect)
    mask = refine_mask(raw)
    alpha = feather_alpha(mask)
    fg_ratio = float((mask > 0).mean())  # 前景が画面に占める割合

    # 4) 背景差し替え（透明 / 白スタジオ / 新背景 / ぼかし）
    white_bg = np.full_like(bgr, STUDIO_WHITE, dtype=np.uint8)
    new_bg, bg_name = load_background(h, w)
    # ポートレート風ぼかし: 背景だけ強くぼかし、前景はシャープに残す
    blurred = cv2.GaussianBlur(bgr, ksize=(0, 0), sigmaX=max(3.0, min(h, w) * 0.02))

    comp_white = composite(bgr, white_bg, alpha)
    comp_newbg = composite(bgr, new_bg, alpha)
    comp_bokeh = composite(bgr, blurred, alpha)
    cutout_rgba = make_cutout_rgba(bgr, alpha)

    # --- 出力（成果物） ---
    cutout_path = OUT_DIR / "use_case_cutout.png"
    Image.fromarray(cutout_rgba, mode="RGBA").save(cutout_path)  # 背景透過 PNG
    bokeh_path = OUT_DIR / "use_case_bokeh.png"
    cv2.imwrite(str(bokeh_path), comp_bokeh)

    # --- 一覧パネル（入力＋矩形 / アルファ / 切り抜き(市松) / 新背景合成） ---
    viz_rect = bgr.copy()
    rx, ry, rw, rh = rect
    cv2.rectangle(viz_rect, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    alpha_gray = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    cutout_on_checker = composite(bgr, checkerboard(h, w), alpha)  # 透過を市松で可視化
    panels = [
        panel(viz_rect, f"input + rect ({rect_method})"),
        panel(alpha_gray, f"alpha matte (fg={fg_ratio:.1%})"),
        panel(cutout_on_checker, "cutout (transparent bg)"),
        panel(comp_newbg, "replaced background"),
    ]
    panel_path = OUT_DIR / "use_case_panel.png"
    cv2.imwrite(str(panel_path), grid(panels, 2))

    # 白背景合成も EC 出品用に保存
    white_path = OUT_DIR / "use_case_white_bg.png"
    cv2.imwrite(str(white_path), comp_white)

    # --- ログ JSON ---
    summary = {
        "input": src,
        "image_size": [w, h],
        "rect_method": rect_method,
        "rect_xywh": list(rect),
        "grabcut_iters": GRABCUT_ITERS,
        "foreground_ratio": round(fg_ratio, 4),
        "replacement_background": bg_name,
        "outputs": [
            cutout_path.name, panel_path.name, white_path.name, bokeh_path.name,
        ],
    }
    json_path = OUT_DIR / "use_case_bg_removal.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- コンソール ---
    print("=== ワンクリック背景除去ツール（GrabCut） ===")
    print(f"input={src}  size={w}x{h}  rect={rect} ({rect_method})")
    print(f"前景占有率 = {fg_ratio:.1%}  差し替え背景 = {bg_name}")
    if fg_ratio < 0.01:
        print("注意: 前景がほとんど検出されませんでした。被写体が中央に無い可能性があります。")
        print("      USE_CASE_INTERACTIVE=1 で矩形を手動指定するか、data/ に中央被写体の写真を置いてください。")
    elif fg_ratio > 0.95:
        print("注意: ほぼ全面が前景になりました。背景と被写体の色が近いか、矩形が大きすぎる可能性があります。")
    print(f"saved: {cutout_path}")
    print(f"       {panel_path}")
    print(f"       {white_path}")
    print(f"       {bokeh_path}")
    print(f"       {json_path}")
    print("\nヒント: data/08_classical_segmentation/ に商品写真、background/ に新しい背景画像を置くと"
          "そのまま実用的に動きます。")


if __name__ == "__main__":
    main()
