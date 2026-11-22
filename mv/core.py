"""Rendering primitives for the Claude ver. Unknown Mother-Goose MV."""
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

ORANGE = np.array([0.851, 0.467, 0.341], np.float32)      # #D97757 claude terracotta
ORANGE_HI = np.array([0.93, 0.56, 0.30], np.float32)
INK = np.array([0.16, 0.16, 0.17], np.float32)
PAPER = np.array([0.935, 0.935, 0.94], np.float32)

FONTS = {
    "serif": r"C:\Users\Aentro\AppData\Local\Microsoft\Windows\Fonts\NotoSerifSC-Regular.otf",
    "sans": r"C:\Windows\Fonts\Deng.ttf",
    "thin": r"C:\Windows\Fonts\Dengl.ttf",
    "bold": r"C:\Windows\Fonts\Dengb.ttf",
    "jp": r"C:\Windows\Fonts\YuGothL.ttc",
    "jpm": r"C:\Windows\Fonts\YuGothM.ttc",
    "jpb": r"C:\Windows\Fonts\YuGothB.ttc",
    "mono": r"C:\Windows\Fonts\CascadiaMono.ttf",
    "min": str(Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSerifCJKjp-Medium.otf"),
    "minr": str(Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSerifCJKjp-Regular.otf"),
    "minb": str(Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSerifCJKjp-SemiBold.otf"),
}

# ---------------------------------------------------------------- timing (measured from the album track)
import json as _json

BEAT = 60.0 / 222.0
BAR = BEAT * 4
_B = _json.load(open(ROOT / "audio" / "beats.json"))
BEATS = np.array(_B["beats"])
_down = np.array(_B["downbeat"])
STRONG = BEATS[_down | np.roll(_down, 2)]
DOWN = BEATS[_down]
KICKS = np.array([k for k, s in _B["kicks"] if s > 0.5])
SNARES = np.array([k for k, s in _B["snares"] if s > 0.5])


def beat_pos(t):
    """Continuous beat index at time t (tracked beats, extrapolated at the ends)."""
    if t <= BEATS[0]:
        return (t - BEATS[0]) / BEAT
    if t >= BEATS[-1]:
        return len(BEATS) - 1 + (t - BEATS[-1]) / BEAT
    i = int(np.searchsorted(BEATS, t)) - 1
    return i + (t - BEATS[i]) / (BEATS[i + 1] - BEATS[i])


def since_beat(t, div=1):
    b = beat_pos(t) * div
    return (b - math.floor(b)) * BEAT / div


def pulse(t, decay=0.12, div=1):
    return math.exp(-since_beat(t, div) / decay)


def beat_index(t, div=1):
    return int(math.floor(beat_pos(t) * div))


def _since(arr, t):
    i = int(np.searchsorted(arr, t + 1e-4)) - 1
    return t - arr[i] if i >= 0 else 1e9


def kick_pulse(t, decay=0.09):
    return math.exp(-_since(KICKS, t) / decay)


def snare_pulse(t, decay=0.07):
    return math.exp(-_since(SNARES, t) / decay)


def strong_pulse(t, decay=0.12):
    return math.exp(-_since(STRONG, t) / decay)


def snap(t, grid="beat", tol=0.09):
    """Snap t to the nearest tracked beat / strong beat / 8th note if within tol."""
    arr = STRONG if grid == "strong" else BEATS
    if grid == "8th":
        arr = np.sort(np.concatenate([BEATS, (BEATS[1:] + BEATS[:-1]) / 2]))
    i = int(np.argmin(np.abs(arr - t)))
    return float(arr[i]) if abs(arr[i] - t) <= tol else t


def strong_index(t):
    return int(np.searchsorted(STRONG, t + 1e-4)) - 1


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, k):
    return a + (b - a) * k


def ramp(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / (t1 - t0))


def ease_out(k):
    return 1 - (1 - k) ** 3


def ease_in(k):
    return k ** 3


def ease_io(k):
    return k * k * (3 - 2 * k)


def ease_out_back(k, s=1.6):
    k -= 1
    return k * k * ((s + 1) * k + s) + 1


def hash01(*a):
    x = math.sin(sum((i + 1) * 12.9898 * v for i, v in enumerate(a)) + 78.233) * 43758.5453
    return x - math.floor(x)


# ---------------------------------------------------------------- canvas
def new_canvas(color=PAPER):
    c = np.empty((H, W, 3), np.float32)
    c[:] = color
    return c


def to_rgba_f(img):
    """PIL / uint8 -> float32 premultiplied RGBA."""
    a = np.asarray(img).astype(np.float32) / 255.0
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones_like(a[..., :1])], 2)
    a[..., :3] *= a[..., 3:4]
    return a


def composite(dst, src, x0, y0, alpha=1.0, mode="normal"):
    """Composite premultiplied RGBA `src` with its top-left at integer (x0, y0)."""
    h, w = src.shape[:2]
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0 or alpha <= 0.001:
        return
    s = src[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    d = dst[Y0:Y1, X0:X1]
    rgb = s[..., :3] * alpha
    a = s[..., 3:4] * alpha
    if mode == "normal":
        d *= 1 - a
        d += rgb
    elif mode == "add":
        d += rgb
        np.minimum(d, 1.5, out=d)
    elif mode == "multiply":
        # src rgb is premultiplied; un-premultiply for multiply blend
        col = np.where(a > 1e-4, rgb / np.maximum(a, 1e-4), 1.0)
        d *= 1 - a + a * col
    elif mode == "screen":
        d[:] = 1 - (1 - d) * (1 - rgb)


def put(dst, src, cx, cy, scale=1.0, rot=0.0, alpha=1.0, mode="normal", anchor=(0.5, 0.5), sx=None, sy=None,
        _nocrop=False):
    """Draw RGBA src with its anchor at (cx, cy), scaled and rotated (degrees)."""
    if alpha <= 0.001:
        return
    h, w = src.shape[:2]
    sx = scale if sx is None else sx
    sy = scale if sy is None else sy
    if _nocrop:
        rot = 0.0
    if abs(rot) < 1e-3 and abs(sx - sy) < 1e-6 and abs(sx - 1) < 1e-6:
        composite(dst, src, int(round(cx - anchor[0] * w)), int(round(cy - anchor[1] * h)), alpha, mode)
        return
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    ax, ay = anchor[0] * w, anchor[1] * h
    A = np.array([[c * sx, -s * sy], [s * sx, c * sy]], np.float32)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32) - [ax, ay]
    pts = corners @ A.T + [cx, cy]
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1c <= x0c or y1c <= y0c:
        return
    if not _nocrop and abs(rot) < 1e-3 and (sx > 1.5 or sy > 1.5):
        # crop the source to the visible part before magnifying
        u0 = int(max(0, math.floor((x0c - cx) / sx + ax) - 2))
        v0 = int(max(0, math.floor((y0c - cy) / sy + ay) - 2))
        u1 = int(min(w, math.ceil((x1c - cx) / sx + ax) + 2))
        v1 = int(min(h, math.ceil((y1c - cy) / sy + ay) + 2))
        if u1 <= u0 or v1 <= v0:
            return
        sub = src[v0:v1, u0:u1]
        put(dst, sub, cx + (u0 - ax) * sx, cy + (v0 - ay) * sy, sx=sx, sy=sy, alpha=alpha, mode=mode, anchor=(0, 0),
            _nocrop=True)
        return
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = A
    M[:, 2] = np.array([cx - x0c, cy - y0c]) - A @ [ax, ay]
    interp = cv2.INTER_AREA if max(sx, sy) < 0.5 else cv2.INTER_LINEAR
    warped = cv2.warpAffine(src, M, (x1c - x0c, y1c - y0c), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    composite(dst, warped, x0c, y0c, alpha, mode)


def put_fit(dst, src, cx, cy, height=None, width=None, **kw):
    h, w = src.shape[:2]
    s = (height / h) if height else (width / w)
    put(dst, src, cx, cy, scale=s * kw.pop("scale", 1.0), **kw)


def fill(dst, color, alpha=1.0, rect=None):
    if rect is None:
        dst[:] = dst * (1 - alpha) + np.asarray(color, np.float32) * alpha
    else:
        x0, y0, x1, y1 = [int(v) for v in rect]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            d = dst[y0:y1, x0:x1]
            d[:] = d * (1 - alpha) + np.asarray(color, np.float32) * alpha


def rect_rgba(w, h, color, radius=0, border=0, border_color=None, fill_alpha=1.0):
    w, h = max(1, int(w)), max(1, int(h))
    im = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = tuple(int(c * 255) for c in color) + (int(255 * fill_alpha),)
    bc = None if not border else tuple(int(c * 255) for c in (border_color if border_color is not None else color)) + (255,)
    d.rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], radius=radius * 2, fill=col, outline=bc, width=border * 2)
    im = im.resize((w, h), Image.LANCZOS)
    return to_rgba_f(im)


@lru_cache(maxsize=256)
def rect_cached(w, h, color, radius=0, border=0, border_color=None, fill_alpha=1.0):
    return rect_rgba(w, h, color, radius, border, border_color, fill_alpha)


# ---------------------------------------------------------------- text
@lru_cache(maxsize=64)
def _font(name, size):
    return ImageFont.truetype(FONTS[name], size)


@lru_cache(maxsize=384)
def text_rgba(s, size=28, font="sans", color=(0.2, 0.2, 0.22), vertical=False, spacing=0.12, line_gap=1.6,
              stroke=0, stroke_color=(1, 1, 1)):
    """Render text to premultiplied RGBA. Supports \n. Vertical = top-to-bottom columns, right-to-left."""
    f = _font(font, size)
    col = tuple(int(c * 255) for c in color) + (255,)
    scol = tuple(int(c * 255) for c in stroke_color) + (255,)
    lines = s.split("\n")
    pad = 6 + stroke
    if not vertical:
        adv = []
        for ln in lines:
            xs, x = [], 0.0
            for ch in ln:
                xs.append(x)
                x += f.getlength(ch) + size * spacing
            adv.append((xs, x - size * spacing if ln else 0))
        wmax = max(a[1] for a in adv) if adv else 1
        lh = size * line_gap
        im = Image.new("RGBA", (int(wmax + pad * 2) + 1, int(lh * (len(lines) - 1) + size * 1.35 + pad * 2)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for li, ln in enumerate(lines):
            for ch, x in zip(ln, adv[li][0]):
                d.text((pad + x, pad + li * lh), ch, font=f, fill=col, stroke_width=stroke, stroke_fill=scol)
    else:
        step = size * (1 + spacing)
        colw = size * line_gap
        hmax = max(len(ln) for ln in lines) * step
        im = Image.new("RGBA", (int(colw * (len(lines) - 1) + size * 1.3 + pad * 2), int(hmax + pad * 2) + 1), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for li, ln in enumerate(lines):
            x = im.width - pad - size * 1.15 - li * colw
            for ci, ch in enumerate(ln):
                cw = f.getlength(ch)
                if ch in "、。，,.":
                    d.text((x + size * 0.55, pad + ci * step - size * 0.45), ch, font=f, fill=col)
                elif ch in "ー—―…「」『』（）()〜~-":
                    tmp = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
                    ImageDraw.Draw(tmp).text((size * 0.5, size * 0.3), ch, font=f, fill=col)
                    tmp = tmp.rotate(-90, resample=Image.BICUBIC)
                    im.alpha_composite(tmp, (int(x - size * 0.35), int(pad + ci * step - size * 0.5)))
                else:
                    d.text((x + (size - cw) / 2, pad + ci * step), ch, font=f, fill=col, stroke_width=stroke, stroke_fill=scol)
    return to_rgba_f(im)


def text(dst, s, x, y, size=28, font="sans", color=(0.2, 0.2, 0.22), anchor=(0.5, 0.5), alpha=1.0, vertical=False,
         spacing=0.12, line_gap=1.6, scale=1.0, rot=0.0, mode="normal", stroke=0, stroke_color=(1, 1, 1)):
    if not s or alpha <= 0.001:
        return None
    img = text_rgba(s, size, font, tuple(float(c) for c in color), vertical, spacing, line_gap, stroke, tuple(stroke_color))
    put(dst, img, x, y, scale=scale, rot=rot, alpha=alpha, anchor=anchor, mode=mode)
    return img.shape[1] * scale, img.shape[0] * scale


def typed(s, k):
    """Return prefix of s for typewriter progress k in [0,1]."""
    n = int(round(len(s) * clamp(k)))
    return s[:n]


def glitch_text(s, k, seed=0):
    """Characters resolve from random glyphs as k goes 0->1."""
    pool = "01アイウエオカキ#%&*<>/\\|=+?█▓▒░"
    out = []
    for i, ch in enumerate(s):
        if ch in " \n":
            out.append(ch)
        elif hash01(i, seed) < k:
            out.append(ch)
        elif k > 0:
            out.append(pool[int(hash01(i, seed, int(k * 20)) * len(pool))])
    return "".join(out)


# ---------------------------------------------------------------- images
def _grade_mono_orange(rgba, keep=1.0, sat_floor=0.30, contrast=1.0, lift=0.0):
    """Desaturate everything except the orange family -> wowaka-like monochrome + one accent."""
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    hsv = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat = hsv[..., 0] * 2, hsv[..., 1] / 255
    m = np.clip((sat - sat_floor) / 0.25, 0, 1) * np.clip(1 - np.abs(hue - 20) / 22, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)[..., None] * keep
    gray = (rgb @ np.array([0.30, 0.59, 0.11], np.float32))[..., None]
    gray = np.clip((gray - 0.5) * contrast + 0.5 + lift, 0, 1)
    out = gray * (1 - m) + rgb * m
    return np.concatenate([out, a], 2)


@lru_cache(maxsize=64)
def load_asset(name, grade="mono", max_h=1600):
    """Load an RGBA asset (raw/<name>.png), straight alpha -> graded, premultiplied float."""
    p = ASSETS / "raw" / f"{name}.png"
    im = Image.open(p).convert("RGBA")
    if im.height > max_h:
        im = im.resize((int(im.width * max_h / im.height), max_h), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    if grade == "mono":
        a = _grade_mono_orange(a, keep=0.95, contrast=1.05)
    elif grade == "gray":
        a = _grade_mono_orange(a, keep=0.0)
    elif grade == "soft":
        a = _grade_mono_orange(a, keep=0.8, contrast=0.9, lift=0.03)
    a[..., :3] *= a[..., 3:4]
    return a


@lru_cache(maxsize=32)
def silhouette(name, color=(0.25, 0.25, 0.27), max_h=1600):
    a = load_asset(name, "none", max_h).copy()
    al = a[..., 3:4]
    a[..., :3] = np.asarray(color, np.float32) * al
    return a


@lru_cache(maxsize=16)
def pixelize(name, px_h=64, max_h=1600):
    a = load_asset(name, "mono", max_h)
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, int(w * px_h / h)), px_h), interpolation=cv2.INTER_AREA)
    al = small[..., 3:4]
    col = np.where(al > 1e-3, small[..., :3] / np.maximum(al, 1e-3), 0)
    col = np.round(col * 5) / 5
    al = (al > 0.45).astype(np.float32)
    small = np.concatenate([col * al, al], 2)
    return small


def big_nearest(src, scale):
    h, w = src.shape[:2]
    return cv2.resize(src, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)


def tint(src, color, amount=1.0):
    out = src.copy()
    out[..., :3] = out[..., :3] * (1 - amount) + np.asarray(color, np.float32) * out[..., 3:4] * amount
    return out


# ---------------------------------------------------------------- post fx
_rng = np.random.default_rng(7)
_GRAIN = [(_rng.standard_normal((H // 2, W // 2)).astype(np.float32)) for _ in range(6)]
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_VIG = (1 - 0.17 * (((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2) ** 1.5).clip(0.5, 1)[..., None]


def post(c, t, grain=0.035, vignette=1.0, flash=0.0, bloom=0.0, ca=0.0, letterbox=0.0, fade_black=0.0, fade_white=0.0):
    if bloom > 0:
        small = cv2.resize(c, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        bright = np.clip(small - 0.65, 0, None)
        bright = cv2.GaussianBlur(bright, (0, 0), 6)
        c += cv2.resize(bright, (W, H), interpolation=cv2.INTER_LINEAR) * bloom
    if ca > 0:
        sh = int(ca)
        if sh:
            c[:, sh:, 0] = c[:, :-sh, 0]
            c[:, :-sh, 2] = c[:, sh:, 2]
    if vignette > 0:
        c *= 1 - (1 - _VIG) * vignette
    if flash > 0:
        c += (1 - c) * clamp(flash)
    if fade_white > 0:
        c += (1 - c) * clamp(fade_white)
    if fade_black > 0:
        c *= 1 - clamp(fade_black)
    if grain > 0:
        g = _GRAIN[int(t * FPS) % len(_GRAIN)]
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
        c += g[..., None] * grain
    if letterbox > 0:
        bh = int(H * 0.125 * letterbox)
        c[:bh] = 0.03
        c[H - bh:] = 0.03
    np.clip(c, 0, 1, out=c)
    return c


def to_u8(c):
    return (c * 255 + 0.5).astype(np.uint8)
