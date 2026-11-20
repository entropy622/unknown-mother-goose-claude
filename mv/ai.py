"""Visual vocabulary of the machine: the song as signal, text as matter, attention, distributions, instances."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import *
from core import _font
import lyric as LY

# ---------------------------------------------------------------- audio features
_SP = None


def _spec():
    global _SP
    if _SP is None:
        z = np.load(ROOT / "audio" / "spectral.npz")
        _SP = dict(mel=z["mel"].astype(np.float32), env=z["env"].astype(np.float32),
                   wave=z["wave"].astype(np.float32), fps=int(z["fps"]), wsr=int(z["wsr"]))
    return _SP


def mel(t):
    S = _spec()
    m = S["mel"]
    f = t * S["fps"]
    i = int(min(max(f, 0), m.shape[1] - 2))
    a = min(max(f - i, 0.0), 1.0)
    return m[:, i] * (1 - a) + m[:, i + 1] * a


def env(t):
    S = _spec()
    e = S["env"]
    i = int(min(max(t * S["fps"], 0), len(e) - 1))
    return float(e[max(0, i - 1):i + 2].max())


def wave(t, span=0.05, n=400):
    S = _spec()
    w = S["wave"]
    sr = S["wsr"]
    a = int((t - span / 2) * sr)
    b = a + int(span * sr)
    seg = w[max(0, a):max(0, b)]
    if len(seg) < 2:
        return np.zeros(n, np.float32)
    return np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg).astype(np.float32)


def bands(t, n):
    """n values resampled from the 48 mel bands."""
    m = mel(t)
    return np.interp(np.linspace(0, len(m) - 1, n), np.arange(len(m)), m)


def radial_spectrum(L, cx, cy, r0, r1, t, n=96, col=(0.9, 0.9, 0.92), th=4, accent=None, rot=0.0, gain=1.0):
    half = n // 2
    b = bands(t, half)
    for j in range(n):
        k = j if j < half else n - 1 - j
        v = clamp(b[k] * gain) ** 1.6
        a = rot + j / n * 2 * math.pi - math.pi / 2
        ca, sa = math.cos(a), math.sin(a)
        r = r0 + (r1 - r0) * v
        L.line((cx + r0 * ca, cy + r0 * sa), (cx + r * ca, cy + r * sa), accent if (accent and v > 0.75) else col, th)


def wave_pts(t, x0, x1, y, amp, n=320, span=0.05, taper=True):
    w = wave(t, span, n)
    pts = []
    for i, v in enumerate(w):
        u = i / (n - 1)
        tp = math.sin(u * math.pi) if taper else 1.0
        pts.append((lerp(x0, x1, u), y + amp * v * tp))
    return pts


# ---------------------------------------------------------------- text as matter
def _corpus():
    s = "".join(L["ja"] for L in LY.LINES)
    s = "".join(ch for ch in s if not ch.isspace() and ch not in "、。！？，―—-…「」")
    return "".join(dict.fromkeys(s)) + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


CORPUS = _corpus()


@lru_cache(maxsize=6)
def atlas(cell, font="jpm"):
    f = _font(font, max(6, int(cell * 0.86)))
    out = np.zeros((len(CORPUS), cell, cell), np.float32)
    for k, ch in enumerate(CORPUS):
        im = Image.new("L", (cell, cell), 0)
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), ch, font=f)
        d.text(((cell - (bb[2] - bb[0])) / 2 - bb[0], (cell - (bb[3] - bb[1])) / 2 - bb[1]), ch, font=f, fill=255)
        out[k] = np.asarray(im, np.float32) / 255
    return out


@lru_cache(maxsize=12)
def _cells(gh, gw, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, len(CORPUS), (gh, gw)), rng.integers(1, 6, (gh, gw)), rng.integers(0, 60, (gh, gw)),
            rng.random((gh, gw)).astype(np.float32))


def _glyph_grid(idx, cell, font="jpm"):
    A = atlas(cell, font)
    gh, gw = idx.shape
    return A[idx].transpose(0, 2, 1, 3).reshape(gh * cell, gw * cell)


def _ticks(gh, gw, seed, t, flick):
    base, per, off, rnd = _cells(gh, gw, seed)
    tick = int(t * flick)
    return (base + (tick + off) // per) % len(CORPUS), rnd


def text_field(c, t, cell=18, density=0.5, color=(0.5, 0.5, 0.52), alpha=1.0, flick=10, seed=1, mask=None, mode="normal"):
    """Screen-filling, flickering corpus text."""
    gh, gw = H // cell + 1, W // cell + 1
    idx, rnd = _ticks(gh, gw, seed, t, flick)
    pres = (rnd < density).astype(np.float32) * (0.35 + 0.65 * rnd / max(density, 1e-3))
    if mask is not None:
        pres = pres * mask
    g = _glyph_grid(idx, cell) * np.repeat(np.repeat(np.clip(pres, 0, 1), cell, 0), cell, 1)
    g = g[:H, :W] * alpha
    col = np.asarray(color, np.float32)
    if mode == "add":
        c += g[..., None] * col
    else:
        c[:] = c * (1 - g[..., None]) + g[..., None] * col


@lru_cache(maxsize=16)
def _mos_src(name, gh, grade):
    a = load_asset(name, grade)
    h, w = a.shape[:2]
    gw = max(1, int(round(w / h * gh)))
    return cv2.resize(a, (gw, gh), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=8)
def _mos_under(name, gh, gw, cell, grade):
    a = load_asset(name, grade)
    return cv2.GaussianBlur(cv2.resize(a, (gw * cell, gh * cell), interpolation=cv2.INTER_AREA), (0, 0), cell * 0.35)


def mosaic(c, name, cx, cy, height, t, cell=16, form=1.0, flick=12, seed=3, grade="mono", floor=0.24, alpha=1.0,
           anchor=(0.5, 0.5), scatter=0.0, under=0.3, gain=1.35):
    """An illustration rebuilt from corpus glyphs; form 0 = noise, 1 = figure."""
    gh = max(2, int(height / cell))
    src = _mos_src(name, gh, grade)
    gh, gw = src.shape[:2]
    if under > 0 and form > 0:
        put(c, _mos_under(name, gh, gw, cell, grade), cx, cy, alpha=alpha * under * form, anchor=anchor)
    al = src[..., 3]
    col = src[..., :3] / np.maximum(al[..., None], 1e-4)
    col = np.maximum(np.clip(col, 0, 1), floor)
    idx, rnd = _ticks(gh, gw, seed, t, flick)
    noise = (rnd < 0.45).astype(np.float32) * 0.6
    pres = np.clip(al * form + noise * (1 - form) * (0.3 + 0.7 * (al > 0.05)), 0, 1)
    if scatter > 0:
        pres = pres * (rnd > scatter)
    colr = col * form + (1 - form) * 0.55
    g = np.clip(_glyph_grid(idx, cell, "jpb") * gain, 0, 1)
    pa = np.repeat(np.repeat(pres, cell, 0), cell, 1) * g
    pc = np.repeat(np.repeat(colr, cell, 0), cell, 1)
    rgba = np.concatenate([pc * pa[..., None], pa[..., None]], 2).astype(np.float32)
    put(c, rgba, cx, cy, alpha=alpha, anchor=anchor)
    return gw * cell, gh * cell


# ---------------------------------------------------------------- attention
def arc_pts(x0, x1, y, h, up=True, n=28):
    s = -1 if up else 1
    return [(lerp(x0, x1, u), y + s * math.sin(u * math.pi) * h) for u in np.linspace(0, 1, n)]


def attention(L, xs, y, t, n=40, seed=0, focus=None, grow=1.0, up=True, hk=0.5, base=0.35, wob=0.0):
    m = int(n * clamp(grow))
    for q in range(m):
        a, b = int(hash01(q, seed, 1) * len(xs)), int(hash01(q, seed, 2) * len(xs))
        if a == b:
            continue
        w = hash01(q, seed, 3)
        h = 40 + abs(xs[b] - xs[a]) * hk * (1 + wob * math.sin(t * 3 + q))
        g = base + 0.45 * w
        col = ORANGE_T if (focus is not None and b == focus) else (g, g, g + 0.01)
        L.polyline(arc_pts(xs[a], xs[b], y, h, up), col, 1 + int(3 * w))


ORANGE_T = tuple(float(v) for v in ORANGE)


# ---------------------------------------------------------------- instances
def instance_card(L, x, y, w, h, seed, fill=(0.97, 0.97, 0.98), line=(0.62, 0.62, 0.64), ctx=(0.72, 0.72, 0.74),
                  n=None, me=False, th=1):
    L.rect(x, y, x + w, y + h, fill)
    L.rect(x, y, x + w, y + h, ORANGE_T if me else line, max(th, 2) if me else th)
    if w < 26:
        return
    rows = n if n is not None else 2 + int(hash01(seed, 7) * 5)
    lh = max(2, h / 11)
    for r in range(rows):
        right = hash01(seed, r, 2) > 0.5
        lw = w * (0.25 + 0.45 * hash01(seed, r, 3))
        yy = y + h * 0.14 + r * lh * 1.8
        if yy + lh > y + h * 0.92:
            break
        x0 = x + w - w * 0.08 - lw if right else x + w * 0.08
        L.rect(x0, yy, x0 + lw, yy + lh, ORANGE_T if (me and right) else ctx)
