"""Kinetic typography: word segmentation + motion primitives for the lyric layer."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import (W, H, put, text_rgba, _font, ease_out, ease_in, ease_io, ease_out_back, clamp, ramp, lerp, hash01,
                  to_rgba_f, kick_pulse)
import lyric as LY

LYS = LY.LINES
PUNCT = set("、。，,.！？!?―—…「」『』 　")

# ---------------------------------------------------------------- segmentation (bunsetsu-ish, via janome)
try:
    from janome.tokenizer import Tokenizer
    _TOK = Tokenizer()
except Exception:
    _TOK = None

CONTENT = {"名詞", "動詞", "形容詞", "副詞", "連体詞", "感動詞", "接頭詞", "形容動詞"}


def _segment(s):
    """Split s into word groups (lists of char indices). Particles / auxiliaries / punctuation attach left."""
    groups = []
    if _TOK is None:
        cur = []
        for i, ch in enumerate(s):
            if ch in " 　":
                if cur:
                    groups.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            groups.append(cur)
        return groups
    pos = 0
    cur = []
    prev_prefix = False
    for tok in _TOK.tokenize(s):
        surf = tok.surface
        p = tok.part_of_speech.split(",")
        i0 = s.find(surf, pos)
        if i0 < 0:
            i0 = pos
        idx = list(range(i0, i0 + len(surf)))
        pos = i0 + len(surf)
        if surf.strip() == "":
            if cur:
                groups.append(cur)
            cur = []
            prev_prefix = False
            continue
        starts = p[0] in CONTENT and p[1] not in ("非自立", "接尾") and not prev_prefix
        if all(s[j] in PUNCT for j in idx):
            starts = False
        if starts and cur:
            groups.append(cur)
            cur = []
        cur += idx
        prev_prefix = p[0] == "接頭詞"
    if cur:
        groups.append(cur)
    # merge tiny leading groups such as a lone kana into the next one
    out = []
    for g in groups:
        g = [j for j in g if s[j] not in " 　"]
        if g:
            out.append(g)
    return out


class Word:
    __slots__ = ("s", "idx", "t", "te", "line", "k")

    def __init__(self, line, k, idx):
        L = LYS[line]
        self.line, self.k, self.idx = line, k, idx
        self.s = "".join(L["ja"][j] for j in idx)
        self.t = L["chars"][idx[0]]
        self.te = L["chars"][idx[-1]]


WORDS = []
for _i, _L in enumerate(LYS):
    WORDS.append([Word(_i, k, g) for k, g in enumerate(_segment(_L["ja"]))])


def words(i):
    return WORDS[i]


def phrase_words(i, k):
    """Words belonging to phrase k (space separated) of line i."""
    ph = set(LYS[i]["phrases"][k])
    return [w for w in WORDS[i] if w.idx[0] in ph]


def join(ws):
    return "".join(w.s for w in ws)


def merge(i, a, b):
    """Pseudo-word made of words a..b (inclusive) of line i."""
    ws = WORDS[i][a:b + 1]
    return Word(i, a, [j for w in ws for j in w.idx])


def sub(i, s, occ=0):
    """Pseudo-word for the substring s of line i (occ-th occurrence)."""
    ja = LYS[i]["ja"]
    p = -1
    for _ in range(occ + 1):
        p = ja.find(s, p + 1)
    if p < 0:
        raise KeyError(s)
    return Word(i, 0, list(range(p, p + len(s))))


def active(i, t, keep=0.0):
    L = LYS[i]
    return L["t"] - 0.08 <= t < L["next"] + keep


# ---------------------------------------------------------------- glyph images
def fit(size):
    """Display sizes are compressed: very large type reads as karaoke, not as a lyric video."""
    size = float(size)
    return int(size if size <= 120 else 120 + (size - 120) * 0.45)


def face(font, size):
    """Large words are set in Mincho; small text stays Gothic."""
    if font == "jpb" or (font == "jpm" and size >= 96):
        return "min"
    return font


@lru_cache(maxsize=96)
def wimg(s, size, font="jpm", color=(0.2, 0.2, 0.22), vertical=False, stroke=0, stroke_color=(1, 1, 1), spacing=0.04,
         raw=False):
    if not raw:
        size = fit(size)
        font = face(font, size)
    img = text_rgba(s, int(size), font, tuple(float(v) for v in color), vertical, spacing, 1.15, 0, (1, 1, 1))
    if stroke <= 0:
        return img
    r = max(4, int(size * 0.12))
    a = np.pad(img[..., 3], r * 2)
    halo = cv2.GaussianBlur(cv2.dilate(a, np.ones((3, 3), np.uint8), iterations=max(1, r // 3)), (0, 0), r)
    halo = np.clip(halo * 0.75, 0, 1)
    out = np.zeros(a.shape + (4,), np.float32)
    out[..., :3] = np.array(stroke_color, np.float32) * halo[..., None]
    out[..., 3] = halo
    p = np.pad(img, ((r * 2, r * 2), (r * 2, r * 2), (0, 0)))
    out = out * (1 - p[..., 3:4]) + p
    return out


@lru_cache(maxsize=48)
def hollow(s, size, font="jpb", color=(0.2, 0.2, 0.22), width=3, vertical=False, spacing=0.04):
    """Formerly outline letters; now a faint filled watermark in the display face."""
    return wimg(s, size, font, color, vertical, 0, (1, 1, 1), spacing) * 0.4


def _hollow_outline(s, size, font="jpb", color=(0.2, 0.2, 0.22), width=3, vertical=False, spacing=0.04):
    size = int(size)
    f = _font(font, size)
    pad = width + 8
    if vertical:
        step = size * (1 + spacing)
        w, h = int(size * 1.4 + pad * 2), int(step * len(s) + pad * 2)
    else:
        w = int(sum(f.getlength(ch) + size * spacing for ch in s) + pad * 2)
        h = int(size * 1.35 + pad * 2)
    A = Image.new("L", (w, h), 0)
    Bm = Image.new("L", (w, h), 0)
    da, db = ImageDraw.Draw(A), ImageDraw.Draw(Bm)
    x = pad
    for k, ch in enumerate(s):
        if vertical:
            cw = f.getlength(ch)
            xy = (pad + (size * 1.4 - cw) / 2, pad + k * size * (1 + spacing))
        else:
            xy = (x, pad)
            x += f.getlength(ch) + size * spacing
        da.text(xy, ch, font=f, fill=255, stroke_width=width, stroke_fill=255)
        db.text(xy, ch, font=f, fill=255)
    a = np.clip(np.asarray(A, np.float32) - np.asarray(Bm, np.float32), 0, 255) / 255.0
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = np.array(color, np.float32) * a[..., None]
    out[..., 3] = a
    return out


def wsize(s, size, font="jpm", spacing=0.04):
    size = fit(size)
    f = _font(face(font, size), int(size))
    return sum(f.getlength(ch) + size * spacing for ch in s) - size * spacing, size


# ---------------------------------------------------------------- motion primitives
def shake(t, amp, seed=0, rate=30):
    if amp <= 0:
        return 0.0, 0.0
    k = int(t * rate)
    return (hash01(k, seed) - 0.5) * 2 * amp, (hash01(k, seed, 7) - 0.5) * 2 * amp


def slam(c, img, x, y, t, t0, s0=1.8, dur=0.10, alpha=1.0, anchor=(0.5, 0.5), rot=0.0, ghost=0.35, sh=0.0,
         mode="normal"):
    """Hit the frame: scale s0 -> 1 in `dur`, with a fading shock echo."""
    if t < t0 - 1e-3:
        return False
    u = clamp((t - t0) / dur)
    e = ease_out(u)
    sc = lerp(s0, 1.0, e)
    dx, dy = shake(t, sh * (1 - ramp(t, t0, t0 + 0.4)) + sh * 0.25, seed=int(x))
    put(c, img, x + dx, y + dy, scale=sc, rot=rot, alpha=alpha * clamp(u * 3 + 0.2), anchor=anchor, mode=mode)
    if ghost > 0:
        g = clamp((t - t0) / 0.22)
        if g < 1:
            put(c, img, x, y, scale=1 + 0.5 * ease_out(g), rot=rot, alpha=ghost * (1 - g) * alpha, anchor=anchor,
                mode=mode)
    return True


def slide(c, img, x, y, t, t0, dx=-260, dy=0, dur=0.14, alpha=1.0, anchor=(0.5, 0.5), rot=0.0, blur=True):
    """Slide in from (x+dx, y+dy) with a short motion trail."""
    if t < t0 - 1e-3:
        return False
    u = ease_out(clamp((t - t0) / dur))
    px, py = x + dx * (1 - u), y + dy * (1 - u)
    if blur and u < 1:
        for k in (3, 2, 1):
            v = max(0.0, u - 0.08 * k)
            put(c, img, x + dx * (1 - v), y + dy * (1 - v), alpha=alpha * 0.18, anchor=anchor, rot=rot)
    put(c, img, px, py, alpha=alpha * clamp(u * 2 + 0.2), anchor=anchor, rot=rot)
    return True


def wipe(c, img, x, y, t, t0, dur=0.18, alpha=1.0, anchor=(0.5, 0.5), axis=1, scale=1.0):
    """Reveal the image progressively along an axis (1 = left->right, 0 = top->bottom)."""
    if t < t0 - 1e-3:
        return False
    u = ease_out(clamp((t - t0) / dur))
    h, w = img.shape[:2]
    if axis == 1:
        n = max(1, int(w * u))
        part = img[:, :n]
        ox = x - anchor[0] * w * scale
        put(c, part, ox, y, alpha=alpha, anchor=(0.0, anchor[1]), scale=scale)
    else:
        n = max(1, int(h * u))
        part = img[:n]
        oy = y - anchor[1] * h * scale
        put(c, part, x, oy, alpha=alpha, anchor=(anchor[0], 0.0), scale=scale)
    return True


def glitch(c, img, x, y, t, amt=1.0, slices=7, alpha=1.0, anchor=(0.5, 0.5), seed=0, scale=1.0):
    """Horizontal slice displacement."""
    h, w = img.shape[:2]
    k = int(t * 24)
    edges = sorted({0, h} | {int(hash01(seed, k, j) * h) for j in range(slices - 1)})
    x0 = x - anchor[0] * w * scale
    y0 = y - anchor[1] * h * scale
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2:
            continue
        off = (hash01(seed, k, a, 3) - 0.5) * 2 * 60 * amt if hash01(seed, k, a, 4) < 0.55 * amt + 0.1 else 0
        put(c, img[a:b], x0 + off, y0 + a * scale, anchor=(0.0, 0.0), alpha=alpha, scale=scale)


def shatter(c, img, x, y, t, t0, alpha=1.0, n=6, force=900, seed=0, anchor=(0.5, 0.5), scale=1.0):
    """Break the image into n x n tiles flying apart after t0 (draw whole before t0)."""
    if t < t0:
        put(c, img, x, y, alpha=alpha, anchor=anchor, scale=scale)
        return
    u = t - t0
    h, w = img.shape[:2]
    x0 = x - anchor[0] * w * scale
    y0 = y - anchor[1] * h * scale
    th, tw = max(2, h // n), max(2, w // n)
    for r in range(0, h, th):
        for q in range(0, w, tw):
            tile = img[r:r + th, q:q + tw]
            if tile[..., 3].max() < 0.05:
                continue
            cx, cy = x0 + (q + tw / 2) * scale, y0 + (r + th / 2) * scale
            ang = math.atan2(cy - y, cx - x) + (hash01(r, q, seed) - 0.5)
            sp = force * (0.4 + hash01(q, r, seed, 1))
            px = cx + math.cos(ang) * sp * u
            py = cy + math.sin(ang) * sp * u + 900 * u * u
            put(c, tile, px, py, rot=(hash01(r, q, 2) - 0.5) * 720 * u, scale=scale,
                alpha=alpha * (1 - ramp(u, 0.25, 0.7)))


def echo_bg(c, s, t, size=60, color=(0.5, 0.5, 0.52), alpha=0.12, speed=90, gap=1.6, rows=None, font="jpb",
            y0=0, y1=H, rot=0.0):
    """Repeated text rows scrolling in alternating directions (typographic texture)."""
    img = wimg(s + "　", size, font, color)
    w = img.shape[1]
    lh = size * gap
    n = rows or int((y1 - y0) / lh) + 1
    for r in range(n):
        y = y0 + (r + 0.5) * lh
        dirn = 1 if r % 2 else -1
        off = (t * speed * dirn + hash01(r) * w) % w
        x = -w + off
        while x < W + w:
            put(c, img, x, y, anchor=(0, 0.5), alpha=alpha, rot=rot)
            x += w


def along_circle(c, s, cx, cy, R, ang0, size=34, font="jpm", color=(0.3, 0.3, 0.32), alpha=1.0, step=None):
    """Lay characters along a circle (tangent orientation)."""
    step = step or (size * 1.15) / R
    for k, ch in enumerate(s):
        a = ang0 + k * step
        g = wimg(ch, size, font, color)
        put(c, g, cx + R * math.cos(a), cy + R * math.sin(a), rot=math.degrees(a) + 90, alpha=alpha)


def invert_box(c, x, y, w, h, t, t0, dur=0.12, color=(0.12, 0.12, 0.13), anchor=(0.5, 0.5), from_left=True):
    from core import fill
    if t < t0:
        return 0.0
    u = ease_out(clamp((t - t0) / dur))
    x0 = x - anchor[0] * w
    y0 = y - anchor[1] * h
    if from_left:
        fill(c, color, 1.0, (x0, y0, x0 + w * u, y0 + h))
    else:
        fill(c, color, 1.0, (x0 + w * (1 - u), y0, x0 + w, y0 + h))
    return u


def zoom_through(c, img, x, y, t, t0, t1, z1=28.0, alpha=1.0):
    """Camera flies into the glyph: exponential scale from 1 to z1 over [t0, t1]."""
    if t < t0:
        put(c, img, x, y, alpha=alpha)
        return
    u = clamp((t - t0) / (t1 - t0))
    sc = math.exp(math.log(z1) * ease_in(u))
    put(c, img, x, y, scale=sc, alpha=alpha)


def tracking(c, s, x, y, t, t0, t1, size=40, font="jpm", color=(0.3, 0.3, 0.32), sp0=0.0, sp1=0.8, alpha=1.0):
    """Letter spacing that widens over time (centered)."""
    k = ease_out(clamp((t - t0) / max(1e-3, t1 - t0)))
    sp = lerp(sp0, sp1, k)
    f = _font(font, size)
    ws = [f.getlength(ch) for ch in s]
    total = sum(ws) + size * sp * (len(s) - 1)
    xx = x - total / 2
    for ch, cw in zip(s, ws):
        put(c, wimg(ch, size, font, color), xx + cw / 2, y, alpha=alpha)
        xx += cw + size * sp


def word_chars(c, w, t, x, y, size=40, font="jpm", color=(0.3, 0.3, 0.32), vertical=False, anchor=(0.5, 0.5),
               pop=1.0, alpha=1.0, spacing=0.04, accent=None, accent_color=None):
    """Per-character reveal for a word (used only where the character-by-character feel is wanted)."""
    return LY.draw_chars(c, LYS[w.line], w.idx, t, x, y, size=int(size), font=font, color=color, vertical=vertical,
                         anchor=anchor, pop=pop, alpha=alpha, spacing=spacing, accent=accent,
                         accent_color=accent_color)


def flash_at(t, times, dur=0.10, amt=0.5):
    best = 0.0
    for t0 in times:
        if t0 <= t < t0 + dur:
            best = max(best, amt * (1 - (t - t0) / dur))
    return best
