"""Japanese lyric typography: characters appear on their sung syllable (timed from the vocal stem)."""
import json
import math
from functools import lru_cache

from core import ROOT, text_rgba, put, _font, ease_out, clamp, hash01

_D = json.load(open(ROOT / "audio" / "lyrics.json", encoding="utf-8"))
LINES = _D["lines"]
for _i, _L in enumerate(LINES):
    _L["idx"] = _i
    s = _L["ja"]
    ph, cur = [], []
    for ci, ch in enumerate(s):
        if ch in " 　":
            if cur:
                ph.append(cur)
            cur = []
        else:
            cur.append(ci)
    if cur:
        ph.append(cur)
    _L["phrases"] = ph

INK = (0.2, 0.2, 0.22)
POP = 0.11


def T(i):
    return LINES[i]["t"]


def SPLIT(i):
    return LINES[i]["split"] or LINES[i]["t"]


def current(t, lo=None, hi=None):
    """Active line at time t (optionally restricted to an index range)."""
    best = None
    for L in LINES:
        if L["t"] - 0.02 <= t and (lo is None or L["idx"] >= lo) and (hi is None or L["idx"] <= hi):
            best = L
    if best is not None and t >= best["next"] - 0.02 and (hi is None or best["idx"] < hi):
        return None
    return best


def phrase_at(L, t):
    """Index of the phrase being sung (by time of its first character)."""
    k = 0
    for i, p in enumerate(L["phrases"]):
        if L["chars"][p[0]] - 0.02 <= t:
            k = i
    return k


@lru_cache(maxsize=8192)
def glyph(ch, size, font, color, stroke=0, stroke_color=(1, 1, 1)):
    return text_rgba(ch, size, font, color, False, 0.0, 1.0, stroke, stroke_color)


ROT = set("ー—―…〜~-(（)）「」『』")
SHIFT = set("、。，,.")
SMALL = set("ゃゅょぁぃぅぇぉっャュョァィゥェォッ")


def _layout(chars, size, font, vertical, spacing):
    """Return per-char offsets (x, y, rot) relative to the start, and the extent."""
    f = _font(font, size)
    pos = []
    if vertical:
        step = size * (1 + spacing)
        for k, ch in enumerate(chars):
            dx, dy, rot = 0.0, k * step, 0.0
            if ch in ROT:
                rot = -90.0
            elif ch in SHIFT:
                dx, dy = size * 0.55, k * step - size * 0.55
            elif ch in SMALL:
                dx, dy = size * 0.08, k * step - size * 0.08
            pos.append((dx, dy, rot))
        return pos, (size, step * len(chars))
    x = 0.0
    for ch in chars:
        w = f.getlength(ch)
        pos.append((x + w / 2, 0.0, 0.0))
        x += w + size * spacing
    return pos, (x - size * spacing, size)


def draw_chars(c, L, idxs, t, x, y, size=34, font="jpm", color=INK, vertical=False, spacing=0.08, anchor=(0.5, 0.5),
               alpha=1.0, pop=1.0, jitter=0.0, stroke=0, stroke_color=(1, 1, 1), accent=None, accent_color=None,
               hold_all=False):
    """Draw characters `idxs` of line L, each appearing at its sung time."""
    s = L["ja"]
    chars = [s[i] for i in idxs]
    pos, (wd, ht) = _layout(chars, size, font, vertical, spacing)
    if vertical:
        ox = x - (anchor[0] - 0.5) * size
        oy = y - anchor[1] * ht
    else:
        ox = x - anchor[0] * wd
        oy = y - (anchor[1] - 0.5) * size
    col = tuple(float(v) for v in color)
    for k, (ci, ch) in enumerate(zip(idxs, chars)):
        ta = L["chars"][ci]
        if not hold_all and t < ta - 0.005:
            continue
        a = 1.0 if hold_all else clamp((t - ta) / 0.05 + 0.25)
        e = 1.0 if hold_all else ease_out(clamp((t - ta) / POP))
        sc = 1 + 0.45 * pop * (1 - e)
        dx, dy, rot = pos[k]
        jx = jy = 0.0
        if jitter:
            jx = (hash01(ci, int(t * 24)) - 0.5) * jitter
            jy = (hash01(ci, int(t * 24), 3) - 0.5) * jitter
        cc = col
        if accent and ch in accent:
            cc = tuple(float(v) for v in accent_color)
        g = glyph(ch, size, font, cc, stroke, tuple(stroke_color))
        if vertical:
            put(c, g, ox + dx + jx, oy + dy + size * 0.62 + jy, scale=sc, rot=rot, alpha=a * alpha)
        else:
            put(c, g, ox + dx + jx, oy + dy + jy, scale=sc, rot=rot, alpha=a * alpha)
    return wd, ht


def extent(L, idxs, size, font="jpm", vertical=False, spacing=0.08):
    return _layout([L["ja"][i] for i in idxs], size, font, vertical, spacing)[1]


def phrase(c, L, k, t, x, y, **kw):
    if k >= len(L["phrases"]):
        return 0, 0
    return draw_chars(c, L, L["phrases"][k], t, x, y, **kw)


def line_h(c, L, t, x, y, gap=1.0, **kw):
    """Whole line on one row, phrases separated by a gap of `gap` ems."""
    idxs = []
    for k, p in enumerate(L["phrases"]):
        idxs += p
        if k < len(L["phrases"]) - 1:
            idxs.append(None)
    s = L["ja"]
    # use real space chars for gaps
    real = []
    for i in idxs:
        if i is None:
            real.append(next(j for j in range(real[-1] + 1, len(s)) if s[j] in " 　"))
        else:
            real.append(i)
    return draw_chars(c, L, real, t, x, y, **kw)


def columns(c, L, t, x, y, size=30, col_gap=1.9, phrases=None, right_to_left=True, **kw):
    """Vertical text: one column per phrase, right-to-left."""
    ph = L["phrases"] if phrases is None else [L["phrases"][k] for k in phrases]
    for k, p in enumerate(ph):
        xx = x - k * size * col_gap if right_to_left else x + k * size * col_gap
        draw_chars(c, L, p, t, xx, y, size=size, vertical=True, anchor=(0.5, 0.0), **kw)


def sung(L, t):
    """Fraction of the line's characters already sung."""
    ch = L["chars"]
    return sum(1 for v in ch if v <= t) / max(1, len(ch))


def char_time(L, ch):
    s = L["ja"]
    i = s.find(ch)
    return L["chars"][i] if i >= 0 else L["t"]
