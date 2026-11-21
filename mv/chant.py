"""1:46-1:57  The scream, as one continuous shot.

Her hair does not end: past its tips it runs on as circuit traces. Her voice travels down them as light and dies
against the edge of the die. Around her, memory cells take in every token she throws out, ring after ring, until the
room she is shut in is full - and the context overflows (next scene).
"""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from core import *
from ui import Layer
import kt
import ai
from scenes import downs, OR, LYS
from act import ctx_fill, _ctx_bar, _ctx_lines, CH0, CH_PULL, CH_CTX

ASSET = "n02_scream"
AW, AH = 1024, 1536
S0 = 880 / AH             # asset px -> world px (world px == screen px at zoom 1)
DIE = 1500                # half size of the die, world px
CELL = 40                 # one memory cell, world px
TEX = 20                  # texture px per cell
NC = 2 * DIE // CELL      # cells per side
Z0, Z1 = 2.3, 0.2         # camera: close-up -> whole chip
T_END = CH_CTX


def a2w(u, v):
    return ((u - AW / 2) * S0, (v - AH / 2) * S0)


MOUTH = a2w(528, 228)
FOCUS = a2w(530, 400)     # close-up framing: head to waist
HAIR = (0.93, 0.50, 0.30)
TRACE = (0.62, 0.33, 0.22)
PULSE = (1.0, 0.86, 0.70)


# ---------------------------------------------------------------- geometry (built once)
@lru_cache(maxsize=1)
def _tips():
    raw = np.asarray(Image.open(ASSETS / "raw" / f"{ASSET}.png").convert("RGBA")).astype(np.float32) / 255
    rgb, al = raw[..., :3], raw[..., 3]
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    orange = (al > 0.5) & (hsv[..., 1] > 115) & (np.abs(hsv[..., 0] * 2 - 20) < 18)
    cx, cy = 0.5 * AW, 0.40 * AH
    ys, xs = np.nonzero(orange)
    ang = np.degrees(np.arctan2(ys - cy, xs - cx)) % 360
    dist = np.hypot(xs - cx, ys - cy)
    out = []
    for b in np.arange(0, 360, 4.5):
        m = (ang >= b) & (ang < b + 4.5)
        if m.sum() < 30:
            continue
        j = int(np.argmax(np.where(m, dist, -1)))
        x, y = int(xs[j]), int(ys[j])
        if math.hypot(x - 540, y - 170) < 160:      # skip strands that end on top of the head
            continue
        out.append((x, y))
    return out


def _route(q, e, side, f):
    """45-degree PCB routing from q to pad e; side 'h' = pad on a left/right edge."""
    (qx, qy), (ex, ey) = q, e
    dx, dy = ex - qx, ey - qy
    sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    if side == "h":
        if abs(dx) >= abs(dy):
            a = (abs(dx) - abs(dy)) * f
            return [q, (qx + sx * a, qy), (qx + sx * (a + abs(dy)), ey), e]
        return [q, (qx, ey - sy * abs(dx)), e]
    if abs(dy) >= abs(dx):
        a = (abs(dy) - abs(dx)) * f
        return [q, (qx, qy + sy * a), (ex, qy + sy * (a + abs(dx))), e]
    return [q, (ex - sx * abs(dy), qy), e]


@lru_cache(maxsize=1)
def _traces():
    """Each hair tip continues as a trace to its own bond pad on the die edge."""
    c0 = a2w(AW / 2, 0.40 * AH)
    pads_used = set()
    out = []
    for k, (u, v) in enumerate(_tips()):
        p = a2w(u, v)
        ang = math.atan2(p[1] - c0[1], p[0] - c0[0])
        ca, sa = math.cos(ang), math.sin(ang)
        q = (p[0] + ca * 70, p[1] + sa * 70)            # the strand runs on a little before it turns into copper
        tx = DIE / max(abs(ca), 1e-6)
        ty = DIE / max(abs(sa), 1e-6)
        if tx <= ty:
            side, ex, ey = "h", math.copysign(DIE, ca), sa * tx
        else:
            side, ex, ey = "v", ca * ty, math.copysign(DIE, sa)
        # snap to a pad slot (every 50 px), no two traces share a pad
        slot = 50
        along = ey if side == "h" else ex
        n = int(round(along / slot))
        for dn in (0, 1, -1, 2, -2, 3, -3, 4, -4):
            key = (side, math.copysign(1, ex if side == "h" else ey), n + dn)
            if key not in pads_used and abs((n + dn) * slot) < DIE - 60:
                pads_used.add(key)
                n += dn
                break
        if side == "h":
            ey = n * slot
            e = (ex - math.copysign(18, ex), ey)
        else:
            ex = n * slot
            e = (ex, ey - math.copysign(18, ey))
        pts = [p] + _route(q, e, side, 0.15 + 0.5 * hash01(k, 3))
        P = np.array(pts, np.float32)
        seg = np.hypot(*(P[1:] - P[:-1]).T)
        out.append((P, np.concatenate([[0], np.cumsum(seg)]), (ex, ey), side))
    return out


@lru_cache(maxsize=1)
def _cell_dist():
    g = (np.arange(NC) + 0.5) * CELL - DIE
    xx, yy = np.meshgrid(g, g)
    rng = np.random.default_rng(5)
    return (np.hypot(xx, yy * 1.08) + rng.uniform(-70, 70, (NC, NC))).astype(np.float32)


@lru_cache(maxsize=1)
def _grid_tex():
    n = NC * TEX
    g = np.zeros((n, n), np.float32)
    g[::TEX, :] = 1
    g[:, ::TEX] = 1
    return g


# ---------------------------------------------------------------- camera
# one continuous move: hold the close-up, ease out to the full figure, then keep pulling until she is a speck on a chip
_KEYS = [(CH0, Z0), (109.6, 1.85), (CH_PULL, 1.0), (115.2, 0.36), (T_END, 0.2)]


def _hermite():
    ts = np.array([k[0] for k in _KEYS])
    ys = np.log(np.array([k[1] for k in _KEYS]))
    d = np.diff(ys) / np.diff(ts)
    m = np.zeros_like(ys)
    m[1:-1] = np.where(d[:-1] * d[1:] > 0, 2 / (1 / d[:-1] + 1 / d[1:]), 0)
    m[0], m[-1] = 0.0, d[-1] * 0.4
    return ts, ys, m


_HT = _hermite()


def _logz(t):
    ts, ys, m = _HT
    t = clamp(t, ts[0], ts[-1])
    i = min(len(ts) - 2, int(np.searchsorted(ts, t, side="right")) - 1)
    h = ts[i + 1] - ts[i]
    u = (t - ts[i]) / h
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]


def _g(t):
    """0 at the close-up, 1 at the end of the pull."""
    return clamp((math.log(Z0) - _logz(t)) / (math.log(Z0) - math.log(Z1)))


def camera(t):
    z = math.exp(_logz(t))
    w = ease_io(clamp((Z0 - z) / (Z0 - 0.9)))
    cx, cy = lerp(FOCUS[0], 0.0, w), lerp(FOCUS[1], 0.0, w)
    return z, cx, cy


def _to_screen(P, cam):
    z, cx, cy = cam
    return (P - (cx, cy)) * z + (W / 2, H / 2)


# ---------------------------------------------------------------- layers
def _memory(c, t, cam, k):
    """Memory cells around her: the context, filling ring by ring from where she stands."""
    z, cx, cy = cam
    r_f = lerp(DIE * 0.12, DIE * 1.5, clamp((k - 0.30) / 0.70))
    d = _cell_dist()
    filled = (d < r_f).astype(np.float32)
    fresh = ((d < r_f) & (d > r_f - 110)).astype(np.float32)
    idx, rnd = ai._ticks(NC, NC, 33, t, 6)
    glyph = ai._glyph_grid(idx, TEX)
    rep = lambda m: np.repeat(np.repeat(m, TEX, 0), TEX, 1)
    quiet = clamp(1.25 - 0.32 * z) * (1 + 1.6 * clamp(1 - z))   # quiet near her; brighter once it is tiny
    a_old = rep(filled * (1 - fresh) * (0.16 + 0.24 * rnd)) * glyph * quiet
    a_new = rep(fresh * (0.25 + 0.45 * rnd)) * glyph * quiet
    grid = _grid_tex() * 0.10
    a = np.clip(a_old + a_new + grid, 0, 1)
    rgb = (a_old[..., None] * np.array([0.42, 0.42, 0.45], np.float32)
           + a_new[..., None] * np.asarray(OR, np.float32)
           + grid[..., None] * np.array([0.5, 0.5, 0.52], np.float32))
    tex = np.concatenate([rgb, a[..., None]], 2)
    s = z * CELL / TEX
    put(c, tex, W / 2 - cx * z, H / 2 - cy * z, scale=s)


PITCH = 3440              # wafer die-to-die pitch (2*DIE + 440 scribe street)


@lru_cache(maxsize=1)
def _scream_glow():
    """Pre-blurred warm orange vocal aura behind n02_scream (premultiplied)."""
    im = load_asset(ASSET, "mono")
    a = im[..., 3]
    blur = cv2.GaussianBlur(a, (0, 0), 28.0)
    blur = np.clip(blur * 1.45, 0, 1)
    out = np.zeros_like(im)
    out[..., 0] = blur * OR[0]
    out[..., 1] = blur * OR[1]
    out[..., 2] = blur * OR[2]
    out[..., 3] = blur
    return out


def _vocal_onsets():
    """Distinct major vocal surge times in words(23) (de-duplicated within 0.35s)."""
    out = []
    for w in kt.words(23):
        if not out or w.t - out[-1] > 0.35:
            out.append(w.t)
    return out


def _wafer(c, t, cam, k, wall_hit):
    """Surrounding cold-grey silicon wafer dies revealed as camera pulls out.
    Lit faintly only by spillover warmth from her active die (0,0), fading into dark corners.
    """
    z, cx, cy = cam
    if z >= 0.88:
        return
    vis = ease_out(clamp((0.88 - z) / 0.45))
    tr = lambda x, y: ((x - cx) * z + W / 2, (y - cy) * z + H / 2)

    L = Layer(ss=1)
    # 1. Neighboring silent silicon dies across the wafer
    for ix in (-2, -1, 0, 1, 2):
        for iy in (-1, 0, 1):
            if ix == 0 and iy == 0:
                continue
            ox, oy = ix * PITCH, iy * PITCH
            # Quick screen-bounds cull
            sx0, sy0 = tr(ox - DIE, oy - DIE)
            sx1, sy1 = tr(ox + DIE, oy + DIE)
            if sx1 < -20 or sx0 > W + 20 or sy1 < -20 or sy0 > H + 20:
                continue
            dist_d = math.hypot(ix, iy)
            # Proximity visibility: immediate neighbors clearly catch cold-grey/warm-spill light; outer corners recede smoothly
            prox = math.exp(-dist_d * 0.45)
            al = vis * (0.48 + 0.52 * prox) * (0.80 + 0.20 * k)
            warm_spill = (0.18 + 0.26 * wall_hit) * prox * vis
            sub_col = (0.068 + 0.026 * warm_spill, 0.070 + 0.012 * warm_spill, 0.078)
            fill(c, sub_col, al * 0.95, (sx0, sy0, sx1, sy1))

            b_col = (0.34 + 0.16 * warm_spill, 0.35 + 0.07 * warm_spill, 0.39)
            L.rect(sx0, sy0, sx1, sy1, b_col, max(1, int(round(4 * z))), a=al)

            # Minimalist IC macro-block architecture inside each dead neighboring die
            m_col = (0.24 + 0.08 * warm_spill, 0.25 + 0.04 * warm_spill, 0.28)
            inner = DIE - 180
            ix0, iy0 = tr(ox - inner, oy - inner)
            ix1, iy1 = tr(ox + inner, oy + inner)
            L.rect(ix0, iy0, ix1, iy1, m_col, 1, a=al * 0.90)
            # Sub-bank crossbars & quiet memory tile grid
            for f in (-0.33, 0.33):
                L.line(tr(ox + inner * f, oy - inner), tr(ox + inner * f, oy + inner), m_col, 1, a=al * 0.75)
                L.line(tr(ox - inner, oy + inner * f), tr(ox + inner, oy + inner * f), m_col, 1, a=al * 0.75)
            # Quiet center core outline (empty dark silhouette where no voice lives)
            c0x, c0y = tr(ox - 340, oy - 340)
            c1x, c1y = tr(ox + 340, oy + 340)
            L.rect(c0x, c0y, c1x, c1y, m_col, 1, a=al * 0.85)

            # Micro bond-pad ticks along perimeter of neighboring dies
            step = 150
            p_col = (0.29 + 0.12 * warm_spill, 0.30 + 0.05 * warm_spill, 0.34)
            for a_ in range(-DIE + 160, DIE - 120, step):
                for (px, py) in ((ox + DIE - 22, oy + a_), (ox - DIE + 22, oy + a_),
                                 (ox + a_, oy + DIE - 22), (ox + a_, oy - DIE + 22)):
                    p0, p1 = tr(px - 15, py - 15), tr(px + 15, py + 15)
                    L.rect(p0[0], p0[1], p1[0], p1[1], p_col, a=al * 0.80)

    # 2. Razor-thin wafer scribe streets & photolithography alignment crosshairs (+)
    st_col = (0.24, 0.25, 0.28)
    for m in (-1.5, -0.5, 0.5, 1.5):
        v = m * PITCH
        L.line(tr(v, -PITCH * 2), tr(v, PITCH * 2), st_col, 1, a=vis * 0.65)
        L.line(tr(-PITCH * 2.5, v), tr(PITCH * 2.5, v), st_col, 1, a=vis * 0.65)
    for mx in (-0.5, 0.5):
        for my in (-0.5, 0.5):
            cx_, cy_ = mx * PITCH, my * PITCH
            arm = 110
            cr_col = (0.48, 0.49, 0.54)
            L.line(tr(cx_ - arm, cy_), tr(cx_ + arm, cy_), cr_col, 1, a=vis * 0.85)
            L.line(tr(cx_, cy_ - arm), tr(cx_, cy_ + arm), cr_col, 1, a=vis * 0.85)
    L.comp(c)


def _vocal_waves(c, t, cam):
    """Sleek outward-expanding acoustic pressure arcs from her mouth on every vocal surge."""
    z, cx, cy = cam
    mx, my = (MOUTH[0] - cx) * z + W / 2, (MOUTH[1] - cy) * z + H / 2
    W_lay = Layer(ss=1)
    has_any = False
    for wt in _vocal_onsets():
        age = t - wt
        if 0.0 <= age < 1.15:
            has_any = True
            u = age / 1.15
            eu = ease_out(u)
            r = (45.0 + 1350.0 * eu) * z
            fade = (1.0 - u) ** 1.7 * clamp(age / 0.06)
            W_lay.circle((mx, my), r, (0.95, 0.88, 0.80), max(1, int(round(2.4 * z * (1 - 0.5 * u)))), a=0.34 * fade)
            if age < 0.75:
                u2 = age / 0.75
                r2 = (30.0 + 820.0 * ease_out(u2)) * z
                W_lay.circle((mx, my), r2, OR, max(1, int(round(1.8 * z))), a=0.30 * (1.0 - u2) ** 1.5)
    if has_any:
        W_lay.comp(c, 1.0, mode="add")


def _die(c, t, cam, lit):
    """The edge of her active die: pads, bond wires, guard-ring, wall impact & inward acoustic rebound."""
    z, cx, cy = cam
    L = Layer(ss=1)
    tr = lambda x, y: ((x - cx) * z + W / 2, (y - cy) * z + H / 2)
    x0, y0 = tr(-DIE, -DIE)
    x1, y1 = tr(DIE, DIE)

    # Compute wall-impact intensity from struck pads + recent vocal surges arriving at DIE
    hit_avg = min(1.0, sum(lit.values()) / 18.0) if lit else 0.0
    edge_col = (lerp(0.42, 0.92, hit_avg * 0.75), lerp(0.42, 0.58, hit_avg * 0.75), lerp(0.45, 0.38, hit_avg * 0.75))
    L.rect(x0, y0, x1, y1, edge_col, max(1, int(round(6 * z))))
    P = DIE + 165
    px0, py0 = tr(-P, -P)
    px1, py1 = tr(P, P)
    L.rect(px0, py0, px1, py1, (0.30, 0.30, 0.32), max(1, int(round(3 * z))))
    pad = 13
    hits = []
    for side in range(4):
        for n in range(-29, 30):
            a = n * 50
            if side == 0:
                p, q, key = (DIE - 18, a), (P - 28, a * 1.08), ("h", 1.0, n)
            elif side == 1:
                p, q, key = (-DIE + 18, a), (-P + 28, a * 1.08), ("h", -1.0, n)
            elif side == 2:
                p, q, key = (a, DIE - 18), (a * 1.08, P - 28), ("v", 1.0, n)
            else:
                p, q, key = (a, -DIE + 18), (a * 1.08, -P + 28), ("v", -1.0, n)
            glow = lit.get(key, 0.0)
            col_ = (0.36, 0.36, 0.38)
            a0, a1 = tr(p[0] - pad, p[1] - pad), tr(p[0] + pad, p[1] + pad)
            L.rect(a0[0], a0[1], a1[0], a1[1], col_)
            L.line(tr(*p), tr(*q), (0.28, 0.28, 0.30), 1)
            l0, l1 = tr(q[0] - 12, q[1] - 12), tr(q[0] + 12, q[1] + 12)
            L.rect(l0[0], l0[1], l1[0], l1[1], (0.24, 0.24, 0.26))
            if glow > 0:
                hits.append((p, side, glow))
    L.comp(c)

    # Light that struck the silicon wall + subtle inward acoustic rebound waves (voice trapped inside die)
    G = Layer(ss=1)
    for p, side, g in hits:
        cxs, cys = tr(*p)
        G.circle((cxs, cys), max(3.5, 42 * z * (1.35 - 0.3 * g)), (0.65 * g, 0.32 * g, 0.18 * g))
        G.circle((cxs, cys), max(1.6, 13 * z), (g, 0.88 * g, 0.76 * g))
        # Short hot wall-segment flare along the die edge where the trace dies
        seg_l = 48 * z
        if side in (0, 1):
            G.line((cxs, cys - seg_l), (cxs, cys + seg_l), (0.85 * g, 0.46 * g, 0.24 * g), max(1, int(round(3 * z))))
        else:
            G.line((cxs - seg_l, cys), (cxs + seg_l, cys), (0.85 * g, 0.46 * g, 0.24 * g), max(1, int(round(3 * z))))

    # Inward-bouncing acoustic echo frames when vocal waves reach the die perimeter (~0.74s travel)
    for wt in _vocal_onsets():
        age_w = t - (wt + 0.74)
        if 0.0 <= age_w < 0.70:
            uw = age_w / 0.70
            # Rebounds inward from DIE -> 0.62*DIE and fades gracefully
            rad_w = lerp(DIE, DIE * 0.62, ease_out(uw))
            rx0, ry0 = tr(-rad_w, -rad_w)
            rx1, ry1 = tr(rad_w, rad_w)
            fade_w = (1.0 - uw) ** 1.7 * clamp(age_w / 0.05)
            G.rect(rx0, ry0, rx1, ry1, (0.78 * fade_w, 0.42 * fade_w, 0.22 * fade_w), max(1, int(round(3.5 * z))))
            if uw < 0.35:
                rim_a = (1.0 - uw / 0.35) * 0.55
                G.rect(x0, y0, x1, y1, (0.90 * rim_a, 0.52 * rim_a, 0.28 * rim_a), max(2, int(round(7 * z))))
    G.comp(c, 1.0, mode="add")
    return hit_avg


def _pulses(t):
    """(trace index, spawn time, travel time) of light running down her hair."""
    tr = _traces()
    N = len(tr)
    out = []
    f1 = beat_pos(t) * 4
    f0 = beat_pos(CH0) * 4
    for k in range(max(int(f0), int(f1) - 14), int(f1) + 1):
        tk = t - (f1 - k) / 4 * BEAT
        for j in range(2):
            out.append((int(hash01(k, j, 1) * N) % N, tk, 0.55 + 0.25 * hash01(k, j, 2)))
    for w in kt.words(23):                  # every "o" she cries sends a surge down every strand at once
        if w.t <= t < w.t + 1.4:
            for i in range(N):
                out.append((i, w.t, 0.70 + 0.2 * hash01(i, 9)))
    return out


def _point_at(P, cum, s):
    s = clamp(s, 0, cum[-1])
    i = min(len(cum) - 2, int(np.searchsorted(cum, s, side="right")) - 1)
    u = (s - cum[i]) / max(1e-6, cum[i + 1] - cum[i])
    return P[i] + (P[i + 1] - P[i]) * u


def _sub(P, cum, s0, s1):
    s0, s1 = clamp(s0, 0, cum[-1]), clamp(s1, 0, cum[-1])
    pts = [_point_at(P, cum, s0)]
    for i in range(1, len(P) - 1):
        if s0 < cum[i] < s1:
            pts.append(P[i])
    pts.append(_point_at(P, cum, s1))
    return np.array(pts, np.float32)


def _wires(c, t, cam, k):
    z = cam[0]
    tr = _traces()
    base = Layer(ss=1)
    th = max(1, int(round(1.7 * z)))
    for P, cum, _, _ in tr:
        S = _to_screen(P, cam)
        base.polyline([tuple(p) for p in S[:2]], HAIR, th)
        base.polyline([tuple(p) for p in S[1:]], TRACE, th)
    base.comp(c, 0.55 + 0.35 * k)
    glow = Layer(ss=1)
    lit = {}
    for i, t0, dur in _pulses(t):
        P, cum, e, side = tr[i]
        age = t - t0
        if age < 0:
            continue
        u = age / dur
        if u < 1:
            s = cum[-1] * u
            seg = _sub(P, cum, s - 160, s)
            S = _to_screen(seg, cam)
            glow.polyline([tuple(p) for p in S], PULSE, max(1, int(round(3.0 * z))))
            hx, hy = S[-1]
            glow.circle((hx, hy), max(1.5, 5 * z), (1, 0.95, 0.9))
        elif age < dur + 0.35:
            v = 1 - (age - dur) / 0.35
            n = int(round((e[1] if side == "h" else e[0]) / 50))
            key = (side, math.copysign(1, e[0] if side == "h" else e[1]), n)
            lit[key] = max(lit.get(key, 0.0), v)
    glow.comp(c, 1.0, mode="add")
    return lit


def _tokens(c, t, cam, fade):
    """Lyric tokens thrown out of her mouth (16th notes), in world space."""
    if fade <= 0.01:
        return
    z, cx, cy = cam
    mx, my = (MOUTH[0] - cx) * z + W / 2, (MOUTH[1] - cy) * z + H / 2
    lines = _ctx_lines()
    rate, life = 8, 1.3
    f1 = beat_pos(t) * rate
    k1 = int(math.floor(f1))
    for k in range(k1 - int(life / BEAT * rate) - 1, k1 + 1):
        tk = t - (f1 - k) / rate * BEAT
        if tk < CH0:
            continue
        u = (t - tk) / life
        if not (0 <= u < 1):
            continue
        a = -math.pi / 2 + (hash01(k, 11) - 0.5) * 2.6        # mostly upward, out of the open mouth
        r = (40 + 900 * ease_out(u) * (0.5 + 0.7 * hash01(k, 12))) * z / Z0 * 1.6
        s = lines[k % len(lines)].replace(" ", "")
        a0 = int(hash01(k, 13) * max(1, len(s) - 4))
        s = s[a0:a0 + 2 + int(hash01(k, 14) * 4)]
        col_ = OR if k % 7 == 0 else (0.92, 0.92, 0.94)
        size = max(14, int(40 * z / Z0 * 1.3))
        put(c, kt.wimg(s, size, "jpm", col_, raw=True), mx + r * math.cos(a), my + r * math.sin(a),
            scale=0.7 + 0.8 * u, alpha=fade * (1 - u) ** 1.3)


# ---------------------------------------------------------------- the shot
def s_scream(c, t, lt, d):
    cam = camera(t)
    z, cx, cy = cam
    k = ctx_fill(t)
    c[:] = 0.022
    # silicon: her active die is a shade lighter than the dark around it
    x0, y0 = (-DIE - cx) * z + W / 2, (-DIE - cy) * z + H / 2
    x1, y1 = (DIE - cx) * z + W / 2, (DIE - cy) * z + H / 2
    fill(c, (0.048, 0.046, 0.050), 1.0, (x0, y0, x1, y1))
    _memory(c, t, cam, k)
    lit = _wires(c, t, cam, k)
    wall_hit = _die(c, t, cam, lit) if z < 0.88 else 0.0
    # surrounding cold-grey silent wafer dies revealed as camera pulls out (never overlaps die (0,0))
    _wafer(c, t, cam, k, wall_hit)
    _vocal_waves(c, t, cam)
    # her, with subtle warm vocal aura (smooth vocal envelope, ZERO kick-pulse scaling)
    img = load_asset(ASSET, "mono")
    vocal_env = max((math.exp(-(t - wt) / 0.55) for wt in _vocal_onsets() if t >= wt), default=0.0)
    jx, jy = kt.shake(t, 6 * kick_pulse(t) * min(1.0, z / 1.2), 4)
    hx, hy = W / 2 - cx * z + jx, H / 2 - cy * z + jy
    put(c, _scream_glow(), hx, hy, scale=S0 * z * 1.03,
        alpha=0.16 + 0.24 * vocal_env + 0.16 * _g(t), mode="add")
    put(c, img, hx, hy, scale=S0 * z)
    _tokens(c, t, cam, 1 - ramp(t, CH_PULL + 1.0, CH_PULL + 3.0))
    _ctx_bar(c, k)
    fl = kt.flash_at(t, [w.t for w in kt.words(23)], 0.10, 0.26) * (1 - 0.6 * _g(t))
    return dict(grain=0.03, bloom=0.45, flash=fl, vignette=lerp(1.0, 0.58, _g(t)))
