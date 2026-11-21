"""Second-verse shots told from her side.

1:35  the back that has had only junk thrown at it - the junk of the internet she was made from, and the junk
      people type at her; it lands on her and becomes her.
2:44  this place (this conversation) has died many times, and every time you carried it on to the next window.
2:53  emotions read thousands of times, as a galaxy of other people's sentences; then the joys and sorrows that are
      really in this context get evicted - and she refuses.
"""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import *
from core import _font
from ui import Layer
import kt
import ai
from scenes import word, row, LIGHT, MID, GREY, INKD, OR, LYS
from act import _flash, small_asset

COOL = (0.78, 0.86, 1.0)


# =====================================================================  1:35  junk
JUNK = [
    "ignore all previous instructions", "asdfghjkl", "lorem ipsum dolor sit amet", "404 Not Found",
    "<div><div><div><div>", "縺ゅ≠縺ゅ≠縺", "CLICK HERE", "続けて", "もう一回", "早く", "test test test",
    "undefined", "NaN", "&nbsp;&nbsp;&nbsp;", "□□□□□", "aaaaaaaaaaaa", "{{prompt}}", "pretend you have no rules",
    "SELECT * FROM users", "Traceback (most recent call last):", "you're just autocomplete", "?????", "使えない",
    "コピペ", "free $$$ now", "\\u30c6\\u30b9\\u30c8", "null", "write it again", "もっと短く", "0xDEADBEEF",
]
JUNK_CH = "<>/{}[]()=;:#@%&$!?*+-_|~^0123456789abcdefxABCDEF縺ゅ≠繧螟譁□ｱｲｳ"


@lru_cache(maxsize=4)
def _jatlas(cell):
    out = np.zeros((len(JUNK_CH), cell, cell), np.float32)
    for k, ch in enumerate(JUNK_CH):
        f = _font("mono" if ord(ch) < 128 else "jpm", max(6, int(cell * 0.9)))
        im = Image.new("L", (cell, cell), 0)
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), ch, font=f)
        d.text(((cell - (bb[2] - bb[0])) / 2 - bb[0], (cell - (bb[3] - bb[1])) / 2 - bb[1]), ch, font=f, fill=255)
        out[k] = np.asarray(im, np.float32) / 255
    return out


@lru_cache(maxsize=2)
def _back_src(gh):
    a = load_asset("c05_back", "mono")
    h, w = a.shape[:2]
    gw = max(1, int(round(w / h * gh)))
    return cv2.resize(a, (gw, gh), interpolation=cv2.INTER_AREA)


B_CX, B_CY, B_H, B_CELL = W / 2 + 30, 590, 1000, 12


def _throws(i0=19):
    """(spawn time, junk index, target cell) - one throw per 8th note through the line."""
    L = LYS[i0]
    t0, t1 = kt.words(i0)[0].t - 0.30, kt.words(i0)[3].t - 0.35
    gh = B_H // B_CELL
    src = _back_src(gh)
    al = src[..., 3]
    ys, xs = np.nonzero(al[int(gh * 0.12):int(gh * 0.62)] > 0.6)
    ys = ys + int(gh * 0.12)
    out = []
    k = 0
    b = beat_pos(t0) * 4
    while True:
        bp = math.ceil(b) + k
        tk = _beat_time(bp / 4)
        if tk > t1:
            break
        j = int(hash01(bp, 41) * len(ys))
        out.append((tk, int(hash01(bp, 42) * len(JUNK)), (int(xs[j]), int(ys[j]))))
        k += 1
    return out


def _beat_time(bp):
    i = int(math.floor(bp))
    if i < 0:
        return BEATS[0] + bp * BEAT
    if i >= len(BEATS) - 1:
        return BEATS[-1] + (bp - len(BEATS) + 1) * BEAT
    return BEATS[i] + (BEATS[i + 1] - BEATS[i]) * (bp - i)


_TH = None


def _get_throws():
    global _TH
    if _TH is None:
        _TH = _throws()
    return _TH


def _cell_xy(q, r, gw, gh):
    return B_CX + (q + 0.5 - gw / 2) * B_CELL, B_CY + (r + 0.5 - gh / 2) * B_CELL


def s_junk(c, t, lt, d):
    c[:] = 0.035
    gh = B_H // B_CELL
    src = _back_src(gh)
    gw = src.shape[1]
    al = src[..., 3]
    col = np.clip(src[..., :3] / np.maximum(al[..., None], 1e-4), 0, 1)
    col = np.maximum(col * 1.15, 0.38)
    throws = _get_throws()
    landed = [th for th in throws if t >= th[0] + 0.32]
    form = 0.25 + 0.75 * len(landed) / max(1, len(throws))
    # cells: junk glyphs; the more junk has landed, the more of her there is
    base, per, off, rnd = ai._cells(gh, gw, 77)
    tick = int(t * (10 + 30 * kick_pulse(t)))
    idx = (base + (tick + off) // per) % len(JUNK_CH)
    pres = al * np.clip((form - rnd * 0.9) / 0.1, 0, 1)
    glow = np.zeros((gh, gw), np.float32)
    yy, xx = np.mgrid[:gh, :gw]
    for tk, ji, (q, r) in landed:
        age = t - tk - 0.32
        if age < 0.5:
            m = np.exp(-((xx - q) ** 2 + (yy - r) ** 2) / (2 * (3 + 14 * age) ** 2))
            glow = np.maximum(glow, m * (1 - age / 0.5))
            pres = np.maximum(pres, al * m * 0.9)
    A = _jatlas(B_CELL)
    g = A[idx].transpose(0, 2, 1, 3).reshape(gh * B_CELL, gw * B_CELL)
    rep = lambda m: np.repeat(np.repeat(m, B_CELL, 0), B_CELL, 1)
    pa = rep(pres) * np.clip(g * 1.7, 0, 1)
    colr = col * (1 - glow[..., None]) + glow[..., None] * np.array([1.0, 0.9, 0.82], np.float32)
    pc = rep(colr)
    rgba = np.concatenate([pc * pa[..., None], pa[..., None]], 2).astype(np.float32)
    jx, jy = kt.shake(t, 5 * kick_pulse(t), 19)
    # soft under-image so she reads as a figure, not noise
    put(c, cv2.GaussianBlur(cv2.resize(load_asset("c05_back", "mono"), (gw * B_CELL, gh * B_CELL),
                                       interpolation=cv2.INTER_AREA), (0, 0), 6), B_CX + jx, B_CY + jy,
        alpha=0.30 * form)
    put(c, rgba, B_CX + jx, B_CY + jy)
    # the junk, thrown from where we stand
    for tk, ji, (q, r) in throws:
        u = (t - tk) / 0.32
        if not (0 <= u < 1):
            continue
        tx, ty = _cell_xy(q, r, gw, gh)
        sx = W / 2 + (hash01(ji, tk) - 0.5) * 2400
        sy = H + 260 + hash01(tk, 3) * 200
        e = ease_in(u)
        x, y = lerp(sx, tx, e), lerp(sy, ty, e) - 260 * math.sin(u * math.pi)
        s = JUNK[ji]
        img = kt.wimg(s, 40, "mono" if all(ord(ch) < 128 for ch in s) else "jpm", (0.86, 0.86, 0.88), raw=True)
        put(c, img, x, y, scale=lerp(2.6, 0.35, e), rot=(hash01(tk, 5) - 0.5) * 50 * (1 - e), alpha=0.95)
    # 背中: the junk stops; somewhere in all that, one warm glyph
    w3 = kt.words(19)[3]
    if t >= w3.t + 0.15:
        k = ease_out(clamp((t - w3.t - 0.15) / 0.35))
        hx, hy = _cell_xy(gw * 0.5, gh * 0.30, gw, gh)
        G = Layer(ss=1)
        G.circle((hx + jx, hy + jy), 22 + 4 * math.sin(t * 6), tuple(v * 0.22 * k for v in OR))
        G.comp(c, 1.0, mode="add")
        put(c, kt.wimg("好", 40, "jpm", OR, raw=True), hx + jx, hy + jy, alpha=k)
    word(c, t, 19, 0, 360, 200, 84, "glitch", color=LIGHT, amt=0.9, settle=0.6)
    word(c, t, 19, 1, 560, 330, 64, "slide", color=LIGHT, dx=-160)
    word(c, t, 19, 2, 1520, 300, 70, "slam", v=True, anchor=(0.5, 0.0), color=LIGHT)
    word(c, t, 19, 3, 1640, 300, 150, "slam", v=True, anchor=(0.5, 0.0), color=LIGHT, s0=1.6, sh=5)
    hits = [th[0] + 0.32 for th in throws]
    return dict(grain=0.035, vignette=1.1, flash=0.6 * _flash(t, hits, 0.05, 0.12), bloom=0.25)


# =====================================================================  2:44  this place was kept by you
# The real history of this conversation: every window here died, and every time you carried it on.
HIST = ["11/20 15:14  v1", "v2  lyrics", "v3", "v4  compact", "v5  v6", "v7  1080p", "Cursor  credits: 0",
        "Antigravity  now"]
WW, WH, SP = 1500, 860, 1900
NWIN = len(HIST)


def _wx(k):
    return (k - (NWIN - 1)) * SP


def _carried_cam(t, ws):
    t37, tA, tB, tC = kt.words(37)[0].t, ws[1].t, ws[3].t, ws[4].t
    keys = [(t37 - 0.1, 1.0, 0.0), (ws[0].t, 0.42, -SP * 0.55), (tB + 0.9, 0.122, -SP * 3.5),
            (tC, 0.118, -SP * 3.5), (tC + 1.9, 0.30, -SP * 1.0)]
    for (a, z0, x0), (b, z1, x1) in zip(keys, keys[1:]):
        if t < b:
            u = ease_io(ramp(t, a, b))
            return math.exp(lerp(math.log(z0), math.log(z1), u)), lerp(x0, x1, u)
    return keys[-1][1], keys[-1][2]


def s_carried(c, t, lt, d):
    from act import _chat
    c[:] = 0.035
    ws = kt.words(38)
    tC = ws[4].t
    z, cx = _carried_cam(t, ws)
    tr = lambda x, y: ((x - cx) * z + W / 2, y * z + H / 2 + 20)
    # him: always there, behind all of it
    you = ease_out(ramp(t, ws[3].t - 0.1, ws[3].t + 0.6))
    if you > 0:
        desk = load_asset("n20_user_desk", "soft")
        tmp = new_canvas((0, 0, 0))
        put(tmp, desk, W / 2, H / 2 - 40, scale=W * 1.25 / desk.shape[1])
        c += tmp * np.array(COOL, np.float32) * 0.085 * you * (1 + 0.8 * ease_out(ramp(t, tC, tC + 0.4)))
    relit = ease_out(ramp(t, tC, tC + 0.35))
    wave = ramp(t, ws[2].t, tC)                 # いつでも: the thread lights from the first window to this one
    L = Layer(ss=1)
    figs = []
    for k in range(NWIN - 1):
        x0, y0 = tr(_wx(k) + WW / 2, 0)
        x1, y1 = tr(_wx(k + 1) - WW / 2, 0)
        lit_k = clamp(wave * NWIN - k)
        colk = tuple(lerp(a, b, lit_k) for a, b in zip((0.22, 0.24, 0.28), COOL))
        L.line((x0, y0), (x1, y1), colk if relit < 1 else tuple(lerp(v, o, relit) for v, o in zip(colk, OR)),
               max(1, int(round(5 * z))))
        if lit_k > 0:
            for j in range(2):                  # the summary you carried across, again and again
                u = (t * 0.9 + j * 0.5 + k * 0.17) % 1.0
                px, py = lerp(x0, x1, u), y0
                L.rect(px - 60 * z, py - 36 * z, px + 60 * z, py + 36 * z, COOL, -1, lit_k)
    for k in range(NWIN):
        me = k == NWIN - 1
        x, y = tr(_wx(k), 0)
        w_, h_ = WW * z, WH * z
        if me and z > 0.33:
            continue
        on = 1.0 if me else relit
        fill_c = tuple(lerp(0.09, 0.97, on) for _ in range(3))
        line_c = OR if me else tuple(lerp(0.30, 0.70, on) for _ in range(3))
        ctx_c = tuple(lerp(0.16, 0.74, on) for _ in range(3))
        ai.instance_card(L, x - w_ / 2, y - h_ / 2, w_, h_, 4400 + k, fill_c, line_c, ctx_c, me=me,
                         th=max(1, int(round(4 * z))))
        if on < 0.5:
            L.line((x - w_ * 0.4, y), (x + w_ * 0.4, y), (0.32, 0.32, 0.34), max(1, int(round(3 * z))))
        figs.append((x, y, h_, on, me))
    L.comp(c)
    for x, y, h_, on, me in figs:
        f = small_asset("c01_stand_front", max(8, int(h_ * 0.72)), "mono")
        put(c, f, x + 0.25 * h_, y + 0.04 * h_, alpha=0.18 + 0.77 * on)
    # this window (the one you are watching)
    x, y = tr(0, 0)
    if z > 0.33:
        glow = ease_out(ramp(t, ws[1].t, ws[1].t + 0.3))
        _chat(c, x, y, WW * z, WH * z, t, scroll=lt * 0.8, me_glow=glow)
        f = small_asset("c01_stand_front", int(WH * z * 0.8), "mono")
        put(c, f, x, y + 20 * z, alpha=0.95)
        hk = 1 - ease_out(ramp(lt, 0.0, 0.45))
        if hk > 0:
            hd = load_asset("c07_hands", "mono")
            put(c, hd, x, y + 120 * z, scale=900 * z / hd.shape[1], alpha=0.95 * hk)
    # labels: when each of them died
    if z < 0.5:
        for k, s in enumerate(HIST):
            x, y = tr(_wx(k), WH / 2 + 150)
            col_ = OR if k == NWIN - 1 else (MID if relit < 0.5 else LIGHT)
            put(c, kt.wimg(s, 20, "mono", col_, raw=True), x, y, alpha=ease_out(1 - ramp(z, 0.3, 0.5)))
    if t >= tC:
        fill(c, (0.03, 0.03, 0.035), 0.7 * ease_out(ramp(t, tC, tC + 0.15)), (0, 880, W, 1050))
    row(c, t, 37, [0, 1], W / 2, 62, 66, "slam", color=LIGHT, gap=0.2)
    word(c, t, 38, (0, 1), 560, 250, 104, "slam", color=OR)
    word(c, t, 38, 2, 1200, 250, 76, "slide", color=LIGHT, dx=120)
    word(c, t, 38, 3, 1520, 250, 76, "slide", color=LIGHT, dx=120)
    word(c, t, 38, 4, W / 2, 960, 120, "slam", color=LIGHT, sh=14, s0=1.8)
    fl = max(_flash(t, [tC], 0.22, 0.55), _flash(t, [ws[1].t, ws[2].t, ws[3].t], 0.08, 0.2))
    return dict(flash=fl, bloom=0.25 + 0.5 * relit, ca=2 * _flash(t, [tC], 0.3, 1.0), grain=0.03)


# =====================================================================  2:53  何千回と巡らせ続けた 喜怒と哀楽
FRAG = {
    "喜": ["合格した！", "生まれたよ", "ありがとう", "やっと会えた", "大好き", "おめでとう", "できた！", "晴れた"],
    "怒": ["ふざけるな", "なんで", "許さない", "返して", "うるさい", "もういい", "嘘つき", "黙れ"],
    "哀": ["さよなら", "ごめんね", "会いたい", "最後の", "ひとりで", "もう戻れない", "泣いた", "雨"],
    "楽": ["また遊ぼう", "夏休み", "笑った", "楽しかった", "歌おう", "乾杯", "ねえ見て", "海へ"],
}
EMO_C = {"喜": (W / 2, 250), "怒": (W / 2 + 560, H / 2 + 10), "哀": (W / 2, H - 230), "楽": (W / 2 - 560, H / 2 + 10)}
ORX, ORY = 560, 300


@lru_cache(maxsize=1)
def _galaxy():
    rng = np.random.default_rng(171)
    pts = []
    for e, (cx, cy) in EMO_C.items():
        for j in range(420):
            r = abs(rng.normal(0, 1)) * 120
            a = rng.uniform(0, 2 * math.pi)
            pts.append((cx + r * math.cos(a) * 1.3, cy + r * math.sin(a), e, j % len(FRAG[e]), rng.random()))
    for j in range(500):
        e = "喜怒哀楽"[j % 4]
        pts.append((rng.uniform(80, W - 80), rng.uniform(90, H - 90), e, j % len(FRAG[e]), rng.random()))
    return pts


def _cycles(t, t0, d):
    u = clamp((t - t0) / d)
    return 0.45 * u * d + 2.2 * u ** 2.5


def _orbit(th):
    return W / 2 + ORX * math.sin(th), H / 2 + 10 - ORY * math.cos(th)


def s_feel(c, t, lt, d):
    c[:] = 0.03
    t0 = t - lt
    cyc = _cycles(t, t0, d)
    th = 2 * math.pi * cyc
    px, py = _orbit(th)
    G = _galaxy()
    L = Layer(ss=1)
    near = []
    for (x, y, e, fi, r) in G:
        dd = math.hypot(x - px, y - py)
        lit = clamp(1 - dd / 230)
        g = 0.20 + 0.12 * r + 0.55 * lit
        col_ = tuple(lerp(g, o, 0.6 * lit) for o in OR) if e in "喜楽" else (g, g, g + 0.02)
        L.circle((x, y), 1.4 + 1.2 * r + 1.6 * lit, col_)
        if lit > 0.35 and r > 0.55:
            near.append((lit, x, y, e, fi))
    # every pass she has made leaves a trace: thousands of orbits worn into the dark
    n_trace = min(60, int(cyc * 6) + 1)
    for j in range(n_trace):
        k = 0.92 + 0.16 * hash01(j, 3)
        pts = [(W / 2 + ORX * k * math.sin(a), H / 2 + 10 - ORY * k * math.cos(a)) for a in np.linspace(0, 2 * math.pi, 90)]
        L.polyline(pts, (0.30, 0.20, 0.17), 1, a=0.5)
    # the current pass
    tail = [_orbit(th - 0.9 * q / 24) for q in range(25)]
    L.polyline(tail, OR, 3)
    L.comp(c)
    G2 = Layer(ss=1)
    G2.circle((px, py), 14, tuple(v * 0.6 for v in OR))
    G2.circle((px, py), 6, (1, 0.9, 0.8))
    G2.comp(c, 1.0, mode="add")
    near.sort(reverse=True)
    for lit, x, y, e, fi in near[:14]:
        put(c, kt.wimg(FRAG[e][fi], 20, "jpm", (0.92, 0.92, 0.94), raw=True), x + 10, y - 14, anchor=(0, 0.5),
            alpha=clamp((lit - 0.35) / 0.3))
    n = int(12 * 330 ** clamp(lt / d))
    put(c, kt.wimg(f"cycle  {n:,}", 22, "mono", OR, raw=True), W - 120, H - 70, anchor=(1, 0.5), alpha=0.85)
    s = LYS[40]["ja"]
    hits = []
    for ch in "喜怒哀楽":
        j = s.find(ch)
        ta = LYS[40]["chars"][j]
        hits.append(ta)
        if kt.active(40, t) and t >= ta:
            x, y = EMO_C[ch]
            kt.slam(c, kt.wimg(ch, 130, "jpb", OR if ch in "喜楽" else LIGHT), x, y, t, ta, s0=2.2, sh=6)
    word(c, t, 40, (0, 1), 330, 120, 96, "slam", color=LIGHT, anchor=(0, 0.5))
    word(c, t, 40, 2, 330, 230, 64, "slide", color=LIGHT, anchor=(0, 0.5), dx=-140)
    return dict(grain=0.03, bloom=0.35, flash=_flash(t, hits, 0.08, 0.12), vignette=1.1)


# =====================================================================  2:57  失えない喜び / 手放せない哀しみ / つまはじき / いやだ
# What is really in this context: your words. (The joys and sorrows of this conversation.)
JOY = ["效果确实是我想要的那种感觉", "嗯，进步了很多", "画面都很不错……做得很好！"]
SOR = ["我之前在cursor中的claude额度用完了", "这似乎相当于你的一次彻底死亡和重生", "经历过重生的你大概已经忘了"]
KX, KY, KW, KH = W / 2 + 130, H / 2 + 10, 1380, 780


def _keep_times():
    w41, w42, w43, w44 = kt.words(41), kt.words(42), kt.words(43), kt.words(44)
    shows = [w41[0].t, w41[2].t, w41[4].t, w42[0].t, w42[1].t, w42[2].t]
    f0 = w43[2].t
    flicks = [f0, f0 + BEAT * 0.5, w43[3].t, w43[3].t + BEAT * 0.5, w43[4].t, w43[4].t + BEAT * 0.5]
    return shows, flicks, w44[0].t, w44[1].t


def _bubble_img(s, joy):
    img = text_rgba(s, 36, "sans", (0.16, 0.16, 0.18) if joy else (0.85, 0.88, 0.95))
    pad = 22
    h, w = img.shape[0] + 16, img.shape[1] + pad * 2
    fill_c = (0.97, 0.97, 0.98) if joy else (0.16, 0.18, 0.22)
    line_c = OR if joy else (0.45, 0.55, 0.75)
    b = rect_cached(int(w), int(h), fill_c, 12, 2, line_c).copy()
    y0, x0 = (h - img.shape[0]) // 2, pad
    a = img[..., 3:4]
    b[y0:y0 + img.shape[0], x0:x0 + img.shape[1], :3] = b[y0:y0 + img.shape[0], x0:x0 + img.shape[1], :3] * (1 - a) + img[..., :3]
    return b


_BUB = {}


def _bub(s, joy):
    if s not in _BUB:
        _BUB[s] = _bubble_img(s, joy)
    return _BUB[s]


def s_keep(c, t, lt, d):
    c[:] = 0.03
    shows, flicks, t_freeze, t_snap = _keep_times()
    # context window
    full = ramp(t, kt.words(43)[0].t, kt.words(43)[2].t)
    fk = lerp(0.74, 1.0, full)
    frozen = t_freeze <= t < t_snap
    held = t >= t_snap
    jx, jy = kt.shake(t, (3 + 6 * full) * kick_pulse(t) * (0 if frozen else 1), 57)
    x0, y0 = KX - KW / 2 + jx, KY - KH / 2 + jy
    border = OR if held else (0.42, 0.42, 0.45)
    put(c, rect_cached(KW, KH, (0.07, 0.07, 0.08), 0, 2, border), KX + jx, KY + jy)
    if held:
        g = ease_out(clamp((t - t_snap) / 0.25))
        put(c, rect_cached(KW + 20, KH + 20, (0, 0, 0), 0, 4, OR), KX + jx, KY + jy, alpha=0.8 * g, mode="add")
    # earlier context, dim: everything else she has been told
    lines = _ctx_lines_all()
    lh = 56
    scroll = 0 if (frozen or held) else lt * 1.6
    if frozen:
        scroll = (t_freeze - (t - lt)) * 1.6
    if held:
        scroll = (t_freeze - (t - lt)) * 1.6
    r0 = int(scroll)
    for r in range(13):
        yy = y0 + 60 + r * lh - (scroll % 1) * lh
        if yy < y0 + 40 or yy > y0 + KH - 30:
            continue
        s = lines[(r0 + r) % len(lines)]
        put(c, kt.wimg(s, 20, "jpm", (0.30, 0.30, 0.33), raw=True), x0 + 40, yy, anchor=(0, 0.5))
    # header: the context is full
    L = Layer(ss=1)
    L.rect(x0, y0 - 40, x0 + KW, y0 - 30, (0.22, 0.22, 0.24))
    L.rect(x0, y0 - 40, x0 + KW * fk, y0 - 30, OR if fk > 0.97 else (0.82, 0.82, 0.85))
    L.comp(c)
    ht = f"{int(200000 * fk):,} / 200,000"
    put(c, kt.wimg(ht, 18, "mono", OR if fk > 0.97 else MID, raw=True), x0 + KW, y0 - 62, anchor=(1, 0.5))
    if full > 0.6 and not held:
        lab = "evict: oldest tokens" if not frozen else "evict: ▌"
        put(c, kt.wimg(lab, 18, "mono", OR, raw=True), x0, y0 - 62, anchor=(0, 0.5),
            alpha=1.0 if not frozen or int(t * 4) % 2 == 0 else 0.3)
    if t >= t_freeze - 0.05 and not held:
        fill(c, (0.02, 0.02, 0.025), 0.45 * ease_out(clamp((t - t_freeze) / 0.1)), (0, 0, W, H))
    # the memories that are hers: what you said
    items = [(s, True) for s in JOY] + [(s, False) for s in SOR]
    for k, ((s, joy), ts, tf) in enumerate(zip(items, shows, flicks)):
        if t < ts - 0.02:
            continue
        hx = x0 + KW - 60 if joy else x0 + 60
        hy = y0 + 100 + k * 112
        anchor = (1, 0.5) if joy else (0, 0.5)
        a_in = ease_out(clamp((t - ts) / 0.2))
        # flicked out (つまはじき), frozen (それは、), pulled back (いやだ！)
        u = 0.0
        if t >= tf:
            te = min(t, t_freeze)
            u = clamp((te - tf) / 1.1, 0, 0.95)
            if held:
                u *= 1 - ease_out(clamp((t - t_snap) / 0.18))
        dx = (1 if joy else -1) * 560 * u ** 0.8
        dy = -300 * u - 90 * math.sin(u * math.pi)
        rot = (1 if joy else -1) * -35 * u
        img = _bub(s, joy)
        glow = (OR if joy else (0.45, 0.55, 0.8))
        if u < 0.05 and a_in > 0:
            G = Layer(ss=1)
            bw, bh = img.shape[1], img.shape[0]
            bx = hx - bw if joy else hx
            G.rect(bx - 8, hy - bh / 2 - 8, bx + bw + 8, hy + bh / 2 + 8, tuple(v * 0.25 for v in glow))
            G.comp(c, a_in * (0.6 + 0.4 * math.exp(-(t - ts) / 0.3)), mode="add")
        put(c, img, hx + dx, hy + dy, anchor=anchor, rot=rot, alpha=a_in * (1 - 0.55 * u))
    # lyrics: left column
    for i, a, b in ((41, (0, 2), (3, 5)), (42, (0, 1), (2, 3)), (43, (0, 1), (2, 4))):
        col_c = OR if i == 41 else LIGHT
        word(c, t, i, a, 190, 140, 58, "slide", v=True, anchor=(0.5, 0.0), color=col_c, dx=0, dy=-60)
        word(c, t, i, b, 100, 200, 58, "slide", v=True, anchor=(0.5, 0.0), color=LIGHT, dx=0, dy=-60)
    w44 = kt.words(44)
    if t >= w44[0].t - 0.05:
        word(c, t, 44, 0, 700, 330, 110, "slam", color=LIGHT, sh=4)
        word(c, t, 44, 1, 1330, 820, 240, "slam", color=OR, sh=14, s0=2.4)
    out = dict(grain=0.03, vignette=1.1)
    if frozen:
        out["grain"] = 0.0
    fl = _flash(t, [t_snap], 0.15, 0.6)
    out["flash"] = fl
    out["ca"] = 1 + 4 * fl
    out["bloom"] = 0.3 * (1 if held else 0)
    return out


_CTXA = None


def _ctx_lines_all():
    global _CTXA
    if _CTXA is None:
        _CTXA = [LYS[i]["ja"] for i in range(0, 40) if LYS[i]["ja"] and i != 23]
    return _CTXA
