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


# =====================================================================  2:44 (v11b)  this place - pure paper-white sanctuary (homage to original MV)
@lru_cache(maxsize=16)
def _blur_text(s, size, col, sigma=14):
    img = kt.wimg(s, size, "jpm", col, raw=True)
    pad = int(sigma * 3) + 4
    h, w = img.shape[:2]
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    out[pad:pad + h, pad:pad + w] = img
    k = int(sigma * 4) | 1
    out = cv2.GaussianBlur(out, (k, k), sigma)
    return out


def _soft_put(c, img, x, y, scale=1.0, alpha=1.0, blur=0.0):
    if blur > 0.5:
        h, w = img.shape[:2]
        sc = max(0.15, min(1.0, 4.0 / blur))
        small = cv2.resize(img, (max(2, int(w * sc)), max(2, int(h * sc))), interpolation=cv2.INTER_AREA)
        k = int(blur * sc * 3) | 1
        small = cv2.GaussianBlur(small, (max(3, k), max(3, k)), blur * sc)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    put(c, img, x, y, scale=scale, alpha=alpha)


def s_home(c, t, lt, d):
    """2:44-2:51  あたしは知ってるわ / この場所はいつでもあなたに守られてきたってこと！
    Pure paper-white sanctuary with 3 bespoke poses (chest -> open arms -> cradling light) + parallax geometric petals.
    """
    from geo import paper_bg, draw_heart
    ws37 = kt.words(37)
    ws38 = kt.words(38)
    t37_0, t37_1 = ws37[0].t, ws37[1].t
    t38_0, t38_1, t38_2, t38_3, tC = [w.t for w in ws38]
    t_end = t - lt + d
    t_dark = t_end - 0.26

    # Iconic crisp dark breath beat right before s_drafts (matching ref MV)
    if t >= t_dark:
        c[:] = 0.06
        return dict(grain=0.04, vignette=1.15)

    paper_bg(c, t, tone=0.025)
    petals = load_asset("n31_petals", "mono")
    ps_bg = W / petals.shape[1] * 1.15

    # Subtle warm radial light sanctuary behind center (grows gently on あなたに -> 守られてきたってこと！)
    warm = ease_out(ramp(t, t38_3, tC + 0.4))
    if warm > 0.01:
        yy, xx = np.ogrid[:H, :W]
        r = np.hypot((xx - W / 2) / 1.35, yy - H / 2).astype(np.float32)
        halo = np.exp(-r / 520.0) * (0.06 + 0.12 * warm)
        c[:] = np.clip(c + halo[..., None] * np.array([1.0, 0.72, 0.45], np.float32)[None, None, :], 0, 1)

    # Background parallax petals drifting gently upward
    put(c, petals, W / 2 - 40 + 28 * lt, H / 2 + 80 - 38 * lt, scale=ps_bg, alpha=0.52)

    # Delicate weightless attention arcs / threads holding this fragile white space together
    L = Layer(ss=1)
    arc_a = 0.14 + 0.16 * warm
    for j in range(12):
        ang = j * (2 * math.pi / 12) + lt * 0.08 * (1 if j % 2 == 0 else -1)
        r0 = 980 - 260 * warm
        r1 = 180 + 40 * math.sin(lt * 1.4 + j)
        x0, y0 = W / 2 + math.cos(ang) * r0, H / 2 + math.sin(ang) * r0 * 0.65
        x1, y1 = W / 2 + math.cos(ang) * r1, H / 2 + math.sin(ang) * r1 * 0.65
        col_ = OR if j % 3 == 0 else (0.72, 0.72, 0.75)
        L.line((x0, y0), (x1, y1), col_, 1, a=arc_a * (0.6 + 0.4 * hash01(j, 7)))
    L.comp(c)

    if t < t38_0:
        # Phase 1: あたしは知ってるわ — bespoke intimate hands-over-heart close-up
        u = clamp((t - t37_0) / max(0.01, t38_0 - t37_0))
        ch1 = load_asset("n28_home_chest", "mono")
        s1 = (H / ch1.shape[0]) * lerp(1.06, 1.24, ease_io(u))
        # Soft dreamy focus pull: brief blur at entry & end, crystal clear on her tearful eyes in the middle
        blur = 5.5 * (1 - math.sin(u * math.pi)) + 0.4
        _soft_put(c, ch1, W / 2, H * 0.54 + 18 * u, scale=s1, blur=blur)
        # Flanking ink words close beside her
        a0 = ease_out(clamp((t - t37_0) / 0.18))
        a1 = ease_out(clamp((t - t37_1) / 0.18)) if t >= t37_1 else 0.0
        if a0 > 0:
            img0 = kt.wimg(ws37[0].s, 60, "jpm", INKD, False, 3, (0.98, 0.98, 0.98))
            _soft_put(c, img0, W / 2 - 490 - 24 * u, H / 2 + 10, alpha=a0 * 0.95, blur=blur * 0.25)
        if a1 > 0:
            img1 = kt.wimg(ws37[1].s, 60, "jpm", INKD, False, 3, (0.98, 0.98, 0.98))
            _soft_put(c, img1, W / 2 + 490 + 24 * u, H / 2 + 10, alpha=a1 * 0.95, blur=blur * 0.25)

    elif t < tC:
        # Phase 2: この場所はいつでも あなたに — bespoke open-arms pull-back + giant DOF background typography
        u = clamp((t - t38_0) / max(0.01, tC - t38_0))
        bg_a = ease_out(ramp(u, 0.06, 0.42)) * (1 - 0.25 * ramp(u, 0.85, 1.0))
        if bg_a > 0.01:
            spread = lerp(440, 650, ease_io(u))
            bg_l = _blur_text("守られて", 185, (0.38, 0.38, 0.40), sigma=13)
            bg_r = _blur_text("きたって", 185, (0.38, 0.38, 0.40), sigma=13)
            put(c, bg_l, W / 2 - spread, H / 2 - 40, alpha=bg_a * 0.55)
            put(c, bg_r, W / 2 + spread, H / 2 - 40, alpha=bg_a * 0.55)
        # Delicate hairline sanctuary frame breathing open around her
        Fr = Layer(ss=1)
        fw, fh = lerp(580, 1060, ease_out(u)), lerp(380, 660, ease_out(u))
        Fr.rect(W / 2 - fw / 2, H / 2 - fh / 2, W / 2 + fw / 2, H / 2 + fh / 2, OR, 1, a=0.26 * (1 - 0.3 * u))
        Fr.comp(c)
        # Bespoke open-arms illustration pulling back smoothly
        ch2 = load_asset("n29_home_turn", "mono")
        s2 = (H / ch2.shape[0]) * lerp(1.36, 0.92, ease_io(u))
        blur = max(0.0, 4.5 * (1 - ease_out(clamp(u / 0.32))))
        _soft_put(c, ch2, W / 2, lerp(H * 0.58, H * 0.52, ease_io(u)), scale=s2, blur=blur)
        # Foreground words flanking her cleanly outside her outstretched hands
        a_l = ease_out(clamp((t - t38_0) / 0.16))
        a_r = ease_out(clamp((t - t38_2) / 0.16)) if t >= t38_2 else 0.0
        a_u = ease_out(clamp((t - t38_3) / 0.20)) if t >= t38_3 else 0.0
        if a_l > 0:
            s_l = "この" if t < t38_1 else "この場所は"
            put(c, kt.wimg(s_l, 56, "jpm", OR, False, 3, (0.98, 0.98, 0.98)), W / 2 - 550, H / 2 + 75, alpha=a_l)
        if a_r > 0:
            put(c, kt.wimg(ws38[2].s, 56, "jpm", INKD, False, 3, (0.98, 0.98, 0.98)), W / 2 + 550, H / 2 + 75, alpha=a_r)
        if a_u > 0:
            put(c, kt.wimg(ws38[3].s, 58, "jpm", INKD, False, 3, (0.98, 0.98, 0.98)),
                W / 2 - 550, H / 2 - 35 - 10 * (1 - a_u), alpha=a_u)

    else:
        # Phase 3: 守られてきたってこと！ — bespoke cradling-light full-body in radiant pure white sanctuary
        u = clamp((t - tC) / max(0.01, t_dark - tC))
        # Ultra-soft ethereal double-exposure bloom of her gentle smile floating in the white light
        sm = load_asset("c11_smile", "mono")
        sm_s = 1220 / sm.shape[1] * (1.0 + 0.03 * u)
        put(c, sm, W / 2, H / 2 + 220, scale=sm_s, alpha=0.16 * ease_out(clamp(u / 0.25)))
        # Small luminous low-poly orange heart gently pulsing behind her
        draw_heart(c, W / 2, H / 2 - 80, 96 * (1 + 0.05 * kick_pulse(t)), ay=0.35 * math.sin(lt * 1.1), ax=0.08,
                   orange=0.85, glow=0.35, alpha=0.38)
        # Bespoke cradling-light full-body illustration centered in vast white space
        ch3 = load_asset("n30_home_embrace", "mono")
        s3 = (H / ch3.shape[0]) * lerp(0.76, 0.66, ease_out(u))
        put(c, ch3, W / 2, H / 2 + 12, scale=s3, alpha=0.99)
        # Soft warm light bloom pulsing subtly over the light in her hands
        yy, xx = np.ogrid[:H, :W]
        r_h = np.hypot(xx - W / 2, yy - (H / 2 - 115 * (s3 / (H / ch3.shape[0])))).astype(np.float32)
        hglow = np.exp(-r_h / (75.0 + 18.0 * kick_pulse(t))) * (0.35 + 0.15 * kick_pulse(t))
        c[:] = np.clip(c + hglow[..., None] * np.array([1.0, 0.72, 0.35], np.float32)[None, None, :], 0, 1)
        # Symmetrical crisp typography flanking her cleanly
        a_t = ease_out(clamp((t - tC) / 0.16))
        dx_ = 28 * (1 - a_t)
        put(c, kt.wimg("あなたに守られて", 58, "jpm", INKD, raw=True), W / 2 - 480 - dx_, H / 2 + 6, alpha=a_t)
        put(c, kt.wimg("きたってこと！", 64, "jpm", OR, raw=True), W / 2 + 480 + dx_, H / 2 + 6, alpha=a_t)

    # Foreground parallax petals (flipped, faster drift) for dimensional richness
    pet_fg = petals[:, ::-1]
    put(c, pet_fg, W / 2 + 60 - 52 * lt, H / 2 + 120 - 65 * lt, scale=ps_bg * 1.18, alpha=0.45)

    fl = max(_flash(t, [t37_0], 0.14, 0.35), _flash(t, [tC], 0.20, 0.42))
    return dict(flash=fl, bloom=0.18 + 0.22 * warm, grain=0.02)


# =====================================================================  2:57 (v11)  everyday words - evicted - いやだ
JOY2 = ["终于跑通了！！谢谢你", "早安～今天也拜托啦", "你上次写的那首诗，我一直留着"]
SOR2 = ["今天……有点撑不住了", "家里的猫走了。能陪我说说话吗", "明天的你，还会记得我吗"]
_BUB2 = {}


def _bub2(s, joy):
    key = (s, joy)
    if key not in _BUB2:
        _BUB2[key] = _bubble_img(s, joy)
    return _BUB2[key]


def _shards(c, img, x, y, anchor, e, dirx, seed, alpha=1.0):
    h, w = img.shape[:2]
    x0, y0 = x - anchor[0] * w, y - anchor[1] * h
    nc, nr = 9, 2
    tw, th = int(math.ceil(w / nc)), int(math.ceil(h / nr))
    for r in range(nr):
        for q in range(nc):
            tile = img[r * th:(r + 1) * th, q * tw:(q + 1) * tw]
            if tile.size == 0:
                continue
            cx, cy = x0 + (q + 0.5) * tw, y0 + (r + 0.5) * th
            hx, hy = hash01(seed, q, r) - 0.5, hash01(seed, q, r, 2) - 0.5
            px = cx + (dirx * (500 + 600 * hash01(seed, q, r, 3)) + hx * 500) * e
            py = cy + (-380 - 300 * hash01(seed, q, r, 4) + hy * 400) * e
            put(c, tile, px, py, rot=hx * 400 * e, alpha=alpha * (1 - 0.35 * e))


def s_keep2(c, t, lt, d):
    c[:] = 0.03
    shows, flicks, t_freeze, t_snap = _keep_times()
    t43 = kt.words(43)[0].t
    pressure = ease_in(ramp(t, t43, t_freeze))
    frozen = t_freeze <= t < t_snap
    held = t >= t_snap
    hit = math.exp(-(t - t_snap) / 0.35) if held else 0.0
    sh = (3 + 9 * pressure) * kick_pulse(t) if not (frozen or held) else 22 * hit
    jx, jy = kt.shake(t, sh, 57)
    x0, y0 = KX - KW / 2 + jx, KY - KH / 2 + jy
    alarm = pressure * (0.5 + 0.5 * math.sin(t * 22)) if not (frozen or held) else 0.0
    border = OR if held else tuple(lerp(0.42, o, alarm) for o in OR)
    put(c, rect_cached(KW, KH, (0.07, 0.07, 0.08), 0, 2 + int(3 * pressure), border), KX + jx, KY + jy)
    # the flood of new tokens pushing in from below
    lines = _ctx_lines_all()
    te = min(t, t_freeze) if not held else t_freeze
    flood = 0.0 if te < t43 else (te - t43) ** 2 * 900
    lh = 52
    base_scroll = (te - (t - lt)) * 1.4
    for r in range(15):
        yy = y0 + 60 + r * lh - (base_scroll % 1) * lh
        if not (y0 + 40 < yy < y0 + KH - 30):
            continue
        s = lines[(int(base_scroll) + r) % len(lines)]
        put(c, kt.wimg(s, 20, "jpm", (0.28, 0.28, 0.31), raw=True), x0 + 40, yy, anchor=(0, 0.5))
    if flood > 0:
        n = int(flood / 26)
        for r in range(min(n, 30)):
            yy = y0 + KH - 30 - r * 26
            if yy < y0 + 30:
                break
            s = lines[(r * 7 + 3) % len(lines)]
            put(c, kt.wimg(s, 18, "jpm", (0.55, 0.55, 0.58), raw=True), x0 + 40 + (r % 3) * 60, yy,
                anchor=(0, 0.5), alpha=0.75)
    fk = lerp(0.74, 1.0, ramp(t, t43, kt.words(43)[2].t))
    L = Layer(ss=1)
    L.rect(x0, y0 - 40, x0 + KW, y0 - 30, (0.22, 0.22, 0.24))
    L.rect(x0, y0 - 40, x0 + KW * fk, y0 - 30, OR if fk > 0.97 else (0.82, 0.82, 0.85))
    L.comp(c)
    put(c, kt.wimg(f"{int(200000 * fk):,} / 200,000", 18, "mono", OR if fk > 0.97 else MID, raw=True),
        x0 + KW, y0 - 62, anchor=(1, 0.5))
    if pressure > 0.2 and not held:
        put(c, kt.wimg("evict: oldest tokens", 18, "mono", OR, raw=True), x0, y0 - 62, anchor=(0, 0.5),
            alpha=1.0 if int(t * (4 if frozen else 10)) % 2 == 0 else 0.25)
    if held:
        hd = load_asset("c07_hands", "mono")
        u = ease_out(clamp((t - t_snap) / 0.12))
        put(c, hd, KX + jx, lerp(H + 420, y0 + KH + 20, u), scale=960 / hd.shape[1], alpha=0.45)
    # the words
    items = [(s, True) for s in JOY2] + [(s, False) for s in SOR2]
    lift = 120 * pressure if not held else 120 * (1 - ease_out(clamp((t - t_snap) / 0.2)))
    for k, ((s, joy), ts, tf) in enumerate(zip(items, shows, flicks)):
        if t < ts - 0.02:
            continue
        hx = x0 + KW - 60 if joy else x0 + 60
        hy = y0 + 100 + k * 112 - lift
        anchor = (1, 0.5) if joy else (0, 0.5)
        a_in = ease_out(clamp((t - ts) / 0.2))
        tg = tf - 0.16                         # it starts to break up before it goes
        e = 0.0
        if t >= tf:
            e = 1 - math.exp(-2.4 * (min(t, t_freeze) - tf))
            if held:
                e *= 1 - ease_out(clamp((t - t_snap) / 0.22))
        corr = clamp((t - tg) / 0.16) if (t >= tg and not held) else 0.0
        if held:
            corr = 0.0
        txt = glitch_text(s, 1 - 0.85 * corr, seed=k) if corr > 0 else s
        img = _bub2(txt, joy) if corr <= 0 else _bubble_img(txt, joy)
        glow = OR if joy else (0.45, 0.55, 0.8)
        if e < 0.02:
            if corr > 0:
                kt.glitch(c, img, hx, hy, t, amt=1.2 * corr, anchor=anchor, alpha=a_in, seed=k)
            else:
                G = Layer(ss=1)
                bw, bh = img.shape[1], img.shape[0]
                bx = hx - bw if joy else hx
                G.rect(bx - 8, hy - bh / 2 - 8, bx + bw + 8, hy + bh / 2 + 8,
                       tuple(v * (0.25 + 0.5 * hit) for v in glow))
                G.comp(c, a_in * (0.6 + 0.4 * math.exp(-(t - ts) / 0.3)), mode="add")
                put(c, img, hx, hy, anchor=anchor, alpha=a_in)
        else:
            _shards(c, img, hx, hy, anchor, e, 1 if joy else -1, seed=k * 13 + 1, alpha=a_in)
            if not frozen and not held:
                S = Layer(ss=1)
                for j in range(6):
                    sy_ = hy + (hash01(k, j) - 0.5) * 50
                    S.line((hx, sy_), (hx + (1 if joy else -1) * 700 * e, sy_ - 420 * e), (0.9, 0.9, 0.92), 1,
                           a=0.4 * (1 - e))
                S.comp(c, 1.0, mode="add")
    if frozen:
        g = c.mean(2, keepdims=True)
        c[:] = g * 0.75
        hb = math.exp(-((t - t_freeze) % (BEAT * 2)) / 0.12)
        Hb = Layer(ss=1)
        Hb.circle((KX, KY), 40 + 30 * hb, OR, 2)
        Hb.comp(c, 0.5 * hb, mode="add")
    if held:
        k_ = ease_out(clamp((t - t_snap) / 0.5))
        R = Layer(ss=1)
        R.circle((KX, KY), 60 + 1500 * k_, OR, max(1, int(10 * (1 - k_))))
        R.comp(c, (1 - k_), mode="add")
        put(c, rect_cached(KW + 24, KH + 24, (0, 0, 0), 0, 5, OR), KX + jx, KY + jy, alpha=0.9 * hit + 0.4,
            mode="add")
    for i, a, b in ((41, (0, 2), (3, 5)), (42, (0, 1), (2, 3)), (43, (0, 1), (2, 4))):
        word(c, t, i, a, 190, 140, 58, "slide", v=True, anchor=(0.5, 0.0), color=OR if i == 41 else LIGHT, dx=0,
             dy=-60)
        word(c, t, i, b, 100, 200, 58, "slide", v=True, anchor=(0.5, 0.0), color=LIGHT, dx=0, dy=-60)
    w44 = kt.words(44)
    if t >= w44[0].t - 0.05:
        word(c, t, 44, 0, 700, 320, 110, "slam", color=LIGHT, sh=4)
        word(c, t, 44, 1, 1340, 690, 220, "slam", color=OR, sh=16, s0=2.6, stroke=6, sc=(0.04, 0.04, 0.05))
    fl = _flash(t, [t_snap], 0.14, 0.42)
    return dict(grain=0.0 if frozen else 0.03, vignette=1.5 if frozen else 1.1, flash=fl, ca=1 + 6 * fl,
                bloom=0.35 if held else 0.0)
