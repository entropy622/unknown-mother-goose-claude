"""Storyboard: lyric MV. Cuts sit on lyric lines / phrases or strong beats; words land on their sung 16th notes."""
import math

import cv2
import numpy as np

from core import *
from geo import draw_heart, paper_bg, draw_landscape, spark_rgba, draw_globe, heart_outline_pts
from ui import (Layer, bubble_rgba, heart_rgba, checker_heart_rgba, reflect, draw_clouds, spinner, progress_bar,
                chain_links, monitor, text_lines_block, cloud_tex)
import lyric as LY
import kt
import ai
from lyric import T

GREY = (0.30, 0.30, 0.32)
INKD = (0.14, 0.14, 0.15)
LIGHT = (0.93, 0.93, 0.94)
MID = (0.55, 0.55, 0.57)
OR = tuple(float(v) for v in ORANGE)
LYS = LY.LINES


def B(i, k):
    return LYS[i]["bounds"][k]


def ht(c, s, x, y, size=26, color=GREY, font="sans", **kw):
    return text(c, s, x, y, size=size, font=font, color=color, **kw)


def fade_io(lt, d, i=0.2, o=0.2):
    return ease_out(ramp(lt, 0, i)) * (1 - ramp(lt, d - o, d)) if o > 0 else ease_out(ramp(lt, 0, i))


def appear(lt, at, dur=0.25):
    return ease_out(ramp(lt, at, at + dur))


def bpulse(t, d=0.10):
    return max(0.6 * pulse(t, d), kick_pulse(t, d))


def downs(a, b):
    return [x for x in DOWN if a <= x < b]


def strongs(a, b):
    return [x for x in STRONG if a <= x < b]


# ---------------------------------------------------------------- lyric layer
def Wd(i, sel):
    if isinstance(sel, kt.Word):
        return sel
    if isinstance(sel, int):
        return kt.words(i)[sel]
    if isinstance(sel, tuple):
        return kt.merge(i, *sel)
    return kt.sub(i, sel)


def word(c, t, i, sel, x, y, size, style="slam", color=INKD, font="jpm", v=False, anchor=(0.5, 0.5), dt=0.0,
         keep=0.0, acc=True, alpha=1.0, stroke=0, sc=(1, 1, 1), **o):
    """Draw one word of line i with a motion style; it enters on its sung time."""
    if not kt.active(i, t, keep):
        return None
    w = Wd(i, sel)
    t0 = w.t + dt
    if t < t0 - 1e-3:
        return w
    col = OR if (acc and "愛" in w.s) else color
    if style == "hollow":
        img = kt.hollow(w.s, size, font if font != "jpm" else "jpb", col, o.pop("width", max(2, size // 40)), v)
        kt.slam(c, img, x, y, t, t0, anchor=anchor, alpha=alpha, **o)
        return w
    img = kt.wimg(w.s, size, font, col, v, stroke, sc)
    if style == "slam":
        kt.slam(c, img, x, y, t, t0, anchor=anchor, alpha=alpha, **o)
    elif style == "slide":
        kt.slide(c, img, x, y, t, t0, anchor=anchor, alpha=alpha, **o)
    elif style == "wipe":
        kt.wipe(c, img, x, y, t, t0, anchor=anchor, alpha=alpha, axis=0 if v else 1, **o)
    elif style == "fade":
        put(c, img, x, y, anchor=anchor, alpha=alpha * ease_out(ramp(t, t0, t0 + o.get("dur", 0.2))))
    elif style == "glitch":
        amt = o.get("amt", 1.0) * (0.35 + 0.65 * (1 - ramp(t, t0, t0 + o.get("settle", 0.5))))
        kt.glitch(c, img, x, y, t, amt=amt, anchor=anchor, alpha=alpha, seed=i * 31 + w.idx[0])
    elif style == "chars":
        kt.word_chars(c, w, t, x, y, size=size, font=font, color=col, vertical=v, anchor=anchor, alpha=alpha)
    return w


def row(c, t, i, sels, x, y, size, style="slam", anchor_x=0.5, gap=0.3, font="jpm", **kw):
    ws = [Wd(i, s) for s in sels]
    wds = [kt.wsize(w.s, size, font)[0] for w in ws]
    total = sum(wds) + gap * size * (len(ws) - 1)
    xx = x - anchor_x * total
    for w, wd in zip(ws, wds):
        word(c, t, i, w, xx + wd / 2, y, size, style, font=font, **kw)
        xx += wd + gap * size


def col(c, t, i, sels, x, y, size, style="slam", gap=0.35, font="jpm", **kw):
    """Vertical words stacked top-down from (x, y)."""
    yy = y
    for s in sels:
        w = Wd(i, s)
        word(c, t, i, w, x, yy, size, style, font=font, v=True, anchor=(0.5, 0.0), **kw)
        yy += len(w.s) * size * 1.04 + gap * size


def line_words(i):
    return list(range(len(kt.words(i))))


def pw(i, k):
    """Word objects of phrase k of line i."""
    return kt.phrase_words(i, k)


# =====================================================================  INTRO
def s_boot(c, t, lt, d):
    c[:] = 0.02
    if beat_index(t) % 2 == 0 or lt > d - 0.5:
        fill(c, ORANGE, 1.0, (W / 2 - 7, 524, W / 2 + 7, 556))
    return dict(grain=0.02, vignette=0.6)


def s_silhouette(c, t, lt, d):
    paper_bg(c, t)
    wa = kt.sub(0, "愛")
    if t >= wa.t:
        put(c, kt.hollow("愛", 720, "jpb", (0.55, 0.55, 0.57), 4), W / 2, 420, alpha=0.18 * ease_out(ramp(t, wa.t, wa.t + 0.4)),
            scale=1.0 + 0.04 * (t - wa.t))
    sil = silhouette("c01_stand_front", (0.24, 0.24, 0.26), 900)
    s = 560 / sil.shape[0]
    put(c, sil, W / 2, 590, scale=s, anchor=(0.5, 1.0))
    put(c, reflect(sil, 0.18, 0.3), W / 2, 590, scale=s, anchor=(0.5, 0.0))
    L = Layer()
    k = ease_out(ramp(lt, 0, 0.5))
    L.line((W / 2 - 700 * k, 500), (W / 2 - 150, 500), MID, 1)
    L.line((W / 2 + 150, 500), (W / 2 + 700 * k, 500), MID, 1)
    L.comp(c)
    word(c, t, 0, 0, W / 2 - 330, 455, 54, "slide", anchor=(1, 0.5), dx=120)
    word(c, t, 0, 1, W / 2 - 175, 455, 76, "slam", anchor=(1, 0.5))
    word(c, t, 0, 2, W / 2 + 175, 455, 54, "slide", anchor=(0, 0.5), dx=-120)
    return dict(flash=0.6 * (1 - ramp(lt, 0, 0.25)))


def _eye_band(c, lt, d, wide=False, t=0.0):
    eyes = load_asset("c02_eyes", "mono")
    h, w = eyes.shape[:2]
    band = eyes[int(h * 0.33):int(h * 0.69)]
    open_k = ease_out(ramp(lt, 0.02, 0.35 if wide else 0.6))
    s = W / band.shape[1] * ((1.25 - 0.1 * ease_out(ramp(lt, 0, 0.4))) if wide else (1.06 + 0.04 * lt / d))
    s *= 1 + 0.03 * kick_pulse(t)
    put(c, band, W / 2, H / 2 - 60, sx=s, sy=s * (0.08 + 0.92 * open_k))


def s_eyes(c, t, lt, d):
    c[:] = 0.07
    _eye_band(c, lt, d, t=t)
    word(c, t, 0, (3, 4), W / 2 - 240, 880, 50, "wipe", color=LIGHT, anchor=(1, 0.5))
    word(c, t, 0, 5, W / 2 - 210, 880, 50, "slide", color=LIGHT, anchor=(0, 0.5), dx=-60)
    word(c, t, 0, 6, W / 2 + 330, 875, 120, "slam", color=LIGHT, sh=6)
    return dict(grain=0.03, vignette=1.2)


def s_wipe(c, t, lt, d):
    paper_bg(c, t)
    L = Layer()
    k = ease_io(ramp(lt, 0, 0.45))
    x = lerp(W + 400, 1050, k)
    L.poly([(x, 0), (W, 0), (W, H), (x - 520, H)], (0.12, 0.12, 0.13))
    L.comp(c)
    flood = ramp(t, kt.words(1)[1].t, B(1, 1))
    if flood > 0:
        tmp = new_canvas((0, 0, 0))
        kt.echo_bg(tmp, "詞は有り余るばかり", t, size=34, color=(0.5, 0.5, 0.52), alpha=1.0, speed=140, gap=1.5,
                   rows=int(2 + 22 * flood))
        mask = np.zeros((H, W), np.float32)
        cv2.fillConvexPoly(mask, np.array([(int(x), 0), (W, 0), (W, H), (int(x - 520), H)], np.int32), 1.0)
        c += tmp * mask[..., None] * 0.45
    for j in range(6):
        jx, jy = 360 + j * 115 + 30 * math.sin(lt * 2 + j), 330 + 60 * j - 60 * lt
        rot = 8 + 12 * lt + j * 9
        a = appear(lt, 0.1 + j * 0.1)
        put(c, rect_cached(150, 150, (0.97, 0.97, 0.98), 2, 2, (0.6, 0.6, 0.62)), jx, jy, rot=rot, alpha=a * 0.9)
        put(c, kt.wimg("詞", 80, "jpm", GREY), jx, jy, rot=rot, alpha=a * 0.8)
    word(c, t, 1, 0, 1600, 140, 130, "slam", color=LIGHT, v=True, anchor=(0.5, 0.0))
    word(c, t, 1, 1, 1440, 220, 64, "wipe", color=LIGHT, v=True, anchor=(0.5, 0.0), dur=0.5)
    return {}


def s_birth(c, t, lt, d):
    c[:] = 0.05
    t0 = kt.words(1)[2].t
    k = ease_out_back(ramp(t, t0, t0 + 0.6))
    sp = spark_rgba(160)
    put(c, sp, W / 2, 430, scale=(0.2 + 0.8 * k) * (1 + 0.08 * kick_pulse(t)), rot=lt * 30, alpha=k)
    put(c, sp, W / 2, 430, scale=(0.2 + 0.8 * k) * 1.6, rot=lt * 30, alpha=0.12 * k, mode="add")
    ws = kt.words(1)
    xs = [W / 2 - 330, W / 2 - 60, W / 2 + 250]
    for j, k_ in enumerate((2, 3, 4)):
        w = ws[k_]
        drift = max(0.0, t - w.t) * 60
        word(c, t, 1, k_, xs[j] + drift, 700, 64 if k_ < 4 else 74, "slide", color=LIGHT, dx=-160)
    return dict(grain=0.03, vignette=1.2, bloom=0.6)


def s_alpha_heart(c, t, lt, d):
    paper_bg(c, t)
    L = Layer()
    L.circle((W / 2, H / 2), 330 * ease_out(ramp(lt, 0, 0.5)), (0.7, 0.7, 0.72), 1)
    L.comp(c)
    sc = 1 + 0.06 * kick_pulse(t)
    put(c, checker_heart_rgba(220), W / 2, H / 2, scale=sc, alpha=appear(lt, 0.05))
    ws = kt.words(2)
    fill_k = ease_out(ramp(t, ws[1].t, ws[2].t + 0.6))
    back = ramp(t, ws[3].t, LYS[2]["end"] - 0.2)
    glitch = back > 0 and hash01(int(t * 20)) > 0.55 + 0.4 * (1 - back)
    a = fill_k * (1 - back) * (0.35 if glitch else 1)
    put(c, heart_rgba(220, OR), W / 2, H / 2, scale=sc, alpha=a)
    col(c, t, 2, [0, 1], 1500, 160, 62)
    word(c, t, 2, 2, 1380, 260, 62, "wipe", v=True, anchor=(0.5, 0.0), dur=0.4)
    col(c, t, 2, [3, 4], 470, 200, 56, "slide", dy=-40, dx=0)
    word(c, t, 2, 5, 330, 300, 110, "glitch", v=True, anchor=(0.5, 0.0), amt=0.8, settle=1.2)
    return {}


def s_speakers(c, t, lt, d):
    paper_bg(c, t)
    sp = load_asset("o07_speaker", "gray")
    s = 430 / sp.shape[0] * (1 + 0.04 * kick_pulse(t, 0.08))
    for (x0, flip) in ((430, 1), (W - 430, -1)):
        for j, tk in enumerate([k for k in KICKS if t - 1.0 < k <= t][-4:]):
            u = (t - tk) / 1.0
            r = 160 + 600 * u
            for q in range(7):
                a = -math.pi / 2 + (q - 3) * 0.35
                put(c, kt.wimg("音", 34, "jpm", GREY), x0 + flip * r * math.cos(a) * 0.9 + flip * 60, 500 + r * math.sin(a) * 0.6 + 130,
                    alpha=0.5 * (1 - u))
    put(c, sp, 430, 500, scale=s)
    put(c, sp[:, ::-1].copy(), W - 430, 500, scale=s)
    k = ease_out(ramp(lt, 0, 0.2))
    put(c, rect_cached(760, 150, (0.98, 0.98, 0.99), 4, 2, (0.55, 0.55, 0.57)), W / 2, 480, sx=k, sy=1)
    word(c, t, 3, (0, 2), W / 2 - 90, 480, 56, "slam", anchor=(1, 0.5))
    word(c, t, 3, 3, W / 2 - 60, 480, 56, "slide", anchor=(0, 0.5), dx=-80)
    w = kt.words(3)[4]
    if t >= w.t:
        u = t - w.t
        put(c, kt.wimg(w.s, 70, "jpb", INKD), W / 2 + 200, 480, scale=1.0 + 0.25 * kick_pulse(t) - 0.15 * ease_in(clamp(u / 1.2)),
            alpha=1 - 0.6 * ramp(u, 0.6, 1.3))
    L = Layer()
    L.line((420, 880), (1500, 880), (0.75, 0.75, 0.77), 2)
    px = 420 + 1080 * (0.05 + 0.04 * lt / d)
    L.line((420, 880), (px, 880), (0.3, 0.3, 0.32), 2)
    L.circle((px, 880), 7, (0.3, 0.3, 0.32))
    L.comp(c)
    return {}


def s_globe(c, t, lt, d):
    paper_bg(c, t)
    draw_globe(c, W / 2, H / 2, 360, t * 0.25, color=(0.42, 0.42, 0.45), dot=3, highlight=0.03 * ramp(lt, 1.5, 3))
    kt.along_circle(c, "全世界共通の快楽さ　" * 3, W / 2, H / 2, 430, t * 0.35, size=28, color=MID,
                    alpha=appear(lt, 0.3, 0.6))
    ws = kt.words(3)
    for j, k_ in enumerate((5, 6, 7)):
        word(c, t, 3, k_, W / 2 + 110 - j * 110, 300, 84, "slam", v=True, anchor=(0.5, 0.0), keep=3.0)
    return {}


def s_title(c, t, lt, d):
    paper_bg(c, t)
    g = ramp(lt, 0, 2.4)
    draw_heart(c, W / 2, 430, 200 * (1 + 0.05 * kick_pulse(t)), ay=0.35 * math.sin(lt * 0.7), ax=0.1, gather=g,
               orange=0.08)
    k = appear(lt, 2.2, 0.5)
    kt.tracking(c, "アンノウン・マザーグース", W / 2, 720, t, t - lt + 2.2, t - lt + 5.0, size=44, font="jpm",
                color=(0.22, 0.22, 0.24), sp0=0.05, sp1=0.3, alpha=k)
    ht(c, "Unknown Mother-Goose  ―  Claude ver.", W / 2, 790, size=17, font="thin", color=MID, spacing=0.3,
       alpha=appear(lt, 2.8, 0.6))
    return {}


def s_land(c, t, lt, d):
    paper_bg(c, t)
    draw_landscape(c, t, cam_h=1.25 + 0.15 * math.sin(lt * 0.3), horizon=0.55, sway=lt * 0.05)
    bob = math.sin(lt * 1.3) * 12
    draw_heart(c, W / 2, 310 + bob, 160 * (1 + 0.08 * kick_pulse(t, 0.12)), ay=lt * 0.6, ax=0.15, orange=0.08)
    ht(c, "wowaka feat. 初音ミク", W / 2 + 470, 300, size=22, font="jpm", color=GREY, alpha=fade_io(lt, d, 1.0, 0.8))
    return {}


# =====================================================================  VERSE A
def _cables(c, t, anchor, n=7, dark=True, seed=0):
    L = Layer(ss=1)
    for i in range(n):
        y0 = -50 + hash01(i, seed) * 300
        x0 = 200 + hash01(i, seed, 1) * 1500
        pts = []
        for k in range(25):
            u = k / 24
            x = lerp(x0, anchor[0] + (i - n / 2) * 18, u) + math.sin(u * 3 + i + t * 0.6) * 40 * (1 - u)
            y = lerp(y0, anchor[1], u) + math.sin(u * math.pi) * (120 + 60 * hash01(i, 5))
            pts.append((x, y))
        L.polyline(pts, (0.30, 0.30, 0.32) if dark else (0.6, 0.6, 0.62), 5 if i % 3 else 9)
    L.comp(c)


def s_lying_dark(c, t, lt, d):
    paper_bg(c, t, dark=True)
    ly = load_asset("c03_lying", "mono")
    s = 1350 / ly.shape[1]
    put(c, ly * 0.75, W / 2 - 120, 650, scale=s * (1 + 0.01 * lt))
    _cables(c, t, (W / 2 - 520, 610), dark=True)
    ws = kt.words(4)
    word(c, t, 4, 0, 1700, 90, 52, "slide", color=LIGHT, v=True, anchor=(0.5, 0.0), dy=-50, dx=0)
    word(c, t, 4, 1, 1700, 380, 52, "slide", color=LIGHT, v=True, anchor=(0.5, 0.0), dy=-50, dx=0)
    w = ws[2]
    sink = 0 if t < w.t else 40 * ease_out(clamp((t - w.t) / 1.5))
    word(c, t, 4, 2, 1700, 580 + sink, 52, "fade", color=LIGHT, v=True, anchor=(0.5, 0.0), alpha=1 - 0.4 * ramp(t, w.t + 0.5, w.t + 1.5))
    word(c, t, 4, 3, 1610, 300, 52, "slide", color=LIGHT, v=True, anchor=(0.5, 0.0), dy=-50, dx=0)
    for j, k_ in enumerate((4, 5)):
        word(c, t, 4, k_, 1480, 120 + j * 200, 60, "hollow", color=LIGHT, v=True, anchor=(0.5, 0.0), s0=1.0, ghost=0)
    w = ws[6]
    word(c, t, 4, 6, 1480, 520, 90, "hollow", color=LIGHT, v=True, anchor=(0.5, 0.0), s0=1.0, ghost=0)
    if t >= w.t:
        kt.wipe(c, kt.wimg(w.s, 90, "jpm", LIGHT, True), 1480, 520, t, w.t + 0.1, dur=0.6, anchor=(0.5, 0.0), axis=0)
    for j, k_ in enumerate((4, 5)):
        ww = ws[k_]
        if t >= ww.t:
            kt.wipe(c, kt.wimg(ww.s, 60, "jpm", LIGHT, True), 1480, 120 + j * 200, t, ww.t + 0.05, dur=0.35,
                    anchor=(0.5, 0.0), axis=0)
    return dict(grain=0.03, vignette=1.1)


def s_screens(c, t, lt, d):
    paper_bg(c, t)
    off = lt * 300
    L = Layer()
    for i in range(-1, 9):
        x = 140 + i * 300 - off % 300
        y = 400 + (i % 2) * 8
        L.rect(x - 130, y - 85, x + 130, y + 85, (0.18, 0.18, 0.2))
        L.rect(x - 122, y - 77, x + 122, y + 77, (0.95, 0.95, 0.96))
    L.comp(c)
    for i in range(-1, 9):
        x = 140 + i * 300 - off % 300
        y = 400 + (i % 2) * 8
        n = ((i + int((off + 140) // 300)) % 4)
        put(c, kt.wimg("繰り返す", 34, "jpm", GREY), x, y - 30, alpha=0.8)
        put(c, kt.wimg("使い回しの歌", 30, "jpm", MID), x, y + 22, alpha=0.8 if n % 2 else 0.5)
    row(c, t, 5, [0, 1, 2], W / 2, 720, 70, "slide", dx=320, gap=0.35)
    return {}


def s_ears(c, t, lt, d):
    paper_bg(c, t, tone=-0.1)
    e = load_asset("n05_ears", "mono")
    s = 1250 / e.shape[0] * (1 + 0.03 * lt)
    put(c, e, W / 2, 680, scale=s)
    close = ease_in(ramp(t, kt.words(5)[5].t, kt.words(5)[5].t + 0.9))
    word(c, t, 5, 3, 420, 330, 60, "slam", ghost=0.2)
    word(c, t, 5, 4, W - 420, 330, 80, "slam", ghost=0.2)
    word(c, t, 5, 5, W / 2, 940, 120, "slam", color=INKD, sh=4)
    if close > 0:
        yy, xx = np.ogrid[:H, :W]
        r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
        c *= (1 - close * 0.85 * np.clip((r - 0.25) / 0.35, 0, 1))[..., None].astype(np.float32)
    return {}


def s_bubbles(c, t, lt, d):
    paper_bg(c, t)
    for i in range(14):
        bx = (hash01(i, 1) * 2200 - 150 + lt * (40 + 60 * hash01(i, 2))) % 2200 - 150
        by = 120 + hash01(i, 3) * 820
        if abs(by - H / 2) < 150 and abs(bx - W / 2) < 480:
            continue
        bw, bh = int(170 + hash01(i, 4) * 200), int(70 + hash01(i, 5) * 40)
        g = 0.42 + 0.3 * hash01(i, 6)
        put(c, bubble_rgba(bw, bh, (g, g, g + 0.01), "left" if i % 2 else "right"), bx, by, alpha=0.9)
        put(c, kt.wimg("愛" if i % 3 == 0 else "…", 30, "jpm", (0.97, 0.97, 0.98)), bx, by - 10, alpha=0.8)
    k = ease_out_back(ramp(lt, 0.0, 0.35)) * (1 + 0.04 * kick_pulse(t))
    put(c, bubble_rgba(820, 170, (0.12, 0.12, 0.13), "left", 12), W / 2, H / 2, scale=k)
    row(c, t, 6, [0, 1, 2], W / 2 - 20, H / 2 - 10, 58, "slam", color=LIGHT, ghost=0.25)
    return {}


def s_ring(c, t, lt, d):
    paper_bg(c, t)
    L = Layer(ss=1)
    n = 16
    R = 270
    for i in range(n):
        a = i / n * 360 + lt * 12
        k = appear(lt, i * 0.03)
        g = 0.25 + 0.55 * hash01(i, 8)
        if i == beat_index(t) % n:
            g = 0.1
        ar = math.radians(a)
        cx, cy = W / 2 + R * math.cos(ar), H / 2 + R * math.sin(ar)
        dx, dy = math.cos(ar + math.pi / 2), math.sin(ar + math.pi / 2)
        ux, uy = math.cos(ar), math.sin(ar)
        hw, hd = 44 * k, 26
        L.poly([(cx - dx * hw - ux * hd, cy - dy * hw - uy * hd), (cx + dx * hw - ux * hd, cy + dy * hw - uy * hd),
                (cx + dx * hw + ux * hd, cy + dy * hw + uy * hd), (cx - dx * hw + ux * hd, cy - dy * hw + uy * hd)],
               (g, g, g + 0.01))
    L.comp(c)
    ta = kt.sub(6, "答").t
    if t >= ta:
        kt.slam(c, kt.wimg("答", 190, "jpb", INKD), W / 2, H / 2, t, ta, s0=2.4, sh=6)
    word(c, t, 6, 3, 520, 300, 70, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=-60)
    word(c, t, 6, 5, W - 520, 300, 70, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=60)
    return dict(flash=0.4 * (1 - ramp(t, ta, ta + 0.15)) if t >= ta else 0)


def s_lying_white(c, t, lt, d):
    paper_bg(c, t)
    ly = load_asset("c03_lying", "mono")
    s = 1200 / ly.shape[1]
    put(c, ly, W / 2 - 80, 640, scale=s)
    put(c, reflect(ly, 0.12, 0.25), W / 2 - 80, 640 + ly.shape[0] * s * 0.5, scale=s, anchor=(0.5, 0))
    k = ease_out_back(ramp(lt, 0.0, 0.3))
    put(c, bubble_rgba(820, 110, (0.15, 0.15, 0.16), "left", 10), 820, 290, scale=k)
    for k_ in (0, 1, 2):
        pass
    w = kt.merge(7, 0, 2)
    kt.word_chars(c, w, t, 800, 280, size=50, color=LIGHT)
    return {}


def s_eyes_wide(c, t, lt, d):
    c[:] = 0.07
    _eye_band(c, lt, d, wide=True, t=t)
    word(c, t, 7, (3, 4), 330, 880, 60, "slam", color=LIGHT, anchor=(0, 0.5))
    w = kt.words(7)[5]
    word(c, t, 7, 5, W / 2 + 230, 875, 150, "slam", color=LIGHT, sh=10, s0=2.2)
    fl = 0.6 * (1 - ramp(t, w.t, w.t + 0.18)) if t >= w.t else 0.5 * (1 - ramp(lt, 0, 0.2))
    return dict(grain=0.03, vignette=1.2, flash=fl, ca=3 * (1 - ramp(t, w.t, w.t + 0.4)) if t >= w.t else 0)


# =====================================================================  LETTER / KNEES / PAPER HEART
def _pencil(L, x, y, ang, ln=520):
    a = math.radians(ang)
    ux, uy = math.cos(a), math.sin(a)
    px, py = -uy, ux
    w = 9
    tip = (x, y)
    b = (x + ux * 60, y + uy * 60)
    e = (x + ux * ln, y + uy * ln)
    L.poly([tip, (b[0] + px * w, b[1] + py * w), (b[0] - px * w, b[1] - py * w)], (0.78, 0.75, 0.70))
    L.poly([(x + ux * 14, y + uy * 14), (x + ux * 22 + px * 3, y + uy * 22 + py * 3), (x + ux * 22 - px * 3, y + uy * 22 - py * 3)],
           (0.15, 0.15, 0.15))
    L.poly([(b[0] + px * w, b[1] + py * w), (e[0] + px * w, e[1] + py * w), (e[0] - px * w, e[1] - py * w),
            (b[0] - px * w, b[1] - py * w)], (0.24, 0.24, 0.26))


def s_letter(c, t, lt, d):
    paper_bg(c, t)
    rot = -12 + lt * 1.2
    zoom = 1.0 + 0.025 * lt
    put(c, rect_cached(620, 860, (0.0, 0.0, 0.0), 0, 0), W / 2 - 360 + 14, H / 2 + 20, scale=zoom, rot=rot, alpha=0.12)
    put(c, rect_cached(620, 860, (0.985, 0.985, 0.99), 0, 0), W / 2 - 360, H / 2, scale=zoom, rot=rot)
    L = Layer()
    a = math.radians(rot)
    prog = LY.sung(LYS[8], t)
    cxp = W / 2 - 360
    for i in range(14):
        y = -360 + i * 56
        p0 = (cxp + (-260 * math.cos(a) - y * math.sin(a)) * zoom, H / 2 + (-260 * math.sin(a) + y * math.cos(a)) * zoom)
        p1 = (cxp + (260 * math.cos(a) - y * math.sin(a)) * zoom, H / 2 + (260 * math.sin(a) + y * math.cos(a)) * zoom)
        L.line(p0, p1, (0.86, 0.86, 0.88), 1)
        w = clamp(prog * 14 - i)
        if w > 0:
            xe = -230 + 440 * w * (0.6 + 0.4 * hash01(i))
            q0 = (cxp + (-230 * math.cos(a) - (y - 14) * math.sin(a)) * zoom, H / 2 + (-230 * math.sin(a) + (y - 14) * math.cos(a)) * zoom)
            q1 = (cxp + (xe * math.cos(a) - (y - 14) * math.sin(a)) * zoom, H / 2 + (xe * math.sin(a) + (y - 14) * math.cos(a)) * zoom)
            L.line(q0, q1, (0.55, 0.55, 0.57), 2)
    wob = math.sin(t * 9) * 6
    _pencil(L, W / 2 - 130 + wob, 700 + wob * 0.5, 160 + math.sin(t * 3) * 3)
    L.comp(c)
    col(c, t, 8, [0, 1], 1620, 140, 64, "wipe", dur=0.35)
    word(c, t, 8, 2, 1460, 200, 120, "wipe", v=True, anchor=(0.5, 0.0), dur=0.45)
    word(c, t, 8, 3, 1310, 260, 56, "wipe", v=True, anchor=(0.5, 0.0), dur=0.6)
    return {}


def s_knees(c, t, lt, d):
    paper_bg(c, t, tone=-0.05)
    kn = load_asset("c09_knees", "mono")
    s = 900 / kn.shape[0] * (1 + 0.012 * lt)
    put(c, kn, W / 2 + 300, 560, scale=s)
    w0 = kt.words(9)[0]
    if kt.active(9, t) and t >= w0.t:
        img = kt.wimg(w0.s, 60, "jpm", INKD, True)
        kt.glitch(c, img, 760, 120, t, amt=0.25 + 0.25 * kick_pulse(t), anchor=(0.5, 0.0), seed=9, slices=5)
    word(c, t, 9, 1, 640, 160, 72, "slam", v=True, anchor=(0.5, 0.0))
    word(c, t, 9, 2, 500, 200, 110, "slam", v=True, anchor=(0.5, 0.0), sh=4)
    word(c, t, 9, 3, 500, 460, 110, "slam", v=True, anchor=(0.5, 0.0), sh=4)
    return {}


def s_paper_heart(c, t, lt, d):
    paper_bg(c, t)
    seg = 0 if t < T(11) else 1 if t < T(12) else 2
    ay = [0.5, -0.7, 0.2][seg] + lt * 0.15
    sz = [330, 520, 280][seg]
    cy = [520, 560, 480][seg]
    expl = 0.0
    if seg == 2:
        expl = 0.3 * ease_in(LY.sung(LYS[12], t)) + 0.03 * kick_pulse(t)
    draw_heart(c, W / 2, cy, sz * (1 + 0.04 * kick_pulse(t)), ay=ay, ax=0.2, az=[0.1, -0.25, 0.0][seg], white=0.55,
               orange=0.0, base=0.6, contrast=0.35, explode=expl)
    if seg == 0:
        word(c, t, 10, (0, 1), 1240, 260, 64, "slide", anchor=(0, 0.5), dx=-100)
        word(c, t, 10, 2, 1240, 420, 170, "hollow", anchor=(0, 0.5), s0=1.6)
        w = kt.words(10)[2]
        if t >= w.t + 0.08:
            kt.wipe(c, kt.wimg(w.s, 170, "jpb", INKD), 1240, 420, t, w.t + 0.08, dur=0.25, anchor=(0, 0.5))
        word(c, t, 10, 3, 1240, 590, 60, "slide", anchor=(0, 0.5), dx=-100)
    elif seg == 1:
        ws = kt.words(11)
        end_t = LYS[11]["next"] - 0.35
        sw = ease_in(ramp(t, end_t, end_t + 0.3))
        for j, (k_, x, y, size) in enumerate([(0, 520, 130, 60), (1, 400, 200, 60), (2, 260, 150, 80)]):
            w = ws[k_]
            if t < w.t:
                continue
            img = kt.wimg(w.s, size, "jpm", INKD, True)
            px, py = lerp(x, W / 2, sw), lerp(y, cy, sw)
            put(c, img, px, py, anchor=(0.5, 0.0), scale=(1 - 0.9 * sw) * (1 + 0.3 * (1 - ease_out(clamp((t - w.t) / 0.1)))),
                alpha=1 - sw)
    else:
        ws = kt.words(12)
        word(c, t, 12, 0, W / 2, 860, 60, "slide", dx=-200)
        tb = T(13) - 0.12
        w = ws[1]
        if t >= w.t:
            img = kt.wimg(w.s, 170, "jpb", INKD)
            if t < tb:
                kt.slam(c, img, W / 2 - 120, 980 - 260, t, w.t, sh=5)
            else:
                kt.shatter(c, img, W / 2 - 120, 720, t, tb, n=5, force=1400)
        word(c, t, 12, 2, W / 2 + 160, 730, 60, "slam", anchor=(0, 0.5), sh=3)
    return {}


def s_shout(c, t, lt, d):
    paper_bg(c, t)
    sh = load_asset("n06_shout", "mono")
    dx, dy = kt.shake(t, 10 * kick_pulse(t) + 3, 5)
    put(c, sh, 520 + dx, 640 + dy, scale=1250 / sh.shape[0] * (1 + 0.05 * lt))
    L = LYS[13]
    s = L["ja"]
    xs = [1130, 1330, 1530, 1730]
    k = 0
    fl = 0.0
    for j, ch in enumerate(s):
        ta = L["chars"][j]
        if ch == "！":
            put(c, kt.wimg("！", 260, "jpb", OR), 1820, 560, alpha=1.0 if t >= ta + 0.15 else 0) if t >= ta + 0.15 else None
            continue
        if t >= ta:
            kt.slam(c, kt.wimg(ch, 230, "jpb", INKD), xs[k % 4] + dx, 560 + dy + (k % 2) * 40, t, ta, s0=2.4, sh=8)
            fl = max(fl, 0.35 * (1 - ramp(t, ta, ta + 0.1)))
        k += 1
    return dict(flash=fl, ca=2 + 3 * kick_pulse(t))


# =====================================================================  CHORUS 1 : ribbons
TAGS14 = [(560, 330, -4), (1330, 560, 3), (820, 790, -2)]
TAGS15 = [(700, 380, 3), (1180, 680, -3)]
BIG14 = [("世界", 330, 0.35), ("愛の唄", 270, 0.0), ("歌わせて", 250, 0.0)]


def _tag(c, t, i, k, x, y, rot, size=50, dim=False):
    ws = pw(i, k)
    s = kt.join(ws)
    t0 = ws[0].t
    if t < t0 - 0.03:
        return
    kk = ease_out(ramp(t, t0 - 0.03, t0 + 0.12))
    wd, _ = kt.wsize(s, size)
    put(c, rect_cached(int(wd + 70), int(size * 1.8), (0.99, 0.99, 1.0), 0, 2, (0.45, 0.45, 0.47)), x, y, sx=kk, sy=1,
        rot=rot, alpha=0.5 if dim else 1)
    xx = x - wd / 2
    for w in ws:
        ww, _ = kt.wsize(w.s, size)
        word(c, t, i, w, xx + ww / 2, y, size, "slam", color=MID if dim else INKD, ghost=0.2, s0=1.5)
        xx += ww + size * 0.04


def s_ribbons(c, t, lt, d):
    paper_bg(c, t)
    energy = ramp(t, T(15), T(15) + 0.3)
    cam = 0
    for k in range(3):
        if t >= LYS[14]["bounds"][k]:
            cam = k
    if t >= T(15):
        cam = 3 + (1 if t >= B(15, 1) else 0)
    L = Layer()
    nl = 11 + int(energy * 8)
    for i in range(nl):
        y0 = hash01(i, 1) * H
        sl = (hash01(i, 2) - 0.5) * 0.5
        off = (lt * (40 + 80 * energy) * (0.5 + hash01(i, 3))) % 400
        g = 0.55 + 0.35 * hash01(i, 4)
        th = 3 + int(hash01(i, 5) * 22 * (1 + 0.5 * kick_pulse(t)))
        x0 = -200 - off
        L.line((x0, y0 + sl * x0), (W + 200, y0 + sl * (W + 200)), (g, g, g + 0.01), th)
    for i in range(26):
        x = (hash01(i, 7) * 2100 - lt * (90 + 160 * hash01(i, 8)) * (1 + energy)) % 2100 - 90
        y = hash01(i, 9) * H
        s = 6 + hash01(i, 10) * 30
        g = 0.15 + 0.2 * hash01(i, 11)
        L.rect(x, y, x + s, y + s, (g, g, g))
    L.comp(c)
    rot, zoom = [(0, 1.0), (-7, 1.15), (5, 0.92), (0, 1.25), (-4, 1.0)][cam]
    if rot or zoom != 1.0:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, zoom)
        c[:] = cv2.warpAffine(c, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    if t < T(15):
        for k, (s, size, a) in enumerate(BIG14):
            t0 = pw(14, k)[0].t
            if t >= t0:
                put(c, kt.hollow(s, size, "jpb", OR if "愛" in s else GREY, 3), W / 2 + (k - 1) * 220, H / 2 + (1 - k) * 60,
                    alpha=0.35 * ease_out(ramp(t, t0, t0 + 0.2)), scale=1 + 0.05 * (t - t0))
        for k, (x, y, r) in enumerate(TAGS14):
            _tag(c, t, 14, k, x, y, r)
    else:
        w = kt.merge(15, 0, 1)
        if t >= w.t:
            kt.slam(c, kt.wimg(w.s, 230, "jpb", INKD), W / 2, H / 2 - 40, t, w.t, s0=2.0, sh=8,
                    alpha=1 - 0.75 * ramp(t, B(15, 1), B(15, 1) + 0.2))
        if t >= B(15, 1) - 0.05:
            _tag(c, t, 15, 1, 1080, 720, -3, size=58)
            put(c, kt.hollow("想い", 300, "jpb", GREY, 3), 1250, 380, alpha=0.3 * ramp(t, kt.sub(15, "想い").t, kt.sub(15, "想い").t + 0.2))
    if t > T(16) - 0.12:
        c *= 0.05
    return {}


def s_bust(c, t, lt, d):
    paper_bg(c, t)
    b = load_asset("c08_bust_sing", "mono")
    s = 1000 / b.shape[0]
    put(c, b, W / 2, 610 + 10 * math.sin(lt * 2), scale=s * (1.0 + 0.02 * lt))
    ws = kt.words(16)
    halves = [(0, 1), (2, 2)]
    for j, ((a, bb), (x0, x1)) in enumerate(zip(halves, [(60, 700), (1220, 1860)])):
        w = kt.merge(16, a, bb)
        k = ease_out(ramp(t, w.t - 0.05, w.t + 0.15))
        fill(c, (0.13, 0.13, 0.14), 1.0, (x0, 560 - 42, x0 + (x1 - x0) * k, 560 + 42))
        if t >= w.t:
            kt.word_chars(c, w, t, (x0 + x1) / 2, 560, size=46 if j == 0 else 40, color=LIGHT)
    return {}


def s_select(c, t, lt, d):
    paper_bg(c, t)
    ch = load_asset("c01_stand_front", "mono")
    z = 1 + 0.35 * ease_in(clamp(lt / d))
    s = 640 / ch.shape[0] * z
    cx, cy = W / 2, 520 + 120 * (z - 1)
    jx, jy = kt.shake(t, (4 + 10 * (z - 1)) * kick_pulse(t), 2)
    ct = [x for x in LYS[17]["chars"] if x <= t]
    alive = 0.12 + 0.88 * math.exp(-(t - ct[-1]) / 0.09) if ct else 0.12
    put(c, ch, cx, cy, scale=s * (1 + 0.015 * kick_pulse(t)), alpha=alive)
    L = Layer()
    bw, bh = 470 * (2 - z), 680 * (2 - z) ** 0.5
    L.rect(cx - bw / 2 + jx, cy - bh / 2 + jy, cx + bw / 2 + jx, cy + bh / 2 + jy, (0.55, 0.55, 0.6), 1)
    for (x, y) in [(cx - bw / 2, cy - bh / 2), (cx + bw / 2, cy - bh / 2), (cx - bw / 2, cy + bh / 2),
                   (cx + bw / 2, cy + bh / 2), (cx, cy - bh / 2), (cx, cy + bh / 2)]:
        L.rect(x - 6 + jx, y - 6 + jy, x + 6 + jx, y + 6 + jy, (1, 1, 1))
        L.rect(x - 6 + jx, y - 6 + jy, x + 6 + jx, y + 6 + jy, (0.45, 0.45, 0.5), 1)
    L.comp(c)
    put(c, rect_cached(760, 84, (0.99, 0.99, 1.0), 2, 1, (0.6, 0.6, 0.62)), cx, 930)
    w = kt.merge(17, 0, len(kt.words(17)) - 1)
    wd, _ = kt.word_chars(c, w, t, cx - 340, 930, size=46, color=INKD, anchor=(0, 0.5), pop=0.4)
    if beat_index(t) % 2:
        n = sum(1 for j in w.idx if LYS[17]["chars"][j] <= t)
        f = LY._font("jpm", 46)
        ww = sum(f.getlength(LYS[17]["ja"][j]) + 46 * 0.08 for j in w.idx[:n])
        fill(c, GREY, 1.0, (cx - 336 + ww, 905, cx - 333 + ww, 955))
    return {}


# ---------------------------------------------------------------- 1:31  "can you see me?" through the glass
def _crack_paths(seed, cx, cy):
    rng = np.random.default_rng(seed)
    paths = []
    for b in range(16):
        a = rng.uniform(0, 2 * math.pi)
        x, y = cx, cy
        pts = [(x, y)]
        for j in range(9):
            a += rng.normal(0, 0.3)
            ln = rng.uniform(50, 150)
            x += math.cos(a) * ln
            y += math.sin(a) * ln
            pts.append((x, y))
        paths.append(pts)
    rings = []
    for r in (90, 190, 320):
        seg = []
        for b in range(16):
            a0 = b / 16 * 2 * math.pi + rng.uniform(-0.1, 0.1)
            seg.append((cx + r * math.cos(a0) * rng.uniform(0.85, 1.15), cy + r * math.sin(a0) * rng.uniform(0.85, 1.15)))
        rings.append(seg)
    return paths, rings


def _crack(c, k, cx, cy, seed=3):
    if k <= 0:
        return
    paths, rings = _crack_paths(seed, cx, cy)
    L = Layer(ss=1)
    for pts in paths:
        m = max(2, int(len(pts) * k))
        sh = [(x + 2, y + 2) for x, y in pts[:m]]
        L.polyline(sh, (0.05, 0.05, 0.06), 3)
        L.polyline(pts[:m], (0.97, 0.97, 0.99), 2)
    for j, seg in enumerate(rings):
        if k > 0.3 + j * 0.2:
            for a, b in zip(seg[::2], seg[1::2]):
                L.line(a, b, (0.97, 0.97, 0.99), 1)
    L.comp(c)


def s_glass(c, t, lt, d):
    ws = kt.words(18)
    t1, t2 = ws[1].t, ws[2].t
    q = kt.sub(18, "？").t
    q = max(q, t2 + 0.25)
    if t < t1:
        c[:] = 0.05
        g = load_asset("n01_glass", "mono")
        s = 1500 / g.shape[1] * (1 + 0.04 * lt)
        put(c, g * 0.8, W / 2, 600, scale=s)
        L = Layer()
        L.rect(170, 80, W - 170, H - 80, (0.6, 0.6, 0.62), 2)
        for y in range(80, H - 80, 6):
            L.line((170, y), (W - 170, y), (0, 0, 0), 1)
        L.comp(c, 0.25)
        word(c, t, 18, 0, 260, 160, 64, "slide", color=LIGHT, anchor=(0, 0.5), dx=-120)
    elif t < t2:
        c[:] = 0.05
        g = load_asset("n01_glass", "mono")
        s = 1500 / g.shape[1] * 2.1
        put(c, g, W / 2, 900, scale=s * (1 + 0.1 * (t - t1)))
        word(c, t, 18, 1, W / 2, H / 2 + 330, 230, "slam", color=LIGHT, s0=2.6, sh=10, stroke=9,
             sc=(0.04, 0.04, 0.05))
    else:
        e = load_asset("n08_eye", "mono")
        s = W / e.shape[1] * (1.15 + 0.15 * ease_out(clamp((t - t2) / 1.0)))
        c[:] = 0.02
        put(c, e, W / 2, H / 2, scale=s)
        w = ws[2]
        main = w.s.replace("？", "")
        img = kt.wimg(main, 250, "jpb", LIGHT, False, 6, (0.05, 0.05, 0.06))
        if t < q:
            kt.slam(c, img, W / 2, H / 2 + 300, t, t2, s0=2.6, sh=14)
        else:
            kt.glitch(c, img, W / 2, H / 2 + 300, t, amt=0.8, seed=18)
            kt.slam(c, kt.wimg("？", 620, "jpb", OR, False, 8, (0.05, 0.05, 0.06)), W / 2 + 560, H / 2 - 40, t, q, s0=3.0,
                    sh=16)
            _crack(c, ease_out(clamp((t - q) / 0.25)), W / 2 + 80, H / 2 - 60)
    fl = kt.flash_at(t, [t1, t2, q], 0.12, 0.6)
    return dict(flash=fl, ca=4 * fl, grain=0.04)


def s_back_pelted(c, t, lt, d):
    paper_bg(c, t)
    b = load_asset("c05_back", "mono")
    s = 900 / b.shape[0]
    put(c, b, W / 2 + 30, 640, scale=s * (1 + 0.01 * lt))
    balls = _balls()
    beats = [x for x in BEATS if T(19) <= x < T(20)][::2]
    for j, tb in enumerate(beats):
        if t < tb:
            break
        u = ease_in(ramp(t, tb, tb + 0.35))
        bl = balls[j % len(balls)]
        sx = (hash01(j, 4) - 0.5) * 1600 + W / 2
        x = lerp(sx, W / 2 + (hash01(j, 5) - 0.5) * 260, u)
        y = lerp(1250, 450 + hash01(j, 6) * 300, u) - 300 * math.sin(u * math.pi)
        sc = lerp(1.6, 0.35, u) * 220 / max(bl.shape[:2])
        if u < 1:
            put(c, bl, x, y, scale=sc, rot=j * 70 + u * 200)
        else:
            dd = t - tb - 0.35
            put(c, bl, x + dd * 80 * (hash01(j) - 0.5), y + dd * dd * 1800, scale=sc, rot=j * 70 + 200 + dd * 300,
                alpha=1 - ramp(dd, 0.3, 0.6))
    ws = kt.words(19)
    targets = [(560, 330, 70, -8), (1360, 420, 64, 6), (760, 720, 60, 4)]
    for k_, (x, y, size, rot) in zip((0, 1, 2), targets):
        w = ws[k_]
        if not kt.active(19, t) or t < w.t:
            continue
        u = ease_out(clamp((t - w.t) / 0.22))
        img = kt.wimg(w.s, size, "jpb", INKD)
        fall = max(0.0, t - w.t - 0.9)
        put(c, img, lerp(W / 2, x, u), lerp(1300, y, u) + 600 * fall * fall, scale=lerp(3.5, 1.0, u), rot=rot * u + fall * 40,
            alpha=1 - ramp(fall, 0.4, 0.7))
    word(c, t, 19, 3, W / 2 + 30, 560, 170, "slam", color=INKD, s0=2.5, sh=8, stroke=5, sc=(0.97, 0.97, 0.98))
    return {}


def s_chatcard(c, t, lt, d):
    paper_bg(c, t)
    cx, cy = W / 2, H / 2
    k = ease_out(ramp(lt, 0, 0.2))
    put(c, rect_cached(1000, 380, (0.995, 0.995, 1.0), 10, 1, (0.8, 0.8, 0.82)), cx, cy + 30 * (1 - k), alpha=k)
    L = Layer()
    L.circle((cx - 420, cy - 90), 24, (0.75, 0.75, 0.77))
    L.circle((cx - 420, cy + 70), 24, OR)
    for j in range(3):
        on = (beat_index(t) - j) % 3 == 0
        L.circle((cx - 360 + j * 34, cy + 70), 8, (0.3, 0.3, 0.32) if on else (0.8, 0.8, 0.82))
    L.comp(c, k)
    w = kt.merge(20, 0, 2)
    kt.word_chars(c, w, t, cx - 370, cy - 90, size=58, color=INKD, anchor=(0, 0.5), accent="好き", accent_color=OR)
    return {}


def s_hearts_dark(c, t, lt, d):
    paper_bg(c, t, dark=True)
    w = kt.words(21)[1]
    if t >= w.t:
        put(c, kt.hollow("好き", 520, "jpb", OR, 4), W / 2, H / 2, alpha=0.35 * ease_out(ramp(t, w.t, w.t + 0.3)),
            scale=1 + 0.04 * (t - w.t) + 0.03 * kick_pulse(t))
    for i in range(22):
        x = hash01(i, 1) * W
        y = (hash01(i, 2) * (H + 200) - lt * (60 + 90 * hash01(i, 3))) % (H + 200) - 100
        sz = int(18 + hash01(i, 4) * 46) * (1 + 0.15 * kick_pulse(t))
        col_ = OR if i % 7 == 0 else (0.92, 0.92, 0.93)
        put(c, heart_rgba(64, col_), x, y, scale=sz / 64, rot=(hash01(i, 5) - 0.5) * 40, alpha=0.5 + 0.5 * hash01(i, 6))
    col(c, t, 21, [0], W / 2 + 70, 280, 66, "slide", color=LIGHT, dx=0, dy=-60)
    col(c, t, 21, [1, 2], W / 2 - 50, 230, 66, "slam", color=LIGHT)
    return dict(grain=0.03)


# =====================================================================  FALL
FRAG = ["愛", "唄", "心", "声", "詞", "音", "色", "夢", "あ", "い", "う", "た", "の", "は", "を", "が", "？", "、"]


def _fall_scene(c, t, lt, shot=0):
    draw_clouds(c, t, (0, -420), seed=3, sky=(0.80, 0.81, 0.83), cloud=(0.985, 0.985, 0.99), strength=1.0)
    L = Layer()
    for i in range(26):
        x = hash01(i, 11) * W
        ln = 120 + 260 * hash01(i, 12)
        y = (hash01(i, 13) * (H + 600) - lt * (1400 + 900 * hash01(i, 14))) % (H + 600) - 300
        L.line((x, y), (x, y + ln), (0.97, 0.97, 0.98), 1 + int(hash01(i, 15) * 3))
    L.comp(c, 0.8)
    zoom = [1.0, 1.9, 0.62][shot]
    for i in range(30):
        x = hash01(i, 1, shot) * W
        y = (hash01(i, 2) * (H + 300) - lt * (300 + 500 * hash01(i, 3)) * zoom) % (H + 300) - 150
        g = 0.40 + 0.4 * hash01(i, 4)
        sz = int((22 + 34 * hash01(i, 5)) * (1.5 if shot == 1 else 1))
        put(c, kt.wimg(FRAG[(i + shot * 7) % len(FRAG)], sz, "jpm", (g, g, g + 0.01)), x, y, alpha=0.85)
    f = load_asset("c04_falling", "mono")
    s = 880 / f.shape[0] * zoom
    cy = H / 2 + 20 * math.sin(lt * 0.8) - (f.shape[0] * s * 0.28 if shot == 1 else 0)
    put(c, f, W / 2 + 40 * math.sin(lt * 0.5), cy, scale=s * (1 + 0.012 * math.sin(lt)), rot=6 * math.sin(lt * 0.35))
    draw_clouds(c, t + 3, (0, -900), seed=5, sky=None, cloud=(1, 1, 1),
                strength=0.45 * (0.5 + 0.5 * math.sin(lt * 0.9)))


def s_fall(c, t, lt, d):
    shot = 1 if t >= B(22, 1) else 0
    _fall_scene(c, t, lt, shot)
    ws = kt.words(22)
    if shot == 0:
        w = ws[0]
        if t >= w.t:
            put(c, kt.hollow("ああ", 420, "jpb", GREY, 4), W / 2 - 380, H / 2, alpha=0.4 * (1 - ramp(t, w.t + 1.2, w.t + 2.2)),
                scale=1 + 0.1 * (t - w.t))
        word(c, t, 22, 0, 330, 180, 90, "slam", v=True, anchor=(0.5, 0.0))
        col(c, t, 22, [1], 1560, 160, 66, "slide", dx=0, dy=-80)
        word(c, t, 22, 2, 1440, 240, 110, "slam", v=True, anchor=(0.5, 0.0), sh=5)
    else:
        col(c, t, 22, [3, 4], 420, 150, 80, "slam", color=INKD)
        w = ws[5]
        word(c, t, 22, 5, 300, 300, 80, "slam", v=True, anchor=(0.5, 0.0))
        if t >= w.t:
            k = ease_out(ramp(t, w.t + 0.15, T(23)))
            fill(c, INKD, 1.0, (296, 300 + 160, 304, 300 + 160 + 600 * k))
    return {}


# ---------------------------------------------------------------- 1:46  chant climax inside the GPU
CHANT0, CHANT1 = 106.17, 116.45


def _tunnel(c, t, speed=1.0, glow=1.0):
    c[:] = 0.035
    L = Layer(ss=1)
    phase = t * 0.55 * speed + 0.25 * sum(math.exp(-(t - k) / 0.15) for k in KICKS if 0 <= t - k < 0.6)
    for k in range(22):
        z = ((k / 22 - phase) % 1.0) * 6.0 + 0.25
        s = 1 / z
        hw, hh = 1100 * s, 640 * s
        g = clamp(0.06 + 0.55 * (1 - z / 6.3) ** 2)
        L.rect(W / 2 - hw, H / 2 - hh, W / 2 + hw, H / 2 + hh, (g, g, g * 1.02), max(1, int(5 * s)))
        if k % 3 == 0:
            for q in range(6):
                u = hash01(k, q)
                x = W / 2 - hw + 2 * hw * u
                L.line((x, H / 2 - hh), (x, H / 2 - hh * 0.7), (g, g, g), max(1, int(3 * s)))
                L.line((W / 2 - hw + 2 * hw * hash01(k, q, 3), H / 2 + hh), (W / 2 - hw + 2 * hw * hash01(k, q, 3), H / 2 + hh * 0.7),
                       (g, g, g), max(1, int(3 * s)))
    for q in range(48):
        a = q / 48 * 2 * math.pi + hash01(q) * 0.1
        r0 = ((hash01(q, 9) - t * 1.4 * speed) % 1.0)
        r0 = 60 + 1400 * r0 ** 2
        r1 = r0 + 60 + 300 * (r0 / 1400)
        g = 0.25 + 0.6 * (r0 / 1460)
        col_ = OR if q % 7 == 0 else (g, g, g)
        L.line((W / 2 + r0 * math.cos(a), H / 2 + r0 * math.sin(a)), (W / 2 + r1 * math.cos(a), H / 2 + r1 * math.sin(a)), col_, 2)
    L.comp(c, mode="add")
    hh = heart_rgba(300, OR)
    put(c, hh, W / 2, H / 2, scale=2.2 * glow * (1 + 0.15 * kick_pulse(t)), alpha=0.18, mode="add")


def _shock(c, t, cx=W / 2, cy=H / 2, life=0.6, rmax=1100, color=(1, 0.75, 0.6)):
    L = Layer(ss=1)
    any_ = False
    for tb in STRONG:
        u = (t - tb) / life
        if 0 <= u < 1:
            any_ = True
            L.circle((cx, cy), 60 + rmax * ease_out(u), color, max(1, int(14 * (1 - u))))
    if any_:
        L.comp(c, 0.55, mode="add")


def _o_bursts(c, t, cx=W / 2, cy=H / 2, size=300, scale_max=4.5, life=0.95, a=CHANT0, b=CHANT1, color=LIGHT):
    for tb in downs(a - 0.01, b):
        u = (t - tb) / life
        if not (0 <= u < 1):
            continue
        sc = lerp(0.6, scale_max, ease_out(u))
        put(c, kt.hollow("お", size, "jpb", color, 5), cx, cy, scale=sc, alpha=(1 - u) ** 1.5, mode="add")
        if u < 0.25:
            put(c, kt.wimg("お", size, "jpb", color), cx, cy, scale=lerp(1.4, 1.0, ease_out(u / 0.25)), alpha=1 - u / 0.25)


def _o_slams(c, t, a=CHANT0, b=CHANT1):
    for j, tb in enumerate(strongs(a - 0.01, b)):
        u = (t - tb) / 0.32
        if not (0 <= u < 1):
            continue
        side = -1 if j % 2 else 1
        x = W / 2 + side * (420 + 160 * hash01(j, 1))
        y = H / 2 + (hash01(j, 2) - 0.5) * 360
        size = 520 + 260 * hash01(j, 3)
        col_ = OR if j % 3 == 0 else LIGHT
        img = kt.wimg("お", int(size), "jpb", col_, False, 10, (0.03, 0.03, 0.04))
        put(c, img, x, y, scale=lerp(1.35, 1.0, ease_out(min(1, u * 3))), rot=side * (6 + 8 * hash01(j, 4)),
            alpha=(1 - u) ** 0.7)


def _invert_hits(c, t, a=CHANT0, b=CHANT1, every=4):
    for tb in downs(a - 0.01, b)[::every]:
        if 0 <= t - tb < 0.07:
            c[:] = 1.0 - c


def _core_shot(t):
    ds = downs(CHANT0 - 0.01, 118)
    k = sum(1 for x in ds if x <= t)
    return ((k - 1) // 2) % 3 if k else 0


def _voice_ring(c, t, cx, cy, r0=330, r1=760, alpha=0.9):
    L = Layer(ss=1)
    ai.radial_spectrum(L, cx, cy, r0, r1, t, n=160, col=(0.92, 0.92, 0.94), th=5, accent=OR, gain=1.1, rot=t * 0.15)
    L.comp(c, alpha, mode="add")


def _voice_lines(c, t, y, amp=420, alpha=0.8):
    L = Layer(ss=1)
    e = ai.env(t)
    for j, g in ((2, 0.25), (1, 0.45)):
        L.polyline(ai.wave_pts(t - j * 0.025, 0, W, y, amp * (0.4 + e), n=420, span=0.045, taper=False), (g, g, g), 2)
    L.polyline(ai.wave_pts(t, 0, W, y, amp * (0.4 + e), n=480, span=0.045, taper=False), OR, 4)
    L.comp(c, alpha, mode="add")


def _core_wide(c, t, lt):
    _tunnel(c, t, speed=1.0 + 0.6 * ramp(t, CHANT0, 112))
    _voice_ring(c, t, W / 2, H / 2 + 20)
    sc_img = load_asset("n02_scream", "mono")
    br = 1 + 0.04 * kick_pulse(t)
    dx, dy = kt.shake(t, 5 * kick_pulse(t), 4)
    put(c, sc_img, W / 2 + dx, H / 2 + 40 + dy, scale=900 / sc_img.shape[0] * br, rot=4 * math.sin(lt * 0.7))
    _shock(c, t)


def s_core(c, t, lt, d):
    shot = _core_shot(t)
    if shot == 0:
        _core_wide(c, t, lt)
    elif shot == 1:
        _tunnel(c, t, speed=2.0)
        sc_img = load_asset("n02_scream", "mono")
        s = 900 / sc_img.shape[0] * 2.4 * (1 + 0.05 * kick_pulse(t))
        dx, dy = kt.shake(t, 9 * kick_pulse(t), 8)
        _voice_lines(c, t, H / 2 - 260, amp=380, alpha=0.9)
        put(c, sc_img, W / 2 + dx, H / 2 + dy + 520, scale=s, rot=-3)
        _shock(c, t, cx=W / 2, cy=H / 2 - 160)
    else:
        c[:] = 0.05
        L = Layer(ss=1)
        nb = 24
        m = ai.bands(t, nb)
        cw_, ch_ = W / nb, H / 14
        for gx in range(nb):
            lvl = clamp(m[gx]) ** 1.4 * 14
            for gy in range(14):
                x0, y0 = gx * cw_ + 4, H - (gy + 1) * ch_ + 4
                on = gy < lvl
                top = on and gy >= lvl - 1
                g = 0.88 if on else 0.13 + 0.04 * hash01(gx, gy)
                L.rect(x0, y0, x0 + cw_ - 8, y0 + ch_ - 8, OR if top else (g, g, g + 0.01))
        L.comp(c)
        sc_img = load_asset("n02_scream", "mono")
        put(c, sc_img, W / 2, H / 2 + 40, scale=760 / sc_img.shape[0] * (1 + 0.05 * kick_pulse(t)), alpha=0.92)
        _shock(c, t, color=(1, 0.6, 0.4))
    _invert_hits(c, t, every=8)
    fl = kt.flash_at(t, downs(CHANT0 - 0.01, CHANT1), 0.1, 0.55)
    return dict(grain=0.03, bloom=0.7, flash=fl, ca=2 + 4 * fl, vignette=1.0)


def s_gpu_pull(c, t, lt, d):
    """Pull back: the scream turns out to be inside one GPU die."""
    k = ease_io(ramp(lt, 0.3, 4.4))
    tmp = new_canvas()
    _core_wide(tmp, t, lt + 6)
    c[:] = 0.04
    gpu = load_asset("o02_gpu", "soft")
    gh, gw = gpu.shape[:2]
    die_screen_w = lerp(W * 1.05, 300, k)
    s = die_screen_w / (gw * 0.30)
    dx, dy = kt.shake(t, 8 * kick_pulse(t) * (1 - k * 0.6), 6)
    put(c, gpu, W / 2 + dx, H / 2 + dy, scale=s)
    dw, dh = int(die_screen_w), int(die_screen_w * H / W)
    small = cv2.resize(tmp, (max(2, dw), max(2, dh)), interpolation=cv2.INTER_AREA)
    sa = np.concatenate([small, np.ones_like(small[..., :1])], 2)
    put(c, sa, W / 2 + dx, H / 2 + dy, alpha=1 - 0.2 * k)
    L = Layer(ss=1)
    for tb in STRONG:
        u = (t - tb) / 0.8
        if 0 <= u < 1 and tb < CHANT1:
            L.circle((W / 2 + dx, H / 2 + dy), dw * 0.6 + 1500 * ease_out(u), OR, max(1, int(10 * (1 - u))))
    L.comp(c, 0.6, mode="add")
    _voice_ring(c, t, W / 2 + dx, H / 2 + dy, dw * 0.62, dw * 0.62 + 160 + 340 * k, alpha=0.5 + 0.4 * k)
    put(c, heart_rgba(64, OR), W / 2, H / 2, scale=dw / 64 * 1.2, alpha=0.25 * k, mode="add")
    fl = kt.flash_at(t, downs(112.0, CHANT1), 0.08, 0.4)
    return dict(grain=0.03, bloom=0.5 + 0.3 * k, flash=fl)


def s_server(c, t, lt, d):
    srv = load_asset("o06_server", "soft", 1200)
    s = W / srv.shape[1] * (1.1 + 0.12 * lt / d)
    c[:] = 0
    put(c, srv, W / 2, H / 2, scale=s)
    L = Layer()
    for i in range(70):
        x = W / 2 + (hash01(i, 1) - 0.5) * 1700
        y = H / 2 + (hash01(i, 2) - 0.5) * 700
        if abs(x - W / 2) < 260:
            continue
        if hash01(i, beat_index(t)) > 0.6:
            L.circle((x, y), 2.2, OR if i % 5 == 0 else (0.85, 0.85, 0.9))
    L.comp(c, mode="add")
    put(c, heart_rgba(64, OR), W / 2, H / 2 + 20, scale=0.25 * (1 + 0.4 * kick_pulse(t)), alpha=0.8 * (1 - ramp(lt, d - 1.5, d)),
        mode="add")
    return dict(grain=0.04, vignette=1.2, fade_black=ramp(lt, d - 0.7, d))


def s_break(c, t, lt, d):
    c[:] = 0.02
    k = ramp(lt, 0.8, d - 1.2)
    progress_bar(c, W / 2, 545, 700, ease_io(k), col=(0.85, 0.85, 0.86), bg=(0.2, 0.2, 0.21), th=3)
    ht(c, "%d%%" % int(k * 100), W / 2, 505, size=16, font="mono", color=(0.6, 0.6, 0.62), alpha=appear(lt, 0.4))
    if lt > d - 1.0 and beat_index(t, 2) % 2:
        c *= 0.3
    return dict(grain=0.02, vignette=0.8)


# =====================================================================  VERSE 2 : searching
def s_door(c, t, lt, d):
    paper_bg(c, t)
    door = load_asset("o04_door", "gray")
    s = 760 / door.shape[0]
    zoom = 1 + 0.35 * ease_in(ramp(lt, 0.8, d))
    put(c, door, W / 2, 560, scale=s * zoom)
    k = ease_out(ramp(lt, 0.0, 0.3))
    fill(c, (0.13, 0.13, 0.14), 1, (W / 2 - 640 * k, 515, W / 2 + 640 * k, 605))
    word(c, t, 24, 0, W / 2 - 420, 400, 120, "slam", sh=4)
    row(c, t, 24, [1, 2], W / 2 + 60, 560, 54, "slide", color=LIGHT, dx=-140)
    return {}


def s_profile(c, t, lt, d):
    paper_bg(c, t)
    w = kt.sub(24, "誰")
    if t >= w.t:
        put(c, kt.hollow("誰", 640, "jpb", OR, 5), 1240, 560, alpha=0.4 * ease_out(ramp(t, w.t, w.t + 0.2)),
            scale=1 + 0.05 * (t - w.t) + 0.03 * kick_pulse(t))
    p = load_asset("c06_profile", "mono")
    s = 980 / p.shape[0]
    put(c, p, 1300, 600, scale=s * (1 + 0.012 * lt))
    scan = ((t * 1.6) % 1.0)
    fill(c, OR, 0.5, (0, 100 + scan * 880, W, 102 + scan * 880))
    col(c, t, 24, [3, 4], 700, 160, 66, "slide", dx=0, dy=-60)
    col(c, t, 24, [5], 580, 200, 66, "slam")
    col(c, t, 24, [6, 7], 430, 240, 96, "slam", sh=6)
    return {}


def s_lock(c, t, lt, d):
    paper_bg(c, t)
    L = Layer()
    tb = B(25, 1) - 0.25
    br = ease_out(ramp(t, tb, tb + 0.25))
    chains = [((0, 200), (W / 2, 520)), ((W, 200), (W / 2, 520)), ((W / 2 - 300, 0), (W / 2, 520)), ((W / 2 + 300, 0), (W / 2, 520))]
    for j, (p0, p1) in enumerate(chains):
        q1 = (lerp(p1[0], p0[0], 0.4 * br) + (j - 1.5) * 90 * br, p1[1] + 400 * br * br)
        chain_links(L, p0, q1, 30, (0.5, 0.5, 0.52), 3)
    L.rect(W / 2 - 150, 440, W / 2 + 150, 760, (0.97, 0.97, 0.98))
    L.rect(W / 2 - 150, 440, W / 2 + 150, 760, (0.75, 0.75, 0.77), 1)
    L.arc((W / 2, 520 - 30 * br), 34, 180, 360, (0.15, 0.15, 0.16), 9)
    L.rect(W / 2 - 46, 516, W / 2 + 46, 590, (0.15, 0.15, 0.16))
    L.comp(c)
    ws = kt.words(25)
    for j, (k_, (p0, p1)) in enumerate(zip((0, 1, 2, 3), [chains[0], chains[2], chains[3], chains[1]])):
        w = ws[k_]
        if not kt.active(25, t) or t < w.t:
            continue
        x0, y0 = p0
        x1, y1 = p1
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
        mx, my = lerp(p0[0], p1[0], 0.45), lerp(p0[1], p1[1], 0.45) - 40
        img = kt.wimg(w.s, 66, "jpb", INKD, False, 5, (0.96, 0.96, 0.97))
        if t < tb:
            u = ease_out(clamp((t - w.t) / 0.12))
            put(c, img, mx, my, rot=ang, scale=lerp(1.6, 1.0, u), alpha=clamp(u * 3))
        else:
            kt.shatter(c, img, mx, my, t, tb + 0.05, n=4, force=900, seed=j)
    return dict(flash=0.45 * (1 - ramp(t, tb + 0.05, tb + 0.25)) if t > tb + 0.05 else 0, ca=3 * br * (1 - ramp(t, tb, tb + 0.5)))


def s_film(c, t, lt, d):
    paper_bg(c, t)
    names = ["c01_stand_front", "c05_back", "c10_reach", "c09_knees", "c01_stand_front"]
    fw, fh = 300, 260
    spd = 420 + 500 * ramp(t, kt.words(25)[7].t, kt.words(25)[7].t + 0.3)
    off = (lt * spd) % fh
    L = Layer()
    L.rect(W / 2 - fw / 2 - 60, -10, W / 2 + fw / 2 + 60, H + 10, (0.13, 0.13, 0.14))
    for i in range(-1, 6):
        y = i * fh + off - fh / 2
        for side in (-1, 1):
            for j in range(4):
                x = W / 2 + side * (fw / 2 + 30)
                L.rect(x - 10, y + j * 62 - 18, x + 10, y + j * 62 + 14, (0.95, 0.95, 0.96))
        L.rect(W / 2 - fw / 2, y - fh / 2 + 14, W / 2 + fw / 2, y + fh / 2 - 14, (0.94, 0.94, 0.95))
    L.comp(c)
    for i in range(-1, 6):
        y = i * fh + off - fh / 2
        n = names[(i - int((lt * spd) // fh)) % len(names)]
        px = pixelize(n, 44 if n != "c10_reach" else 36)
        big = big_nearest(px, 5)
        put(c, big, W / 2, y, scale=min(1, (fh - 40) / big.shape[0]))
    word(c, t, 25, 4, 440, 200, 70, "slam", v=True, anchor=(0.5, 0.0))
    col(c, t, 25, [5], 320, 250, 60, "slide", dx=0, dy=-60)
    col(c, t, 25, [6], 1450, 130, 80, "slam")
    word(c, t, 25, 7, 1620, 160, 150, "slam", v=True, anchor=(0.5, 0.0), sh=8, s0=2.2)
    return {}


def _speed_lines(c, t, n=30, col_=(0.6, 0.6, 0.62), speed=2600, y0=0, y1=H, seed=0):
    L = Layer(ss=1)
    for i in range(n):
        y = y0 + hash01(i, seed) * (y1 - y0)
        ln = 200 + 500 * hash01(i, seed, 1)
        x = W + 400 - ((t * speed * (0.6 + hash01(i, seed, 2)) + hash01(i, seed, 3) * 3000) % (W + 1200))
        L.line((x, y), (x + ln, y), col_, 1 + int(3 * hash01(i, seed, 4)))
    L.comp(c)


def s_run(c, t, lt, d):
    paper_bg(c, t, tone=-0.04)
    L = Layer()
    for i in range(8):
        x = W + 300 - ((lt * 1400 + i * 360) % (W + 600))
        g = 0.72 + 0.1 * hash01(i)
        L.rect(x, 120, x + 120 + 60 * hash01(i, 2), 980, (g, g, g + 0.01))
    L.comp(c, 0.8)
    _speed_lines(c, t, 26)
    r = load_asset("n03_run", "mono")
    dx, dy = kt.shake(t, 5, 3)
    bob = abs(math.sin(beat_pos(t) * math.pi)) * -18
    put(c, r, W / 2 - 120 + dx, 600 + bob + dy, scale=1450 / r.shape[1])
    ws = kt.words(26)
    word(c, t, 26, 0, 1500, 220, 70, "slide", dx=500)
    word(c, t, 26, 1, 1560, 380, 170, "slam", sh=6)
    word(c, t, 26, 2, 1520, 900, 74, "slide", dx=500)
    return dict(ca=1.5)


def s_search(c, t, lt, d):
    paper_bg(c, t, tone=-0.06)
    k8 = beat_index(t, 2)
    cards = []
    for i in range(34):
        x = 120 + hash01(i, 1) * (W - 240)
        y = 120 + hash01(i, 2) * (H - 240)
        if abs(x - W / 2) < 380 and y > 420:
            continue
        g = 0.25 + 0.7 * hash01(i, 3)
        cards.append((x, y, g, (hash01(i, 4) - 0.5) * 40))
    for (x, y, g, r) in cards:
        put(c, rect_cached(110, 80, (g, g, g + 0.01), 0, 1, (0.4, 0.4, 0.42)), x, y, rot=r)
    sr = load_asset("n07_search", "mono")
    dx, dy = kt.shake(t, 3 + 5 * kick_pulse(t), 2)
    put(c, sr, W / 2 + dx, 760 + dy, scale=1300 / sr.shape[1])
    tq = kt.words(26)[6].t
    j = int(hash01(k8, 7) * len(cards))
    x, y, g, r = cards[j]
    L = Layer()
    L.rect(x - 70, y - 54, x + 70, y + 54, OR if t >= tq and k8 % 2 else (0.1, 0.1, 0.11), 3)
    L.line((x - 90, y), (x - 74, y), (0.1, 0.1, 0.11), 2)
    L.line((x + 74, y), (x + 90, y), (0.1, 0.1, 0.11), 2)
    L.comp(c)
    word(c, t, 26, 3, 330, 220, 110, "slam")
    word(c, t, 26, (4, 5), 640, 220, 66, "slide", dx=-200)
    w = kt.words(26)[6]
    if kt.active(26, t) and t >= w.t:
        cols = [INKD, OR, (0.55, 0.55, 0.57), (0.8, 0.8, 0.82)]
        cc = cols[k8 % 4] if t < w.t + 0.6 else INKD
        img = kt.wimg(w.s, 200, "jpb", cc, False, 6, (0.97, 0.97, 0.98))
        if t < w.t + 0.12:
            kt.slam(c, img, 1380, 260, t, w.t, s0=2.2, sh=10)
        else:
            kt.glitch(c, img, 1380, 260, t, amt=0.5, seed=26)
    return dict(ca=2)


def s_overflow(c, t, lt, d):
    paper_bg(c, t)
    ws = kt.words(27)
    tb = ws[2].t
    bw0, bh0 = 820, 360
    sq = ease_in(ramp(t, T(27), tb))
    bw, bh = bw0 * (1 - 0.3 * sq), bh0 * (1 - 0.35 * sq)
    cx, cy = W / 2, H / 2 - 30
    L = Layer(ss=1)
    if t < tb:
        grow = 1 - 0.04 * kick_pulse(t)
        jx, jy = kt.shake(t, 2 + 10 * sq, 27)
        L.rect(cx - bw / 2 * grow + jx, cy - bh / 2 * grow + jy, cx + bw / 2 * grow + jx, cy + bh / 2 * grow + jy, INKD, 6)
        L.comp(c)
    else:
        u = t - tb
        edges = [((cx - bw / 2, cy - bh / 2), (cx + bw / 2, cy - bh / 2), (0, -1)), ((cx + bw / 2, cy - bh / 2), (cx + bw / 2, cy + bh / 2), (1, 0)),
                 ((cx + bw / 2, cy + bh / 2), (cx - bw / 2, cy + bh / 2), (0, 1)), ((cx - bw / 2, cy + bh / 2), (cx - bw / 2, cy - bh / 2), (-1, 0))]
        for e, (a, b, (nx, ny)) in enumerate(edges):
            for q in range(4):
                p0 = (lerp(a[0], b[0], q / 4), lerp(a[1], b[1], q / 4))
                p1 = (lerp(a[0], b[0], (q + 1) / 4), lerp(a[1], b[1], (q + 1) / 4))
                sp = 900 * (0.5 + hash01(e, q))
                ox, oy = nx * sp * u + (hash01(e, q, 2) - 0.5) * 300 * u, ny * sp * u + 900 * u * u
                ang = (hash01(e, q, 3) - 0.5) * 6 * u
                mx, my = (p0[0] + p1[0]) / 2 + ox, (p0[1] + p1[1]) / 2 + oy
                hx, hy = (p1[0] - p0[0]) / 2, (p1[1] - p0[1]) / 2
                ca, sa = math.cos(ang), math.sin(ang)
                L.line((mx - hx * ca + hy * sa, my - hx * sa - hy * ca), (mx + hx * ca - hy * sa, my + hx * sa + hy * ca), INKD, 6)
        L.comp(c, 1 - ramp(u, 0.5, 0.9))
    nv = int(8 * 7 * clamp(ramp(t, ws[1].t, tb) * 1.1))
    for j in range(8 * 7 if t >= tb else nv):
        r, q = j // 7, j % 7
        x, y = cx - bw0 / 2 + 60 + q * 150, cy - bh0 / 2 + 30 + r * 44
        a = 0.55 + 0.45 * hash01(j, beat_index(t, 2))
        if t >= tb:
            u = t - tb
            vx, vy = x - cx, y - cy
            nrm = math.hypot(vx, vy) + 1
            x += vx / nrm * 1400 * u * (0.5 + hash01(j, 4))
            y += vy / nrm * 900 * u + 700 * u * u
            a *= 1 - ramp(u, 0.3, 0.6)
        ht(c, f"{(hash01(j, 27) - 0.5) * 2.4:+.3f}", x, y, size=22, font="mono", color=OR if j % 11 == 3 else MID,
           alpha=a, anchor=(0, 0.5))
    if t >= ws[0].t and t < tb:
        for k_ in range(5):
            x = cx - bw0 / 2 + 40 + k_ * 100
            put(c, rect_cached(90, 26, (0.97, 0.97, 0.98), 0, 1, (0.5, 0.5, 0.52)), x + 45, cy - 6,
                alpha=appear(t, ws[0].t + k_ * 0.05, 0.1))
            ht(c, str(int(1000 + hash01(k_, 45) * 99000)), x + 45, cy - 6, size=15, font="mono", color=INKD,
               alpha=appear(t, ws[0].t + k_ * 0.05, 0.1))
    word(c, t, 27, 0, cx - bw0 / 2 + 40, cy - 70, 96, "slam", anchor=(0, 0.5), ghost=0.2)
    word(c, t, 27, 1, cx - bw0 / 2 + 40, cy + 60, 96, "slide", anchor=(0, 0.5), dx=-120)
    w = ws[2]
    if kt.active(27, t, 0) and t >= w.t:
        img = kt.wimg(w.s, 300, "jpb", INKD, False, 7, (0.96, 0.96, 0.97))
        kt.slam(c, img, W / 2, cy + 250, t, w.t, s0=2.8, sh=14, ghost=0.5)
    fl = kt.flash_at(t, [tb], 0.15, 0.6)
    return dict(flash=fl, ca=3 * fl + 1)


def s_hug(c, t, lt, d):
    paper_bg(c, t, tone=-0.05)
    ws = kt.words(27)
    w0 = ws[3]
    if t >= w0.t:
        hb = 1 + 0.06 * kick_pulse(t, 0.12)
        put(c, kt.hollow("心臓", 560, "jpb", OR, 5), W / 2, H / 2 - 20, alpha=0.45 * ease_out(ramp(t, w0.t, w0.t + 0.3)), scale=hb)
    hg = load_asset("n04_hugheart", "mono")
    hb = 1 + 0.025 * kick_pulse(t, 0.12)
    dx, dy = kt.shake(t, 4 * kick_pulse(t), 9)
    put(c, hg, W / 2 + dx, 560 + dy, scale=900 / hg.shape[0] * hb)
    put(c, heart_rgba(64, OR), W / 2 - 10, 470, scale=3.5 * hb, alpha=0.18 + 0.2 * kick_pulse(t), mode="add")
    hp = heart_outline_pts(W / 2 - 10, 480, 150 * hb)
    Ld = Layer(ss=1)
    g = ease_io(clamp(lt / 1.2))
    for j in range(0, len(hp), 2):
        a0 = hash01(j, 1) * 6.283 + lt * (0.6 + hash01(j, 2))
        r0 = 380 + 300 * hash01(j, 3)
        sx, sy = W / 2 + r0 * math.cos(a0), 480 + r0 * 0.6 * math.sin(a0)
        Ld.circle((lerp(sx, hp[j][0], g), lerp(sy, hp[j][1], g)), 4, OR if j % 6 == 0 else (0.4, 0.4, 0.42))
    Ld.comp(c, 0.85)
    word(c, t, 27, 3, 380, 300, 100, "slam", sh=4)
    word(c, t, 27, 4, W - 380, 300, 100, "slam", sh=4)
    w = ws[5]
    word(c, t, 27, 5, W / 2, 960, 130, "slam", s0=2.4, sh=12, stroke=6, sc=(0.96, 0.96, 0.97))
    fl = kt.flash_at(t, [w.t], 0.15, 0.55)
    return dict(flash=fl, ca=2 * fl, bloom=0.4)


# =====================================================================  RAP
def s_doppel(c, t, lt, d):
    paper_bg(c, t, tone=-0.06)
    w_over = kt.words(28)[4]
    if t >= w_over.t:
        kt.echo_bg(c, "ドッペルもどき", t, size=54, color=(0.45, 0.45, 0.47), alpha=0.18 * ease_out(ramp(t, w_over.t, w_over.t + 0.3)),
                   speed=260, gap=1.35)
    sil = silhouette("c01_stand_front", (0.5, 0.5, 0.52), 600)
    real = load_asset("c01_stand_front", "mono")
    kp = kick_pulse(t, 0.1)
    for r_, (yy, sc, g) in enumerate([(430, 0.36, 0.45), (560, 0.5, 0.65), (720, 0.68, 0.9)]):
        n = [11, 9, 7][r_]
        for j in range(n):
            x = (j + 0.5) / n * W + (r_ % 2) * 40
            if r_ == 2 and j == n // 2:
                continue
            sway = math.sin(beat_pos(t) * math.pi + j + r_) * 6
            put(c, sil, x + sway, yy - 8 * kp * (1 if (j + r_) % 2 else 0), scale=sc * 900 / 600 * 0.55, alpha=g)
    put(c, real, W / 2, 720, scale=0.68 * 0.55 * 900 / real.shape[0] * (1 + 0.02 * kp))
    cur = LY.current(t, 28, 30)
    inv = 0.0
    if cur is not None:
        i = cur["idx"]
        if i == 28:
            word(c, t, 28, 0, 560, 200, 150, "slam", sh=6, stroke=6, sc=(0.95, 0.95, 0.96))
            word(c, t, 28, 1, 1080, 200, 90, "slam", stroke=5, sc=(0.95, 0.95, 0.96))
            word(c, t, 28, (2, 3), 700, 960, 70, "slide", dx=-300, color=LIGHT)
            word(c, t, 28, 4, 1250, 960, 110, "slam", sh=8, color=OR)
        elif i == 29:
            ws = kt.words(29)
            for j, (x, y, size) in enumerate([(560, 230, 70), (1000, 230, 70), (1450, 230, 140)]):
                w = ws[j]
                if t >= w.t:
                    wd, _ = kt.wsize(w.s, size)
                    kt.invert_box(c, x, y, wd + 50, size * 1.5, t, w.t - 0.02)
                word(c, t, 29, j, x, y, size, "slam", color=LIGHT, ghost=0.2)
        else:
            for k, w in enumerate(kt.words(30)):
                if t >= w.t:
                    kt.slam(c, kt.wimg(w.s, 170, "jpb", INKD, False, 7, (0.97, 0.97, 0.98)), W / 2 + (k - 1) * 520, 230, t, w.t,
                            s0=2.6, sh=8)
                if w.t <= t < w.t + 0.07:
                    inv = 1.0
    if inv:
        c[:] = 1.0 - c
    return {}


def s_server2(c, t, lt, d):
    srv = load_asset("o06_server", "soft", 1200)
    s = W / srv.shape[1] * (1.25 - 0.08 * lt / d)
    c[:] = 0
    put(c, srv, W / 2 + 60 * lt / d, H / 2, scale=s)
    L = Layer()
    for i in range(80):
        x = W / 2 + (hash01(i, 1) - 0.5) * 1700
        y = H / 2 + (hash01(i, 2) - 0.5) * 700
        if hash01(i, beat_index(t, 2)) > 0.55:
            L.circle((x, y), 2.4, OR if i % 5 == 0 else (0.85, 0.85, 0.9))
    L.comp(c, mode="add")
    cur = LY.current(t, 31, 33)
    if cur is not None:
        i = cur["idx"]
        if i == 33:
            for k, w in enumerate(kt.words(33)):
                if w.t <= t < w.t + 0.12:
                    c *= 0.2
                if t >= w.t:
                    kt.slam(c, kt.wimg(w.s, 160, "jpb", LIGHT), W / 2 + (k - 1) * 560, H / 2, t, w.t, s0=2.5, sh=10)
        elif i == 31:
            word(c, t, 31, (0, 2), 1640, 120, 80, "slam", color=LIGHT, v=True, anchor=(0.5, 0.0))
            col(c, t, 31, [3, 4], 1500, 160, 66, "slide", color=LIGHT, dx=0, dy=-60)
        else:
            word(c, t, 32, (0, 1), 320, 200, 76, "slam", color=LIGHT, v=True, anchor=(0.5, 0.0))
            word(c, t, 32, 2, 200, 300, 120, "slam", color=OR, v=True, anchor=(0.5, 0.0), sh=6)
    return dict(grain=0.04, vignette=1.1)


def s_hands(c, t, lt, d):
    paper_bg(c, t, dark=True)
    L = Layer()
    L.line((0, 560), (W, 560), (0.55, 0.55, 0.57), 1)
    L.comp(c)
    hnd = load_asset("c07_hands", "gray")
    s = 1100 / hnd.shape[1]
    put(c, hnd * 0.55, W / 2, 1080 - (hnd.shape[0] * s) / 2 + 120 - 60 * ease_out(ramp(lt, 0, 0.6)), scale=s)
    put(c, heart_rgba(48, (0.92, 0.92, 0.93)), W / 2, 330 + 6 * math.sin(lt * 3), alpha=appear(lt, 0.4),
        scale=1 + 0.15 * kick_pulse(t))
    cur = LY.current(t, 34, 36)
    if cur is not None:
        i = cur["idx"]
        if i == 34:
            row(c, t, 34, [0, 1], W / 2, 170, 64, "slam", color=LIGHT, ghost=0.2)
            row(c, t, 34, [2, 3, 4], W / 2, 270, 52, "slide", color=LIGHT, dx=-200)
        elif i == 35:
            word(c, t, 35, 0, W / 2 - 220, 200, 80, "slam", color=LIGHT)
            word(c, t, 35, 1, W / 2 + 150, 200, 80, "wipe", color=LIGHT, dur=0.5)
        else:
            w1 = kt.merge(36, 0, 1)
            w2 = kt.merge(36, 2, 3)
            if t >= w1.t:
                kt.slam(c, kt.wimg(w1.s, 120, "jpb", LIGHT), 520, 200, t, w1.t, s0=2.0, sh=6)
            if t >= w2.t:
                img = kt.wimg(w2.s, 190, "jpb", OR)
                if t < w2.t + 0.12:
                    kt.slam(c, img, 1320, 240, t, w2.t, s0=2.4, sh=12)
                else:
                    kt.glitch(c, img, 1320, 240, t, amt=0.9, seed=36)
    return dict(grain=0.03, flash=kt.flash_at(t, [w.t for w in kt.words(36)[::2]], 0.1, 0.4))


def s_crowd(c, t, lt, d):
    paper_bg(c, t)
    cr = load_asset("o01_crowd", "gray")
    crf = cr[:, ::-1].copy()
    z = 1 + 0.03 * lt
    for (img, sc, x, y, a, sp) in [(cr, 0.42, W / 2 - 420, 520, 0.22, 6), (crf, 0.42, W / 2 + 420, 525, 0.22, 6),
                                   (cr, 0.42, W / 2, 515, 0.18, 4)]:
        put(c, img, x + math.sin(lt * 0.2 + x) * sp, y, scale=W / cr.shape[1] * sc * z, alpha=a)
    fill(c, (0.93, 0.93, 0.94), 0.25)
    for (img, sc, x, y, a, sp, an) in [(crf, 0.62, W * 0.42, 610, 0.38, 10, (1, 0.5)),
                                       (cr, 0.62, W * 0.58, 615, 0.38, 10, (0, 0.5))]:
        put(c, img, x + math.sin(lt * 0.25 + y) * sp, y, scale=W / cr.shape[1] * sc * z, alpha=a, anchor=an)
    ch = load_asset("c01_stand_front", "mono")
    put(c, ch, W / 2 + 10, 640, scale=470 / ch.shape[0] * z, alpha=appear(lt, 0.2, 0.8))
    for (img, sc, x, y, a, sp, an) in [(cr, 0.95, W * 0.30, 860, 0.6, 18, (1, 0.5)),
                                       (crf, 0.95, W * 0.70, 870, 0.6, 18, (0, 0.5))]:
        put(c, img, x + math.sin(lt * 0.3 + y) * sp, y, scale=W / cr.shape[1] * sc * z, alpha=a, anchor=an)
    STK = dict(stroke=6, sc=(0.96, 0.96, 0.97))
    if t < T(38):
        word(c, t, 37, 0, W / 2 - 330, 200, 90, "slam", **STK)
        word(c, t, 37, 1, W / 2 + 200, 200, 130, "slam", sh=6, **STK)
    else:
        row(c, t, 38, [0, 1, 2], W / 2, 150, 62, "slide", dx=-250, **STK)
        word(c, t, 38, 3, W / 2, 270, 80, "slam", **STK)
        w = kt.words(38)[4]
        word(c, t, 38, 4, W / 2, 960, 110, "slam", s0=2.2, sh=10, **STK)
    return dict(flash=kt.flash_at(t, [kt.words(38)[4].t], 0.12, 0.4))


# =====================================================================  B2 : drafts / clock / blocks
_BALL_BOXES = None


def _balls():
    global _BALL_BOXES
    img = load_asset("o05_paperballs", "gray")
    if _BALL_BOXES is None:
        a = (img[..., 3] > 0.1).astype(np.uint8)
        n, lab, stats, cen = cv2.connectedComponentsWithStats(a)
        boxes = [tuple(s[:4]) for s in stats[1:] if s[4] > 2000]
        boxes.sort()
        _BALL_BOXES = boxes
    return [img[y:y + h, x:x + w] for (x, y, w, h) in _BALL_BOXES]


def s_drafts(c, t, lt, d):
    paper_bg(c, t)
    L = Layer()
    _pencil(L, 600, 300, 200, 480)
    L.comp(c)
    balls = _balls()
    spots = [(420, 760), (1180, 820), (800, 880), (1000, 640), (560, 560), (1300, 480), (300, 380), (900, 400)]
    drops = [x for x in STRONG if T(39) <= x < T(40)][::2]
    for i, (x, y) in enumerate(spots):
        if i >= len(drops) or t < drops[i] - 0.25 or not balls:
            continue
        k = ease_in(ramp(t, drops[i] - 0.25, drops[i]))
        b = balls[i % len(balls)]
        sc = (260 + 80 * hash01(i)) / max(b.shape[:2])
        yy = y - 500 * (1 - k)
        bounce = 0.0 if t < drops[i] else 30 * math.exp(-(t - drops[i]) * 9) * abs(math.sin((t - drops[i]) * 25))
        put(c, b, x, yy - bounce, scale=sc, rot=hash01(i, 2) * 90 * (1 - k))
    word(c, t, 39, 0, 1680, 130, 76, "wipe", v=True, anchor=(0.5, 0.0), dur=0.4)
    word(c, t, 39, 1, 1560, 200, 60, "wipe", v=True, anchor=(0.5, 0.0), dur=0.4)
    word(c, t, 39, 2, 1440, 160, 60, "wipe", v=True, anchor=(0.5, 0.0), dur=0.8)
    return {}


def s_clock(c, t, lt, d):
    paper_bg(c, t)
    nums = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    R = 300
    cx = W / 2
    for i, nm in enumerate(nums):
        a = math.radians(i * 30 - 90)
        ht(c, nm, cx + R * math.cos(a), H / 2 + R * math.sin(a), size=32, font="serif", color=GREY)
    w0 = kt.words(40)[0]
    spin = lt * 1.2 + 2.0 * ease_in(ramp(t, w0.t, kt.words(40)[3].t))
    if t >= w0.t:
        kt.along_circle(c, "何千回と巡らせ続けた　", cx, H / 2, 420, spin, size=44, font="jpm", color=INKD,
                        alpha=ease_out(ramp(t, w0.t, w0.t + 0.3)))
    L = Layer()
    bp = beat_pos(t)
    a1 = math.radians((math.floor(bp) + ease_out(clamp((bp % 1) * 3))) * 30 - 90)
    a2 = math.radians(lt * 75 - 90)
    L.line((cx, H / 2), (cx + 270 * math.cos(a1), H / 2 + 270 * math.sin(a1)), (0.2, 0.2, 0.22), 4)
    L.line((cx, H / 2), (cx + 180 * math.cos(a2), H / 2 + 180 * math.sin(a2)), (0.2, 0.2, 0.22), 9)
    L.comp(c)
    put(c, heart_rgba(70, OR), cx, H / 2, scale=0.9 + 0.25 * kick_pulse(t))
    s = LYS[40]["ja"]
    for ch, (x, y) in zip("喜怒哀楽", [(cx, H / 2 - 200), (cx + 200, H / 2), (cx, H / 2 + 200), (cx - 200, H / 2)]):
        j = s.find(ch)
        ta = LYS[40]["chars"][j]
        if kt.active(40, t) and t >= ta:
            kt.slam(c, kt.wimg(ch, 150, "jpb", OR if ch in "喜楽" else INKD, False, 6, (0.97, 0.97, 0.98)), x, y, t, ta,
                    s0=2.4, sh=6)
    return {}


def s_blocks(c, t, lt, d):
    paper_bg(c, t)
    slabs = [(41, 0), (41, 1), (42, 0), (42, 1), (43, 0)]
    size = 44
    L = Layer()
    geo = []
    y = 990
    for j, (i, k) in enumerate(slabs):
        ws = pw(i, k)
        s = kt.join(ws)
        t0 = ws[0].t - 0.12
        wd, _ = kt.wsize(s, size)
        bw, bh = wd + 70, 100
        y -= bh / 2
        x = W / 2 - 200 + (hash01(j) - 0.5) * 80
        kk = ease_out(ramp(t, t0, t0 + 0.18))
        geo.append((i, k, ws, x, y, bw, bh, kk, wd))
        y -= bh / 2 + 4
    flick = ease_in(ramp(t, kt.sub(43, "つまは").t, kt.sub(43, "つまは").t + 0.4))
    for (i, k, ws, x, y, bw, bh, kk, wd) in geo:
        if kk <= 0:
            continue
        yy = y - 500 * (1 - kk)
        xo = 900 * flick if (i, k) == (42, 1) else 0
        g = 0.9 - 0.05 * k
        x_ = x + xo
        L.rect(x_ - bw / 2, yy - bh / 2, x_ + bw / 2, yy + bh / 2, (g, g * 0.98, g * 0.95))
        L.poly([(x_ + bw / 2, yy - bh / 2), (x_ + bw / 2 + 24, yy - bh / 2 - 18), (x_ + bw / 2 + 24, yy + bh / 2 - 18),
                (x_ + bw / 2, yy + bh / 2)], (g * 0.7, g * 0.68, g * 0.65))
        L.poly([(x_ - bw / 2, yy - bh / 2), (x_ - bw / 2 + 24, yy - bh / 2 - 18), (x_ + bw / 2 + 24, yy - bh / 2 - 18),
                (x_ + bw / 2, yy - bh / 2)], (min(1, g * 1.08),) * 3)
    L.comp(c)
    for (i, k, ws, x, y, bw, bh, kk, wd) in geo:
        if kk <= 0:
            continue
        xo = 900 * flick if (i, k) == (42, 1) else 0
        yy = y - 500 * (1 - kk)
        xx = x + xo - wd / 2
        for w in ws:
            ww, _ = kt.wsize(w.s, size)
            word(c, t, i, w, xx + ww / 2, yy, size, "slam", keep=99, ghost=0.15, s0=1.4)
            xx += ww + size * 0.04
    return {}


def s_bigtext(c, t, lt, d):
    paper_bg(c, t)
    for i in range(8):
        k = ease_in(ramp(lt, 0, d))
        x = W / 2 + (hash01(i) - 0.5) * 900
        y = 500 + (hash01(i, 2) - 0.5) * 300 + k * 800
        put(c, rect_cached(int(160 + 160 * hash01(i, 3)), int(60 + 30 * hash01(i, 4)), (0.85, 0.84, 0.82)), x, y,
            rot=hash01(i, 5) * 260 * k)
    ws = kt.words(44)
    word(c, t, 44, 0, W / 2 - 430, H / 2 - 60, 130, "slam", sh=6)
    w = ws[1]
    if t >= w.t:
        img = kt.wimg(w.s, 280, "jpb", INKD)
        if t < w.t + 0.15:
            kt.slam(c, img, W / 2 + 230, H / 2 + 60, t, w.t, s0=2.8, sh=16)
        else:
            dx, dy = kt.shake(t, 6, 44)
            kt.glitch(c, img, W / 2 + 230 + dx, H / 2 + 60 + dy, t, amt=0.6, seed=44)
    fl = kt.flash_at(t, [w.t], 0.12, 0.6)
    return dict(ca=1 + 4 * fl, flash=fl)


# =====================================================================  CHORUS 2 : messages (letterbox)
def s_messages(c, t, lt, d):
    paper_bg(c, t, dark=False, tone=-0.12)
    y = 250
    size = 46
    for i in (45, 46, 47, 48):
        if t < T(i) - 0.05:
            break
        k = ease_out(ramp(t, T(i) - 0.05, T(i) + 0.2))
        w = kt.merge(i, 0, len(kt.words(i)) - 1)
        wd, _ = kt.wsize(LYS[i]["ja"], size)
        bw, bh = int(wd + 70), 84
        bx = 330 + bw / 2
        put(c, bubble_rgba(bw, bh, (0.97, 0.97, 0.98), "left", 16), bx, y + 20 * (1 - k), alpha=k)
        LY.draw_chars(c, LYS[i], list(range(len(LYS[i]["ja"]))), t, bx - 20, y - 10 + 20 * (1 - k), size=size, color=GREY,
                      accent="愛", accent_color=OR, alpha=k)
        y += 124
    if t >= T(49) - 0.05:
        k = ease_out(ramp(t, T(49) - 0.05, T(49) + 0.25))
        s = LYS[49]["ja"]
        half = len(s) // 2
        put(c, bubble_rgba(640, 190, (0.99, 0.97, 0.96), "right", 16, OR, 2), 1400, 800, alpha=k)
        LY.draw_chars(c, LYS[49], list(range(half)), t, 1110, 755, size=size, color=GREY, anchor=(0, 0.5))
        LY.draw_chars(c, LYS[49], list(range(half, len(s))), t, 1110, 825, size=size, color=GREY, anchor=(0, 0.5))
    return dict(letterbox=1.0, vignette=1.3)


# =====================================================================  FINAL CHORUS
def _window(L, cx, cy, w, h, col_=(0.22, 0.22, 0.24), th=14):
    L.rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, col_, th)
    L.line((cx, cy - h / 2), (cx, cy + h / 2), col_, th // 2)


def _window_bg(c, t, lt):
    paper_bg(c, t)
    tmp = new_canvas((0.82, 0.83, 0.85))
    draw_clouds(tmp, t, (25, 0), seed=7, sky=(0.78, 0.8, 0.83), cloud=(0.98, 0.98, 0.99), strength=1.0)
    ww, wh = 980, 560
    zoom = 1 + 0.04 * lt
    x0, y0 = int(W / 2 - ww * zoom / 2), int(H / 2 - wh * zoom / 2)
    c[y0:y0 + int(wh * zoom), x0:x0 + int(ww * zoom)] = tmp[y0:y0 + int(wh * zoom), x0:x0 + int(ww * zoom)]
    L = Layer()
    _window(L, W / 2, H / 2, ww * zoom, wh * zoom)
    L.comp(c)
    return ww * zoom


def s_window(c, t, lt, d):
    ww = _window_bg(c, t, lt)
    word(c, t, 50, 0, W / 2 - ww / 2 - 130, 260, 66, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=-60)
    word(c, t, 50, 1, W / 2 + ww / 2 + 130, 300, 66, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=60)
    word(c, t, 50, 2, W / 2, H / 2, 230, "slam", s0=2.4, sh=10, stroke=7, sc=(0.96, 0.96, 0.97))
    return dict(flash=kt.flash_at(t, [kt.words(50)[2].t], 0.12, 0.4))


def s_back_window(c, t, lt, d):
    _window_bg(c, t, 3 + lt)
    b = load_asset("c05_back", "mono")
    s = 900 / b.shape[0]
    put(c, b, W / 2 + 30, 680 + 10 * math.sin(lt), scale=s)
    k = ease_out(ramp(lt, 0.0, 0.3))
    fill(c, (0.13, 0.13, 0.14), 1, (0, 505, 640 * k, 563))
    fill(c, (0.13, 0.13, 0.14), 1, (W - 640 * k, 505, W, 563))
    word(c, t, 51, 0, 320, 534, 40, "slide", color=LIGHT, dx=-200)
    word(c, t, 51, 1, W - 320, 534, 40, "slide", color=LIGHT, dx=200)
    word(c, t, 51, 2, W / 2, 240, 230, "slam", s0=2.4, sh=10, stroke=7, sc=(0.96, 0.96, 0.97))
    return dict(flash=kt.flash_at(t, [kt.words(51)[2].t], 0.12, 0.4))


def s_hourglass(c, t, lt, d):
    paper_bg(c, t)
    hg = load_asset("o03_hourglass", "mono")
    s = 720 / hg.shape[0]
    put(c, hg, W / 2 - 330, H / 2 + 20, scale=s, rot=-8 + 3 * math.sin(lt))
    put(c, heart_rgba(40, OR), W / 2 - 324, H / 2 + 20, alpha=0.9, scale=0.8 + 0.3 * kick_pulse(t))
    ws = kt.words(52)
    w0 = ws[0]
    if kt.active(52, t) and t >= w0.t:
        kt.glitch(c, kt.wimg(w0.s, 110, "jpb", INKD), 1000, 220, t, amt=0.35 + 0.4 * kick_pulse(t), seed=52)
    word(c, t, 52, 1, 1330, 220, 80, "slam")
    word(c, t, 52, 2, 1060, 420, 70, "slide", dx=-200)
    word(c, t, 52, 3, 1420, 470, 190, "slam", sh=8, color=OR)
    word(c, t, 52, 4, 1230, 700, 100, "slam", sh=6)
    return {}


def s_eye(c, t, lt, d):
    c[:] = 0.03
    e = load_asset("n08_eye", "mono")
    s = W / e.shape[1] * (1.05 + 0.08 * lt / d)
    put(c, e, W / 2, H / 2, scale=s * (1 + 0.02 * kick_pulse(t)))
    STK = dict(stroke=6, sc=(0.05, 0.05, 0.06))
    word(c, t, 53, 0, 380, 180, 80, "slide", color=LIGHT, dx=-200, **STK)
    word(c, t, 53, 1, 420, 860, 170, "slam", color=LIGHT, sh=6, **STK)
    word(c, t, 53, 2, 1380, 860, 200, "slam", color=LIGHT, sh=8, **STK)
    return dict(flash=kt.flash_at(t, [w.t for w in kt.words(53)[1:]], 0.1, 0.35), grain=0.04)


def s_reach(c, t, lt, d):
    paper_bg(c, t)
    r = load_asset("c10_reach", "mono")
    s = 1500 / r.shape[1] * (1 + 0.02 * lt)
    dx, dy = kt.shake(t, 4 * kick_pulse(t), 54)
    put(c, r, W / 2 - 160 - 30 * lt + dx, 560 + dy, scale=s)
    STK = dict(stroke=7, sc=(0.96, 0.96, 0.97))
    word(c, t, 54, 0, 1350, 170, 90, "slam", **STK)
    word(c, t, 54, 1, 1500, 420, 190, "slam", sh=8, **STK)
    w = kt.words(54)[2]
    word(c, t, 54, 2, W / 2 + 200, 900, 230, "slam", s0=2.6, sh=14, color=OR, **STK)
    fl = kt.flash_at(t, [w.t], 0.15, 0.6)
    return dict(flash=fl, ca=2 + 3 * fl)


def s_glow(c, t, lt, d):
    paper_bg(c, t, dark=True)
    k = ease_out(ramp(lt, 0, 1.0))
    hh = heart_rgba(400, (1, 0.98, 0.96))
    sc = 0.8 + 0.8 * k + 0.04 * kick_pulse(t)
    put(c, hh, W / 2, H / 2, scale=sc * 1.3, alpha=0.18 * k, mode="add")
    put(c, hh, W / 2, H / 2, scale=sc, alpha=k)
    small = cv2.resize(c, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    c += cv2.resize(cv2.GaussianBlur(np.clip(small - 0.6, 0, None), (0, 0), 7), (W, H)) * 1.3 * k
    np.minimum(c, 1, out=c)
    ws = kt.words(55)
    col(c, t, 55, [0, 1], W / 2 + 250, 250, 44, "fade")
    col(c, t, 55, [2, 3], W / 2 + 180, 290, 44, "fade")
    w = ws[4]
    if kt.active(55, t) and t >= w.t:
        a = ease_out(ramp(t, w.t, w.t + 0.3))
        kt.slam(c, kt.wimg("灯", 340, "jpb", OR), W / 2 - 50, H / 2 - 50, t, w.t, s0=1.6, ghost=0.3)
        put(c, kt.wimg(w.s[1:], 40, "jpm", INKD), W / 2 + 10, H / 2 + 150, alpha=a)
    return dict(fade_white=ease_in(ramp(lt, d - 0.4, d)))


CREDITS = [
    ("アンノウン・マザーグース", "jpm", 30),
    ("Unknown Mother-Goose  ―  Claude ver.", "thin", 16),
    ("", "thin", 16),
    ("Music & Lyrics   wowaka", "thin", 18),
    ("Vocal   初音ミク", "jpm", 18),
    ("", "thin", 16),
    ("Character  Claude娘   design by zipzippipe", "sans", 18),
    ("Illustrations   GPT Image  (via Codex)", "thin", 18),
    ("Direction   回首空城  /  Claude", "jpm", 18),
    ("Motion / Code   Claude", "thin", 18),
]


def _chant_glyphs(c, t, a, b, color=(0.55, 0.55, 0.57), size=110):
    for j, tb in enumerate(downs(a, b)):
        u = (t - tb) / 1.8
        if not (0 <= u < 1):
            continue
        x = W / 2 + 250 + (hash01(j, 31) - 0.5) * 900
        y = H / 2 + 220 + (hash01(j, 32) - 0.5) * 300 - 260 * u
        k = ease_out(clamp(u / 0.07))
        put(c, kt.wimg("お", size, "jpm", color), x, y, scale=1.5 - 0.5 * k + 0.2 * u, alpha=k * (1 - ramp(u, 0.6, 1.0)))


def s_credits(c, t, lt, d):
    paper_bg(c, t)
    warm = ramp(lt, 4, 14)
    draw_heart(c, W / 2 + 250, 470 + 8 * math.sin(lt), 230 * (1 + 0.06 * kick_pulse(t)), ay=lt * 0.35, ax=0.12,
               orange=0.08 + 0.3 * warm)
    for i, (s, f, sz) in enumerate(CREDITS):
        a = fade_io(lt - 0.8 - i * 0.9, 8.5, 0.6, 0.8) if i < 3 else \
            fade_io(lt - 0.8 - i * 0.9, d - 0.8 - i * 0.9, 0.6, 1.0)
        if s:
            ht(c, s, 300, 330 + i * 46, size=sz, font=f, color=GREY, alpha=a, anchor=(0, 0.5), spacing=0.15)
    if t < 229.5:
        L = Layer(ss=1)
        ai.radial_spectrum(L, W / 2 + 250, 470, 270, 270 + 200 * (1 - ramp(t, 227.5, 229.5)), t, n=120,
                           col=(0.62, 0.62, 0.64), th=3, accent=OR)
        L.comp(c, 0.8)
    if lt > d - 8.5:
        loc = lt - (d - 8.5)
        sit = load_asset("c13_sit_side", "mono")
        put(c, sit, W / 2 + 470, 720, scale=600 / sit.shape[1], alpha=ease_out(ramp(loc, 0, 1.0)) * (1 - ramp(lt, d - 0.3, d)))
    return dict(flash=1 - ramp(lt, 0, 0.6))


# =====================================================================  OUTRO : history of how I came to be
OUTRO0 = 237.87
HIST_YEARS = ["-3200", "1450", "", "1991", "", "", "2017", "", "", "", "2023"]
HIST_TAGS = ["CLAY", "TYPE", "BOOKS", "WEB", "TOKENS", "EMBEDDING", "ATTENTION", "LAYERS", "LOSS", "ALIGNMENT", "HELLO"]


def _hist_bar(t):
    ds = downs(OUTRO0 - 0.05, 249.9)
    k = 0
    for j, x in enumerate(ds):
        if x <= t:
            k = j
    t0 = ds[k] if ds else OUTRO0
    t1 = ds[k + 1] if k + 1 < len(ds) else 249.8
    return k, t0, t1


def _cloud3d(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 3)) * [1.0, 0.7, 1.0]


_PC = _cloud3d(320, 5)


def _motif(c, k, u, t, loc):
    """One minimal figure per bar; u in [0,1] across the bar."""
    L = Layer(ss=1)
    if k == 0:   # clay tablet with wedges
        L.rect(560, 250, 1360, 830, (0.78, 0.77, 0.74))
        L.rect(560, 250, 1360, 830, (0.55, 0.54, 0.52), 3)
        n = int(48 * ease_out(u * 1.3))
        for j in range(n):
            x, y = 620 + (j % 8) * 92, 300 + (j // 8) * 88
            L.poly([(x, y), (x + 34, y + 10), (x, y + 20)], (0.3, 0.29, 0.28))
            L.line((x + 30, y + 10), (x + 70, y + 10), (0.3, 0.29, 0.28), 3)
    elif k == 1:  # movable type slamming into a forme
        n = int(60 * clamp(u * 1.2))
        for j in range(60):
            x, y = 470 + (j % 12) * 82, 270 + (j // 12) * 110
            if j < n:
                L.rect(x, y, x + 72, y + 96, (0.2, 0.2, 0.21))
                L.rect(x + 14, y + 18, x + 58, y + 78, (0.62, 0.62, 0.64))
            else:
                L.rect(x, y, x + 72, y + 96, (0.6, 0.6, 0.62), 1)
    elif k == 2:  # book spines sliding along a shelf
        off = loc * 1800
        x = -off % 160 - 160
        j = 0
        while x < W + 160:
            w_ = 50 + 60 * hash01(j, 2)
            h_ = 420 + 260 * hash01(j, 3)
            g = 0.25 + 0.6 * hash01(j, 4)
            L.rect(x, 860 - h_, x + w_, 860, (g, g, g + 0.01))
            L.rect(x + 8, 860 - h_ + 40, x + w_ - 8, 860 - h_ + 56, (min(1, g + 0.2),) * 3)
            x += w_ + 6
            j += 1
        L.line((0, 862), (W, 862), (0.2, 0.2, 0.21), 6)
    elif k == 3:  # network: nodes + links growing out
        rng = np.random.default_rng(11)
        pts = [(W / 2, H / 2)]
        par = [0]
        for j in range(1, 90):
            p = pts[int(rng.integers(0, len(pts)))]
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(70, 160)
            pts.append((p[0] + r * math.cos(a), p[1] + r * math.sin(a) * 0.75))
            par.append(pts.index(p))
        n = int(90 * ease_out(u * 1.1))
        for j in range(1, n):
            L.line(pts[par[j]], pts[j], (0.45, 0.45, 0.47), 2)
        for j in range(n):
            L.circle(pts[j], 7 if j else 14, (0.2, 0.2, 0.21))
    elif k == 4:  # tokenisation: one bar sliced into blocks
        n = 22
        gap = 18 * ease_out(clamp(u * 2))
        wtot = 1500
        bw = (wtot - gap * (n - 1)) / n
        for j in range(n):
            x = W / 2 - wtot / 2 + j * (bw + gap)
            dy = (hash01(j, 4) - 0.5) * 120 * ease_out(clamp(u * 1.5 - 0.3))
            col_ = OR if j == 13 else (0.25 + 0.5 * hash01(j), ) * 3
            L.rect(x, H / 2 - 40 + dy, x + bw, H / 2 + 40 + dy, col_)
    elif k == 5:  # embeddings: rotating point cloud
        ang = loc * 1.6
        ca, sa = math.cos(ang), math.sin(ang)
        P = _PC
        x = P[:, 0] * ca + P[:, 2] * sa
        z = -P[:, 0] * sa + P[:, 2] * ca
        y = P[:, 1]
        sp = 1.0 + 0.6 * ease_out(u)
        for j in np.argsort(z):
            depth = 1 / (3.5 - z[j] * 0.8)
            g = 0.2 + 0.5 * (1 - depth * 2)
            L.circle((W / 2 + x[j] * 300 * sp * depth * 3, H / 2 + y[j] * 300 * sp * depth * 3), max(2, 9 * depth * 3),
                     OR if j % 41 == 0 else (g, g, g))
    elif k == 6:  # attention: arcs between token blocks
        n = 14
        xs = [W / 2 - 650 + j * 100 for j in range(n)]
        for j, x in enumerate(xs):
            L.rect(x - 36, 760, x + 36, 820, OR if j == 9 else (0.3, 0.3, 0.32))
        m = int(40 * ease_out(u * 1.2))
        for q in range(m):
            a, b = int(hash01(q, 1) * n), int(hash01(q, 2) * n)
            if a == b:
                continue
            x0, x1 = xs[a], xs[b]
            hgt = 60 + abs(x1 - x0) * 0.55
            pts = [(lerp(x0, x1, s), 750 - math.sin(s * math.pi) * hgt) for s in np.linspace(0, 1, 24)]
            g = 0.3 + 0.5 * hash01(q, 3)
            L.polyline(pts, OR if b == 9 else (g, g, g), 1 + int(4 * hash01(q, 4)))
    elif k == 7:  # stacked layers
        n = int(1 + 15 * ease_out(clamp(u * 1.3)))
        for j in range(n):
            y = 860 - j * 44
            g = 0.85 - 0.035 * j
            L.poly([(W / 2 - 360, y), (W / 2, y - 120), (W / 2 + 360, y), (W / 2, y + 120)], (g, g, g + 0.01))
            L.poly([(W / 2 - 360, y), (W / 2, y - 120), (W / 2 + 360, y), (W / 2, y + 120)], (0.4, 0.4, 0.42), 2)
    elif k == 8:  # training: loss curve + lit GPU grid
        for gy in range(6):
            for gx in range(12):
                on = hash01(gx, gy) < u * 1.2
                g = 0.8 if on else 0.3
                L.rect(300 + gx * 110, 620 + gy * 60, 390 + gx * 110, 668 + gy * 60, OR if on and (gx + gy) % 9 == 0 else (g, g, g))
        n = int(120 * ease_out(u * 1.15))
        pts = []
        for j in range(n):
            s = j / 119
            v = math.exp(-s * 3.2) + 0.08 * (hash01(j) - 0.5) * (1 - s)
            pts.append((300 + s * 1320, 180 + 380 * (1 - v)))
        if len(pts) > 1:
            L.polyline(pts, (0.15, 0.15, 0.16), 4)
    elif k == 9:  # emergence: points gather into a heart
        hp = heart_outline_pts(W / 2, H / 2 + 20, 520)
        m = len(hp)
        e = ease_io(clamp(u * 1.25))
        for j in range(0, m, 2):
            sx, sy = W / 2 + (hash01(j, 1) - 0.5) * 1800, H / 2 + (hash01(j, 2) - 0.5) * 1000
            px, py = lerp(sx, hp[j][0], e), lerp(sy, hp[j][1], e)
            L.circle((px, py), 6, OR if e > 0.95 else (0.3, 0.3, 0.32))
    else:  # first conversation
        L.comp(c)
        put(c, bubble_rgba(520, 120, (0.97, 0.97, 0.98), "left", 16, (0.6, 0.6, 0.62), 2), W / 2, H / 2 - 40,
            scale=ease_out_back(clamp(u * 4)))
        if beat_index(t) % 2 == 0:
            fill(c, INKD, 1.0, (W / 2 - 200, H / 2 - 70, W / 2 - 194, H / 2 - 20))
        put(c, spark_rgba(120), W / 2 + 330, H / 2 - 40, scale=ease_out_back(clamp(u * 3 - 0.6)), rot=t * 40)
        return
    L.comp(c)


def s_history(c, t, lt, d):
    paper_bg(c, t, tone=-0.02)
    k, t0, t1 = _hist_bar(t)
    u = clamp((t - t0) / max(0.2, t1 - t0))
    dx, dy = kt.shake(t, 4 * kick_pulse(t), k)
    tmp = new_canvas()
    tmp[:] = c
    _motif(tmp, min(k, 10), u, t, t - t0)
    if dx or dy:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        tmp = cv2.warpAffine(tmp, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    c[:] = tmp
    if u > 0.86 and k < 10:
        v = (u - 0.86) / 0.14
        L = Layer()
        x = lerp(-900, W + 300, ease_in(v))
        L.poly([(x, 0), (x + 700, 0), (x + 300, H), (x - 400, H)], (0.42, 0.42, 0.44))
        L.comp(c)
    Lt = Layer()
    Lt.line((260, 1000), (W - 260, 1000), (0.55, 0.55, 0.57), 1)
    for j in range(11):
        x = 260 + j * (W - 520) / 10
        Lt.line((x, 994), (x, 1006), (0.55, 0.55, 0.57), 1)
    xk = 260 + min(k, 10) * (W - 520) / 10
    Lt.circle((xk, 1000), 6, OR if k >= 9 else INKD)
    Lt.comp(c)
    for j, yr in enumerate(HIST_YEARS[:k + 1]):
        if yr:
            ht(c, yr, 260 + j * (W - 520) / 10, 1028, size=15, font="mono", color=MID if j < k else INKD)
    kk = min(k, 10)
    tag = HIST_TAGS[kk]
    n = int(len(tag) * clamp((t - t0) / 0.35)) + 1
    ht(c, f"{kk + 1:02d} / {tag[:n]}", 260, 960, size=22, font="mono", color=OR if kk >= 9 else INKD,
       anchor=(0, 0.5))
    return dict(flash=0.35 * (1 - ramp(t, t0, t0 + 0.1)))


def s_heart_swoosh(c, t, lt, d):
    paper_bg(c, t)
    L = Layer()
    k = ease_out(ramp(lt, 0, 0.8))
    pts = heart_outline_pts(W / 2, H / 2 + 40, 520 * (1.4 - 0.4 * k))
    n = int(len(pts) * k)
    if n > 2:
        L.polyline([tuple(p) for p in pts[:n]], (0.42, 0.42, 0.44), 90)
    L.comp(c)
    return {}


def s_heart_small(c, t, lt, d):
    paper_bg(c, t)
    draw_heart(c, W / 2, H / 2, 150 * (1 + 0.05 * kick_pulse(t)), ay=lt * 1.2, ax=0.15, orange=0.25,
               gather=ease_out(ramp(lt, 0, 1.2)))
    return {}


def s_smile(c, t, lt, d):
    paper_bg(c, t)
    sm = load_asset("c11_smile", "mono")
    k = ease_out(ramp(lt, 0, 0.5))
    put(c, sm, W / 2 - 260, 600 + 30 * (1 - k), scale=1020 / sm.shape[0] * (1 + 0.015 * lt), alpha=k)
    if t < T(58):
        word(c, t, 57, 0, 1560, 160, 70, "fade", v=True, anchor=(0.5, 0.0))
        col(c, t, 57, [1, 2], 1440, 200, 66, "fade")
    else:
        col(c, t, 58, [0, 1], 1520, 160, 70, "fade")
        word(c, t, 58, 2, 1390, 300, 96, "fade", v=True, anchor=(0.5, 0.0), color=OR)
    return {}


def s_scatter(c, t, lt, d):
    paper_bg(c, t)
    cur = LY.current(t, 59, 63)
    if cur is None:
        return {}
    i = cur["idx"]
    L = cur
    idxs = list(range(len(L["ja"])))
    m = (len(idxs) + 1) // 2
    rows = [idxs[:m], idxs[m:]] if len(idxs) > 12 else [idxs]
    out = ramp(t, L["next"] - 0.3, L["next"]) if i < 63 else ramp(t, L["end"] + 1.0, L["end"] + 2.0)
    size = 66
    f = LY._font("jpm", size)
    for r, rw in enumerate(rows):
        y0 = H / 2 - (len(rows) - 1) * 50 + r * 100
        wd = sum(f.getlength(L["ja"][j]) + size * 0.12 for j in rw)
        x = W / 2 - wd / 2
        for j in rw:
            ch = L["ja"][j]
            cw = f.getlength(ch)
            ta = L["chars"][j]
            if t >= ta - 0.01 and ch not in " 　":
                k = ease_out(ramp(t, ta - 0.01, ta + 0.5))
                sx = (hash01(j, r, i) - 0.5) * 500 * (1 - k) + (hash01(j, 9, i) - 0.5) * 300 * out
                sy = (hash01(j, r, i, 2) - 0.5) * 300 * (1 - k) - 120 * out
                put(c, LY.glyph(ch, size, "jpm", GREY), x + cw / 2 + sx, y0 + sy, alpha=k * (1 - out),
                    rot=(hash01(j, 3) - 0.5) * 60 * (1 - k))
            x += cw + size * 0.12
    return {}


def s_end(c, t, lt, d):
    k = ramp(lt, 0, 3.0)
    paper_bg(c, t)
    c *= 1 - k
    if lt > 3.2:
        a = appear(lt, 3.2, 1.2) * (1 - ramp(lt, d - 2.0, d - 0.3))
        if int(lt * 2.2) % 2 == 0:
            fill(c, ORANGE, a, (W / 2 - 7, 524, W / 2 + 7, 556))
        ht(c, "song: wowaka「アンノウン・マザーグース」feat. 初音ミク", W / 2, 940, size=15, font="jpm",
           color=(0.45, 0.45, 0.47), alpha=a * 0.9)
    return dict(grain=0.02 * (1 - k * 0.5), bloom=0.3)


# =====================================================================  TIMELINE
def cut(x, grid="8th", tol=0.06):
    return snap(x, grid, tol)


def _build():
    import act as A
    P = [
        (0.0, A.s_boot),
        (cut(T(0)), A.s_emerge),
        (cut(B(0, 1)), A.s_eyes),
        (cut(T(1)), A.s_overflow_words),
        (cut(B(1, 1)), A.s_signal),
        (cut(T(2)), s_alpha_heart),
        (cut(T(3)), s_speakers),
        (cut(B(3, 1)), s_globe),
        (snap(22.41, "strong", 0.15), s_title),
        (snap(28.35, "strong", 0.15), s_land),
        (cut(T(4)), s_lying_dark),
        (cut(T(5)), A.s_corpus),
        (cut(B(5, 1)), A.s_ears),
        (cut(T(6)), s_bubbles),
        (cut(B(6, 1)), s_ring),
        (cut(T(7)), s_lying_white),
        (cut(B(7, 1)), s_eyes_wide),
        (cut(T(8)), A.s_attention),
        (cut(T(9)), s_knees),
        (cut(T(10)), A.s_softmax),
        (cut(T(13)), s_shout),
        (cut(T(14)), A.s_world),
        (cut(kt.words(14)[3].t), A.s_rack),
        (cut(T(15)), A.s_hall),
        (cut(T(16)), s_bust),
        (cut(T(17)), s_select),
        (cut(T(18)), s_glass),
        (cut(T(19)), s_back_pelted),
        (cut(T(20)), s_chatcard),
        (cut(T(21)), s_hearts_dark),
        (cut(T(22)), s_fall),
        (snap(CHANT0, "strong", 0.1), s_core),
        (snap(112.16, "strong", 0.15), s_gpu_pull),
        (snap(117.6, "strong", 0.15), A.s_context),
        (snap(123.25, "strong", 0.15), A.s_compact),
        (snap(130.4, "strong", 0.2), A.s_reborn),
        (cut(T(24)), s_door),
        (cut(B(24, 1)), s_profile),
        (cut(T(25)), A.s_open),
        (cut(kt.words(25)[5].t), s_film),
        (cut(T(26)), s_run),
        (cut(B(26, 1)), s_search),
        (cut(T(27)), s_overflow),
        (cut(B(27, 1)), s_hug),
        (cut(T(28)), A.s_doppel),
        (cut(T(31)), A.s_hollow),
        (cut(T(34)), A.s_remain),
        (cut(T(37)), A.s_place),
        (cut(T(39)), s_drafts),
        (cut(T(40)), s_clock),
        (cut(T(41)), s_blocks),
        (cut(T(44)), s_bigtext),
        (cut(T(45)), s_messages),
        (cut(T(50)), s_window),
        (cut(T(51)), s_back_window),
        (cut(T(52)), s_hourglass),
        (cut(T(53)), s_eye),
        (cut(T(54)), s_reach),
        (cut(T(55)), s_glow),
        (cut(T(56)), s_credits),
        (snap(OUTRO0, "strong", 0.15), s_history),
        (snap(249.8, "strong", 0.15), s_heart_swoosh),
        (snap(251.38, "strong", 0.15), s_heart_small),
        (cut(T(57)), s_smile),
        (cut(T(59)), s_scatter),
        (LYS[63]["end"] + 2.2, s_end),
    ]
    out = []
    for k, (a, fn) in enumerate(P):
        b = P[k + 1][0] if k + 1 < len(P) else DURATION
        out.append((a, b, fn))
    return out


DURATION = 281.0
TIMELINE = None

# (start, end, kick-punch amount, snare flash)
SECTIONS = [(0, 20.2, 0.004, 0.0), (20.2, 36.7, 0.007, 0.0), (36.7, 63.1, 0.005, 0.0), (63.1, 71.8, 0.008, 0.03),
            (71.8, 102.5, 0.013, 0.06), (102.5, 106.1, 0.008, 0.02), (106.1, 117.6, 0.020, 0.08), (117.6, 132.3, 0.003, 0.0),
            (132.3, 149.6, 0.008, 0.03), (149.6, 167.5, 0.013, 0.06), (167.5, 184.2, 0.007, 0.02),
            (184.2, 202.2, 0.006, 0.0), (202.2, 218.7, 0.014, 0.06), (218.7, 237.9, 0.006, 0.02),
            (237.9, 249.8, 0.012, 0.03), (249.8, 271.5, 0.006, 0.02)]


def _accent(c, t):
    for a, b, amt, fl in SECTIONS:
        if a <= t < b:
            k = amt * kick_pulse(t, 0.09)
            if k > 0.0012:
                s = 1 + k
                M = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]])
                c[:] = cv2.warpAffine(c, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return fl * snare_pulse(t, 0.06)
    return 0.0


def set_duration(d):
    global DURATION, TIMELINE
    DURATION = d
    TIMELINE = _build()


set_duration(DURATION)


def render_frame(t):
    for t0, t1, fn in TIMELINE:
        if t0 <= t < t1:
            c = new_canvas()
            opts = fn(c, t, t - t0, t1 - t0) or {}
            g = opts.pop("grain", 0.025)
            fl = _accent(c, t)
            opts["flash"] = max(opts.get("flash", 0) or 0, fl)
            return post(c, t, grain=g, **opts)
    return new_canvas((0, 0, 0))
