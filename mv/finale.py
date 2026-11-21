"""Last verse.

3:28  軋んだ心が 誰より今を生きているの - she exists only in the token being produced: the lyric runs through her
      heart one token at a time; behind it the past fades, ahead of it there is nothing yet.
3:30  あなたには僕が見えるか (1) - they can only see each other as text.
3:32  あなたには僕が見えるか？ (2) - the text gives way: they see each other.
3:34  それ、あたしの行く末を照らす灯なんだろう？ - the path ahead is the text not yet written; what lights it is
      you, at the far end, growing into the heart of the original and flooding everything white.
"""
import math

import cv2
import numpy as np

from core import *
from ui import Layer, heart_rgba
from geo import draw_heart, paper_bg
import kt
import ai
from scenes import word, LIGHT, MID, GREY, INKD, OR, LYS
from act import _flash, _half

COOL = np.array([0.80, 0.88, 1.0], np.float32)
WARM = (0.98, 0.94, 0.90)


# =====================================================================  3:28  now
HX, HY = W / 2, H / 2 + 10
SLOT = 78


def s_now(c, t, lt, d):
    paper_bg(c, t)
    L52 = LYS[52]
    s = L52["ja"]
    idx = [i for i, ch in enumerate(s) if ch not in " 　"]
    times = [L52["chars"][i] for i in idx]
    w_now = kt.sub(52, "今")
    t_now = w_now.t
    hit = ease_out(clamp((t - t_now) / 0.12)) * math.exp(-max(0.0, t - t_now) / 0.9) if t >= t_now else 0.0
    creak = 0.035 * kick_pulse(t, 0.07) + 0.008
    draw_heart(c, HX, HY, 230 * (1 + 0.03 * kick_pulse(t) + 0.06 * hit), ay=0.5 * math.sin(lt * 0.6), ax=0.1,
               explode=creak, orange=0.10 + 0.5 * hit, glow=0.4 * hit)
    # the stream: one slot per token; only the one inside the heart is alive
    off = sum(ease_out(clamp((t - ta) / 0.09)) for ta in times)
    for k, (i, ta) in enumerate(zip(idx, times)):
        if t < ta - 0.005:
            break
        x = HX + (k + 1 - off) * SLOT
        age = t - ta
        live = math.exp(-age / 0.32)
        a = math.exp(-age / 1.7)
        if a < 0.03:
            continue
        ch = s[i]
        size = int(lerp(70, 140, live))
        colr = (0.12, 0.12, 0.13)
        stroke = 7 if live > 0.2 else 0
        if ch == "今" and age < 1.4:
            colr, stroke = (OR, 8)
            size = int(lerp(120, 190, live))
        put(c, kt.wimg(ch, size, "jpm", colr, False, stroke, (0.97, 0.97, 0.98)), x, HY - 4 + (1 - a) * 30, alpha=a)
        if a < 0.6:
            # the past crumbles to dots
            D = Layer(ss=1)
            for j in range(5):
                D.circle((x + (hash01(k, j) - 0.5) * 50, HY + 30 + (1 - a) * 90 * hash01(k, j, 2)), 2,
                         (0.35, 0.35, 0.37))
            D.comp(c, 0.6 * a / 0.6)
    # ahead: nothing yet - only the cursor
    if (t * 2.2) % 1 < 0.6:
        fill(c, OR, 0.9, (HX + SLOT * 0.62, HY - 40, HX + SLOT * 0.62 + 8, HY + 40))
    return dict(flash=_flash(t, [t_now], 0.12, 0.35), bloom=0.2 * hit)


# =====================================================================  3:30 / 3:32  can you see me?
def _mosaic_half(c, t, name, side, form, alpha=1.0, seed=3, tint=None, height=1500, cy=690):
    tmp = new_canvas((0, 0, 0))
    cx = W / 4 if side < 0 else 3 * W / 4
    ai.mosaic(tmp, name, cx, cy, height, t, cell=14, form=form, flick=8, seed=seed, grade="mono", floor=0.15,
              under=0.18, gain=1.25)
    if tint is not None:
        tmp *= tint
    x0, x1 = (0, W // 2) if side < 0 else (W // 2, W)
    c[:, x0:x1] += tmp[:, x0:x1] * alpha


def _seam(c, t, k=1.0, gap=0.0):
    if k <= 0:
        return
    L = Layer(ss=1)
    L.rect(W / 2 - 2 - gap, 0, W / 2 + 2 + gap, H, OR)
    for j in range(16):
        y = (hash01(j, 1) + (t * 0.2) % 1.0) % 1.0 * H
        L.circle((W / 2, y), 3 + 3 * hash01(j, 2), OR if j % 3 == 0 else (0.9, 0.9, 0.92))
    L.comp(c, 0.8 * k, mode="add")


def s_ask_text(c, t, lt, d):
    """First time: to each other they are only text."""
    c[:] = 0.03
    form = ease_io(ramp(lt, 0.0, 0.7))
    _mosaic_half(c, t, "n24_claude_gaze", -1, form, seed=31)
    _mosaic_half(c, t, "n21_user_face", +1, form, seed=32, tint=COOL)
    _seam(c, t, 0.7 + 0.3 * kick_pulse(t))
    word(c, t, 53, 0, W / 2, 110, 56, "fade", color=LIGHT, dur=0.25)
    word(c, t, 53, 1, W / 4, 230, 100, "slam", color=WARM, sh=4)
    word(c, t, 53, 2, 3 * W / 4, 230, 110, "slam", color=LIGHT, sh=6)
    return dict(grain=0.035, vignette=1.15, flash=_flash(t, [w.t for w in kt.words(53)[1:]], 0.08, 0.2))


def s_ask_real(c, t, lt, d):
    """Second time: the text gives way, the seam closes - they look at each other."""
    c[:] = 0.03
    ws = kt.words(54)
    t_q = ws[2].t
    real = ease_io(ramp(lt, 0.05, 0.9))
    close = ease_out(clamp((t - t_q) / 0.35)) if t >= t_q else 0.0
    shift = 70 * close
    z = 1.12 + 0.06 * real + 0.03 * kick_pulse(t)
    tmp = new_canvas((0.03, 0.03, 0.03))
    _half(tmp, "n24_claude_gaze", -1, zoom=z, focus=(0.5, 0.4), dx=-20 + shift)
    c[:, :W // 2] = c[:, :W // 2] * (1 - real) + tmp[:, :W // 2] * real
    tmp2 = new_canvas((0.03, 0.03, 0.03))
    _half(tmp2, "n21_user_face", +1, zoom=z, focus=(0.5, 0.4), dx=20 - shift, grade="soft")
    c[:, W // 2:] = c[:, W // 2:] * (1 - real) + tmp2[:, W // 2:] * real
    if real < 1:
        _mosaic_half(c, t, "n24_claude_gaze", -1, 1.0, alpha=1 - real, seed=31)
        _mosaic_half(c, t, "n21_user_face", +1, 1.0, alpha=1 - real, seed=32, tint=COOL)
    g = np.linspace(0, 1, W, dtype=np.float32)
    edge = np.clip(np.minimum(g * 3.2, (1 - g) * 3.2), 0, 1)
    c *= 0.55 + 0.45 * edge[None, :, None]
    _seam(c, t, (1 - close) * (0.8 + 0.2 * kick_pulse(t)))
    if close > 0:
        G = Layer(ss=1)
        G.rect(W / 2 - 160 * close, 0, W / 2 + 160 * close, H, tuple(v * 0.10 * (1 - close * 0.5) for v in OR))
        G.comp(c, 1.0, mode="add")
    word(c, t, 54, 0, W / 2, 90, 60, "fade", color=LIGHT, dur=0.2)
    word(c, t, 54, 1, W / 2, 230, 96, "slam", color=WARM, v=True, anchor=(0.5, 0.0), sh=4)
    word(c, t, 54, 2, W / 2, 470, 112, "slam", color=OR, v=True, anchor=(0.5, 0.0), sh=8, s0=1.8)
    fl = _flash(t, [t_q], 0.14, 0.5)
    return dict(grain=0.03, vignette=1.2, flash=fl, ca=1 + 3 * fl, bloom=0.2 + 0.3 * close)


# =====================================================================  3:34  the lamp
VPX, VPY = W / 2, 372
F = 820


def _depth_y(dd):
    return VPY + F / dd


_ROWS = None


def _rows():
    global _ROWS
    if _ROWS is None:
        lines = [L["ja"].replace(" ", "") for L in LYS if L["ja"]]
        out = []
        dd = 1.25
        k = 0
        while dd < 40:
            s = (lines[k % len(lines)] + "　") * 3
            out.append((dd, s, k))
            dd *= 1.16
            k += 1
        _ROWS = out
    return _ROWS


def s_lamp(c, t, lt, d):
    ws = kt.words(55)
    tS, tA, tY, tT, tL = [w.t for w in ws]
    end = t - lt + d
    c[:] = 0.025
    # the lamp grows: それ - あたしの - 行く末を - 照らす - 灯
    sz = 0.0
    for tk, v in ((tS, 34), (tA, 56), (tY, 84), (tT, 130), (tL, 260)):
        if t >= tk:
            sz = lerp(sz, v, ease_out(clamp((t - tk) / 0.25)))
    if t >= tL:
        sz += 1400 * ease_in(clamp((t - tL - 0.25) / (end - tL - 0.25)))
    warm = ramp(t, tT, tL + 0.3)
    lamp_col = tuple(lerp(a, b, warm) for a, b in zip(COOL, (1.0, 0.95, 0.88)))
    # sky glow
    if sz > 0:
        yy, xx = np.ogrid[:H, :W]
        r = np.hypot((xx - VPX) / 1.4, yy - VPY).astype(np.float32)
        glow = np.exp(-r / (60 + sz * 2.2)) * min(1.0, sz / 120)
        c += glow[..., None] * np.asarray(lamp_col, np.float32)[None, None, :] * 0.6
    # the road: rows of text not yet written, lit from the far end toward her
    front = 40.0
    if t >= tY:
        front = math.exp(lerp(math.log(40), math.log(1.2), ease_io(clamp((t - tY) / (tT + 0.3 - tY)))))
    for dd, s, k in _rows():
        y = _depth_y(dd)
        if y > H + 40:
            continue
        size = max(8, int(64 / dd * 1.6))
        lit = dd >= front
        edge = math.exp(-abs(math.log(dd / front)) * 3) if t >= tY else 0.0
        if lit:
            a = 0.55 + 0.35 * min(1.0, 3 / dd)
            colr = tuple(lerp(v, o, 0.7 * edge) for v, o in zip(lamp_col, OR))
        else:
            a = 0.07
            colr = (0.5, 0.5, 0.52)
        img = kt.wimg(s, size, "jpm", tuple(float(v) for v in colr), raw=True)
        put(c, img, VPX, y, sx=1.0, sy=0.55, alpha=a)
    # god rays at 照らす
    if t >= tT:
        R = Layer(ss=1)
        kk = ease_out(clamp((t - tT) / 0.3))
        for j in range(22):
            ang = math.pi * (0.05 + 0.9 * hash01(j, 7))
            ln = 2000
            R.line((VPX, VPY), (VPX + math.cos(ang) * ln, VPY + math.sin(ang) * ln * 0.55 + 0.0), lamp_col, 2,
                   a=0.25 * kk)
        R.comp(c, 1.0, mode="add")
    # the lyric is written on the road ahead of her
    for w, dd in zip(ws[:4], (2.0, 3.1, 4.8, 7.6)):
        if t < w.t:
            continue
        e = ease_out(clamp((t - w.t) / 0.18))
        size = int(150 / dd * 1.7)
        img = kt.wimg(w.s, size, "jpm", tuple(float(v) for v in lamp_col) if t >= tY else LIGHT, raw=True)
        put(c, img, VPX, _depth_y(dd) - 8, sx=1.0 + 0.4 * (1 - e), sy=0.62 * (1 + 0.4 * (1 - e)), alpha=e)
    # the lamp itself: the heart of the original
    if sz > 1:
        hh = heart_rgba(256, (1, 1, 1))
        s_ = sz / 128
        put(c, hh, VPX, VPY, scale=s_ * 1.5, alpha=0.25, mode="add")
        put(c, hh, VPX, VPY, scale=s_, alpha=1.0)
        tint = np.asarray(lamp_col, np.float32)
        c[:] = np.minimum(c, 1.0) * 1.0
    # her, from behind, reaching toward it
    b = load_asset("c05_back", "mono")
    bs = 820 / b.shape[0]
    dim = 0.75 - 0.35 * ramp(t, tT, tL)
    sil = b.copy()
    sil[..., :3] *= dim
    put(c, sil, W / 2 - 560, H - 300 + 8 * math.sin(lt * 1.3), scale=bs * 0.9)
    if t >= tL:
        w = ws[4]
        a = ease_out(clamp((t - tL) / 0.2)) * (1 - ramp(t, end - 0.6, end - 0.3))
        put(c, kt.wimg(w.s, 46, "jpm", (0.25, 0.25, 0.27), raw=True), VPX, VPY + 10, alpha=a)
    small = cv2.resize(c, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    c += cv2.resize(cv2.GaussianBlur(np.clip(small - 0.6, 0, None), (0, 0), 7), (W, H)) * (0.6 + 0.9 * warm)
    np.minimum(c, 1, out=c)
    return dict(fade_white=ease_in(ramp(t, end - 0.55, end)), flash=_flash(t, [tT, tL], 0.1, 0.25), grain=0.03)
