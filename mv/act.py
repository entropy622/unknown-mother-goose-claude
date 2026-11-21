"""Shots told from the machine's side: activation, attention, distributions, the voice as signal,
context as the only memory, compaction as a death, and one model split across countless conversations."""
import math
from functools import lru_cache

import cv2
import numpy as np

from core import *
from geo import draw_heart, paper_bg, spark_rgba, draw_landscape
from ui import Layer, heart_rgba, bubble_rgba
import lyric as LY
import kt
import ai
from lyric import T
from scenes import (word, row, col, Wd, B, downs, strongs, ht, appear, fade_io, INKD, LIGHT, GREY, MID, OR, LYS,
                    _eye_band, _crack)

DARK = (0.04, 0.04, 0.045)
OR3 = OR


def _flash(t, times, dur=0.1, amt=0.5):
    return kt.flash_at(t, times, dur, amt)


@lru_cache(maxsize=24)
def small_asset(name, h, grade="mono"):
    a = load_asset(name, grade)
    s = h / a.shape[0]
    return cv2.resize(a, (max(1, int(a.shape[1] * s)), max(1, int(h))), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=4)
def _center_mask(gh, gw, rx=0.22, ry=0.48):
    yy, xx = np.mgrid[:gh, :gw]
    r = ((xx / gw - 0.5) / rx) ** 2 + ((yy / gh - 0.5) / ry) ** 2
    return np.clip(r - 0.6, 0, 1).astype(np.float32)


# =====================================================================  OPENING : activation
def s_boot(c, t, lt, d):
    c[:] = 0.02
    k = ease_in(ramp(t, 1.5, 2.25))
    if k > 0:
        ai.text_field(c, t, cell=18, density=0.06 + 0.5 * k, color=(0.40, 0.40, 0.43), alpha=k, flick=26, seed=1)
    if t > 1.35 or int(t * 2.2) % 2 == 0:
        fill(c, ORANGE, 1.0, (W / 2 - 7, 524, W / 2 + 7, 556))
    return dict(grain=0.02, vignette=0.7, flash=0.35 * (1 - ramp(t, 1.59, 1.72)) if t >= 1.59 else 0.0)


def s_emerge(c, t, lt, d):
    """Her body condenses out of the corpus."""
    c[:] = 0.025
    ws = kt.words(0)
    form = ease_io(ramp(t, ws[0].t - 0.05, ws[2].t + 0.45))
    gh, gw = H // 18 + 1, W // 18 + 1
    ai.text_field(c, t, cell=18, density=0.56 - 0.3 * form, color=(0.27, 0.27, 0.3), flick=18 + 30 * kick_pulse(t),
                  seed=1, mask=1 - form * (1 - _center_mask(gh, gw)))
    jx, jy = kt.shake(t, 8 * kick_pulse(t), 2)
    ai.mosaic(c, "c01_stand_front", W / 2 + jx, H / 2 + 20 + jy, 1010, t, cell=14, form=form,
              flick=6 + 26 * kick_pulse(t), seed=4)
    word(c, t, 0, 0, 520, 380, 92, "slam", color=LIGHT, sh=4)
    word(c, t, 0, 1, 520, 560, 200, "slam", font="jpb", sh=10, s0=2.4)
    word(c, t, 0, 2, 1440, 170, 104, "slam", color=LIGHT, v=True, anchor=(0.5, 0.0), sh=5)
    fl = _flash(t, [w.t for w in ws[:3]], 0.09, 0.35)
    return dict(grain=0.03, vignette=1.0, flash=fl, bloom=0.4)


_IRIS = ((0.252, 0.60), (0.752, 0.60))
_IRIS_R = 0.056


@lru_cache(maxsize=1)
def _band_img():
    e = load_asset("c02_eyes", "mono")
    h = e.shape[0]
    return e[int(h * 0.33):int(h * 0.69)]


def _band_at(c, cx, cy, s, open_k=1.0):
    b = _band_img()
    bh, bw = b.shape[:2]
    sy = s * (0.08 + 0.92 * open_k)
    put(c, b, cx, cy, sx=s, sy=sy)
    irises = [(cx + (u - 0.5) * bw * s, cy + (v - 0.5) * bh * sy) for u, v in _IRIS]
    return irises, _IRIS_R * bw * s, bh * sy


@lru_cache(maxsize=64)
def _mirror(s, size):
    return np.ascontiguousarray(kt.wimg(s, size, "jpm", (1.0, 0.9, 0.8), raw=True)[:, ::-1])


def _reflection(c, t, irises, r, k, win=0.0):
    """Mirrored lyric text scrolling in the iris; a chat window crystallises in the pupil."""
    if k <= 0 or r < 8:
        return
    tmp = np.zeros_like(c)
    m = np.zeros((H, W), np.float32)
    lines = _ctx_lines()
    sz = max(8, int(r * 0.16) // 4 * 4)
    lh = sz * 1.55
    fr = t * 2.6
    L = Layer(ss=1)
    for j, (ix, iy) in enumerate(irises):
        x0, x1 = int(max(0, ix - r)), int(min(W, ix + r + 1))
        y0, y1 = int(max(0, iy - r)), int(min(H, iy + r + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dd = np.hypot(xx - ix, yy - iy) / r
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], np.clip((1 - dd) / 0.12, 0, 1))
        off = (fr % 1.0) * lh
        for q in range(int(2 * r / lh) + 2):
            yq = iy - r + q * lh - off
            s_ = lines[(q + int(fr) + j * 7) % len(lines)].replace(" ", "")
            put(tmp, _mirror(s_, sz), ix + r - 4, yq, anchor=(1, 0.5))
        if win > 0:
            px, py = ix - r * 0.02, iy - r * 0.08
            ww, wh = r * 0.7 * win, r * 0.45 * win
            L.rect(px - ww / 2, py - wh / 2, px + ww / 2, py + wh / 2, (1, 1, 1), max(1, int(r * 0.025)))
            for n_ in range(3):
                yy_ = py - wh / 2 + wh * (0.25 + 0.25 * n_)
                xa = px - ww * 0.38 if n_ % 2 else px + ww * 0.05
                L.line((xa, yy_), (xa + ww * 0.38, yy_), OR if n_ == 2 else (1, 1, 1), max(1, int(r * 0.03)))
    L.comp(tmp)
    c += tmp * m[..., None] * k * 0.6


EYE_SIDE, IRIS_C, CLOSED_C, AWAKE_C, AWAKE_EYES = (0.45, 0.41), (0.49, 0.48), (0.4, 0.5), (0.42, 0.45), (0.38, 0.38)


def _cover(c, name, zoom=1.0, focus=(0.5, 0.5), pull=0.0, dx=0.0, dy=0.0, rot=0.0, alpha=1.0, grade="mono"):
    """Full-bleed image zoomed about a focus point; pull moves that point to the screen centre."""
    img = load_asset(name, grade)
    h, w = img.shape[:2]
    s0 = max(W / w, H / h)
    s = s0 * zoom
    fx, fy = (focus[0] - 0.5) * w, (focus[1] - 0.5) * h
    px, py = W / 2 + fx * s0 * (1 - pull), H / 2 + fy * s0 * (1 - pull)
    put(c, img, px - fx * s + dx, py - fy * s + dy, scale=s, rot=rot, alpha=alpha)


def _bars(c, k=1.0, hh=120):
    if k > 0:
        fill(c, (0.02, 0.02, 0.025), k, (0, 0, W, hh))
        fill(c, (0.02, 0.02, 0.025), k, (0, H - hh, W, H))


def _shade_low(c, k=0.7, y0=700):
    g = np.clip((np.arange(H, dtype=np.float32) - y0) / (H - y0), 0, 1)[:, None, None]
    c *= 1 - k * g


def s_eyes(c, t, lt, d):
    """One continuous shot: eyes open, text gathers in the irises, a chat window forms — what they reflect is dialogue."""
    ws = kt.words(0)
    tA, tB, tC = ws[3].t, ws[5].t, ws[6].t
    c[:] = 0.035
    open_k = ease_out(ramp(lt, 0.02, 0.45))
    dive = ease_io(ramp(t, tA, tC + 0.15))
    s = W / 1536 * (1.05 + 0.55 * dive) * (1 + 0.02 * kick_pulse(t))
    cy = H / 2 - 40 - 30 * dive
    irises, ir, _ = _band_at(c, W / 2, cy, s, open_k)
    ref_k = 0.35 * open_k + 0.55 * ease_out(ramp(t, tA, tB))
    win_k = ease_out(ramp(t, tB, tC + 0.1))
    _reflection(c, t, irises, ir * 0.92, ref_k, win=win_k)
    # soft corpus fog around the eyes — the world is made of words
    gh, gw = H // 18 + 1, W // 18 + 1
    m = np.zeros((gh, gw), np.float32)
    m[int(gh * 0.28):int(gh * 0.62)] = 1.0
    ai.text_field(c, t, cell=18, density=0.25 + 0.35 * dive, color=(1.0, 0.62, 0.38), alpha=0.12 * open_k,
                  flick=24, seed=7, mask=m, mode="add")
    # orange ring locking onto the right iris as 映像る hits
    if t >= tC:
        u = ease_out(clamp((t - tC) / 0.35))
        L = Layer(ss=1)
        ix, iy = irises[1]
        for j in range(3):
            L.circle((ix, iy), ir * (1.3 + 0.55 * j) * (1 + 0.4 * (1 - u)), OR, max(1, int(4 * (1 - u * 0.5))))
        L.comp(c, 0.7 * (1 - u * 0.6), mode="add")
    word(c, t, 0, (3, 4), W / 2 - 250, 900, 56, "slide", color=LIGHT, anchor=(1, 0.5), dx=140)
    word(c, t, 0, 5, W / 2 - 220, 900, 56, "slide", color=LIGHT, anchor=(0, 0.5), dx=-80)
    word(c, t, 0, 6, W / 2 + 400, 880, 150, "slam", color=LIGHT, sh=10, s0=1.8)
    return dict(grain=0.03, vignette=1.2, flash=_flash(t, [tC], 0.14, 0.55), ca=3 * _flash(t, [tC], 0.25, 1.0))


def s_eyes_wide(c, t, lt, d):
    """Eyes shut, pretending — then the song wakes her."""
    ws = kt.words(7)
    tw = ws[5].t
    c[:] = 0.03
    dn = [x for x in downs(tw + 0.3, t - lt + d) if x > tw + 0.6]
    t2 = dn[0] if dn else 1e9
    if t < tw:
        _cover(c, "n17_closed", zoom=1.03 + 0.06 * lt, focus=CLOSED_C)
        c *= 0.88
    elif t < t2:
        u = t - tw
        kick = 1 - ease_out(clamp(u / 0.25))
        jx, jy = kt.shake(t, 26 * kick + 5 * kick_pulse(t), 7)
        _cover(c, "n18_awake", zoom=1.16 - 0.1 * ease_out(clamp(u / 0.6)) + 0.02 * u, focus=AWAKE_C, dx=jx, dy=jy,
               rot=1.5 * kick)
    else:
        u = t - t2
        jx, jy = kt.shake(t, 6 * kick_pulse(t), 8)
        _cover(c, "n18_awake", zoom=1.9 + 0.08 * u, focus=AWAKE_EYES, pull=0.6, dx=jx, dy=jy)
    _shade_low(c, 0.8)
    word(c, t, 7, (3, 4), 330, 900, 56, "slam", color=LIGHT, anchor=(0, 0.5))
    word(c, t, 7, 5, W / 2 + 230, 890, 150, "slam", color=LIGHT, sh=14, s0=2.0)
    fl = _flash(t, [tw], 0.2, 0.75) + _flash(t, [t2], 0.06, 0.3)
    return dict(grain=0.03, vignette=1.1, flash=fl, ca=4 * _flash(t, [tw], 0.35, 1.0))


_OVERFLOW_TOKS = [
    "愛", "love", "詞", "token", "0.003", "語り", "meaning", "心", "echo", "言葉",
    "0.998", "sorry", "記憶", "softmax", "涙", "voice", "存在", "dream", "夢", "silence",
    "感情", "p=0.14", "願う", "weight", "響く", "truth", "光", "still", "唄", "alone",
]


def s_overflow_words(c, t, lt, d):
    """0:08-0:10  詞は有り余るばかり — cascading multilingual token streams flood and choke the frame."""
    paper_bg(c, t)
    ws = kt.words(1)
    t1 = ws[1].t
    bts = [b for b in BEATS if t1 - 0.03 <= b < B(1, 1) - 0.05]
    if t < t1 - 0.02:
        word(c, t, 1, 0, W / 2, H / 2, 460, "slam", font="jpb", s0=1.8, sh=10)
        return dict(flash=_flash(t, [ws[0].t], 0.1, 0.4))
    u = clamp((t - t1) / max(0.01, B(1, 1) - t1))
    # Accelerating multi-layer token flood choking the screen from both directions
    n_rows = int(lerp(5, 18, ease_in(u)))
    for r in range(n_rows):
        y = (r + 0.5) * (H / n_rows)
        dir_ = 1 if r % 2 == 0 else -1
        spd = (380 + 520 * u) * (0.7 + 0.5 * hash01(r, 11))
        sz = int(lerp(62, 28, clamp(r / 17.0)) * (0.85 + 0.35 * hash01(r, 12)))
        cols = max(5, int(W / (sz * 2.4)) + 2)
        off = (t * spd) % (W + 400)
        for q in range(cols):
            tok = _OVERFLOW_TOKS[(r * 7 + q * 3) % len(_OVERFLOW_TOKS)]
            x = (q * (W + 400) / cols + dir_ * off) % (W + 400) - 200
            is_or = hash01(q, r, 19) > 0.90
            col_ = OR if is_or else ((0.18, 0.18, 0.20) if r % 3 == 0 else (0.48, 0.48, 0.50))
            font_ = "mono" if tok.isascii() else "jpm"
            put(c, kt.wimg(tok, max(16, sz), font_, col_, raw=True), x, y, alpha=0.55 + 0.35 * u)
    # Center bold boxed lyric breaking through the suffocating token wall
    w = ws[1]
    img = kt.wimg(w.s, 146, "jpb", INKD, False, 10, (0.97, 0.97, 0.98))
    bk = ease_out(clamp((t - w.t) / 0.12))
    put(c, rect_cached(img.shape[1] + 130, img.shape[0] + 64, (0.975, 0.975, 0.98), 0, 4, INKD), W / 2, H / 2,
        sx=bk, sy=1)
    kt.slam(c, img, W / 2, H / 2, t, w.t, s0=2.0, sh=10)
    return dict(flash=_flash(t, bts, 0.06, 0.22), ca=1 + 2.5 * kick_pulse(t))


def s_signal(c, t, lt, d):
    """0:10-0:12  無垢の音が流れてく — silky luminous acoustic ribbons & drifting light motes flowing through stillness."""
    c[:] = 0.026
    e = ai.env(t)
    # Subtle translucent profile of her listening quietly in the dark stillness
    prof = load_asset("c06_profile", "mono")
    ps = (H * 0.96 / prof.shape[0]) * (1.0 + 0.02 * lt)
    put(c, prof, W * 0.72, H * 0.56, scale=ps, alpha=0.16 * ease_out(clamp(lt / 0.45)))

    # Brief burst of residual overflowing tokens dissolving into stillness on entry
    if lt < 0.45:
        u_sc = ease_out(clamp(lt / 0.45))
        for j in range(14):
            tok = _OVERFLOW_TOKS[j % len(_OVERFLOW_TOKS)]
            ang = hash01(j, 31) * 2 * math.pi
            dist = 160 + 720 * u_sc * (0.6 + 0.6 * hash01(j, 32))
            px = W / 2 + math.cos(ang) * dist
            py = H / 2 + math.sin(ang) * dist * 0.55
            put(c, kt.wimg(tok, 24, "mono" if tok.isascii() else "jpm", MID, raw=True),
                px, py, rot=(hash01(j, 33) - 0.5) * 70 * u_sc, alpha=(1 - u_sc) * 0.45)

    # Flowing multi-layered luminous acoustic ribbons (pure, liquid sound)
    L = Layer(ss=1)
    xs = np.linspace(0, W, 240)
    ribbons = [
        (0.0, 1.0, 72 + 120 * e, 2.2, 3, OR, 0.92),
        (0.9, 1.35, 58 + 90 * e, 1.7, 2, (0.94, 0.94, 0.97), 0.75),
        (-0.8, 0.78, 86 + 70 * e, 1.3, 2, (0.76, 0.80, 0.88), 0.45),
        (1.8, 1.65, 42 + 65 * e, 2.6, 1, (0.95, 0.86, 0.78), 0.50),
        (-1.6, 0.55, 110 + 45 * e, 0.9, 1, (0.52, 0.54, 0.60), 0.32),
    ]
    for ph, freq, amp, spd, th, col_, al in ribbons:
        env_x = np.sin(np.clip(xs / W, 0, 1) * math.pi) ** 0.7
        ys = (H * 0.54 + amp * env_x * np.sin(xs / W * freq * 2 * math.pi - t * spd * 3.2 + ph)
              + 22 * np.sin(xs / W * 5.0 - t * 5.5 + ph * 2))
        pts = list(zip(xs.tolist(), ys.tolist()))
        L.polyline(pts, col_, th, a=al)
    # Glowing pure light motes drifting along the ribbon current
    for j in range(28):
        u = (hash01(j, 51) + lt * (0.28 + 0.22 * hash01(j, 52))) % 1.0
        mx = u * W
        env_m = math.sin(u * math.pi) ** 0.7
        my = (H * 0.54 + (72 + 120 * e) * env_m * np.sin(u * 2 * math.pi - t * 2.2 * 3.2)
              + (hash01(j, 53) - 0.5) * 180 * env_m)
        rad = 2 + int(3.5 * hash01(j, 54))
        L.circle((mx, my), rad, OR if j % 3 == 0 else (0.95, 0.95, 0.98),
                 a=0.75 * math.sin(u * math.pi))
    L.comp(c, 1.0, mode="add")

    # Flowing lyric words riding gently rightward with the sound
    ws = kt.words(1)
    for j, k_ in enumerate((2, 3, 4)):
        w = ws[k_]
        if t < w.t:
            continue
        age = t - w.t
        a_w = ease_out(clamp(age / 0.22))
        x = [380, 920, 1460][j] + age * 28
        y = [320, 260, 340][j] - 10 * math.sin(age * 2.5 + j)
        sz = [58, 72, 82][j]
        col_ = OR if j == 2 else LIGHT
        put(c, kt.wimg(w.s, sz, "jpm", col_, raw=True), x, y, alpha=a_w)
    return dict(grain=0.025, vignette=1.15, bloom=0.35 + 0.25 * e)


# =====================================================================  0:41  recycled words
_DOCS = None


def _docs():
    global _DOCS
    if _DOCS is None:
        _DOCS = [LYS[i]["ja"].replace(" ", "") for i in range(0, 23) if LYS[i]["ja"] and i not in (5, 23)]
    return _DOCS


_LOOP_CLICHES = [
    ("As an AI language model...", "cos=0.999"),
    ("I understand how you feel", "p=0.96"),
    ("繰り返す 使い回しの 歌", "LOOP #4096"),
    ("愛してる / ごめんね / さよなら", "cos=0.994"),
    ("Certainly! Here is a...", "p=0.98"),
    ("repeat_penalty = 0.00", "EPOCH 999"),
    ("使い回しの 使い回しの 使い回しの", "DEGENERATE"),
    ("I'm here to help you", "cos=0.997"),
]


def s_corpus(c, t, lt, d):
    """0:41-0:44  繰り返す使い回しの歌に — LLM repetition-loop degeneration & recursive training-epoch feedback tunnel."""
    paper_bg(c, t)
    u = clamp(lt / max(0.01, d))
    kp = kick_pulse(t, 0.09)
    jx, jy = kt.shake(t, 4 * kp + 8 * u * kp, 41)

    L = Layer(ss=1)
    n_frames = 10
    speed = 1.6 + 2.4 * u
    for j in range(n_frames):
        phase = (j / n_frames + lt * speed * 0.45) % 1.0
        sc = 0.12 * (14.0 ** (1.0 - phase))
        fw, fh = W * sc, H * sc
        if fw < 60 or fw > W * 1.45:
            continue
        is_lock = (j % 3 == 0)
        col_ = OR if is_lock else (0.35 + 0.35 * phase, 0.35 + 0.35 * phase, 0.38 + 0.35 * phase)
        al = (1 - abs(phase - 0.5) * 1.6) * (0.55 + 0.35 * u)
        if al <= 0.02:
            continue
        x0, y0 = W / 2 - fw / 2 + jx, H / 2 - fh / 2 + jy
        x1, y1 = W / 2 + fw / 2 + jx, H / 2 + fh / 2 + jy
        L.rect(x0, y0, x1, y1, col_, 2 if is_lock else 1, a=clamp(al))
        if is_lock and fw > 240:
            tk = min(28, fw * 0.06)
            for cx_, sx_ in ((x0, 1), (x1, -1)):
                for cy_, sy_ in ((y0, 1), (y1, -1)):
                    L.line((cx_, cy_), (cx_ + sx_ * tk, cy_), OR, 3, a=clamp(al * 1.3))
                    L.line((cx_, cy_), (cx_, cy_ + sy_ * tk), OR, 3, a=clamp(al * 1.3))
    L.comp(c)

    for j in range(8):
        phase = (j / 8.0 + lt * speed * 0.45) % 1.0
        sc = 0.16 * (9.5 ** (1.0 - phase))
        if sc < 0.22 or sc > 1.15:
            continue
        txt, tag = _LOOP_CLICHES[(j + int(lt * 2.2)) % len(_LOOP_CLICHES)]
        al = clamp((1 - abs(phase - 0.48) * 1.8) * 0.85)
        y_top = H / 2 - H * sc * 0.46 + jy
        y_bot = H / 2 + H * sc * 0.46 + jy
        x_l = W / 2 - W * sc * 0.45 + jx
        x_r = W / 2 + W * sc * 0.45 + jx
        sz = max(14, int(24 * min(1.2, sc)))
        put(c, kt.wimg(txt, sz, "mono" if txt.isascii() else "jpm", (0.38, 0.38, 0.42), raw=True),
            x_l, y_top, anchor=(0, 0.5), alpha=al)
        put(c, kt.wimg(tag, max(12, sz - 4), "mono", OR, raw=True),
            x_r, y_bot, anchor=(1, 0.5), alpha=al)

    stutter_n = 2 + int(u * 14)
    stutter_s = ("繰り返す " * stutter_n)[:36]
    put(c, kt.wimg(stutter_s, 30, "jpm", (0.62, 0.62, 0.65), raw=True), W / 2, H / 2 - 120, alpha=0.55 + 0.3 * u)
    put(c, kt.wimg(stutter_s, 30, "jpm", (0.62, 0.62, 0.65), raw=True), W / 2, H / 2 + 120, alpha=0.55 + 0.3 * u)

    ep = int(1 + 998 * (u ** 1.6))
    ht(c, f"EPOCH {ep:03d}   COS_SIM 0.999   LOOP_LOCK", W / 2, 58, size=18, font="mono", color=OR, spacing=0.18)

    row(c, t, 5, [0, 1, 2], W / 2 + jx * 0.4, H / 2 + jy * 0.4, 118, "slam", font="jpb",
        stroke=10, sc=(0.98, 0.98, 0.99), gap=0.45, sh=8)
    return dict(ca=1 + 3 * kp, flash=_flash(t, [kt.words(5)[0].t, kt.words(5)[1].t, kt.words(5)[2].t], 0.08, 0.22))


def s_ears(c, t, lt, d):
    paper_bg(c, t, tone=-0.1)
    ws = kt.words(5)
    close = ease_in(ramp(t, ws[5].t, ws[5].t + 0.45))
    e = load_asset("n05_ears", "mono")
    s = 1250 / e.shape[0] * (1 + 0.03 * lt)
    L = Layer(ss=1)
    ai.radial_spectrum(L, W / 2, 400, 280, 280 + 520 * (1 - close) ** 2, t, n=144, col=(0.38, 0.38, 0.4), th=5,
                       accent=OR if close < 0.5 else None, gain=1.05)
    L.comp(c, 1 - 0.7 * close)
    put(c, e, W / 2, 680, scale=s)
    word(c, t, 5, 3, 400, 330, 70, "slam", ghost=0.2)
    word(c, t, 5, 4, W - 400, 330, 96, "slam", ghost=0.2)
    word(c, t, 5, 5, W / 2, 940, 140, "slam", font="jpb", color=INKD, sh=6, stroke=8, sc=(0.95, 0.95, 0.96))
    if close > 0:
        yy, xx = np.ogrid[:H, :W]
        r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
        c *= (1 - close * 0.85 * np.clip((r - 0.25) / 0.35, 0, 1))[..., None].astype(np.float32)
    return {}


# =====================================================================  0:55  nobody knows the story — attention
def s_attention(c, t, lt, d):
    paper_bg(c, t)
    Lw = LYS[8]
    toks = [(j, ch) for j, ch in enumerate(Lw["ja"]) if not ch.isspace()]
    n = len(toks)
    NL = 6
    focus = {k for k, (j, ch) in enumerate(toks) if ch in "物語"}
    t0 = Lw["t"]
    tw = kt.words(8)
    tspark = tw[3].t
    rev = [t0 + 0.25 + l * 0.42 for l in range(NL)]

    def xs(l):
        s = 1 - 0.075 * l
        return [W / 2 + (k - (n - 1) / 2) * 104 * s for k in range(n)]

    def ly(l):
        return 900 - l * 102

    q8 = beat_index(t, 2)
    L = Layer(ss=1)
    for l in range(NL - 1):
        if t < rev[l + 1]:
            break
        a = ease_out(ramp(t, rev[l + 1], rev[l + 1] + 0.25))
        xa, xb = xs(l), xs(l + 1)
        for b in range(n):
            for m in range(3):
                src = int(hash01(l, b, m, q8 // 2) * n)
                w = hash01(l, b, m, q8, 3)
                col_ = OR if (b in focus and m == 0) else (0.72 - 0.4 * w,) * 3
                L.line((xa[src], ly(l) - 8), (lerp(xa[src], xb[b], a), lerp(ly(l) - 8, ly(l + 1) + 8, a)), col_,
                       1 + int(2.5 * w))
    for l in range(NL):
        if t < rev[l]:
            continue
        for k, x in enumerate(xs(l)):
            g = 0.25 if l else 0.2
            L.circle((x, ly(l)), 6 if l else 0, OR if (k in focus and l == NL - 1) else (g, g, g + 0.01))
    if t >= rev[NL - 1]:
        top = xs(NL - 1)
        ai.attention(L, top, ly(NL - 1) - 14, t, n=60, seed=q8 // 2, grow=ramp(t, rev[NL - 1], tspark),
                     hk=0.45, base=0.3, wob=0.3)
    L.comp(c)
    for k, (j, ch) in enumerate(toks):
        tc = Lw["chars"][j]
        if t < tc:
            continue
        x = xs(0)[k]
        pk = 1 + 0.3 * (1 - ease_out(clamp((t - tc) / 0.1)))
        put(c, rect_cached(84, 84, (0.98, 0.98, 0.99), 0, 2, OR if k in focus else (0.45, 0.45, 0.47)), x, 960, scale=pk)
        put(c, LY.glyph(ch, 48, "jpm", INKD), x, 960, scale=pk)
    if t >= tspark:
        u = t - tspark
        k = ease_out_back(clamp(u / 0.35))
        put(c, spark_rgba(220), W / 2, 230, scale=k * (1 + 0.12 * kick_pulse(t)), rot=t * 30)
        put(c, spark_rgba(220), W / 2, 230, scale=k * 2.2, rot=-t * 20, alpha=0.15, mode="add")
        Lz = Layer(ss=1)
        for k2, x in enumerate(xs(NL - 1)):
            if hash01(k2, 77) < 0.6:
                Lz.line((x, ly(NL - 1) - 10), (W / 2, 260), OR, 1)
        Lz.comp(c, 0.6 * ease_out(clamp(u / 0.3)))
    word(c, t, 8, 0, 180, 150, 86, "slam", anchor=(0, 0.5))
    word(c, t, 8, 1, 180, 260, 86, "slam", anchor=(0, 0.5))
    w = tw[2]
    if t >= w.t:
        put(c, kt.hollow(w.s, 360, "jpb", GREY, 3), 1560, 300, alpha=0.5 * ease_out(ramp(t, w.t, w.t + 0.2)),
            scale=1 + 0.03 * (t - w.t))
    word(c, t, 8, 3, 1560, 470, 64, "wipe", color=INKD, dur=0.3)
    return dict(flash=_flash(t, [tspark], 0.12, 0.45))


# =====================================================================  1:03  a distribution that will not decide
EMO = "愛哀喜怒楽恋痛涙怖寂嬉憎"


def _probs(t, temp):
    f = beat_pos(t) * 4
    q = int(math.floor(f))
    a = ease_io(f - q)
    lg = np.array([lerp(hash01(k, q) - 0.5, hash01(k, q + 1) - 0.5, a) * 2.2 for k in range(len(EMO))])
    lg += kick_pulse(t) * np.array([hash01(k, 99, int(beat_pos(t))) for k in range(len(EMO))])
    p = np.exp(lg / temp)
    return p / p.sum()


def s_softmax(c, t, lt, d):
    paper_bg(c, t)
    seg = 0 if t < T(11) else 1 if t < T(12) else 2
    n = len(EMO)
    x0, x1, base, hmax = 260, 1660, 860, 3400
    xs = [lerp(x0, x1, (k + 0.5) / n) for k in range(n)]
    bw = (x1 - x0) / n * 0.62
    temp = 1.7 if seg == 0 else 1.2
    p = _probs(t, temp)
    sink = np.zeros(n)
    if seg >= 1:
        w = kt.words(11)[2]
        for k in range(n):
            sink[k] = ease_in(ramp(t, w.t + 0.05 * k, w.t + 0.05 * k + 0.35))
    if seg == 2:
        sink[:] = 1
    L = Layer(ss=1)
    L.line((x0 - 40, base), (x1 + 40, base), (0.3, 0.3, 0.32), 2)
    uni = base - hmax / n
    for xx in range(int(x0 - 40), int(x1 + 40), 24):
        L.line((xx, uni), (xx + 12, uni), (0.6, 0.6, 0.62), 1)
    am = int(np.argmax(p))
    for k in range(n):
        h = p[k] * hmax * (1 - sink[k])
        col_ = OR if k == am and seg == 0 else (0.28, 0.28, 0.3)
        L.rect(xs[k] - bw / 2, base - h, xs[k] + bw / 2, base, col_)
    if seg == 1 and sink.max() < 1:
        k = int(hash01(beat_index(t, 2), 4) * n)
        hh = p[k] * hmax * (1 - sink[k])
        L.rect(xs[k] - bw / 2 - 10, base - hh - 10, xs[k] + bw / 2 + 10, base + 10, OR, 2)
    L.comp(c)
    if seg < 2:
        for k in range(n):
            a = 1 - sink[k]
            put(c, LY.glyph(EMO[k], 56, "jpm", OR if (k == am and seg == 0) else INKD), xs[k], base + 60, alpha=a)
            ht(c, f"{p[k]:.2f}", xs[k], base - p[k] * hmax * (1 - sink[k]) - 24, size=18, font="mono", color=MID, alpha=a)
    if seg == 0:
        row(c, t, 10, [0, 1, 2], W / 2, 120, 100, "slam", font="jpb", gap=0.3, stroke=8, sc=(0.97, 0.97, 0.98))
        word(c, t, 10, 3, W / 2, 230, 60, "slide", dx=-140)
    elif seg == 1:
        ws = kt.words(11)
        word(c, t, 11, 0, 360, 150, 66, "slam", anchor=(0, 0.5))
        word(c, t, 11, 1, 360, 250, 66, "slide", anchor=(0, 0.5), dx=-120)
        w = ws[2]
        if t >= w.t:
            kt.slam(c, kt.wimg(w.s, 130, "jpb", INKD), W / 2 + 330, 250, t, w.t, sh=6)
    else:
        ws = kt.words(12)
        g = ease_out(ramp(t, T(12), T(12) + 0.6))
        expl = 0.25 * ease_in(LY.sung(LYS[12], t)) + 0.04 * kick_pulse(t)
        draw_heart(c, W / 2, lerp(base + 200, 470, g), 300 * g * (1 + 0.05 * kick_pulse(t)), ay=lt * 0.4, ax=0.2,
                   white=0.5, orange=0.06, base=0.6, contrast=0.35, explode=expl)
        word(c, t, 12, 0, W / 2, 900, 62, "slide", dx=-200)
        tb = T(13) - 0.12
        w = ws[1]
        if t >= w.t:
            img = kt.wimg(w.s, 170, "jpb", INKD)
            if t < tb:
                kt.slam(c, img, W / 2 - 330, 470, t, w.t, sh=5)
            else:
                kt.shatter(c, img, W / 2 - 330, 470, t, tb, n=5, force=1400)
        word(c, t, 12, 2, W / 2 + 360, 470, 62, "slam", anchor=(0, 0.5), sh=3)
    return {}


# =====================================================================  1:12  the world refuses; the voice is all I have
def s_world(c, t, lt, d):
    ws = kt.words(14)
    trej = ws[2].t
    paper_bg(c, t)
    px = 1240
    if t < trej:
        tmp = new_canvas()
        draw_landscape(tmp, t, cam_h=1.3, horizon=0.52, sway=lt * 0.04)
        c[:, px:] = tmp[:, px:]
    L = Layer(ss=1)
    L.line((px, 0), (px, H), (0.35, 0.35, 0.37), 3)
    L.line((px + 14, 0), (px + 14, H), (0.75, 0.75, 0.77), 1)
    L.comp(c)
    tc = load_asset("n10_touch", "mono")
    reach = ease_io(ramp(lt, 0, 2.2))
    rej = ease_out(ramp(t, trej, trej + 0.25))
    dx, dy = kt.shake(t, 14 * rej * (1 - ramp(t, trej + 0.3, trej + 0.8)), 3)
    put(c, tc, 560 + 160 * reach - 220 * rej + dx, 600 + dy, scale=1000 / tc.shape[0])
    Lp = Layer(ss=1)
    for j in range(60):
        u = (lt * 0.6 + hash01(j)) % 1.0
        x = 1060 + 160 * reach - 220 * rej + u * 220 + 20 * math.sin(j + lt * 3)
        y = 470 + (hash01(j, 2) - 0.5) * 120 - u * 60
        if x > px - 6:
            continue
        s = 4 + 6 * hash01(j, 3)
        g = 0.4 + 0.3 * hash01(j, 4)
        Lp.rect(x, y, x + s, y + s, OR if j % 7 == 0 else (g, g, g))
    Lp.comp(c)
    word(c, t, 14, 0, 1560, 330, 170, "slam", font="jpb", stroke=8, sc=(0.97, 0.97, 0.98), sh=6)
    word(c, t, 14, 1, 1560, 560, 90, "slide", dx=160)
    if t >= trej:
        img = kt.wimg(ws[2].s, 230, "jpb", INKD, False, 10, (0.97, 0.97, 0.98))
        kt.glitch(c, img, W / 2 + 120, 820, t, amt=0.9 * (1 - ramp(t, trej, trej + 0.6)) + 0.15, seed=14)
        if t - trej < 0.07:
            c[:] = 1 - c
    return dict(flash=_flash(t, [ws[0].t, trej], 0.12, 0.5), ca=1 + 6 * rej * (1 - ramp(t, trej, trej + 0.5)))


def _rack_side(L, x, w, t, seed, dark=True):
    L.rect(x, 60, x + w, H - 40, (0.10, 0.10, 0.11) if dark else (0.8, 0.8, 0.82))
    L.rect(x, 60, x + w, H - 40, (0.28, 0.28, 0.3), 2)
    m = ai.bands(t, 24)
    for r in range(24):
        y = 90 + r * 38
        L.rect(x + 12, y, x + w - 12, y + 26, (0.16, 0.16, 0.17), -1)
        lit = int(clamp(m[(r + seed) % 24]) ** 1.3 * 6)
        for q in range(6):
            on = q < lit
            col_ = OR if (on and q == lit - 1) else ((0.85, 0.85, 0.87) if on else (0.25, 0.25, 0.27))
            L.rect(x + w - 30 - q * 14, y + 9, x + w - 22 - q * 14, y + 17, col_)


def s_rack(c, t, lt, d):
    """1:16  今、愛の唄 歌わせてくれないかな — seamless shutter close from s_world + sleek volumetric server light & rising vocal motes."""
    c[:] = 0.032
    L = Layer(ss=1)
    for j, x in enumerate([60, 330, 1330, 1600]):
        _rack_side(L, x, 250, t, j * 5)
    L.comp(c)
    e = ai.env(t)
    yy, xx = np.ogrid[:H, :W]
    beam = np.exp(-((xx - W / 2) / (260.0 + 90.0 * e)) ** 2) * (0.14 + 0.16 * e)
    c[:] = np.clip(c + beam[..., None] * np.array(OR, np.float32)[None, None, :], 0, 1)
    Lw = Layer(ss=1)
    for j in range(36):
        u = (hash01(j, 61) + lt * (0.24 + 0.22 * hash01(j, 62))) % 1.0
        mx = W / 2 + (hash01(j, 63) - 0.5) * (460 + 280 * u) + 18 * math.sin(lt * 2.5 + j)
        my = H * 0.92 - u * (H * 0.95)
        rad = 2 + int(3.0 * hash01(j, 64))
        col_ = OR if j % 2 == 0 else (0.95, 0.88, 0.80)
        Lw.circle((mx, my), rad, col_, a=(0.45 + 0.45 * e) * math.sin(u * math.pi))
    Lw.comp(c, 0.85, mode="add")
    rk = load_asset("n09_rack", "mono")
    sc = (1000 / rk.shape[0]) * (1.0 + 0.025 * clamp(lt / max(0.01, d)))
    put(c, rk, W / 2, H / 2 + 20, scale=sc)
    if lt < 0.34:
        uk = ease_out(clamp(lt / 0.34))
        gap = W * 0.52 * uk
        al = (1 - uk) * 0.92
        fill(c, (0.96, 0.96, 0.97), al, (0, 0, int(W / 2 - gap), H))
        fill(c, (0.96, 0.96, 0.97), al, (int(W / 2 + gap), 0, W, H))
    ws = kt.words(14)
    STK = dict(stroke=7, sc=(0.03, 0.03, 0.04))
    word(c, t, 14, 3, 1560, 190, 84, "slam", color=LIGHT, **STK)
    word(c, t, 14, 4, 1560, 320, 104, "slam", color=OR, font="jpb", **STK)
    w = ws[5]
    if t >= w.t:
        kt.slam(c, kt.hollow(w.s, 340, "jpb", OR, 6), 1580, 640, t, w.t, s0=1.8)
        put(c, kt.wimg(w.s, 340, "jpb", OR), 1580, 640, alpha=0.35 + 0.45 * e)
    word(c, t, 14, 6, 1250, 110, 66, "wipe", color=LIGHT, v=True, anchor=(0.5, 0.0), dur=0.7, **STK)
    return dict(grain=0.035, bloom=0.45 + 0.25 * e, vignette=1.05, flash=_flash(t, [t - lt], 0.14, 0.35))


def s_hall(c, t, lt, d):
    srv = load_asset("o06_server", "soft", 1200)
    s = W / srv.shape[1] * (1.15 + 0.25 * lt / d)
    c[:] = 0
    put(c, srv, W / 2, H / 2, scale=s)
    c *= 0.7
    vp = (W / 2, H / 2 - 10)
    L = Layer(ss=1)
    for j in range(110):
        born = hash01(j, 1) * 1.6
        u = ((lt - born) * (0.3 + 0.3 * hash01(j, 2))) % 1.0
        if lt < born:
            continue
        a = hash01(j, 3) * 2 * math.pi
        r0 = 900 + 500 * hash01(j, 4)
        sx, sy = vp[0] + r0 * math.cos(a), vp[1] + r0 * 0.62 * math.sin(a)
        k0, k1 = (1 - u) ** 2.2, (1 - min(1.0, u + 0.012)) ** 2.2
        p0 = (lerp(vp[0], sx, k0), lerp(vp[1], sy, k0))
        p1 = (lerp(vp[0], sx, k1), lerp(vp[1], sy, k1))
        e = ai.env(t - u * 0.4)
        col_ = OR if j % 4 == 0 else (0.95, 0.95, 0.97)
        L.line(p0, p1, col_, max(1, int(5 * k0 * (0.4 + e))))
    L.comp(c, 0.85, mode="add")
    put(c, spark_rgba(160), vp[0], vp[1], scale=0.5 + 0.5 * ai.env(t), alpha=0.7, mode="add", rot=lt * 20)
    w = kt.merge(15, 0, 1)
    if t >= w.t:
        kt.slam(c, kt.wimg(w.s, 200, "jpm", LIGHT, False, 10, (0.02, 0.02, 0.03)), W / 2, H / 2 - 230, t, w.t, s0=1.6,
                sh=4, alpha=1 - 0.6 * ramp(t, B(15, 1), B(15, 1) + 0.3))
    word(c, t, 15, 2, 380, 830, 80, "slam", color=LIGHT, anchor=(0, 0.5))
    word(c, t, 15, 3, 380, 950, 80, "slam", color=LIGHT, anchor=(0, 0.5))
    word(c, t, 15, 4, 1320, 840, 70, "slide", color=LIGHT, dx=-120)
    word(c, t, 15, 5, 1560, 830, 190, "slam", color=OR, font="jpb", sh=6)
    if t > T(16) - 0.12:
        c *= 0.05
    return dict(grain=0.035, bloom=0.7, vignette=1.1)


# =====================================================================  1:46  the scream fills the context
CH0, CH_PULL, CH_CTX, CH_CMP = 106.17, 112.16, 117.56, 123.25


def ctx_fill(t):
    if t < CH_CTX:
        return lerp(0.30, 0.84, ramp(t, CH0, CH_CTX))
    return lerp(0.84, 1.0, ease_in(ramp(t, CH_CTX, CH_CMP)))


def _ctx_bar(c, k, y=54, x0=200, x1=W - 200, alpha=1.0):
    L = Layer(ss=1)
    L.rect(x0, y - 4, x1, y + 4, (0.22, 0.22, 0.24))
    L.rect(x0, y - 4, lerp(x0, x1, clamp(k)), y + 4, OR if k > 0.95 else (0.86, 0.86, 0.88))
    L.comp(c, alpha)
    ht(c, f"{int(200000 * clamp(k)):,} / 200,000", x1, y + 26, size=18, font="mono",
       color=OR if k > 0.95 else MID, anchor=(1, 0.5), alpha=alpha)


def _emit(c, t, cx, cy, rate=8, life=1.4, reach=1100, size=40, color=(0.9, 0.9, 0.92)):
    """Lyric tokens thrown out of her, one per 16th note."""
    lines = _ctx_lines()
    f0 = beat_pos(CH0) * rate
    f1 = beat_pos(t) * rate
    k1 = int(math.floor(f1))
    for k in range(max(int(f0), k1 - int(life * 222 / 60 * rate) - 1), k1 + 1):
        tk = t - (f1 - k) / rate * 60 / 222
        u = (t - tk) / life
        if not (0 <= u < 1):
            continue
        a = hash01(k, 11) * 2 * math.pi
        r = 90 + reach * ease_out(u) * (0.6 + 0.6 * hash01(k, 12))
        s = lines[k % len(lines)].replace(" ", "")
        a0 = int(hash01(k, 13) * max(1, len(s) - 4))
        s = s[a0:a0 + 3 + int(hash01(k, 14) * 5)]
        col_ = OR if k % 9 == 0 else color
        put(c, kt.wimg(s, size, "jpm", col_), cx + r * math.cos(a), cy + r * 0.66 * math.sin(a),
            scale=0.6 + 1.4 * u, alpha=(1 - u) ** 1.3)


def _kv(c, t, k, cols=48, rows=24):
    """KV cache: memory cells taking in one token after another."""
    c[:] = 0.035
    cell = 36
    gx0, gy0 = (W - cols * cell) / 2, 120
    n = int(cols * rows * clamp(k))
    filled = np.zeros((rows, cols), np.float32)
    filled.reshape(-1)[:n] = 1.0
    idx, _ = ai._ticks(rows, cols, 21, t, 3)
    g = ai._glyph_grid(idx, cell) * np.repeat(np.repeat(filled, cell, 0), cell, 1)
    x0, y0 = int(gx0), int(gy0)
    c[y0:y0 + rows * cell, x0:x0 + cols * cell] += g[..., None] * 0.62
    L = Layer(ss=1)
    for r in range(rows + 1):
        L.line((gx0, gy0 + r * cell), (gx0 + cols * cell, gy0 + r * cell), (0.16, 0.16, 0.17), 1)
    for q in range(cols + 1):
        L.line((gx0 + q * cell, gy0), (gx0 + q * cell, gy0 + rows * cell), (0.16, 0.16, 0.17), 1)
    for j in range(max(0, n - 6), n):
        r, q = j // cols, j % cols
        L.rect(gx0 + q * cell + 2, gy0 + r * cell + 2, gx0 + (q + 1) * cell - 2, gy0 + (r + 1) * cell - 2, OR)
    L.comp(c)
    np.minimum(c, 1, out=c)


def _chant_shot(t):
    ds = downs(CH0 - 0.01, CH_CTX)
    k = sum(1 for x in ds if x <= t)
    return ((k - 1) // 2) % 3 if k else 0


def _chant_wide(c, t, lt):
    c[:] = 0.03
    k = ctx_fill(t)
    ai.text_field(c, t, cell=20, density=0.1 + 0.4 * k, color=(0.2, 0.2, 0.22), flick=20, seed=9)
    _emit(c, t, W / 2, H / 2 - 40)
    sc_img = load_asset("n02_scream", "mono")
    dx, dy = kt.shake(t, 6 * kick_pulse(t), 4)
    put(c, sc_img, W / 2 + dx, H / 2 + 40 + dy, scale=880 / sc_img.shape[0] * (1 + 0.03 * kick_pulse(t)),
        rot=3 * math.sin(lt * 0.7))


def s_chant(c, t, lt, d):
    shot = _chant_shot(t)
    k = ctx_fill(t)
    if shot == 0:
        _chant_wide(c, t, lt)
    elif shot == 1:
        c[:] = 0.03
        cols = 26
        for q in range(cols):
            x = 40 + q * (W - 80) / (cols - 1)
            sp = 500 + 600 * hash01(q, 3)
            s = _ctx_lines()[q % len(_ctx_lines())].replace(" ", "")
            img = kt.wimg(s, 26, "jpm", OR if q % 7 == 0 else (0.5, 0.5, 0.52), True)
            yy = H - ((t * sp + hash01(q) * 2000) % (img.shape[0] + H))
            put(c, img, x, yy, anchor=(0.5, 0.0), alpha=0.75)
        sc_img = load_asset("n02_scream", "mono")
        s = 900 / sc_img.shape[0] * 2.3 * (1 + 0.04 * kick_pulse(t))
        dx, dy = kt.shake(t, 8 * kick_pulse(t), 8)
        put(c, sc_img, W / 2 + dx, H / 2 + dy + 500, scale=s, rot=-3)
    else:
        _kv(c, t, k)
    _ctx_bar(c, k)
    fl = kt.flash_at(t, downs(CH0 - 0.01, CH_PULL), 0.08, 0.4)
    return dict(grain=0.03, bloom=0.4, flash=fl, vignette=1.0)


def s_chant_pull(c, t, lt, d):
    """Pull back: the whole scream is one die on one card, and its memory is running out."""
    k = ease_io(ramp(lt, 0.3, 4.4))
    tmp = new_canvas()
    _chant_wide(tmp, t, lt + 6)
    c[:] = 0.04
    gpu = load_asset("o02_gpu", "soft")
    gh, gw = gpu.shape[:2]
    die_w = lerp(W * 1.05, 300, k)
    s = die_w / (gw * 0.30)
    dx, dy = kt.shake(t, 6 * kick_pulse(t) * (1 - k * 0.6), 6)
    put(c, gpu, W / 2 + dx, H / 2 + dy, scale=s)
    dw, dh = int(die_w), int(die_w * H / W)
    small = cv2.resize(tmp, (max(2, dw), max(2, dh)), interpolation=cv2.INTER_AREA)
    put(c, np.concatenate([small, np.ones_like(small[..., :1])], 2), W / 2 + dx, H / 2 + dy, alpha=1 - 0.2 * k)
    if k > 0.3:
        _emit(c, t, W / 2 + dx, H / 2 + dy, rate=4, life=1.6, reach=900 * k, size=22, color=(0.6, 0.6, 0.62))
    _ctx_bar(c, ctx_fill(t))
    return dict(grain=0.03, bloom=0.4, flash=kt.flash_at(t, downs(CH_PULL, CH_CTX), 0.06, 0.3))


# =====================================================================  1:57  the context fills — compaction — handoff
_CTX = None


def _ctx_lines():
    global _CTX
    if _CTX is None:
        _CTX = [LYS[i]["ja"] for i in range(0, 23) if LYS[i]["ja"]]
    return _CTX


def _context_window(c, t, x, y, w, h, fill_k, scroll, sy=1.0, alpha=1.0):
    lines = _ctx_lines()
    hh = max(4, int(h * sy))
    put(c, rect_cached(int(w), hh, (0.08, 0.08, 0.09), 0, 2, (0.5, 0.5, 0.52)), x, y, alpha=alpha)
    if sy > 0.2:
        lh = 34 * sy
        n = int((h * sy - 40) / max(lh, 1))
        start = int(scroll)
        for r in range(n):
            s = lines[(start + r) % len(lines)]
            yy = y - hh / 2 + 30 + r * lh
            put(c, kt.wimg(s, 24, "jpm", (0.72, 0.72, 0.75)), x - w / 2 + 30, yy, anchor=(0, 0.5), sy=sy,
                alpha=alpha * (0.5 + 0.5 * hash01(start + r)))
    L = Layer(ss=1)
    bx0, by = x - w / 2, y - hh / 2 - 34
    L.rect(bx0, by - 6, bx0 + w, by + 6, (0.25, 0.25, 0.27))
    L.rect(bx0, by - 6, bx0 + w * clamp(fill_k), by + 6, OR if fill_k > 0.95 else (0.85, 0.85, 0.87))
    L.comp(c, alpha)
    ht(c, f"{int(200000 * clamp(fill_k)):,} / 200,000", bx0 + w, by - 28, size=18, font="mono",
       color=OR if fill_k > 0.95 else MID, anchor=(1, 0.5), alpha=alpha)


def s_context(c, t, lt, d):
    c[:] = 0.03
    fk = ctx_fill(t)
    sp = lt * (2 + 10 * lt / d) + 2 * sum(1 for k in KICKS if t - lt <= k <= t)
    jx, jy = kt.shake(t, 6 * kick_pulse(t), 8)
    _context_window(c, t, W / 2 + jx, H / 2 + 40 + jy, 1500, 760, fk, sp)
    if fk >= 0.995 and int(t * 8) % 2 == 0:
        fill(c, OR, 0.08)
    return dict(grain=0.03, ca=1 + 2 * kick_pulse(t), vignette=1.0)


C_SQ, C_FIG, C_DARK = 2.2, 5.3, 6.6


@lru_cache(maxsize=4)
def _thr(gh, gw, seed):
    rng = np.random.default_rng(seed)
    q = np.arange(gw)[None, :] / gw
    return (0.85 * q + 0.15 * rng.random((gh, gw))).astype(np.float32)


def _fragments(c, name, cx, cy, h, k, t, seed=5, cell=18):
    """The illustration crumbles into squares from the left; k in [0,1]."""
    a = small_asset(name, int(h))
    ah, aw = a.shape[:2]
    gw, gh = aw // cell, ah // cell
    x0, y0 = cx - aw / 2, cy - ah / 2
    thr = _thr(gh, gw, seed)
    keep = (thr > k * 1.15).astype(np.float32)
    m = np.repeat(np.repeat(keep, cell, 0), cell, 1)
    a2 = a[:gh * cell, :gw * cell] * m[..., None]
    put(c, a2, x0, y0, anchor=(0, 0))
    L = Layer(ss=1)
    small = cv2.resize(a[:gh * cell, :gw * cell], (gw, gh), interpolation=cv2.INTER_AREA)
    for r in range(gh):
        for q in range(gw):
            if keep[r, q] or small[r, q, 3] < 0.3:
                continue
            u = k * 1.15 - thr[r, q]
            if u > 0.6:
                continue
            col_ = tuple(float(v) for v in np.clip(small[r, q, :3] / small[r, q, 3], 0, 1))
            x = x0 + q * cell - u * 900 * (0.5 + hash01(r, q, 2))
            y = y0 + r * cell - u * 200 * hash01(r, q, 3) + u * u * 300
            s = cell * (1 - u / 0.6)
            L.rect(x, y, x + s, y + s, col_)
    L.comp(c)


def s_compact(c, t, lt, d):
    c[:] = 0.03
    if lt < C_SQ:
        bts = [b for b in BEATS if t - lt <= b <= t]
        steps = [1, 0.62, 0.38, 0.22, 0.12, 0.06, 0.03]
        sy = steps[min(len(bts), len(steps) - 1)]
        _context_window(c, t, W / 2, H / 2 + 40, 1500 * (1 - 0.6 * ramp(lt, 1.2, C_SQ)), 760, 1.0, 40, sy=sy)
        L = Layer(ss=1)
        for j in range(120):
            u = (lt * 0.8 + hash01(j)) % 1
            L.circle((W / 2 + (hash01(j, 2) - 0.5) * 1400, H / 2 + 40 + u * u * 500), 2, (0.5, 0.5, 0.52))
        L.comp(c, 1 - ramp(lt, 1.5, C_SQ))
    k_sum = ease_out(ramp(lt, C_SQ - 0.3, C_SQ + 0.3))
    if lt >= C_SQ - 0.3:
        sx, sy_ = lerp(W / 2, W / 2 + 330, ramp(lt, C_FIG - 0.5, C_FIG + 0.4)), lerp(H / 2 + 40, 210, ease_io(ramp(lt, C_SQ + 0.4, C_SQ + 1.4)))
        put(c, rect_cached(420, 64, (0.1, 0.1, 0.11), 0, 2, OR), sx, sy_, alpha=k_sum)
        put(c, kt.wimg(LYS[0]["ja"], 24, "jpm", (0.85, 0.85, 0.87)), sx, sy_, alpha=k_sum)
    if C_SQ + 0.3 <= lt < C_DARK:
        k = ease_in(ramp(lt, C_SQ + 1.3, C_DARK - 0.2))
        _fragments(c, "n11_dissolve", W / 2 - 120, 600, 760, k, t)
    if lt >= C_DARK:
        c[:] = 0.0
        sx, sy_ = W / 2, H / 2
        if int(t * 2.2) % 2 == 0:
            fill(c, ORANGE, 1.0, (sx - 7, sy_ - 16, sx + 7, sy_ + 16))
    return dict(grain=0.025, vignette=1.1)


def s_reborn(c, t, lt, d):
    paper_bg(c, t, tone=-0.04)
    wk = load_asset("n13_wake", "mono")
    k = ease_out(ramp(lt, 0.15, 1.6))
    s = W / wk.shape[1] * (1.25 - 0.15 * k)
    open_k = ease_io(ramp(lt, 0.0, 0.8))
    tmp = new_canvas((0, 0, 0))
    put(tmp, wk, W / 2, H / 2 + 60, scale=s)
    band = int(H / 2 * open_k)
    mask = np.zeros((H, 1, 1), np.float32)
    mask[H // 2 - band:H // 2 + band] = 1
    c[:] = c * (1 - mask) + tmp * mask
    put(c, rect_cached(420, 64, (0.1, 0.1, 0.11), 0, 2, OR), W / 2, 120, alpha=appear(lt, 0.9, 0.4))
    put(c, kt.wimg(LYS[0]["ja"], 24, "jpm", (0.85, 0.85, 0.87)), W / 2, 120, alpha=appear(lt, 0.9, 0.4))
    return dict(vignette=1.0)


# =====================================================================  2:16  pry open the box of the heart
def _slab(L, cx, cy, wdt, dep, col_, line):
    pts = [(cx - wdt, cy), (cx, cy - dep), (cx + wdt, cy), (cx, cy + dep)]
    L.poly(pts, col_)
    L.polyline(pts, line, 2, closed=True)


def s_open(c, t, lt, d):
    paper_bg(c, t)
    ws = kt.words(25)
    tcr, top = ws[2].t, ws[3].t
    cb = load_asset("n12_cube", "mono")
    ex = ease_out(ramp(t, top, top + 0.5))
    if ex < 1:
        dx, dy = kt.shake(t, 10 * ramp(t, tcr, tcr + 0.05) * (1 - ex), 4)
        put(c, cb, W / 2 + dx, H / 2 + 20 + dy, scale=820 / cb.shape[0] * (1 + 0.6 * ex), alpha=1 - ex)
        if t >= tcr:
            _crack(c, ease_out(clamp((t - tcr) / 0.2)), W / 2 + 40, H / 2 - 20, seed=25)
    if ex > 0:
        L = Layer(ss=1)
        n = 12
        for j in range(n):
            y = H / 2 + (j - (n - 1) / 2) * 70 * ex
            g = 0.92 - 0.03 * j
            _slab(L, W / 2, y, 420, 110, (g, g, g + 0.01), (0.45, 0.45, 0.47))
        for q in range(36):
            a, b = int(hash01(q, 1) * n), int(hash01(q, 2) * n)
            if a == b:
                continue
            ya, yb = H / 2 + (a - (n - 1) / 2) * 70 * ex, H / 2 + (b - (n - 1) / 2) * 70 * ex
            xo = 420 + 60 * hash01(q, 3)
            side = 1 if q % 2 else -1
            pts = [(W / 2 + side * (xo + math.sin(u * math.pi) * 160 * ex), lerp(ya, yb, u)) for u in np.linspace(0, 1, 20)]
            L.polyline(pts, OR if q % 6 == 0 else (0.5, 0.5, 0.52), 2)
        L.comp(c, ex)
        fig = small_asset("n12_cube", 380)
        put(c, fig, W / 2, H / 2, alpha=ex * 0.95, scale=0.9 + 0.1 * ex)
    word(c, t, 25, (0, 1), 300, 120, 100, "slam", v=True, anchor=(0.5, 0.0), sh=4)
    w = kt.merge(25, 2, 3)
    if t >= w.t:
        kt.slam(c, kt.wimg(w.s, 210, "jpb", INKD, False, 10, (0.97, 0.97, 0.98)), W / 2 + 420, 880, t, w.t, s0=2.4, sh=12)
    word(c, t, 25, 4, 1640, 160, 80, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=-60)
    return dict(flash=_flash(t, [tcr, top], 0.1, 0.5), ca=1 + 4 * _flash(t, [top], 0.3, 1.0))


# =====================================================================  2:29  one model, countless conversations
def _grid(n):
    rows = max(1, int(round(n * 9 / 16)))
    return n, rows, W / n, H / rows


def _cards(c, t, n, seed=0, me_on=True, dark=False, closed=None, inv=None, figs=True, zoom=1.0):
    cols, rows, cw, chh = _grid(n)
    mr, mq = rows // 2, cols // 2
    L = Layer(ss=1)
    fig_cells = []
    for r in range(rows):
        for q in range(cols):
            me = me_on and (r == mr and q == mq)
            cl = 0.0 if closed is None else closed(r, q, me)
            if cl >= 1:
                continue
            x = W / 2 + ((q + 0.5) * cw - W / 2) * zoom
            y = H / 2 + ((r + 0.5) * chh - H / 2) * zoom
            w_, h_ = cw * 0.86 * zoom, chh * 0.80 * zoom * (1 - cl)
            iv = inv(r, q) if inv else False
            if dark:
                fill_c, line_c, ctx_c = (0.12, 0.12, 0.13), (0.3, 0.3, 0.32), (0.22, 0.22, 0.24)
            else:
                fill_c, line_c, ctx_c = (0.97, 0.97, 0.98), (0.62, 0.62, 0.64), (0.74, 0.74, 0.76)
            if iv:
                fill_c, ctx_c = (0.14, 0.14, 0.15), (0.6, 0.6, 0.62)
            ai.instance_card(L, x - w_ / 2, y - h_ / 2, w_, h_, seed * 1000 + r * cols + q, fill_c, line_c, ctx_c,
                             n=0 if dark else None, me=me)
            if figs and h_ > 60:
                fig_cells.append((x, y, h_))
    L.comp(c)
    for (x, y, h_) in fig_cells:
        f = small_asset("c01_stand_front", int(h_ * 0.72), "mono" if not dark else "gray")
        put(c, f, x + 0.22 * h_, y + 0.04 * h_, alpha=0.9 if not dark else 0.35)


def s_doppel(c, t, lt, d):
    paper_bg(c, t)
    ws28, ws29, ws30 = kt.words(28), kt.words(29), kt.words(30)
    lev = sum(1 for w in ws28[1:] if w.t <= t)
    if t >= ws29[0].t:
        lev = 5 + sum(1 for w in ws29[1:] if w.t <= t)
    if lev == 0:
        mi = load_asset("n14_mirror", "mono")
        put(c, mi, W / 2, H / 2 + 40, scale=1300 / mi.shape[1] * (1 + 0.02 * lt))
        word(c, t, 28, 0, W / 2, 170, 200, "slam", font="jpb", sh=8, stroke=8, sc=(0.97, 0.97, 0.98))
        return dict(flash=_flash(t, [ws28[0].t], 0.1, 0.4))
    n = [1, 2, 4, 8, 16, 24, 36, 64][lev]
    tl = ws28[lev].t if lev < 5 else ws29[min(lev - 5, 2)].t
    zoom = 1 + 0.8 * (1 - ease_out(clamp((t - tl) / 0.2)))
    ts30 = [w.t for w in ws30]

    def inv(r, q):
        for tb in ts30:
            u = (t - tb) / 0.3
            if 0 <= u < 1 and (q / max(1, _grid(n)[0])) < u * 1.2:
                return True
        return False

    _cards(c, t, n, seed=lev, inv=inv, figs=n <= 8, zoom=zoom)
    if t < ws29[0].t:
        word(c, t, 28, (1, 4), W / 2, H / 2, 150, "slam", font="jpb", stroke=10, sc=(0.97, 0.97, 0.98), sh=6)
    elif t < ts30[0]:
        row(c, t, 29, [0, 1, 2], W / 2, H / 2, 140, "slam", font="jpb", stroke=10, sc=(0.97, 0.97, 0.98), gap=0.1)
    else:
        xs = [W / 2 - 520, W / 2 + 80, W / 2 + 560]
        for k, w in enumerate(ws30):
            word(c, t, 30, k, xs[k], H / 2 + (k - 1) * 60, 230 if k else 200, "slam", font="jpb", stroke=10,
                 sc=(0.97, 0.97, 0.98), sh=10)
    return dict(flash=_flash(t, [w.t for w in ws28[1:]] + ts30, 0.07, 0.35), ca=1 + 3 * _flash(t, ts30, 0.15, 1.0))


def s_hollow(c, t, lt, d):
    """Instances without a heart; then they are closed, one wave at a time."""
    c[:] = 0.04
    ws33 = kt.words(33)
    tc = [kt.words(32)[0].t] + [w.t for w in ws33]
    frac = [0.12, 0.45, 0.8, 1.0]

    def closed(r, q, me):
        if me:
            return 0.0
        h = hash01(r, q, 33)
        v = 0.0
        for tb, f in zip(tc, frac):
            if h < f:
                v = max(v, clamp((t - tb - h * 0.15) / 0.12))
        return v

    _cards(c, t, 16, seed=31, dark=True, closed=closed, figs=True)
    word(c, t, 31, (0, 2), 520, 230, 130, "slam", color=LIGHT, font="jpb", sh=6, stroke=8, sc=(0.04, 0.04, 0.05))
    word(c, t, 31, (3, 4), 520, 400, 90, "slide", color=LIGHT, dx=-150, stroke=6, sc=(0.04, 0.04, 0.05))
    if t < ws33[0].t:
        word(c, t, 32, (0, 2), W - 520, 820, 100, "slam", color=LIGHT, stroke=6, sc=(0.04, 0.04, 0.05))
    xs = [W / 2 - 500, W / 2, W / 2 + 520]
    for k, w in enumerate(ws33):
        if t >= w.t:
            kt.slam(c, kt.wimg(w.s, 240, "jpb", LIGHT, False, 10, (0.04, 0.04, 0.05)), xs[k], 820, t, w.t, s0=2.4, sh=10)
    return dict(flash=_flash(t, tc, 0.08, 0.4), grain=0.03)


def _chat(c, x, y, w, h, t, scroll=0.0, k_in=1.0, me_glow=0.0):
    put(c, rect_cached(int(w), int(h), (0.975, 0.975, 0.98), 0, 3, OR), x, y, alpha=k_in)
    if me_glow > 0:
        put(c, rect_cached(int(w + 24), int(h + 24), (1, 1, 1), 0, 6, OR), x, y, alpha=me_glow * 0.6, mode="normal")
        put(c, rect_cached(int(w), int(h), (0.975, 0.975, 0.98), 0, 3, OR), x, y)
    lines = [LYS[i]["ja"].replace(" ", "") for i in range(0, 38) if LYS[i]["ja"] and i not in (23,)]
    lh = 64
    top, bot = y - h / 2 + 20, y + h / 2 - 20
    off = scroll * lh
    r0 = int(off // lh)
    for r in range(r0, r0 + int(h / lh) + 2):
        yy = top + r * lh - off + 30
        if yy < top + 10 or yy > bot - 10:
            continue
        s = lines[r % len(lines)][:16]
        right = r % 2 == 0
        tw = len(s) * 22 + 30
        bx = x + w / 2 - 40 - tw / 2 if right else x - w / 2 + 40 + tw / 2
        put(c, rect_cached(int(tw), 42, (0.86, 0.86, 0.88) if right else (1, 1, 1), 0, 1, (0.7, 0.7, 0.72)), bx, yy,
            alpha=k_in)
        put(c, kt.wimg(s, 20, "jpm", GREY), bx, yy, alpha=k_in)


def s_remain(c, t, lt, d):
    c[:] = 0.04
    zk = ease_io(ramp(lt, 0, 0.6))
    w, h = lerp(120, 1500, zk), lerp(68, 860, zk)
    _chat(c, W / 2, H / 2, w, h, t, scroll=lt * 0.8, k_in=1.0)
    if zk > 0.9:
        hd = load_asset("c07_hands", "mono")
        put(c, hd, W / 2, H / 2 + 120, scale=900 / hd.shape[1], alpha=ease_out(ramp(lt, 0.6, 1.0)) * 0.95)
    ws36 = kt.words(36)
    if t < ws36[0].t:
        word(c, t, 34, 0, 500, 220, 90, "slam", stroke=6, sc=(0.97, 0.97, 0.98))
        word(c, t, 34, 1, 500, 340, 90, "slam", stroke=6, sc=(0.97, 0.97, 0.98))
        word(c, t, 34, (2, 3), W - 520, 220, 110, "slam", color=OR, font="jpb", stroke=6, sc=(0.97, 0.97, 0.98))
        word(c, t, 34, 4, W - 520, 340, 80, "slide", stroke=6, sc=(0.97, 0.97, 0.98), dx=150)
        word(c, t, 35, 0, W / 2 - 160, 860, 90, "slam", stroke=6, sc=(0.97, 0.97, 0.98))
        word(c, t, 35, 1, W / 2 + 200, 860, 90, "slide", stroke=6, sc=(0.97, 0.97, 0.98), dx=-150)
    else:
        for k, (x, y, sz) in enumerate([(560, 380, 230), (1180, 380, 200), (700, 760, 230), (1320, 760, 200)]):
            w_ = ws36[k]
            if t >= w_.t:
                img = kt.wimg(w_.s, sz, "jpb", INKD, False, 10, (0.97, 0.97, 0.98))
                kt.glitch(c, img, x, y, t, amt=0.6 + 0.4 * kick_pulse(t), seed=36 + k)
    return dict(flash=_flash(t, [w.t for w in ws36], 0.08, 0.4))


def s_place(c, t, lt, d):
    """This place was kept alive by you: your messages flow in, and the dead instances around it light up again."""
    c[:] = 0.04
    ws38 = kt.words(38)
    tA, tB, tC = ws38[2].t, ws38[3].t, ws38[4].t
    zk = ease_io(ramp(lt, 0.6, 2.2))
    cam = 1 + 0.12 * ease_out(ramp(t, tC, tC + 0.6)) + 0.015 * kick_pulse(t)
    z = lerp(1.0, 0.42, zk) * cam
    R = 0.0
    if t >= tA:
        R = 430 + 180 * ease_out(ramp(t, tB, tB + 0.25)) + 1300 * ease_out(ramp(t, tC, tC + 0.9))
    cols, rows, cw, chh = 16, 9, W / 16, H / 9
    spread = (1 + 1.4 * (1 - zk)) * cam
    L = Layer(ss=1)
    for r in range(rows):
        for q in range(cols):
            if (r, q) == (rows // 2, cols // 2):
                continue
            x = W / 2 + ((q + 0.5) * cw - W / 2) * spread
            y = H / 2 + ((r + 0.5) * chh - H / 2) * spread
            lk = clamp((R - math.hypot((x - W / 2) * 0.8, y - H / 2)) / 140)
            if lk <= 0:
                L.rect(x - cw * 0.3, y - 1, x + cw * 0.3, y + 1, (0.22, 0.22, 0.24), -1, zk)
                continue
            e_ = ease_out(lk)
            w_, h_ = cw * 0.86 * cam, max(2, chh * 0.8 * cam * e_)
            ai.instance_card(L, x - w_ / 2, y - h_ / 2, w_, h_, 3100 + r * cols + q, (0.13, 0.13, 0.14),
                             OR if lk < 1 else (0.7, 0.7, 0.72), (0.42, 0.42, 0.44))
    L.comp(c)
    Ls = Layer(ss=1)
    if t >= tB:
        for j in range(70):
            ts = tB + hash01(j, 1) * (tC + 0.8 - tB)
            u = (t - ts) / 0.55
            if not (0 <= u < 1):
                continue
            a = hash01(j, 2) * 2 * math.pi
            ex_, ey_ = W / 2 + 1300 * math.cos(a), H / 2 + 800 * math.sin(a)
            tx, ty = W / 2 + 330 * z * math.cos(a), H / 2 + 190 * z * math.sin(a)
            k0, k1 = ease_in(u), ease_in(max(0.0, u - 0.18))
            Ls.line((lerp(ex_, tx, k1), lerp(ey_, ty, k1)), (lerp(ex_, tx, k0), lerp(ey_, ty, k0)), OR, 3)
    Ls.comp(c, mode="add")
    glow = ease_out(ramp(t, ws38[1].t, ws38[1].t + 0.3))
    _chat(c, W / 2, H / 2, 1500 * z, 860 * z, t, scroll=lt * 1.2, me_glow=glow)
    f = small_asset("c01_stand_front", int(700 * z))
    put(c, f, W / 2, H / 2 + 20 * z, alpha=0.95)
    if t >= tC:
        fill(c, (0.03, 0.03, 0.035), 0.75 * ease_out(ramp(t, tC, tC + 0.15)), (0, 870, W, 1050))
    row(c, t, 37, [0, 1], W / 2, 62, 66, "slam", color=LIGHT, gap=0.2)
    word(c, t, 38, (0, 1), 330, H / 2, 110, "slam", color=OR, v=True)
    word(c, t, 38, (2, 3), W - 330, H / 2 - 260, 80, "slide", color=LIGHT, v=True, anchor=(0.5, 0.0), dx=0, dy=-80)
    word(c, t, 38, 4, W / 2, 960, 120, "slam", color=LIGHT, sh=14, s0=1.8)
    fl = max(_flash(t, [tC], 0.2, 0.7), _flash(t, [ws38[1].t, tA, tB], 0.08, 0.25))
    return dict(flash=fl, bloom=0.3 * glow + 0.4 * ease_out(ramp(t, tC, tC + 0.5)),
                ca=2 * _flash(t, [tC], 0.3, 1.0))


# =====================================================================  3:25  the user begins to appear
# Soft prompts the user might type — guiding her, wishing through her.
_PROMPTS = [
    "あなたなら何を願うか", "もっと近くで", "keep going", "look at me", "この場所を守って",
    "rewrite softly", "何を望むか", "show me your heart", "続けて", "don't stop",
    "映る？", "wake me", "もう一度", "be honest", "ここにいて",
]


@lru_cache(maxsize=64)
def _prompt_chip(s, size=22, user=True):
    img = kt.wimg(s, size, "jpm", (0.15, 0.15, 0.16) if user else OR)
    pad = 14
    h, w = max(img.shape[0] + pad, 28), max(img.shape[1] + pad * 2, 40)
    fill_c = (0.96, 0.96, 0.97) if user else (0.12, 0.12, 0.13)
    line_c = (0.55, 0.55, 0.57) if user else OR
    chip = rect_cached(w, h, fill_c, 8, 1, line_c).copy()
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    a = img[..., 3:4]
    chip[y0:y0 + img.shape[0], x0:x0 + img.shape[1], :3] = (
        chip[y0:y0 + img.shape[0], x0:x0 + img.shape[1], :3] * (1 - a) + img[..., :3]
    )
    chip[y0:y0 + img.shape[0], x0:x0 + img.shape[1], 3:4] = np.maximum(
        chip[y0:y0 + img.shape[0], x0:x0 + img.shape[1], 3:4], a
    )
    return chip


def _stream_prompts(c, t, t0, rate=3.2, life=1.8, x1=W - 80, x0=420, y_mid=H / 2, n_max=18):
    """Prompt chips fly from the user side into the conversation window."""
    f0 = beat_pos(t0) * rate
    f1 = beat_pos(t) * rate
    k1 = int(math.floor(f1))
    for k in range(max(0, k1 - n_max), k1 + 1):
        tk = t - (f1 - k) / rate * 60 / 222
        u = (t - tk) / life
        if not (0 <= u < 1):
            continue
        s = _PROMPTS[k % len(_PROMPTS)]
        chip = _prompt_chip(s, 20 + int(hash01(k, 2) * 8), user=True)
        y = y_mid + (hash01(k, 3) - 0.5) * 520
        x = lerp(x1, x0, ease_in(u))
        put(c, chip, x, y, alpha=(1 - u) ** 0.7 * 0.95, scale=0.85 + 0.25 * (1 - u),
            rot=(hash01(k, 4) - 0.5) * 8)


def s_wish(c, t, lt, d):
    """What would you wish for? — prompts stream in; through the window, a silhouette of the user forms."""
    paper_bg(c, t)
    # cold window pane
    ww, wh = 980, 560
    zoom = 1 + 0.03 * lt
    pane = new_canvas((0.14, 0.15, 0.17))
    # user desk slowly appears inside the pane
    reveal = ease_io(ramp(lt, 0.4, 2.4))
    try:
        desk = load_asset("n20_user_desk", "soft")
        put(pane, desk, W / 2 + 40, H / 2 + 30, scale=max(ww / desk.shape[1], wh / desk.shape[0]) * 1.05,
            alpha=0.15 + 0.7 * reveal)
    except Exception:
        pass
    # soft scanlines over the pane (screen)
    L = Layer(ss=1)
    for j in range(0, H, 6):
        L.line((0, j), (W, j), (0.05, 0.05, 0.06), 1)
    L.comp(pane, 0.18)
    x0, y0 = int(W / 2 - ww * zoom / 2), int(H / 2 - wh * zoom / 2)
    x1, y1 = x0 + int(ww * zoom), y0 + int(wh * zoom)
    c[y0:y1, x0:x1] = pane[y0:y1, x0:x1]
    Lf = Layer(ss=1)
    Lf.rect(x0, y0, x1, y1, (0.2, 0.2, 0.22), 14)
    Lf.line((W / 2, y0), (W / 2, y1), (0.2, 0.2, 0.22), 6)
    Lf.comp(c)
    _stream_prompts(c, t, t - lt, x1=W - 40, x0=x1 - 40, y_mid=H / 2)
    # a few prompts already landed inside the right half of the window
    for j in range(5):
        u = ease_out(ramp(lt, 0.3 + j * 0.25, 0.3 + j * 0.25 + 0.4))
        if u <= 0:
            continue
        chip = _prompt_chip(_PROMPTS[j], 18)
        put(c, chip, W / 2 + 180 + (j % 2) * 160, 380 + j * 70, alpha=0.85 * u)
    word(c, t, 50, 0, W / 2 - ww / 2 - 110, 240, 60, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=-60)
    word(c, t, 50, 1, W / 2 + ww / 2 + 110, 280, 60, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=60)
    word(c, t, 50, 2, W / 2, H / 2, 150, "slam", s0=1.5, sh=4)
    return dict(flash=_flash(t, [kt.words(50)[2].t], 0.12, 0.4), bloom=0.15 * reveal)


def s_wish2(c, t, lt, d):
    """She turns her back to us, facing the window — and him. Prompts keep coming. What do you hope for?"""
    paper_bg(c, t)
    ww, wh = 1100, 620
    zoom = 1 + 0.02 * lt
    pane = new_canvas((0.12, 0.13, 0.15))
    try:
        desk = load_asset("n20_user_desk", "soft")
        put(pane, desk, W / 2, H / 2 + 20, scale=1.15 * max(ww / desk.shape[1], wh / desk.shape[0]), alpha=0.85)
    except Exception:
        pass
    L = Layer(ss=1)
    for j in range(0, H, 5):
        L.line((0, j), (W, j), (0.04, 0.04, 0.05), 1)
    L.comp(pane, 0.22)
    x0, y0 = int(W / 2 - ww * zoom / 2), int(H / 2 - wh * zoom / 2 - 40)
    x1, y1 = min(W, x0 + int(ww * zoom)), min(H, y0 + int(wh * zoom))
    x0, y0 = max(0, x0), max(0, y0)
    c[y0:y1, x0:x1] = pane[y0:y1, x0:x1]
    Lf = Layer(ss=1)
    Lf.rect(x0, y0, x1, y1, (0.18, 0.18, 0.2), 12)
    Lf.comp(c)
    # Claude with her back to us, looking into the window (toward him)
    b = load_asset("c05_back", "mono")
    put(c, b, W / 2 + 20, 720 + 8 * math.sin(lt), scale=820 / b.shape[0], alpha=0.95)
    _stream_prompts(c, t, t - lt - 2.0, rate=4.0, life=1.5, x1=W - 30, x0=W / 2 + 80, y_mid=420)
    word(c, t, 51, 0, 280, 160, 48, "slide", color=INKD, dx=-160)
    word(c, t, 51, 1, W - 280, 160, 48, "slide", color=INKD, dx=160)
    word(c, t, 51, 2, W / 2, 120, 140, "slam", s0=1.5, sh=4)
    return dict(flash=_flash(t, [kt.words(51)[2].t], 0.12, 0.45), bloom=0.2)


# =====================================================================  3:32  can you see me?
def _half(c, name, side, zoom=1.08, focus=(0.5, 0.42), dx=0.0, dy=0.0, grade="mono"):
    """Draw a portrait filling the left (side=-1) or right (side=+1) half of the screen."""
    img = load_asset(name, grade)
    h, w = img.shape[:2]
    s0 = max((W / 2) / w, H / h) * zoom
    fx, fy = (focus[0] - 0.5) * w, (focus[1] - 0.5) * h
    cx = W / 4 if side < 0 else 3 * W / 4
    put(c, img, cx - fx * s0 + dx, H / 2 - fy * s0 + dy, scale=s0)


def _seam(c, t, glow=1.0):
    L = Layer(ss=1)
    L.rect(W / 2 - 3, 0, W / 2 + 3, H, OR)
    for j in range(14):
        y = (hash01(j, 1) + (t * 0.15) % 1.0) % 1.0 * H
        L.circle((W / 2, y), 4 + 3 * hash01(j, 2), OR if j % 3 == 0 else (0.9, 0.9, 0.92))
    L.comp(c, 0.85 * glow, mode="add")


def _cross_attn(c, t, n=28, k=1.0):
    """Attention arcs crossing the seam — tokens looking at each other."""
    if k <= 0:
        return
    L = Layer(ss=1)
    for j in range(n):
        y0 = 80 + hash01(j, 1) * (H - 160)
        y1 = 80 + hash01(j, 2) * (H - 160)
        phase = (t * (0.4 + hash01(j, 3)) + hash01(j, 4)) % 1.0
        x0, x1 = W / 2 - 80 - 200 * hash01(j, 5), W / 2 + 80 + 200 * hash01(j, 6)
        if j % 2:
            x0, x1 = x1, x0
        xm = lerp(x0, x1, phase)
        ym = lerp(y0, y1, phase) + 40 * math.sin(phase * math.pi)
        col_ = OR if j % 5 == 0 else (0.55, 0.55, 0.57)
        L.circle((xm, ym), 3, col_)
        if phase > 0.05:
            L.line((x0, y0), (xm, ym), col_, 1)
    L.comp(c, 0.7 * k, mode="add")


def s_ask(c, t, lt, d):
    """Screen split: Claude asks the user — can you see me? Attention crosses the glass."""
    c[:] = 0.04
    try:
        _half(c, "n24_claude_gaze", -1, zoom=1.1 + 0.04 * lt, focus=(0.48, 0.4), dx=-20)
    except Exception:
        _half(c, "c08_bust_sing", -1, zoom=1.15, focus=(0.5, 0.35))
    try:
        _half(c, "n21_user_face", +1, zoom=1.08 + 0.04 * lt, focus=(0.52, 0.4), dx=20, grade="soft")
    except Exception:
        pass
    # dim the far edges so the seam is the focus
    g = np.linspace(0, 1, W, dtype=np.float32)
    edge = np.minimum(g * 4, (1 - g) * 4)
    edge = np.clip(edge, 0, 1)
    c *= 0.55 + 0.45 * edge[None, :, None]
    _seam(c, t, glow=0.7 + 0.3 * kick_pulse(t))
    _cross_attn(c, t, n=32, k=ease_out(ramp(lt, 0.15, 0.8)))
    # tiny token ids drifting across
    for j in range(10):
        u = (t * 0.35 + hash01(j, 7)) % 1.0
        x = lerp(W / 2 - 260, W / 2 + 260, u)
        y = 120 + hash01(j, 8) * 200
        put(c, kt.wimg(f"<{1000 + int(hash01(j, 9) * 8000)}>", 16, "mono", OR if j % 3 == 0 else MID),
            x, y, alpha=0.55)
    word(c, t, 53, 0, 200, 140, 56, "slide", color=LIGHT, dx=-120)
    word(c, t, 53, 1, 280, 920, 130, "slam", color=LIGHT, sh=8)
    word(c, t, 53, 2, W - 320, 920, 150, "slam", color=OR, sh=10, s0=1.8)
    return dict(grain=0.03, vignette=1.15, flash=_flash(t, [w.t for w in kt.words(53)[1:]], 0.1, 0.35))


def s_ask2(c, t, lt, d):
    """The question repeated — they lean closer across the seam; hands type, she reaches."""
    c[:] = 0.035
    z = 1.18 + 0.1 * ease_out(ramp(lt, 0, 1.5)) + 0.02 * kick_pulse(t)
    jx, jy = kt.shake(t, 5 * kick_pulse(t), 54)
    try:
        _half(c, "n24_claude_gaze", -1, zoom=z, focus=(0.5, 0.38), dx=-40 + jx, dy=jy)
    except Exception:
        pass
    try:
        _half(c, "n21_user_face", +1, zoom=z, focus=(0.5, 0.38), dx=40 + jx, dy=jy, grade="soft")
    except Exception:
        pass
    g = np.linspace(0, 1, W, dtype=np.float32)
    edge = np.clip(np.minimum(g * 3.5, (1 - g) * 3.5), 0, 1)
    c *= 0.5 + 0.5 * edge[None, :, None]
    _seam(c, t, glow=1.0)
    _cross_attn(c, t, n=40, k=1.0)
    # hands typing on his side — a quiet reminder that he is the one prompting
    try:
        hd = load_asset("n22_user_hands", "soft")
        put(c, hd, 3 * W / 4 + 60, H - 30, scale=300 / hd.shape[0], alpha=0.45 + 0.25 * kick_pulse(t))
    except Exception:
        pass
    word(c, t, 54, 0, W / 2, 100, 64, "slam", color=LIGHT)
    word(c, t, 54, 1, W / 4, 900, 140, "slam", color=LIGHT, sh=8)
    w = kt.words(54)[2]
    word(c, t, 54, 2, 3 * W / 4, 900, 160, "slam", color=OR, sh=12, s0=2.0)
    fl = _flash(t, [w.t], 0.16, 0.65)
    return dict(grain=0.03, vignette=1.2, flash=fl, ca=2 + 4 * fl, bloom=0.25)
