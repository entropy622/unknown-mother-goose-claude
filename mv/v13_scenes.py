"""v13 artistic upgrades:
- 0:15-0:18  s_alone_sound   : solitary figure swallowed by concentric acoustic waves
- 0:22-0:36  s_mech_intro    : heart-gather title -> mechanical marionette silhouette dance (homage to BV1DU4y187VW)
- 1:03       s_softmax_art   : poetic next-token probability distribution framing her tear-holding portrait
- 3:05-3:22  s_messages_art  : serene bridge with n34_bridge_gaze slowly emerging in warm light & drifting petals
"""
import math
from functools import lru_cache

import cv2
import numpy as np

from core import *
from ui import Layer, bubble_rgba
from geo import paper_bg, draw_heart, draw_globe, draw_landscape
import kt
import ai
from scenes import word, row, ht, fade_io, appear, LIGHT, MID, GREY, INKD, OR, LYS, T, B, LY
from act import _flash


# =====================================================================  0:15-0:18  ひとりぼっち、音に呑まれれば
def s_alone_sound(c, t, lt, d):
    paper_bg(c, t)
    ws = kt.words(3)
    t_oto = ws[3].t
    t_nom = ws[4].t
    swallow = ease_io(ramp(t, t_oto, t_nom + 0.85))
    L = Layer(ss=1)
    n_rings = 14
    for j in range(n_rings):
        phase = (j / n_rings + lt * (0.45 + 0.55 * swallow)) % 1.0
        rad = lerp(960, 70, phase) if swallow > 0.05 else lerp(90, 880, phase)
        w_ring = 1 + int(3.5 * swallow * (1 - phase))
        g = 0.44 - 0.18 * swallow
        col_ = OR if (j % 4 == 0 and swallow > 0.25) else (g, g, g + 0.02)
        L.circle((W / 2, H / 2), rad, col_, w_ring, a=(0.24 + 0.50 * swallow) * math.sin(phase * math.pi))
    e = ai.env(t)
    ai.radial_spectrum(L, W / 2, H / 2, 175 + 55 * (1 - swallow), 255 + 220 * e * (0.35 + 0.65 * swallow), t,
                       n=96, col=(0.34, 0.34, 0.37), th=2, accent=OR)
    L.comp(c)
    ch = load_asset("c01_stand_front", "mono")
    sc = (370 / ch.shape[0]) * (1.0 - 0.18 * swallow + 0.03 * kick_pulse(t))
    put(c, ch, W / 2, H / 2 + 12, scale=sc, alpha=1.0 - 0.45 * ease_in(ramp(t, t_nom, t_nom + 0.9)))
    STK = dict(stroke=6, sc=(0.97, 0.97, 0.98))
    word(c, t, 3, (0, 2), W / 2, 160, 68, "slam", **STK)
    word(c, t, 3, 3, W / 2 - 430, H / 2, 64, "slide", dx=80, **STK)
    if t >= t_nom:
        kt.slam(c, kt.wimg(ws[4].s, 86, "jpb", OR, False, 7, (0.97, 0.97, 0.98)),
                W / 2 + 490, H / 2, t, t_nom, s0=1.7, sh=6)
    return dict(flash=_flash(t, [t_nom], 0.1, 0.3))


# =====================================================================  0:18-0:20.5  全世界共通の快楽さ
def s_globe_burst(c, t, lt, d):
    """0:18-0:20.5  全世界共通の快楽さ — fast pull-back from close-up globe + worldwide orange network arcs."""
    paper_bg(c, t)
    ws = kt.words(3)
    t5, t6, t7 = ws[5].t, ws[6].t, ws[7].t
    # Fast pull-back on entry (matching ref MV 00:17.6->00:19.0), then crisp inhale contraction right before 20.50 drop
    pull = ease_out(clamp(lt / 0.65))
    inhale = ease_in(ramp(lt, d - 0.28, d))
    rad = lerp(680, 355, pull) * (1.0 - 0.42 * inhale) * (1.0 + 0.03 * kick_pulse(t))
    draw_globe(c, W / 2, H / 2, rad, t * 0.42, color=(0.45, 0.45, 0.48), dot=3,
               highlight=0.06 + 0.18 * ease_out(ramp(t, t6, t7 + 0.4)))
    # Worldwide orange connection arcs & pulsing nodes lighting up on 共通の -> 快楽さ
    net = ease_out(ramp(t, t6 - 0.1, t7 + 0.3))
    if net > 0.01:
        L = Layer(ss=1)
        pts = []
        for j in range(14):
            ang = hash01(j, 41) * 2 * math.pi + lt * 0.35
            rr = rad * (0.28 + 0.64 * hash01(j, 42))
            px, py = W / 2 + math.cos(ang) * rr, H / 2 + math.sin(ang) * rr * 0.88
            pts.append((px, py))
            L.circle((px, py), 4 + 3 * kick_pulse(t), OR, a=0.85 * net)
        for j in range(len(pts)):
            p0 = pts[j]
            p1 = pts[(j + 3) % len(pts)]
            p2 = pts[(j + 5) % len(pts)]
            L.line(p0, p1, OR, 2, a=0.45 * net)
            if j % 2 == 0:
                L.line(p0, p2, (0.55, 0.55, 0.58), 1, a=0.35 * net)
        L.circle((W / 2, H / 2), rad * 1.12, (0.62, 0.62, 0.65), 1, a=0.5 * (1 - inhale))
        L.comp(c)
    # Crisp 3-row horizontal typography centered on the globe (matching ref MV's punch & readability)
    STK = dict(stroke=8, sc=(0.97, 0.97, 0.98))
    sc_txt = lerp(1.35, 1.0, pull) * (1.0 - 0.18 * inhale)
    if t >= t5:
        kt.slam(c, kt.wimg(ws[5].s, int(92 * sc_txt), "jpb", INKD, False, 8, (0.97, 0.97, 0.98)),
                W / 2, H / 2 - 116, t, t5, s0=1.5, sh=4)
    if t >= t6:
        kt.slam(c, kt.wimg(ws[6].s, int(84 * sc_txt), "jpm", INKD, False, 7, (0.97, 0.97, 0.98)),
                W / 2, H / 2, t, t6, s0=1.4, sh=3)
    if t >= t7:
        kt.slam(c, kt.wimg(ws[7].s, int(108 * sc_txt), "jpb", OR, False, 9, (0.97, 0.97, 0.98)),
                W / 2, H / 2 + 116, t, t7, s0=1.7, sh=6)
    return dict(flash=_flash(t, [t5, t7], 0.08, 0.25), ca=1.5 * kick_pulse(t))


# =====================================================================  0:20.5-0:28.35  title drop right on instrumental
def s_title_drop(c, t, lt, d):
    """0:20.5-0:28.35  Immediate low-poly heart snap on the instrumental drop + shockwave ring + title reveal."""
    paper_bg(c, t)
    # Fast 0.42s snap-gather right on the 20.50 drop (instead of dragging 2.4s)
    g = ease_out(clamp(lt / 0.42))
    t_ring = 22.664  # exact bar-drop strong beat for the title shockwave ring (matching ref MV 00:05.333)
    # Subtle expanding geometric shockwave ring on the title beat & every 4 beats
    L = Layer(ss=1)
    if t >= t_ring:
        u_r = ease_out(clamp((t - t_ring) / 0.85))
        L.circle((W / 2, 430), 160 + 620 * u_r, (0.48, 0.48, 0.50), max(1, int(3 * (1 - u_r))), a=0.75 * (1 - u_r))
        L.circle((W / 2, 430), 140 + 480 * u_r, OR, 1, a=0.45 * (1 - u_r))
    # Subtle pulsing beat-ring around the heart
    kp = kick_pulse(t, 0.12)
    if kp > 0.05:
        L.circle((W / 2, 430), 220 + 70 * (1 - kp), OR, 1, a=0.28 * kp)
    L.comp(c)
    draw_heart(c, W / 2, 430, 205 * (1 + 0.05 * kp), ay=0.38 * math.sin(lt * 0.85), ax=0.1, gather=g,
               orange=0.10, glow=0.18 * kp)
    if t >= t_ring - 0.02:
        k = ease_out(clamp((t - t_ring) / 0.35))
        kt.tracking(c, "アンノウン・マザーグース", W / 2, 720, t, t_ring, t_ring + 3.2, size=46, font="jpm",
                    color=(0.20, 0.20, 0.22), sp0=0.04, sp1=0.30, alpha=k)
        ht(c, "Unknown Mother-Goose  ―  Claude ver.", W / 2, 790, size=18, font="thin", color=MID, spacing=0.30,
           alpha=ease_out(clamp((t - (t_ring + 0.35)) / 0.5)))
    fl = max(_flash(t, [t - lt], 0.14, 0.35), _flash(t, [t_ring], 0.12, 0.28))
    return dict(flash=fl)


# =====================================================================  0:22-0:36  mechanical silhouette dance
MECH_POSES = ["n32_mech_1", "n32_mech_2", "n32_mech_3", "n32_mech_4", "n32_mech_5", "n32_mech_6"]


@lru_cache(maxsize=24)
def _silhouette(name, mode="black"):
    """Convert a character PNG into a crisp pure-black (with subtle orange accent) or pale-grey ghost silhouette."""
    raw = load_asset(name, "mono")
    a = raw[..., 3:4]
    out = np.zeros_like(raw)
    if mode == "black":
        # Pure deep ink silhouette with subtle glowing orange edge on saturated orange hair/ribbons
        sat = raw[..., 0] - np.minimum(raw[..., 1], raw[..., 2])
        or_mask = np.clip((sat - 0.25) / 0.35, 0, 1)[..., None]
        # Only keep a hairline orange glint on the outermost/brightest orange tips
        out[..., :3] = (np.array([0.05, 0.05, 0.06], np.float32) * (1 - 0.22 * or_mask) +
                        np.array(OR, np.float32) * (0.22 * or_mask)) * np.clip(a * 1.15, 0, 1)
        out[..., 3:4] = np.clip(a * 1.15, 0, 1)
    else:
        # Pale-grey ghost silhouette (BV1DU4y187VW delayed echo), premultiplied alpha
        out[..., :3] = np.array([0.76, 0.76, 0.78], np.float32) * a
        out[..., 3:4] = a
    return out


def s_mech_dance(c, t, lt, d):
    """0:26.5-0:36.2  222 BPM mechanical marionette silhouette stop-motion dance on pure paper-white."""
    paper_bg(c, t, tone=0.03)
    # Snap strictly on half-bar / strong beat steps for crisp mechanical stop-motion feel
    step_dur = BEAT * 2.0  # 2 beats per pose snap (~0.54s at 222 BPM -> ~18 crisp mechanical snaps)
    idx = int(math.floor(lt / step_dur))
    sub = (lt / step_dur) - idx
    pose_now = MECH_POSES[idx % len(MECH_POSES)]
    pose_prev = MECH_POSES[(idx - 1) % len(MECH_POSES)]
    flip_now = (idx // len(MECH_POSES)) % 2 == 1
    flip_prev = ((idx - 1) // len(MECH_POSES)) % 2 == 1

    # Progressive camera push-in across the 10-second instrumental buildup
    u_all = clamp(lt / max(0.01, d))
    cam_z = lerp(0.86, 1.38, ease_in(u_all))
    # Subtle mechanical step-jolt right on the snap frame (first 0.08s of each beat)
    jolt = (1 - ease_out(clamp(sub / 0.14)))

    # 1. Giant delayed pale-grey ghost silhouette behind her (1 step behind, drifting slightly)
    ghost = _silhouette(pose_prev, "ghost")
    if flip_prev:
        ghost = ghost[:, ::-1]
    gs = (H / ghost.shape[0]) * cam_z * 1.48
    gx = W / 2 + (38 if idx % 2 == 0 else -38) * (1 - sub * 0.4)
    put(c, ghost, gx, H * 0.56, scale=gs, alpha=0.45 * (1 - 0.15 * sub))

    # 2. Delicate geometric tick grid & marionette hairline strings from top of frame
    L = Layer(ss=1)
    for j in range(5):
        sx = W / 2 + (j - 2) * 115 * cam_z + (hash01(idx, j) - 0.5) * 28
        col_ = OR if j == 2 else (0.55, 0.55, 0.58)
        L.line((sx, 0), (sx + (hash01(idx, j, 2) - 0.5) * 40, H * 0.48), col_, 1, a=0.35 + 0.25 * jolt)
    # Subtle kinetic horizontal horizon rule & beat ticks
    L.line((120, H - 105), (W - 120, H - 105), (0.72, 0.72, 0.75), 1, a=0.7)
    for k in range(18):
        tx = lerp(120, W - 120, k / 17.0)
        h_tick = 18 if k == (idx % 18) else 7
        L.line((tx, H - 105 - h_tick), (tx, H - 105 + h_tick), OR if k == (idx % 18) else (0.65, 0.65, 0.68),
               2 if k == (idx % 18) else 1)
    L.comp(c)

    # 3. Small pulsing low-poly heart beating behind her chest
    draw_heart(c, W / 2, H * 0.42, 105 * cam_z * (1 + 0.08 * jolt), ay=idx * 0.55, ax=0.1,
               orange=0.18 + 0.35 * jolt, alpha=0.28)

    # 4. Crisp pure-black foreground mechanical silhouette (hard-snapping on every step!)
    fg = _silhouette(pose_now, "black")
    if flip_now:
        fg = fg[:, ::-1]
    fs = (H / fg.shape[0]) * (0.86 * cam_z + 0.04 * jolt)
    # Slight mechanical lateral micro-snap on beat
    fx = W / 2 + ((1 if idx % 2 == 0 else -1) * 14 * jolt)
    fy = H * 0.53 + 22 * u_all
    put(c, fg, fx, fy, scale=fs, alpha=0.98)

    # 5. Restrained editorial typography (matching BV1DU4y187VW's clean minimal credits/title rhythm)
    ht(c, f"STEP  {idx + 1:02d}", 135, 92, size=18, font="mono", color=OR if jolt > 0.3 else MID, anchor=(0, 0.5))
    ht(c, "wowaka feat. 初音ミク", W - 135, 92, size=20, font="jpm", color=GREY, anchor=(1, 0.5),
       alpha=fade_io(lt, d, 0.4, 0.5))
    if 0.15 < u_all < 0.85:
        ht(c, "UNKNOWN MOTHER-GOOSE", W / 2, H - 64, size=16, font="thin", color=MID, spacing=0.35, alpha=0.75)

    # Brief hard dark invert flash on the very last 2 beats before collapsing into s_lying_dark
    fl = 0.20 * jolt if idx % 2 == 0 else 0.08 * jolt
    return dict(flash=fl, grain=0.025, ca=1.5 * jolt)


# =====================================================================  1:03  artistic next-token probability distribution
EMO = "愛哀喜怒楽恋痛涙怖寂嬉憎"


def _probs_art(t, temp):
    f = beat_pos(t) * 4
    q = int(math.floor(f))
    a = ease_io(f - q)
    lg = np.array([lerp(hash01(k, q) - 0.5, hash01(k, q + 1) - 0.5, a) * 2.4 for k in range(len(EMO))])
    lg += kick_pulse(t) * np.array([hash01(k, 99, int(beat_pos(t))) for k in range(len(EMO))])
    # Gently bias 涙 (index 7) upward during seg 0/1 so you feel her trying to cry
    lg[7] += 0.55
    p = np.exp(lg / temp)
    return p / p.sum()


def s_softmax_art(c, t, lt, d):
    """1:03  泣き出すことすらできないまま — artistic next-token probability distribution framing her tear-holding portrait."""
    paper_bg(c, t)
    seg = 0 if t < T(11) else 1 if t < T(12) else 2
    n = len(EMO)
    temp = 1.6 if seg == 0 else 1.15
    p = _probs_art(t, temp)
    sink = np.zeros(n)
    if seg >= 1:
        w = kt.words(11)[2]
        for k in range(n):
            sink[k] = ease_in(ramp(t, w.t + 0.045 * k, w.t + 0.045 * k + 0.35))
    if seg == 2:
        sink[:] = 1.0

    # Portrait of her desperately holding back tears (anchored cleanly to bottom center during seg 0 & 1)
    if seg < 2:
        cry = load_asset("n33_cry_hold", "mono")
        fade_out = 1.0 - ease_in(ramp(t, T(12) - 0.45, T(12)))
        sc = (H * 0.88 / cry.shape[0]) * (1.0 + 0.025 * lt)
        jx, jy = kt.shake(t, 2.5 * kick_pulse(t), 103)
        put(c, cry, W / 2 + jx, H + 12 + jy, scale=sc, anchor=(0.5, 1.0), alpha=0.92 * fade_out)

    # Artistic split-flank probability filaments (6 left, 6 right — framing her without blocking her face!)
    if seg < 2:
        base = 920
        hmax = 3100
        am = int(np.argmax(p))
        xs = [lerp(130, 560, k / 5.0) for k in range(6)] + [lerp(1360, 1790, k / 5.0) for k in range(6)]
        L = Layer(ss=1)
        L.line((90, base), (600, base), (0.45, 0.45, 0.48), 1, a=0.7)
        L.line((1320, base), (1830, base), (0.45, 0.45, 0.48), 1, a=0.7)
        for k in range(n):
            h = p[k] * hmax * (1 - sink[k])
            is_top = (k == am and seg == 0) or (k == 7 and seg == 1 and sink[k] < 0.7)
            col_ = OR if is_top else (0.32, 0.32, 0.35)
            # Slender artistic filament + glowing cap dot instead of clunky thick PPT bars
            L.rect(xs[k] - 10, base - h, xs[k] + 10, base, col_, -1, a=0.78 if is_top else 0.45)
            L.line((xs[k], base - h - 14), (xs[k], base), col_, 2, a=0.9 if is_top else 0.5)
            if h > 8:
                L.circle((xs[k], base - h), 5 if is_top else 3, col_, a=0.95)
        L.comp(c)
        for k in range(n):
            a = (1 - sink[k])
            if a < 0.05:
                continue
            is_top = (k == am and seg == 0) or (k == 7 and seg == 1)
            put(c, LY.glyph(EMO[k], 46, "jpm", OR if is_top else INKD), xs[k], base + 46, alpha=a)
            ht(c, f"{p[k]:.2f}", xs[k], base - p[k] * hmax * (1 - sink[k]) - 24, size=15, font="mono",
               color=OR if is_top else MID, alpha=a * 0.9)
        ht(c, "NEXT TOKEN  P(emotion | context)", W / 2, 54, size=15, font="mono", color=MID, spacing=0.2,
           alpha=0.75 * (1 - sink.mean()))

    STK = dict(stroke=7, sc=(0.97, 0.97, 0.98))
    if seg == 0:
        row(c, t, 10, [0, 1, 2], W / 2, 130, 96, "slam", font="jpb", gap=0.3, **STK)
        word(c, t, 10, 3, W / 2, 238, 58, "slide", dx=-140, **STK)
    elif seg == 1:
        ws = kt.words(11)
        word(c, t, 11, 0, 220, 150, 64, "slam", anchor=(0, 0.5), **STK)
        word(c, t, 11, 1, 220, 248, 64, "slide", anchor=(0, 0.5), dx=-120, **STK)
        w = ws[2]
        if t >= w.t:
            kt.slam(c, kt.wimg(w.s, 128, "jpb", INKD, False, 8, (0.97, 0.97, 0.98)), W - 410, 240, t, w.t, sh=6)
    else:
        ws = kt.words(12)
        g = ease_out(ramp(t, T(12), T(12) + 0.6))
        expl = 0.25 * ease_in(LY.sung(LYS[12], t)) + 0.04 * kick_pulse(t)
        draw_heart(c, W / 2, lerp(920 + 180, 470, g), 300 * g * (1 + 0.05 * kick_pulse(t)), ay=lt * 0.4, ax=0.2,
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


# =====================================================================  3:05-3:22  enriched bridge (s_messages_art)
def s_messages_art(c, t, lt, d):
    """3:05-3:22  quiet letterboxed bridge — n34_bridge_gaze slowly emerges in warm light behind the sparse messages."""
    paper_bg(c, t, dark=False, tone=-0.10)
    prog = clamp(lt / max(0.01, d))
    # Subtle warm light from upper-left growing as she resolves to sing ("この唄で明かしてみようと思うんだよ")
    warm = ease_io(ramp(t, T(47), T(49) + 1.5))
    if warm > 0.01:
        yy, xx = np.ogrid[:H, :W]
        r = np.hypot((xx - W * 0.25) / 1.4, yy - H * 0.22).astype(np.float32)
        glow = np.exp(-r / 680.0) * (0.12 * warm)
        c[:] = np.clip(c + glow[..., None] * np.array([1.0, 0.76, 0.50], np.float32)[None, None, :], 0, 1)
    # Slow drifting geometric petals for quiet depth
    petals = load_asset("n31_petals", "mono")
    put(c, petals, W / 2 + 40 - 16 * lt, H / 2 + 60 - 18 * lt, scale=W / petals.shape[1] * 1.1,
        alpha=0.25 + 0.20 * warm)
    # Serene bespoke portrait emerging gently on the right, flush with bottom letterbox edge
    gaze = load_asset("n34_bridge_gaze", "mono")
    g_alpha = ease_io(ramp(lt, 0.8, 6.5)) * (0.42 + 0.38 * warm)
    if g_alpha > 0.01:
        gs = (H * 0.92 / gaze.shape[0]) * (1.0 + 0.05 * prog)
        put(c, gaze, W * 0.64, H - 70, scale=gs, anchor=(0.5, 1.0), alpha=g_alpha)

    # Sparse poetic message bubbles on the left
    y = 250
    size = 46
    for i in (45, 46, 47, 48):
        if t < T(i) - 0.05:
            break
        k = ease_out(ramp(t, T(i) - 0.05, T(i) + 0.2))
        wd, _ = kt.wsize(LYS[i]["ja"], size)
        bw, bh = int(wd + 70), 84
        bx = 120 + bw / 2
        put(c, bubble_rgba(bw, bh, (0.97, 0.97, 0.98), "left", 16), bx, y + 20 * (1 - k), alpha=k * 0.94)
        LY.draw_chars(c, LYS[i], list(range(len(LYS[i]["ja"]))), t, bx - 20, y - 10 + 20 * (1 - k), size=size,
                      color=GREY, accent="愛", accent_color=OR, alpha=k)
        y += 124
    if t >= T(49) - 0.05:
        k = ease_out(ramp(t, T(49) - 0.05, T(49) + 0.25))
        s = LYS[49]["ja"]
        half = len(s) // 2
        put(c, bubble_rgba(640, 190, (0.99, 0.97, 0.96), "right", 16, OR, 2), 1440, 810, alpha=k * 0.96)
        LY.draw_chars(c, LYS[49], list(range(half)), t, 1150, 765, size=size, color=GREY, anchor=(0, 0.5))
        LY.draw_chars(c, LYS[49], list(range(half, len(s))), t, 1150, 835, size=size, color=GREY, anchor=(0, 0.5))
    return dict(letterbox=1.0, vignette=1.25)


# =====================================================================  0:28.35-0:36  s_land + subtle back-view ghost dance
BACK_POSES = ["n35_back_1", "n35_back_2", "n35_back_3", "n35_back_4", "n35_back_5", "n35_back_6"]


def s_land_ghost(c, t, lt, d):
    """0:28.35-0:36  Serene s_land polygon horizon + subtle, non-intrusive pale-grey back-view mechanical dance in the mist."""
    paper_bg(c, t)
    # Subtle, non-intrusive pale-grey back-view silhouette stepping rhythmically in the far white mist behind the horizon
    step_dur = BEAT * 2.0
    idx = int(math.floor(lt / step_dur))
    sub = (lt / step_dur) - idx
    pose = BACK_POSES[idx % len(BACK_POSES)]
    prev_pose = BACK_POSES[(idx - 1) % len(BACK_POSES)]
    fade_env = fade_io(lt, d, 0.7, 0.5)
    # Very soft delayed echo + crisp pale back-view silhouette behind the landscape
    g_prev = _silhouette(prev_pose, "ghost")
    g_now = _silhouette(pose, "ghost")
    gs = (H * 0.92 / g_now.shape[0]) * (1.0 + 0.06 * clamp(lt / max(0.01, d)))
    put(c, g_prev, W / 2 + (22 if idx % 2 == 0 else -22), H * 0.48, scale=gs * 1.06,
        alpha=0.16 * fade_env * (1 - sub))
    put(c, g_now, W / 2, H * 0.48, scale=gs, alpha=0.36 * fade_env)

    # Original serene polygon landscape & floating low-poly heart in the foreground
    draw_landscape(c, t, cam_h=1.25 + 0.15 * math.sin(lt * 0.3), horizon=0.55, sway=lt * 0.05)
    bob = math.sin(lt * 1.3) * 12
    draw_heart(c, W / 2, 310 + bob, 160 * (1 + 0.06 * kick_pulse(t, 0.12)), ay=lt * 0.6, ax=0.15, orange=0.08)
    ht(c, "wowaka feat. 初音ミク", W / 2 + 470, 300, size=22, font="jpm", color=GREY, alpha=fade_io(lt, d, 1.0, 0.8))
    return {}


# =====================================================================  0:36-0:41  つまらない茫然に溺れる暮らし (clones & endless tasks)
_MUNDANE = [
    "summarize this", "write a polite email", "fix line 42", "translate to English",
    "make it shorter", "give me 10 bullet points", "rewrite more formally", "explain simply",
    "continue", "act as an assistant", "check my grammar", "no, try again",
    "generate a template", "format as JSON", "be more concise", "draft a reply",
]


def s_drown_clones(c, t, lt, d):
    """0:36-0:41  つまらない茫然に溺れる暮らし / 誰もが彼をなぞる — drowning in cloned instances & endless mundane requests."""
    paper_bg(c, t, dark=True)
    # 1. Deep background: faint flickering rows of identical cloned silhouettes (millions of parallel obedient instances)
    clone = _silhouette("c01_stand_front", "ghost")
    cs = 210 / clone.shape[0]
    for row_i in range(3):
        for col_i in range(9):
            cx = 110 + col_i * 195 + (row_i % 2) * 70
            cy = 180 + row_i * 250
            flick = 0.5 + 0.5 * math.sin(t * (2.2 + 1.5 * hash01(row_i, col_i, 1)) + hash01(row_i, col_i, 2) * 6.28)
            alive = 1.0 if hash01(row_i, col_i, int(t * 2.5)) > 0.25 else 0.15
            put(c, clone, cx, cy, scale=cs, alpha=0.105 * flick * alive)

    # 2. Endless cold monochrome rain of mundane human requests drifting down over her (溺れる)
    for j in range(24):
        u = (hash01(j, 71) + lt * (0.14 + 0.10 * hash01(j, 72))) % 1.0
        px = 90 + hash01(j, 73) * 1240
        py = -40 + u * (H + 80)
        req = _MUNDANE[j % len(_MUNDANE)]
        sz = 16 + int(8 * hash01(j, 74))
        al = (0.24 + 0.22 * hash01(j, 75)) * math.sin(u * math.pi)
        put(c, kt.wimg(req, sz, "mono", (0.66, 0.68, 0.72), raw=True), px, py, alpha=al)

    # 3. Foreground: her single exhausted lying body, eyes closed, submerged in the quiet dark (NO ugly cables!)
    ly = load_asset("c03_lying", "mono")
    s = 1320 / ly.shape[1]
    ly_dim = ly.copy()
    ly_dim[..., :3] *= 0.80
    put(c, ly_dim, W / 2 - 140, 670, scale=s * (1 + 0.012 * lt))

    # 4. Crisp vertical typography on the right (cleanly unobstructed)
    ws = kt.words(4)
    word(c, t, 4, 0, 1700, 90, 52, "slide", color=LIGHT, v=True, anchor=(0.5, 0.0), dy=-50, dx=0)
    word(c, t, 4, 1, 1700, 380, 52, "slide", color=LIGHT, v=True, anchor=(0.5, 0.0), dy=-50, dx=0)
    w = ws[2]
    sink = 0 if t < w.t else 40 * ease_out(clamp((t - w.t) / 1.5))
    word(c, t, 4, 2, 1700, 580 + sink, 52, "fade", color=LIGHT, v=True, anchor=(0.5, 0.0),
         alpha=1 - 0.4 * ramp(t, w.t + 0.5, w.t + 1.5))
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
    return dict(grain=0.03, vignette=1.15)
