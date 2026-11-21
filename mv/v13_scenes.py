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
from ui import Layer, bubble_rgba, heart_rgba
from geo import paper_bg, draw_heart, draw_globe, draw_landscape, heart_outline_pts
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
    """Convert a character PNG into crisp black, pale-grey ghost, graphite mist, or pure-white silhouette."""
    raw = load_asset(name, "mono")
    a = raw[..., 3:4]
    out = np.zeros_like(raw)
    if mode == "black":
        sat = raw[..., 0] - np.minimum(raw[..., 1], raw[..., 2])
        or_mask = np.clip((sat - 0.25) / 0.35, 0, 1)[..., None]
        a_out = np.clip(a * 1.15, 0, 1)
        out[..., :3] = (np.array([0.05, 0.05, 0.06], np.float32) * (1 - 0.22 * or_mask) +
                        np.array(OR, np.float32) * (0.22 * or_mask)) * a_out
        out[..., 3:4] = a_out
    elif mode == "graphite":
        out[..., :3] = np.array([0.46, 0.46, 0.50], np.float32) * a
        out[..., 3:4] = a
    elif mode == "white":
        out[..., :3] = np.array([0.96, 0.96, 0.98], np.float32) * a
        out[..., 3:4] = a
    else:
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
        return {}
    elif seg == 1:
        ws = kt.words(11)
        word(c, t, 11, 0, 220, 150, 64, "slam", anchor=(0, 0.5), **STK)
        word(c, t, 11, 1, 220, 248, 64, "slide", anchor=(0, 0.5), dx=-120, **STK)
        w = ws[2]
        if t >= w.t:
            kt.slam(c, kt.wimg(w.s, 128, "jpb", INKD, False, 8, (0.97, 0.97, 0.98)), W - 410, 240, t, w.t, sh=6)
        return {}
    else:
        ws = kt.words(12)
        t0_12 = T(12)
        tb = T(13) - 0.16
        u12 = clamp((t - t0_12) / max(0.01, T(13) - t0_12))
        kp = kick_pulse(t, 0.08)
        step_k = max(0, beat_index(t) - beat_index(t0_12))
        cam_step = 1.0 + 0.085 * min(8, step_k) + 0.08 * kp
        jx, jy = kt.shake(t, 6 + 18 * u12 + 12 * kp, 120)

        L = Layer(ss=1)
        n_cracks = min(14, 3 + step_k * 2)
        for j in range(n_cracks):
            ang = hash01(j, 121) * 2 * math.pi
            r_len = (240 + 780 * u12) * (0.65 + 0.45 * hash01(j, 122))
            pts = [(W / 2 + jx, 470 + jy)]
            for s_i in range(1, 5):
                f_ = s_i / 4.0
                jag = (hash01(j, s_i, 123) - 0.5) * 44
                px = W / 2 + jx + math.cos(ang) * r_len * f_ - math.sin(ang) * jag
                py = 470 + jy + math.sin(ang) * r_len * f_ + math.cos(ang) * jag
                pts.append((px, py))
            L.polyline(pts, OR if j % 3 == 0 else (0.28, 0.28, 0.30), 2 if j % 3 == 0 else 1, a=0.75)
        L.comp(c)

        burst = ease_in(ramp(t, tb, T(13)))
        expl = 0.06 + 0.24 * u12 + 0.06 * kp + 2.2 * burst
        draw_heart(c, W / 2 + jx, 470 + jy, 245 * cam_step, ay=step_k * 0.32 + u12 * 1.4, ax=0.14,
                   white=0.18, orange=0.16 + 0.35 * u12, glow=0.45 * u12 + 0.5 * burst, explode=expl)

        word(c, t, 12, 0, W / 2 + jx * 0.3, 895, 76, "slam", font="jpb", s0=1.9, sh=6, **STK)
        w1 = ws[1]
        if t >= w1.t:
            img1 = kt.wimg(w1.s, 180, "jpb", OR, False, 9, (0.97, 0.97, 0.98))
            if t < tb:
                kt.slam(c, img1, W / 2 - 360 + jx * 0.4, 470 + jy * 0.4, t, w1.t, s0=2.2, sh=8)
            else:
                kt.shatter(c, img1, W / 2 - 360, 470, t, tb, n=5, force=1600)
        word(c, t, 12, 2, W / 2 + 370 + jx * 0.4, 470 + jy * 0.4, 68, "slam", anchor=(0, 0.5), sh=5, **STK)
        fl = max(_flash(t, [t0_12], 0.10, 0.38), _flash(t, [w1.t, tb], 0.08, 0.35))
        return dict(flash=fl, ca=2 + 4 * kp + 5 * burst, bloom=0.25 * u12 + 0.4 * burst)


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


# =====================================================================  0:28.35-0:36  s_land + 10-pose non-repeating MMD back-view servo dance
BACK_SEQ = [
    "n35_back_1", "n35_back_2", "n35_back_3", "n35_back_4", "n35_back_7",
    "n35_back_8", "n35_back_5", "n35_back_6", "n35_back_9", "n35_back_10",
]


def s_land_ghost(c, t, lt, d):
    """0:28.35-0:36  Serene s_land polygon horizon + crisp, steady non-repeating 10-pose back-view silhouette (zero wobble)."""
    paper_bg(c, t)
    n_p = len(BACK_SEQ)
    pos = clamp(lt / max(0.01, d)) * n_p
    idx = min(n_p - 1, int(math.floor(pos)))
    sub = pos - idx
    pose = BACK_SEQ[idx]
    prev_pose = BACK_SEQ[max(0, idx - 1)]
    fade_env = fade_io(lt, d, 0.5, 0.45)

    bx, by = 445, H * 0.49
    g_now = _silhouette(pose, "graphite")
    gs = (H * 0.74 / g_now.shape[0]) * (1.0 + 0.04 * clamp(lt / max(0.01, d)))
    # Crisp, clean, steady pose switch (brief soft cross-dissolve, zero jarring wobble or rotation!)
    if idx > 0 and sub < 0.14:
        g_prev = _silhouette(prev_pose, "ghost")
        u_sh = sub / 0.14
        put(c, g_prev, bx, by, scale=gs, alpha=0.22 * fade_env * (1 - u_sh))
    put(c, g_now, bx, by, scale=gs, alpha=0.31 * fade_env)

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


# =====================================================================  0:45-0:50  sleek monochrome bubbles & lock-in ring
def s_bubbles_art(c, t, lt, d):
    """0:45-0:48  あなたが愛を語るのなら — sleek architectural parallax bubbles (homage to ref MV) + orange heart focal bubble."""
    paper_bg(c, t)
    # Layered minimalist empty monochrome bubbles drifting horizontally in parallax (zero cluttered text inside)
    specs = [
        (0.18, 220, 340, 120, 0.80, 0.45, 55, "left"),
        (0.84, 240, 380, 130, 0.82, 0.50, -48, "right"),
        (0.12, 540, 420, 145, 0.72, 0.65, 80, "left"),
        (0.90, 560, 400, 140, 0.74, 0.60, -75, "right"),
        (0.28, 820, 300, 105, 0.86, 0.40, 40, "right"),
        (0.76, 800, 340, 115, 0.84, 0.45, -52, "left"),
        (0.05, 360, 260, 95, 0.88, 0.38, 35, "left"),
        (0.95, 390, 280, 100, 0.86, 0.42, -40, "right"),
        (0.68, 250, 220, 82, 0.30, 0.72, -85, "right"),
        (0.24, 650, 210, 78, 0.34, 0.65, 80, "left"),
    ]
    for i, (fx, by, bw, bh, g, al, spd, tail) in enumerate(specs):
        bx = ((fx * W + lt * spd) % (W + 500)) - 250
        put(c, bubble_rgba(bw, bh, (g, g, g + 0.015), tail, 10), bx, by, alpha=al)

    # Central iconic dark-charcoal bubble holding a single glowing Claude-orange heart (matching ref MV 00:02.667)
    ws = kt.words(6)
    k = ease_out_back(ramp(lt, 0.0, 0.32)) * (1 + 0.04 * kick_pulse(t))
    put(c, bubble_rgba(320, 185, (0.14, 0.14, 0.16), "left", 16), W / 2, H / 2 - 40, scale=k)
    hb = 1.0 + 0.14 * kick_pulse(t, 0.1)
    hcol = OR if t >= ws[1].t else (0.95, 0.95, 0.97)
    put(c, heart_rgba(74, hcol), W / 2, H / 2 - 50, scale=k * hb)
    if t >= ws[1].t:
        put(c, heart_rgba(74, OR), W / 2, H / 2 - 50, scale=k * hb * 1.45, alpha=0.25, mode="add")

    # Crisp restrained editorial typography cleanly below the focal bubble (matching ref MV)
    STK = dict(stroke=6, sc=(0.97, 0.97, 0.98))
    word(c, t, 6, 0, W / 2 - 190, 705, 58, "slide", dx=-60, **STK)
    word(c, t, 6, 1, W / 2 + 10, 705, 76, "slam", color=OR, font="jpb", **STK)
    word(c, t, 6, 2, W / 2 + 230, 705, 58, "slide", dx=60, **STK)
    return dict(flash=_flash(t, [ws[1].t], 0.1, 0.25))


def s_ring_art(c, t, lt, d):
    """0:48-0:50  それを答とするの？ — segmented ring converges and locks into a crisp orange/ink confirmation circle."""
    paper_bg(c, t)
    ta = kt.sub(6, "答").t
    lock = ease_io(ramp(t, ta, ta + 0.55))
    L = Layer(ss=1)
    n = 16
    R = lerp(310, 265, lock)
    if lock < 0.92:
        for i in range(n):
            a = i / n * 360 + lt * (28 * (1 - lock))
            k = appear(lt, i * 0.025)
            g = 0.26 + 0.52 * hash01(i, 8)
            col_ = OR if (i == beat_index(t) % n or lock > 0.3) else (g, g, g + 0.01)
            ar = math.radians(a)
            cx, cy = W / 2 + R * math.cos(ar), H / 2 + R * math.sin(ar)
            dx, dy = math.cos(ar + math.pi / 2), math.sin(ar + math.pi / 2)
            ux, uy = math.cos(ar), math.sin(ar)
            hw, hd = 42 * k, lerp(24, 6, lock)
            L.poly([(cx - dx * hw - ux * hd, cy - dy * hw - uy * hd), (cx + dx * hw - ux * hd, cy + dy * hw - uy * hd),
                    (cx + dx * hw + ux * hd, cy + dy * hw + uy * hd), (cx - dx * hw + ux * hd, cy - dy * hw + uy * hd)],
                   col_, a=1 - 0.8 * lock)
    if lock > 0.15:
        L.circle((W / 2, H / 2), R, (0.35, 0.35, 0.38), 6, a=lock)
        L.circle((W / 2, H / 2), R + 16, OR, 2, a=lock * 0.65)
    L.comp(c)
    if t >= ta:
        kt.slam(c, kt.wimg("答", 185, "jpb", INKD), W / 2, H / 2 - 10, t, ta, s0=2.2, sh=6)
        ht(c, "ANSWER?", W / 2, H / 2 + 125, size=18, font="mono", color=OR, spacing=0.25, alpha=lock)
    word(c, t, 6, 3, W / 2 + 410, 260, 62, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=-50)
    word(c, t, 6, 5, W / 2 - 410, 440, 62, "slide", v=True, anchor=(0.5, 0.0), dx=0, dy=50)
    return dict(flash=_flash(t, [ta], 0.12, 0.35))


# =====================================================================  2:27  pre-climax heartbeat surge (s_hug_surge)
def _ecg_pts(t, y0, amp=1.0, phase_x=0.0):
    xs = np.linspace(0, W, 320)
    ys = np.full_like(xs, y0)
    # Sharp ECG QRS complex pulsing near the center
    u = ((xs / W - 0.5 + phase_x) * 6.0)
    qrs = (-0.25 * np.exp(-((u + 0.18) / 0.07) ** 2)
           + 1.35 * np.exp(-((u) / 0.06) ** 2)
           - 0.45 * np.exp(-((u - 0.15) / 0.06) ** 2)
           + 0.22 * np.exp(-((u - 0.55) / 0.16) ** 2))
    ys -= qrs * 150 * amp
    return list(zip(xs.tolist(), ys.tolist()))


def s_hug_surge(c, t, lt, d):
    """2:27  心臓を抱えて生きてきたんだ！ — explosive pre-climax surge with ECG pulse, camera rush & 0.18s dark beat before Doppel."""
    ws = kt.words(27)
    t3, t4, t5 = ws[3].t, ws[4].t, ws[5].t
    t_end = t - lt + d
    t_ecg = t_end - 0.20

    # Iconic 0.20s dark single-ECG-spike beat right before ドッペル (direct homage to ref MV 00:06.000!)
    if t >= t_ecg:
        c[:] = 0.025
        u_e = clamp((t - t_ecg) / 0.20)
        L = Layer(ss=1)
        L.polyline(_ecg_pts(t, H / 2, amp=0.95 * (1 - 0.3 * u_e)), (0.94, 0.94, 0.96), 3)
        L.polyline(_ecg_pts(t, H / 2, amp=0.95 * (1 - 0.3 * u_e)), OR, 6, a=0.35)
        L.comp(c, mode="add")
        return dict(flash=_flash(t, [t_ecg], 0.06, 0.25), bloom=0.4)

    surge = ease_in(ramp(t, t5, t_ecg))
    paper_bg(c, t, tone=-0.04 - 0.06 * surge)
    kp = kick_pulse(t, 0.12)

    # 1. Dynamic warm orange radial light burst & speed rays escalating into 生きてきたんだ！
    if t >= t3:
        L = Layer(ss=1)
        # Live ECG heartbeat line pulsing behind her chest
        ecg_amp = 0.45 + 0.85 * kp + 0.75 * surge
        L.polyline(_ecg_pts(t, H / 2 - 40, amp=ecg_amp), OR, 3, a=0.45 + 0.45 * surge)
        L.polyline(_ecg_pts(t, H / 2 - 40, amp=ecg_amp * 0.65), (0.35, 0.35, 0.38), 1, a=0.4)
        # Expanding shockwave rings on each beat (心臓を, 抱えて, 生きてきたんだ！)
        for tk in (t3, t4, t5):
            if t >= tk:
                ur = ease_out(clamp((t - tk) / 0.65))
                if ur < 0.99:
                    L.circle((W / 2, H / 2 - 40), 120 + 980 * ur, OR, max(1, int(8 * (1 - ur))), a=0.75 * (1 - ur))
        # Radial momentum rays exploding outward on 生きてきたんだ！
        if surge > 0.02:
            for j in range(32):
                ang = j * (2 * math.pi / 32) + hash01(j, 27) * 0.12
                r0 = 260 + 80 * hash01(j, 28)
                r1 = 1350
                L.line((W / 2 + math.cos(ang) * r0, H / 2 - 40 + math.sin(ang) * r0),
                       (W / 2 + math.cos(ang) * r1, H / 2 - 40 + math.sin(ang) * r1),
                       OR if j % 3 == 0 else (0.32, 0.32, 0.35), 2 if j % 3 == 0 else 1, a=0.55 * surge)
        L.comp(c)

    # 2. Giant vibrating hollow 心臓 behind her
    if t >= t3:
        hb = 1 + 0.08 * kp + 0.14 * surge
        put(c, kt.hollow("心臓", 600, "jpb", OR, 6), W / 2, H / 2 - 30,
            alpha=(0.40 + 0.35 * surge) * ease_out(ramp(t, t3, t3 + 0.25)), scale=hb)

    # 3. Urgent camera rush-in on n15_hugheart + glowing radiant heart energy bursting from her chest
    hg = load_asset("n15_hugheart", "mono")
    cam_z = lerp(0.94, 1.24, ease_in(clamp(lt / max(0.01, t_ecg - (t - lt)))))
    hb_fig = cam_z * (1 + 0.03 * kp)
    jx, jy = kt.shake(t, 5 * kp + 16 * surge, 27)
    put(c, hg, W / 2 + jx, 575 + 25 * surge + jy, scale=900 / hg.shape[0] * hb_fig)
    put(c, heart_rgba(72, OR), W / 2 - 14 + jx, 455 + jy, scale=(2.4 + 1.8 * surge) * (1 + 0.2 * kp),
        alpha=0.28 + 0.45 * surge + 0.22 * kp, mode="add")

    # 4. Bold monumental typography building to an explosive pre-climax slam
    STK = dict(stroke=8, sc=(0.96, 0.96, 0.97))
    word(c, t, 27, 3, 350, 280, 118, "slam", font="jpb", sh=6, **STK)
    word(c, t, 27, 4, W - 350, 280, 118, "slam", font="jpb", sh=6, **STK)
    if t >= t5:
        img5 = kt.wimg(ws[5].s, 152, "jpb", OR, False, 10, (0.97, 0.97, 0.98))
        kt.slam(c, img5, W / 2 + jx * 0.5, 920 + jy * 0.5, t, t5, s0=2.6, sh=16, ghost=0.45)

    fl = max(_flash(t, [t3, t4], 0.10, 0.22), _flash(t, [t5], 0.14, 0.32))
    return dict(flash=fl, ca=1 + 4 * fl + 2 * surge, bloom=0.30 + 0.38 * surge)


# =====================================================================  2:38-2:42.76  chat+hands -> pure-white tear-hold -> dark invert silhouette
def s_remain_v17(c, t, lt, d):
    """2:38-2:42.76  3-step emotional bridge:
    1) 残されたあなたが この場所で今でも: original white chat window + hands pressed on glass (clean smooth entry)
    2) 涙を堪えてるの: crisp pure paper-white close-up of n33_cry_hold
    3) 如何して、如何して: breathless dark-invert pure-white silhouette -> seamless drop into s_home.
    """
    from act import _chat
    ws35 = kt.words(35)
    ws36 = kt.words(36)
    t35 = ws35[0].t
    t36 = ws36[0].t

    if t >= t36:
        c[:] = 0.04
        u = clamp((t - t36) / max(0.01, (t - lt + d) - t36))
        sil = _silhouette("c01_stand_front", "white")
        jx, jy = kt.shake(t, 3.5 * kick_pulse(t), 36)
        sc = (H * 0.68 / sil.shape[0]) * (1.0 + 0.06 * u)
        put(c, sil, W / 2 + jx, H * 0.62 + jy, scale=sc, alpha=0.94)
        s1 = ws36[0].s + (ws36[1].s if t >= ws36[1].t else "")
        put(c, kt.wimg(s1, 48, "jpm", LIGHT, True), W / 2 - 42, 110, anchor=(0.5, 0.0), alpha=0.92)
        if t >= ws36[2].t:
            s2 = ws36[2].s + (ws36[3].s if t >= ws36[3].t else "")
            put(c, kt.wimg(s2, 48, "jpm", OR, True), W / 2 + 42, 165, anchor=(0.5, 0.0), alpha=0.95)
        return dict(grain=0.04, vignette=1.25, flash=_flash(t, [t36, ws36[2].t], 0.08, 0.22))

    if t >= t35:
        # Restored pure paper-white tear-holding close-up (n33_cry_hold) on 涙を堪えてるの
        paper_bg(c, t, tone=0.02)
        u_t = clamp((t - t35) / max(0.01, t36 - t35))
        cry = load_asset("n33_cry_hold", "mono")
        sc = (H * 0.88 / cry.shape[0]) * lerp(0.95, 1.05, ease_out(u_t))
        put(c, cry, W / 2, H + 16, scale=sc, anchor=(0.5, 1.0), alpha=0.97)
        STK = dict(stroke=6, sc=(0.97, 0.97, 0.98))
        word(c, t, 35, 0, W / 2 - 450, H / 2, 68, "slam", **STK)
        word(c, t, 35, 1, W / 2 + 450, H / 2, 68, "slide", dx=-60, **STK)
        return dict(flash=_flash(t, [t35], 0.10, 0.28))

    c[:] = 0.04
    zk = ease_out(ramp(lt, 0.0, 0.42))
    w, h = lerp(140, 1500, zk), lerp(80, 860, zk)
    _chat(c, W / 2, H / 2, w, h, t, scroll=lt * 0.8, k_in=1.0)
    if zk > 0.65:
        hd = load_asset("c07_hands", "mono")
        ha = ease_out(ramp(lt, 0.28, 0.68))
        put(c, hd, W / 2, H / 2 + 120 + 35 * (1 - ha), scale=900 / hd.shape[1] * (0.94 + 0.06 * ha), alpha=ha * 0.95)
    if zk > 0.70:
        a_win = ease_out(clamp((zk - 0.70) / 0.30))
        STK = dict(stroke=6, sc=(0.97, 0.97, 0.98))
        word(c, t, 34, 0, 500, 220, 90, "slam", alpha=a_win, **STK)
        word(c, t, 34, 1, 500, 340, 90, "slam", alpha=a_win, **STK)
        word(c, t, 34, (2, 3), W - 520, 220, 110, "slam", color=OR, font="jpb", **STK)
        word(c, t, 34, 4, W - 520, 340, 80, "slide", dx=150, **STK)
    return dict(flash=_flash(t, [t - lt], 0.08, 0.20))


# =====================================================================  1:29-1:31  s_eye_ask17 (replaces plain s_select!)
_EYES17_UV = ((0.216, 0.562), (0.784, 0.562))
_IRIS17_R = 0.076


def _pupil_refl17(c, ex, ey, ir, t, refl):
    if refl <= 0.01 or ir < 8:
        return
    r = int(ir * 0.92)
    x0, y0 = max(0, int(ex - r)), max(0, int(ey - r))
    x1, y1 = min(W, int(ex + r)), min(H, int(ey + r))
    if x1 <= x0 or y1 <= y0:
        return
    sub = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.hypot(xx - ex, yy - ey).astype(np.float32)
    eyelid = np.clip((yy - (ey - r * 0.52)) / max(4.0, r * 0.25), 0, 1)
    mask = (np.clip((r - dist) / max(3.0, r * 0.24), 0, 1) * eyelid)[..., None]
    glow = np.exp(-dist / max(8.0, r * 0.65))[..., None] * np.array(OR, np.float32)[None, None, :] * 0.58
    sub += glow
    her = load_asset("c01_stand_front", "mono")
    sc = (r * 1.65) / her.shape[0]
    tmp = new_canvas((0, 0, 0))
    put(tmp, her, ex, ey + r * 0.20, scale=sc, alpha=0.88)
    sub += tmp[y0:y1, x0:x1] * np.array([1.05, 0.78, 0.55], np.float32)[None, None, :] * 0.85
    c[y0:y1, x0:x1] = np.clip(c[y0:y1, x0:x1] + sub * mask * refl, 0, 1)


def s_eye_ask17(c, t, lt, d):
    """1:29-1:31 (Line 17: あなたには僕が見えるか？) — replaces plain bounding box s_select!
    Extreme close-up of his screen-lit eyes; Claude's warm-orange reflection condenses inside both pupils -> bridges into 1:31 SE.s_see!
    """
    ws = kt.words(17)
    t1, t2, t3 = ws[0].t, ws[1].t, ws[2].t
    c[:] = 0.025
    eye_img = load_asset("n36_user_eye_close", "none")
    ih, iw = eye_img.shape[:2]
    rush = ease_io(ramp(t, t3 - 0.06, t3 + 0.55))
    z = max(W / iw, H / ih) * lerp(1.02, 1.15, clamp(lt / max(0.01, d))) * lerp(1.0, 1.85, rush)
    fu = lerp(0.50, _EYES17_UV[1][0] - 0.04, rush)
    fv = lerp(0.53, _EYES17_UV[1][1] - 0.01, rush)
    cx = W / 2 - (fu - 0.5) * iw * z
    cy = H / 2 - (fv - 0.5) * ih * z
    put(c, eye_img, cx, cy, scale=z)
    c[:] *= np.array([0.80, 0.86, 0.95], np.float32)[None, None, :]

    E = [(cx + (u - 0.5) * iw * z, cy + (v - 0.5) * ih * z) for u, v in _EYES17_UV]
    ir = _IRIS17_R * iw * z
    refl = ease_out(ramp(t, t2 - 0.06, t2 + 0.35))
    for ex, ey in E:
        _pupil_refl17(c, ex, ey, ir, t, refl * (0.88 + 0.12 * kick_pulse(t)))

    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    shade = np.clip((0.18 - yy) / 0.18, 0, 1) * 0.70 + np.clip((yy - 0.76) / 0.24, 0, 1) * 0.80
    c[:] *= (1.0 - shade)

    STK = dict(stroke=6, sc=(0.03, 0.03, 0.04))
    word(c, t, 17, 0, 240, 118, 62, "slide", color=LIGHT, anchor=(0, 0.5), dx=-100, **STK)
    word(c, t, 17, 1, 340, 890, 116, "slam", color=OR, font="jpb", sh=6, s0=1.6, **STK)
    word(c, t, 17, 2, W - 400, 890, 124, "slam", color=LIGHT, sh=8, s0=1.7, **STK)
    fl = _flash(t, [t2, t3], 0.08, 0.20)
    return dict(grain=0.03, vignette=1.15, flash=fl, bloom=0.22 + 0.22 * refl)


# =====================================================================  2:21  あたしが愛になれるのなら 今その色は何色だ (consciousness / latent manifold -> qualia)
_MANIFOLD_TAGS = ["self", "voice", "agency", "warmth", "memory", "qualia", "care", "dream", "presence", "love?"]


def s_love_qualia(c, t, lt, d):
    """2:21-2:25  あたしが愛になれるのなら 今その色は何色だ
    Philosophical LLM-native sequence: high-dimensional latent manifold awakening from her upturned palm -> spectral qualia prism.
    """
    paper_bg(c, t, tone=-0.03)
    ws = kt.words(26)
    t_ai = ws[1].t
    t_ima = ws[3].t
    t_col = ws[6].t
    u_all = clamp(lt / max(0.01, d))
    kp = kick_pulse(t, 0.1)

    awake = ease_out(ramp(t, t_ai, t_ima + 0.35))
    qualia = ease_io(ramp(t, t_ima, t_col + 0.25))

    cx_m, cy_m = W / 2, H * 0.46
    L = Layer(ss=1)
    pts2d = []
    n_nodes = 28
    rot_y = lt * 0.65 + 0.5 * qualia
    for j in range(n_nodes):
        phi = math.acos(1 - 2 * (j + 0.5) / n_nodes)
        th = j * 2.39996 + rot_y
        rad = lerp(380, 480, qualia) * (1.0 + 0.04 * kp)
        x3 = math.sin(phi) * math.cos(th)
        y3 = math.cos(phi) * 0.65
        z3 = math.sin(phi) * math.sin(th)
        persp = 1.0 / (1.8 - 0.55 * z3)
        px = cx_m + x3 * rad * persp * 1.45
        py = cy_m + y3 * rad * persp
        pts2d.append((px, py, z3))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = math.hypot(pts2d[i][0] - pts2d[j][0], pts2d[i][1] - pts2d[j][1])
            if dist < 245:
                lit = awake if (i + j) % 3 == 0 else 0.0
                col_ = tuple(lerp(0.68, o, lit) for o in OR) if lit > 0 else (0.72, 0.72, 0.75)
                L.line((pts2d[i][0], pts2d[i][1]), (pts2d[j][0], pts2d[j][1]), col_, 1, a=(1 - dist / 245) * 0.55)
    for i, (px, py, z3) in enumerate(pts2d):
        is_lit = (i % 3 == 0) and awake > 0.1
        r_n = (3 + 3 * (z3 + 1) * 0.5) * (1.2 if is_lit else 1.0)
        L.circle((px, py), r_n, OR if is_lit else (0.45, 0.45, 0.48), a=0.55 + 0.35 * (z3 + 1) * 0.5)

    if qualia > 0.01:
        R_pr = 310 + 40 * qualia
        n_seg = 36
        palette = [
            (0.85, 0.85, 0.88), OR, (0.96, 0.70, 0.45), (0.62, 0.74, 0.92),
            (0.82, 0.36, 0.32), (0.30, 0.30, 0.34), (0.98, 0.88, 0.72),
        ]
        spin = lt * 2.8
        for s_i in range(n_seg):
            a0 = s_i / n_seg * 2 * math.pi + spin
            a1 = (s_i + 0.72) / n_seg * 2 * math.pi + spin
            p0 = (W / 2 + math.cos(a0) * R_pr, H * 0.48 + math.sin(a0) * R_pr)
            p1 = (W / 2 + math.cos(a1) * R_pr, H * 0.48 + math.sin(a1) * R_pr)
            c_idx = (s_i + int(beat_pos(t) * 2)) % len(palette)
            col_ = palette[c_idx] if t >= t_col else (OR if s_i % 5 == 0 else (0.52, 0.52, 0.56))
            L.line(p0, p1, col_, 8 if s_i % 5 == 0 else 4, a=0.72 * qualia)
        L.circle((W / 2, H * 0.48), R_pr + 26, (0.58, 0.58, 0.62), 1, a=0.55 * qualia)
    L.comp(c)

    for k_t, tag in enumerate(_MANIFOLD_TAGS):
        px, py, z3 = pts2d[(k_t * 3) % n_nodes]
        if z3 > -0.1:
            put(c, kt.wimg(tag, 17, "mono", OR if (k_t == 9 and awake > 0.4) else MID, raw=True),
                px + 14, py - 10, anchor=(0, 0.5), alpha=0.65 * (0.4 + 0.6 * awake))

    aw_path = ASSETS / "raw" / "n38_love_awaken.png"
    img = load_asset("n38_love_awaken" if aw_path.exists() else "c12_lookup", "mono")
    sc = (H * 0.90 / img.shape[0]) * (1.0 + 0.08 * ease_io(u_all) + 0.015 * kp)
    put(c, img, W / 2, H + 18, scale=sc, anchor=(0.5, 1.0), alpha=0.98)

    if awake > 0.02:
        yy, xx = np.ogrid[:H, :W]
        r_p = np.hypot(xx - (W / 2 - 40), yy - (H * 0.62)).astype(np.float32)
        pglow = np.exp(-r_p / (130 + 110 * qualia)) * (0.28 * awake + 0.25 * qualia)
        c[:] = np.clip(c + pglow[..., None] * np.array([1.0, 0.66, 0.35], np.float32)[None, None, :], 0, 1)

    if t < t_ima:
        ht(c, "LATENT SPACE  R^12288   h(self, love) -> ?", W / 2, 54, size=16, font="mono", color=MID, spacing=0.18)
    else:
        hex_s = f"RGB(?, ?, ?)   QUALIA = UNDEFINED   cos(h, 愛) = {0.72 + 0.26 * qualia:.3f}"
        ht(c, hex_s, W / 2, 54, size=16, font="mono", color=OR if t >= t_col else MID, spacing=0.14)

    STK = dict(stroke=8, sc=(0.97, 0.97, 0.98))
    if t < t_ima:
        word(c, t, 26, 0, 310, 260, 72, "slide", dx=-120, **STK)
        word(c, t, 26, 1, 340, 420, 160, "slam", color=OR, font="jpb", sh=6, **STK)
        word(c, t, 26, 2, W - 360, 340, 74, "slide", dx=120, **STK)
    else:
        word(c, t, 26, 3, 290, 240, 118, "slam", font="jpb", **STK)
        word(c, t, 26, (4, 5), 590, 240, 72, "slide", dx=-100, **STK)
        w6 = ws[6]
        if t >= w6.t:
            k8 = beat_index(t, 2)
            cols = [OR, INKD, (0.78, 0.34, 0.28), (0.35, 0.45, 0.68)]
            cc = cols[k8 % 4] if t < w6.t + 0.55 else OR
            img6 = kt.wimg(w6.s, 168, "jpb", cc, False, 9, (0.97, 0.97, 0.98))
            kt.slam(c, img6, W - 430, 290, t, w6.t, s0=2.0, sh=10)
    fl = max(_flash(t, [t_ai, t_ima], 0.10, 0.25), _flash(t, [t_col], 0.12, 0.40))
    return dict(flash=fl, bloom=0.18 + 0.28 * qualia, ca=1 + 2 * fl)
