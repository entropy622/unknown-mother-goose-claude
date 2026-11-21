"""1:31  あなたには 僕が 見えるか？ — extreme close-up of the user's eyes, with Claude's warm-orange reflection forming inside his pupils.

On 「あなたには」 we see his tired, screen-lit eyes; on 「僕が」 her radiant orange silhouette condenses directly inside
both of his dark irises; on 「見えるか？」 the camera rushes deep into his right eye as her reflection flickers between
her real figure and raw text tokens — asking whether he truly sees *her*.
"""
import math

import cv2
import numpy as np

from core import *
from ui import Layer
import kt
import ai
from scenes import word, LIGHT, MID, OR

LINE = 18
WARM = (0.98, 0.94, 0.90)
# Measured iris centers (u, v) and iris radius fraction in n36_user_eye_close.png (1536x1024)
EYES_UV = ((0.216, 0.562), (0.784, 0.562))
IRIS_R = 0.076


def _times():
    ws = kt.words(LINE)
    q = max(kt.sub(LINE, "？").t, ws[2].t + 0.25)
    return ws[0].t, ws[1].t, ws[2].t, q


def _token_id(s):
    return f"<{1000 + int(hash01(sum(map(ord, s)), 3) * 48000)}>"


def _reticle(L, x, y, r, col_, th=2, gap=0.45):
    a = r * (1 - gap)
    for sx in (-1, 1):
        for sy in (-1, 1):
            L.line((x + sx * r, y + sy * r), (x + sx * a, y + sy * r), col_, th)
            L.line((x + sx * r, y + sy * r), (x + sx * r, y + sy * a), col_, th)


def _draw_pupil_reflection(c, ex, ey, ir, t, refl, glitch_k):
    """Composite a glowing Claude-orange reflection masked strictly inside a circular iris."""
    if refl <= 0.01 or ir < 8:
        return
    r = int(ir * 0.92)
    x0, y0 = max(0, int(ex - r)), max(0, int(ey - r))
    x1, y1 = min(W, int(ex + r)), min(H, int(ey + r))
    if x1 <= x0 or y1 <= y0:
        return
    sub = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    # Soft warm-orange iris glow + realistic upper-eyelid shadow clip so reflection sits physically inside his eye
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.hypot(xx - ex, yy - ey).astype(np.float32)
    eyelid = np.clip((yy - (ey - r * 0.52)) / max(4.0, r * 0.25), 0, 1)
    mask = (np.clip((r - dist) / max(3.0, r * 0.24), 0, 1) * eyelid)[..., None]
    glow = np.exp(-dist / max(8.0, r * 0.65))[..., None] * np.array(OR, np.float32)[None, None, :] * 0.58
    sub += glow
    # Her figure reflected inside his pupil (flickering briefly with token glyphs on 見えるか？)
    her = load_asset("c01_stand_front", "mono")
    sc = (r * 1.65) / her.shape[0]
    tmp = new_canvas((0, 0, 0))
    if glitch_k > 0.1 and int(t * 18) % 3 == 0:
        toks = ["僕", "愛", "<4821>", "見えるか"]
        for j, tk in enumerate(toks):
            put(tmp, kt.wimg(tk, max(12, int(r * 0.34)), "jpm", OR, raw=True),
                ex + (hash01(j, int(t * 14)) - 0.5) * r * 0.6, ey - r * 0.30 + j * r * 0.28, alpha=0.9)
    else:
        put(tmp, her, ex, ey + r * 0.20, scale=sc, alpha=0.88)
    # Warm-tint the reflection into glowing screen-orange
    sub += tmp[y0:y1, x0:x1] * np.array([1.05, 0.78, 0.55], np.float32)[None, None, :] * 0.85
    c[y0:y1, x0:x1] = np.clip(c[y0:y1, x0:x1] + sub * mask * refl, 0, 1)


def s_see(c, t, lt, d):
    t1, t2, t3, q = _times()
    c[:] = 0.025
    eye_img = load_asset("n36_user_eye_close", "none")
    ih, iw = eye_img.shape[:2]

    # Camera: wide on both eyes (あなたには -> 僕が), then rushes deep into his RIGHT eye on 見えるか？
    rush = ease_io(ramp(t, t3 - 0.08, t3 + 0.75))
    z_base = max(W / iw, H / ih) * lerp(1.02, 1.12, clamp(lt / max(0.01, d)))
    z = z_base * lerp(1.0, 2.25, rush)
    # Focus shifts from center (0.5, 0.52) smoothly onto his right eye (0.784, 0.562)
    fu = lerp(0.50, EYES_UV[1][0] - 0.02, rush)
    fv = lerp(0.53, EYES_UV[1][1] - 0.01, rush)
    jx, jy = kt.shake(t, 4 * kick_pulse(t) * (0.3 + 0.7 * rush), 18)
    cx = W / 2 - (fu - 0.5) * iw * z + jx
    cy = H / 2 - (fv - 0.5) * ih * z + jy
    put(c, eye_img, cx, cy, scale=z)
    # Cool moody screen-light grade so the orange reflection inside his eyes pops dramatically
    c[:] *= np.array([0.78, 0.84, 0.94], np.float32)[None, None, :]

    # Screen coordinates of his two irises
    E = [(cx + (u - 0.5) * iw * z, cy + (v - 0.5) * ih * z) for u, v in EYES_UV]
    ir = IRIS_R * iw * z

    # Claude's warm-orange reflection forming directly inside BOTH pupils on 僕が -> 見えるか？
    refl = ease_out(ramp(t, t2 - 0.05, t2 + 0.40))
    glitch_k = ease_out(ramp(t, t3, q + 0.3))
    for ex, ey in E:
        _draw_pupil_reflection(c, ex, ey, ir, t, refl * (0.85 + 0.15 * kick_pulse(t)), glitch_k)

    # Subtle attention ring locking gently onto his right iris on 見えるか？
    if t >= t3 - 0.05:
        lock = ease_out(clamp((t - t3) / 0.25))
        tight = ease_out(clamp((t - q) / 0.18)) if t >= q else 0.0
        R = Layer(ss=1)
        ex, ey = E[1]
        rr = ir * lerp(1.65, 1.12, lock) - 10 * tight
        R.circle((ex, ey), rr, OR, 2 + int(tight), a=0.75 * lock)
        _reticle(R, ex, ey, rr * 1.15, OR, 2, gap=0.55)
        R.comp(c, 1.0, mode="add")

    # Dark cinematic gradient bars at top & bottom for razor-sharp lyric legibility
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    shade = np.clip((0.18 - yy) / 0.18, 0, 1) * 0.72 + np.clip((yy - 0.76) / 0.24, 0, 1) * 0.82
    c[:] *= (1.0 - shade)

    # Lyrics with token IDs
    STK = dict(stroke=6, sc=(0.03, 0.03, 0.04))
    labels = []
    w0 = word(c, t, LINE, 0, 220, 118, 62, "slide", color=LIGHT, anchor=(0, 0.5), dx=-100, **STK)
    labels.append((w0, 220, 162, (0, 0.5)))
    w1 = word(c, t, LINE, 1, 330, 885, 118, "slam", color=OR, font="jpb", sh=6, s0=1.6, **STK)
    labels.append((w1, 330, 965, (0.5, 0.5)))
    w2 = kt.sub(LINE, "見えるか")
    word(c, t, LINE, w2, 1420, 885, 128, "slam", color=LIGHT, sh=8, s0=1.7, **STK)
    labels.append((w2, 1420, 968, (0.5, 0.5)))
    wq = kt.sub(LINE, "？")
    word(c, t, LINE, wq, 1420 + kt.wsize("見えるか", 128)[0] / 2 + 58, 885, 128, "slam", color=OR, sh=10, s0=2.2,
         dt=max(0.0, q - wq.t), **STK)
    for w, x, y, an in labels:
        if w is not None and t >= w.t:
            ht = kt.wimg(_token_id(w.s), 18, "mono", MID, raw=True)
            put(c, ht, x, y, anchor=an, alpha=0.75 * ease_out(clamp((t - w.t) / 0.2)))

    fl = kt.flash_at(t, [t2, t3], 0.09, 0.22)
    return dict(grain=0.03, vignette=1.15, flash=fl, bloom=0.25 + 0.25 * refl, ca=1 + 2 * fl)
