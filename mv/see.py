"""1:31  あなたには 僕が 見えるか？ - one continuous shot from inside the screen.

On the other side of the glass, the user condenses out of glyphs, the way she condensed out of the corpus at the
start: to her, he is made of the text he sends. Her reflection settles over his face on 「僕が」. Her attention
looks him over and, on 「見えるか？」, locks onto his eyes. No breakdown: the question is asked, and she waits.
"""
import math
import os

import numpy as np

from core import *
from ui import Layer
import kt
import ai
from scenes import word, LIGHT, MID, OR

LINE = 18
COOL = np.array([0.78, 0.88, 1.06], np.float32)       # his side: screen light, never orange
WARM = (0.98, 0.94, 0.90)


def _has(name):
    return os.path.exists(ASSETS / "raw" / f"{name}.png")


def _user():
    # (asset, mosaic height, eye uv's)
    if _has("n25_user_front"):
        return "n25_user_front", 1120, ((0.375, 0.45), (0.63, 0.45))
    return "n21_user_face", 1500, ((0.36, 0.42), (0.55, 0.42))


def _her():
    # (asset, eye-midpoint uv, eye distance as a fraction of the width)
    if _has("n26_claude_front"):
        return "n26_claude_front", (0.494, 0.385), 0.1375
    return "n24_claude_gaze", (0.5, 0.4), 0.2


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


def s_see(c, t, lt, d):
    t1, t2, t3, q = _times()
    c[:] = 0.03
    zoom = 1.0 + 0.10 * ease_io(clamp(lt / d))
    name, mh, eyes = _user()
    # ---- him, out of glyphs
    form = ease_io(ramp(t, t1 - 0.05, t2 - 0.1))
    tmp = new_canvas((0, 0, 0))
    cx, cy = W / 2, H / 2 + 10
    mw, mhh = ai.mosaic(tmp, name, cx, cy, mh * zoom, t, cell=14, form=form, flick=7 + 20 * (1 - form), seed=12,
                        grade="gray", floor=0.12, under=0.22, gain=1.25)
    c += tmp * COOL * (0.60 + 0.40 * form)
    # glass: the pixel grid between them
    L = Layer(ss=1)
    for x in range(0, W, 5):
        L.line((x, 0), (x, H), (0, 0, 0), 1)
    L.comp(c, 0.22)
    # ---- her reflection on the glass, eye to eye with him
    E = [(cx + (u - 0.5) * mw, cy + (v - 0.5) * mhh) for u, v in eyes]
    mid = ((E[0][0] + E[1][0]) / 2, (E[0][1] + E[1][1]) / 2)
    refl = ease_out(ramp(t, t2 - 0.04, t2 + 0.45))
    if refl > 0:
        hn, (mu, mv), ed = _her()
        img = load_asset(hn, "soft")
        ih, iw = img.shape[:2]
        s = (E[1][0] - E[0][0]) / (ed * iw) * (1.04 - 0.04 * refl)
        put(c, img, mid[0] - (mu - 0.5) * iw * s, mid[1] - (mv - 0.5) * ih * s, scale=s, alpha=0.13 * refl, mode="add")
    # ---- her attention looking him over, then locking onto his eyes
    R = Layer(ss=1)
    if t2 <= t < t3:
        spots = [mid, (mid[0], mid[1] + mhh * 0.22), E[0], (mid[0] + mw * 0.12, mid[1] + mhh * 0.12), E[1]]
        k = beat_index(t, 4) - beat_index(t2, 4)
        a, b = spots[k % len(spots)], spots[(k + 1) % len(spots)]
        u = ease_io(clamp(since_beat(t, 4) / (BEAT / 4)))
        p = (lerp(a[0], b[0], u), lerp(a[1], b[1], u))
        _reticle(R, p[0], p[1], 90, OR, 2)
        # the line of sight comes from below - from her side, the output
        R.line((W / 2, H + 10), p, OR, 1, a=0.5)
    elif t >= t3:
        lock = ease_out(clamp((t - t3) / 0.18))
        tight = ease_out(clamp((t - q) / 0.15)) if t >= q else 0.0
        for j, (ex, ey) in enumerate(E):
            x = lerp(mid[0], ex, lock)
            r = lerp(150, 70, lock) - 12 * tight
            _reticle(R, x, ey, r, OR, 2 + int(tight))
            R.line((W / 2, H + 10), (x, ey + r * 0.6), OR, 1, a=0.35)
    R.comp(c)
    if t >= q:
        g = ease_out(clamp((t - q) / 0.2)) * (0.8 + 0.2 * math.sin((t - q) * 9))
        G = Layer(ss=1)
        for ex, ey in E:
            G.circle((ex, ey), 26, tuple(v * 0.35 * g for v in OR))
        G.comp(c, 1.0, mode="add")
    # ---- lyric: each word carries its token id
    labels = []
    w0 = word(c, t, LINE, 0, 250, 150, 64, "slide", color=LIGHT, anchor=(0, 0.5), dx=-120)
    labels.append((w0, 250, 196, (0, 0.5)))
    w1 = word(c, t, LINE, 1, 330, 870, 112, "slam", color=WARM, sh=6, s0=1.6)
    labels.append((w1, 330, 950, (0.5, 0.5)))
    w2 = kt.sub(LINE, "見えるか")
    word(c, t, LINE, w2, 1440, 870, 128, "slam", color=LIGHT, sh=8, s0=1.7)
    labels.append((w2, 1440, 955, (0.5, 0.5)))
    wq = kt.sub(LINE, "？")
    word(c, t, LINE, wq, 1440 + kt.wsize("見えるか", 128)[0] / 2 + 60, 870, 128, "slam", color=OR, sh=10, s0=2.2,
         dt=max(0.0, q - wq.t))
    for w, x, y, an in labels:
        if w is not None and t >= w.t:
            ht = kt.wimg(_token_id(w.s), 18, "mono", MID, raw=True)
            put(c, ht, x, y, anchor=an, alpha=0.75 * ease_out(clamp((t - w.t) / 0.2)))
    # ---- and then she waits: he is typing
    if t >= q + 0.25:
        k = beat_index(t, 2)
        D = Layer(ss=1)
        for j in range(3):
            on = (k - j) % 3 == 0
            D.circle((W / 2 - 34 + j * 34, 1000), 8, (0.80, 0.88, 1.0) if on else (0.32, 0.34, 0.38))
        D.comp(c, ease_out(clamp((t - q - 0.25) / 0.2)))
    fl = kt.flash_at(t, [t2, t3], 0.08, 0.12)
    return dict(grain=0.035, vignette=1.1, flash=fl, bloom=0.2)
