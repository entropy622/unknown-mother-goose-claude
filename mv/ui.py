"""Vector UI pieces (anti-aliased) in the spirit of the reference MV's interface motifs."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import (W, H, ORANGE, INK, composite, put, text, text_rgba, to_rgba_f, hash01, clamp, ease_out, ramp,
                  load_asset, rect_cached)


def _c8(col, a=1.0):
    return (int(col[0] * 255 * a), int(col[1] * 255 * a), int(col[2] * 255 * a), int(255 * a))


class Layer:
    """Premultiplied uint8 RGBA scratch layer covering a region of the frame."""

    def __init__(self, x0=0, y0=0, w=W, h=H, ss=1):
        self.x0, self.y0, self.ss = x0, y0, ss
        self.buf = np.zeros((h * ss, w * ss, 4), np.uint8)

    def _p(self, x, y):
        return (int(round((x - self.x0) * self.ss * 16)), int(round((y - self.y0) * self.ss * 16)))

    def line(self, p, q, col, th=1, a=1.0):
        cv2.line(self.buf, self._p(*p), self._p(*q), _c8(col, a), max(1, int(th * self.ss)), cv2.LINE_AA, 4)

    def polyline(self, pts, col, th=1, a=1.0, closed=False):
        P = np.array([self._p(*p) for p in pts], np.int32)
        cv2.polylines(self.buf, [P], closed, _c8(col, a), max(1, int(th * self.ss)), cv2.LINE_AA, 4)

    def poly(self, pts, col, a=1.0):
        P = np.array([self._p(*p) for p in pts], np.int32)
        cv2.fillPoly(self.buf, [P], _c8(col, a), cv2.LINE_AA, 4)

    def circle(self, c, r, col, th=-1, a=1.0):
        cv2.circle(self.buf, self._p(*c), int(r * self.ss * 16), _c8(col, a), -1 if th < 0 else max(1, int(th * self.ss)),
                   cv2.LINE_AA, 4)

    def arc(self, c, r, a0, a1, col, th=2, a=1.0):
        cv2.ellipse(self.buf, self._p(*c), (int(r * self.ss * 16), int(r * self.ss * 16)), 0, a0, a1, _c8(col, a),
                    max(1, int(th * self.ss)), cv2.LINE_AA, 4)

    def rect(self, x0, y0, x1, y1, col, th=-1, a=1.0):
        if th < 0:
            self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], col, a)
        else:
            self.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], col, th, a, closed=True)

    def comp(self, c, alpha=1.0, mode="normal"):
        b = self.buf
        if self.ss > 1:
            b = cv2.resize(b, (b.shape[1] // self.ss, b.shape[0] // self.ss), interpolation=cv2.INTER_AREA)
        ys, xs = np.nonzero(b[..., 3])
        if len(ys) == 0:
            return
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        composite(c, b[y0:y1, x0:x1].astype(np.float32) / 255, self.x0 + x0, self.y0 + y0, alpha, mode)


# --------------------------------------------------------------- shapes as cached RGBA
@lru_cache(maxsize=64)
def bubble_rgba(w, h, fill=(0.62, 0.62, 0.64), tail="left", radius=10, outline=None, th=2, fill_alpha=1.0):
    k = 3
    im = Image.new("RGBA", ((w + 40) * k, (h + 40) * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = tuple(int(v * 255) for v in fill) + (int(255 * fill_alpha),)
    o = None if outline is None else tuple(int(v * 255) for v in outline) + (255,)
    x0, y0 = 20 * k, 10 * k
    d.rounded_rectangle([x0, y0, x0 + w * k, y0 + h * k], radius=radius * k, fill=f, outline=o, width=th * k)
    if tail == "left":
        pts = [(x0 + 18 * k, y0 + h * k - 2), (x0 + 6 * k, y0 + (h + 22) * k), (x0 + 40 * k, y0 + h * k - 2)]
    elif tail == "right":
        pts = [(x0 + (w - 18) * k, y0 + h * k - 2), (x0 + (w - 6) * k, y0 + (h + 22) * k), (x0 + (w - 40) * k, y0 + h * k - 2)]
    else:
        pts = None
    if pts:
        d.polygon(pts, fill=f)
        if o:
            d.line([pts[0], pts[1], pts[2]], fill=o, width=th * k)
    return to_rgba_f(im.resize((w + 40, h + 40), Image.LANCZOS))


@lru_cache(maxsize=8)
def heart_rgba(size, color=(1, 1, 1), outline=0):
    k = 4
    S = size * k
    t = np.linspace(0, 2 * np.pi, 200)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], 1)
    pts -= (pts.max(0) + pts.min(0)) / 2
    pts = pts / np.abs(pts).max() * S * 0.46 + S / 2
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = tuple(int(v * 255) for v in color) + (255,)
    if outline:
        d.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=col, width=outline * k, joint="curve")
    else:
        d.polygon([tuple(p) for p in pts], fill=col)
    return to_rgba_f(im.resize((size, size), Image.LANCZOS))


@lru_cache(maxsize=4)
def checker_heart_rgba(size, cell=14):
    h = heart_rgba(size, (1, 1, 1))
    yy, xx = np.mgrid[0:size, 0:size]
    chk = (((xx // cell) + (yy // cell)) % 2).astype(np.float32)
    col = (0.80 + 0.15 * chk)[..., None]
    out = h.copy()
    out[..., :3] = col * h[..., 3:4]
    return out


def reflect(src, fade=0.35, length=0.35):
    r = src[::-1].copy()
    h = r.shape[0]
    g = np.clip(1 - np.arange(h) / (h * length), 0, 1).astype(np.float32)[:, None, None] * fade
    return r * g


# --------------------------------------------------------------- clouds
@lru_cache(maxsize=4)
def cloud_tex(w, h, seed=1, contrast=2.2, base=0.5):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(6):
        cell = max(2, 256 >> o)
        n = rng.random((h // cell + 3, w // cell + 3)).astype(np.float32)
        up = cv2.resize(n, ((w // cell + 3) * cell, (h // cell + 3) * cell), interpolation=cv2.INTER_CUBIC)[:h, :w]
        acc += up * amp
        tot += amp
        amp *= 0.55
    acc /= tot
    return np.clip((acc - base) * contrast + 0.5, 0, 1)


def draw_clouds(c, t, speed=(0, -60), seed=1, sky=(0.80, 0.81, 0.83), cloud=(0.97, 0.97, 0.975), strength=1.0,
                scale=1.0):
    tw, th = int(W * 1.0), int(H * 1.0)
    tex = cloud_tex(tw, th, seed)
    ox = int(t * speed[0]) % tw
    oy = int(t * speed[1]) % th
    m = np.roll(np.roll(tex, -oy, 0), -ox, 1)
    m = m[..., None] * strength
    c[:] = c * (1 - m) + np.asarray(cloud, np.float32) * m if sky is None else \
        np.asarray(sky, np.float32) * (1 - m) + np.asarray(cloud, np.float32) * m


# --------------------------------------------------------------- misc widgets
def spinner(c, cx, cy, r, t, col=(0.85, 0.85, 0.86), th=6):
    L = Layer(int(cx - r - 10), int(cy - r - 10), int(2 * r + 20), int(2 * r + 20))
    a0 = (t * 400) % 360
    span = 200 + 80 * math.sin(t * 5)
    L.arc((cx, cy), r, a0, a0 + span, col, th)
    L.comp(c)


def progress_bar(c, cx, cy, w, k, col=(0.85, 0.85, 0.86), bg=(0.3, 0.3, 0.32), th=4):
    L = Layer(int(cx - w / 2 - 4), int(cy - 8), int(w + 8), 16)
    L.line((cx - w / 2, cy), (cx + w / 2, cy), bg, th)
    if k > 0:
        L.line((cx - w / 2, cy), (cx - w / 2 + w * clamp(k), cy), col, th)
    L.comp(c)


def chain_links(L, p0, p1, link=26, col=(0.45, 0.45, 0.47), th=3):
    n = int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / link)
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    for i in range(n):
        x = p0[0] + (p1[0] - p0[0]) * (i + 0.5) / n
        y = p0[1] + (p1[1] - p0[1]) * (i + 0.5) / n
        rx, ry = (link * 0.62, link * 0.3) if i % 2 == 0 else (link * 0.62, link * 0.12)
        cv2.ellipse(L.buf, L._p(x, y), (int(rx * 16), int(ry * 16)), ang, 0, 360, _c8(col), th, cv2.LINE_AA, 4)


def monitor(L, cx, cy, w, h, col=(0.2, 0.2, 0.22), screen=(0.96, 0.96, 0.97), bezel=14):
    L.rect(cx - w / 2 - bezel, cy - h / 2 - bezel, cx + w / 2 + bezel, cy + h / 2 + bezel, col)
    L.rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, screen)


def text_lines_block(L, x0, y0, w, n, col=(0.7, 0.7, 0.72), gap=16, th=6, seed=0):
    for i in range(n):
        ln = w * (0.45 + 0.55 * hash01(i, seed))
        L.line((x0, y0 + i * gap), (x0 + ln, y0 + i * gap), col, th)
