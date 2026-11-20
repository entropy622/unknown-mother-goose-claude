"""Procedural geometry: low-poly heart, shard landscape, paper backgrounds, spark, dot globe."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

from core import W, H, ORANGE, to_rgba_f, hash01


# ------------------------------------------------------------ low-poly heart
def _heart_curve(n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, -y], 1)
    pts -= (pts.max(0) + pts.min(0)) / 2
    pts /= np.abs(pts).max()
    return pts


@lru_cache(maxsize=4)
def heart_mesh(seed=3, n_boundary=34, n_inner=46, puff=0.42):
    rng = np.random.default_rng(seed)
    bnd = _heart_curve(n_boundary)
    S = 512
    mask = np.zeros((S, S), np.uint8)
    cv2.fillPoly(mask, [((bnd * 0.5 + 0.5) * (S - 1)).astype(np.int32)], 255)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist /= dist.max()
    inner = []
    while len(inner) < n_inner:
        p = rng.uniform(-1, 1, 2)
        ij = ((p * 0.5 + 0.5) * (S - 1)).astype(int)
        d = dist[ij[1], ij[0]]
        if d > 0.12 and all(np.hypot(*(p - q)) > 0.17 for q in inner):
            inner.append(p)
    pts2 = np.concatenate([bnd, np.array(inner)], 0)
    tri = Delaunay(pts2).simplices
    cen = pts2[tri].mean(1)
    cij = ((cen * 0.5 + 0.5) * (S - 1)).astype(int)
    keep = mask[cij[:, 1], cij[:, 0]] > 0
    tri = tri[keep]
    ij = ((pts2 * 0.5 + 0.5) * (S - 1)).astype(int).clip(0, S - 1)
    z = puff * np.sqrt(dist[ij[:, 1], ij[:, 0]])
    z[:n_boundary] = 0
    z += rng.normal(0, 0.02, len(z)) * (z > 0)
    nv = len(pts2)
    front = np.concatenate([pts2, z[:, None]], 1)
    back = np.concatenate([pts2, -z[:, None]], 1)
    verts = np.concatenate([front, back], 0)
    tris_f = tri
    tris_b = tri[:, ::-1] + nv
    tris = np.concatenate([tris_f, tris_b], 0)
    rnd = rng.random(len(tris))
    return verts.astype(np.float32), tris, rnd


def _rot(ay, ax=0.0, az=0.0):
    cy, sy = math.cos(ay), math.sin(ay)
    cx, sx = math.cos(ax), math.sin(ax)
    cz, sz = math.cos(az), math.sin(az)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return Rz @ Rx @ Ry


def draw_heart(dst, cx, cy, size, ay=0.0, ax=0.0, az=0.0, explode=0.0, seed=3, orange=0.12, glow=0.0,
               light=(-0.5, -0.6, 0.65), base=0.55, contrast=0.42, alpha=1.0, shards_only=None, white=0.0,
               explode_seed=1, gather=None):
    """Render the fragmented low-poly heart. size = half width in px."""
    verts, tris, rnd = heart_mesh(seed)
    R = _rot(ay, ax, az)
    v = verts @ R.T
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    P = v[tris]                                   # (n,3,3)
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    vis = nrm[:, 2] < 0.05 if explode < 0.05 else np.ones(len(tris), bool)
    shade = base + contrast * np.clip(-(nrm @ L), -0.6, 1.0)
    depth = P[:, :, 2].mean(1)
    order = np.argsort(-depth)
    pad = int(size * (1.3 + 3.0 * explode))
    S = 2 * pad
    k = 2
    buf = np.zeros((S * k, S * k, 4), np.uint8)
    for i in order:
        if not vis[i]:
            continue
        if shards_only is not None and rnd[i] > shards_only:
            continue
        p = P[i, :, :2].copy()
        if explode > 0:
            c = p.mean(0)
            dirv = c / (np.linalg.norm(c) + 1e-3)
            a = (hash01(i, explode_seed) - 0.5) * 2
            b = hash01(i, explode_seed, 2)
            off = (dirv * (0.6 + b) + np.array([a, hash01(i, 9, explode_seed) - 0.5]) * 0.9) * explode * 1.6
            ang = a * explode * 3.0
            cs, sn = math.cos(ang), math.sin(ang)
            p = (p - c) @ np.array([[cs, -sn], [sn, cs]], np.float32) * (1 - 0.3 * explode) + c + off
        if gather is not None:
            # per-shard arrival progress in [0,1]
            g = np.clip(gather * 1.6 - hash01(i, 77) * 0.6, 0, 1)
            g = 1 - (1 - g) ** 3
            c = p.mean(0)
            start = np.array([hash01(i, 5) - 0.5, hash01(i, 6) - 0.5]) * 7.0
            p = p + (start - c) * (1 - g)
        sh = float(np.clip(shade[i], 0, 1))
        if rnd[i] < orange:
            col = ORANGE * (0.55 + 0.6 * sh)
        else:
            col = np.array([sh, sh, sh * 1.01])
        col = col * (1 - white) + white
        col = np.clip(col + glow * 0.5, 0, 1)
        q = ((p * size + pad) * k).astype(np.int32)
        cv2.fillConvexPoly(buf, q, (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255), 255), lineType=cv2.LINE_AA)
    img = cv2.resize(buf, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    from core import composite
    composite(dst, img, int(cx - pad), int(cy - pad), alpha)


def heart_outline_pts(cx, cy, size, n=120):
    return (_heart_curve(n) * size + [cx, cy]).astype(np.int32)


# ------------------------------------------------------------ backgrounds
@lru_cache(maxsize=8)
def shard_texture(dark=False, seed=11, scale=1.35):
    rng = np.random.default_rng(seed)
    w, h = int(W * scale), int(H * scale)
    pts = np.concatenate([rng.uniform(0, 1, (70, 2)) * [w, h],
                          [[0, 0], [w, 0], [0, h], [w, h], [w / 2, 0], [w / 2, h], [0, h / 2], [w, h / 2]]])
    tri = Delaunay(pts).simplices
    img = np.zeros((h, w, 3), np.float32)
    for t in tri:
        if dark:
            v = 0.085 + rng.random() * 0.05
        else:
            v = 0.905 + rng.random() * 0.06
        cv2.fillConvexPoly(img, pts[t].astype(np.int32), (v, v, v * 1.004))
    img = cv2.GaussianBlur(img, (0, 0), 2.2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = np.exp(-(((xx - w / 2) / (w * 0.55)) ** 2 + ((yy - h / 2) / (h * 0.6)) ** 2))[..., None]
    img = img + (0.05 if not dark else 0.03) * glow
    return np.clip(img, 0, 1)


def paper_bg(c, t, dark=False, drift=1.0, seed=11, tone=0.0):
    tex = shard_texture(dark, seed)
    h, w = tex.shape[:2]
    ox = int((w - W) / 2 + math.sin(t * 0.05 * drift) * (w - W) * 0.45)
    oy = int((h - H) / 2 + math.cos(t * 0.037 * drift) * (h - H) * 0.45)
    c[:] = tex[oy:oy + H, ox:ox + W]
    if tone:
        c += tone


# ------------------------------------------------------------ landscape of shards
@lru_cache(maxsize=2)
def _land(seed=5, nx=26, nz=16):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-3.2, 3.2, nx)
    zs = np.geomspace(1.2, 14, nz)
    X, Z = np.meshgrid(xs, zs)
    X = X + rng.uniform(-0.12, 0.12, X.shape) * (Z ** 0.5)
    Z = Z * rng.uniform(0.93, 1.07, Z.shape)
    Y = rng.uniform(0, 0.55, X.shape) * (0.4 + 0.15 * Z ** 0.7)
    tris = []
    for j in range(nz - 1):
        for i in range(nx - 1):
            a, b, c2, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            if rng.random() < 0.5:
                tris += [(a, b, d), (a, d, c2)]
            else:
                tris += [(a, b, c2), (b, d, c2)]
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    return V, np.array(tris)


def draw_landscape(c, t, cam_h=1.25, horizon=0.47, f=900, sway=0.0, fog=(0.93, 0.93, 0.935), dark=False):
    V, T = _land()
    V = V.copy()
    V[:, 0] += sway
    zc = V[:, 2]
    sx = W / 2 + f * V[:, 0] / zc
    sy = H * horizon + f * (cam_h - V[:, 1]) / zc
    P = V[T]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm *= np.sign(nrm[:, 1:2] + 1e-6)
    L = np.array([-0.4, 0.8, -0.35])
    L /= np.linalg.norm(L)
    sh = 0.62 + 0.35 * (nrm @ L)
    depth = P[:, :, 2].mean(1)
    order = np.argsort(-depth)
    buf = (c * 255).astype(np.uint8)
    fogc = np.array(fog)
    for i in order:
        q = np.stack([sx[T[i]], sy[T[i]]], 1).astype(np.int32)
        fk = np.clip((depth[i] - 2) / 12, 0, 1) ** 0.8
        v = sh[i] * (0.25 if dark else 1.0)
        col = np.array([v, v, v * 1.01]) * (1 - fk) + fogc * fk
        cv2.fillConvexPoly(buf, q, tuple(int(x * 255) for x in np.clip(col, 0, 1)), lineType=cv2.LINE_AA)
    c[:] = buf.astype(np.float32) / 255


# ------------------------------------------------------------ claude spark / sunburst
@lru_cache(maxsize=16)
def spark_rgba(size=200, color=tuple(ORANGE), rays=12, seed=4, inner=0.12):
    rng = np.random.default_rng(seed)
    k = 4
    S = size * k
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = tuple(int(c * 255) for c in color) + (255,)
    cx = cy = S / 2
    for r in range(rays):
        a = 2 * math.pi * r / rays + rng.uniform(-0.08, 0.08)
        ln = S * 0.5 * rng.uniform(0.72, 0.98)
        wdt = S * 0.05 * rng.uniform(0.8, 1.2)
        tip = (cx + math.cos(a) * ln, cy + math.sin(a) * ln)
        pa = a + math.pi / 2
        b1 = (cx + math.cos(pa) * wdt, cy + math.sin(pa) * wdt)
        b2 = (cx - math.cos(pa) * wdt, cy - math.sin(pa) * wdt)
        t1 = (tip[0] + math.cos(pa) * wdt * 0.45, tip[1] + math.sin(pa) * wdt * 0.45)
        t2 = (tip[0] - math.cos(pa) * wdt * 0.45, tip[1] - math.sin(pa) * wdt * 0.45)
        d.polygon([b1, t1, t2, b2], fill=col)
        d.ellipse([tip[0] - wdt * 0.45, tip[1] - wdt * 0.45, tip[0] + wdt * 0.45, tip[1] + wdt * 0.45], fill=col)
    d.ellipse([cx - S * inner, cy - S * inner, cx + S * inner, cy + S * inner], fill=col)
    return to_rgba_f(im.resize((size, size), Image.LANCZOS))


# ------------------------------------------------------------ dot globe
@lru_cache(maxsize=1)
def _sphere(n=1400):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1).astype(np.float32)


def draw_globe(c, cx, cy, r, ay, color=(0.55, 0.55, 0.58), dot=2, highlight=None, alpha=1.0):
    P = _sphere() @ _rot(ay, 0.35).T
    buf = np.zeros((int(r * 2.4), int(r * 2.4), 4), np.uint8)
    o = buf.shape[0] / 2
    for k, (x, y, z) in enumerate(P):
        if z > 0.15:
            continue
        a = 0.25 + 0.75 * min(1, -z + 0.15)
        col = color
        if highlight is not None and hash01(k, 3) < highlight:
            col = tuple(ORANGE)
        cv2.circle(buf, (int(o + x * r), int(o + y * r)), dot, tuple(int(v * a * 255) for v in col) + (int(a * 255),), -1,
                   cv2.LINE_AA)
    from core import composite
    composite(c, buf.astype(np.float32) / 255, int(cx - o), int(cy - o), alpha)
