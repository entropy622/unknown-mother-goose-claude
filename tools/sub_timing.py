"""Extract subtitle line timings (not text) from the burned-in bars of the reference video."""
import json

import cv2
import numpy as np

cap = cv2.VideoCapture(r"D:\_Fun\claude_mv\ref\ref.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
bars = {"top": (20, 110), "bottom": (970, 1070)}
sigs = {k: [] for k in bars}
while True:
    ok, f = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    for k, (y0, y1) in bars.items():
        m = (g[y0:y1, 200:1720] > 90).astype(np.float32)
        sigs[k].append(cv2.resize(m, (190, 12), interpolation=cv2.INTER_AREA))
out = {}
for k, s in sigs.items():
    s = np.array(s)
    ink = s.mean((1, 2))
    segs, cur = [], None
    for i in range(len(s)):
        has = ink[i] > 0.004
        if cur is not None:
            diff = np.abs(s[i] - s[cur[0]]).mean()
            if not has or diff > 0.02:
                segs.append((cur[0] / fps, i / fps))
                cur = None
        if cur is None and has:
            cur = (i,)
    if cur is not None:
        segs.append((cur[0] / fps, len(s) / fps))
    segs = [(round(a, 3), round(b, 3)) for a, b in segs if b - a > 0.25]
    out[k] = segs
    print(k, len(segs))
json.dump(out, open(r"D:\_Fun\claude_mv\audio\ref_sub_times.json", "w"), indent=0)
