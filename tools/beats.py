"""Beat / downbeat / drum-onset analysis of the album track, output in MV time."""
import json

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, find_peaks

OFFSET = 1.592          # mv_time = album_time + OFFSET
BPM = 222.0

sr, y = wavfile.read(r"D:\_Fun\claude_mv\audio\album.wav")
y = y.astype(np.float32).mean(1) / 32768
hop = 256
f, t, Z = stft(y, fs=sr, nperseg=2048, noverlap=2048 - hop)
fe = sr / hop
mag = np.log1p(np.abs(Z) * 200)


def band_flux(lo, hi):
    m = mag[(f >= lo) & (f < hi)]
    e = np.maximum(0, np.diff(m, axis=1)).sum(0)
    e = np.concatenate([[0], e])
    e = e - np.convolve(e, np.ones(64) / 64, mode="same")
    e = np.maximum(e, 0)
    return e / (np.percentile(e, 99) + 1e-9)


kick = band_flux(30, 130)
snare = band_flux(1500, 6000)
full = band_flux(30, 11000)
onset = 0.5 * full + 0.3 * kick + 0.2 * snare

# ---- dynamic-programming beat tracker (Ellis 2007), fixed tempo prior
P = 60 * fe / BPM
N = len(onset)
score = onset.copy()
back = np.full(N, -1)
tight = 400.0
lo, hi = int(P * 0.85), int(P * 1.15) + 1
for i in range(hi, N):
    prev = np.arange(i - hi, i - lo)
    pen = -tight * (np.log((i - prev) / P)) ** 2
    cand = score[prev] + pen
    j = int(np.argmax(cand))
    score[i] = onset[i] + cand[j]
    back[i] = prev[j]
i = int(np.argmax(score[-int(P * 2):])) + N - int(P * 2)
beats = []
while i >= 0:
    beats.append(i)
    i = back[i]
beats = np.array(beats[::-1])
bt = beats / fe

# ---- downbeat phase: kick-heavy beats vs snare-heavy beats, local windows
def strength(env, idx, w=3):
    return np.array([env[max(0, k - w):k + w + 1].max() for k in idx])


ks, ss = strength(kick, beats), strength(snare, beats)
nb = len(beats)
WIN = 16
phase_local = []
for s0 in range(0, nb, 4):
    seg = slice(max(0, s0 - WIN), min(nb, s0 + WIN))
    idx = np.arange(nb)[seg]
    best, bp = -1e9, 0
    for p in range(4):
        m = (idx - p) % 4
        sc = ks[idx][m == 0].mean() * 1.5 + ks[idx][m == 2].mean() * 0.5 + ss[idx][(m == 1) | (m == 3)].mean() \
            - ss[idx][(m == 0) | (m == 2)].mean() * 0.5
        if sc > best:
            best, bp = sc, p
    phase_local.append(bp)
# smooth: majority in sliding window of 5
pl = np.array(phase_local)
sm = pl.copy()
for i in range(len(pl)):
    w = pl[max(0, i - 2):i + 3]
    sm[i] = np.bincount(w, minlength=4).argmax()
down = np.zeros(nb, bool)
for i in range(nb):
    down[i] = (i - sm[min(len(sm) - 1, i // 4)]) % 4 == 0

kp, _ = find_peaks(kick, height=0.35, distance=int(fe * 0.12))
sp, _ = find_peaks(snare, height=0.35, distance=int(fe * 0.12))

out = dict(
    offset=OFFSET, bpm=BPM,
    beats=[round(float(b + OFFSET), 4) for b in bt],
    downbeat=[bool(d) for d in down],
    kicks=[[round(float(k / fe + OFFSET), 4), round(float(kick[k]), 3)] for k in kp],
    snares=[[round(float(s / fe + OFFSET), 4), round(float(snare[s]), 3)] for s in sp],
    env_fps=fe, env=[round(float(v), 3) for v in onset[::2]],
)
json.dump(out, open(r"D:\_Fun\claude_mv\audio\beats.json", "w"))
ib = np.diff(bt)
print("beats", nb, "mean ibi", ib.mean(), "bpm", 60 / ib.mean(), "ibi std", ib.std())
print("phase per 4-beat block (changes):", [(round(float(bt[i * 4] + OFFSET), 2), int(p)) for i, p in enumerate(sm)
                                            if i == 0 or p != sm[i - 1]])
print("kicks", len(kp), "snares", len(sp))
