import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, find_peaks

sr, y = wavfile.read(r"D:\_Fun\claude_mv\audio\song_mono.wav")
y = y.astype(np.float32) / 32768.0
hop = 512
f, t, Z = stft(y, fs=sr, nperseg=2048, noverlap=2048 - hop)
mag = np.log1p(np.abs(Z) * 100)
flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
flux = np.concatenate([[0], flux])
flux = (flux - flux.mean()) / (flux.std() + 1e-9)
fps_env = sr / hop

# tempo via autocorrelation of onset envelope (60..240 bpm)
env = flux - np.convolve(flux, np.ones(16) / 16, mode="same")
env = np.maximum(env, 0)
ac = np.correlate(env, env, mode="full")[len(env) - 1:]
lags = np.arange(len(ac))
bpm_of = 60 * fps_env / np.maximum(lags, 1)
mask = (bpm_of > 70) & (bpm_of < 240)
best = lags[mask][np.argmax(ac[mask])]
bpm = 60 * fps_env / best
print("bpm estimate", bpm)
cands = sorted(zip(ac[mask], bpm_of[mask]), reverse=True)[:12]
print("top", [(round(b, 1)) for a, b in cands])

# refine bpm on a fine grid with comb scoring + phase
def comb_score(bpm_try):
    period = 60 * fps_env / bpm_try
    best_s, best_p = -1, 0
    for ph in np.linspace(0, period, 64, endpoint=False):
        idx = np.arange(ph, len(env) - 1, period).astype(int)
        s = env[idx].mean()
        if s > best_s:
            best_s, best_p = s, ph
    return best_s, best_p

grid = np.arange(bpm * 0.97, bpm * 1.03, 0.05)
scores = [comb_score(b) for b in grid]
i = int(np.argmax([s for s, _ in scores]))
bpm_f, phase = grid[i], scores[i][1] / fps_env
print("refined bpm", bpm_f, "phase sec", phase)

# loudness per 0.5s
rms_win = int(sr * 0.5)
n = len(y) // rms_win
rms = np.sqrt((y[: n * rms_win].reshape(n, rms_win) ** 2).mean(axis=1))
db = 20 * np.log10(rms + 1e-6)
line = []
for k in range(0, n, 2):
    line.append(f"{k*0.5:6.1f}s {db[k]:6.1f} " + "#" * int(max(0, db[k] + 40)))
open(r"D:\_Fun\claude_mv\audio\loudness.txt", "w").write("\n".join(line))

# strong onsets
pk, _ = find_peaks(flux, height=2.5, distance=int(fps_env * 0.2))
json.dump(
    {
        "bpm": float(bpm_f),
        "phase": float(phase),
        "duration": len(y) / sr,
        "onsets": [round(float(p / fps_env), 3) for p in pk],
        "env_fps": fps_env,
        "env": [round(float(v), 3) for v in np.maximum(flux, 0)],
        "db_halfsec": [round(float(v), 2) for v in db],
    },
    open(r"D:\_Fun\claude_mv\audio\analysis.json", "w"),
)
