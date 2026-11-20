"""Find the time offset between the MV audio and the purchased album track."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


def env(path):
    sr, y = wavfile.read(path)
    y = y.astype(np.float32) / 32768
    _, _, Z = stft(y, fs=sr, nperseg=2048, noverlap=2048 - 256)
    m = np.log1p(np.abs(Z) * 100)
    e = np.maximum(0, np.diff(m, axis=1)).sum(0)
    return (e - e.mean()) / e.std(), sr / 256


a, fa = env(r"D:\_Fun\claude_mv\audio\song_mono.wav")
b, fb = env(r"D:\_Fun\claude_mv\audio\album_mono.wav")
assert abs(fa - fb) < 1e-6
for start in [10, 60, 120, 180, 240]:
    seg = b[int(start * fb):int((start + 20) * fb)]
    best, lag = -1, 0
    for L in range(int((start - 6) * fa), int((start + 10) * fa)):
        w = a[L:L + len(seg)]
        if len(w) < len(seg):
            break
        s = float(np.dot(w, seg)) / len(seg)
        if s > best:
            best, lag = s, L
    print(f"album t={start:4d}s  ->  mv t={lag / fa:8.3f}s   offset={lag / fa - start:+.3f}s  corr={best:.3f}")
