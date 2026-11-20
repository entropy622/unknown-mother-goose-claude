"""Audio-reactive features for the renderer: 60 fps log-mel spectrum of the mix, vocal envelope and a light vocal waveform."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

ROOT = r"D:\_Fun\claude_mv\audio"
OFFSET = 1.592
FPS = 60
NB = 48


def mono(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    return (x.mean(1) if x.ndim > 1 else x), sr


mix, sr = mono(ROOT + r"\album_padded.wav")
hop = sr // FPS
f, _, Z = stft(mix, sr, nperseg=4096, noverlap=4096 - hop, boundary=None, padded=True)
P = np.abs(Z) ** 2
edges = np.searchsorted(f, np.geomspace(50, 14000, NB + 1))
edges = np.maximum.accumulate(np.maximum(edges, np.arange(len(edges)) + edges[0]))
mel = np.stack([P[a:max(b, a + 1)].mean(0) for a, b in zip(edges[:-1], edges[1:])])
mel = 10 * np.log10(mel + 1e-10)
lo = np.percentile(mel, 5, axis=1, keepdims=True)
hi = np.percentile(mel, 99.5, axis=1, keepdims=True)
mel = np.clip((mel - lo) / (hi - lo), 0, 1).astype(np.float16)

voc, vsr = mono(ROOT + r"\stem_vocals.wav")
voc = np.concatenate([np.zeros(int(OFFSET * vsr), np.float32), voc])
v8 = resample_poly(voc, 8000, vsr).astype(np.float32)
n = len(v8) // (8000 // FPS)
env = np.sqrt((v8[: n * (8000 // FPS)].reshape(n, -1) ** 2).mean(1))
env = np.clip(env / np.percentile(env, 99.5), 0, 1).astype(np.float16)
v8 = (v8 / (np.abs(v8).max() + 1e-6)).astype(np.float16)

np.savez_compressed(ROOT + r"\spectral.npz", mel=mel, env=env, wave=v8, fps=FPS, wsr=8000)
print(mel.shape, env.shape, v8.shape)
