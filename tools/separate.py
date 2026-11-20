"""Vocal stem separation with demucs (htdemucs) on GPU, without torchaudio."""
import sys
import types

import numpy as np
import torch
from scipy.io import wavfile

sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

sr, y = wavfile.read(r"D:\_Fun\claude_mv\audio\album.wav")
assert sr == 44100
wav = torch.from_numpy(y.T.astype(np.float32) / 32768)
model = get_model("htdemucs").cuda().eval()
ref = wav.mean(0)
x = (wav - ref.mean()) / ref.std()
with torch.no_grad():
    out = apply_model(model, x[None].cuda(), device="cuda", split=True, overlap=0.25, progress=True)[0]
out = out * ref.std() + ref.mean()
for name, stem in zip(model.sources, out):
    if name in ("vocals", "drums"):
        a = (stem.cpu().numpy().T.clip(-1, 1) * 32767).astype(np.int16)
        wavfile.write(rf"D:\_Fun\claude_mv\audio\stem_{name}.wav", sr, a)
        print("wrote", name)
