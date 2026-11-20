"""Parse the user's LRC, time every character from vocal-stem onsets, export lyrics.json + external zh subtitles."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, find_peaks

LRC = Path(r"C:\Users\Aentro\Downloads\LrcHelper_v3_0_0\アンノウン・マザーグース.lrc")
OUT = Path(r"D:\_Fun\claude_mv\audio\lyrics.json")
SUBS = Path(r"D:\_Fun\claude_mv\out")
OFFSET = 1.592
TAIL = 285.0

# ------------------------------------------------------------------ parse
rows = []
for ln in LRC.read_text(encoding="utf-8-sig").splitlines():
    m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", ln.strip())
    if m:
        rows.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
rows.sort(key=lambda r: r[0])

JA = re.compile(r"[\u3040-\u30ff]")
lines = []
for t, s in rows:
    is_ja = bool(JA.search(s)) or s.startswith("お—")
    if is_ja:
        lines.append({"t": round(t + OFFSET, 3), "ja": s, "zh": ""})
    elif lines and not lines[-1]["zh"] and abs(t - (lines[-1]["t"] - OFFSET) - 1.0) < 0.05:
        lines[-1]["zh"] = s
for i, L in enumerate(lines):
    nxt = lines[i + 1]["t"] if i + 1 < len(lines) else TAIL
    L["next"] = nxt

# ------------------------------------------------------------------ vocal onsets
sr, y = wavfile.read(r"D:\_Fun\claude_mv\audio\stem_vocals.wav")
y = y.astype(np.float32).mean(1) / 32768
hop = 256
f, tt, Z = stft(y, fs=sr, nperseg=1024, noverlap=1024 - hop)
fe = sr / hop
mag = np.abs(Z)
band = (f > 150) & (f < 6000)
lm = np.log1p(mag[band] * 300)
flux = np.concatenate([[0], np.maximum(0, np.diff(lm, axis=1)).sum(0)])
flux = flux - np.convolve(flux, np.ones(32) / 32, mode="same")
flux = np.maximum(flux, 0)
flux /= np.percentile(flux, 99.5)
rms = np.sqrt((mag[band] ** 2).mean(0))
rms_db = 20 * np.log10(rms + 1e-9)
gate = rms_db > np.percentile(rms_db, 35)
pk, props = find_peaks(flux * gate, height=0.08, distance=int(fe * 0.055))
onsets = pk / fe + OFFSET
strength = props["peak_heights"]
active = gate

SMALL = set("ゃゅょぁぃぅぇぉっャュョァィゥェォッ")
PUNCT = set(" 　、。，,.！？!?―—「」『』…")


def units(s):
    """Indices of characters that start a new sung mora."""
    out = []
    for i, ch in enumerate(s):
        if ch in PUNCT or ch in SMALL:
            continue
        out.append(i)
    return out


_bt = np.array(json.load(open(r"D:\_Fun\claude_mv\audio\beats.json"))["beats"])
GRID16 = np.sort(np.concatenate([_bt] + [_bt[:-1] + (_bt[1:] - _bt[:-1]) * k / 4 for k in (1, 2, 3)]))
LEAD = 0.035

fr = 0.01
nfr = int(len(y) / sr / fr)
edb = np.array([20 * np.log10(np.sqrt((y[int(k * fr * sr):int((k + 1) * fr * sr)] ** 2).mean()) + 1e-9)
                for k in range(nfr)])


def gaps(a, b):
    """Low-energy runs inside [a, b] (MV time): list of (center, duration)."""
    i0, i1 = int((a - OFFSET) / fr), int((b - OFFSET) / fr)
    seg = edb[i0:i1]
    if len(seg) < 5:
        return []
    thr = np.median(seg) - 14
    low = seg < thr
    out, k = [], 0
    while k < len(low):
        if low[k]:
            j = k
            while j < len(low) and low[j]:
                j += 1
            if j - k >= 4:
                out.append(((i0 + (k + j) / 2) * fr + OFFSET, (j - k) * fr, (i0 + j) * fr + OFFSET))
            k = j
        else:
            k += 1
    return out


def assign(U, a, b):
    """Times for sung units U inside [a, b] from the strongest onsets."""
    n = len(U)
    sel = (onsets >= a - 0.06) & (onsets < b)
    O, S = onsets[sel], strength[sel]
    if len(O) >= n:
        top = np.sort(np.argsort(-S)[:n])
        top[0] = 0
        pick = list(O[np.sort(top)])
    elif len(O) >= 2:
        pick = list(np.interp(np.linspace(0, len(O) - 1, n), np.arange(len(O)), O))
    else:
        pick = list(np.linspace(a, a + (b - a) * 0.8, n))
    pick[0] = max(min(pick[0], a + 0.06), a - 0.06)
    return pick


for L in lines:
    s = L["ja"]
    i0, i1 = int((L["t"] - OFFSET) * fe), int((min(L["next"], L["t"] + 10) - OFFSET) * fe)
    act = np.nonzero(active[i0:i1])[0]
    vend = float((i0 + act[-1]) / fe + OFFSET) if len(act) else L["t"] + 2.0
    vend = min(vend, L["next"] - 0.05)
    L["end"] = round(min(vend + 0.25, L["next"]), 3)
    # phrases split on spaces; boundaries snapped to the clearest vocal gap near the mora-proportional point
    ph, cur = [], []
    for i, ch in enumerate(s):
        if ch in " 　":
            if cur:
                ph.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        ph.append(cur)
    mor = [max(1, len(units("".join(s[i] for i in p)))) for p in ph]
    tot = sum(mor)
    bounds = [L["t"]]
    G = gaps(L["t"] + 0.25, vend - 0.2)
    for k in range(1, len(ph)):
        exp = L["t"] + (vend - L["t"]) * sum(mor[:k]) / tot
        best, bs = None, -1e9
        for (cen, dur, gend) in G:
            if gend <= bounds[-1] + 0.2:
                continue
            sc = dur - 0.25 * abs(cen - exp)
            if abs(cen - exp) < 1.0 and sc > bs:
                best, bs = gend, sc
        bounds.append(best if best is not None else exp)
    bounds.append(vend)
    times = [None] * len(s)
    for k, p in enumerate(ph):
        U = [i for i in p if s[i] not in PUNCT and s[i] not in SMALL]
        if not U:
            continue
        pick = assign(U, bounds[k], bounds[k + 1])
        for j, i in enumerate(U):
            times[i] = float(pick[j])
    # quantise sung units to the 16th-note grid (vocal consonants lead the note by ~30-40 ms)
    prev = -1.0
    for i in range(len(s)):
        if times[i] is None:
            continue
        j = int(np.argmin(np.abs(GRID16 - (times[i] + LEAD))))
        while GRID16[j] <= prev + 1e-3 and j + 1 < len(GRID16):
            j += 1
        times[i] = float(GRID16[j])
        prev = times[i]
    last = L["t"]
    for i in range(len(s)):
        if times[i] is None:
            times[i] = last
        last = times[i]
    L["chars"] = [round(x, 3) for x in times]
    L["split"] = round(times[ph[1][0]], 3) if len(ph) > 1 else None
    L["bounds"] = [round(x, 3) for x in bounds]
    L["n_onsets"] = int(((onsets >= L["t"]) & (onsets < vend)).sum())
    L["n_units"] = len(units(s))

json.dump({"offset": OFFSET, "lines": lines, "onsets": [round(float(o), 3) for o in onsets]},
          OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ external subtitles (zh)
def ts_ass(x):
    h, r = divmod(x, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ts_srt(x):
    ms = int(round(x * 1000))
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


subs = []
for L in lines:
    if not L["zh"]:
        continue
    st = L["t"] - 0.1
    en = max(L["end"] + 0.6, st + 1.5)
    en = min(en, L["next"] - 0.15)
    subs.append((st, en, L["zh"]))

ass = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "WrapStyle: 0", "",
       "[V4+ Styles]",
       "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
       "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
       "MarginR, MarginV, Encoding",
       "Style: zh,Noto Serif SC,40,&H00F4F4F4,&H000000FF,&H00303030,&H64000000,0,0,0,0,100,100,2,0,1,2.2,0.6,2,60,60,46,1",
       "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
for st, en, s in subs:
    ass.append(f"Dialogue: 0,{ts_ass(st)},{ts_ass(en)},zh,,0,0,0,,{{\\fad(120,160)}}{s}")
base = SUBS / "Unknown_Mother-Goose_Claude_ver"
(base.with_suffix(".zh.ass")).write_text("\n".join(ass) + "\n", encoding="utf-8-sig")
srt = []
for i, (st, en, s) in enumerate(subs, 1):
    srt += [str(i), f"{ts_srt(st)} --> {ts_srt(en)}", s, ""]
(base.with_suffix(".zh.srt")).write_text("\n".join(srt), encoding="utf-8-sig")

print("lines", len(lines), "with zh", len(subs), "onsets", len(onsets))
bad = [(round(L["t"], 2), L["n_units"], L["n_onsets"]) for L in lines if L["n_onsets"] < L["n_units"] * 0.6]
print("lines with few onsets (t, units, onsets):", bad)
