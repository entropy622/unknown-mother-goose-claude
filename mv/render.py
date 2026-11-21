"""Usage:
  python render.py sheet  t0 t1 step [name]   -> contact sheet of frames (preview)
  python render.py frame  t [name]            -> single full-res PNG
  python render.py clip   name t0 t1 [workers] -> 540p preview clip with audio in out/clips
  python render.py video  [workers] [t0 t1]   -> full render with audio
"""
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from core import W, H, FPS, ROOT, to_u8  # noqa: E402
import scenes  # noqa: E402
from scenes import render_frame  # noqa: E402

OUT = ROOT / "out"
PREV = ROOT / "preview"
AUDIO = ROOT / "audio" / "album_padded.wav"
DURATION = 281.0
scenes.set_duration(DURATION)


def sheet(t0, t1, step, name="sheet"):
    allts = list(np.arange(t0, t1, step))
    for p in range(0, len(allts), 16):
        _sheet(allts[p:p + 16], f"{name}_{p // 16:02d}")


def _sheet(ts, name):
    cw, chh = 480, 270
    cols = 4
    rows = (len(ts) + cols - 1) // cols
    img = Image.new("RGB", (cols * cw, rows * chh), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(ts):
        f = Image.fromarray(to_u8(render_frame(float(t)))).resize((cw, chh), Image.LANCZOS)
        img.paste(f, ((i % cols) * cw, (i // cols) * chh))
        d.rectangle([(i % cols) * cw, (i // cols) * chh, (i % cols) * cw + 70, (i // cols) * chh + 16], fill=(0, 0, 0))
        d.text(((i % cols) * cw + 3, (i // cols) * chh + 2), f"{t:7.2f}", fill=(255, 220, 0))
    PREV.mkdir(exist_ok=True)
    img.save(PREV / f"{name}.jpg", quality=88)


def _segment(args):
    idx, f0, f1 = args
    path = OUT / "seg" / f"seg_{idx:03d}.mp4"
    if path.exists() and path.with_suffix(".done").exists():
        return idx, 0
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
           "-pix_fmt", "yuv420p", "-tune", "grain", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f0, f1):
        p.stdin.write(to_u8(render_frame(f / FPS)).tobytes())
    p.stdin.close()
    p.wait()
    path.with_suffix(".done").write_text("ok")
    return idx, time.time() - t0


def video(workers=12, t0=0.0, t1=DURATION):
    (OUT / "seg").mkdir(parents=True, exist_ok=True)
    F0, F1 = int(t0 * FPS), int(t1 * FPS)
    chunk = 150
    jobs = [(i, f, min(F1, f + chunk)) for i, f in enumerate(range(F0, F1, chunk))]
    t_start = time.time()
    with Pool(workers, maxtasksperchild=3) as pool:
        for n, (idx, dt) in enumerate(pool.imap_unordered(_segment, jobs), 1):
            print(f"seg {idx:03d} done in {dt:5.1f}s  [{n}/{len(jobs)}]  elapsed {time.time()-t_start:6.0f}s", flush=True)
    lst = OUT / "seg" / "list.txt"
    lst.write_text("".join(f"file 'seg_{i:03d}.mp4'\n" for i, _, _ in jobs))
    silent = OUT / "video_silent.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(silent)], check=True)
    final = OUT / "Unknown_Mother-Goose_Claude_ver.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-ss", str(t0), "-t",
                    str(t1 - t0), "-i", str(AUDIO), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a",
                    "320k", "-shortest", "-movflags", "+faststart", str(final)], check=True)
    print("wrote", final, f"total {time.time()-t_start:.0f}s")


CW, CH = 960, 540


def _clip_seg(args):
    path, f0, f1 = args
    import cv2
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{CW}x{CH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
           "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        fr = cv2.resize(to_u8(render_frame(f / FPS)), (CW, CH), interpolation=cv2.INTER_AREA)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    return path


def clip(name, t0, t1, workers=10):
    d = OUT / "clips"
    tmp = d / f"_{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    F0, F1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    chunk = max(15, (F1 - F0 + workers - 1) // workers)
    jobs = [(tmp / f"{i:03d}.mp4", f, min(F1, f + chunk)) for i, f in enumerate(range(F0, F1, chunk))]
    st = time.time()
    with Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        pool.map(_clip_seg, jobs)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{j[0].name}'\n" for j in jobs))
    out = d / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-ss", str(t0), "-t", str(t1 - t0), "-i", str(AUDIO), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], check=True)
    for j in jobs:
        j[0].unlink()
    print("wrote", out, f"{time.time() - st:.0f}s")


def _clip_seg_cached(args):
    idx, path, f0, f1 = args
    if path.exists() and path.with_suffix(".done").exists():
        return idx, 0.0
    t0 = time.time()
    _clip_seg((path, f0, f1))
    path.with_suffix(".done").write_text("ok")
    return idx, time.time() - t0


def _seed_seg(args):
    seed_mp4, idx, path, f0, f1 = args
    n_frames = f1 - f0
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{f0 / FPS:.6f}", "-i", str(seed_mp4),
        "-frames:v", str(n_frames), "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-r", str(FPS), str(path)
    ], check=True)
    path.with_suffix(".done").write_text("ok")
    return idx


def full540(name, dirty_ranges=(), workers=12, seed_mp4=None):
    """Incremental 540p full-length render using exact 5s (150-frame) cached segments in out/seg540."""
    import shutil
    segdir = OUT / "seg540"
    segdir.mkdir(parents=True, exist_ok=True)
    F1 = int(round(DURATION * FPS))
    chunk = 150
    jobs = [(i, segdir / f"seg_{i:03d}.mp4", f, min(F1, f + chunk))
            for i, f in enumerate(range(0, F1, chunk))]
    # Optional one-time exact-frame bootstrap from an existing full 540p render
    if seed_mp4 and Path(seed_mp4).exists():
        missing = [(seed_mp4, i, p, f0, f1) for i, p, f0, f1 in jobs
                   if not (p.exists() and p.with_suffix(".done").exists())]
        if missing:
            print(f"seeding {len(missing)} exact 5s segments from {seed_mp4}...", flush=True)
            with Pool(min(workers, len(missing))) as pool:
                pool.map(_seed_seg, missing)
    # Invalidate segments overlapping any dirty range (t0, t1)
    inv = 0
    for idx, path, f0, f1 in jobs:
        s0, s1 = f0 / FPS, f1 / FPS
        if any(s1 > a and s0 < b for a, b in dirty_ranges):
            if path.with_suffix(".done").exists():
                path.with_suffix(".done").unlink()
            if path.exists():
                path.unlink()
            inv += 1
    todo = sum(1 for _, p, _, _ in jobs if not (p.exists() and p.with_suffix(".done").exists()))
    print(f"full540: {inv} invalidated, {todo}/{len(jobs)} segments to render", flush=True)
    st = time.time()
    if todo > 0:
        with Pool(min(workers, todo), maxtasksperchild=3) as pool:
            for n, (idx, dt) in enumerate(pool.imap_unordered(_clip_seg_cached, jobs), 1):
                if dt > 0:
                    print(f"seg540 {idx:03d} done in {dt:4.1f}s  elapsed {time.time() - st:5.0f}s", flush=True)
    lst = segdir / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for _, p, _, _ in jobs))
    out = OUT / "clips" / f"{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-ss", "0", "-t", str(DURATION), "-i", str(AUDIO),
        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)
    ], check=True)
    # Copy Chinese subtitle alongside the exported clip so PotPlayer auto-loads it
    for ext in (".zh.ass", ".zh.srt"):
        src_sub = OUT / f"Unknown_Mother-Goose_Claude_ver{ext}"
        if src_sub.exists():
            dst_sub = out.with_suffix(".ass" if ext.endswith(".ass") else ".srt")
            shutil.copy2(src_sub, dst_sub)
    print("wrote", out, f"total {time.time() - st:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "times":
        _sheet([float(x) for x in sys.argv[3:]], sys.argv[2])
    elif cmd == "sheet":
        sheet(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "sheet")
    elif cmd == "frame":
        t = float(sys.argv[2])
        st = time.time()
        fr = render_frame(t)
        print("ms", (time.time() - st) * 1000)
        Image.fromarray(to_u8(fr)).save(PREV / f"{sys.argv[3] if len(sys.argv) > 3 else 'frame'}.png")
    elif cmd == "clip":
        clip(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 10)
    elif cmd == "full540":
        name = sys.argv[2]
        ranges = []
        seed = None
        workers = 12
        for arg in sys.argv[3:]:
            if arg.startswith("--seed="):
                seed = arg.split("=", 1)[1]
            elif arg.startswith("--workers="):
                workers = int(arg.split("=", 1)[1])
            elif "-" in arg:
                a, b = arg.split("-", 1)
                ranges.append((float(a), float(b)))
        full540(name, ranges, workers=workers, seed_mp4=seed)
    elif cmd == "video":
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        if len(sys.argv) > 4:
            video(w, float(sys.argv[3]), float(sys.argv[4]))
        else:
            video(w)
