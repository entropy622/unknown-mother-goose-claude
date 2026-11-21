"""Batch image generation through the local codex CLI (image_generation tool)."""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODEX = r"C:\Users\Aentro\.vscode\extensions\openai.chatgpt-26.1117.31558-win32-x64\bin\windows-x86_64\codex.exe"
ROOT = Path(r"D:\_Fun\claude_mv\assets")
RAW = ROOT / "raw"
STYLE = (ROOT / "prompts" / "_style.txt").read_text(encoding="utf-8")
OBJ_STYLE = """STYLE:
- Quiet, minimal, monochrome look matching wowaka's "Unknown Mother-Goose" music video: white / light gray / charcoal tones, soft studio lighting, clean edges.
- The ONLY saturated color allowed is a warm orange (#D97757 to #E8853D), and only where the description asks for it.
- No text, no logos, no watermark, no people unless asked.
- Background: fully TRANSPARENT (alpha) PNG if supported, otherwise perfectly flat pure #00FF00 green.
"""
USER_STYLE = """STYLE:
- A young Chinese software engineer (male, mid-20s): short black hair, soft tired eyes, calm quiet face, dark simple hoodie.
- Art style MUST match wowaka's music-video artwork: clean flat cel-shaded anime, thin crisp line art, very few shading tones, slightly desaturated, elegant and quiet, lots of negative space.
- Palette: white / light gray / charcoal / black. The ONLY saturated accent allowed is a cool screen-blue glow where described (not orange).
- No text, no watermark, no signature, no logos.
- Background: fully TRANSPARENT (alpha) PNG if supported, otherwise perfectly flat pure #00FF00 green.
"""

JOBS = json.loads((ROOT / "prompts" / "jobs.json").read_text(encoding="utf-8"))


def run(job):
    out = RAW / f"{job['id']}.png"
    if out.exists():
        return job["id"], "skip"
    kind = job.get("kind", "char")
    style = {"char": STYLE, "obj": OBJ_STYLE, "user": USER_STYLE}.get(kind, OBJ_STYLE)
    if job.get("opaque"):
        style = style.replace(
            "- Background: fully TRANSPARENT (alpha) PNG if your image tool supports it. If transparency is not supported, use a perfectly flat, uniform pure chroma green #00FF00 background with no shadows, no gradient, no floor.",
            "- Background: as described (opaque full-frame image).",
        ).replace(
            "- Background: fully TRANSPARENT (alpha) PNG if supported, otherwise perfectly flat pure #00FF00 green.",
            "- Background: as described (opaque full-frame image).",
        )
    prompt = (
        "Use your image generation tool to create exactly ONE image, then copy the resulting PNG file to "
        f"{out} (create the folder if needed). Do nothing else. Reply only with the saved path.\n\n"
        f"IMAGE: {job['prompt']}\nAspect / size: {job.get('size', '1024x1536')}.\n\n{style}"
    )
    args = [CODEX, "exec", "--skip-git-repo-check", "-s", "workspace-write", "-C", str(ROOT),
            "-c", 'model_reasoning_effort="low"']
    if kind == "char":
        args += ["-i", str(ROOT / job.get("ref", "ref_sheet.jpg"))]
    args += ["-"]
    t0 = time.time()
    for attempt in range(3):
        p = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if out.exists():
            return job["id"], f"ok {time.time()-t0:.0f}s"
        time.sleep(15)
    (RAW / f"{job['id']}.log").write_text(f"rc={p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-6000:]}", encoding="utf-8")
    return job["id"], "FAILED"


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    jobs = [j for j in JOBS if not only or j["id"] in only]
    workers = 2

    def staggered(ij):
        i, job = ij
        if i < workers:
            time.sleep(i * 20)
        return run(job)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for jid, status in ex.map(staggered, enumerate(jobs)):
            print(jid, status, flush=True)
