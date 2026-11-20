import subprocess
import sys

sys.path.insert(0, r"D:\_Fun\claude_mv\tools")
import gen_images as g

args = [g.CODEX, "exec", "--skip-git-repo-check", "-s", "workspace-write", "-C", str(g.ROOT),
        "-c", 'model_reasoning_effort="low"', "-i", str(g.ROOT / "ref_sheet.jpg"), "-"]
p = subprocess.run(args, input="say hi", capture_output=True, text=True, encoding="utf-8", errors="replace")
print(p.returncode)
print(p.stdout[-2000:])
print(p.stderr[-3000:])
