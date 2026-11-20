import sys
from pathlib import Path

from PIL import Image, ImageDraw

raw = Path(r"D:\_Fun\claude_mv\assets\raw")
names = sys.argv[1:] or sorted(p.stem for p in raw.glob("*.png"))
cell = 360
cols = 5
rows = (len(names) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell + 24)), (200, 200, 205))
d = ImageDraw.Draw(sheet)
for i, n in enumerate(names):
    im = Image.open(raw / f"{n}.png").convert("RGBA")
    im.thumbnail((cell - 10, cell - 10))
    x, y = (i % cols) * cell, (i // cols) * (cell + 24)
    bg = Image.new("RGBA", im.size, (200, 200, 205, 255))
    bg.alpha_composite(im)
    sheet.paste(bg.convert("RGB"), (x + (cell - im.width) // 2, y + 24 + (cell - im.height) // 2))
    d.text((x + 6, y + 4), f"{n} {Image.open(raw / f'{n}.png').size}", fill=(0, 0, 0))
sheet.save(r"D:\_Fun\claude_mv\preview\assets_contact.jpg", quality=88)
