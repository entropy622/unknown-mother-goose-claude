import sys
import time

sys.path.insert(0, r"D:\_Fun\claude_mv\mv")
from PIL import Image

from core import *
from geo import *

t0 = time.time()
c = new_canvas()
paper_bg(c, 3.0)
draw_heart(c, 1300, 500, 210, ay=0.5, ax=0.15, orange=0.15)
ch = load_asset("c01_stand_front")
put_fit(c, ch, 560, 560, height=900)
text(c, "我由别人的句子\n组成", 860, 300, size=30, font="serif", vertical=True, color=(0.25, 0.25, 0.27))
text(c, "アンノウン・マザーグース", 1300, 800, size=30, font="jp")
text(c, "Unknown Mother-Goose  /  Claude ver.", 1300, 845, size=16, font="thin", spacing=0.25)
put(c, spark_rgba(120), 1600, 220)
post(c, 3.0)
print("frame ms", (time.time() - t0) * 1000)
Image.fromarray(to_u8(c)).save(r"D:\_Fun\claude_mv\preview\test_look.png")

t0 = time.time()
c = new_canvas()
draw_landscape(c, 0)
draw_heart(c, W / 2, 330, 120, ay=1.0, orange=0.1)
post(c, 0)
print("land ms", (time.time() - t0) * 1000)
Image.fromarray(to_u8(c)).save(r"D:\_Fun\claude_mv\preview\test_land.png")
