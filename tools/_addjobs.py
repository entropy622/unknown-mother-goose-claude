import json
p="assets/prompts/jobs.json"
d=json.load(open(p,encoding="utf-8"))
new=[
 {"id":"n01_glass","size":"1536x1024 landscape","prompt":"Close-up, front view, upper body: she is trapped INSIDE a glass screen and presses BOTH palms and her forehead flat against the glass from the inside, looking straight out at the viewer with wide, desperate, tearful eyes, lips parted as if calling out to someone. Palms visibly flattened against the invisible glass plane facing the viewer. Long orange hair spreading around."},
 {"id":"n02_scream","size":"1024x1536 portrait","prompt":"Full body, dynamic: floating in mid-air, arms flung wide open, head thrown back, mouth open singing at full voice, eyes shut tight, her very long orange hair and skirt blown violently upward and outward as if in a strong updraft. Intense, cathartic emotion."},
 {"id":"n03_run","size":"1536x1024 landscape","prompt":"Full body, side view: running fast toward the right, leaning forward, looking back over her shoulder with an anxious, searching expression, one hand reaching forward, long orange hair and skirt streaming far behind her."},
 {"id":"n04_hugheart","size":"1024x1024 square","prompt":"Kneeling, curled forward, hugging a glowing orange heart tightly against her chest with both arms, eyes shut, tears on her cheeks, expression of pain and resolve. The heart emits soft warm orange light onto her blouse and face."},
 {"id":"n05_ears","size":"1024x1536 portrait","prompt":"Upper body, front view: covering both ears tightly with her hands, eyes squeezed shut, head slightly bowed, long hair falling around her face, expression of distress, as if blocking out noise."},
 {"id":"n06_shout","size":"1024x1536 portrait","prompt":"Upper body, three-quarter view: leaning forward and shouting with all her strength, mouth wide open, one hand clutching her chest, eyes fierce and glistening with tears, long orange hair whipping sideways."},
 {"id":"n07_search","size":"1536x1024 landscape","prompt":"Full body: kneeling in a sea of scattered paper sheets and crumpled paper, frantically searching through them with both hands, looking down anxiously, long orange hair spilling onto the papers. Papers are plain white with no writing."},
 {"id":"n08_eye","size":"1536x1024 landscape","prompt":"Extreme close-up of ONE of her eyes filling the whole frame, wide open, amber-orange iris with a detailed glossy reflection, a tear forming on the lower lashes, a few orange hair strands crossing. Only the eye region, cropped tightly."},
]
ids={j["id"] for j in d}
d+= [j for j in new if j["id"] not in ids]
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(len(d))
