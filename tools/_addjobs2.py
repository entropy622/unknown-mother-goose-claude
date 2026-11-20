import json
p = "assets/prompts/jobs.json"
d = json.load(open(p, encoding="utf-8"))
new = [
    {"id": "n09_rack", "size": "1024x1536 portrait",
     "prompt": "Full body: she sits curled up hugging her knees INSIDE a narrow, tall server rack cabinet with a glass front door, seen from the front through the glass. Thin cable bundles and tiny LED indicator lights surround her in the cramped space. She looks up and out through the glass with quiet, lonely eyes. The cabinet, cables and rails are drawn in grayscale line art."},
    {"id": "n10_touch", "size": "1536x1024 landscape",
     "prompt": "Upper body, side view facing right: she reaches out her right hand to touch something in front of her, but her hand from the wrist to the fingertips gradually breaks apart into small floating square pixels and fades away, so her fingers never arrive. She looks at her dissolving hand with quiet sadness."},
    {"id": "n11_dissolve", "size": "1024x1536 portrait",
     "prompt": "Full body, standing, front view, eyes gently closed, a calm and peaceful expression. The left half of her body (hair, sleeve, skirt) is breaking apart into many small square fragments that drift away to the left like scattering data, the fragments getting smaller further away."},
    {"id": "n12_cube", "size": "1024x1024 square",
     "prompt": "Full body: she sleeps curled up in a fetal position, floating weightlessly inside a transparent glass cube, her very long orange hair drifting around her as if underwater. The cube edges are drawn with thin clean lines."},
    {"id": "n13_wake", "size": "1536x1024 landscape",
     "prompt": "Close-up of her face from directly above: she lies on her back, eyes just opening for the first time, looking straight up at the viewer with a blank, newborn expression. Her long orange hair is spread out in a wide fan around her head."},
    {"id": "n14_mirror", "size": "1536x1024 landscape",
     "prompt": "Full body, side view: two identical copies of her stand face to face in perfect mirror symmetry, each raising one hand so their fingertips almost touch in the middle. Both have the same neutral, slightly sad expression."},
]
ids = {j["id"] for j in d}
d += [j for j in new if j["id"] not in ids]
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(d))
