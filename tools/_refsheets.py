import subprocess, sys
from PIL import Image, ImageDraw
def sheet(t0,t1,step,name):
    ts=[];t=t0
    while t<t1: ts.append(round(t,2)); t+=step
    ims=[]
    for t in ts:
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",str(t),"-i","ref/ref.mp4","-frames:v","1","-vf","scale=384:216","preview/_r.png"],check=True)
        im=Image.open("preview/_r.png").convert("RGB"); ImageDraw.Draw(im).text((3,3),f"{t:.1f}",fill=(255,220,0)); ims.append(im)
    cols=5; rows=(len(ims)+cols-1)//cols
    o=Image.new("RGB",(cols*384,rows*216))
    for i,im in enumerate(ims): o.paste(im,((i%cols)*384,(i//cols)*216))
    o.save(f"preview/ref_{name}.jpg",quality=85)
sheet(84,96,0.6,"a")
sheet(102,118,0.8,"b")
sheet(132,150,0.9,"c")
sheet(236,252,0.8,"d")
sheet(70,84,0.7,"e")
sheet(150,168,0.9,"f")
