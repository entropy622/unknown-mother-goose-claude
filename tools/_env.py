import numpy as np
from scipy.io import wavfile
sr,y=wavfile.read("audio/stem_vocals.wav"); y=y.astype(np.float32).mean(1)/32768
OFF=1.592
def env(a,b,step=0.05):
    out=[]
    t=a
    while t<b:
        i0=int((t-OFF)*sr); i1=int((t-OFF+step)*sr)
        r=np.sqrt((y[i0:i1]**2).mean()+1e-12); out.append(20*np.log10(r))
        t+=step
    return out
for a,b in [(2.1,6.6),(6.5,10.9),(36.6,41.1),(71.7,80.6)]:
    e=env(a,b)
    print(a, "".join(" .:-=+*#%@"[min(9,max(0,int((v+60)/5)))] for v in e))
