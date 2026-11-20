import numpy as np, json
from scipy.io import wavfile
from scipy.signal import find_peaks
sr,y=wavfile.read("audio/stem_vocals.wav"); y=y.astype(np.float32).mean(1)/32768
OFF=1.592; fr=0.01
B=np.array(json.load(open("audio/beats.json"))["beats"])
for a,b in [(106.0,116.6),(218.6,229.2)]:
    i0=int((a-OFF)*sr); i1=int((b-OFF)*sr); seg=y[i0:i1]
    n=int(len(seg)/(fr*sr)); e=np.array([np.sqrt((seg[int(k*fr*sr):int((k+1)*fr*sr)]**2).mean()+1e-12) for k in range(n)])
    db=20*np.log10(e); d=np.diff(np.convolve(db,np.ones(3)/3,'same'),prepend=db[0])
    pk,_=find_peaks(d,height=2.0,distance=40)
    ts=a+pk*fr
    bi=[ (round(t,2), round(((t-B[np.searchsorted(B,t)-1])/(B[1]-B[0])),2)) for t in ts]
    print(a, len(ts), bi)
    print("".join(" .:-=+*#%@"[min(9,max(0,int((v+50)/4)))] for v in db[::5]))
