"""S: walk-forward sizing of cheap levers on stored arrays (79 days). Causal: day D uses errors of days <= D-1 only
(D-1 prices are known at the gate; the forecast for D-1 was issued on D-2)."""
import numpy as np, pandas as pd, glob, os, sys
SP=sys.argv[1]
def load(d):
    out={}
    for f in sorted(glob.glob(f'{SP}/faithful/{d}/*.npz')):
        z=np.load(f,allow_pickle=True)
        if len(z['actual'])!=96: continue
        out[os.path.basename(f)[:-4]]={k:z[k] for k in ['trees','timestral','market','issued','actual','rl_trees','rl_true']}
    return out
S=load('out_strict'); S.update(load('out_fresh_strict'))
days=sorted(S); print(len(days),days[0],days[-1], 'missing:',[d for d in pd.date_range(days[0],days[-1]).strftime('%Y-%m-%d') if d not in S])
def arr(k): return np.array([S[d][k] for d in days])
y=arr('actual'); P={m:arr(m) for m in ['trees','timestral','market','issued']}
i61=np.array([d<='2026-11-07' and d>='2026-09-08' for d in days]); ifr=np.array([d>='2026-11-09' for d in days])
def rep(name,pred):
    e=np.abs(pred-y).mean(1); b=np.abs(P['issued']-y).mean(1); d=e-b
    def bb(ix):
        x=d[ix]; n=len(x); rng=np.random.default_rng(0); bs=[]
        for _ in range(2000):
            st=rng.integers(0,n,size=int(np.ceil(n/7))); idx=np.concatenate([(s+np.arange(7))%n for s in st])[:n]; bs.append(x[idx].mean())
        return np.percentile(bs,[5,95])
    # skip first 14 days as warm-up in report as well
    w=np.arange(len(days))>=14
    a,b2=bb(i61&w); c,d2=bb(ifr)
    print(f'{name:44s} 61(after 14d warmup): {e[i61&w].mean():.3f} vs {b[i61&w].mean():.3f} d={d[i61&w].mean():+.3f} [{a:+.3f},{b2:+.3f}] | fresh18: {e[ifr].mean():.3f} vs {b[ifr].mean():.3f} d={d[ifr].mean():+.3f} [{c:+.3f},{d2:+.3f}]')
E=P['issued']-y
hb=lambda x: x.reshape(len(x),24,4).mean(2).repeat(4,axis=1)
for k in [1,2,3,5,7,14]:
    for sh in [0.5,1.0]:
        pr=P['issued'].copy()
        for i in range(len(days)):
            if i>=k: pr[i]=P['issued'][i]-sh*hb(E[i-k:i]).mean(0)
        rep(f'bias corr: mean hourly err last {k}d x{sh}',pr)
for k in [3,7]:
    pr=P['issued'].copy()
    for i in range(k,len(days)): pr[i]=P['issued'][i]-0.5*E[i-k:i].mean()
    rep(f'bias corr: scalar level last {k}d x0.5',pr)
# same weekday-type? AR(1) on yesterday's error by hour with fitted coefficient walk-forward
pr=P['issued'].copy()
for i in range(15,len(days)):
    X=hb(E[1:i-0])[:-1] if False else None
    x=hb(E[:i-1]).ravel(); t=hb(E[1:i]).ravel(); c=(x@t)/(x@x); pr[i]=P['issued'][i]-c*hb(E[i-1:i])[0]
rep('AR1 on yesterday hourly err (wf coef)',pr)
# walk-forward weights: LS on last k days, nonneg-ish via clipping
M=np.stack([P['trees'],P['timestral'],P['market']],-1)
for k in [14,28]:
    pr=P['issued'].copy()
    for i in range(k,len(days)):
        A=M[i-k:i].reshape(-1,3); w=np.linalg.lstsq(A,y[i-k:i].ravel(),rcond=None)[0]; w=np.clip(w,0,None); w/=w.sum(); pr[i]=M[i]@w
    rep(f'wf LS weights last {k}d (clip, sum1)',pr)
    pr=P['issued'].copy()
    for i in range(k,len(days)):
        ma=np.abs(M[i-k:i]-y[i-k:i,:,None]).mean((0,1)); w=(1/ma**4); w/=w.sum(); pr[i]=M[i]@w
    rep(f'wf inverse-MAE^4 weights last {k}d',pr)
# wf weights + bias both
k=28; pr=P['issued'].copy()
for i in range(k,len(days)):
    A=M[i-k:i].reshape(-1,3); w=np.clip(np.linalg.lstsq(A,y[i-k:i].ravel(),rcond=None)[0],0,None); w/=w.sum()
    pe=M[i-3:i]@w-y[i-3:i]; pr[i]=M[i]@w-0.5*hb(pe).mean(0)
rep('wf LS 28d + bias 3d x0.5',pr)
# RL-error-informed: how much if RL error were removed (walk-forward slope) => bound for perfect RL
pr=P['issued'].copy(); rle=arr('rl_trees')-arr('rl_true')
for i in range(14,len(days)):
    b=np.polyfit(rle[:i].ravel(),E[:i].ravel(),1); pr[i]=P['issued'][i]-b[0]*rle[i]
rep('BOUND perfect-RL linear (wf slope, true RL err)',pr)
pr=P['issued'].copy()
for i in range(14,len(days)):
    b=np.polyfit(rle[:i].ravel(),E[:i].ravel(),1); pr[i]=P['issued'][i]-b[0]*rle[i]*0.5
rep('BOUND half of RL err removed',pr)
# trees-only checks with bias
for m in ['trees','timestral','market']:
    Em=P[m]-y; pr=P[m].copy()
    for i in range(3,len(days)): pr[i]=P[m][i]-0.5*hb(Em[i-3:i]).mean(0)
    e=np.abs(pr-y).mean(1); b=np.abs(P[m]-y).mean(1)
    print(f'{m} bias3d x0.5: 61: {e[i61].mean():.3f} vs {b[i61].mean():.3f}; fresh {e[ifr].mean():.3f} vs {b[ifr].mean():.3f}')
