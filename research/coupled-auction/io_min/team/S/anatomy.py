"""S: error anatomy on stored per-day npz (61 champion days + 18 fresh). Offline, seconds."""
import numpy as np, pandas as pd, glob, os, sys
SP=sys.argv[1]
def load(d, lo=None, hi=None):
    rows=[]
    for f in sorted(glob.glob(f'{SP}/faithful/{d}/*.npz')):
        day=os.path.basename(f)[:-4]
        if lo and not (lo<=day<=hi): continue
        z=np.load(f,allow_pickle=True)
        t=pd.to_datetime(z['idx'],unit='us',utc=True).tz_convert('Europe/Berlin')
        df=pd.DataFrame({k:z[k] for k in ['trees','timestral','market','issued','actual','rl_trees','rl_mixed','rl_true']})
        if len(df)!=96: 
            pass
        df['t']=t; df['day']=day; df['wk']=z['weather_k_DE'] if 'weather_k_DE' in z.files else np.nan
        rows.append(df)
    return pd.concat(rows,ignore_index=True)
A=load('out_strict','2026-09-08','2026-11-07'); A['set']='61'
B=load('out_fresh_strict'); B['set']='fresh18'
M=load('out_strict_rlmatched','2026-09-08','2026-11-07'); MB=load('out_fresh_strict_rlmatched')
X=pd.concat([A,B],ignore_index=True)
mem=['trees','timestral','market','issued']
for m in mem: X['e_'+m]=X[m]-X.actual; X['a_'+m]=X['e_'+m].abs()
X['h']=X.t.dt.hour
X['win']=pd.cut(X.h,[-1,5,9,15,20,23],labels=['00-06','06-10','10-16','16-21','21-24'])
X['dow']=X.t.dt.day_name().str[:3]; X['month']=X.t.dt.strftime('%m')
X['reg']=pd.cut(X.actual,[-1e9,0.0,3,8,13,1e9],labels=['<=0','0-3','3-8','8-13','>13'])
X['rle']=(X.rl_trees-X.rl_true); X['rlabs']=pd.cut(X.rle.abs(),[-1,1,2,4,99],labels=['<1','1-2','2-4','>4GW'])
X['rlsign']=pd.cut(X.rle,[-99,-2,2,99],labels=['RL low>2','ok','RL high>2'])
def tab(by,df=X):
    g=df.groupby(by,observed=True)
    o=g[['a_'+m for m in mem]].mean().round(2)
    o['bias_bl']=g.e_issued.mean().round(2)
    o['share_n']=(g.size()/len(df)).round(3)
    o['share_err']=(g.a_issued.sum()/df.a_issued.sum()).round(3)
    return o
pd.set_option('display.width',200)
for s in ['61','fresh18']:
    D=X[X.set==s]; print(f'\n######## {s}: days={D.day.nunique()} blend MAE={D.a_issued.mean():.3f} actual mean={D.actual.mean():.2f} std={D.actual.std():.2f}')
    for by in ['win','dow','reg','rlabs','rlsign','month','wk']:
        print(f'\n-- by {by}'); print(tab(by,D).to_string())
    print('\n-- win x reg share of blend error'); print((D.pivot_table(index='win',columns='reg',values='a_issued',aggfunc='sum',observed=True)/D.a_issued.sum()).round(3).to_string())
    # daily concentration
    dd=D.groupby('day').a_issued.mean().sort_values(ascending=False)
    print('\ntop-10 days share of error: %.3f ; worst days:'%(dd.head(10).sum()/dd.sum()), dd.head(8).round(2).to_dict())
    # error decomposition: daily-level (mean bias per day) vs shape
    lvl=D.groupby('day').e_issued.transform('mean'); 
    print('level-corrected (per-day oracle shift) MAE: %.3f ; per-day-window(5) shift: %.3f'%((D.e_issued-lvl).abs().mean(), (D.e_issued-D.groupby(['day','win'],observed=True).e_issued.transform('mean')).abs().mean()))
    # member correlation & bounds
    E=D[['e_trees','e_timestral','e_market']]; print('member err corr:\n',E.corr().round(2).to_string())
    P=D[['trees','timestral','market']].values; y=D.actual.values
    # best fixed weights (in-sample bound)
    w=np.linalg.lstsq(P,y,rcond=None)[0]; print('best fixed LS weights',w.round(2),'MAE %.3f'%np.abs(P@w-y).mean())
    # per-window LS weights
    pr=np.zeros(len(D)); 
    for wn,ix in D.groupby('h').indices.items():
        ww=np.linalg.lstsq(np.c_[P[ix],np.ones(len(ix))],y[ix],rcond=None)[0]; pr[ix]=np.c_[P[ix],np.ones(len(ix))]@ww
    print('per-hour LS weights+intercept (in-sample bound) MAE %.3f'%np.abs(pr-y).mean())
    # per-day oracle member
    dm=D.groupby('day')[['a_trees','a_timestral','a_market','a_issued']].mean(); print('per-day oracle best-of-4 MAE %.3f ; per-slot oracle best member %.3f ; median-of-3 %.3f'%(dm.min(axis=1).mean(), D[['a_trees','a_timestral','a_market']].min(axis=1).mean(), np.abs(np.median(P,axis=1)-y).mean()))
    # hourly bias correction bound (in-sample per-hour mean error removal)
    print('remove per-hour mean bias (in-sample): %.3f'%(D.e_issued-D.groupby('h').e_issued.transform('mean')).abs().mean())
    # RL sensitivity: regress blend err on rl err
    for m in mem:
        b=np.polyfit(D.rle,D['e_'+m],1); r=np.corrcoef(D.rle,D['e_'+m])[0,1]
        res=D['e_'+m]-np.polyval(b,D.rle)
        print(f'{m}: err ~ {b[0]:.3f} ct per GW RL err, r={r:.2f}; MAE after removing RL-linear part {res.abs().mean():.3f} (from {D["a_"+m].mean():.3f})')
    # by hour detail
    print('\n-- by hour'); print(tab('h',D)[['a_issued','bias_bl','share_err']].T.round(2).to_string())
    # naive week-ago not available here.
# matched comparisons by window
for nm,base,var in [('61',A,M),('fresh18',B,MB)]:
    j=base.merge(var,on=['day','t'],suffixes=('','_m'))
    j['h']=j.t.dt.hour; j['win']=pd.cut(j.h,[-1,5,9,15,20,23],labels=['00-06','06-10','10-16','16-21','21-24'])
    o={}
    for m in mem: o[m]=(j[m+'_m']-j.actual).abs().groupby(j.win,observed=True).mean()-(j[m]-j.actual).abs().groupby(j.win,observed=True).mean()
    print(f'\nrlmatched - base by window ({nm}, {j.day.nunique()} days)'); print(pd.DataFrame(o).round(3).to_string())
    print('overall', {m: round(float((j[m+'_m']-j.actual).abs().mean()-(j[m]-j.actual).abs().mean()),3) for m in mem}, 'RL MAE', round(float((j.rl_trees-j.rl_true).abs().mean()),2),'->',round(float((j.rl_trees_m-j.rl_true).abs().mean()),2))
X.to_pickle(f'{SP}/team_S_anatomy.pkl') if False else None
