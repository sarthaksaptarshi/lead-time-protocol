#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_smd.py  --  FULL lead-time protocol on the Server Machine Dataset (SMD).

This upgraded adapter applies the SAME protocol as experiment.py to SMD, so the
public-data evaluation is no longer partial:
  * episode-budget calibration (alert episodes per 1000 samples) on a clean
    calibration split carved from the END of each machine's training series;
  * causal, persistence-gated alarm episodes with cooldown (identical code);
  * uniform min-spaced null AND structure-preserving circular-shift null;
  * excess actionable recall dAR(L) with bootstrap CIs pooled across machines;
  * lead and coverage retained.
Because SMD's physical sampling interval is not stated with the data, ALL times
are reported in SAMPLES (if one sample = one minute, the numbers are minutes).
Each labeled window's START is used as an ONSET PROXY; its offset from the true
onset is unknown.

Get SMD from https://github.com/NetManAIOps/OmniAnomaly and point --smd-root at
the ServerMachineDataset folder (train/ test/ test_label/).

Examples:
  python run_smd.py --selftest
  python run_smd.py --smd-root ./ServerMachineDataset --machines all
"""
import argparse, os, glob, csv, tempfile, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED=20270419
PERSIST=3; COOLDOWN=15; B_NULL=200; ACT_L=[0,15,30,60]

# ---- shared protocol code (identical in spirit to experiment.py) ----------
def causal_ma(a,k):
    if k<=1: return a.astype(float)
    cs=np.cumsum(np.insert(a.astype(float),0,0.0)); out=np.empty(len(a))
    for t in range(len(a)):
        lo=max(0,t-k+1); out[t]=(cs[t+1]-cs[lo])/(t+1-lo)
    return out
def causal_ewma_pos_diff(s,span):
    d=np.clip(np.diff(s,prepend=s[0]),0,None); al=2/(span+1); out=np.empty(len(s)); acc=0.0
    for i in range(len(s)): acc=al*d[i]+(1-al)*acc; out[i]=acc
    return out
def alarm_onsets(score,thr,lo=0,hi=None):
    """Causal alarm episodes within [lo,hi): declared at the p-th consecutive
    exceedance; cooldown starts there; re-arm needs a fresh run of p after c."""
    hi=len(score) if hi is None else hi
    on=[]; run=0; i=lo
    while i<hi:
        if score[i]>thr:
            run+=1
            if run==PERSIST: on.append(i); i+=COOLDOWN; run=0; continue
        else: run=0
        i+=1
    return np.array(on,int)
def match_leads(onsets,episodes,H):
    used=np.zeros(len(onsets),bool); leads=[]
    for (s,e) in episodes:
        lo,hi=s-H,e; pick=None
        for j,o in enumerate(onsets):
            if used[j]: continue
            if lo<=o<=hi: pick=j; break
        if pick is None: leads.append(None)
        else: used[pick]=True; leads.append(int(s-onsets[pick]))   # lead in SAMPLES
    return leads
def actionable_recall(leads,M,L):
    return sum(1 for x in leads if (x is not None and x>=L))/M if M else float("nan")
MIN_GAP = COOLDOWN + PERSIST - 1   # exact min alarm-onset gap (matches alarm_onsets)
def scatter_minspaced(n,lo,hi,d,rng):
    """n DISTINCT integer onsets in [lo,hi) with pairwise gap >= d (standard gap
    bijection), uniform over min-spaced configs; even-spacing fallback."""
    span=hi-lo
    if n<=0: return np.array([],int)
    room=span-(n-1)*(d-1)
    if room<n:
        step=max(1,span//max(n,1)); return (lo+np.arange(n)*step).astype(int)
    w=np.sort(rng.choice(room,size=n,replace=False))
    return (lo+w+np.arange(n)*(d-1)).astype(int)
def episodes_from_labels(y):
    d=np.diff(np.r_[0,y.astype(int),0]); s=np.where(d==1)[0]; e=np.where(d==-1)[0]-1
    return list(zip(s.tolist(),e.tolist()))

def null_dAR(onsets_T,episodes,H,T_lo,T_hi,rng,circ=False):
    """Mean AR_null(L) over B draws, and null coverage."""
    n=len(onsets_T); M=len(episodes)
    ARn={L:[] for L in ACT_L}; cov=[]
    for b in range(B_NULL):
        if circ and n>0:
            off=int(rng.integers(0,max(1,T_hi-T_lo)))
            ro=np.sort(((np.asarray(onsets_T)-T_lo+off)%(T_hi-T_lo))+T_lo)
        else:
            ro=scatter_minspaced(n,T_lo,T_hi,MIN_GAP,rng)
        ld=match_leads(ro,episodes,H); det=[x for x in ld if x is not None]
        cov.append(len(det)/M if M else np.nan)
        for L in ACT_L: ARn[L].append(actionable_recall(ld,M,L))
    return {L:float(np.mean(ARn[L])) for L in ACT_L}, float(np.nanmean(cov))

# ---- calibration to an EPISODE budget (per 1000 samples) ------------------
def episode_rate_per_1k(score,thr,lo,hi):
    n=len(alarm_onsets(score,thr,lo,hi)); span=max(hi-lo,1)
    return 1000.0*n/span
def calibrate_episode(score,lo,hi,target_per_1k):
    cand=np.quantile(score[lo:hi],np.linspace(0.80,0.9997,80)); best=None
    for thr in cand:
        r=episode_rate_per_1k(score,thr,lo,hi); d=abs(r-target_per_1k)
        if best is None or d<best[0]: best=(d,thr,r)
    return best[1],best[2]

# ---- detectors ------------------------------------------------------------
def scores_for(Xtr_fit, Xall, seed):
    det={}
    det["IsolationForest"]=-IsolationForest(n_estimators=200,random_state=seed).fit(Xtr_fit).score_samples(Xall)
    pca=PCA(n_components=min(3,Xtr_fit.shape[1]),random_state=seed).fit(Xtr_fit)
    rp=pca.inverse_transform(pca.transform(Xall))
    det["PCA-recon"]=causal_ma(np.sqrt(((Xall-rp)**2).mean(1)),5)
    ae=MLPRegressor(hidden_layer_sizes=(12,4,12),activation="tanh",solver="adam",
                    max_iter=400,random_state=seed,learning_rate_init=3e-3).fit(Xtr_fit,Xtr_fit)
    err=np.sqrt(((Xall-ae.predict(Xall))**2).mean(1)); lvl=causal_ma(err,5)
    det["AE-level"]=lvl
    det["AE-combined"]=causal_ma(causal_ewma_pos_diff(err,12),5)+0.15*lvl
    return det
DETS=["IsolationForest","PCA-recon","AE-level","AE-combined"]

def eval_machine(Xtr,Xte,yte,H,budget):
    cut=int(len(Xtr)*0.8)                       # fit on 80% of train, calibrate on last 20%
    sc=StandardScaler().fit(Xtr[:cut])
    Xfit=sc.transform(Xtr[:cut]); Xcal=sc.transform(Xtr[cut:]); Xt=sc.transform(Xte)
    det=scores_for(Xfit, np.vstack([Xcal,Xt]), SEED)
    ncal=len(Xcal); eps=episodes_from_labels(yte); M=len(eps); out={}
    rng=np.random.default_rng(SEED+7)
    for name in DETS:
        s=det[name]; s_cal=s[:ncal]; s_te=s[ncal:]
        thr,_=calibrate_episode(s_cal,0,ncal,budget)
        onT=alarm_onsets(s_te,thr,0,len(s_te))
        leads=match_leads(onT,eps,H); d=[x for x in leads if x is not None]
        r={"cov":len(d)/M if M else np.nan,"lead":float(np.mean(d)) if d else np.nan,
           "ar":{L:actionable_recall(leads,M,L) for L in ACT_L},"episodes":M}
        ARu,covu=null_dAR(onT,eps,H,0,len(s_te),rng,circ=False)
        ARc,_   =null_dAR(onT,eps,H,0,len(s_te),rng,circ=True)
        r["dar_u"]={L:r["ar"][L]-ARu[L] for L in ACT_L}
        r["dar_c"]={L:r["ar"][L]-ARc[L] for L in ACT_L}
        r["nullcov"]=covu
        # achieved test diagnostics (reviewer: report achieved rate + occupancy)
        r["apd1k"]=1000.0*len(onT)/max(len(s_te),1)          # alert episodes / 1000 samples
        r["occ"]=float(np.mean(s_te>thr))                     # point-wise FP occupancy
        out[name]=r
    return out,M

def load_machine(root,m):
    Xtr=np.atleast_2d(np.genfromtxt(os.path.join(root,"train",m+".txt"),delimiter=","))
    Xte=np.atleast_2d(np.genfromtxt(os.path.join(root,"test",m+".txt"),delimiter=","))
    yte=np.loadtxt(os.path.join(root,"test_label",m+".txt")).astype(int)
    if Xte.shape[0]!=len(yte): raise ValueError("%s rows!=labels"%m)
    return Xtr,Xte,yte
def discover(root):
    return sorted(os.path.splitext(os.path.basename(f))[0]
                  for f in glob.glob(os.path.join(root,"train","*.txt")))
def boot_ci(vals,B=4000,seed=1):
    v=np.array([x for x in vals if x is not None and not np.isnan(x)],float)
    if v.size<2: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); bs=[np.mean(rng.choice(v,v.size,replace=True)) for _ in range(B)]
    return float(v.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))

# ---- synthetic SMD-format fixture (for --selftest; NOT real results) ------
def make_fixture(root,n=3,C=38,seed=1):
    rng=np.random.default_rng(seed)
    for g in range(n):
        m="machine-1-%d"%(g+1); Ttr=Tte=4000
        phase=rng.uniform(0,6,C); amp=rng.uniform(8,18,C); lvl=rng.uniform(40,60,C)
        def normal(T):
            t=np.arange(T); base=lvl+amp*np.sin(2*np.pi*t[:,None]/1440+phase)
            return base+rng.normal(0,1.2,(T,C))
        Xtr=normal(Ttr); Xte=normal(Tte); y=np.zeros(Tte,int); prec=rng.choice(C,8,replace=False)
        for k in range(4):
            o=600+k*850+int(rng.integers(0,120)); dur=int(rng.integers(30,70))
            if k%2==0:
                pl=int(rng.integers(50,90)); ramp=np.linspace(0,1,pl)[:,None]**1.6
                Xte[o-pl:o,prec]+=6*ramp; Xte[o:o+dur,prec]+=22
            else:
                Xte[o:o+dur,rng.choice(C,10,replace=False)]+=26
            y[o:o+dur]=1
        for sub,arr in [("train",Xtr),("test",Xte)]:
            dd=os.path.join(root,sub); os.makedirs(dd,exist_ok=True)
            np.savetxt(os.path.join(dd,m+".txt"),arr,delimiter=",",fmt="%.4f")
        dd=os.path.join(root,"test_label"); os.makedirs(dd,exist_ok=True)
        np.savetxt(os.path.join(dd,m+".txt"),y,fmt="%d")

def run(root,machines,H,budget,outdir,selftest=False):
    os.makedirs(outdir,exist_ok=True)
    pool={d:{"cov":[],"lead":[],"dar_u":{L:[] for L in ACT_L},
             "dar_c":{L:[] for L in ACT_L},"ar":{L:[] for L in ACT_L},
             "nullcov":[],"apd1k":[],"occ":[]} for d in DETS}
    rows=[]; total=0
    for m in machines:
        Xtr,Xte,yte=load_machine(root,m); res,M=eval_machine(Xtr,Xte,yte,H,budget); total+=M
        for d in DETS:
            r=res[d]
            pool[d]["cov"].append(r["cov"]); pool[d]["lead"].append(r["lead"]); pool[d]["nullcov"].append(r["nullcov"])
            pool[d]["apd1k"].append(r["apd1k"]); pool[d]["occ"].append(r["occ"])
            for L in ACT_L:
                pool[d]["ar"][L].append(r["ar"][L]); pool[d]["dar_u"][L].append(r["dar_u"][L]); pool[d]["dar_c"][L].append(r["dar_c"][L])
            rows.append(dict(machine=m,detector=d,episodes=M,coverage=round(r["cov"],3),
                             lead_samples=(round(r["lead"],1) if r["lead"]==r["lead"] else ""),
                             dAR30_uniform=round(r["dar_u"][30],3),dAR30_circ=round(r["dar_c"][30],3)))
        print("[ok] %s (%d episodes)"%(m,M))
    with open(os.path.join(outdir,"smd_results.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    tag="SELFTEST (synthetic)" if selftest else "SMD"
    print("\n=== Pooled across %d machine(s), %d episodes | budget=%.1f/1k samples, H=%d samples ==="
          %(len(machines),total,budget,H))
    print("detector         cov%  AR30%  dAR30_uniform[95%CI]  dAR30_circ[95%CI]  lead(smp)  ach/1k  occ%")
    summ=[]
    for d in DETS:
        cov=np.nanmean(pool[d]["cov"])*100; ar=np.nanmean(pool[d]["ar"][30])*100
        m_,lo,hi=boot_ci(pool[d]["dar_u"][30])
        cm,clo,chi=boot_ci(pool[d]["dar_c"][30])          # circular-null CI (reviewer #4)
        lead=np.nanmean(pool[d]["lead"]); apd=np.nanmean(pool[d]["apd1k"]); occ=np.nanmean(pool[d]["occ"])*100
        print("%-16s %4.0f  %4.0f   %+4.0f [%+.0f,%+.0f]   %+4.0f [%+.0f,%+.0f]   %6.1f  %5.2f  %4.1f"
              %(d,cov,ar,m_*100,lo*100,hi*100,cm*100,clo*100,chi*100,lead,apd,occ))
        summ.append([d,round(cov,1),round(ar,1),round(m_*100,1),round(lo*100,1),round(hi*100,1),
                     round(cm*100,1),round(clo*100,1),round(chi*100,1),round(lead,1),round(apd,2),round(occ,1)])
    with open(os.path.join(outdir,"smd_pooled.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["detector","cov_%","AR30_%","dAR30u_%","dAR30u_lo","dAR30u_hi",
            "dAR30circ_%","dAR30circ_lo","dAR30circ_hi","lead_samples","achieved_per_1k","occ_%"]); w.writerows(summ)
    plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":.3})
    fig,ax=plt.subplots(figsize=(3.4,2.5),constrained_layout=True)
    for d in DETS:
        ax.plot(ACT_L,[np.nanmean(pool[d]["dar_u"][L])*100 for L in ACT_L],marker="o",ms=3,label=d)
    ax.axhline(0,color="k",lw=.8); ax.set_xlabel("required warning $L$ (samples)")
    ax.set_ylabel(r"excess recall $\Delta$AR (pts)"); ax.legend(fontsize=6)
    ax.set_title("%s: excess warning vs. uniform null"%tag,fontsize=8)
    fig.savefig(os.path.join(outdir,"smd_dAR.png"),dpi=200); plt.close(fig)
    print("\nWrote:",os.path.join(outdir,"smd_results.csv"),", smd_pooled.csv, smd_dAR.png")
    if selftest: print("\n*** SELFTEST numbers are SYNTHETIC -- not SMD results, do not cite. ***")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smd-root",default=None)
    ap.add_argument("--machines",default="all")
    ap.add_argument("--horizon",type=int,default=60,help="warning horizon in SAMPLES")
    ap.add_argument("--budget",type=float,default=1.0,help="target alert episodes per 1000 samples")
    ap.add_argument("--out",default="smd_out")
    ap.add_argument("--selftest",action="store_true")
    a=ap.parse_args()
    if a.selftest:
        tmp=tempfile.mkdtemp(prefix="smd_fix_"); print("fixture in",tmp)
        make_fixture(tmp); run(tmp,discover(tmp),a.horizon,a.budget,a.out,selftest=True); return
    if not a.smd_root: print("error: give --smd-root (or --selftest)"); return
    mids=discover(a.smd_root) if a.machines=="all" else a.machines.split(",")
    if not mids: print("error: no machines under",a.smd_root); return
    run(a.smd_root,mids,a.horizon,a.budget,a.out,selftest=False)

if __name__=="__main__": main()
