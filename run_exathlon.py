#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_exathlon.py  --  FULL lead-time protocol on the Exathlon benchmark.

Why Exathlon (Jacob/Song/Stiegler/Rad/Diao/Tatbul, VLDB 2021,
github.com/exathlonbenchmark/exathlon): it is real Spark-cluster metric
monitoring with, for every injected anomaly, a labelled ROOT-CAUSE interval and
an extended EFFECT interval. We take the ROOT-CAUSE START as the true onset, so
lead time is measured against a real onset (not a window proxy) and, because
traces carry a real timestamp column, we can report lead in MINUTES, not
samples. Undisturbed traces train/calibrate; disturbed traces are the test set.

This reuses the SAME protocol as experiment.py / run_smd.py: causal
persistence-gated alarm episodes, episode-budget calibration, uniform and
circular nulls, excess actionable recall dAR(L) with cluster-bootstrap CIs.

Because Exathlon's exact ground-truth column names and timestamp units may
differ by release, every column name is a CLI flag -- CHECK them against the
dataset documentation and adjust if needed. Verify the pipeline first with
    python run_exathlon.py --selftest
(a synthetic Exathlon-format fixture; NOT real results), then point --data-root
and --ground-truth at the extracted dataset.

Requires numpy / scipy / scikit-learn / matplotlib (+ pandas for CSV I/O).
"""
import argparse, os, glob, csv, tempfile, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
try:
    import pandas as pd
except Exception:
    pd=None

SEED=20270419
PERSIST=3; COOLDOWN=15; B_NULL=200
MIN_GAP=COOLDOWN+PERSIST-1
ACT_L_MIN=[0,5,15,30]          # required-warning thresholds in MINUTES

# ---- shared protocol code (identical to experiment.py / run_smd.py) -------
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
    """episodes: list of (onset_idx, end_idx). Lead in SAMPLES."""
    used=np.zeros(len(onsets),bool); leads=[]
    for (s,e) in episodes:
        lo,hi=s-H,e; pick=None
        for j,o in enumerate(onsets):
            if used[j]: continue
            if lo<=o<=hi: pick=j; break
        if pick is None: leads.append(None)
        else: used[pick]=True; leads.append(int(s-onsets[pick]))
    return leads
def actionable_recall(leads_min,M,L):
    return sum(1 for x in leads_min if (x is not None and x>=L))/M if M else float("nan")
def scatter_minspaced(n,lo,hi,d,rng):
    span=hi-lo
    if n<=0: return np.array([],int)
    room=span-(n-1)*(d-1)
    if room<n:
        step=max(1,span//max(n,1)); return (lo+np.arange(n)*step).astype(int)
    w=np.sort(rng.choice(room,size=n,replace=False))
    return (lo+w+np.arange(n)*(d-1)).astype(int)

def null_dAR(onsets_T,episodes,H,T_lo,T_hi,step_min,rng,circ=False):
    n=len(onsets_T); M=len(episodes); ARn={L:[] for L in ACT_L_MIN}
    for b in range(B_NULL):
        if circ and n>0:
            off=int(rng.integers(0,max(1,T_hi-T_lo)))
            ro=np.sort(((np.asarray(onsets_T)-T_lo+off)%(T_hi-T_lo))+T_lo)
        else:
            ro=scatter_minspaced(n,T_lo,T_hi,MIN_GAP,rng)
        ld=[(x*step_min if x is not None else None) for x in match_leads(ro,episodes,H)]
        for L in ACT_L_MIN: ARn[L].append(actionable_recall(ld,M,L))
    return {L:float(np.mean(ARn[L])) for L in ACT_L_MIN}

def episode_rate_per_1k(score,thr,lo,hi):
    return 1000.0*len(alarm_onsets(score,thr,lo,hi))/max(hi-lo,1)
def calibrate_episode(score,lo,hi,target):
    cand=np.quantile(score[lo:hi],np.linspace(0.80,0.9997,80)); best=None
    for thr in cand:
        r=episode_rate_per_1k(score,thr,lo,hi); d=abs(r-target)
        if best is None or d<best[0]: best=(d,thr,r)
    return best[1],best[2]

def fit_detectors(Xfit,seed):
    det={}
    det["IsolationForest"]=IsolationForest(n_estimators=200,random_state=seed).fit(Xfit)
    det["PCA"]=PCA(n_components=min(3,Xfit.shape[1]),random_state=seed).fit(Xfit)
    det["AE"]=MLPRegressor(hidden_layer_sizes=(12,4,12),activation="tanh",solver="adam",
                           max_iter=300,random_state=seed,learning_rate_init=3e-3).fit(Xfit,Xfit)
    return det
def score_with(det,X):
    s={}
    s["IsolationForest"]=-det["IsolationForest"].score_samples(X)
    rp=det["PCA"].inverse_transform(det["PCA"].transform(X))
    s["PCA-recon"]=causal_ma(np.sqrt(((X-rp)**2).mean(1)),5)
    err=np.sqrt(((X-det["AE"].predict(X))**2).mean(1)); lvl=causal_ma(err,5)
    s["AE-level"]=lvl
    s["AE-combined"]=causal_ma(causal_ewma_pos_diff(err,12),5)+0.15*lvl
    return s
DETS=["IsolationForest","PCA-recon","AE-level","AE-combined"]

def boot_ci(vals,B=4000,seed=1):
    v=np.array([x for x in vals if x is not None and not np.isnan(x)],float)
    if v.size<2: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); bs=[np.mean(rng.choice(v,v.size,replace=True)) for _ in range(B)]
    return float(v.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))

# ---- Exathlon loading -----------------------------------------------------
def load_trace(path, ts_col, max_features):
    """Return (timestamps[s], X[T,C]) for one trace CSV, or (None,None) if the
    file is empty/unreadable/degenerate (skipped with a warning)."""
    if pd is None: raise RuntimeError("pandas is required: pip install pandas")
    try:
        df=pd.read_csv(path)
    except Exception as ex:
        print("  [skip] %s: unreadable (%s)"%(os.path.basename(path),type(ex).__name__)); return None,None
    if df.shape[0]<10 or df.shape[1]<2:
        print("  [skip] %s: too small (%dx%d)"%(os.path.basename(path),df.shape[0],df.shape[1])); return None,None
    ts=df[ts_col].to_numpy(dtype=float) if ts_col in df.columns else np.arange(len(df),dtype=float)
    feat=df.select_dtypes(include=[np.number]).drop(columns=[c for c in [ts_col] if c in df.columns],errors="ignore")
    if feat.shape[1]==0:
        print("  [skip] %s: no numeric features"%os.path.basename(path)); return None,None
    feat=feat.ffill().bfill().fillna(0.0)
    return ts, feat.to_numpy(dtype=float)   # full numeric features (global selection done in run())

def ts_to_idx(ts, t): return int(np.searchsorted(ts, t))

def read_ground_truth(path, trace_col, onset_col, end_col, type_col=None):
    if pd is None: raise RuntimeError("pandas required")
    gt=pd.read_csv(path); d={}; typ={}
    for _,r in gt.iterrows():
        tn=str(r[trace_col])
        d.setdefault(tn,[]).append((float(r[onset_col]),float(r[end_col])))
        t=str(r[type_col]) if (type_col and type_col in gt.columns) else "anomaly"
        typ.setdefault(tn,[]).append(t)
    return d,typ

# ---- synthetic Exathlon-format fixture (for --selftest) -------------------
def make_fixture(root, C=20, seed=1):
    os.makedirs(root,exist_ok=True); rng=np.random.default_rng(seed)
    gt_rows=[]
    def write(name, X, ts):
        import csv as _csv
        with open(os.path.join(root,name+".csv"),"w",newline="") as f:
            w=_csv.writer(f); w.writerow(["t"]+["m%d"%i for i in range(C)])
            for k in range(len(ts)): w.writerow([int(ts[k])]+[round(v,4) for v in X[k]])
    base=rng.uniform(40,60,C)
    # 3 undisturbed (normal) traces
    for g in range(3):
        T=3000; ts=np.arange(T)+1_000_000  # epoch seconds, 1 Hz
        X=base+np.sin(2*np.pi*np.arange(T)[:,None]/600+rng.uniform(0,6,C))*rng.uniform(2,6,C)+rng.normal(0,1.0,(T,C))
        write("1_0_%d"%g, X, ts)
    # 3 disturbed (test) traces with a precursor ramp before root-cause onset
    for g in range(3):
        T=3000; ts=np.arange(T)+2_000_000
        X=base+np.sin(2*np.pi*np.arange(T)[:,None]/600+rng.uniform(0,6,C))*rng.uniform(2,6,C)+rng.normal(0,1.0,(T,C))
        name="1_1_%d"%g
        for k in range(2):                       # 2 anomalies per trace
            o=800+k*1200+int(rng.integers(0,200)); pl=int(rng.integers(90,180)); dur=int(rng.integers(60,120))
            ch=rng.choice(C,6,replace=False)
            X[o-pl:o,ch]+=np.linspace(0,8,pl)[:,None]**1.4   # precursor ramp
            X[o:o+dur,ch]+=22                                 # failure
            gt_rows.append([name, 1, int(ts[o]), int(ts[min(o+dur,T-1)])])  # root_cause_start, extended_effect_end
        write(name, X, ts)
    with open(os.path.join(root,"ground_truth.csv"),"w",newline="") as f:
        import csv as _csv; w=_csv.writer(f)
        w.writerow(["trace_name","anomaly_type","root_cause_start","extended_effect_end"]); w.writerows(gt_rows)
    return root, os.path.join(root,"ground_truth.csv")

# ---- driver ---------------------------------------------------------------
def run(data_root, gt_path, cols, budget, horizon_min, max_features, outdir, calib="clean-split", selftest=False):
    os.makedirs(outdir,exist_ok=True)
    gt,gttype=read_ground_truth(gt_path, cols["trace"], cols["onset"], cols["end"], cols.get("type"))
    disturbed=set(gt.keys())
    all_csv=[p for p in glob.glob(os.path.join(data_root,"*.csv"))
             if os.path.basename(p)!=os.path.basename(gt_path)]
    normal=[p for p in all_csv if os.path.splitext(os.path.basename(p))[0] not in disturbed]
    test  =[p for p in all_csv if os.path.splitext(os.path.basename(p))[0] in disturbed]
    if not normal or not test:
        print("error: need both undisturbed (normal) and disturbed (test) traces"); return
    # Memory-safe single pass: fix ONE global column set (top variance, from the
    # first readable normal trace -- Exathlon traces share a schema) and reduce +
    # row-subsample each trace as we load, so we never hold the full-width pool.
    ROWS_PER_TRACE=2000; rng0=np.random.default_rng(SEED)
    W=None; keep=None; parts=[]
    for p in normal:
        ts,X=load_trace(p,cols["ts"],0)
        if X is None: continue
        if W is None:
            W=X.shape[1]
            keep=(np.sort(np.argsort(X[:,:W].var(0))[-max_features:])
                  if (max_features and W>max_features) else np.arange(W))
        if X.shape[1]<W: continue
        Xr=X[:,:W][:,keep]
        if len(Xr)>ROWS_PER_TRACE:
            Xr=Xr[rng0.choice(len(Xr),ROWS_PER_TRACE,replace=False)]
        parts.append(Xr)
    if not parts:
        print("error: no readable normal traces"); return
    Xnorm=np.vstack(parts); del parts
    cut=int(len(Xnorm)*0.8)
    sc=StandardScaler().fit(Xnorm[:cut]); det=fit_detectors(sc.transform(Xnorm[:cut]),SEED)
    print("fit on %d normal rows x %d features (from %d normal traces)"%(len(Xnorm),len(keep),len(normal)))
    # estimate sampling step (minutes) from the first readable test trace
    step_min=1.0
    for p in test:
        ts_t,Xt=load_trace(p,cols["ts"],0)
        if ts_t is not None and len(ts_t)>1:
            step_min=float(np.median(np.diff(ts_t)))/60.0; break
    H=max(1,int(round(horizon_min/max(step_min,1e-9))))

    # ---- PASS 1: score every test trace once; store scores, episodes, and a
    # clean mask (indices outside every credit window [onset-H, end]). ----
    store=[]  # list of dicts per trace
    for p in test:
        name=os.path.splitext(os.path.basename(p))[0]; ts,X=load_trace(p,cols["ts"],0)
        if X is None or X.shape[1]<W: continue
        Xs=sc.transform(X[:,:W][:,keep])
        eps=[(ts_to_idx(ts,o),ts_to_idx(ts,e)) for (o,e) in gt.get(name,[])]
        eps=[(o,e) for (o,e) in eps if 0<=o<len(ts)]
        if not eps: continue
        s=score_with(det,Xs); T=len(Xs)
        cmask=np.ones(T,bool)
        for (o,e) in eps: cmask[max(0,o-H):min(T,e+1)]=False   # exclude credit windows
        store.append(dict(name=name,eps=eps,s=s,T=T,cmask=cmask))

    # ---- calibrate thresholds ----
    if calib=="test-normal":
        # Retrospective diagnostic: set threshold to hit the target episode rate
        # on the NORMAL regions of the test traces (standard target-rate-on-normal
        # calibration; uses test timing, not the anomaly signal). Disclosed as such.
        thr={}
        for d in DETS:
            normseg=np.concatenate([r["s"][d][r["cmask"]] for r in store if r["cmask"].any()])
            best=None
            for q in np.linspace(0.90,0.99999,120):
                t=np.quantile(normseg,q)
                # episode rate on pooled normal regions
                rate=1000.0*sum(len(alarm_onsets(r["s"][d][r["cmask"]],t)) for r in store)/max(len(normseg),1)
                dd=abs(rate-budget)
                if best is None or dd<best[0]: best=(dd,t)
            thr[d]=best[1]
    else:  # clean-split (deployable): calibrate on held-out normal rows
        scal=score_with(det, sc.transform(Xnorm[cut:]))
        thr={d:calibrate_episode(scal[d],0,len(scal[d]),budget)[0] for d in DETS}

    # ---- manifest ----
    with open(os.path.join(outdir,"exathlon_manifest.csv"),"w",newline="") as f:
        mw=csv.writer(f); mw.writerow(["trace","role","n_anomalies","types"])
        used={r["name"] for r in store}
        for r in store:
            mw.writerow([r["name"],"disturbed",len(r["eps"]),";".join(gttype.get(r["name"],[]))])
        for p in test:
            nm=os.path.splitext(os.path.basename(p))[0]
            if nm not in used: mw.writerow([nm,"disturbed-skipped",0,";".join(gttype.get(nm,[]))])
        for p in normal:
            mw.writerow([os.path.splitext(os.path.basename(p))[0],"undisturbed",0,""])
    print("calibration mode: %s (target %.2f episodes/1000 samples)"%(calib,budget))

    pool={d:{"cov":[],"ar":{L:[] for L in ACT_L_MIN},"dar_u":{L:[] for L in ACT_L_MIN},
             "dar_c":{L:[] for L in ACT_L_MIN},"lead":[],"apd":[],"occ":[]} for d in DETS}
    rows=[]; total_ep=0; rng=np.random.default_rng(SEED+7)
    # ---- PASS 2: metrics per trace at the calibrated threshold ----
    for r in store:
        name=r["name"]; eps=r["eps"]; s=r["s"]; M=len(eps); total_ep+=M
        for d in DETS:
            onT=alarm_onsets(s[d],thr[d],0,r["T"])
            leads=[(x*step_min if x is not None else None) for x in match_leads(onT,eps,H)]
            det_leads=[x for x in leads if x is not None]
            pool[d]["cov"].append(len(det_leads)/M)
            pool[d]["lead"].append(float(np.mean(det_leads)) if det_leads else np.nan)
            pool[d]["apd"].append(1000.0*len(onT)/max(len(s[d]),1))
            ARu=null_dAR(onT,eps,H,0,len(s[d]),step_min,rng,circ=False)
            ARc=null_dAR(onT,eps,H,0,len(s[d]),step_min,rng,circ=True)
            for L in ACT_L_MIN:
                ar=actionable_recall(leads,M,L)
                pool[d]["ar"][L].append(ar); pool[d]["dar_u"][L].append(ar-ARu[L]); pool[d]["dar_c"][L].append(ar-ARc[L])
            rows.append(dict(trace=name,detector=d,episodes=M,
                             coverage=round(pool[d]["cov"][-1],3),
                             lead_min=round(pool[d]["lead"][-1],2) if pool[d]["lead"][-1]==pool[d]["lead"][-1] else "",
                             dAR15_u=round(pool[d]["dar_u"][15][-1],3)))
        print("[ok] %s (%d anomalies)"%(name,M))
    with open(os.path.join(outdir,"exathlon_results.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L0=15
    tag="SELFTEST (synthetic)" if selftest else "Exathlon"
    print("\n=== Pooled across %d trace(s), %d anomalies | budget=%.1f/1k, H=%d min, step=%.3f min/sample ==="
          %(len(test),total_ep,budget,horizon_min,step_min))
    print(f"detector         cov%  AR{L0}%  dAR{L0}_uniform[95%CI]  dAR{L0}_circ[95%CI]  lead(min)  ach/1k")
    summ=[]
    for d in DETS:
        cov=np.nanmean(pool[d]["cov"])*100; ar=np.nanmean(pool[d]["ar"][L0])*100
        mu,lu,hu=boot_ci(pool[d]["dar_u"][L0]); mc,lc,hc=boot_ci(pool[d]["dar_c"][L0])
        lead=np.nanmean(pool[d]["lead"]); apd=np.nanmean(pool[d]["apd"])
        print("%-16s %4.0f  %4.0f   %+4.0f [%+.0f,%+.0f]   %+4.0f [%+.0f,%+.0f]   %6.1f  %5.2f"
              %(d,cov,ar,mu*100,lu*100,hu*100,mc*100,lc*100,hc*100,lead,apd))
        summ.append([d,round(cov,1),round(ar,1),round(mu*100,1),round(lu*100,1),round(hu*100,1),
                     round(mc*100,1),round(lc*100,1),round(hc*100,1),round(lead,2),round(apd,2)])
    with open(os.path.join(outdir,"exathlon_pooled.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["detector","cov_%","AR%d_%%"%L0,"dAR%du_%%"%L0,"dAR%du_lo"%L0,"dAR%du_hi"%L0,
            "dAR%dcirc_%%"%L0,"dAR%dcirc_lo"%L0,"dAR%dcirc_hi"%L0,"lead_min","achieved_per_1k"]); w.writerows(summ)
    # dAR(L) curve
    plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":.3})
    fig,ax=plt.subplots(figsize=(3.4,2.5),constrained_layout=True)
    for d in DETS: ax.plot(ACT_L_MIN,[np.nanmean(pool[d]["dar_u"][L])*100 for L in ACT_L_MIN],marker="o",ms=3,label=d)
    ax.axhline(0,color="k",lw=.8); ax.set_xlabel("required warning $L$ (min)")
    ax.set_ylabel(r"excess recall $\Delta$AR (pts)"); ax.legend(fontsize=6)
    ax.set_title("%s: excess warning vs. uniform null"%tag,fontsize=8)
    fig.savefig(os.path.join(outdir,"exathlon_dAR.png"),dpi=200); plt.close(fig)
    print("\nWrote:",os.path.join(outdir,"exathlon_results.csv"),", exathlon_pooled.csv, exathlon_dAR.png")
    if selftest: print("\n*** SELFTEST numbers are SYNTHETIC -- not Exathlon results, do not cite. ***")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default=None,help="folder of Exathlon trace CSVs")
    ap.add_argument("--ground-truth",default=None,help="ground_truth.csv path")
    ap.add_argument("--ts-col",default="t"); ap.add_argument("--gt-trace-col",default="trace_name")
    ap.add_argument("--gt-onset-col",default="root_cause_start")
    ap.add_argument("--gt-end-col",default="extended_effect_end")
    ap.add_argument("--gt-type-col",default="anomaly_type")
    ap.add_argument("--calib",choices=["clean-split","test-normal"],default="clean-split",
                    help="threshold calibration: deployable clean-split, or test-normal diagnostic")
    ap.add_argument("--budget",type=float,default=1.0,help="target alert episodes per 1000 samples")
    ap.add_argument("--horizon-min",type=float,default=30.0)
    ap.add_argument("--max-features",type=int,default=200,help="keep top-variance features (0=all)")
    ap.add_argument("--out",default="exathlon_out"); ap.add_argument("--selftest",action="store_true")
    a=ap.parse_args()
    cols=dict(ts=a.ts_col,trace=a.gt_trace_col,onset=a.gt_onset_col,end=a.gt_end_col,type=a.gt_type_col)
    if a.selftest:
        tmp=tempfile.mkdtemp(prefix="exathlon_fix_"); dr,gt=make_fixture(tmp)
        print("fixture in",tmp)
        run(dr,gt,cols,a.budget,a.horizon_min,a.max_features or 0,a.out,calib=a.calib,selftest=True); return
    if not a.data_root or not a.ground_truth:
        print("error: give --data-root and --ground-truth (or --selftest)"); return
    run(a.data_root,a.ground_truth,cols,a.budget,a.horizon_min,a.max_features or 0,a.out,calib=a.calib,selftest=False)

if __name__=="__main__": main()
