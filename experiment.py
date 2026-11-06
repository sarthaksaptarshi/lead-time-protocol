#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment.py  --  Single audited pipeline for the lead-time study (v3).

v3 answers the second-round review. In addition to the v2 fixes (causal
smoothing; disjoint train/cal/test; persistence+cooldown alarm episodes;
coverage + actionable recall; 15 seeds; horizon sweep; ablation) it adds:

  [R1] ONE precise null definition: the detector's own test-region alarm-
       episode COUNT, scattered over the test exposure [CAL_END,N) with a
       minimum spacing equal to the cooldown. Plus a structure-preserving
       CIRCULAR-SHIFT null for sensitivity. Matching is identical to the
       detector's.
  [R2] Control detectors -- random, onset-only, precursor-oracle -- to
       validate what the corrected metric means (no "refuses warning" claim).
  [R3] Coverage-aware correction  dAR(L) = AR_det(L) - mean AR_null(L),
       all failures in both denominators; null coverage reported; empty-draw
       handling defined (conditional-null excludes empty draws, reported).
  [R4] Multiple calibration TARGETS {3,6,12} alerts/day with ACHIEVED rates.
  [R6] Bootstrap 95% CIs across streams for paired contrasts; Monte-Carlo
       standard error for the null; stated Spearman test.
  [R7] Operational-vs-oracle table; episode-level output dump; NAB-style and
       range-recall parameters documented in code.

Outputs -> ./out_v2/  (kept name for continuity with the manuscript paths).
Deterministic: master seed below; per-stream seeds derived from it.
"""
import os, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

MASTER_SEED = 20270419
# Output dir sits next to this script, so it runs anywhere (not a fixed path).
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_v2")
os.makedirs(OUT, exist_ok=True)

STEP_MIN=2; DAYS=10; N=DAYS*24*60//STEP_MIN; C=6
TRAIN_END=int(2.0*24*60/STEP_MIN); CAL_END=int(3.0*24*60/STEP_MIN)
S_SEEDS=15
H_MIN=120; H=H_MIN//STEP_MIN
PERSIST=3; COOLDOWN=15
TARGETS=[3.0,6.0,12.0]; TARGET_MAIN=6.0
ACT_L=[0,15,30,60]
B_NULL=200
REAL=["IsolationForest","PCA-recon","AE-level","AE-combined"]
ABLA=["AE-level","AE-trend","AE-combined"]
CTRL=["Random","Onset-only","Precursor-oracle"]

# ---- shared seasonal model (used by the generator and the precursor-oracle) --
PER=24*60//STEP_MIN; PHASE=2*np.pi*(np.arange(N)%PER)/PER
LVL=np.array([45,55,120,20,30,1.5]); AMP=np.array([18,6,35,5,12,0.6])
PH =np.array([0.6,1.2,0.4,1.0,0.5,0.8]); NZ=np.array([2.5,1.5,6,1.5,3,0.25])
SEASON=np.stack([LVL[c]+AMP[c]*np.sin(PHASE-PH[c]) for c in range(C)],axis=1)  # (N,C)

# =========================================================================
# 1. stream generator (known onsets + precursor structure)
# =========================================================================
def gen_stream(seed):
    rng=np.random.default_rng(seed)
    X=SEASON+np.stack([rng.normal(0,NZ[c],N) for c in range(C)],axis=1)
    labels=np.zeros(N,int); episodes=[]
    PREC=[1,2,3,4]
    def gradual(o,pl,dur):
        p0=o-pl; ramp=np.linspace(0,1,pl)**1.7
        X[p0:o,1]+=(6+rng.uniform(0,4))*ramp; X[p0:o,3]+=(5+rng.uniform(0,3))*ramp
        X[p0:o,2]+=(20+rng.uniform(0,12))*ramp; X[p0:o,4]+=(9+rng.uniform(0,5))*ramp
        fw=np.linspace(1,1.5,dur)
        X[o:o+dur,1]+=30*fw; X[o:o+dur,2]+=230*fw; X[o:o+dur,3]+=34*fw
        X[o:o+dur,4]+=65*fw; X[o:o+dur,5]+=11*fw
        labels[o:o+dur]=1; episodes.append((o,o+dur-1,"gradual"))
    def sudden(o,dur):
        X[o:o+dur,2]+=220; X[o:o+dur,5]+=11; X[o:o+dur,0]+=30
        labels[o:o+dur]=1; episodes.append((o,o+dur-1,"sudden"))
    kinds=["gradual"]*6+["sudden"]*4; rng.shuffle(kinds); cur=CAL_END+H+60
    for k in kinds:
        cur+=int(rng.integers(140,240))
        if k=="gradual":
            pl=int(rng.integers(70,105)); dur=int(rng.integers(25,45))
            if cur-pl<CAL_END+H: cur=CAL_END+H+pl+5
            gradual(cur,pl,dur); cur+=dur
        else:
            dur=int(rng.integers(25,45)); sudden(cur,dur); cur+=dur
        if cur>N-H-80: break
    for c in (0,1): X[:,c]=np.clip(X[:,c],0,100)
    X[:,5]=np.clip(X[:,5],0,None)
    return X,labels,episodes,PREC

# =========================================================================
# 2. causal processing + detectors (incl. controls)
# =========================================================================
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

def scores_for_stream(X,labels,prec,seed):
    scaler=StandardScaler().fit(X[:TRAIN_END]); Xs=scaler.transform(X); Xtr=Xs[:TRAIN_END]
    rng=np.random.default_rng(seed+13)
    s={}
    s["IsolationForest"]=-IsolationForest(n_estimators=200,random_state=seed).fit(Xtr).score_samples(Xs)
    pca=PCA(n_components=3,random_state=seed).fit(Xtr); rp=pca.inverse_transform(pca.transform(Xs))
    s["PCA-recon"]=causal_ma(np.sqrt(((Xs-rp)**2).mean(1)),5)
    ae=MLPRegressor(hidden_layer_sizes=(12,4,12),activation="tanh",solver="adam",
                    max_iter=600,random_state=seed,learning_rate_init=3e-3).fit(Xtr,Xtr)
    err=np.sqrt(((Xs-ae.predict(Xs))**2).mean(1))
    lvl=causal_ma(err,5); trend=causal_ma(causal_ewma_pos_diff(err,12),5)
    s["AE-level"]=lvl; s["AE-trend"]=trend; s["AE-combined"]=trend+0.15*lvl
    # ---- control detectors (R2) ----
    # random: independent noise -> a random alarm process
    s["Random"]=rng.normal(0,1,N)
    # onset-only: high strictly inside the failure window (fires AT onset)
    s["Onset-only"]=labels.astype(float)+rng.normal(0,1e-3,N)
    # precursor-oracle: causal positive departure of the precursor channels
    # from the (known) seasonal baseline, in noise units -> rises during a ramp,
    # ~0 during normal seasonality. Uses generator internals by design (oracle).
    dev=np.clip((X[:,prec]-SEASON[:,prec])/NZ[prec],0,None).sum(1)
    s["Precursor-oracle"]=causal_ma(dev,5)
    return s

# =========================================================================
# 3. alarm episodes, matching, metrics, NULLS
# =========================================================================
def alarm_onsets(score,thr):
    on=[]; run=0; i=0
    while i<N:
        if score[i]>thr:
            run+=1
            if run==PERSIST: on.append(i); i+=COOLDOWN; run=0; continue
        else: run=0
        i+=1
    return np.array(on,int)

def match_leads(onsets,episodes,horizon):
    used=np.zeros(len(onsets),bool); leads=[]
    for (s,e,_k) in episodes:
        lo,hi=s-horizon,e; pick=None
        for j,o in enumerate(onsets):
            if used[j]: continue
            if lo<=o<=hi: pick=j; break
        if pick is None: leads.append(None)
        else: used[pick]=True; leads.append((s-onsets[pick])*STEP_MIN)
    return leads

def clean_mask(episodes,horizon):
    m=np.zeros(N,bool); m[CAL_END:]=True
    for (s,e,_k) in episodes: m[max(0,s-horizon):e+1]=False
    return m

def alarm_rate_cal(score,thr):
    cal=score[TRAIN_END:CAL_END]; days=len(cal)*STEP_MIN/1440.0
    # recompute onsets within the calibration slice only
    on=[]; run=0; i=0
    while i<len(cal):
        if cal[i]>thr:
            run+=1
            if run==PERSIST: on.append(i); i+=COOLDOWN; run=0; continue
        else: run=0
        i+=1
    return len(on)/days

def calibrate(score,target):
    """Attainable-budget search over the FULL quantile grid (R4): alarm-episode
    rate is NOT monotone in the threshold because lowering it merges runs, so
    we scan and take the threshold whose achieved cal rate is closest to target."""
    cand=np.quantile(score[TRAIN_END:CAL_END],np.linspace(0.80,0.9997,80))
    best=None
    for thr in cand:
        apd=alarm_rate_cal(score,thr); d=abs(apd-target)
        if best is None or d<best[0]: best=(d,thr,apd)
    return best[1],best[2]

def actionable_recall(leads,M,L):
    return sum(1 for x in leads if (x is not None and x>=L))/M

# ---- nulls (R1) ----
# Minimum alarm-onset gap enforced by the alarm process (see alarm_onsets):
# after an alarm the scanner skips COOLDOWN samples and then needs PERSIST more
# consecutive exceedances, so consecutive onsets are >= COOLDOWN+PERSIST-1 apart.
MIN_GAP = COOLDOWN + PERSIST - 1
def scatter_minspaced(n,lo,hi,d,rng):
    """n DISTINCT integer onsets in [lo,hi) with pairwise gap >= d, sampled
    uniformly over all such configurations (standard gap bijection): draw n
    distinct sorted positions w in [0, room), then x_i = lo + w_i + i*(d-1),
    where room = (hi-lo) - (n-1)*(d-1). Falls back to even spacing if n onsets
    cannot fit with gap d."""
    span=hi-lo
    if n<=0: return np.array([],int)
    room=span-(n-1)*(d-1)
    if room<n:                                   # infeasible -> even spacing
        step=max(1,span//max(n,1))
        return (lo+np.arange(n)*step).astype(int)
    w=np.sort(rng.choice(room,size=n,replace=False))
    return (lo+w+np.arange(n)*(d-1)).astype(int)

def null_distributions(n_test,episodes,horizon,rng,kind_sets,circ_src=None):
    """Return, for each requested kind in kind_sets, arrays over B draws of
    (conditional mean lead on NON-empty draws), coverage, and AR(L) vectors.
    Two null families: uniform min-spaced (default) and, if circ_src given,
    circular-shift of the detector's own test onsets (structure-preserving)."""
    lo,hi=CAL_END,N
    res={k:{"condmeans":[],"cov":[],"AR":{L:[] for L in ACT_L}} for k in kind_sets}
    for b in range(B_NULL):
        if circ_src is None:
            ro=scatter_minspaced(n_test,lo,hi,MIN_GAP,rng)
        else:
            off=int(rng.integers(0,max(1,hi-lo)))
            ro=np.sort(((np.asarray(circ_src)-lo+off)%(hi-lo))+lo)
        for k in kind_sets:
            eps=[ep for ep in episodes if (k is None or ep[2]==k)]
            M=len(eps)
            if M==0:
                res[k]["cov"].append(np.nan); continue
            ld=match_leads(ro,eps,horizon); det=[x for x in ld if x is not None]
            res[k]["cov"].append(len(det)/M)
            if det: res[k]["condmeans"].append(np.mean(det))  # empty draws excluded
            for L in ACT_L: res[k]["AR"][L].append(actionable_recall(ld,M,L))
    return res

def pointwise_f1(score,thr,labels):
    al=score>thr; tp=np.sum(al&(labels==1)); fp=np.sum(al&(labels==0)); fn=np.sum((~al)&(labels==1))
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    return (2*p*r/(p+r)) if (p+r) else 0.0

def nab_style(onsets,episodes,horizon,clean_alarm):
    """SIMPLIFIED NAB-style reward (NOT the official NAB implementation).
    Standard profile A_TP=1,A_FP=0.11,A_FN=1; sigmoid position reward for the
    first in-window alarm with relative position y=-lead/(H*step) in [-1,0];
    FP penalty per clean-region alarm; FN per miss; min-max normalised to [0,1]."""
    A_TP,A_FP,A_FN=1.0,0.11,1.0; leads=match_leads(onsets,episodes,horizon); raw=0.0
    for ld in leads:
        if ld is None: raw-=A_FN
        else:
            y=-ld/(horizon*STEP_MIN); raw+=A_TP*(2.0/(1.0+np.exp(5*y))-1.0)
    raw-=A_FP*clean_alarm
    perfect=A_TP*len(episodes); worst=-(A_FN*len(episodes)+A_FP*max(clean_alarm,1))
    return float((raw-worst)/(perfect-worst+1e-9))

def affiliation_style(onsets,episodes,T_lo,T_hi):
    """Affiliation-STYLE precision/recall (our reimplementation in the spirit of
    Huet et al. 2022; NOT the official package): directed distance to the
    nearest counterpart, normalized by the expected nearest distance under a
    uniform placement at the same count. Returns (P,R,F1) in [0,1]."""
    span=max(T_hi-T_lo,1)
    ons=np.asarray(sorted(onsets),float); ev=np.asarray([e[0] for e in episodes],float)
    if ons.size==0 or ev.size==0: return 0.0,0.0,0.0
    drand_r=span/(2*ons.size)                      # expected nearest-alarm dist (uniform)
    R=float(np.mean([max(0.0,1.0-np.min(np.abs(ons-s))/drand_r) for s in ev]))
    drand_p=span/(2*ev.size)
    P=float(np.mean([max(0.0,1.0-np.min(np.abs(ev-o))/drand_p) for o in ons]))
    F=2*P*R/(P+R) if (P+R)>0 else 0.0
    return P,R,F

def range_recall_front(onsets,episodes,horizon,alpha=0.5):
    """Range-based recall (Tatbul et al. 2018 style): existence + positional
    overlap with FRONT bias; alpha=0.5; cardinality handled by max. Unitless."""
    vals=[]
    for (s,e,_k) in episodes:
        lo,hi=s-horizon,e; L=hi-lo+1; hit=[o for o in onsets if lo<=o<=hi]
        if not hit: vals.append(0.0); continue
        pos=np.array([(hi-o+1)/L for o in hit]); vals.append(alpha*1.0+(1-alpha)*float(np.clip(pos.max(),0,1)))
    return float(np.mean(vals)) if vals else 0.0

# =========================================================================
# 4. per-stream evaluation
# =========================================================================
ALL_DET=["IsolationForest","PCA-recon","AE-level","AE-trend","AE-combined",
         "Random","Onset-only","Precursor-oracle"]

def eval_stream(seed,target=TARGET_MAIN,horizon=H,want_null=True,want_circ=False):
    X,labels,episodes,prec=gen_stream(seed); sc=scores_for_stream(X,labels,prec,seed)
    rng=np.random.default_rng(seed+777); cm=clean_mask(episodes,horizon)
    clean_days=cm.sum()*STEP_MIN/1440.0; M=len(episodes)
    out={}
    for name in ALL_DET:
        s=sc[name]; thr,apd_cal=calibrate(s,target); ons=alarm_onsets(s,thr)
        ons_test=ons[ons>=CAL_END]; n_test=len(ons_test)
        clean_alarm=int(np.sum([cm[o] for o in ons_test])) if n_test else 0
        leads=match_leads(ons,episodes,horizon)
        d={"thr":thr,"apd_cal":apd_cal,"n_test":n_test}
        # coverage, conditional lead, AR by type
        for kind in (None,"gradual","sudden"):
            idx=[i for i,ep in enumerate(episodes) if (kind is None or ep[2]==kind)]
            det=[leads[i] for i in idx if leads[i] is not None]; tag=kind or "all"
            d[f"cov_{tag}"]=len(det)/len(idx) if idx else np.nan
            d[f"lead_{tag}"]=float(np.mean(det)) if det else np.nan
            for L in ACT_L:
                d[f"AR_{tag}_{L}"]=actionable_recall([leads[i] for i in idx],len(idx),L) if idx else np.nan
        d["occ_test"]=float(np.mean(s[cm]>thr))
        d["apd_test"]=len([o for o in ons_test if cm[o]])/clean_days
        # oracle F1 threshold (uses test labels -> disclosed)
        d["f1_op"]=pointwise_f1(s,thr,labels)
        qs=np.quantile(s[CAL_END:],np.linspace(0.80,0.9997,80)); f1s=[pointwise_f1(s,q,labels) for q in qs]
        bi=int(np.argmax(f1s)); d["f1_best"]=f1s[bi]
        lb=[x for x in match_leads(alarm_onsets(s,qs[bi]),episodes,horizon) if x is not None]
        d["lead_best"]=float(np.mean(lb)) if lb else 0.0
        d["nab"]=nab_style(ons,episodes,horizon,clean_alarm); d["range_rec"]=range_recall_front(ons,episodes,horizon)
        d["affil"]=affiliation_style(ons[ons>=CAL_END],episodes,CAL_END,N)[2]
        # nulls + coverage-aware correction
        if want_null:
            nd=null_distributions(n_test,episodes,horizon,rng,[None,"gradual","sudden"])
            for kind in (None,"gradual","sudden"):
                tag=kind or "all"; cond=nd[kind]["condmeans"]
                nmean=float(np.mean(cond)) if cond else np.nan
                d[f"nullcond_{tag}"]=nmean
                d[f"nullcov_{tag}"]=float(np.nanmean(nd[kind]["cov"]))
                d[f"nullcond_frac_empty_{tag}"]=1.0-len(cond)/B_NULL
                d[f"beyond_{tag}"]=(d[f"lead_{tag}"]-nmean) if (not np.isnan(d[f"lead_{tag}"]) and not np.isnan(nmean)) else np.nan
                d[f"nullse_{tag}"]=(float(np.std(cond))/np.sqrt(max(len(cond),1))) if cond else np.nan
                for L in ACT_L:
                    draws=np.asarray(nd[kind]["AR"][L],float); arn=draws.mean()
                    d[f"dAR_{tag}_{L}"]=d[f"AR_{tag}_{L}"]-arn; d[f"ARnull_{tag}_{L}"]=arn
                    # Monte-Carlo s.e. of the null AR mean (R5: MC uncertainty on primary measure)
                    d[f"dARse_{tag}_{L}"]=float(draws.std()/np.sqrt(len(draws)))
            # structure-preserving circular-shift null (R5), real detectors, all L
            if want_circ and name in REAL:
                ndc=null_distributions(n_test,episodes,horizon,rng,[None],circ_src=ons_test)
                for L in ACT_L:
                    d[f"dAR_all_{L}_circ"]=d[f"AR_all_{L}"]-float(np.mean(ndc[None]["AR"][L]))
        out[name]=d
    return out,M

# =========================================================================
# 5. run seeds, aggregate, bootstrap CIs
# =========================================================================
def boot_ci(vals,B=4000,seed=1):
    v=np.array([x for x in vals if x is not None and not np.isnan(x)],float)
    if v.size<2: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); bs=[np.mean(rng.choice(v,v.size,replace=True)) for _ in range(B)]
    return float(v.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))

print("Running %d seeds (main target=%g alerts/day) ..."%(S_SEEDS,TARGET_MAIN))
seeds=[MASTER_SEED+i for i in range(S_SEEDS)]
per_seed=[]
for i,sd in enumerate(seeds):
    rec,M=eval_stream(sd,want_null=True,want_circ=True); per_seed.append(rec)
acc={d:{} for d in ALL_DET}
for rec in per_seed:
    for d in ALL_DET:
        for k,v in rec[d].items(): acc[d].setdefault(k,[]).append(v)
def col(d,k): return acc[d].get(k,[])
def ms(d,k):
    a=np.array([x for x in col(d,k) if x is not None and not np.isnan(x)],float)
    return (np.nan,np.nan) if a.size==0 else (a.mean(),a.std())

# =========================================================================
# 6. tables
# =========================================================================
def wtab(fn,header,rows):
    with open(os.path.join(OUT,fn),"w",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

# Table I: real detectors, coverage-aware (dAR with CI), at target 6
t1=[]
for d in REAL:
    m,lo,hi=boot_ci(col(d,"dAR_all_30"))
    bym,bylo,byhi=boot_ci(col(d,"beyond_all"))
    t1.append([d,f"{ms(d,'cov_all')[0]*100:.0f}",f"{ms(d,'AR_all_30')[0]*100:.0f}",
               f"{m*100:+.0f} [{lo*100:+.0f},{hi*100:+.0f}]",
               f"{ms(d,'lead_all')[0]:.0f}",
               f"{bym:+.0f} [{bylo:+.0f},{byhi:+.0f}]"])
wtab("table1_main.csv",["detector","cov_%","AR30_%","dAR30_%_95CI","lead|det_min","beyond_chance_min_95CI"],t1)

# Table Ib: null VALIDATION on GRADUAL faults (where a precursor exists).
# Expected: precursor-oracle >> 0, real detector > 0, random ~ 0, onset-only < 0.
t1b=[]
for d in ["Precursor-oracle","AE-level","Random","Onset-only"]:
    m,lo,hi=boot_ci(col(d,"dAR_gradual_30"))
    t1b.append([d,f"{ms(d,'cov_gradual')[0]*100:.0f}",
                f"{m*100:+.0f} [{lo*100:+.0f},{hi*100:+.0f}]",
                f"{ms(d,'beyond_gradual')[0]:+.0f}"])
wtab("table1b_validation.csv",["detector","cov_gradual_%","dAR30_gradual_%_95CI","beyond_gradual_min"],t1b)

# Table II: multi-target achieved rates (R4)
t2=[]
mt={}
for tg in TARGETS:
    recs=[eval_stream(sd,target=tg,want_null=False)[0] for sd in seeds]
    for d in REAL:
        a=np.array([r[d]["apd_test"] for r in recs]); o=np.array([r[d]["occ_test"] for r in recs])
        mt.setdefault(d,{})[tg]=(a.mean(),a.std(),o.mean()*100)
for d in REAL:
    row=[d]+[f"{mt[d][tg][0]:.1f}±{mt[d][tg][1]:.1f}" for tg in TARGETS]+[f"{mt[d][TARGET_MAIN][2]:.1f}"]
    t2.append(row)
wtab("table2_opreal.csv",["detector"]+[f"achieved@target{int(t)}" for t in TARGETS]+["occ@6_%"],t2)

# --- Comparable achieved-budget comparison (R5) ---
# Re-evaluate at a lower target (3/day) where achieved rates are closer, and
# report dAR(30) there, so warning quality is compared near a common achieved
# paging frequency rather than only a common calibration target.
TG_LOW=3.0
recs_low=[eval_stream(sd,target=TG_LOW,want_null=True)[0] for sd in seeds]
print("\n== Comparable-achieved-budget check (target %.0f/day) =="%TG_LOW)
print("detector\tachieved/day\tdAR30_%%")
cab=[]
for d in REAL:
    ach=np.nanmean([r[d]["apd_test"] for r in recs_low])
    dar=np.nanmean([r[d]["dAR_all_30"] for r in recs_low])*100
    print("%-16s %.1f\t%+.0f"%(d,ach,dar)); cab.append((d,ach,dar))
wtab("table_cab.csv",["detector","achieved_per_day","dAR30_pct"],
     [[d,f"{a:.1f}",f"{x:+.0f}"] for d,a,x in cab])

# Table III: ablation with dAR
t3=[]
for d in ABLA:
    t3.append([d,f"{ms(d,'AR_all_30')[0]*100:.0f}",
               f"{ms(d,'dAR_all_30')[0]*100:+.0f}",
               f"{ms(d,'beyond_gradual')[0]:+.0f}",f"{ms(d,'beyond_sudden')[0]:+.0f}"])
wtab("table3_ablation.csv",["variant","AR30_%","dAR30_%","beyond_gradual_min","beyond_sudden_min"],t3)

# Table IV: operational vs oracle (R7)
t4=[]
for d in REAL:
    t4.append([d,f"{ms(d,'f1_op')[0]:.2f}",f"{ms(d,'lead_all')[0]:.0f}",
               f"{ms(d,'cov_all')[0]*100:.0f}",f"{ms(d,'apd_test')[0]:.1f}",
               f"{ms(d,'f1_best')[0]:.2f}",f"{ms(d,'lead_best')[0]:+.0f}"])
wtab("table4_opvsoracle.csv",["detector","F1_op","lead_op_min","cov_op_%","alerts/day_op",
                              "F1_oracle","lead_oracle_min"],t4)

# Table V: metric comparison + decision (R7)
t5=[]
for d in REAL:
    t5.append([d,f"{ms(d,'nab')[0]:.2f}",f"{ms(d,'range_rec')[0]:.2f}",
               f"{ms(d,'dAR_all_30')[0]*100:+.0f}",f"{ms(d,'beyond_all')[0]:+.0f}"])
wtab("table5_compare.csv",["detector","NABstyle","range_rec","dAR30_%","beyond_min"],t5)

# Spearman at operational point, per stream across the 4 REAL detectors
rhos=[]
for rec in per_seed:
    f1=[rec[d]["f1_op"] for d in REAL]; ld=[rec[d]["lead_all"] for d in REAL]
    if np.all(~np.isnan(ld)): rhos.append(spearmanr(f1,ld).correlation)
rho_m,rho_lo,rho_hi=boot_ci(rhos)

# --- Null self-calibration (R6): a uniform d-spaced "detector" evaluated
# against fresh null draws should give dAR ~ 0 (the metric is unbiased under
# its EXACT null). This is distinct from the persistence-gated RANDOM control,
# whose onset distribution differs from uniform d-spaced. ---
def null_selfcal():
    vals=[]
    for sd in seeds:
        X,lab,eps,prec=gen_stream(sd); rng=np.random.default_rng(sd+321)
        # pick a representative alarm count (AE-level's test count on this stream)
        sc=scores_for_stream(X,lab,prec,sd)["AE-level"]; thr,_=calibrate(sc,TARGET_MAIN)
        n=int(np.sum(alarm_onsets(sc,thr)>=CAL_END))
        pseudo=scatter_minspaced(n,CAL_END,N,MIN_GAP,rng)  # uniform d-spaced "detector"
        arp=actionable_recall(match_leads(pseudo,eps,H),len(eps),30)
        nd=null_distributions(n,eps,H,rng,[None]); arn=np.mean(nd[None]["AR"][30])
        vals.append((arp-arn)*100)
    return boot_ci(vals)
sc_m,sc_lo,sc_hi=null_selfcal()
print("\nNull self-calibration (uniform d-spaced detector) dAR30: %.1f [%.1f,%.1f] pts (expect ~0)"
      %(sc_m,sc_lo,sc_hi))
print("Affiliation-style F1 by detector:", {d:round(ms(d,'affil')[0],3) for d in REAL})

# circular-shift sensitivity (dAR under both nulls, per detector, with CI) [R5]
circ_rows=[]
for d in REAL:
    um,ulo,uhi=boot_ci(col(d,"dAR_all_30"))
    cm_,clo,chi=boot_ci(col(d,"dAR_all_30_circ"))
    se=ms(d,"dARse_all_30")[0]
    circ_rows.append((d,um*100,ulo*100,uhi*100,cm_*100,clo*100,chi*100,se*100))

# paired AE-level - IsolationForest contrasts [R4]
def paired_ci(da,db,B=4000,seed=2):
    a=np.array(col(da,"dAR_all_30"),float); b=np.array(col(db,"dAR_all_30"),float)
    la=np.array(col(da,"beyond_all"),float); lb=np.array(col(db,"beyond_all"),float)
    rng=np.random.default_rng(seed); n=len(a)
    def ci(x):
        bs=[np.mean(rng.choice(x,n,replace=True)) for _ in range(B)]
        return float(x.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
    return ci(a-b),ci(la-lb)
pc_dAR,pc_beyond=paired_ci("AE-level","IsolationForest")

print("\n== Table I (target 6/day, 15 streams; dAR & beyond-chance with 95%% CI) ==")
for r in t1: print("\t".join(r))
print("\n== Table IV operational vs oracle ==")
for r in t4: print("\t".join(r))
print("\nSpearman(F1_op,lead) across 4 detectors, mean over streams: %.2f [%.2f,%.2f]"%(rho_m,rho_lo,rho_hi))
print("\nPaired AE-level - IsolationForest (95%% CI over streams):")
print("  dAR30: %+.0f [%+.0f,%+.0f] pts    beyond-chance lead: %+.1f [%+.1f,%+.1f] min"%(
    pc_dAR[0]*100,pc_dAR[1]*100,pc_dAR[2]*100,pc_beyond[0],pc_beyond[1],pc_beyond[2]))
print("\nNull sensitivity dAR30 (uniform [CI] vs circular [CI]; MC s.e.):")
for d,um,ul,uh,cmv,cl,ch,se in circ_rows:
    print("  %-16s uniform %+.0f [%+.0f,%+.0f]  circ %+.0f [%+.0f,%+.0f]  MCse %.1f"%(d,um,ul,uh,cmv,cl,ch,se))
print("\nAR(L) %% by detector (L=0/15/30/60):")
for d in REAL:
    print("  %-16s %s"%(d,"  ".join("%.0f"%(ms(d,f"AR_all_{L}")[0]*100) for L in ACT_L)))
print("\nNull coverage by detector x fault type (all/gradual/sudden):")
for d in REAL+["Onset-only","Random","Precursor-oracle"]:
    print("  %-16s %.2f / %.2f / %.2f"%(d,ms(d,"nullcov_all")[0],ms(d,"nullcov_gradual")[0],ms(d,"nullcov_sudden")[0]))
print("MC s.e. of dAR30 (AE-level): %.1f pts ; null empty-draw frac: %.3f"%(ms("AE-level","dARse_all_30")[0]*100,ms("AE-level","nullcond_frac_empty_all")[0]))

# episode-level dump (R7)
with open(os.path.join(OUT,"episode_level.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["stream","detector","episode_idx","type","onset_idx","lead_min"])
    for si,sd in enumerate(seeds):
        X,labels,episodes,prec=gen_stream(sd); sc=scores_for_stream(X,labels,prec,sd)
        for d in REAL:
            s=sc[d]; thr,_=calibrate(s,TARGET_MAIN); ld=match_leads(alarm_onsets(s,thr),episodes,H)
            for ei,((o,e,k),l) in enumerate(zip(episodes,ld)):
                w.writerow([sd,d,ei,k,o,"" if l is None else f"{l}"])

# =========================================================================
# 7. horizon sweep (lead only, real detectors)
# =========================================================================
H_grid=[40,80,120,160,200]; sweep={d:[] for d in REAL}
for Hm in H_grid:
    Hs=Hm//STEP_MIN; tmp={d:[] for d in REAL}
    for sd in seeds:
        r,_=eval_stream(sd,horizon=Hs,want_null=False)
        for d in REAL: tmp[d].append(r[d]["lead_all"])
    for d in REAL: sweep[d].append(np.nanmean(tmp[d]))
print("\nHorizon sweep: mean lead|det (min) at H =",H_grid)
for d in REAL: print("  %-16s %s"%(d,"  ".join("%.0f"%v for v in sweep[d])))

# =========================================================================
# 8. figures
# =========================================================================
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,
    "grid.alpha":0.3,"grid.linewidth":0.5,"axes.linewidth":0.7,"lines.linewidth":1.3,"figure.dpi":200})
STY={"IsolationForest":dict(color="#4c4c4c",ls="--"),"PCA-recon":dict(color="#7a7a7a",ls=":"),
     "AE-level":dict(color="#1f5fa8",ls="-"),"AE-combined":dict(color="#b5261e",ls="-")}

# Fig 1: causal concept (unchanged intent; caption fixed in manuscript)
X0,lab0,eps0,prec0=gen_stream(MASTER_SEED); sc0=scores_for_stream(X0,lab0,prec0,MASTER_SEED)
grad=[e for e in eps0 if e[2]=="gradual"][0]; s0,e0,_=grad; err=sc0["AE-level"]
thr_op,_=calibrate(err,TARGET_MAIN); qs=np.quantile(err[CAL_END:],np.linspace(0.80,0.9997,80))
f1s=[pointwise_f1(err,q,lab0) for q in qs]; thr_hi=qs[int(np.argmax(f1s))]
lo,hi=s0-H-40,e0+40; mins=np.arange(N)*STEP_MIN
fig,ax=plt.subplots(figsize=(3.4,2.5),constrained_layout=True)
ax.axvspan(mins[s0],mins[e0],color="0.80",label="failure window")
ax.axvspan(mins[max(lo,CAL_END)],mins[s0],color="0.92",label="pre-onset (incl. ramp)")
ax.plot(mins[lo:hi],err[lo:hi],color="#1f5fa8",label="AE error (causal)")
ax.axhline(thr_op,color="#b5261e",ls="--",lw=1.0,label="operational thr.")
ax.axhline(thr_hi,color="black",ls=":",lw=1.0,label="$F_1$-optimal (oracle)")
for thr,cl in [(thr_op,"#b5261e"),(thr_hi,"black")]:
    o=alarm_onsets(err,thr); o=o[(o>=s0-H)&(o<=e0)]
    if len(o):
        ld=(s0-o[0])*STEP_MIN; ax.axvline(mins[o[0]],color=cl,lw=0.9)
        ax.annotate("lead %+d min"%ld,(mins[o[0]],ax.get_ylim()[1]*0.5),fontsize=6,color=cl,rotation=90,va="center",ha="right")
ax.set_xlabel("time (min)"); ax.set_ylabel("reconstruction error")
ax.legend(fontsize=5.4,loc="upper left",framealpha=0.9); ax.set_title("Causal detector, two thresholds",fontsize=8)
fig.savefig(os.path.join(OUT,"fig1_concept.png")); plt.close(fig)

# Fig 2: coverage-aware excess actionable recall dAR(L) on GRADUAL faults,
# real detectors + the three control detectors (null validation).
fig,ax=plt.subplots(figsize=(3.4,2.6),constrained_layout=True)
for d in REAL:
    ys=[ms(d,f"dAR_gradual_{L}")[0]*100 for L in ACT_L]; ax.plot(ACT_L,ys,label=d,**STY[d])
for d,c,mk in [("Precursor-oracle","#6a3d9a","o"),("Random","#2a9d8f","^"),("Onset-only","#8a5a00","v")]:
    ys=[ms(d,f"dAR_gradual_{L}")[0]*100 for L in ACT_L]
    ax.plot(ACT_L,ys,color=c,ls="-.",lw=1.0,marker=mk,ms=3,label=d)
ax.axhline(0,color="black",lw=0.9)
ax.set_xlabel("required warning $L$ (min)"); ax.set_ylabel(r"excess recall $\Delta$AR$(L)$ (pts)")
ax.legend(fontsize=5.0,loc="upper right",ncol=1,framealpha=0.9)
ax.set_title("Excess warning vs. null (gradual faults)",fontsize=8)
fig.savefig(os.path.join(OUT,"fig2_beyond_chance.png")); plt.close(fig)

# Fig 3: raw actionable recall
fig,ax=plt.subplots(figsize=(3.4,2.5),constrained_layout=True)
for d in REAL:
    ys=[ms(d,f"AR_all_{L}")[0]*100 for L in ACT_L]; es=[ms(d,f"AR_all_{L}")[1]*100 for L in ACT_L]
    ax.errorbar(ACT_L,ys,yerr=es,label=d,capsize=2,**STY[d])
ax.set_xlabel("required warning $L$ (min before onset)"); ax.set_ylabel("actionable recall (%)")
ax.legend(fontsize=5.8,loc="upper right",framealpha=0.9); ax.set_title("Coverage-aware early warning",fontsize=8)
fig.savefig(os.path.join(OUT,"fig3_actionable_recall.png")); plt.close(fig)

# Fig 4: horizon sweep
fig,ax=plt.subplots(figsize=(3.4,2.5),constrained_layout=True)
for d in REAL: ax.plot(H_grid,sweep[d],label=d,**STY[d])
ax.set_xlabel("warning horizon $H$ (min)"); ax.set_ylabel("mean lead | detected (min)")
ax.legend(fontsize=5.8,loc="upper left",framealpha=0.9); ax.set_title("Sensitivity to the horizon",fontsize=8)
fig.savefig(os.path.join(OUT,"fig4_horizon.png")); plt.close(fig)

# summary
with open(os.path.join(OUT,"summary.txt"),"w") as f:
    f.write("rho=%.3f CI[%.2f,%.2f]\n"%(rho_m,rho_lo,rho_hi))
    for d in ALL_DET:
        f.write("%s cov=%.2f AR30=%.2f dAR30=%.3f lead=%.1f beyond_all=%.1f beyond_grad=%.1f "
                "beyond_sud=%.1f nullcov=%.2f occ=%.3f apd=%.2f f1op=%.3f f1best=%.3f leadbest=%.1f "
                "nab=%.3f range=%.3f\n"%(d,ms(d,'cov_all')[0],ms(d,'AR_all_30')[0],ms(d,'dAR_all_30')[0],
                ms(d,'lead_all')[0],ms(d,'beyond_all')[0],ms(d,'beyond_gradual')[0],ms(d,'beyond_sudden')[0],
                ms(d,'nullcov_all')[0],ms(d,'occ_test')[0],ms(d,'apd_test')[0],ms(d,'f1_op')[0],
                ms(d,'f1_best')[0],ms(d,'lead_best')[0],ms(d,'nab')[0],ms(d,'range_rec')[0]))
print("\nAll outputs written to",OUT)
