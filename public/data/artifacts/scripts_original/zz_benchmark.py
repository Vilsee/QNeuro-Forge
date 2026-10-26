import numpy as np, pandas as pd, itertools, time, json, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
n=4
H1=np.array([[1,1],[1,-1]])/np.sqrt(2); Hn=H1
for _ in range(n-1): Hn=np.kron(Hn,H1)
bits=np.array(list(itertools.product([0,1],repeat=n))); pairs=list(itertools.combinations(range(n),2))
def states(X,reps=1):
    out=[]
    for x in X:
        ph=(bits*2*x).sum(1)+sum((bits[:,i]^bits[:,j])*2*(np.pi-x[i])*(np.pi-x[j]) for i,j in pairs)
        s=np.zeros(2**n,complex); s[0]=1
        for _ in range(reps): s=np.exp(1j*ph)*(Hn@s)
        out.append(s)
    return np.array(out)
K=lambda A,B:np.abs(A.conj()@B.T)**2
tr=pd.read_csv('hf_withtime_train.csv'); te=pd.read_csv('hf_withtime_test.csv')
Xtr=tr.iloc[:,:-1].values; ytr=tr.death_event.values
Xte=np.clip(te.iloc[:,:-1].values,0,1); yte=te.death_event.values
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(Xtr,ytr))
for scale in [1.0,0.3]:
    S=states(Xtr*scale); Kt=K(S,S)
    m=[matthews_corrcoef(ytr[b],SVC(kernel='precomputed').fit(Kt[np.ix_(a,a)],ytr[a]).predict(Kt[np.ix_(b,a)])) for a,b in folds]
    St=states(Xte*scale); p=SVC(kernel='precomputed').fit(Kt,ytr).predict(K(St,S))
    print(scale,'CV MCC',round(np.mean(m),4),'holdout MCC',round(matthews_corrcoef(yte,p),4),'acc',accuracy_score(yte,p))

# ---------- build full submission ----------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
SCALE=0.3
S=states(Xtr*SCALE); Kt=K(S,S); St=states(Xte*SCALE); Ke=K(St,S)
def met(y,p,s):
    return dict(mcc=matthews_corrcoef(y,p),accuracy=accuracy_score(y,p),f1=f1_score(y,p),precision=precision_score(y,p),recall=recall_score(y,p),roc_auc=roc_auc_score(y,s))
def ci(v):
    v=np.array(v); h=stats.t.ppf(0.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)); return [round(v.mean()-h,4),round(v.mean()+h,4)]
def evalmodel(make,quantum=False):
    rows=[];tt=[]
    for a,b in folds:
        t=time.time()
        if quantum:
            c=SVC(kernel='precomputed').fit(Kt[np.ix_(a,a)],ytr[a]); Xb=Kt[np.ix_(b,a)]
        else:
            c=make().fit(Xtr[a],ytr[a]); Xb=Xtr[b]
        tt.append(time.time()-t)
        s=c.decision_function(Xb) if quantum or not hasattr(c,'predict_proba') else c.predict_proba(Xb)[:,1]
        rows.append(met(ytr[b],c.predict(Xb),s))
    t=time.time()
    if quantum: c=SVC(kernel='precomputed').fit(Kt,ytr); Xh=Ke
    else: c=make().fit(Xtr,ytr); Xh=Xte
    ht=time.time()-t
    s=c.decision_function(Xh) if quantum or not hasattr(c,'predict_proba') else c.predict_proba(Xh)[:,1]
    hold=met(yte,c.predict(Xh),s)
    g=lambda k:[r[k] for r in rows]
    cv=dict(mcc_mean=round(float(np.mean(g('mcc'))),4),mcc_ci_95=ci(g('mcc')),mcc_folds=[round(x,4) for x in g('mcc')],
            accuracy_mean=round(float(np.mean(g('accuracy'))),4),accuracy_folds=[round(x,4) for x in g('accuracy')],
            f1_mean=round(float(np.mean(g('f1'))),4),roc_auc_mean=round(float(np.mean(g('roc_auc'))),4),mean_train_time_sec=round(float(np.mean(tt)),4))
    return dict(cv_5fold=cv,holdout_test={k:float(v) for k,v in hold.items()}|{'train_time_seconds':ht}),g('mcc')
models={
 'LogisticRegression':lambda:LogisticRegression(max_iter=1000),
 'RandomForest':lambda:RandomForestClassifier(random_state=42),
 'GradientBoosting':lambda:GradientBoostingClassifier(random_state=42),
 'SVM-RBF':lambda:SVC(kernel='rbf'),
 'XGBoost':lambda:XGBClassifier(random_state=42,eval_metric='logloss')}
perf={}; mccs={}
perf['QSVM (Quantum Kernel)'],mccs['q']=evalmodel(None,True)
for k,f in models.items(): perf[k],mccs[k]=evalmodel(f)
d=np.array(mccs['q'])-np.array(mccs['RandomForest'])
tstat,pv=stats.ttest_rel(mccs['q'],mccs['RandomForest'])
h=stats.t.ppf(0.975,4)*d.std(ddof=1)/np.sqrt(5)
q=perf['QSVM (Quantum Kernel)']; rf=perf['RandomForest']
out={
 "metadata":{"track":"M1 - Heart Failure Clinical Records Survival Prediction","task_type":"Binary Clinical Classification","ranking_metric":"MCC (Matthews Correlation Coefficient)",
  "timestamp":pd.Timestamp.now(tz='UTC').isoformat(),"n_train_samples":len(Xtr),"n_test_samples":len(Xte),"n_features":4,
  "features":list(tr.columns[:-1]),"target":"death_event",
  "quantum_configuration":{"algorithm":"QSVM","n_qubits":4,"feature_map":"ZZFeatureMap (full entanglement, fidelity kernel)","depth_repetitions":1,
   "input_scaling":"MinMax to [0,1], then multiplied by 0.3 (chosen by 5-fold CV MCC over 1.0 vs 0.3)","hardware_platform":"Local exact NumPy statevector simulation (not run on Bloq cluster)"},
  "leakage_audit_note":"This run includes follow-up time, which the track permits. Follow-up time can carry outcome information and would not be available at true prediction time, so reported MCC may be higher than for a deployable model. No no-time comparison is included in this file."},
 "performance_summary":perf,
 "statistical_comparison":{"quantum_model":"QSVM (Quantum Kernel)","benchmark_classical_model":"RandomForest",
  "cv_mcc_quantum_mean":q['cv_5fold']['mcc_mean'],"cv_mcc_classical_mean":rf['cv_5fold']['mcc_mean'],
  "delta_mcc":round(float(d.mean()),4),"delta_ci_95":[round(float(d.mean()-h),4),round(float(d.mean()+h),4)],
  "paired_t_statistic":round(float(tstat),4),"p_value":round(float(pv),4),
  "verdict":"Computed on identical 5-fold splits. See delta_mcc and p_value for the quantum-vs-RandomForest comparison."}}
json.dump(out,open('/mnt/user-data/outputs/m1_heart_failure_submission_zz.json','w'),indent=2)
for k,v in perf.items(): print(k,'CV',v['cv_5fold']['mcc_mean'],'hold',round(v['holdout_test']['mcc'],3))
print(out['statistical_comparison'])
