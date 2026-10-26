import numpy as np, pandas as pd, itertools, warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import QuantileTransformer
n=4
H1=np.array([[1,1],[1,-1]])/np.sqrt(2); Hn=H1
for _ in range(n-1): Hn=np.kron(Hn,H1)
bits=np.array(list(itertools.product([0,1],repeat=n)))
pairs=list(itertools.combinations(range(n),2))
def states(X,reps=1):
    out=[]
    for x in X:
        ph=(bits*2*x).sum(1)+sum((bits[:,i]^bits[:,j])*2*(np.pi-x[i])*(np.pi-x[j]) for i,j in pairs)
        s=np.zeros(2**n,complex); s[0]=1
        for _ in range(reps): s=np.exp(1j*ph)*(Hn@s)
        out.append(s)
    return np.array(out)
K=lambda A,B:np.abs(A.conj()@B.T)**2
def prep(name,mode,scale):
    tr=pd.read_csv(f'{name}_train.csv'); te=pd.read_csv(f'{name}_test.csv')
    Xtr=tr.iloc[:,:-1].values; Xte=te.iloc[:,:-1].values
    if mode=='quantile':
        q=QuantileTransformer(n_quantiles=100,output_distribution='uniform').fit(Xtr)
        Xtr,Xte=q.transform(Xtr),q.transform(Xte)
    elif mode=='clip':
        lo,hi=np.percentile(Xtr,5,0),np.percentile(Xtr,95,0)
        Xtr=np.clip(Xtr,lo,hi); Xte=np.clip(Xte,lo,hi)
        Xtr=(Xtr-lo)/(hi-lo); Xte=(Xte-lo)/(hi-lo)
    else:
        Xtr,Xte=Xtr,np.clip(Xte,0,1)
    return Xtr*scale,Xte*scale,tr.death_event.values,te.death_event.values
def ev(name,mode,scale,reps=1,C=1,cw=None):
    Xtr,Xte,ytr,yte=prep(name,mode,scale)
    S,St=states(Xtr,reps),states(Xte,reps); Kt,Ke=K(S,S),K(St,S)
    ms=[]
    for a,b in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=1).split(Kt,ytr):
        p=SVC(kernel='precomputed',C=C,class_weight=cw).fit(Kt[np.ix_(a,a)],ytr[a]).predict(Kt[np.ix_(b,a)])
        ms.append(matthews_corrcoef(ytr[b],p))
    pt=SVC(kernel='precomputed',C=C,class_weight=cw).fit(Kt,ytr).predict(Ke)
    return np.mean(ms),matthews_corrcoef(yte,pt)
res=[]
for name in ['hf_notime','hf_withtime']:
    for mode in ['minmax','quantile','clip']:
        for scale in [0.3,0.5,0.75,1.0,1.5]:
            cv,t=ev(name,mode,scale); res.append((name,mode,scale,cv,t))
df=pd.DataFrame(res,columns=['data','mode','scale','cvMCC','testMCC'])
print(df.round(3).to_string())
