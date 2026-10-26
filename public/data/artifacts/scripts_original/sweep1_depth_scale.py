import numpy as np, pandas as pd, itertools
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict
n=4
H1=np.array([[1,1],[1,-1]])/np.sqrt(2)
Hn=H1
for _ in range(n-1): Hn=np.kron(Hn,H1)
bits=np.array(list(itertools.product([0,1],repeat=n)))
def states(X,reps):
    out=[]
    for x in X:
        ph=np.zeros(2**n)
        ph+=(bits*2*x).sum(1)
        for i,j in itertools.combinations(range(n),2):
            ph+= (bits[:,i]^bits[:,j])*2*(np.pi-x[i])*(np.pi-x[j])
        D=np.exp(1j*ph)
        s=np.zeros(2**n,complex); s[0]=1
        for _ in range(reps): s=Hn@s; s=D*s
        out.append(s)
    return np.array(out)
def K(A,B): return np.abs(A.conj()@B.T)**2
def run(name,scale,reps,cw,C=1):
    tr=pd.read_csv(f'{name}_train.csv'); te=pd.read_csv(f'{name}_test.csv')
    Xtr=tr.iloc[:,:-1].values*scale; ytr=tr.death_event.values
    Xte=np.clip(te.iloc[:,:-1].values,0,1)*scale; yte=te.death_event.values
    S=states(Xtr,reps); St=states(Xte,reps)
    Ktr=K(S,S); Kte=K(St,S)
    clf=lambda: SVC(kernel='precomputed',C=C,class_weight=cw)
    cvp=cross_val_predict(clf(),Ktr,ytr,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict') if False else None
    # manual CV on precomputed
    p=np.zeros_like(ytr)
    for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(Ktr,ytr):
        p[b]=clf().fit(Ktr[np.ix_(a,a)],ytr[a]).predict(Ktr[np.ix_(b,a)])
    m=clf().fit(Ktr,ytr); pt=m.predict(Kte)
    return matthews_corrcoef(ytr,p),matthews_corrcoef(yte,pt),pt.sum()
for name in ['hf_notime','hf_withtime']:
    print(name)
    for scale in [1,np.pi,2*np.pi]:
        for reps in [1,2]:
            for cw in [None,'balanced']:
                cv,t,np_=run(name,scale,reps,cw)
                print(f' scale={scale:.2f} reps={reps} cw={cw}: CV MCC={cv:.3f} test MCC={t:.3f} pos_pred={np_}')
