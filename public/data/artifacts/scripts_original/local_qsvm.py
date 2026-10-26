import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def overlap(a, b):
    qml.AngleEmbedding(a * np.pi, wires=range(4))
    qml.adjoint(qml.AngleEmbedding)(b * np.pi, wires=range(4))
    return qml.probs(wires=range(4))

def gram(A, B):
    return np.array([[overlap(a, b)[0] for b in B] for a in A])

for name in ["hf_notime", "hf_withtime"]:
    tr = pd.read_csv(f"{name}_train.csv")
    te = pd.read_csv(f"{name}_test.csv")
    Xtr, ytr = tr.iloc[:, :-1].values, tr["death_event"].values
    Xte, yte = te.iloc[:, :-1].values, te["death_event"].values

    print(f"Computing Gram matrix for {name} ({len(Xtr)} train x {len(Xtr)} train)...")
    K_train = gram(Xtr, Xtr)
    K_test = gram(Xte, Xtr)

    clf = SVC(kernel="precomputed").fit(K_train, ytr)
    pred = clf.predict(K_test)
    mcc = matthews_corrcoef(yte, pred)
    print(f"{name} local quantum-kernel MCC: {round(mcc, 3)}")
