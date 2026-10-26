import os, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef

LOCAL = "heart_failure_clinical_records_dataset.csv"
if os.path.exists(LOCAL):
    df = pd.read_csv(LOCAL)
else:
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=519)
    df = pd.concat([d.data.features, d.data.targets], axis=1)

df.columns = [c.lower() for c in df.columns]
print(df.shape)
print(df.columns.tolist())
print(df["death_event"].value_counts())

# One split of patients, shared by both variants
train, test = train_test_split(
    df, test_size=0.2, stratify=df["death_event"], random_state=7)

VARIANTS = {
    "hf_notime":   ["ejection_fraction", "serum_creatinine", "age", "serum_sodium"],
    "hf_withtime": ["ejection_fraction", "serum_creatinine", "age", "time"],
}

for name, cols in VARIANTS.items():
    sc = MinMaxScaler().fit(train[cols])          # fit on train only
    out = {}
    for part, frame in (("train", train), ("test", test)):
        o = pd.DataFrame(sc.transform(frame[cols]), columns=cols)
        o["death_event"] = frame["death_event"].values
        fn = f"{name}_{part}.csv"
        o.to_csv(fn, index=False)
        out[part] = o
        h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
        print(fn, o.shape, h)

    # classical reference score, for comparing against Bloq's MCC later
    clf = SVC(kernel="rbf").fit(out["train"][cols], out["train"]["death_event"])
    pred = clf.predict(out["test"][cols])
    print(name, "classical RBF-SVM test MCC:",
          round(matthews_corrcoef(out["test"]["death_event"], pred), 3))
