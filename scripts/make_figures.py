"""Draw the README figures as SVG from the signed study. Re-run after `qneuro study build`:
    python scripts/make_figures.py
Every number is read from public/data/study.json, public/data/audit.jsonl or data/morphos/. SVG only, with
system fonts, because GitHub does not load web fonts inside images. Palette follows DESIGN.md (no blue or purple)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
ENV = json.loads((ROOT / "public/data/study.json").read_text("utf-8"))
S = ENV["study"]
AUDIT = [json.loads(l) for l in (ROOT / "public/data/audit.jsonl").read_text("utf-8").splitlines() if l.strip()]

PAPER, PAPER2, INK, INK2 = "#E8E7E2", "#DDDBD3", "#1C1B18", "#57544C"
VERD, BRASS, OX, RULE = "#1E6B55", "#86600F", "#A83224", "#CAC7BC"
TINT = {"reproduced": "#D3E0D6", "supported": "#D3E0D6", "partly": "#E6DDC3", "hypothesis": "#E6DDC3",
        "not_reproduced": "#E8D0C8", "not_verifiable": "#DDDBD3"}
SANS = "ui-sans-serif, system-ui, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
f3 = lambda x: ("\u2212" if x < 0 else "") + f"{abs(x):.3f}"
sg = lambda x, d=3: ("+" if x >= 0 else "\u2212") + f"{abs(x):.{d}f}"


class Svg:
    def __init__(self, w: int, h: int, title: str, desc: str):
        self.w, self.h, self.p = w, h, []
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" role="img" '
                     f'aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
                     f'<rect width="{w}" height="{h}" fill="{PAPER}"/>')

    def add(self, s: str): self.p.append(s)

    def text(self, x, y, s, size=13, fill=INK, anchor="start", weight=400, mono=False, italic=False):
        fam = MONO if mono else SANS
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{st}>{escape(str(s))}</text>')

    def rect(self, x, y, w, h, fill="none", stroke=INK, sw=1, r=2):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def circle(self, x, y, r, fill=INK, stroke="none", sw=1):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def arrow(self, x1, y1, x2, y2, stroke=INK, sw=1.4):
        a = math.atan2(y2 - y1, x2 - x1); s = 7
        self.line(x1, y1, x2, y2, stroke, sw)
        self.add(f'<polygon points="{x2:.1f},{y2:.1f} {x2 - s * math.cos(a - .45):.1f},{y2 - s * math.sin(a - .45):.1f} '
                 f'{x2 - s * math.cos(a + .45):.1f},{y2 - s * math.sin(a + .45):.1f}" fill="{stroke}"/>')

    def save(self, name):
        (OUT / name).write_text(self.head + "".join(self.p) + "</svg>\n", encoding="utf-8")
        print("wrote", name)


def lin(d0, d1, r0, r1): return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def csv_counts(name):
    rows = list(csv.DictReader(open(ROOT / "data/morphos/original" / name, encoding="utf-8")))
    return len(rows), sum(int(float(r["death_event"])) for r in rows)


def hero():
    d = S["meta"]["dataset"]; v = S["guard"]["withtime"]["verdicts"]["mcc"]
    st = {}
    for c in S["claims"]: st[c["status"]] = st.get(c["status"], 0) + 1
    holds = st.get("reproduced", 0) + st.get("supported", 0)
    g = Svg(880, 330, "QNeuro Forge at a glance", "From 299 patients to a signed ledger, with the four headline results.")
    g.text(24, 38, "Does the quantum kernel hold up?", 22, INK, weight=600)
    g.text(24, 60, "Team Morphos heart-failure QSVM, every number recomputed and signed", 13, INK2)
    stages = [("299 patients", "UCI heart-failure\nrecords"), ("4 numbers each", "age, ejection frac.,\ncreatinine, time"),
              ("4 qubits", "one simulated state\nper patient"), ("kernel", "overlap of two\npatients' states"),
              ("SVM", "draws the line\nbetween outcomes"), ("signed ledger", "hashes, ML-DSA-65,\naudit chain")]
    x0, w, gap, y = 24, 130, 10, 84
    for i, (a, b) in enumerate(stages):
        x = x0 + i * (w + gap)
        g.rect(x, y, w, 74, PAPER, INK, 1)
        g.text(x + 10, y + 24, a, 13, VERD if i == 5 else INK, weight=600)
        for j, ln in enumerate(b.split("\n")): g.text(x + 10, y + 44 + j * 15, ln, 11, INK2)
        if i < len(stages) - 1: g.arrow(x + w + 1, y + 37, x + w + gap - 1, y + 37)
    g.line(24, 184, 856, 184, INK, 1)
    figs = [(f3(S["e5"]["holdout"]["mcc"]), f"submitted QSVM, holdout MCC\n({d['test']} patients)"),
            (f"{f3(v['quantum_score'])} / {f3(v['classical_score'])}", "best quantum vs best classical,\nMCC over 15 paired splits"),
            (f"{holds} of {len(S['claims'])}", "claims from the report\nhold after re-checking"),
            (f"{len(S['manifest'])} + {ENV['audit']['entries']}", "files hashed + signed\naudit entries")]
    for i, (n, l) in enumerate(figs):
        x = 24 + i * 210
        g.text(x, 232, n, 23, INK, mono=True, weight=500)
        for j, ln in enumerate(l.split("\n")): g.text(x, 256 + j * 16, ln, 12, INK2)
        if i: g.line(x - 14, 200, x - 14, 290, RULE, 1)
    g.text(24, 316, "No hardware ran any circuit: every quantum state here is simulated on an ordinary computer.", 11.5, INK2, italic=True)
    g.save("hero.svg")


def architecture():
    g = Svg(880, 420, "How the pieces fit together", "Inputs feed the study layer, which reuses the engine's guard, signing and audit log, and writes signed outputs that the page reads.")
    g.text(24, 30, "How the pieces fit together", 17, INK, weight=600)
    cols = [("Inputs, as submitted", "data/morphos/", [("4 prepared CSVs", ""), ("Bloq raw outputs E1, E2", ""), ("submitted file (E5)", ""), ("original Bloq export", ""), ("local E6 file", ""), ("the 5 original scripts", "")], INK),
            ("Study layer (new)", "qneuro/study/", [("kernels.py", "ZZ and angle kernels"), ("experiments.py", "re-runs E3 to E6"), ("claims.py", "20 claims, a status each"), ("build.py", "manifest, signing, log")], INK),
            ("Engine (reused)", "qneuro/", [("guard.py", "corrected test, Holm, verdict"), ("vault.py", "ML-DSA-65 signing"), ("audit.py", "hash-chained log")], VERD),
            ("Outputs", "public/ and runs/", [("study.json", "the signed study"), ("audit.jsonl", "27 signed entries"), ("artifacts/", "18 hashed copies")], VERD)]
    w, gap, y, h = 190, 24, 56, 214
    for i, (t, sub, items, col) in enumerate(cols):
        x = 24 + i * (w + gap)
        g.rect(x, y, w, h, PAPER, col, 2 if col == VERD else 1)
        g.text(x + 12, y + 26, t, 13.5, col, weight=600); g.text(x + 12, y + 44, sub, 11, INK2, mono=True)
        g.line(x + 12, y + 54, x + w - 12, y + 54, RULE, 1)
        for j, (a_, b_) in enumerate(items):
            if i == 0:
                g.text(x + 12, y + 78 + j * 25, a_, 12, INK)
            else:
                g.text(x + 12, y + 78 + j * 36, a_, 12, INK, mono=True, weight=600); g.text(x + 12, y + 94 + j * 36, b_, 11, INK2)
        if i < 3: g.arrow(x + w + 2, y + 100, x + w + gap - 2, y + 100)
    g.text(24 + 2 * (w + gap) - 12, y + h + 18, "scores out, verdict back; sign and log each step", 10.5, INK2, anchor="middle")
    g.arrow(24 + 3 * (w + gap) + w / 2, y + h + 2, 24 + 3 * (w + gap) + w / 2, y + h + 40)
    g.rect(24, y + h + 42, 832, 76, PAPER, INK, 1)
    g.text(38, y + h + 66, "public/index.html: the page and its 11-step walkthrough", 13.5, INK, weight=600)
    g.text(38, y + h + 86, "Reads study.json and audit.jsonl, then re-hashes each of the 18 files in your browser with SHA-256", 11.8, INK2)
    g.text(38, y + h + 104, "and compares them with the signed manifest. The terminal checks the ML-DSA-65 signature and the audit chain.", 11.8, INK2)
    g.save("architecture.svg")


def concept():
    g = Svg(880, 330, "How a quantum-kernel SVM works", "Two patients become two quantum states; their overlap is the kernel; the kernel matrix feeds a normal SVM.")
    g.text(24, 30, "How a quantum-kernel SVM decides", 18, INK, weight=600)
    g.rect(24, 56, 170, 150, PAPER, INK)
    g.text(36, 78, "two patients", 12.5, INK2, weight=600)
    rows = [("", "a", "b"), ("age", "0.59", "0.27"), ("ejection", "0.30", "0.55"), ("creatinine", "0.07", "0.03"), ("time", "0.12", "0.80")]
    for i, (k, a, b) in enumerate(rows):
        y = 100 + i * 20
        g.text(36, y, k, 11.5, INK2); g.text(128, y, a, 11.5, INK, anchor="end", mono=True); g.text(176, y, b, 11.5, INK, anchor="end", mono=True)
    g.text(30, 222, "illustrative values, scaled to 0 to 1", 11, INK2, italic=True)
    g.arrow(198, 130, 236, 130)
    g.rect(240, 56, 200, 150, PAPER, INK)
    g.text(252, 78, "feature map (4 qubits)", 12.5, INK2, weight=600)
    for q in range(4):
        y = 104 + q * 24
        g.line(254, y, 428, y, INK, 1)
        g.rect(270, y - 9, 38, 18, PAPER2, INK); g.text(289, y + 4, f"x{q + 1}", 11, INK, anchor="middle", mono=True)
        g.text(250, y + 4, f"q{q}", 10.5, INK2, anchor="end", mono=True)
    g.rect(334, 94, 78, 88, PAPER, VERD, 1.5, 3); g.text(373, 142, "entangle?", 11, VERD, anchor="middle")
    g.text(373, 156, "(ZZ yes,", 10.5, VERD, anchor="middle"); g.text(373, 169, "angle no)", 10.5, VERD, anchor="middle")
    g.arrow(444, 130, 482, 130)
    g.rect(486, 56, 190, 150, PAPER, INK)
    g.text(498, 78, "one state per patient", 12.5, INK2, weight=600)
    g.text(498, 108, "|\u03c6(a)\u27e9   |\u03c6(b)\u27e9", 14, INK, mono=True)
    g.text(498, 136, "kernel = overlap", 12.5, INK, weight=600)
    g.text(498, 158, "K(a,b) = |\u27e8\u03c6(a)|\u03c6(b)\u27e9|\u00b2", 12, VERD, mono=True, weight=500)
    g.text(498, 182, "1 = identical, 0 = unrelated", 11.5, INK2)
    g.arrow(680, 130, 718, 130)
    g.rect(722, 56, 134, 150, PAPER, INK)
    g.text(734, 78, "ordinary SVM", 12.5, INK2, weight=600)
    for i in range(5):
        for j in range(5):
            g.rect(738 + j * 16, 94 + i * 12, 14, 10, PAPER2 if (i + j) % 3 else "#D3E0D6", "none", 0, 0)
    g.text(734, 168, "239 \u00d7 239 matrix", 11.5, INK, mono=True)
    g.text(734, 184, "of overlaps feeds", 11.5, INK2); g.text(734, 198, "the usual SVM", 11.5, INK2)
    g.line(24, 244, 856, 244, RULE, 1)
    notes = ["The quantum part is only the similarity measure. Training and prediction are classical.",
             "Here the qubits are simulated: one state per patient, then a matrix product gives every overlap.",
             "On real hardware each pair of patients would need its own circuit run many times (see the cost estimate below)."]
    for i, n in enumerate(notes): g.text(24, 268 + i * 18, n, 12.5, INK2)
    g.save("concept-qsvm.svg")


def circuits():
    g = Svg(880, 412, "The two kernels in this study as circuits", "Angle embedding uses one rotation per qubit and no entanglement; the ZZ feature map adds Hadamards, phases and pairwise entangling blocks.")
    g.text(24, 30, "Two different kernels: what was submitted, and what the platform default is", 17, INK, weight=600)
    g.rect(24, 50, 400, 300, PAPER, INK)
    g.text(40, 76, "Angle embedding  (the submitted E5 kernel)", 13.5, VERD, weight=600)
    for q in range(4):
        y = 118 + q * 40
        g.line(70, y, 396, y, INK, 1); g.text(60, y + 4, f"q{q}", 11, INK2, anchor="end", mono=True)
        g.rect(150, y - 13, 70, 26, PAPER2, INK); g.text(185, y + 5, f"RX(x{q + 1})", 12, INK, anchor="middle", mono=True)
    g.text(40, 292, "Each qubit is rotated on its own. Nothing connects them,", 12, INK2)
    g.text(40, 309, "so the state is a product state: no entanglement.", 12, INK2)
    g.text(40, 332, "K(a,b) = \u220f\u1d62 cos\u00b2((a\u1d62 \u2212 b\u1d62)/2)", 13.5, INK, mono=True, weight=500)
    g.rect(456, 50, 400, 300, PAPER, INK)
    g.text(472, 76, "ZZ feature map  (Bloq default; local E3, E4, E6)", 13, BRASS, weight=600)
    ys = [118 + q * 40 for q in range(4)]
    for q, y in enumerate(ys):
        g.line(502, y, 840, y, INK, 1); g.text(492, y + 4, f"q{q}", 11, INK2, anchor="end", mono=True)
        g.rect(512, y - 12, 26, 24, PAPER2, INK); g.text(525, y + 4, "H", 12, INK, anchor="middle", mono=True)
        g.rect(546, y - 12, 62, 24, PAPER2, INK); g.text(577, y + 4, f"P(2x{q + 1})", 11.5, INK, anchor="middle", mono=True)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for k, (i, j) in enumerate(pairs):
        x = 626 + k * 34
        g.line(x + 9, ys[i], x + 9, ys[j], VERD, 2.4)
        g.circle(x + 9, ys[i], 4, VERD); g.circle(x + 9, ys[j], 4, VERD)
    g.text(626, 94, "one entangling block per pair", 11, VERD)
    g.text(472, 292, "Hadamards, single-qubit phases, then a phase on every pair", 12, INK2)
    g.text(472, 309, "of qubits (6 pairs for 4 qubits). The qubits become entangled.", 12, INK2)
    g.text(472, 332, "pair phase: P(2(\u03c0 \u2212 x\u1d62)(\u03c0 \u2212 x\u2c7c))", 13, INK, mono=True, weight=500)
    g.text(24, 376, "Same four numbers, same SVM, different kernels, different answers.", 12.5, INK2)
    g.text(24, 394, f"On the submitted holdout the angle kernel scores {f3(S['e5']['holdout']['mcc'])} and the ZZ kernel {f3(S['e5']['zz_same_range_holdout_mcc'])}.", 12.5, INK2)
    g.save("circuits.svg")


def split():
    ntr, dtr = csv_counts("hf_withtime_train.csv"); nte, dte = csv_counts("hf_withtime_test.csv")
    g = Svg(880, 290, "How 299 patients were split", f"{ntr} training and {nte} holdout patients, stratified by outcome.")
    g.text(24, 30, "The data: 299 patients, split once, scaled on the training rows only", 17, INK, weight=600)
    W = 832; wtr = W * ntr / (ntr + nte)
    g.rect(24, 52, wtr, 56, "#D3E0D6", INK); g.rect(24 + wtr, 52, W - wtr, 56, PAPER2, INK)
    g.text(36, 76, f"{ntr} training patients", 14, INK, weight=600); g.text(36, 96, f"{dtr} died ({dtr / ntr * 100:.1f}%)", 12, INK2)
    g.text(24 + wtr + 12, 76, f"{nte} holdout", 14, INK, weight=600); g.text(24 + wtr + 12, 96, f"{dte} died ({dte / nte * 100:.1f}%)", 12, INK2)
    g.text(24, 128, "Stratified 80/20 split by outcome, random_state 7. The MinMax scaler is fitted on the training rows only,", 12, INK2)
    g.text(24, 144, "so holdout values can fall slightly outside 0 to 1; they are clipped before encoding.", 12, INK2)
    for k, (title, feats, col) in enumerate([("With follow-up time", ["ejection_fraction", "serum_creatinine", "age", "time"], VERD),
                                              ("Without time", ["ejection_fraction", "serum_creatinine", "age", "serum_sodium"], INK2)]):
        x = 24 + k * 424
        g.rect(x, 158, 408, 98, PAPER, col, 1.5)
        g.text(x + 14, 180, title, 13.5, col, weight=600)
        for i, ft in enumerate(feats):
            g.rect(x + 14 + (i % 2) * 190, 192 + (i // 2) * 30, 180, 24, PAPER2, "none", 0)
            g.text(x + 24 + (i % 2) * 190, 209 + (i // 2) * 30, f"{i + 1}  {ft}", 12, INK, mono=True)
    g.text(24, 278, "Four features because the track allows four qubits. Each feature is one qubit.", 12, INK2)
    g.save("data-split.svg")


def experiments():
    L = S["ledger"]; row = {r["id"]: r for r in L}
    g = Svg(880, 360, "The six experiments", "E1 and E2 on Bloq, E3 and E4 local sweeps, E5 submitted, E6 local not submitted, each with a re-check status.")
    g.text(24, 30, "Six experiments and what the re-check says about each", 17, INK, weight=600)
    for name, x in (("Bloq platform", 24), ("Local diagnosis", 318), ("Submitted entry", 612)): g.text(x, 62, name, 12.5, INK2, weight=600)
    cw, ch = 132, 150
    layout = {"E1": (24, 74), "E2": (164, 74), "E3": (318, 74), "E4": (458, 74), "E5": (612, 74)}
    short = {"E1": ("default ZZ map,", "no time"), "E2": ("default ZZ map,", "with time"), "E3": ("sweep: depth", "and input range"),
             "E4": ("sweep: pre-", "processing, range"), "E5": ("submitted: angle", "QSVM + 5 classical")}
    nums = {"E1": (f3(row["E1"]["holdout_mcc"]), "MCC, Bloq 48-row split"), "E2": (f3(row["E2"]["holdout_mcc"]), "MCC, Bloq 48-row split"),
            "E3": (row["E3"].get("span", "").replace("CV MCC ", "").replace(" (with time)", ""), "CV MCC, with time"),
            "E4": (f3(row["E4"]["cv_mcc"]), "CV MCC, range 0.3"), "E5": (f3(row["E5"]["holdout_mcc"]), f"holdout, CV {f3(row['E5']['cv_mcc'])}")}
    for id_, (x, y) in layout.items():
        r = row[id_]
        g.rect(x, y, cw, ch, TINT[r["status"]], INK)
        g.text(x + 10, y + 26, id_, 18, INK, mono=True, weight=600)
        g.text(x + 10, y + 46, r["where"], 11.5, INK2)
        for i, ln in enumerate(short[id_]): g.text(x + 10, y + 70 + i * 15, ln, 11.5, INK)
        g.text(x + 10, y + ch - 36, nums[id_][0], 14 if len(nums[id_][0]) < 8 else 12, INK, mono=True, weight=500)
        g.text(x + 10, y + ch - 16, nums[id_][1], 10, INK2)
        g.text(x + cw - 10, y + 26, r["status_label"].split(" ")[0], 10.5, INK2, anchor="end")
    r = row["E6"]; x, y = 612, 74 + ch + 20
    g.rect(x, y, 244, 100, TINT[r["status"]], INK, 1, 2)
    g.text(x + 10, y + 24, "E6", 18, INK, mono=True, weight=600); g.text(x + 52, y + 24, "Local, not submitted", 12, INK2)
    g.text(x + 10, y + 46, "True ZZ kernel and five classical models", 11.5, INK)
    g.text(x + 10, y + 62, "on identical folds", 11.5, INK)
    g.text(x + 10, y + 88, f"CV {f3(r['cv_mcc'])}   holdout {f3(r['holdout_mcc'])}", 13, INK, mono=True, weight=500)
    g.arrow(298, 150, 314, 150); g.arrow(592, 150, 608, 150); g.arrow(734, 226, 734, 244)
    lx = 24
    for key, lab in [("reproduced", "reproduced / supported"), ("partly", "partly / hypothesis"), ("not_reproduced", "not reproduced")]:
        g.rect(lx, 262, 18, 14, TINT[key], INK); g.text(lx + 26, 274, lab, 11.5, INK2); lx += 190
    g.text(24, 306, "E1, E2 and E5 were run on Bloq: only their files can be checked, not where they came from.", 12, INK2)
    g.text(24, 326, "E3, E4 and E6 are local, so every number can be re-run with one command.", 12, INK2)
    g.save("experiments.svg")


def holdout():
    (tn, fp), (fn, tp) = S["e5"]["holdout_confusion"]; sens = S["sensitivity"]
    g = Svg(880, 330, "The 60-patient holdout, patient by patient", "Circles are survivors and squares are patients who died; red marks the seven wrong predictions.")
    g.text(24, 30, "The 60-patient holdout, one dot per patient", 17, INK, weight=600)
    g.text(24, 52, "Circles survived, squares died. Grey: the submitted QSVM predicted correctly. Red: it did not.", 12.5, INK2)

    def grid(x0, y0, ok, wrong, cols, square):
        for i, c in enumerate([True] * ok + [False] * wrong):
            cx, cy = x0 + (i % cols) * 26 + 10, y0 + (i // cols) * 26 + 10
            col = INK2 if c else OX
            if square: g.rect(cx - 9, cy - 9, 18, 18, col, "none", 0, 2)
            else: g.circle(cx, cy, 9.5, col)
    g.text(24, 84, f"{tn + fp} survived", 12.5, INK, weight=600); grid(24, 94, tn, fp, 7, False)
    g.text(232, 84, f"{tp + fn} died", 12.5, INK, weight=600); grid(232, 94, tp, fn, 5, True)
    x = 396
    g.text(x, 84, "confusion matrix", 12.5, INK, weight=600)
    g.rect(x, 96, 230, 110, PAPER, RULE)
    g.text(x + 108, 118, "pred. survived", 11, INK2, anchor="end", mono=True); g.text(x + 218, 118, "pred. died", 11, INK2, anchor="end", mono=True)
    g.line(x, 126, x + 230, 126, INK, 1)
    g.text(x + 8, 150, "survived", 11, INK2, mono=True); g.text(x + 108, 150, tn, 14, INK, anchor="end", mono=True); g.text(x + 218, 150, fp, 14, OX, anchor="end", mono=True)
    g.text(x + 8, 184, "died", 11, INK2, mono=True); g.text(x + 108, 184, fn, 14, OX, anchor="end", mono=True); g.text(x + 218, 184, tp, 14, INK, anchor="end", mono=True)
    g.text(660, 84, "MCC on this holdout", 12.5, INK, weight=600)
    g.text(660, 128, f3(sens["baseline"]), 38, INK, mono=True, weight=500)
    g.text(660, 152, f"accuracy {S['e5']['holdout']['accuracy']:.3f}, recall {S['e5']['holdout']['recall']:.3f}", 11.5, INK2)
    g.line(24, 250, 856, 250, RULE, 1)
    g.text(24, 276, f"Flip one prediction and MCC lands between {f3(sens['one_patient_min'])} and {f3(sens['one_patient_max'])}.", 14, INK, weight=600)
    g.text(24, 298, f"Flip two and it lands between {f3(sens['two_patients_min'])} and {f3(sens['two_patients_max'])}. A gap of a few hundredths between two models is inside that noise.", 12.5, INK2)
    g.save("holdout.svg")


NAMES = {"qsvm_zz_x0.3": "QSVM, ZZ kernel, inputs x0.3", "qsvm_zz_x1.0": "QSVM, ZZ kernel, inputs x1.0", "qsvm_angle_x1.0": "QSVM, angle kernel, x1.0",
         "logistic_regression": "Logistic regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting", "svm_rbf": "SVM, RBF kernel", "xgboost": "XGBoost"}


def interval(variant, metric, fname, title):
    t = sorted(S["guard"][variant]["table"], key=lambda r: -r[metric + "_mean"])
    bestc = next(r for r in t if r["kind"] == "classical")
    lo = min(r[metric + "_ci"][0] for r in t); hi = max(r[metric + "_ci"][1] for r in t)
    d0, d1 = math.floor((lo - .02) * 10) / 10, math.ceil((hi + .02) * 10) / 10
    H = 74 + len(t) * 32 + 60
    g = Svg(880, H, title, f"Interval plot of {metric} for {len(t)} models over 15 paired splits.")
    g.text(24, 30, title, 17, INK, weight=600)
    v = S["guard"][variant]["verdicts"][metric]
    g.text(24, 52, f"Guard verdict: {v['category'].replace('_', ' ')}.  Best quantum {f3(v['quantum_score'])}, best classical {f3(v['classical_score'])}, "
                   f"difference {sg(v['delta'])}, corrected p {v['p_value']:.4f}.", 12.5, INK2)
    x = lin(d0, d1, 250, 760); top = 74
    step = 0.2 if (d1 - d0) > .55 else 0.1
    k = math.ceil(d0 / step - 1e-9)
    while k * step <= d1 + 1e-9:
        tv = k * step; g.line(x(tv), top, x(tv), H - 46, RULE, 1, "3 3"); g.text(x(tv), H - 28, f"{tv:.1f}", 11, INK2, anchor="middle", mono=True); k += 1
    for i, r in enumerate(t):
        y = top + i * 32 + 16; col = VERD if r["kind"] == "quantum" else (INK if r is bestc else INK2)
        g.text(24, y + 4, NAMES.get(r["model"], r["model"]), 12.5, INK if r["kind"] == "quantum" else INK2, weight=600 if r["kind"] == "quantum" else 400)
        a, b = r[metric + "_ci"]; g.line(x(a), y, x(b), y, col, 2.2); g.line(x(a), y - 5, x(a), y + 5, col, 2.2); g.line(x(b), y - 5, x(b), y + 5, col, 2.2)
        g.circle(x(r[metric + "_mean"]), y, 5.5, col); g.text(856, y + 4, f3(r[metric + "_mean"]), 12.5, INK, anchor="end", mono=True)
    g.line(250, H - 46, 760, H - 46, INK, 1)
    g.text(24, H - 8, f"{metric.upper()}: mean with 95% t-interval over 15 paired splits (5-fold x 3 repeats). Green: quantum. Black: best classical.", 11.5, INK2)
    g.save(fname)


def line_panel(g, x0, y0, w, h, series, xt, yd, yt, title, xlabel):
    xs = [p[0] for s in series for p in s[2]]; x = lin(min(xs), max(xs), x0 + 36, x0 + w - 96); y = lin(yd[0], yd[1], y0 + h - 40, y0 + 30)
    g.text(x0, y0 + 14, title, 12.5, INK, weight=600)
    for t in yt: g.line(x0 + 36, y(t), x0 + w - 96, y(t), RULE, 1, "3 3"); g.text(x0 + 30, y(t) + 4, f"{t:.1f}", 11, INK2, anchor="end", mono=True)
    g.line(x0 + 36, y(0), x0 + w - 96, y(0), INK, 1)
    for v, l in xt: g.text(x(v), y0 + h - 20, l, 11, INK2, anchor="middle", mono=True)
    g.text((x0 + 36 + x0 + w - 96) / 2, y0 + h - 2, xlabel, 11, INK2, anchor="middle")
    ends = sorted([(y(s[2][-1][1]), s) for s in series], key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 15: ends[i] = (ends[i - 1][0] + 15, ends[i][1])
    for name, col, pts in series:
        g.add(f'<polyline fill="none" stroke="{col}" stroke-width="2.2" points="{" ".join(f"{x(p[0]):.1f},{y(p[1]):.1f}" for p in pts)}"/>')
    for ly, s in ends: g.text(x(s[2][-1][0]) + 8, ly + 4, s[0], 11.5, s[1], weight=600)


def sweeps():
    e3, e4 = S["sweeps"]["e3"], S["sweeps"]["e4"]
    g = Svg(880, 330, "Input range sweeps", "Cross-validated MCC against input range for E3 and E4, with and without time.")
    g.text(24, 30, "Does a wider input range break the ZZ kernel?", 17, INK, weight=600)
    pts3 = lambda var: [(i, next(r["cv_mcc"] for r in e3 if r["data"] == var and r["range"] == rg and r["depth"] == 1 and r["class_weight"] == "none")) for i, rg in enumerate(["0 to 1", "0 to pi", "0 to 2pi"])]
    line_panel(g, 24, 50, 420, 260, [("with time", VERD, pts3("withtime")), ("no time", INK2, pts3("notime"))], [(0, "0 to 1"), (1, "0 to \u03c0"), (2, "0 to 2\u03c0")], (-0.1, 0.6), [0, .2, .4, .6], "E3, depth 1: cross-validated MCC", "input range fed to the ZZ feature map")
    pts4 = lambda var: sorted((r["scale"], r["cv_mcc"]) for r in e4 if r["data"] == var and r["mode"] == "minmax")
    line_panel(g, 456, 50, 400, 260, [("with time", VERD, pts4("withtime")), ("no time", INK2, pts4("notime"))], [(v, str(v)) for v in (0.3, 0.5, 0.75, 1.0, 1.5)], (0, 0.7), [0, .2, .4, .6], "E4: cross-validated MCC by range multiplier", "input range multiplier (min-max inputs)")
    g.save("sweeps.svg")


def leakage():
    L = S["leakage"]; tn = sorted([r for r in L["single_feature"] if r["variant"] == "withtime"], key=lambda r: -r["auroc_alone"])
    g = Svg(880, 330, "Follow-up time carries the outcome", "Single-feature AUROC, and MCC with time versus serum sodium for four models.")
    g.text(24, 30, "Follow-up time carries the outcome", 17, INK, weight=600)
    g.text(24, 50, "Left: each feature alone (AUROC, 239 training patients). Right: CV MCC with time (filled) and serum sodium (open).", 12, INK2)
    x = lin(0.5, 0.9, 180, 395); top = 84
    g.text(24, 76, "single-feature AUROC (0.5 = no signal)", 11.5, INK2, weight=600)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9): g.line(x(t), top, x(t), top + 4 * 40, RULE, 1, "3 3"); g.text(x(t), top + 4 * 40 + 16, f"{t:.1f}", 11, INK2, anchor="middle", mono=True)
    g.line(x(.5), top, x(.5), top + 160, INK, 1)
    for i, r in enumerate(tn):
        y = top + i * 40 + 20; col = BRASS if r["feature"] == "time" else INK2
        g.text(24, y + 4, r["feature"], 12, INK, mono=True, weight=600 if r["feature"] == "time" else 400)
        g.line(x(.5), y, x(r["auroc_alone"]), y, col, 2.2); g.circle(x(r["auroc_alone"]), y, 5.5, col); g.text(x(r["auroc_alone"]) + 10, y + 4, f3(r["auroc_alone"]), 12, INK, mono=True)
    P = L["paired"]; x2 = lin(0.2, 0.7, 590, 800); g.text(470, 76, "cross-validated MCC", 11.5, INK2, weight=600)
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7): g.line(x2(t), top, x2(t), top + 160, RULE, 1, "3 3"); g.text(x2(t), top + 176, f"{t:.1f}", 11, INK2, anchor="middle", mono=True)
    pretty = {"LogisticRegression": "Logistic regression", "RandomForest": "Random forest", "QSVM (ZZ kernel, inputs x 0.3)": "QSVM, ZZ kernel"}
    for i, r in enumerate(P):
        y = top + i * 40 + 20; col = VERD if r["model"].startswith("QSVM") else INK
        g.text(470, y + 4, pretty.get(r["model"], r["model"]), 12, INK if col == INK else VERD, weight=600 if col == VERD else 400)
        g.line(x2(r["mcc_without_time"]), y, x2(r["mcc_with_time"]), y, col, 2.2)
        g.circle(x2(r["mcc_without_time"]), y, 5.5, PAPER, col, 2); g.circle(x2(r["mcc_with_time"]), y, 5.5, col)
        g.text(856, y + 4, sg(-r["mcc_drop"]), 12, INK, anchor="end", mono=True)
    drops = [r["mcc_drop"] for r in P]; t = next(r for r in tn if r["feature"] == "time")
    g.line(24, 276, 856, 276, RULE, 1)
    g.text(24, 300, f"time alone: AUROC {f3(t['auroc_alone'])}, the strongest single feature. Removing it costs every model {f3(min(drops))} to {f3(max(drops))} MCC.", 13, INK, weight=600)
    g.text(24, 318, "A model that needs follow-up time cannot predict ahead, so the with-time scores are optimistic for real use.", 12, INK2)
    g.save("time-leakage.svg")


def claims():
    C = S["claims"]; cols, cw, ch = 4, 208, 62; rows = math.ceil(len(C) / cols)
    H = 90 + rows * (ch + 8) + 56
    g = Svg(880, H, "The 20 claims and their status", "Each claim from the report coloured by whether it reproduced.")
    g.text(24, 30, "Every claim in the report, recomputed", 17, INK, weight=600)
    st = {}
    for c in C: st[c["status"]] = st.get(c["status"], 0) + 1
    holds = st.get("reproduced", 0) + st.get("supported", 0)
    g.text(24, 52, f"{holds} of {len(C)} hold ({st.get('reproduced', 0)} reproduced, {st.get('supported', 0)} supported by local analysis).", 12.5, INK2)
    for i, c in enumerate(C):
        x, y = 24 + (i % cols) * (cw + 5), 70 + (i // cols) * (ch + 8)
        g.rect(x, y, cw, ch, TINT[c["status"]], INK)
        g.text(x + 9, y + 20, c["id"], 12, INK, mono=True, weight=600)
        g.text(x + 9, y + 38, c["status_label"], 11.5, INK)
        s = ""
        for w in c["claim"].split():
            if len(s + " " + w) > 31: break
            s = (s + " " + w).strip()
        g.text(x + 9, y + 54, s + ("..." if s != c["claim"] else ""), 10, INK2)
    ly = 70 + rows * (ch + 8) + 14; lx = 24
    for key, lab in [("reproduced", "reproduced / supported"), ("partly", "partly / hypothesis"), ("not_reproduced", "not reproduced"), ("not_verifiable", "not verifiable here")]:
        g.rect(lx, ly, 18, 14, TINT[key], INK); g.text(lx + 26, ly + 12, lab, 11.5, INK2); lx += 205
    g.save("claims.svg")


def audit():
    pick = [0, 1, 2, len(AUDIT) - 1]
    g = Svg(880, 330, "The hash-chained audit log", "Each audit entry stores the previous entry's hash and an ML-DSA signature; editing one entry breaks the chain from there on.")
    g.text(24, 30, f"The audit log: {len(AUDIT)} signed entries, each linked to the one before", 17, INK, weight=600)
    w, gap = 190, 24
    for k, i in enumerate(pick):
        e = AUDIT[i]; x = 24 + k * (w + gap); bad = (k == 2)
        g.rect(x, 54, w, 168, PAPER, OX if bad else INK, 2 if bad else 1)
        g.text(x + 10, 76, f"entry {e['index']}" if i != len(AUDIT) - 1 else f"entry {e['index']}  (last)", 13, OX if bad else INK, weight=600, mono=True)
        g.text(x + 10, 96, e["action"], 10.5, INK, mono=True)
        for j, (lab, val) in enumerate([("prev hash", e["prev_hash"][:12]), ("payload", e["payload_hash"][:12]), ("entry hash", e["entry_hash"][:12]), ("signature", e["signature_hex"][:12] + "...")]):
            g.text(x + 10, 120 + j * 22, lab, 10.5, INK2); g.text(x + w - 10, 120 + j * 22, val, 10.5, OX if (bad and j == 2) else INK, anchor="end", mono=True)
        g.line(x + 10, 206, x + w - 10, 206, RULE, 1); g.text(x + 10, 216, "ML-DSA-65 over the entry", 9.5, INK2)
        if k < 3:
            g.arrow(x + w + 2, 140, x + w + gap - 2, 140, OX if k == 2 else INK)
            g.text(x + w + gap / 2, 130, "..." if k == 2 else "", 14, INK2, anchor="middle", weight=600)
    g.text(24, 252, "entry hash = SHA-256 of the entry's contents, including the previous entry's hash.", 12, INK2)
    g.text(24, 270, "Edit entry 2 (red, as an example) and its hash changes, the next entry's 'prev hash' no longer matches, and its signature fails.", 12, INK2)
    g.text(24, 288, f"The last entry seals the study: it records study SHA-256 {ENV['study_sha256'][:16]}... and is signed too.", 12, INK2)
    g.text(24, 310, "hashes shown are the first 12 characters of the real entries in public/data/audit.jsonl", 11, INK2, italic=True)
    g.save("audit-chain.svg")


if __name__ == "__main__":
    hero(); architecture(); concept(); circuits(); split(); experiments(); holdout()
    interval("withtime", "mcc", "fair-comparison.svg", "Fair comparison with follow-up time: MCC over 15 paired splits")
    if "notime" in S["guard"]: interval("notime", "mcc", "fair-comparison-notime.svg", "Fair comparison without time: MCC over 15 paired splits")
    sweeps(); leakage(); claims(); audit()
