"""Write docs/DEMO_SCRIPT.md. The spoken lines come from the page itself, so every number matches
the signed study. Re-run after `qneuro study build`:   python scripts/make_demo_script.py"""
from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

TERMINAL = [
    ("Re-run the submitted model from scratch", "qneuro study reproduce",
     "I rebuilt the submitted QSVM with my own code, not the platform's. The five class-based metrics match the submitted file to nine decimals and the confusion matrix is identical. A true ZZ kernel on the same inputs scores differently, so the submitted kernel has no entanglement."),
    ("Verify signature, hashes and audit chain", "qneuro study verify",
     "Six checks: the ML-DSA-65 signature on the study, the study hash, all 18 artifact hashes against the signed manifest, the audit chain, the sealing entry, and the published copy."),
    ("Break it on purpose", "qneuro study tamper-demo",
     "I change one digit in a copy of the submitted file and one digit in a copy of the signed study. Both edits must be detected. The untouched control must still pass."),
    ("Show the audit chain", "qneuro audit --audit-file runs/morphos/audit.jsonl",
     "Every step of the analysis is a signed entry linked to the previous one by hash. Editing any entry breaks every link after it."),
]

QA = [
    ("What is a quantum kernel SVM, in one breath?",
     "Each patient's four numbers are turned into the state of four qubits. The kernel is how much two patients' states overlap. A normal SVM then uses that overlap as its similarity measure. Here the qubits are simulated on an ordinary computer."),
    ("Why MCC and not accuracy?",
     "About 32% of patients died, so a model that says 'survived' for everyone is 68% accurate. E1 did exactly that: accuracy 0.6875, MCC 0.000. MCC is 0 for no skill and 1 for perfect, whatever the class balance."),
    ("Why did the Bloq runs E1 and E2 fail?",
     "I do not know. My sweep shows the ZZ kernel collapses when inputs are stretched to 0 to 2 pi. But with inputs in 0 to 1 the local ZZ kernel never predicted 'no deaths' in 300 random splits. So I call the range explanation a hypothesis. Whether the platform rescales inputs internally cannot be checked from outside."),
    ("Is the quantum model better than the classical ones?",
     "Not demonstrably. On 15 paired splits the best quantum kernel and the best classical model are tied, 0.600 against 0.599 MCC. The submitted holdout MCC of 0.746 is on 60 patients, where flipping one prediction moves it between 0.702 and 0.790."),
    ("You edited the submitted file. Why?",
     "Three text fields: the feature-map name, a leakage note and a verdict sentence. Every number is identical, and the diff is on screen. The original note said time inflates MCC across models, which the file never tested, and it named a ZZ map although the kernel that reproduces the numbers has no entanglement."),
    ("How do you know the submitted kernel is not ZZ?",
     "An independent angle-embedding kernel reproduces the submitted holdout metrics exactly, to nine decimals, with the same confusion matrix. A true ZZ kernel at the same input range gives holdout MCC 0.672. Two kernels, one match."),
    ("Can you prove the run happened on Bloq?",
     "No, and the tool says so. It can sign and hash what I give it. It cannot attest to where a number came from. I keep the experiment ids and screenshots from the platform."),
    ("Why does one cross-validation fold not reproduce?",
     "Fold 4 is 0.617 in my run and 0.633 in the file, a gap of 0.016 MCC. The other four folds and the holdout match exactly. I do not know the cause, and the claim table marks it as partly reproduced."),
    ("The tree-model baselines do not reproduce either?",
     "Right. Logistic regression and the RBF SVM in the submitted file reproduce exactly, logistic regression only when inputs are multiplied by 0.3, which is why it collapsed to the majority class. RandomForest, GradientBoosting and XGBoost do not reproduce with default settings and seed 42. The platform's settings for them are not visible to me."),
    ("What is wrong with the time feature?",
     "It is how long a patient was followed, and patients who die are followed for less time. Alone it scores AUROC 0.812, higher than any other single feature. Replacing it with serum sodium lowers MCC for every model by 0.164 to 0.229. The track allows it, but a model that needs it cannot predict ahead."),
    ("Why a corrected t-test and why Holm?",
     "Cross-validation folds share training patients, so a plain paired t-test is overconfident. The Nadeau-Bengio correction widens the variance. Holm corrects for trying three quantum variants, because picking the best of several looks better than it is."),
    ("What is ML-DSA and why use it here?",
     "A NIST post-quantum signature standard, FIPS 204. Results about patients stay relevant for decades, and signatures made today with classical algorithms could be forged later by a quantum computer. The library is a pure-Python reference implementation, unaudited, so this shows the pattern and is not production security."),
    ("Would this run on real quantum hardware?",
     "Not tried. A hardware kernel needs one circuit per pair of patients. For 239 training patients that is 239 x 238 / 2 = 28,441 circuits for the training matrix and 60 x 239 = 14,340 more for the holdout, 42,781 in total, each repeated for shots. The simulator needs one state per patient and a matrix product."),
    ("Is this usable clinically?",
     "No. It is research tooling on a public dataset of 299 patients from one pair of hospitals in Faisalabad, Pakistan. It is not a diagnostic tool."),
    ("What would you do next?",
     "Run the no-time comparison on the Bloq platform, lock a holdout before choosing anything, find out whether the platform rescales inputs, and run a small hardware job."),
]


def collect() -> list[dict]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(Quiet, directory=str(ROOT / "public")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto("http://127.0.0.1:8766/")
            pg.wait_for_function("window.QN && window.QN.ready === true", timeout=20000)
            steps = pg.evaluate("window.QN.tour.steps.map(s => ({...s, say: window.QN.tour.fill(s.say)}))")
            b.close()
        return steps
    finally:
        srv.shutdown()


def main() -> None:
    steps = collect()
    L: list[str] = []
    L += ["# Professor demo script", "",
          "About 12 minutes of demo, then questions. Left screen: terminal. Right screen: the study page.",
          "Spoken lines are generated from the signed study, so every number matches what is on screen.", "",
          "## Before you start", "",
          "1. Run `.\\scripts\\setup.ps1` once. It installs, tests, rebuilds and signs the study.",
          "2. Start the page: `.\\scripts\\serve.ps1`, then open http://127.0.0.1:8080/ (or your Vercel address).",
          "3. In the page press **Start walkthrough**. Arrow keys move between steps. Escape closes it.",
          "4. Keep the terminal in the repository root with the virtual environment active (`.\\.venv\\Scripts\\Activate.ps1`).",
          "5. Optional: `python demo\\run_demo.py --pace 1.3` plays the whole browser part by itself while you talk.", "",
          "## One-line opening", "",
          "> We built a quantum-kernel classifier for heart-failure survival on the Bloq platform. This session shows what survived a re-check, what did not, and how anyone can verify it.", "",
          "## Part A: the browser walkthrough", ""]
    for i, s in enumerate(steps, 1):
        L += [f"### A{i}. {s['title']}", "", f"**On screen:** {s['show']}", ""]
        if s.get("action"):
            a = s["action"]
            L += [f"**The page does this for you:** clicks `{a['click']}`" + (f", then `{a['then']}`." if a.get("then") else "."), ""]
        L += [f"**Say:** {s['say']}", ""]
        if s.get("terminal"):
            L += ["**Then switch to the terminal for Part B.**", ""]
    L += ["## Part B: the terminal", ""]
    for i, (t, cmd, say) in enumerate(TERMINAL, 1):
        L += [f"### B{i}. {t}", "", "```powershell", cmd, "```", "", f"**Say:** {say}", ""]
    L += ["Or run all four with pauses: `.\\scripts\\demo.ps1`.", "",
          "## If something goes wrong", "",
          "| Problem | What to do |", "| --- | --- |",
          "| The page shows 'Could not load the study data' | You opened the file directly. Use `.\\scripts\\serve.ps1` or the Vercel address. |",
          "| Hash buttons say hashing needs https | Use localhost or the Vercel address, not a file path or a plain-http IP. |",
          "| `qneuro study verify` fails | Run `qneuro study build` once, then verify again. The public key is created on your machine on first build. |",
          "| Browser demo window does not open | Run `python -m playwright install chromium`, or present manually with the arrow keys. |", "",
          "## Questions you may get", ""]
    for q, a in QA:
        L += [f"**{q}**", "", a, ""]
    L += ["## Do not say", "",
          "- That the quantum model beats classical models. The guard says tied.",
          "- That the run is verified on Bloq. Only the files are verified.",
          "- That the collapse of E1 and E2 is explained. It is a hypothesis.",
          "- That this is a clinical tool.", ""]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "DEMO_SCRIPT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote docs/DEMO_SCRIPT.md")


if __name__ == "__main__":
    main()
