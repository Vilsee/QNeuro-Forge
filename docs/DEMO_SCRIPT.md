# Professor demo script

About 12 minutes of demo, then questions. Left screen: terminal. Right screen: the study page.
Spoken lines are generated from the signed study, so every number matches what is on screen.

## Before you start

1. Run `.\scripts\setup.ps1` once. It installs, tests, rebuilds and signs the study.
2. Start the page: `.\scripts\serve.ps1`, then open http://127.0.0.1:8080/ (or your Vercel address).
3. In the page press **Start walkthrough**. Arrow keys move between steps. Escape closes it.
4. Keep the terminal in the repository root with the virtual environment active (`.\.venv\Scripts\Activate.ps1`).
5. Optional: `python demo\run_demo.py --pace 1.3` plays the whole browser part by itself while you talk.

## One-line opening

> We built a quantum-kernel classifier for heart-failure survival on the Bloq platform. This session shows what survived a re-check, what did not, and how anyone can verify it.

## Part A: the browser walkthrough

### A1. The question and the answer

**On screen:** The sentence under the title, and the four figures beneath it.

**Say:** This is the study we submitted to the Quantum Hackathon with Bloq. The question: can a quantum-kernel support vector machine predict death in heart-failure patients as well as classical models? Everything on this page is computed from the files we submitted, nothing is typed in. The answer is in the sentence: on 239 training patients the best quantum kernel and the best classical model are tied, 0.600 against 0.599 on MCC. I will now show how every number was checked.

### A2. The six experiments

**On screen:** The table of E1 to E6. The last column is the re-check status. Row E5 opens to show its claims.

**The page does this for you:** clicks `#ledgerBody tr[data-id='E5']`.

**Say:** E1 and E2 ran on the Bloq platform and came out near zero. E3 and E4 are local sweeps I ran to understand why. E5 is the file we submitted. E6 is a stricter local benchmark we did not submit. The new part is the right-hand column: code re-ran each row and says whether its numbers reproduce.

### A3. Why the platform runs collapsed

**On screen:** Bloq settings and confusion matrices, then the range chart and the platform-gap box.

**Say:** On Bloq, the default ZZ feature map with four qubits gave MCC 0.0000 without time and 0.0116 with time, close to no skill. My first explanation was kernel concentration at large input ranges. The sweep confirms that effect: at range zero to two pi the cross-validated MCC falls to −0.01. But I tested the other half. With inputs in zero to one, our local ZZ kernel never predicted no-deaths in 300 random splits. So split luck does not explain the platform result, and whether Bloq rescales inputs internally cannot be checked from outside. I label that a hypothesis.

### A4. Every claim, recomputed

**On screen:** The claim table with a status for each row.

**Say:** Here is every claim from our report, the value we recomputed, and a status. 16 of 20 hold. The submitted QSVM holdout MCC of 0.746 reproduces exactly, with the same confusion matrix. It reproduces with an angle-embedding kernel, which has no entanglement. A true ZZ kernel on the same inputs gives 0.672, so the file's feature-map label had to be corrected. Some things did not reproduce: the tree-model baselines and one cross-validation fold. And one claim, that the run happened on Bloq, cannot be verified by any tool outside the platform.

### A5. The holdout is fragile

**On screen:** Sixty dots, one per patient. Grey is correct, red is an error. Clicking a dot flips that prediction.

**The page does this for you:** clicks `#holdBody button.wrong`.

**Say:** The holdout is only 60 patients. Each dot is one patient. There are 7 errors in red. Watch the MCC when I flip one prediction. One flipped patient moves it between 0.702 and 0.790, two between 0.659 and 0.834. So gaps of a few hundredths on this holdout are noise. That is why I chose the input range by cross-validation, not by the holdout.

### A6. Fair comparison

**On screen:** The interval plot for repeated cross-validation, then the Honesty Guard verdict.

**Say:** For a fair comparison I ran all eight models on the same 15 paired splits and gave the scores to the QNeuro Forge Honesty Guard. It uses a corrected resampled t-test, because cross-validation folds overlap, and it corrects for the 3 quantum configurations I tried. The verdict is no difference. The best quantum kernel scored 0.600 and the best classical model 0.599. The guard is the only code allowed to word a quantum advantage claim, and here it declines to.

### A7. Follow-up time

**On screen:** Single-feature scores, then with-time versus without-time for four models.

**Say:** The track allows follow-up time as a feature. It carries outcome information: alone it scores AUROC 0.812, higher than any other single feature. Swapping it for serum sodium lowers cross-validated MCC for every model, by 0.164 to 0.229. So the with-time numbers are optimistic for a model that has to predict before follow-up exists. Our report listed this comparison as future work. I ran it locally here, not on Bloq.

### A8. What was edited before submission

**On screen:** The three-row diff between the Bloq export and the submitted file.

**Say:** Before submitting, we corrected three text fields in the Bloq export: the feature-map name, a leakage note and a verdict sentence. This diff is computed from the two files. Every number is identical. The original note claimed time inflates MCC across models, which the file had never tested, and it called the kernel a ZZ map although it behaves as a product-state kernel.

### A9. Verify in the browser, then break it

**On screen:** The artifact table. First every hash is checked, then one byte is corrupted.

**The page does this for you:** clicks `#btnVerify`, then `#btnCorrupt`.

**Say:** Every input and output file has a SHA-256 hash in a signed manifest. This button recomputes each hash in your browser. Now I corrupt one byte of the submitted file in memory and check again. Verification fails for exactly that file and for nothing else.

### A10. Signature and audit chain

**On screen:** Signature facts, then the audit chain table.

**Say:** The whole study is signed with ML-DSA-65, a NIST post-quantum signature standard, and every step is recorded in a hash-chained log of 27 entries, each signed. The browser can recompute hashes but not check this signature. The command line can. Switch to the terminal now.

**Then switch to the terminal for Part B.**

### A11. Limits and next steps

**On screen:** The limits list.

**Say:** What this does not show. No hardware run: everything is a simulator. A 60-patient holdout. We cannot verify from outside that E5 ran on Bloq. The post-quantum libraries are unaudited demonstration code. And this is research tooling, not a diagnostic tool. Next steps: run the no-time comparison on the platform, lock a holdout, and try hardware.

## Part B: the terminal

### B1. Re-run the submitted model from scratch

```powershell
qneuro study reproduce
```

**Say:** I rebuilt the submitted QSVM with my own code, not the platform's. The five class-based metrics match the submitted file to nine decimals and the confusion matrix is identical. A true ZZ kernel on the same inputs scores differently, so the submitted kernel has no entanglement.

### B2. Verify signature, hashes and audit chain

```powershell
qneuro study verify
```

**Say:** Six checks: the ML-DSA-65 signature on the study, the study hash, all 18 artifact hashes against the signed manifest, the audit chain, the sealing entry, and the published copy.

### B3. Break it on purpose

```powershell
qneuro study tamper-demo
```

**Say:** I change one digit in a copy of the submitted file and one digit in a copy of the signed study. Both edits must be detected. The untouched control must still pass.

### B4. Show the audit chain

```powershell
qneuro audit --audit-file runs/morphos/audit.jsonl
```

**Say:** Every step of the analysis is a signed entry linked to the previous one by hash. Editing any entry breaks every link after it.

Or run all four with pauses: `.\scripts\demo.ps1`.

## If something goes wrong

| Problem | What to do |
| --- | --- |
| The page shows 'Could not load the study data' | You opened the file directly. Use `.\scripts\serve.ps1` or the Vercel address. |
| Hash buttons say hashing needs https | Use localhost or the Vercel address, not a file path or a plain-http IP. |
| `qneuro study verify` fails | Run `qneuro study build` once, then verify again. The public key is created on your machine on first build. |
| Browser demo window does not open | Run `python -m playwright install chromium`, or present manually with the arrow keys. |

## Questions you may get

**What is a quantum kernel SVM, in one breath?**

Each patient's four numbers are turned into the state of four qubits. The kernel is how much two patients' states overlap. A normal SVM then uses that overlap as its similarity measure. Here the qubits are simulated on an ordinary computer.

**Why MCC and not accuracy?**

About 32% of patients died, so a model that says 'survived' for everyone is 68% accurate. E1 did exactly that: accuracy 0.6875, MCC 0.000. MCC is 0 for no skill and 1 for perfect, whatever the class balance.

**Why did the Bloq runs E1 and E2 fail?**

I do not know. My sweep shows the ZZ kernel collapses when inputs are stretched to 0 to 2 pi. But with inputs in 0 to 1 the local ZZ kernel never predicted 'no deaths' in 300 random splits. So I call the range explanation a hypothesis. Whether the platform rescales inputs internally cannot be checked from outside.

**Is the quantum model better than the classical ones?**

Not demonstrably. On 15 paired splits the best quantum kernel and the best classical model are tied, 0.600 against 0.599 MCC. The submitted holdout MCC of 0.746 is on 60 patients, where flipping one prediction moves it between 0.702 and 0.790.

**You edited the submitted file. Why?**

Three text fields: the feature-map name, a leakage note and a verdict sentence. Every number is identical, and the diff is on screen. The original note said time inflates MCC across models, which the file never tested, and it named a ZZ map although the kernel that reproduces the numbers has no entanglement.

**How do you know the submitted kernel is not ZZ?**

An independent angle-embedding kernel reproduces the submitted holdout metrics exactly, to nine decimals, with the same confusion matrix. A true ZZ kernel at the same input range gives holdout MCC 0.672. Two kernels, one match.

**Can you prove the run happened on Bloq?**

No, and the tool says so. It can sign and hash what I give it. It cannot attest to where a number came from. I keep the experiment ids and screenshots from the platform.

**Why does one cross-validation fold not reproduce?**

Fold 4 is 0.617 in my run and 0.633 in the file, a gap of 0.016 MCC. The other four folds and the holdout match exactly. I do not know the cause, and the claim table marks it as partly reproduced.

**The tree-model baselines do not reproduce either?**

Right. Logistic regression and the RBF SVM in the submitted file reproduce exactly, logistic regression only when inputs are multiplied by 0.3, which is why it collapsed to the majority class. RandomForest, GradientBoosting and XGBoost do not reproduce with default settings and seed 42. The platform's settings for them are not visible to me.

**What is wrong with the time feature?**

It is how long a patient was followed, and patients who die are followed for less time. Alone it scores AUROC 0.812, higher than any other single feature. Replacing it with serum sodium lowers MCC for every model by 0.164 to 0.229. The track allows it, but a model that needs it cannot predict ahead.

**Why a corrected t-test and why Holm?**

Cross-validation folds share training patients, so a plain paired t-test is overconfident. The Nadeau-Bengio correction widens the variance. Holm corrects for trying three quantum variants, because picking the best of several looks better than it is.

**What is ML-DSA and why use it here?**

A NIST post-quantum signature standard, FIPS 204. Results about patients stay relevant for decades, and signatures made today with classical algorithms could be forged later by a quantum computer. The library is a pure-Python reference implementation, unaudited, so this shows the pattern and is not production security.

**Would this run on real quantum hardware?**

Not tried. A hardware kernel needs one circuit per pair of patients. For 239 training patients that is 239 x 238 / 2 = 28,441 circuits for the training matrix and 60 x 239 = 14,340 more for the holdout, 42,781 in total, each repeated for shots. The simulator needs one state per patient and a matrix product.

**Is this usable clinically?**

No. It is research tooling on a public dataset of 299 patients from one pair of hospitals in Faisalabad, Pakistan. It is not a diagnostic tool.

**What would you do next?**

Run the no-time comparison on the Bloq platform, lock a holdout before choosing anything, find out whether the platform rescales inputs, and run a small hardware job.

## Do not say

- That the quantum model beats classical models. The guard says tied.
- That the run is verified on Bloq. Only the files are verified.
- That the collapse of E1 and E2 is explained. It is a hypothesis.
- That this is a clinical tool.
