# Terminal half of the professor demo. Press Enter to run each step; the SAY line is what to explain.
#   powershell -ExecutionPolicy Bypass -File .\scripts\demo.ps1            (terminal steps)
#   powershell -ExecutionPolicy Bypass -File .\scripts\demo.ps1 -Browser   (then the browser walkthrough)
param([double]$Pace = 1.0, [switch]$Browser)
$q = ".\.venv\Scripts\qneuro.exe"
$py = ".\.venv\Scripts\python.exe"

function Step($title, $say, $cmd) {
    Write-Host ""
    Write-Host "=== $title" -ForegroundColor Green
    Write-Host "SAY: $say"
    Read-Host "Press Enter to run:  $cmd" | Out-Null
    Invoke-Expression $cmd
}

Step "1. Re-run the submitted model from scratch" `
     "I rebuilt the submitted QSVM with my own code, not the platform's. The five class-based metrics match the submitted file to nine decimals, and the confusion matrix is identical. A true ZZ kernel on the same inputs scores differently, so the submitted kernel has no entanglement." `
     "$q study reproduce"

Step "2. Verify the signature, every file hash and the audit chain" `
     "Six checks: the ML-DSA-65 signature on the study, the study hash, all 18 artifact hashes against the signed manifest, the audit chain, the sealing entry, and the published copy." `
     "$q study verify"

Step "3. Break it on purpose" `
     "I change one digit in a copy of the submitted file and one digit in a copy of the signed study. Both edits must be detected. The control copy must still pass." `
     "$q study tamper-demo"

Step "4. Show the audit chain" `
     "Every step of the analysis is a signed entry linked to the previous one by hash. Editing any entry breaks every link after it." `
     "$q audit --audit-file runs/morphos/audit.jsonl"

if ($Browser) {
    Step "5. Browser walkthrough" `
         "Now the same study as a page. The narration at the bottom is what I will say. Arrow keys move between steps." `
         "$py demo\run_demo.py --manual"
}
