Team Morphos - Hackathon M1 (Heart Failure QSVM) - file bundle
scripts/         prepare_hf.py, local_qsvm.py (angle embedding), sweep1/sweep2/zz_benchmark (local ZZ simulation)
data/original/   MinMax-scaled train/test CSVs (notime, withtime)
data/scaled_0.3/ same data x0.3 (test clipped to [0,1] first)
bloq_outputs/    E1 (notime) and E2 (withtime) raw Bloq config runs; E2 was mistitled "notime"
submissions/     SUBMITTED_corrected.json is the file that was submitted; E6 is a local run, not submitted
Scripts read their CSVs from the current directory; copy the CSVs next to a script before running it.
Full write-up: the Project Report & PRD document.
