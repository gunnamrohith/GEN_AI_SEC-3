WEEK 6 - GOLDEN QUESTION SET AND BASELINE EVALUATION
Release: v0.1

LAYOUT
------
code/
  main.py, test_main.py, golden_set.json, requirements.txt, .gitignore
week_documentation/
  Week_6_GenAI_Lab_Documentation.docx
  baseline_results.csv
  baseline_run_v0.1.txt
  README.txt

RUN FROM THE REPOSITORY ROOT (macOS/Linux)
-----------------------------------------
cd WEEK_EXPERIMENTS/WEEK-6/code
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m unittest -v
./.venv/bin/python main.py

Install Ollama separately if needed. If Ollama is not already running, start it
in another terminal with: ollama serve
If the model is missing, download it once with: ollama pull llama3.2
No hosted API key or paid hosted API is needed.

The program evaluates the 30 supplied questions against local llama3.2.
Input and generated CSV paths are resolved beside main.py. A new run writes
code/baseline_results.csv; it does not overwrite the archived CSV or transcript
in week_documentation/. No local virtual environments are included in Git.

SAVED BASELINE
--------------
The initial transcript records:
30 successful responses, 12 PASS, 18 REVIEW, 0 failed requests.
Average keyword coverage: 47.22%.
Recorded evaluation date: 2026-11-18T14:28:27+05:30.
The transcript and CSV are preserved as supplied from the local Week 6 folder.
IMPORTANT: the current CSV is from a later failed run: all 30 rows have ERROR
status and no answer text. It does NOT represent the successful initial run
recorded in baseline_run_v0.1.txt. It is included transparently as the current
local output, not as evidence of a successful baseline. Run again with Ollama
and llama3.2 available to generate a new CSV; archive that run separately.

SCORING AND LIMITATIONS
-----------------------
Score = matched keywords / configured keywords * 100, rounded to 2 decimals.
Matching is case-insensitive substring matching; PASS is score >= 60%.
An empty keyword list scores 0. Keyword coverage is not factual accuracy.
Reference answers and sources are provided for manual review, not sent to the
model. Source URLs are not retrieved or independently verified.
Requests use temperature=0 and a 120-second timeout. Failed requests are recorded
as ERROR, excluded from the successful-response average, and cause exit 1.
REVIEW alone does not fail execution. Exit 0 does not imply factual correctness.
Model replies can vary across machines, versions and runs.

VALIDATION BEFORE PUBLICATION
------------------------------
python -m unittest -v: 9 tests passed, 0 failures.
Tests simulate model replies and do not require a live Ollama server.
The saved real baseline was not rerun or replaced during publication.
