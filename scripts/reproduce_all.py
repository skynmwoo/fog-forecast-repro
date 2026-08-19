# -*- coding: utf-8 -*-
"""One-command reproduction of every number, table and figure in the paper.

    python scripts/reproduce_all.py

Equivalent to running, in order:
    scripts/prepare_data.py
    scripts/sanity_check_models.py      <-- hard gate; stops on failure
    scripts/run_experiments.py
    scripts/run_shap.py
    scripts/make_tables.py
    scripts/make_figures.py
    scripts/compare_with_manuscript.py
    scripts/audit_consistency.py

Expected wall-clock time on 2 CPU cores: ~30 minutes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("prepare_data.py", [], "raw ASOS -> processed parquet + prevalence/transition CSVs"),
    ("sanity_check_models.py", [], "structural gate: prove CNN is Conv1d and LSTM is recurrent"),
    ("run_experiments.py", [], "12 stations x 2 tasks x 4 models"),
    ("run_shap.py", [], "SHAP for the reproduced overall XGBoost models"),
    ("significance_tests.py", [], "paired Wilcoxon tests across the twelve stations"),
    ("transition_analysis.py", [], "decompose overall skill by transition type; prevalence gap"),
    ("make_tables.py", [], "all manuscript tables from result CSVs"),
    ("make_figures.py", [], "all manuscript figures from result CSVs"),
    ("compare_with_manuscript.py", [], "old vs. reproduced comparison"),
    ("audit_consistency.py", [], "manuscript / config / result consistency audit"),
]


def main() -> int:
    t0 = time.time()
    for i, (script, extra, desc) in enumerate(STEPS, 1):
        print(f"\n{'='*72}\n[{i}/{len(STEPS)}] {script} — {desc}\n{'='*72}", flush=True)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", script), *extra]
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"\nSTEP FAILED: {script} (exit {rc}) — stopping.")
            return rc
    print(f"\nAll steps completed in {(time.time()-t0)/60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
