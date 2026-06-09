#!/usr/bin/env python
"""
run_all.py: End-to-end pipeline for the ocular TMDD case study.

Runs every Python analysis in dependency order, producing all plots and
tables in results/.

Usage:
    pip install -r requirements.txt
    python run_all.py

Expected runtime: ~15 min on a 4-core workstation.
The Bayesian Phase C (ABC-SMC) is the longest step (~1-2 min).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

PIPELINE = [
    ('00_generate_synthetic_data.py',
     'Generate synthetic IVT dataset for identifiability analyses'),
    ('03_calibration.py',
     'Verify k_va values reproduce Park 2016 rabbit half-lives'),
    ('04_sensitivity_sobol.py',
     'Sobol global sensitivity (~896 evals)'),
    ('05_dose_response.py',
     'Equal-mg vs equimolar comparison (the headline plot)'),
    ('06_human_translation.py',
     'Cross-species + human dose-response'),
    ('07_koff_sweep.py',
     'Binding kinetics: internalization-limited PD'),
    ('08_identifiability_sian.py',
     'Phase A: structural identifiability via Lie derivatives'),
    ('09_identifiability_profile.py',
     'Phase B: profile likelihood (4 key parameters)'),
    ('10_identifiability_abc.py',
     'Phase C: ABC-SMC Bayesian inference (~1-2 min)'),
]


def ensure_dirs():
    for d in ['results/plots', 'results/tables', 'results/posteriors', 'data']:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()
    overall_start = time.time()
    results = []

    for script, desc in PIPELINE:
        path = ROOT / 'analyses' / script
        print('\n' + '='*72)
        print(f"Running {script}")
        print(f"  Purpose: {desc}")
        print('='*72)
        t0 = time.time()
        try:
            subprocess.run([sys.executable, str(path)], check=True, cwd=ROOT)
            elapsed = time.time() - t0
            results.append((script, 'OK', elapsed))
            print(f"  -> Completed in {elapsed:.1f}s")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            results.append((script, f'FAILED ({e.returncode})', elapsed))
            print(f"  -> Failed after {elapsed:.1f}s")
            print(f"  -> Continuing with remaining scripts.")

    total = time.time() - overall_start

    print('\n' + '='*72)
    print("Pipeline summary")
    print('='*72)
    for script, status, elapsed in results:
        marker = '[OK]' if status == 'OK' else '[FAIL]'
        print(f"  {marker} {script:<35s} {elapsed:>7.1f}s  {status}")
    print(f"\nTotal runtime: {total/60:.1f} minutes")
    print(f"\nOutputs in results/: plots/, tables/, posteriors/")
    print(f"See README.md for interpretation.")

    return 0 if all(r[1] == 'OK' for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
