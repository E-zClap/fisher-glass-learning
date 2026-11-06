"""Run the five-seed synthetic experiment and the existing native aggregator.

Default settings reproduce the declared paper protocol. --smoke reduces only
the simulated sample sizes and marks the output as a smoke test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'build/fresh_reproduction/synthetic')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_env, n_per_env = (2000, 32) if args.smoke else (300000, 256)
    runs = []
    for seed in range(11, 16):
        folder = out / f'seed_{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / 'experiments/synthetic_fisher_glass.py'),
                   '--seed', str(seed), '--n-env', str(n_env), '--n-per-env', str(n_per_env),
                   '--gamma', '1', '--delta-max', '.35', '--alpha', '.95', '--eps', '1e-12',
                   '--logit-scale', '4', '--reserve-r', '.05', '--anchor-delta', '.5',
                   '--out-dir', str(folder), '--fig-dir', str(folder / 'figures')]
        print(f'Running synthetic seed {seed}: {n_env} environments, {n_per_env} observations per environment.', flush=True)
        with (folder / 'run.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs.append(json.loads((folder / 'synthetic_fisher_glass_results.json').read_text()))
    path = ROOT / 'experiments/multiseed_fisher_glass_diagnostics.py'
    spec = importlib.util.spec_from_file_location('fisherglass_multiseed_aggregation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = dict(protocol='smoke' if args.smoke else 'paper synthetic protocol', seeds=list(range(11,16)),
                  n_env=n_env, n_per_env=n_per_env,
                  interval_convention='nominal normal approximation: 1.96 * sample SD / sqrt(5)',
                  synthetic=module.aggregate_synthetic(runs),
                  per_seed_results=[f'seed_{seed}/synthetic_fisher_glass_results.json' for seed in range(11,16)])
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
