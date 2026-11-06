"""Regenerate the retained W2 audit tables and check their numerical rows."""
from pathlib import Path
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/w2_audit'
OUTPUT = ROOT / 'build/reproduced_tables/w2_audit'
inputs = [BASE / 'waterbirds_frozen_training.json']
for directory in ('diagnostics', 'end_to_end', 'training_ridge_sensitivity'):
    inputs.extend(sorted((BASE / directory).glob('*.json')))
subprocess.run([sys.executable, str(ROOT / 'experiments/make_ridge_exact_audit_report.py'),
                *map(str, inputs), '--output-dir', str(OUTPUT)], check=True)
expected = json.loads((BASE / 'report/ridge_exact_audit.json').read_text())
actual = json.loads((OUTPUT / 'ridge_exact_audit.json').read_text())

def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in ('source', 'source_file')}
    if isinstance(value, list):
        return sorted([normalize(v) for v in value], key=lambda v: json.dumps(v, sort_keys=True))
    return value

def compare(left,right,path=''):
    if isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left:compare(left[key],right[key],path+'/'+key)
    elif isinstance(left,list):
        assert len(left)==len(right),path
        for index,(a,b) in enumerate(zip(left,right)):compare(a,b,path+'/'+str(index))
    elif isinstance(left,(int,float)) and not isinstance(left,bool):
        if not math.isclose(left,right,rel_tol=2e-12,abs_tol=2e-13):raise AssertionError(path)
    elif left != right:raise AssertionError(path)

compare(normalize(actual),normalize(expected))
print(f"Verified {len(actual['runs'])} W2 audit rows and {len(actual['summary'])} aggregate rows.")
