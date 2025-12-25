#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

nb_path = Path('executed_wheel_momentum.ipynb')
if not nb_path.exists():
    print('Error: executed_wheel_momentum.ipynb not found. Run nbconvert first.', file=sys.stderr)
    sys.exit(2)

with nb_path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

stream_text = ''
for cell in nb.get('cells', []):
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'stream':
            stream_text += out.get('text', '')

lines = [L for L in stream_text.splitlines() if re.match(r'\d+\.\d+\t', L)]
if not lines:
    print('No unit-check lines found in executed notebook output.', file=sys.stderr)
    sys.exit(2)

diffs = []
parsed = []
for L in lines:
    parts = re.split(r'\s+', L.strip())
    if len(parts) < 7:
        continue
    try:
        t = float(parts[0])
        accel = float(parts[1])
        req_torque = float(parts[2])
        allowed_calc = float(parts[3])
        delta_exp = float(parts[4])
        delta_act = float(parts[5])
        curr_mom = float(parts[6])
    except Exception as e:
        print('Parse error on line:', L, file=sys.stderr)
        continue
    diffs.append(abs(delta_exp - delta_act))
    parsed.append((t, accel, req_torque, allowed_calc, delta_exp, delta_act, curr_mom))

maxdiff = max(diffs) if diffs else float('nan')
print(f'Parsed {len(parsed)} unit-check lines; max |delta_exp - delta_act| = {maxdiff:.6e}')

# Print a few sample lines
print('\nSample rows:')
for row in parsed[:10]:
    print('\t'.join(f'{v:.6f}' for v in row))

THRESH = 1e-6
if maxdiff <= THRESH:
    print('\nAutomated check: PASS')
    sys.exit(0)
else:
    print('\nAutomated check: FAIL (difference exceeds threshold)')
    sys.exit(3)
