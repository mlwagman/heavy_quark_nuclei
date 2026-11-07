"""Release dead claims so that preempted or killed lines are rerun.

A claim claims/NAME is dead when results/NAME.npz does not exist and
  - it has a heartbeat file 'alive' older than HB minutes (the worker touches it every 2 min), or
  - it has no heartbeat (claimed by the pre-heartbeat script) and its 'who' file is older than OLD minutes.
Dead claims and their runs/NAME directories are removed.  Usage (from ~/hqn/cfree): python cleanup_claims.py [HB=10] [OLD=60]
"""
import os, sys, time, shutil
HB = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
OLD = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
now = time.time(); n = 0
for name in os.listdir('claims'):
    if os.path.exists(f'results/{name}.npz'):
        continue
    try:   # a line that failed three times keeps its claim
        if int(open(f'logs/retry/{name}').read()) >= 3: continue
    except (FileNotFoundError, ValueError):
        pass
    c = f'claims/{name}'
    try:
        if os.path.exists(f'{c}/alive'):
            dead = now - os.path.getmtime(f'{c}/alive') > 60*HB
        else:
            ref = f'{c}/who' if os.path.exists(f'{c}/who') else c
            dead = now - os.path.getmtime(ref) > 60*OLD
    except FileNotFoundError:
        continue
    if dead:
        shutil.rmtree(f'runs/{name}', ignore_errors=True)
        shutil.rmtree(c, ignore_errors=True)
        n += 1
print(f'cleanup: released {n} dead claims', flush=True)
