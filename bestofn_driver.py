"""Best-of-N replicates of the MLP recipe — mirrors the KAN project's protocol.

The KAN champion is characterised by 3 replicates (`bestofn_v9_driver.py` ->
`bestofn_v9_summary.txt`: valid R2 0.8297 +/- 0.0040, test R2 0.8472 +/- 0.0257).
A single MLP run therefore cannot be compared against it — retrain draw noise on
the 8-sim test bucket spans ~0.05 R2. This driver produces the matching 3-replicate
MLP population, each into its OWN named directory so nothing is ever overwritten
(VARIANCE_RECORD lesson: three sequential retrains once clobbered models/).

Replicate 1 is the existing `models_mlp` (from `python -u train.py`); this driver
runs replicates 2 and 3.

Run unbuffered:  python -u bestofn_driver.py
"""
import os
import time

from config import PROJECT_DIR
from train import main

TAU = 1.8
REPLICATES = ["models_mlp_r2", "models_mlp_r3"]

if __name__ == "__main__":
    t0 = time.time()
    for name in REPLICATES:
        d = os.path.join(PROJECT_DIR, name)
        print(f"\n########## RETRAIN START: {name}  (t={time.time()-t0:.0f}s) ##########",
              flush=True)
        main(model_dir=d, tau=TAU)
        print(f"########## RETRAIN DONE: {name}  (t={time.time()-t0:.0f}s) ##########",
              flush=True)
    print(f"\n########## ALL REPLICATES DONE  (total {time.time()-t0:.0f}s) ##########",
          flush=True)
