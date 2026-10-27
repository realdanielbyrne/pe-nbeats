#!/usr/bin/env python3
"""GPU-saturating worker autoscaler for run_from_yaml.py studies.

Launches claim-file-coordinated workers across GPUs and keeps adding one
worker at a time to whichever GPU has headroom, until either the GPU is
saturated (utilization/VRAM thresholds) or a per-GPU worker cap is hit.
Backs off permanently on any observed CUDA OOM. Exits when every study's
workers have drained (a worker whose study has no unclaimed jobs left
exits on its own, and no new one is spawned for a drained study).

Usage:
  python experiments/scripts/autoscale_workers.py \
      --study experiments/configs/a.yaml --study experiments/configs/b.yaml \
      [--gpus 0,1] [--max-per-gpu 5] [--util-target 85] [--vram-floor-mb 8000] \
      [--interval 180] [--log-dir ~/nbeats_worker_logs] [--seed-topology auto]

Stability rules (conservative by design):
  - never spawn if free VRAM on the GPU is below --vram-floor-mb
  - never spawn if the GPU's utilization is already >= --util-target
  - at most one spawn per GPU per polling interval (lets memory settle)
  - a study observed to OOM lowers that GPU's cap to its current live count
  - crashed workers (nonzero exit, no OOM signature) are respawned at most
    twice per (study, gpu) to avoid crash loops
"""

import argparse
import itertools
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "experiments" / "run_from_yaml.py"


def gpu_stats(gpus):
    """Return {gpu: (util_pct, mem_used_mb, mem_total_mb)} via nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    stats = {}
    for line in out.strip().splitlines():
        idx, util, used, total = [int(x.strip()) for x in line.split(",")]
        if idx in gpus:
            stats[idx] = (util, used, total)
    return stats


class Worker:
    _ids = itertools.count()

    def __init__(self, study, gpu, log_dir):
        self.study = study
        self.gpu = gpu
        self.wid = f"auto-g{gpu}-{next(self._ids)}"
        self.log = log_dir / f"{Path(study).stem}__{self.wid}.log"
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(LAUNCHER),
                study,
                "--gpu-id",
                str(gpu),
                "--worker-id",
                self.wid,
            ],
            stdout=open(self.log, "a"),
            stderr=subprocess.STDOUT,
            cwd=REPO,
            start_new_session=True,  # survive autoscaler death
        )
        print(f"[spawn] {self.wid} study={Path(study).stem} gpu={gpu} pid={self.proc.pid}", flush=True)

    def alive(self):
        return self.proc.poll() is None

    def oomed(self):
        try:
            tail = self.log.read_text(errors="ignore")[-20000:]
        except OSError:
            return False
        return bool(re.search(r"CUDA out of memory|OutOfMemoryError", tail))

    def drained(self):
        """Exited cleanly (rc==0) — study has no unclaimed work for it."""
        return self.proc.poll() == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", action="append", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--max-per-gpu", type=int, default=5)
    ap.add_argument("--util-target", type=int, default=85)
    ap.add_argument("--vram-floor-mb", type=int, default=8000)
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--log-dir", default=str(Path.home() / "nbeats_worker_logs"))
    ap.add_argument(
        "--initial",
        default=None,
        help="comma list of study_idx:gpu pairs for the first wave, e.g. 0:0,0:0,1:1,2:0,3:1",
    )
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    studies = args.study
    cap = {g: args.max_per_gpu for g in gpus}
    crash_count = defaultdict(int)  # (study, gpu) -> crashes
    drained_studies = set()
    workers: list[Worker] = []

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # First wave
    if args.initial:
        for pair in args.initial.split(","):
            si, g = [int(x) for x in pair.split(":")]
            workers.append(Worker(studies[si], g, log_dir))
    else:
        for i, study in enumerate(studies):
            workers.append(Worker(study, gpus[i % len(gpus)], log_dir))

    rr = itertools.cycle(range(len(studies)))
    while True:
        time.sleep(args.interval)
        # Reap
        for w in list(workers):
            if w.alive():
                continue
            workers.remove(w)
            if w.drained():
                print(f"[drain] {w.wid} exited 0 — no unclaimed work in {Path(w.study).stem}", flush=True)
                drained_studies.add(w.study)
            elif w.oomed():
                live_on_gpu = sum(1 for x in workers if x.gpu == w.gpu)
                cap[w.gpu] = max(1, live_on_gpu)
                print(f"[oom] {w.wid} — capping gpu {w.gpu} at {cap[w.gpu]} workers", flush=True)
            else:
                crash_count[(w.study, w.gpu)] += 1
                if crash_count[(w.study, w.gpu)] <= 2:
                    print(f"[respawn] {w.wid} crashed (rc={w.proc.returncode}), retrying", flush=True)
                    workers.append(Worker(w.study, w.gpu, log_dir))
                else:
                    print(f"[giveup] {Path(w.study).stem} on gpu {w.gpu}: 3 crashes — see {w.log}", flush=True)

        active_studies = [s for s in studies if s not in drained_studies]
        if not workers and not active_studies:
            print("[done] all studies drained", flush=True)
            return
        if not active_studies:
            continue  # just wait for the last workers

        # Scale up: at most one spawn per GPU per tick
        stats = gpu_stats(gpus)
        for g in gpus:
            util, used, total = stats[g]
            live = sum(1 for w in workers if w.gpu == g)
            if live >= cap[g] or util >= args.util_target or (total - used) < args.vram_floor_mb:
                continue
            # pick the active study with the fewest live workers (round-robin tiebreak)
            for _ in range(len(studies)):
                si = next(rr)
                if studies[si] in active_studies:
                    counts = {s: sum(1 for w in workers if w.study == s) for s in active_studies}
                    target = min(active_studies, key=lambda s: counts[s])
                    workers.append(Worker(target, g, log_dir))
                    break
        print(
            f"[tick] workers={len(workers)} "
            + " ".join(f"gpu{g}:util={stats[g][0]}%,used={stats[g][1]}MB,n={sum(1 for w in workers if w.gpu == g)}" for g in gpus),
            flush=True,
        )


if __name__ == "__main__":
    main()
