"""GPU-aware queue for the 5M-step multi-seed MLP runs.

Each job is one (d_latent, seed) pair. The queue waits for a GPU with enough
free memory before launching, so it can sit on busy shared machines without
stepping on existing jobs.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_jobs(tokens):
    jobs = []
    for token in tokens:
        dlat, seed = token.split(":")
        jobs.append((int(dlat), int(seed)))
    return jobs


def metrics_complete(run_dir, target_steps):
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return False
    last_step = None
    with metrics.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in row:
                last_step = row["step"]
    return last_step is not None and last_step >= target_steps


def job_is_running(job, args):
    dlat, seed = job
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_mlp_multiseed.py"],
                                      text=True)
    except subprocess.CalledProcessError:
        return False
    needles = [
        f"--d-latents {dlat}",
        f"--seeds {seed}",
        f"--base-dir {args.base_dir}",
    ]
    return any(all(needle in line for needle in needles)
               for line in out.splitlines())


def gpu_status(allowed_gpus=None):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    stats = []
    for line in out.splitlines():
        idx, used, total, util = [x.strip() for x in line.split(",")]
        gpu = {
            "index": int(idx),
            "used": int(used),
            "total": int(total),
            "util": int(util),
        }
        if allowed_gpus is None or gpu["index"] in allowed_gpus:
            stats.append(gpu)
    return stats


def launch_job(job, gpu, args):
    dlat, seed = job
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"d{dlat}_s{seed}.log"
    cmd = [
        args.python,
        "run_mlp_multiseed.py",
        "--base-dir", args.base_dir,
        "--d-latents", str(dlat),
        "--steps", str(args.steps),
        "--eval-interval", str(args.eval_interval),
        "--n-gen", str(args.n_gen),
        "--seeds", str(seed),
    ]
    if args.hidden is not None:
        cmd += ["--hidden", str(args.hidden)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            env=env)
    return {
        "job": job,
        "gpu": gpu,
        "proc": proc,
        "log_file": log_f,
        "log_path": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True,
                        help="Jobs as d_latent:seed, e.g. 20:44")
    parser.add_argument("--base-dir",
                        default="multiseed_runs/exp2_mlp_dlat_sn05_5m")
    parser.add_argument("--log-dir", default="multiseed_5m_logs")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--n-gen", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--min-free-mb", type=int, default=55_000)
    parser.add_argument("--max-util", type=int, default=10)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs to use.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--python", default="/home/trevor/genmol-app/.venv/bin/python")
    parser.add_argument("--seed-barrier", action="store_true",
                        help="Finish all jobs for one seed before launching the next seed.")
    args = parser.parse_args()
    args.gpus = (
        {int(x) for x in args.gpus.split(",") if x.strip()}
        if args.gpus is not None else None
    )

    jobs = parse_jobs(args.jobs)
    running = []
    print(f"queue start jobs={jobs}", flush=True)
    while jobs or running:
        still_running = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                still_running.append(item)
            else:
                item["log_file"].close()
                print(f"finished job={item['job']} gpu={item['gpu']} code={code}",
                      flush=True)
        running = still_running

        jobs_per_gpu = {}
        for item in running:
            jobs_per_gpu[item["gpu"]] = jobs_per_gpu.get(item["gpu"], 0) + 1
        free_gpus = []
        for stat in gpu_status(args.gpus):
            free = stat["total"] - stat["used"]
            gpu_jobs = jobs_per_gpu.get(stat["index"], 0)
            if (gpu_jobs < args.max_jobs_per_gpu
                    and free >= args.min_free_mb
                    and stat["util"] <= args.max_util):
                free_gpus.extend(
                    [stat["index"]] * (args.max_jobs_per_gpu - gpu_jobs)
                )

        launchable_jobs = jobs
        if args.seed_barrier and jobs:
            current_seed = jobs[0][1]
            launchable_jobs = [job for job in jobs if job[1] == current_seed]

        while launchable_jobs and free_gpus:
            job = launchable_jobs.pop(0)
            dlat, seed = job
            run_dir = Path(args.base_dir) / f"di5_d{dlat}_n500_s{seed}"
            if metrics_complete(run_dir, args.steps):
                print(f"skip complete job={job}", flush=True)
                jobs.remove(job)
                continue
            if job_is_running(job, args):
                print(f"wait running job={job}", flush=True)
                continue
            gpu = free_gpus.pop(0)
            item = launch_job(job, gpu, args)
            running.append(item)
            jobs.remove(job)
            print(f"launched job={job} gpu={gpu} log={item['log_path']}",
                  flush=True)

        if args.seed_barrier and jobs:
            current_seed = jobs[0][1]
            current_jobs = [job for job in jobs if job[1] == current_seed]
            for job in list(current_jobs):
                dlat, seed = job
                run_dir = Path(args.base_dir) / f"di5_d{dlat}_n500_s{seed}"
                if metrics_complete(run_dir, args.steps):
                    print(f"skip complete job={job}", flush=True)
                    jobs.remove(job)

        state = {
            "queued": jobs,
            "running": [
                {"job": item["job"], "gpu": item["gpu"],
                 "pid": item["proc"].pid, "log": item["log_path"]}
                for item in running
            ],
        }
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.log_dir) / "queue_state.json").write_text(
            json.dumps(state, indent=2)
        )
        if jobs or running:
            time.sleep(args.poll_seconds)
    print("queue complete", flush=True)


if __name__ == "__main__":
    main()
