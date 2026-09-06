import os
import resource
import sys
import time

import optuna  # ty: ignore[unresolved-import]

optuna.logging.set_verbosity(optuna.logging.WARNING)

D = 50
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = 42


def rosenbrock(xs):
    total = 0.0
    for i in range(len(xs) - 1):
        total += 100.0 * (xs[i + 1] - xs[i] ** 2) ** 2 + (1.0 - xs[i]) ** 2
    return total


def objective(trial):
    xs = [trial.suggest_float(f"x_{i}", -5.0, 5.0) for i in range(D)]
    return rosenbrock(xs)


def main():
    print("=" * 60)
    print(f" Python Optuna CMA-ES Benchmark: D={D}, N={N_TRIALS} Trials")
    print("=" * 60)

    sampler = optuna.samplers.CmaEsSampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    elapsed = time.perf_counter() - t0

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = (
        rusage.ru_maxrss / (1024 * 1024)
        if os.uname().sysname == "Darwin"
        else rusage.ru_maxrss / 1024
    )

    trials_per_sec = N_TRIALS / elapsed
    us_per_trial = (elapsed * 1_000_000.0) / N_TRIALS

    print(f"Elapsed Time:       {elapsed:.3f} s")
    print(f"Throughput:         {trials_per_sec:.1f} trials/s")
    print(f"Latency per trial:  {us_per_trial:.2f} µs")
    print(f"Best Value:         {study.best_value:.6e}")
    print(f"Peak RSS:           {peak_rss_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
