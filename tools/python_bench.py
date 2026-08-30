import sys
import time

import rustuna as optuna  # ty: ignore[unresolved-import]

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)

start = time.perf_counter()

for _ in range(n_trials):
    trial = study.ask()
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_float("y", -10.0, 10.0)
    loss = (x - 2.0) ** 2 + (y + 5.0) ** 2
    study.tell(trial, loss)

elapsed = time.perf_counter() - start
total_ms = elapsed * 1000.0
us_per_trial = (elapsed * 1_000_000.0) / n_trials
throughput = n_trials / elapsed

print(f"RESULT:{total_ms:.3f}:{us_per_trial:.2f}:{throughput:.0f}")
