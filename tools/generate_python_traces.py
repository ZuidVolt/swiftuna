import json
import sys
from pathlib import Path

# Check if optuna or rustuna is available
try:
    import rustuna as optuna  # ty: ignore[unresolved-import]
except ImportError:
    try:
        import optuna
    except ImportError:
        print(
            "Note: Neither 'rustuna' nor 'optuna' is currently installed in this python env."
        )
        print("To install, run: uv pip install optuna")
        sys.exit(0)

out_dir = Path("Tests/Fixtures/ParityCorpus")
out_dir.mkdir(parents=True, exist_ok=True)


# 1. Quadratic
def run_quadratic():
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    traces = []

    for i in range(15):
        trial = study.ask()
        x = trial.suggest_float("x", -10.0, 10.0)
        y = trial.suggest_float("y", -10.0, 10.0)
        loss = (x - 2.0) ** 2 + (y + 5.0) ** 2
        study.tell(trial, loss)
        traces.append(
            {"number": trial.number, "params": {"x": x, "y": y}, "value": loss}
        )

    return {
        "problem_name": "quadratic_python",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
    }


py_quad = run_quadratic()
with open(out_dir / "quadratic_python.json", "w") as f:
    json.dump(py_quad, f, indent=2)

print(
    "✅ Python parity trace generated at Tests/Fixtures/ParityCorpus/quadratic_python.json"
)


# 2. CMA-ES Mathematical Parity Trace
def run_cmaes():
    try:
        import cmaes
        import numpy as np
    except ImportError:
        print("Note: cmaes or numpy not found. Skipping cmaes trace.")
        return None

    cma = cmaes.CMA(mean=np.array([0.5, 0.5]), sigma=0.2, seed=42)
    generations_data = []

    for gen in range(3):
        solutions = []
        for i in range(cma.population_size):
            pt = np.array([0.4 + 0.05 * i + 0.02 * gen, 0.6 - 0.04 * i - 0.01 * gen])
            val = float((pt[0] - 0.2) ** 2 + (pt[1] - 0.8) ** 2)
            solutions.append((pt, val))

        cma.tell(solutions)

        generations_data.append(
            {
                "generation": gen + 1,
                "solutions": [
                    {"point": s[0].tolist(), "value": s[1]} for s in solutions
                ],
                "expected_mean": cma._mean.tolist(),
                "expected_sigma": float(cma._sigma),
                "expected_pc": cma._pc.tolist(),
                "expected_p_sigma": cma._p_sigma.tolist(),
                "expected_cov": cma._C.flatten().tolist(),
            }
        )

    return {
        "problem_name": "cmaes_step",
        "seed": 42,
        "initial_mean": [0.5, 0.5],
        "initial_sigma": 0.2,
        "generations": generations_data,
    }


cma_trace = run_cmaes()
if cma_trace:
    with open(out_dir / "cmaes_step.json", "w") as f:
        json.dump(cma_trace, f, indent=2)
    print(
        "✅ CMA-ES parity trace generated at Tests/Fixtures/ParityCorpus/cmaes_step.json"
    )
