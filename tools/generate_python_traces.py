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
