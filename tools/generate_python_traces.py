import json
import math
import sys
from pathlib import Path

# Check if optuna or rustuna is available
try:
    import rustuna as optuna  # ty: ignore[unresolved-import]
except ImportError:
    try:
        import optuna  # ty: ignore[unresolved-import]
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

    for _ in range(15):
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
        import cmaes  # ty: ignore[unresolved-import]
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


# 3. CMA-ES + MedianPruner Parity Trace
def run_cmaes_pruning():
    try:
        import cmaes  # ty: ignore[unresolved-import]
        import numpy as np
    except ImportError:
        print("Note: cmaes or numpy not found. Skipping cmaes pruning trace.")
        return None

    cma = cmaes.CMA(
        mean=np.array([0.5, 0.5]),
        sigma=1.0 / 6.0,
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        seed=42,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2, n_warmup_steps=1, interval_steps=1
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)

    completed_in_gen = []
    traces = []

    for _ in range(15):
        trial = study.ask()

        if len(completed_in_gen) >= cma.population_size:
            batch = completed_in_gen[: cma.population_size]
            cma.tell(batch)
            completed_in_gen = completed_in_gen[cma.population_size :]

        norm_pt = cma.ask()
        x = -5.0 + norm_pt[0] * 10.0
        y = -5.0 + norm_pt[1] * 10.0

        pruned = False
        reported_intermediates = {}
        for step in range(5):
            val = (x - 1.0) ** 2 + (y + 2.0) ** 2 + (4 - step) * 1.0
            trial.report(val, step)
            reported_intermediates[step] = val
            if trial.should_prune():
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                pruned = True
                break

        if pruned:
            traces.append(
                {
                    "number": trial.number,
                    "state": "PRUNED",
                    "params": {"x": float(x), "y": float(y)},
                    "value": None,
                    "intermediate_values": {
                        str(k): float(v) for k, v in reported_intermediates.items()
                    },
                }
            )
        else:
            loss = float((x - 1.0) ** 2 + (y + 2.0) ** 2)
            study.tell(trial, loss)
            completed_in_gen.append((norm_pt, loss))
            traces.append(
                {
                    "number": trial.number,
                    "state": "COMPLETE",
                    "params": {"x": float(x), "y": float(y)},
                    "value": loss,
                    "intermediate_values": {
                        str(k): float(v) for k, v in reported_intermediates.items()
                    },
                }
            )

    return {
        "problem_name": "cmaes_median_pruning",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
    }


cma_prune_trace = run_cmaes_pruning()
if cma_prune_trace:
    with open(out_dir / "cmaes_pruning.json", "w") as f:
        json.dump(cma_prune_trace, f, indent=2)
    print(
        "✅ CMA-ES + MedianPruner parity trace generated at Tests/Fixtures/ParityCorpus/cmaes_pruning.json"
    )


# 4. BruteForce Flat Parity Trace
def run_bruteforce():
    def objective(trial):
        x = trial.suggest_int("x", 0, 2)
        y = trial.suggest_float("y", 0.0, 1.0, step=0.5)
        return float((x - 1) ** 2 + (y - 0.5) ** 2)

    sampler = optuna.samplers.BruteForceSampler(seed=42)
    study = optuna.create_study(
        study_name="bruteforce_flat", direction="minimize", sampler=sampler
    )
    study.optimize(objective)

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"x": float(t.params["x"]), "y": float(t.params["y"])},
                "value": float(t.value),
            }
        )

    return {
        "problem_name": "bruteforce_flat",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


bf_trace = run_bruteforce()
if bf_trace:
    with open(out_dir / "bruteforce.json", "w") as f:
        json.dump(bf_trace, f, indent=2)
    print(
        "✅ BruteForce parity trace generated at Tests/Fixtures/ParityCorpus/bruteforce.json"
    )


# 5. BruteForce + HyperbandPruner Parity Trace
def run_bruteforce_hyperband():
    def objective(trial):
        x = trial.suggest_int("x", 0, 2)
        y = trial.suggest_float("y", 0.0, 1.0, step=0.5)
        for step in range(8):
            val = float((x - 1) ** 2 + (y - 0.5) ** 2 + (7 - step) * 0.5)
            trial.report(val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float((x - 1) ** 2 + (y - 0.5) ** 2)

    sampler = optuna.samplers.BruteForceSampler(seed=42)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=8, reduction_factor=2
    )
    study = optuna.create_study(
        study_name="bruteforce_hyperband",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective)

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"x": float(t.params["x"]), "y": float(t.params["y"])},
                "value": (
                    float(t.value)
                    if t.state == optuna.trial.TrialState.COMPLETE
                    else None
                ),
                "intermediate_values": {
                    str(k): float(v) for k, v in t.intermediate_values.items()
                },
            }
        )

    return {
        "problem_name": "bruteforce_hyperband",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


bf_hb_trace = run_bruteforce_hyperband()
if bf_hb_trace:
    with open(out_dir / "bruteforce_hyperband.json", "w") as f:
        json.dump(bf_hb_trace, f, indent=2)
    print(
        "✅ BruteForce + HyperbandPruner parity trace generated at Tests/Fixtures/ParityCorpus/bruteforce_hyperband.json"
    )


# 6. ThresholdPruner Parity Trace
def run_threshold_pruning():
    def objective(trial):
        _ = trial.suggest_float("p", 0.0, 10.0)
        n = trial.number
        for step in range(6):
            if n == 0:
                val = 5.0 - step * 0.5
            elif n == 1:
                val = (
                    15.0 - step * 2.0
                )  # step 0: 15 (warmup), step 1: 13 (warmup), step 2: 11 > 10.0 (prunes)
            elif n == 2:
                val = (
                    20.0 if step == 4 else 4.0
                )  # step 4 is in [4, 6) -> 20.0 > 10.0 (prunes)
            elif n == 3:
                val = -2.0 if step == 4 else 2.0  # step 4: -2.0 < 0.0 (prunes)
            elif n == 4:
                val = -10.0 if step == 2 else 5.0  # step 2: -10.0 < 0.0 (prunes)
            else:
                val = 3.0 + step * 0.2
            trial.report(val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return val

    pruner = optuna.pruners.ThresholdPruner(
        lower=0.0, upper=10.0, n_warmup_steps=2, interval_steps=2
    )
    study = optuna.create_study(
        study_name="threshold_pruning", direction="minimize", pruner=pruner
    )
    study.optimize(objective, n_trials=6)

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"p": float(t.params["p"])},
                "value": float(t.value)
                if t.state == optuna.trial.TrialState.COMPLETE
                else None,
                "intermediate_values": {
                    str(k): float(v) for k, v in t.intermediate_values.items()
                },
            }
        )
    return {
        "problem_name": "threshold_pruning",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


thresh_trace = run_threshold_pruning()
if thresh_trace:
    with open(out_dir / "threshold_pruning.json", "w") as f:
        json.dump(thresh_trace, f, indent=2)
    print(
        "✅ ThresholdPruner parity trace generated at Tests/Fixtures/ParityCorpus/threshold_pruning.json"
    )


# 7. PercentilePruner Parity Trace
def run_percentile_pruning():
    trajectories = [
        [10.0, 8.0, 6.0, 4.0, 2.0, 1.0],  # Trial 0 (startup 1)
        [12.0, 10.0, 8.0, 6.0, 4.0, 3.0],  # Trial 1 (startup 2)
        [8.0, 6.0, 4.0, 2.0, 1.0, 0.5],  # Trial 2: completes
        [
            5.0,
            15.0,
            4.0,
            3.0,
            2.0,
            1.0,
        ],  # Trial 3: best-so-far is 5.0 at step 1 -> survives
        [20.0, 25.0, 25.0, 25.0, 25.0, 25.0],  # Trial 4: worst -> prunes at step 1
        [7.0, 7.0, 12.0, 12.0, 2.0, 1.0],  # Trial 5
        [9.0, 9.0, 2.0, 2.0, 1.0, 0.5],  # Trial 6
        [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],  # Trial 7: prunes at step 1
    ]

    def objective(trial):
        _ = trial.suggest_float("p", 0.0, 10.0)
        curve = trajectories[trial.number]
        for step in range(len(curve)):
            val = curve[step]
            trial.report(val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return curve[-1]

    pruner = optuna.pruners.PercentilePruner(
        percentile=25.0,
        n_startup_trials=2,
        n_warmup_steps=1,
        interval_steps=2,
        n_min_trials=2,
    )
    study = optuna.create_study(
        study_name="percentile_pruning", direction="minimize", pruner=pruner
    )
    study.optimize(objective, n_trials=len(trajectories))

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"p": float(t.params["p"])},
                "value": float(t.value)
                if t.state == optuna.trial.TrialState.COMPLETE
                else None,
                "intermediate_values": {
                    str(k): float(v) for k, v in t.intermediate_values.items()
                },
            }
        )
    return {
        "problem_name": "percentile_pruning",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


perc_trace = run_percentile_pruning()
if perc_trace:
    with open(out_dir / "percentile_pruning.json", "w") as f:
        json.dump(perc_trace, f, indent=2)
    print(
        "✅ PercentilePruner parity trace generated at Tests/Fixtures/ParityCorpus/percentile_pruning.json"
    )


# 8. SuccessiveHalvingPruner (ASHA Explicit) Parity Trace
def run_asha_pruning():
    trajectories = [
        [10.0 - s * 0.5 for s in range(10)],
        [15.0 - s * 0.5 for s in range(10)],
        [8.0 - s * 0.5 for s in range(10)],
        [12.0 - s * 0.5 for s in range(10)],
        [6.0 - s * 0.5 for s in range(10)],
        [11.0 - s * 0.5 for s in range(10)],
        [5.0 - s * 0.5 for s in range(10)],
    ]

    def objective(trial):
        _ = trial.suggest_float("p", 0.0, 10.0)
        curve = trajectories[trial.number]
        for step in range(len(curve)):
            val = curve[step]
            trial.report(val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return curve[-1]

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=3, min_early_stopping_rate=0, bootstrap_count=0
    )
    study = optuna.create_study(
        study_name="asha_pruning", direction="minimize", pruner=pruner
    )
    study.optimize(objective, n_trials=len(trajectories))

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"p": float(t.params["p"])},
                "value": float(t.value)
                if t.state == optuna.trial.TrialState.COMPLETE
                else None,
                "intermediate_values": {
                    str(k): float(v) for k, v in t.intermediate_values.items()
                },
            }
        )
    return {
        "problem_name": "asha_pruning",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


asha_trace = run_asha_pruning()
if asha_trace:
    with open(out_dir / "asha_pruning.json", "w") as f:
        json.dump(asha_trace, f, indent=2)
    print(
        "✅ SuccessiveHalvingPruner parity trace generated at Tests/Fixtures/ParityCorpus/asha_pruning.json"
    )


# 9. SuccessiveHalvingPruner (ASHA Auto) Parity Trace
def run_asha_auto_pruning():
    def objective(trial):
        _ = trial.suggest_float("p", 0.0, 10.0)
        n = trial.number
        if n == 0:
            for step in range(200):
                trial.report(50.0 - step * 0.1, step)
            return 30.1
        elif n == 1:
            for step in range(10):
                trial.report(10.0 - step * 0.5, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 5.5
        elif n == 2:
            for step in range(10):
                trial.report(99.0, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 99.0
        elif n == 3:
            for step in range(10):
                trial.report(5.0 - step * 0.5, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 0.5
        else:
            for step in range(10):
                val = 4.0 if step < 2 else 50.0
                trial.report(val, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 50.0

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource="auto", reduction_factor=2
    )
    study = optuna.create_study(
        study_name="asha_auto_pruning", direction="minimize", pruner=pruner
    )
    study.optimize(objective, n_trials=5)

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"p": float(t.params["p"])},
                "value": float(t.value)
                if t.state == optuna.trial.TrialState.COMPLETE
                else None,
                "intermediate_values": {
                    str(k): float(v) for k, v in t.intermediate_values.items()
                },
            }
        )
    return {
        "problem_name": "asha_auto_pruning",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


asha_auto_trace = run_asha_auto_pruning()
if asha_auto_trace:
    with open(out_dir / "asha_auto_pruning.json", "w") as f:
        json.dump(asha_auto_trace, f, indent=2)
    print(
        "✅ SuccessiveHalvingPruner (.auto) parity trace generated at Tests/Fixtures/ParityCorpus/asha_auto_pruning.json"
    )


# 10. PatientPruner (Standalone) Parity Trace
def run_patient_standalone():
    trajectories = [
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],  # Trial 0: steady improvement -> completes
        [10.0, 9.0, 10.0, 10.0, 10.0],         # Trial 1: worsens beyond patience -> prunes at step 4
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1],   # Trial 2: completes
        [8.0, 7.0, 8.0, 8.0, 8.0],             # Trial 3: prunes at step 4
    ]

    def objective(trial):
        _ = trial.suggest_float("p", 0.0, 10.0)
        curve = trajectories[trial.number]
        for step in range(len(curve)):
            val = curve[step]
            trial.report(val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return curve[-1]

    pruner = optuna.pruners.PatientPruner(
        wrapped_pruner=None, patience=2, min_delta=0.5
    )
    study = optuna.create_study(
        study_name="patient_standalone", direction="minimize", pruner=pruner
    )
    study.optimize(objective, n_trials=len(trajectories))

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"p": float(t.params["p"])},
                "value": float(t.value)
                if t.state == optuna.trial.TrialState.COMPLETE
                else None,
                "intermediate_values": {
                    str(k): float(v) for k, v in t.intermediate_values.items()
                },
            }
        )
    return {
        "problem_name": "patient_standalone",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


patient_sa_trace = run_patient_standalone()
if patient_sa_trace:
    with open(out_dir / "patient_standalone.json", "w") as f:
        json.dump(patient_sa_trace, f, indent=2)
    print(
        "✅ PatientPruner (standalone) parity trace generated at Tests/Fixtures/ParityCorpus/patient_standalone.json"
    )


# 11. PatientPruner (Wrapped over MedianPruner) Parity Trace
def run_patient_wrapped():
    trajectories = [
        [2.0, 2.0, 2.0, 2.0, 2.0],             # Trial 0 (startup 1) -> completes
        [3.0, 3.0, 3.0, 3.0, 3.0],             # Trial 1 (startup 2) -> completes
        [10.0, 9.0, 8.0, 7.0, 6.0],            # Trial 2: worse than median but improves by 1.0 > 0.5 -> shielded -> completes!
        [10.0, 5.0, 10.0, 10.0, 10.0],         # Trial 3: worse than median AND stagnates -> delegates to MedianPruner -> prunes at step 4
    ]

    def objective(trial):
        _ = trial.suggest_float("p", 0.0, 10.0)
        curve = trajectories[trial.number]
        for step in range(len(curve)):
            val = curve[step]
            trial.report(val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return curve[-1]

    base = optuna.pruners.MedianPruner(n_startup_trials=2)
    pruner = optuna.pruners.PatientPruner(
        wrapped_pruner=base, patience=2, min_delta=0.5
    )
    study = optuna.create_study(
        study_name="patient_wrapped", direction="minimize", pruner=pruner
    )
    study.optimize(objective, n_trials=len(trajectories))

    traces = []
    for t in study.trials:
        traces.append(
            {
                "number": t.number,
                "state": t.state.name,
                "params": {"p": float(t.params["p"])},
                "value": float(t.value)
                if t.state == optuna.trial.TrialState.COMPLETE
                else None,
                "intermediate_values": {
                    str(k): float(v) for k, v in t.intermediate_values.items()
                },
            }
        )
    return {
        "problem_name": "patient_wrapped",
        "seed": 42,
        "trials": traces,
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
    }


patient_wr_trace = run_patient_wrapped()
if patient_wr_trace:
    with open(out_dir / "patient_wrapped.json", "w") as f:
        json.dump(patient_wr_trace, f, indent=2)
    print(
        "✅ PatientPruner (wrapped) parity trace generated at Tests/Fixtures/ParityCorpus/patient_wrapped.json"
    )
