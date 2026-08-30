use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};



use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::Sampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study_with_arc, Direction};
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::tpe::TpeSampler;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct TrialTrace {
    number: u32,
    params: HashMap<String, f64>,
    value: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct ProblemCorpus {
    problem_name: String,
    seed: u64,
    trials: Vec<TrialTrace>,
    best_trial_number: u32,
    best_value: f64,
}

fn run_quadratic() -> ProblemCorpus {
    let seed = 42u64;
    let sampler: Arc<dyn Sampler> = Arc::new(TpeSampler::seed_from_u64(seed));
    let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
    let study = create_study_with_arc("quadratic", storage, sampler, vec![Direction::Minimize]).unwrap();

    let mut traces = Vec::new();
    let mut best_val = f64::MAX;
    let mut best_num = 0;

    for _ in 0..15 {
        let mut trial = study.ask().unwrap();
        let num = trial.number;

        let dist_x = Distribution::new_float(-10.0, 10.0, None, false);
        let dist_y = Distribution::new_float(-10.0, 10.0, None, false);

        let x = trial.suggest("x", &dist_x).unwrap();
        let y = trial.suggest("y", &dist_y).unwrap();

        let loss = (x - 2.0).powi(2) + (y + 5.0).powi(2);

        let mut params = HashMap::new();
        params.insert("x".to_string(), x);
        params.insert("y".to_string(), y);

        study.tell(num, TrialStateValues::Complete(vec![loss])).unwrap();

        if loss < best_val {
            best_val = loss;
            best_num = num;
        }

        traces.push(TrialTrace {
            number: num,
            params,
            value: loss,
        });
    }

    ProblemCorpus {
        problem_name: "quadratic".to_string(),
        seed,
        trials: traces,
        best_trial_number: best_num,
        best_value: best_val,
    }
}

fn run_constrained_weights() -> ProblemCorpus {
    let seed = 42u64;
    let sampler: Arc<dyn Sampler> = Arc::new(TpeSampler::seed_from_u64(seed));
    let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
    let study = create_study_with_arc("constrained_weights", storage, sampler, vec![Direction::Minimize]).unwrap();

    let mut traces = Vec::new();
    let mut best_val = f64::MAX;
    let mut best_num = 0;

    for _ in 0..15 {
        let mut trial = study.ask().unwrap();
        let num = trial.number;

        let dist_param = Distribution::new_float(0.5, 2.0, Some(0.1), false);
        let dist_mut = Distribution::new_float(0.8, 3.0, Some(0.1), false);
        let dist_sink = Distribution::new_float(0.2, 2.0, Some(0.1), false);

        let p_w = trial.suggest("param_weight", &dist_param).unwrap();
        let m_w = trial.suggest("mutation_weight", &dist_mut).unwrap();
        let s_w = trial.suggest("sink_weight", &dist_sink).unwrap();

        let ceiling = m_w + 4.0 * s_w;
        let loss = if ceiling > 5.0 {
            1000.0 + (ceiling - 5.0) * 100.0
        } else {
            (p_w - 1.0).powi(2) + (m_w - 1.5).powi(2) + (s_w - 0.5).powi(2)
        };

        let mut params = HashMap::new();
        params.insert("param_weight".to_string(), p_w);
        params.insert("mutation_weight".to_string(), m_w);
        params.insert("sink_weight".to_string(), s_w);

        study.tell(num, TrialStateValues::Complete(vec![loss])).unwrap();

        if loss < best_val {
            best_val = loss;
            best_num = num;
        }

        traces.push(TrialTrace {
            number: num,
            params,
            value: loss,
        });
    }

    ProblemCorpus {
        problem_name: "constrained_weights".to_string(),
        seed,
        trials: traces,
        best_trial_number: best_num,
        best_value: best_val,
    }
}

fn run_integer_grid() -> ProblemCorpus {
    let seed = 42u64;
    let sampler: Arc<dyn Sampler> = Arc::new(TpeSampler::seed_from_u64(seed));
    let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
    let study = create_study_with_arc("integer_grid", storage, sampler, vec![Direction::Minimize]).unwrap();

    let mut traces = Vec::new();
    let mut best_val = f64::MAX;
    let mut best_num = 0;

    for _ in 0..15 {
        let mut trial = study.ask().unwrap();
        let num = trial.number;

        let dist_layers = Distribution::new_int(1, 8, 1, false);
        let dist_units = Distribution::new_int(32, 256, 32, false);

        let layers = trial.suggest("n_layers", &dist_layers).unwrap();
        let units = trial.suggest("hidden_units", &dist_units).unwrap();

        let loss = ((layers * units) - 512.0).abs();

        let mut params = HashMap::new();
        params.insert("n_layers".to_string(), layers);
        params.insert("hidden_units".to_string(), units);

        study.tell(num, TrialStateValues::Complete(vec![loss])).unwrap();

        if loss < best_val {
            best_val = loss;
            best_num = num;
        }

        traces.push(TrialTrace {
            number: num,
            params,
            value: loss,
        });
    }

    ProblemCorpus {
        problem_name: "integer_grid".to_string(),
        seed,
        trials: traces,
        best_trial_number: best_num,
        best_value: best_val,
    }
}

fn run_categorical_grid() -> ProblemCorpus {
    let seed = 42u64;
    let sampler: Arc<dyn Sampler> = Arc::new(TpeSampler::seed_from_u64(seed));
    let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
    let study = create_study_with_arc("categorical_grid", storage, sampler, vec![Direction::Minimize]).unwrap();

    let mut traces = Vec::new();
    let mut best_val = f64::MAX;
    let mut best_num = 0;

    for _ in 0..15 {
        let mut trial = study.ask().unwrap();
        let num = trial.number;

        let dist_opt = Distribution::new_categorical(4);
        let idx = trial.suggest("optimizer", &dist_opt).unwrap();

        // idx: 0=adam, 1=sgd, 2=rmsprop, 3=adamw
        let loss = match idx as usize {
            0 => 0.12,
            1 => 0.45,
            2 => 0.28,
            _ => 0.08,
        };

        let mut params = HashMap::new();
        params.insert("optimizer".to_string(), idx);

        study.tell(num, TrialStateValues::Complete(vec![loss])).unwrap();

        if loss < best_val {
            best_val = loss;
            best_num = num;
        }

        traces.push(TrialTrace {
            number: num,
            params,
            value: loss,
        });
    }

    ProblemCorpus {
        problem_name: "categorical_grid".to_string(),
        seed,
        trials: traces,
        best_trial_number: best_num,
        best_value: best_val,
    }
}

fn main() {
    let out_dir = Path::new("Tests/Fixtures/ParityCorpus");
    fs::create_dir_all(out_dir).expect("Failed to create corpus dir");

    let quad = run_quadratic();
    let constr = run_constrained_weights();
    let int_grid = run_integer_grid();
    let cat_grid = run_categorical_grid();

    fs::write(
        out_dir.join("quadratic.json"),
        serde_json::to_string_pretty(&quad).unwrap(),
    ).unwrap();

    fs::write(
        out_dir.join("constrained_weights.json"),
        serde_json::to_string_pretty(&constr).unwrap(),
    ).unwrap();

    fs::write(
        out_dir.join("integer_grid.json"),
        serde_json::to_string_pretty(&int_grid).unwrap(),
    ).unwrap();

    fs::write(
        out_dir.join("categorical_grid.json"),
        serde_json::to_string_pretty(&cat_grid).unwrap(),
    ).unwrap();

    println!("✅ Generated 4 golden parity corpus traces in {:?}", out_dir);
}
