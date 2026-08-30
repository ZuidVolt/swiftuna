use std::sync::{Arc, RwLock};
use std::time::Instant;

use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::Sampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study_with_arc, Direction};
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::tpe::TpeSampler;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_trials: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let sampler: Arc<dyn Sampler> = Arc::new(TpeSampler::seed_from_u64(42));
    let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
    let study = create_study_with_arc("bench", storage, sampler, vec![Direction::Minimize]).unwrap();

    let dist_x = Distribution::new_float(-10.0, 10.0, None, false);
    let dist_y = Distribution::new_float(-10.0, 10.0, None, false);

    let start = Instant::now();

    for _ in 0..n_trials {
        let mut trial = study.ask().unwrap();
        let x = trial.suggest("x", &dist_x).unwrap();
        let y = trial.suggest("y", &dist_y).unwrap();
        let loss = (x - 2.0).powi(2) + (y + 5.0).powi(2);
        study.tell(trial.number, TrialStateValues::Complete(vec![loss])).unwrap();
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let us_per_trial = (elapsed.as_secs_f64() * 1_000_000.0) / (n_trials as f64);
    let throughput = (n_trials as f64) / elapsed.as_secs_f64();

    println!("RESULT:{:.3}:{:.2}:{:.0}", total_ms, us_per_trial, throughput);
}
