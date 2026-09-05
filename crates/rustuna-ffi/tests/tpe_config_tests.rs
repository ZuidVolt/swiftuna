use rustuna_ffi::*;

#[path = "common/mod.rs"]
mod common;

use common::{ask_trial, suggest_float, tell_complete};

fn study_with_tpe_full(multivariate: i8, n_startup: usize, seed: u64) -> *mut RustunaStudy {
    // Tests always seed for determinism. Return-style, like tpe/random/qmc.
    let sampler = rustuna_sampler_tpe_full(seed, true, multivariate, n_startup);
    common::create_study("tpe_full_test", sampler)
}

fn run_trials(study: *mut RustunaStudy, n: u32) -> Vec<f64> {
    let mut values = Vec::new();
    for i in 0..n {
        let trial = ask_trial(study);
        let x = suggest_float(trial, "x", -10.0, 10.0);
        rustuna_trial_free(trial);
        let v = x * x;
        tell_complete(study, i, v);
        values.push(v);
    }
    values
}

#[test]
fn tpe_full_runs_in_all_modes() {
    for mode in [-1i8, 0, 1] {
        let study = study_with_tpe_full(mode, 10, 42);
        let values = run_trials(study, 5);
        assert_eq!(values.len(), 5);
        assert!(values.iter().all(|v| v.is_finite()));
        rustuna_study_free(study);
    }
}

#[test]
fn tpe_full_seeded_determinism() {
    let a = {
        let study = study_with_tpe_full(-1, 10, 123);
        let v = run_trials(study, 8);
        rustuna_study_free(study);
        v
    };
    let b = {
        let study = study_with_tpe_full(-1, 10, 123);
        let v = run_trials(study, 8);
        rustuna_study_free(study);
        v
    };
    assert_eq!(a, b);
}
