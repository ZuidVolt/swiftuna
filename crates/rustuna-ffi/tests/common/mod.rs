//! Shared helpers for FFI integration tests.
//!
//! Each file directly under `tests/` compiles as its own crate; this module
//! (a subdirectory, hence not a target itself) holds the ask/suggest/tell
//! plumbing so the per-feature files cannot drift apart. Include with:
//! `#[path = "common/mod.rs"] mod common;`

use rustuna_ffi::*;
use std::ffi::{CStr, CString};
use std::ptr;

/// Creates an in-memory single-objective study owned by the caller.
pub fn create_study(name: &str, sampler: *mut RustunaSampler) -> *mut RustunaStudy {
    assert!(!sampler.is_null());
    let c_name = CString::new(name).unwrap();
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            c_name.as_ptr(),
            ptr::null(),
            0,
            0,
            ptr::null(),
            false,
            sampler,
            &mut study,
        ),
        0
    );
    rustuna_sampler_free(sampler);
    assert!(!study.is_null());
    study
}

/// Creates a study with a seeded random sampler.
pub fn random_study(name: &str, seed: u64) -> *mut RustunaStudy {
    create_study(name, rustuna_sampler_random_new(seed, true))
}

/// Checks out the next trial.
pub fn ask_trial(study: *mut RustunaStudy) -> *mut RustunaTrial {
    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);
    assert!(!trial.is_null());
    trial
}

/// Suggests a float and returns it.
pub fn suggest_float(trial: *mut RustunaTrial, name: &str, low: f64, high: f64) -> f64 {
    let c_name = CString::new(name).unwrap();
    let mut out = 0.0;
    assert_eq!(
        rustuna_trial_suggest_float(trial, c_name.as_ptr(), low, high, 0.0, false, &mut out),
        0
    );
    out
}

/// Suggests an int and returns it.
pub fn suggest_int(trial: *mut RustunaTrial, name: &str, low: i64, high: i64) -> i64 {
    let c_name = CString::new(name).unwrap();
    let mut out = 0i64;
    assert_eq!(
        rustuna_trial_suggest_int(trial, c_name.as_ptr(), low, high, 1, false, &mut out),
        0
    );
    out
}

/// Suggests a categorical and returns the chosen index.
pub fn suggest_categorical(trial: *mut RustunaTrial, name: &str, choices: &[&str]) -> usize {
    let c_name = CString::new(name).unwrap();
    let c_choices: Vec<CString> = choices
        .iter()
        .map(|c| CString::new(*c).unwrap())
        .collect();
    let ptrs: Vec<*const std::os::raw::c_char> =
        c_choices.iter().map(|c| c.as_ptr()).collect();
    let mut out = 0usize;
    assert_eq!(
        rustuna_trial_suggest_categorical(
            trial,
            c_name.as_ptr(),
            ptrs.as_ptr(),
            ptrs.len(),
            &mut out
        ),
        0
    );
    out
}

/// Records a completed single-objective trial.
pub fn tell_complete(study: *mut RustunaStudy, number: u32, value: f64) {
    let values = [value];
    assert_eq!(
        rustuna_study_tell_multi(study, number, 1, values.as_ptr(), 1, ptr::null()),
        0
    );
}

/// Reads a returned JSON string and frees it.
pub fn read_json_string(ptr: *mut std::os::raw::c_char) -> String {
    assert!(!ptr.is_null());
    let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
    unsafe { rustuna_string_free(ptr) };
    s
}
