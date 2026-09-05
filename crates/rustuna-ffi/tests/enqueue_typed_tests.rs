use rustuna_ffi::*;
use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "common/mod.rs"]
mod common;

use common::{ask_trial, random_study, suggest_categorical, suggest_float, suggest_int, tell_complete};

#[test]
fn typed_enqueue_fixes_all_kinds() {
    let study = random_study("enqueue_typed_test", 7);
    let n_x = CString::new("x").unwrap();
    let n_n = CString::new("n").unwrap();
    let n_o = CString::new("opt").unwrap();
    let n_f = CString::new("flag").unwrap();
    let s_sgd = CString::new("sgd").unwrap();
    let names = [n_x.as_ptr(), n_n.as_ptr(), n_o.as_ptr(), n_f.as_ptr()];
    let kinds = [1u8, 0u8, 2u8, 3u8];
    let nums = [2.5f64, 32.0, 0.0, 1.0];
    let strs = [ptr::null(), ptr::null(), s_sgd.as_ptr(), ptr::null()];
    assert_eq!(
        rustuna_study_enqueue_typed(
            study,
            names.as_ptr(),
            kinds.as_ptr(),
            nums.as_ptr(),
            strs.as_ptr(),
            4,
            ptr::null(),
        ),
        0
    );

    let trial = ask_trial(study);
    assert_eq!(suggest_float(trial, "x", -10.0, 10.0), 2.5);
    assert_eq!(suggest_int(trial, "n", 1, 64), 32);
    assert_eq!(suggest_categorical(trial, "opt", &["adam", "sgd"]), 1);
    rustuna_trial_free(trial);
    rustuna_study_free(study);
    let _ = n_f;
}

#[test]
fn typed_enqueue_rejects_bad_input() {
    let study = random_study("enqueue_typed_test", 7);
    let n = CString::new("x").unwrap();
    let names = [n.as_ptr()];
    let nums = [1.0f64];
    let strs = [ptr::null()];
    // Unknown kind.
    assert_eq!(
        rustuna_study_enqueue_typed(
            study,
            names.as_ptr(),
            [9u8].as_ptr(),
            nums.as_ptr(),
            strs.as_ptr(),
            1,
            ptr::null(),
        ),
        -1
    );
    // Null arrays with nonzero count.
    assert_eq!(
        rustuna_study_enqueue_typed(
            study,
            ptr::null(),
            [1u8].as_ptr(),
            nums.as_ptr(),
            strs.as_ptr(),
            1,
            ptr::null(),
        ),
        -1
    );
    // Null study.
    assert_eq!(
        rustuna_study_enqueue_typed(
            ptr::null_mut(),
            names.as_ptr(),
            [1u8].as_ptr(),
            nums.as_ptr(),
            strs.as_ptr(),
            1,
            ptr::null(),
        ),
        -1
    );
    // Zero count with null arrays is legal (all-Rust-sampled trial).
    assert_eq!(
        rustuna_study_enqueue_typed(
            study,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
        ),
        0
    );
    rustuna_study_free(study);
}

#[test]
fn json_enqueue_still_works() {
    let study = random_study("enqueue_typed_test", 7);
    let pj = CString::new(r#"{"x": 3.5}"#).unwrap();
    assert_eq!(
        rustuna_study_enqueue_trial(study, pj.as_ptr(), ptr::null()),
        0
    );
    let trial = ask_trial(study);
    assert_eq!(suggest_float(trial, "x", -10.0, 10.0), 3.5);
    rustuna_trial_free(trial);
    rustuna_study_free(study);
}

fn fetch_json_since(study: *mut RustunaStudy, from: u32) -> Vec<serde_json::Value> {
    let mut out: *mut std::os::raw::c_char = ptr::null_mut();
    let mut len = 0usize;
    assert_eq!(
        rustuna_study_get_trials_json_since(study, u32::MAX, from, &mut out, &mut len),
        0
    );
    let s = common::read_json_string(out);
    serde_json::from_str::<Vec<serde_json::Value>>(&s).unwrap()
}

#[test]
fn since_fetch_returns_only_the_tail() {
    let study = random_study("enqueue_typed_test", 7);
    for n in 0..5u32 {
        let trial = ask_trial(study);
        let x = suggest_float(trial, "x", -10.0, 10.0);
        rustuna_trial_free(trial);
        tell_complete(study, n, x * x);
    }
    // Full fetch: 5 trials.
    assert_eq!(fetch_json_since(study, 0).len(), 5);
    // Tail fetch: numbers 3, 4 only.
    let tail = fetch_json_since(study, 3);
    assert_eq!(tail.len(), 2);
    let numbers: Vec<u64> = tail
        .iter()
        .map(|t| t.get("number").and_then(|v| v.as_u64()).unwrap())
        .collect();
    assert_eq!(numbers, vec![3, 4]);
    // Past the end: empty array, still success.
    assert_eq!(fetch_json_since(study, 99).len(), 0);
    rustuna_study_free(study);
}

static CB_CALLS: AtomicUsize = AtomicUsize::new(0);
static CB_LAST_TRIAL: AtomicUsize = AtomicUsize::new(usize::MAX);

unsafe extern "C" fn fixed_float_cb(
    _ctx: *mut std::ffi::c_void,
    _name: *const std::os::raw::c_char,
    _low: f64,
    _high: f64,
    _step: f64,
    _log: bool,
    trial_number: u32,
    out_value: *mut f64,
) -> i32 {
    CB_CALLS.fetch_add(1, Ordering::SeqCst);
    CB_LAST_TRIAL.store(trial_number as usize, Ordering::SeqCst);
    unsafe { *out_value = 2.5 };
    0
}

#[test]
fn callback_sampler_suggests_through_upcall() {
    CB_CALLS.store(0, Ordering::SeqCst);
    let vtable = rustuna_ffi::CallbackSamplerVTable {
        ctx: ptr::null_mut(),
        free_ctx: None,
        suggest_float: Some(fixed_float_cb),
        suggest_int: None,
        suggest_categorical: None,
    };
    let mut sampler: *mut RustunaSampler = ptr::null_mut();
    assert_eq!(rustuna_sampler_callback_new(&vtable, &mut sampler), 0);
    assert!(!sampler.is_null());

    let study = common::create_study("cb_study", sampler);

    let trial = ask_trial(study);
    assert_eq!(suggest_float(trial, "x", -10.0, 10.0), 2.5);
    assert!(CB_CALLS.load(Ordering::SeqCst) >= 1);
    // First trial in the study carries number 0 through the upcall.
    assert_eq!(CB_LAST_TRIAL.load(Ordering::SeqCst), 0);
    rustuna_trial_free(trial);
    rustuna_study_free(study);
}

fn suggest_all(study: *mut RustunaStudy) -> (f64, i64, usize) {
    let trial = ask_trial(study);
    let x = suggest_float(trial, "x", -10.0, 10.0);
    let n = suggest_int(trial, "n", 1, 100);
    let idx = suggest_categorical(trial, "opt", &["adam", "sgd"]);
    rustuna_trial_free(trial);
    (x, n, idx)
}

#[test]
fn typed_matches_json_path() {
    // Same logical params through both encodings must suggest identically.
    // This is the debug insurance for the typed path's pointer arithmetic.
    let json_study = random_study("enqueue_typed_test", 7);
    let pj = CString::new(r#"{"x": 2.5, "n": 32, "opt": "sgd"}"#).unwrap();
    assert_eq!(
        rustuna_study_enqueue_trial(json_study, pj.as_ptr(), ptr::null()),
        0
    );

    let typed_study = random_study("enqueue_typed_test", 7);
    let n_x = CString::new("x").unwrap();
    let n_n = CString::new("n").unwrap();
    let n_o = CString::new("opt").unwrap();
    let s_sgd = CString::new("sgd").unwrap();
    let names = [n_x.as_ptr(), n_n.as_ptr(), n_o.as_ptr()];
    let kinds = [1u8, 0u8, 2u8];
    let nums = [2.5f64, 32.0, 0.0];
    let strs = [ptr::null(), ptr::null(), s_sgd.as_ptr()];
    assert_eq!(
        rustuna_study_enqueue_typed(
            typed_study,
            names.as_ptr(),
            kinds.as_ptr(),
            nums.as_ptr(),
            strs.as_ptr(),
            3,
            ptr::null(),
        ),
        0
    );

    assert_eq!(suggest_all(json_study), suggest_all(typed_study));
    rustuna_study_free(json_study);
    rustuna_study_free(typed_study);
}
