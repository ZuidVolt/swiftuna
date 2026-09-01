use rustuna_ffi::*;
use std::ffi::{CStr, CString};
use std::ptr;

#[test]
fn test_null_pointer_free_functions_are_safe() {
    // All free functions must safely handle null pointers without segfaulting.
    rustuna_study_free(ptr::null_mut());
    rustuna_trial_free(ptr::null_mut());
    rustuna_persisted_trial_free(ptr::null_mut());
    rustuna_sampler_free(ptr::null_mut());
    rustuna_string_free(ptr::null_mut());
    rustuna_trials_buffer_free(ptr::null_mut(), 0);
}

#[test]
fn test_study_create_null_arguments() {
    // out_study is null
    let res = rustuna_study_create_full(
        ptr::null(),
        ptr::null(),
        0,
        0,
        ptr::null(),
        false,
        ptr::null_mut(),
        ptr::null_mut(),
    );
    assert_eq!(res, -1);
    assert_eq!(rustuna_last_error_code(), -1);
    unsafe {
        let msg = CStr::from_ptr(rustuna_last_error_message()).to_string_lossy();
        assert!(msg.contains("out_study pointer is null"));
    }
}

#[test]
fn test_study_ask_null_arguments() {
    let mut trial: *mut RustunaTrial = ptr::null_mut();
    // Null study
    let res = rustuna_study_ask(ptr::null_mut(), &mut trial);
    assert_eq!(res, -1);

    // Null out_trial
    let name = CString::new("test_study").unwrap();
    let directions = [1i32]; // MINIMIZE
    let mut study: *mut RustunaStudy = ptr::null_mut();
    let code = rustuna_study_create_full(
        name.as_ptr(),
        directions.as_ptr(),
        1,
        0, // In-memory
        ptr::null(),
        false,
        ptr::null_mut(),
        &mut study,
    );
    assert_eq!(code, 0);
    assert!(!study.is_null());

    let res2 = rustuna_study_ask(study, ptr::null_mut());
    assert_eq!(res2, -1);

    rustuna_study_free(study);
}

#[test]
fn test_trial_suggest_null_arguments() {
    let name = CString::new("study_null_suggest").unwrap();
    let directions = [1i32];
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0,
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);
    assert!(!trial.is_null());

    let param_name = CString::new("param_x").unwrap();
    let mut out_val = 0.0f64;

    // Null trial
    assert_eq!(
        rustuna_trial_suggest_float(
            ptr::null_mut(),
            param_name.as_ptr(),
            0.0,
            1.0,
            0.0,
            false,
            &mut out_val
        ),
        -1
    );

    // Null name
    assert_eq!(
        rustuna_trial_suggest_float(trial, ptr::null(), 0.0, 1.0, 0.0, false, &mut out_val),
        -1
    );

    // Null out_val
    assert_eq!(
        rustuna_trial_suggest_float(
            trial,
            param_name.as_ptr(),
            0.0,
            1.0,
            0.0,
            false,
            ptr::null_mut()
        ),
        -1
    );

    rustuna_trial_free(trial);
    rustuna_study_free(study);
}

#[test]
fn test_suggest_float_invalid_range_negative_testing() {
    let name = CString::new("study_invalid_range").unwrap();
    let directions = [1i32];
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0,
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);

    let param_name = CString::new("invalid_float").unwrap();
    let mut out_val = 0.0f64;

    // low > high (10.0 > 5.0) -> must fail with -4 (DistributionError)
    let res = rustuna_trial_suggest_float(
        trial,
        param_name.as_ptr(),
        10.0,
        5.0,
        0.0,
        false,
        &mut out_val,
    );
    assert_eq!(res, -4);
    assert_eq!(rustuna_last_error_code(), -4);

    // log scale with low <= 0.0 -> must fail with -4
    let res_log = rustuna_trial_suggest_float(
        trial,
        param_name.as_ptr(),
        0.0,
        10.0,
        0.0,
        true,
        &mut out_val,
    );
    assert_eq!(res_log, -4);

    rustuna_trial_free(trial);
    rustuna_study_free(study);
}

#[test]
fn test_suggest_int_invalid_range_negative_testing() {
    let name = CString::new("study_invalid_int").unwrap();
    let directions = [1i32];
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0,
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);

    let param_name = CString::new("invalid_int").unwrap();
    let mut out_val = 0i64;

    // low > high (100 > 10) -> must fail with -4 (DistributionError)
    let res =
        rustuna_trial_suggest_int(trial, param_name.as_ptr(), 100, 10, 1, false, &mut out_val);
    assert_eq!(res, -4);

    // log scale with low <= 0 -> must fail with -4
    let res_log =
        rustuna_trial_suggest_int(trial, param_name.as_ptr(), 0, 100, 1, true, &mut out_val);
    assert_eq!(res_log, -4);

    rustuna_trial_free(trial);
    rustuna_study_free(study);
}

#[test]
fn test_suggest_categorical_empty_choices_negative_testing() {
    let name = CString::new("study_empty_cat").unwrap();
    let directions = [1i32];
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0,
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);

    let param_name = CString::new("empty_cat").unwrap();
    let mut out_idx = 0usize;

    // 0 choices -> must fail with -1 (null or empty choices parameter)
    let res =
        rustuna_trial_suggest_categorical(trial, param_name.as_ptr(), ptr::null(), 0, &mut out_idx);
    assert_eq!(res, -1);

    rustuna_trial_free(trial);
    rustuna_study_free(study);
}

#[test]
fn test_study_tell_mismatched_directions_count_negative_testing() {
    let name = CString::new("study_mismatched_dirs").unwrap();
    // 2 directions (multi-objective)
    let directions = [1i32, 2i32];
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            2,
            0,
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);
    let trial_num = rustuna_trial_get_number(trial);
    rustuna_trial_free(trial);

    // Only provide 1 objective value when study expects 2 -> must fail
    let values = [1.23f64];
    let res = rustuna_study_tell_multi(
        study,
        trial_num,
        1, /* Complete */
        values.as_ptr(),
        1,
        ptr::null(),
    );
    assert_eq!(res, -1);

    rustuna_study_free(study);
}

#[test]
fn test_tell_nan_objective_value_strictly_rejected() {
    let name = CString::new("study_tell_nan").unwrap();
    let directions = [1i32];
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0,
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);
    let trial_num = rustuna_trial_get_number(trial);
    rustuna_trial_free(trial);

    // Passing NaN as objective value must return -1
    let nan_val = [f64::NAN];
    let res = rustuna_study_tell_multi(study, trial_num, 1, nan_val.as_ptr(), 1, ptr::null());
    assert_eq!(res, -1);
    assert_eq!(rustuna_last_error_code(), -1);

    rustuna_study_free(study);
}

#[test]
fn test_tell_complete_with_null_values_strictly_rejected() {
    let name = CString::new("study_tell_null_val").unwrap();
    let directions = [1i32];
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0,
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    let mut trial: *mut RustunaTrial = ptr::null_mut();
    assert_eq!(rustuna_study_ask(study, &mut trial), 0);
    let trial_num = rustuna_trial_get_number(trial);
    rustuna_trial_free(trial);

    // Passing null pointer for Complete state must return -1
    let res = rustuna_study_tell_multi(study, trial_num, 1, ptr::null(), 0, ptr::null());
    assert_eq!(res, -1);

    rustuna_study_free(study);
}
