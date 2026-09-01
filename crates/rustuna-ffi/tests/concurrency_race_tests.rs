use rustuna_ffi::*;
use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

#[test]
fn test_concurrent_multithreaded_ask_tell_no_data_race() {
    let name = CString::new("concurrent_study").unwrap();
    let directions = [1i32]; // MINIMIZE
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0, // In-Memory
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study
        ),
        0
    );

    // Study pointer wrapped into Send pointer
    struct SendStudy(*mut RustunaStudy);
    unsafe impl Send for SendStudy {}
    unsafe impl Sync for SendStudy {}

    let shared_study = Arc::new(SendStudy(study));
    let completed_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    let num_threads = 8;
    let trials_per_thread = 25;

    for _ in 0..num_threads {
        let s_clone = Arc::clone(&shared_study);
        let count_clone = Arc::clone(&completed_count);

        let h = thread::spawn(move || {
            let param_name = CString::new("weight").unwrap();

            for _ in 0..trials_per_thread {
                let mut trial: *mut RustunaTrial = ptr::null_mut();
                let ask_res = rustuna_study_ask(s_clone.0, &mut trial);
                assert_eq!(ask_res, 0);
                assert!(!trial.is_null());

                let mut val = 0.0f64;
                assert_eq!(
                    rustuna_trial_suggest_float(
                        trial,
                        param_name.as_ptr(),
                        -5.0,
                        5.0,
                        0.0,
                        false,
                        &mut val
                    ),
                    0
                );

                let t_num = rustuna_trial_get_number(trial);
                rustuna_trial_free(trial);

                let obj = val * val;
                let tell_res = rustuna_study_tell_multi(s_clone.0, t_num, 1, &obj, 1, ptr::null());
                assert_eq!(tell_res, 0);

                count_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(
        completed_count.load(Ordering::SeqCst),
        num_threads * trials_per_thread
    );

    // Verify all trials were recorded and best trial is retrievable
    let mut best_trial: *mut RustunaPersistedTrial = ptr::null_mut();
    assert_eq!(rustuna_study_get_best_trial(study, &mut best_trial), 0);
    assert!(!best_trial.is_null());
    rustuna_persisted_trial_free(best_trial);

    let mut all_trials: *mut *mut RustunaPersistedTrial = ptr::null_mut();
    let mut total_count = 0usize;
    assert_eq!(
        rustuna_study_get_trials(study, &mut all_trials, &mut total_count),
        0
    );
    assert_eq!(total_count, num_threads * trials_per_thread);
    rustuna_trials_buffer_free(all_trials, total_count);

    rustuna_study_free(study);
}

#[test]
fn test_concurrent_take_last_error_isolation_and_consumption() {
    use std::ffi::CStr;

    let num_threads = 10;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let h = thread::spawn(move || {
            // Create independent study for each thread
            let study_name = CString::new(format!("study_err_{thread_id}")).unwrap();
            let directions = [1i32];
            let mut study: *mut RustunaStudy = ptr::null_mut();
            assert_eq!(
                rustuna_study_create_full(
                    study_name.as_ptr(),
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

            // Trigger a thread-unique error: low > high where low is thread_id + 100
            let param_name = CString::new("x").unwrap();
            let low = 100.0 + (thread_id as f64);
            let high = 50.0;
            let mut val = 0.0f64;
            let code = rustuna_trial_suggest_float(
                trial,
                param_name.as_ptr(),
                low,
                high,
                0.0,
                false,
                &mut val,
            );
            assert_eq!(code, -4);

            // 1. Take last error
            let mut err_code = 0i32;
            let mut err_msg_ptr: *mut std::os::raw::c_char = ptr::null_mut();
            let has_err = rustuna_take_last_error(&mut err_code, &mut err_msg_ptr);
            assert_eq!(has_err, 1);
            assert_eq!(err_code, -4);
            assert!(!err_msg_ptr.is_null());

            unsafe {
                let msg_str = CStr::from_ptr(err_msg_ptr).to_string_lossy();
                // Must contain thread's unique low value, proving NO cross-thread error bleeding
                assert!(
                    msg_str.contains(&format!("low ({low:.1})")) || msg_str.contains(&format!("low ({low})")),
                    "Expected message to contain low ({low}), got: {msg_str}"
                );
                rustuna_string_free(err_msg_ptr);
            }

            // 2. Second take on same thread MUST return 0 (slot consumed and reset)
            let mut second_code = 0i32;
            let mut second_msg: *mut std::os::raw::c_char = ptr::null_mut();
            let second_has = rustuna_take_last_error(&mut second_code, &mut second_msg);
            assert_eq!(second_has, 0);
            assert!(second_msg.is_null());

            rustuna_trial_free(trial);
            rustuna_study_free(study);
        });
        handles.push(h);
    }

    for h in handles {
        h.join().unwrap();
    }
}
