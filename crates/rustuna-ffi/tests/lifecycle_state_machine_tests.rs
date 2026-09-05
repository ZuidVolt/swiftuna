use proptest::prelude::*;
use rustuna_ffi::*;
use serde_json::Value;
use std::ffi::{CStr, CString};
use std::ptr;

#[derive(Debug, Clone)]
enum TrialOutcome {
    Complete(f64),
    Pruned,
    Failed,
}

proptest! {
    #[test]
    fn prop_ask_tell_lifecycle_and_best_trial_optimality(
        outcomes in prop::collection::vec(
            prop_oneof![
                (-1000.0..1000.0f64).prop_map(TrialOutcome::Complete),
                Just(TrialOutcome::Pruned),
                Just(TrialOutcome::Failed),
            ],
            1..20
        )
    ) {
        let name = CString::new("prop_lifecycle_study").unwrap();
        let directions = [1i32]; // MINIMIZE
        let mut study: *mut RustunaStudy = ptr::null_mut();
        prop_assert_eq!(
            rustuna_study_create_full(
                name.as_ptr(),
                directions.as_ptr(),
                1,
                0, // InMemory
                ptr::null(),
                false,
                ptr::null_mut(),
                &mut study
            ),
            0
        );

        let mut expected_min: Option<f64> = None;
        let mut completed_count = 0;
        let mut pruned_count = 0;
        let mut failed_count = 0;

        let param_name = CString::new("lr").unwrap();
        let attr_key = CString::new("model_arch").unwrap();
        let attr_val = CString::new("resnet18").unwrap();

        for (idx, outcome) in outcomes.iter().enumerate() {
            let mut trial: *mut RustunaTrial = ptr::null_mut();
            prop_assert_eq!(rustuna_study_ask(study, &mut trial), 0);
            prop_assert!(!trial.is_null());

            let t_num = rustuna_trial_get_number(trial);
            prop_assert_eq!(t_num, idx as u32);

            let mut lr = 0.0f64;
            rustuna_trial_suggest_float(trial, param_name.as_ptr(), 0.0001, 0.1, 0.0, true, &mut lr);

            // Set user attribute
            rustuna_trial_set_user_attr(trial, attr_key.as_ptr(), attr_val.as_ptr());

            // Set constraint (feasibility)
            let c_key = CString::new("max_mem").unwrap();
            rustuna_trial_set_constraint(trial, c_key.as_ptr(), -50.0);

            rustuna_trial_free(trial);

            match outcome {
                TrialOutcome::Complete(val) => {
                    let intermediate_json = CString::new("{\"1\": 0.95, \"2\": 0.85}").unwrap();
                    let val_arr = [*val];
                    let code = rustuna_study_tell_multi(
                        study,
                        t_num,
                        1, // Complete
                        val_arr.as_ptr(),
                        1,
                        intermediate_json.as_ptr(),
                    );
                    prop_assert_eq!(code, 0);

                    completed_count += 1;
                    expected_min = Some(match expected_min {
                        Some(cur) => cur.min(*val),
                        None => *val,
                    });
                }
                TrialOutcome::Pruned => {
                    let intermediate_json = CString::new("{\"1\": 1.5}").unwrap();
                    let code = rustuna_study_tell_multi(
                        study,
                        t_num,
                        2, // Pruned
                        ptr::null(),
                        0,
                        intermediate_json.as_ptr(),
                    );
                    prop_assert_eq!(code, 0);
                    pruned_count += 1;
                }
                TrialOutcome::Failed => {
                    let code = rustuna_study_tell_multi(
                        study,
                        t_num,
                        4, // Fail
                        ptr::null(),
                        0,
                        ptr::null(),
                    );
                    prop_assert_eq!(code, 0);
                    failed_count += 1;
                }
            }
        }

        // Verify best trial
        let mut best_trial: *mut RustunaPersistedTrial = ptr::null_mut();
        let best_code = rustuna_study_get_best_trial(study, &mut best_trial);

        if let Some(min_val) = expected_min {
            prop_assert_eq!(best_code, 0);
            prop_assert!(!best_trial.is_null());

            let mut json_ptr: *mut std::os::raw::c_char = ptr::null_mut();
            prop_assert_eq!(rustuna_persisted_trial_get_json(best_trial, &mut json_ptr), 0);
            prop_assert!(!json_ptr.is_null());

            unsafe {
                let json_str = CStr::from_ptr(json_ptr).to_string_lossy();
                let parsed: Value = serde_json::from_str(&json_str).unwrap();

                // State must be 1 (Complete)
                prop_assert_eq!(parsed["state"].as_i64().unwrap(), 1);

                // Value must match expected minimum
                let recorded_val = parsed["values"][0].as_f64().unwrap();
                prop_assert!((recorded_val - min_val).abs() < 1e-6);

                // User attribute must be preserved
                prop_assert_eq!(parsed["user_attrs"]["model_arch"].as_str().unwrap(), "resnet18");

                // Constraint must be preserved
                prop_assert_eq!(parsed["constraints"]["max_mem"].as_f64().unwrap(), -50.0);

                rustuna_string_free(json_ptr);
            }
            rustuna_persisted_trial_free(best_trial);
        } else {
            // No completed trials -> get_best_trial must fail
            prop_assert_ne!(best_code, 0);
        }

        // Verify filtered state queries
        let mut all_trials: *mut *mut RustunaPersistedTrial = ptr::null_mut();
        let mut total_len = 0usize;
        prop_assert_eq!(rustuna_study_get_trials(study, &mut all_trials, &mut total_len), 0);
        prop_assert_eq!(total_len, outcomes.len());
        rustuna_trials_buffer_free(all_trials, total_len);

        // Filter only Complete trials (bitmask 1 << 1 = 2)
        let mut complete_trials: *mut *mut RustunaPersistedTrial = ptr::null_mut();
        let mut c_len = 0usize;
        prop_assert_eq!(rustuna_study_get_trials_filtered(study, 1 << 1, &mut complete_trials, &mut c_len), 0);
        prop_assert_eq!(c_len, completed_count);
        rustuna_trials_buffer_free(complete_trials, c_len);

        // Filter only Pruned trials (bitmask 1 << 2 = 4)
        let mut p_trials: *mut *mut RustunaPersistedTrial = ptr::null_mut();
        let mut p_len = 0usize;
        prop_assert_eq!(rustuna_study_get_trials_filtered(study, 1 << 2, &mut p_trials, &mut p_len), 0);
        prop_assert_eq!(p_len, pruned_count);
        rustuna_trials_buffer_free(p_trials, p_len);

        // Filter only Failed trials (bitmask 1 << 4 = 16)
        let mut f_trials: *mut *mut RustunaPersistedTrial = ptr::null_mut();
        let mut f_len = 0usize;
        prop_assert_eq!(rustuna_study_get_trials_filtered(study, 1 << 4, &mut f_trials, &mut f_len), 0);
        prop_assert_eq!(f_len, failed_count);
        rustuna_trials_buffer_free(f_trials, f_len);

        rustuna_study_free(study);
    }

    #[test]
    fn prop_trials_batch_json_matches_individual_trials(
        outcomes in prop::collection::vec(
            prop_oneof![
                (-500.0..500.0f64).prop_map(TrialOutcome::Complete),
                Just(TrialOutcome::Pruned),
                Just(TrialOutcome::Failed),
            ],
            1..15
        ),
        mask in prop_oneof![
            Just(0xFFFFFFFFu32), // All
            Just(1u32 << 1),     // Complete only
            Just(1u32 << 2),     // Pruned only
            Just((1u32 << 1) | (1u32 << 2)), // Complete + Pruned
        ]
    ) {
        let name = CString::new("prop_batch_vs_indiv_study").unwrap();
        let directions = [1i32];
        let mut study: *mut RustunaStudy = ptr::null_mut();
        prop_assert_eq!(
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

        let param_name = CString::new("alpha").unwrap();
        let attr_key = CString::new("batch_size").unwrap();
        let attr_val = CString::new("64").unwrap();

        for (_idx, outcome) in outcomes.iter().enumerate() {
            let mut trial: *mut RustunaTrial = ptr::null_mut();
            prop_assert_eq!(rustuna_study_ask(study, &mut trial), 0);
            let mut alpha = 0.0f64;
            rustuna_trial_suggest_float(trial, param_name.as_ptr(), 0.0, 1.0, 0.0, false, &mut alpha);
            rustuna_trial_set_user_attr(trial, attr_key.as_ptr(), attr_val.as_ptr());
            let t_num = rustuna_trial_get_number(trial);
            rustuna_trial_free(trial);

            match outcome {
                TrialOutcome::Complete(v) => {
                    let v_arr = [*v];
                    rustuna_study_tell_multi(study, t_num, 1, v_arr.as_ptr(), 1, ptr::null());
                }
                TrialOutcome::Pruned => {
                    rustuna_study_tell_multi(study, t_num, 2, ptr::null(), 0, ptr::null());
                }
                TrialOutcome::Failed => {
                    rustuna_study_tell_multi(study, t_num, 4, ptr::null(), 0, ptr::null());
                }
            }
        }

        // 1. Fetch individual trials via filtered buffer
        let mut ind_ptrs: *mut *mut RustunaPersistedTrial = ptr::null_mut();
        let mut ind_len = 0usize;
        prop_assert_eq!(rustuna_study_get_trials_filtered(study, mask, &mut ind_ptrs, &mut ind_len), 0);

        let mut ind_json_strings = Vec::new();
        for i in 0..ind_len {
            unsafe {
                let pt = *ind_ptrs.add(i);
                let mut json_ptr: *mut std::os::raw::c_char = ptr::null_mut();
                prop_assert_eq!(rustuna_persisted_trial_get_json(pt, &mut json_ptr), 0);
                let s = CStr::from_ptr(json_ptr).to_string_lossy().into_owned();
                ind_json_strings.push(s);
                rustuna_string_free(json_ptr);
            }
        }
        rustuna_trials_buffer_free(ind_ptrs, ind_len);

        // 2. Fetch batched JSON array
        let mut batch_json_ptr: *mut std::os::raw::c_char = ptr::null_mut();
        let mut batch_len = 0usize;
        prop_assert_eq!(rustuna_study_get_trials_json_since(study, mask, 0, &mut batch_json_ptr, &mut batch_len), 0);
        prop_assert!(!batch_json_ptr.is_null());

        unsafe {
            let batch_str = CStr::from_ptr(batch_json_ptr).to_string_lossy();
            prop_assert_eq!(batch_len, batch_str.len());
            let batch_array: Vec<Value> = serde_json::from_str(&batch_str).unwrap();

            // Assert lengths match
            prop_assert_eq!(batch_array.len(), ind_len);

            // Assert each item matches bit-for-bit
            for (i, item) in batch_array.iter().enumerate() {
                let ind_val: Value = serde_json::from_str(&ind_json_strings[i]).unwrap();
                prop_assert_eq!(&item["number"], &ind_val["number"]);
                prop_assert_eq!(&item["state"], &ind_val["state"]);
                prop_assert_eq!(&item["values"], &ind_val["values"]);
                prop_assert_eq!(&item["user_attrs"], &ind_val["user_attrs"]);
            }

            rustuna_string_free(batch_json_ptr);
        }

        rustuna_study_free(study);
    }
}
