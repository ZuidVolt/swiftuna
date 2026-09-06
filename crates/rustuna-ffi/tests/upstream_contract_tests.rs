use rustuna_ffi::*;
use serde_json::Value;
use std::ffi::{CStr, CString};
use std::ptr;

#[test]
fn test_tpe_sampler_determinism() {
    let run_optimization = |seed: u64| -> Vec<f64> {
        let sampler = rustuna_sampler_tpe_full(seed, true, -1, 10);
        let name = CString::new("tpe_det").unwrap();
        let directions = [1i32]; // MINIMIZE
        let mut study: *mut RustunaStudy = ptr::null_mut();
        assert_eq!(
            rustuna_study_create_full(
                name.as_ptr(),
                directions.as_ptr(),
                1,
                0,
                ptr::null(),
                false,
                sampler,
                &mut study
            ),
            0
        );

        let param_name = CString::new("x").unwrap();
        let mut samples = Vec::new();

        for _ in 0..10 {
            let mut trial: *mut RustunaTrial = ptr::null_mut();
            assert_eq!(rustuna_study_ask(study, &mut trial), 0);
            let mut x = 0.0f64;
            assert_eq!(
                rustuna_trial_suggest_float(
                    trial,
                    param_name.as_ptr(),
                    -10.0,
                    10.0,
                    0.0,
                    false,
                    &mut x
                ),
                0
            );
            samples.push(x);
            let t_num = rustuna_trial_get_number(trial);
            rustuna_trial_free(trial);

            let obj = x * x;
            assert_eq!(
                rustuna_study_tell_multi(study, t_num, 1, &obj, 1, ptr::null()),
                0
            );
        }

        rustuna_study_free(study);
        samples
    };

    let run1 = run_optimization(999);
    let run2 = run_optimization(999);
    assert_eq!(
        run1, run2,
        "TPE with identical seed must produce identical parameter sequences"
    );

    let run3 = run_optimization(111);
    assert_ne!(
        run1, run3,
        "TPE with different seeds should produce different sequences"
    );
}

#[test]
fn test_storage_cross_copy_parity() {
    // 1. Create In-Memory study and run 5 trials
    let name_mem = CString::new("source_mem_study").unwrap();
    let directions = [1i32];
    let mut study_mem: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name_mem.as_ptr(),
            directions.as_ptr(),
            1,
            0, // In-Memory
            ptr::null(),
            false,
            ptr::null_mut(),
            &mut study_mem
        ),
        0
    );

    let param_name = CString::new("param_a").unwrap();
    let attr_k = CString::new("framework").unwrap();
    let attr_v = CString::new("swiftuna").unwrap();

    for i in 0..5 {
        let mut trial: *mut RustunaTrial = ptr::null_mut();
        assert_eq!(rustuna_study_ask(study_mem, &mut trial), 0);
        let mut a = 0.0f64;
        assert_eq!(
            rustuna_trial_suggest_float(trial, param_name.as_ptr(), 0.0, 100.0, 0.0, false, &mut a),
            0
        );
        assert_eq!(
            rustuna_trial_set_user_attr(trial, attr_k.as_ptr(), attr_v.as_ptr()),
            0
        );
        let t_num = rustuna_trial_get_number(trial);
        rustuna_trial_free(trial);

        let val = (i as f64) * 2.5;
        assert_eq!(
            rustuna_study_tell_multi(study_mem, t_num, 1, &val, 1, ptr::null()),
            0
        );
    }

    // 2. Copy study from In-Memory to a temporary SQLite database
    let db_path = format!(
        "/tmp/test_cross_copy_{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let c_db_path = CString::new(db_path.as_str()).unwrap();
    let name_target = CString::new("copied_sqlite_study").unwrap();

    let mut copied_study: *mut RustunaStudy = ptr::null_mut();
    let copy_res = rustuna_study_copy(
        study_mem,
        1, // SQLite
        c_db_path.as_ptr(),
        name_target.as_ptr(),
        &mut copied_study,
    );
    assert_eq!(copy_res, 0);
    assert!(!copied_study.is_null());
    rustuna_study_free(copied_study);
    rustuna_study_free(study_mem);

    // 3. Reload from SQLite and verify all 5 trials are preserved
    let mut reloaded_study: *mut RustunaStudy = ptr::null_mut();
    let load_res = rustuna_study_load(
        name_target.as_ptr(),
        1, // SQLite
        c_db_path.as_ptr(),
        ptr::null_mut(),
        &mut reloaded_study,
    );
    assert_eq!(load_res, 0);
    assert!(!reloaded_study.is_null());

    let mut trials_buf: *mut *mut RustunaPersistedTrial = ptr::null_mut();
    let mut trials_len = 0usize;
    assert_eq!(
        rustuna_study_get_trials(reloaded_study, &mut trials_buf, &mut trials_len),
        0
    );
    assert_eq!(trials_len, 5);

    // Verify first trial
    unsafe {
        let first_pt = *trials_buf;
        let mut json_ptr: *mut std::os::raw::c_char = ptr::null_mut();
        assert_eq!(rustuna_persisted_trial_get_json(first_pt, &mut json_ptr), 0);
        let json_str = CStr::from_ptr(json_ptr).to_string_lossy();
        let parsed: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(parsed["number"].as_u64().unwrap(), 0);
        assert_eq!(parsed["state"].as_i64().unwrap(), 1); // Complete
        assert_eq!(
            parsed["user_attrs"]["framework"].as_str().unwrap(),
            "swiftuna"
        );
        rustuna_string_free(json_ptr);
    }

    rustuna_trials_buffer_free(trials_buf, trials_len);
    rustuna_study_free(reloaded_study);

    // Cleanup temp db
    let _ = std::fs::remove_file(&db_path);
}

#[test]
fn test_ped_anova_param_importance_contract() {
    let name = CString::new("importance_study").unwrap();
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

    let name_dominant = CString::new("dominant_param").unwrap();
    let name_noise = CString::new("noise_param").unwrap();

    // Objective = 100 * dominant + 0.01 * noise
    for i in 0..15 {
        let mut trial: *mut RustunaTrial = ptr::null_mut();
        assert_eq!(rustuna_study_ask(study, &mut trial), 0);

        let mut dom = 0.0f64;
        let mut noise = 0.0f64;
        rustuna_trial_suggest_float(
            trial,
            name_dominant.as_ptr(),
            0.0,
            10.0,
            0.0,
            false,
            &mut dom,
        );
        rustuna_trial_suggest_float(
            trial,
            name_noise.as_ptr(),
            0.0,
            10.0,
            0.0,
            false,
            &mut noise,
        );
        let t_num = rustuna_trial_get_number(trial);
        rustuna_trial_free(trial);

        let val = 100.0 * dom + 0.01 * noise + (i as f64) * 0.001;
        assert_eq!(
            rustuna_study_tell_multi(study, t_num, 1, &val, 1, ptr::null()),
            0
        );
    }

    let mut out_json: *mut std::os::raw::c_char = ptr::null_mut();
    let imp_res = rustuna_study_get_param_importances(study, true, ptr::null(), &mut out_json);
    assert_eq!(imp_res, 0);
    assert!(!out_json.is_null());

    unsafe {
        let json_str = CStr::from_ptr(out_json).to_string_lossy();
        let importances: std::collections::HashMap<String, f64> =
            serde_json::from_str(&json_str).unwrap();

        let dom_imp = importances.get("dominant_param").copied().unwrap_or(0.0);
        let noise_imp = importances.get("noise_param").copied().unwrap_or(0.0);

        assert!(
            dom_imp > noise_imp,
            "Dominant parameter importance ({dom_imp}) must be greater than noise ({noise_imp})"
        );

        // Normalized sum should be ~1.0
        let total: f64 = importances.values().sum();
        assert!(
            (total - 1.0).abs() < 0.05,
            "Normalized importances must sum to approximately 1.0 (got {total})"
        );

        rustuna_string_free(out_json);
    }

    rustuna_study_free(study);
}

#[test]
fn test_grid_sampler_ordering_and_exhaustion_contract() {
    let space_json = CString::new(r#"{"x": [1.0, 2.0], "y": [10.0, 20.0]}"#).unwrap();
    let mut sampler: *mut RustunaSampler = ptr::null_mut();
    assert_eq!(
        rustuna_sampler_grid_new(space_json.as_ptr(), 0, false, &mut sampler),
        0
    );

    let name = CString::new("grid_contract").unwrap();
    let directions = [0i32]; // MINIMIZE
    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0,
            ptr::null(),
            false,
            sampler,
            &mut study
        ),
        0
    );

    let name_x = CString::new("x").unwrap();
    let name_y = CString::new("y").unwrap();
    let mut visited = Vec::new();

    for _ in 0..4 {
        let mut trial: *mut RustunaTrial = ptr::null_mut();
        assert_eq!(rustuna_study_ask(study, &mut trial), 0);
        let mut x = 0.0f64;
        let mut y = 0.0f64;
        assert_eq!(
            rustuna_trial_suggest_float(trial, name_x.as_ptr(), 0.0, 100.0, 0.0, false, &mut x),
            0
        );
        assert_eq!(
            rustuna_trial_suggest_float(trial, name_y.as_ptr(), 0.0, 100.0, 0.0, false, &mut y),
            0
        );
        let t_num = rustuna_trial_get_number(trial);
        rustuna_trial_free(trial);
        visited.push((x, y));

        let val = x + y;
        assert_eq!(
            rustuna_study_tell_multi(study, t_num, 1, &val, 1, ptr::null()),
            0
        );
    }

    assert_eq!(visited.len(), 4, "Grid must yield all 4 combinations");
    assert_eq!(
        visited,
        vec![(1.0, 10.0), (1.0, 20.0), (2.0, 10.0), (2.0, 20.0)],
        "Unshuffled grid must traverse in deterministic lexicographical order"
    );

    // The 5th ask must fail with searchSpaceExhausted (code 21)
    let mut trial: *mut RustunaTrial = ptr::null_mut();
    let ask_res = rustuna_study_ask(study, &mut trial);
    assert_ne!(ask_res, 0, "Exhausted grid must fail ask");
    assert_eq!(
        rustuna_last_error_code(),
        21,
        "Exhausted grid must produce code 21 (searchSpaceExhausted)"
    );

    rustuna_study_free(study);
}

#[test]
fn test_trial_state_machine_and_json_wire_contract() {
    let name = CString::new("state_machine_wire_study").unwrap();
    let directions = [0i32];
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

    let param_name = CString::new("alpha").unwrap();
    let mut alpha = 0.0f64;
    assert_eq!(
        rustuna_trial_suggest_float(trial, param_name.as_ptr(), 0.0, 1.0, 0.0, false, &mut alpha),
        0
    );

    // Constraint setting
    let c_name = CString::new("latency_c").unwrap();
    assert_eq!(
        rustuna_trial_set_constraint(trial, c_name.as_ptr(), -0.5),
        0
    );

    // User attribute
    let u_key = CString::new("worker").unwrap();
    let u_val = CString::new("node_1").unwrap();
    assert_eq!(
        rustuna_trial_set_user_attr(trial, u_key.as_ptr(), u_val.as_ptr()),
        0
    );

    let t_num = rustuna_trial_get_number(trial);
    rustuna_trial_free(trial);

    // Complete the trial with intermediate values
    let val = 42.0;
    let inter_json = CString::new(r#"{"0": 0.5, "1": 0.25}"#).unwrap();
    assert_eq!(
        rustuna_study_tell_multi(study, t_num, 1, &val, 1, inter_json.as_ptr()),
        0
    );

    // Re-telling the same completed trial must fail with TrialAlreadyFinished (code 12)
    let double_tell = rustuna_study_tell_multi(study, t_num, 1, &val, 1, ptr::null());
    assert_ne!(double_tell, 0, "Double tell must be rejected");
    assert_eq!(
        rustuna_last_error_code(),
        12,
        "Double tell must fail with TrialAlreadyFinished (code 12)"
    );

    // Inspect persisted trial JSON schema
    let mut trials_buf: *mut *mut RustunaPersistedTrial = ptr::null_mut();
    let mut count = 0usize;
    assert_eq!(
        rustuna_study_get_trials(study, &mut trials_buf, &mut count),
        0
    );
    assert_eq!(count, 1);

    unsafe {
        let pt = *trials_buf;
        let mut json_ptr: *mut std::os::raw::c_char = ptr::null_mut();
        assert_eq!(rustuna_persisted_trial_get_json(pt, &mut json_ptr), 0);
        let json_str = CStr::from_ptr(json_ptr).to_string_lossy();
        let parsed: Value = serde_json::from_str(&json_str).unwrap();

        // Pin the wire schema keys that Swiftuna relies on
        assert!(parsed.get("number").is_some(), "JSON wire must include number");
        assert!(parsed.get("state").is_some(), "JSON wire must include state");
        assert!(parsed.get("values").is_some(), "JSON wire must include values");
        assert!(parsed.get("params").is_some(), "JSON wire must include params");
        assert!(parsed.get("param_values").is_some(), "JSON wire must include param_values");
        assert!(parsed.get("user_attrs").is_some(), "JSON wire must include user_attrs");
        assert!(parsed.get("constraints").is_some(), "JSON wire must include constraints");
        assert!(parsed.get("intermediate_values").is_some(), "JSON wire must include intermediate_values");
        assert!(parsed.get("datetime_start").is_some(), "JSON wire must include datetime_start");
        assert!(parsed.get("datetime_complete").is_some(), "JSON wire must include datetime_complete");

        assert_eq!(parsed["state"].as_i64().unwrap(), 1, "Complete state wire integer is 1");
        assert_eq!(parsed["params"]["alpha"].as_f64().unwrap(), alpha);
        assert_eq!(parsed["constraints"]["latency_c"].as_f64().unwrap(), -0.5);
        assert_eq!(parsed["user_attrs"]["worker"].as_str().unwrap(), "node_1");
        assert_eq!(parsed["intermediate_values"]["0"].as_f64().unwrap(), 0.5);
        assert_eq!(parsed["intermediate_values"]["1"].as_f64().unwrap(), 0.25);

        rustuna_string_free(json_ptr);
    }
    rustuna_trials_buffer_free(trials_buf, count);

    rustuna_study_free(study);
}

#[test]
fn test_nsgaii_pareto_dominance_contract() {
    let name = CString::new("nsgaii_pareto_contract").unwrap();
    let directions = [0i32, 0i32]; // [MINIMIZE, MINIMIZE]
    let mut sampler: *mut RustunaSampler = ptr::null_mut();
    assert_eq!(
        rustuna_sampler_nsgaii_new(10, 0.0, 0.0, 0.0, 42, &mut sampler),
        0
    );

    let mut study: *mut RustunaStudy = ptr::null_mut();
    assert_eq!(
        rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            2,
            0,
            ptr::null(),
            false,
            sampler,
            &mut study
        ),
        0
    );

    let param_name = CString::new("t").unwrap();

    // Optimize a 2-objective problem: obj1 = t, obj2 = 1.0 - t
    // Any point t in [0.0, 1.0] is Pareto-optimal because reducing obj1 increases obj2
    for _ in 0..15 {
        let mut trial: *mut RustunaTrial = ptr::null_mut();
        assert_eq!(rustuna_study_ask(study, &mut trial), 0);

        let mut t = 0.0f64;
        assert_eq!(
            rustuna_trial_suggest_float(trial, param_name.as_ptr(), 0.0, 1.0, 0.0, false, &mut t),
            0
        );
        let t_num = rustuna_trial_get_number(trial);
        rustuna_trial_free(trial);

        let objs = [t, 1.0 - t];
        assert_eq!(
            rustuna_study_tell_multi(study, t_num, 1, objs.as_ptr(), 2, ptr::null()),
            0
        );
    }

    // Query Pareto best trials
    let mut trials_buf: *mut *mut RustunaPersistedTrial = ptr::null_mut();
    let mut count = 0usize;
    assert_eq!(
        rustuna_study_get_best_trials(study, &mut trials_buf, &mut count),
        0
    );
    assert!(count > 0, "Pareto frontier must contain at least one trial");

    // Verify non-dominance invariant: no trial in the set may strictly dominate another
    let mut pareto_points = Vec::new();
    unsafe {
        for i in 0..count {
            let pt = *trials_buf.add(i);
            let mut json_ptr: *mut std::os::raw::c_char = ptr::null_mut();
            assert_eq!(rustuna_persisted_trial_get_json(pt, &mut json_ptr), 0);
            let json_str = CStr::from_ptr(json_ptr).to_string_lossy();
            let parsed: Value = serde_json::from_str(&json_str).unwrap();
            let v1 = parsed["values"][0].as_f64().unwrap();
            let v2 = parsed["values"][1].as_f64().unwrap();
            pareto_points.push((v1, v2));
            rustuna_string_free(json_ptr);
        }
    }
    rustuna_trials_buffer_free(trials_buf, count);

    for (i, p1) in pareto_points.iter().enumerate() {
        for (j, p2) in pareto_points.iter().enumerate() {
            if i != j {
                // p1 cannot strictly dominate p2: it cannot be <= in all objectives and < in at least one
                let dominates = (p1.0 <= p2.0 && p1.1 <= p2.1) && (p1.0 < p2.0 || p1.1 < p2.1);
                assert!(
                    !dominates,
                    "Pareto frontier violation: point {i} {:?} dominates point {j} {:?}",
                    p1, p2
                );
            }
        }
    }

    rustuna_study_free(study);
}
