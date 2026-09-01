use proptest::prelude::*;
use rustuna_ffi::*;
use serde_json::json;
use std::collections::HashSet;
use std::ffi::CString;
use std::ptr;

proptest! {
    #[test]
    fn prop_grid_sampler_exact_cartesian_completeness(
        floats in prop::collection::vec(-10.0..10.0f64, 1..4),
        ints in prop::collection::vec(-50..50i64, 1..4),
        categories in prop::collection::vec("[a-z]{3}", 1..3),
    ) {
        // Ensure values within each dimension are unique for proper combinatorics count
        let mut u_floats = floats.clone();
        u_floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        u_floats.dedup();

        let mut u_ints = ints.clone();
        u_ints.sort();
        u_ints.dedup();

        let mut u_cats = categories.clone();
        u_cats.sort();
        u_cats.dedup();

        let expected_total = u_floats.len() * u_ints.len() * u_cats.len();

        let cat_indices: Vec<f64> = (0..u_cats.len()).map(|i| i as f64).collect();
        let int_floats: Vec<f64> = u_ints.iter().map(|&i| i as f64).collect();

        let search_space = json!({
            "param_f": u_floats,
            "param_i": int_floats,
            "param_c": cat_indices
        });
        let space_json = CString::new(search_space.to_string()).unwrap();

        let mut sampler: *mut RustunaSampler = ptr::null_mut();
        let code = rustuna_sampler_grid_new(space_json.as_ptr(), 42, true, &mut sampler);
        prop_assert_eq!(code, 0);
        prop_assert!(!sampler.is_null());

        let name = CString::new("prop_grid_completeness").unwrap();
        let directions = [1i32];
        let mut study: *mut RustunaStudy = ptr::null_mut();
        let s_code = rustuna_study_create_full(
            name.as_ptr(),
            directions.as_ptr(),
            1,
            0, // In-memory
            ptr::null(),
            false,
            sampler,
            &mut study,
        );
        prop_assert_eq!(s_code, 0);

        let mut visited_combinations = HashSet::new();
        let mut trial_count = 0;

        let name_f = CString::new("param_f").unwrap();
        let name_i = CString::new("param_i").unwrap();
        let name_c = CString::new("param_c").unwrap();

        let cat_c_ptrs: Vec<CString> = u_cats.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
        let cat_ptrs: Vec<*const std::os::raw::c_char> = cat_c_ptrs.iter().map(|c| c.as_ptr()).collect();

        loop {
            let mut trial: *mut RustunaTrial = ptr::null_mut();
            let ask_res = rustuna_study_ask(study, &mut trial);
            if ask_res != 0 {
                // Search space exhausted
                break;
            }

            let mut f_val = 0.0f64;
            let mut i_val = 0i64;
            let mut c_idx = 0usize;

            rustuna_trial_suggest_float(trial, name_f.as_ptr(), -10.0, 10.0, 0.0, false, &mut f_val);
            rustuna_trial_suggest_int(trial, name_i.as_ptr(), -50, 50, 1, false, &mut i_val);
            rustuna_trial_suggest_categorical(trial, name_c.as_ptr(), cat_ptrs.as_ptr(), cat_ptrs.len(), &mut c_idx);

            let combo_key = format!("{:.6}_{}_{}", f_val, i_val, c_idx);
            visited_combinations.insert(combo_key);

            let trial_num = rustuna_trial_get_number(trial);
            rustuna_trial_free(trial);

            let obj_val = 0.0f64;
            rustuna_study_tell_multi(study, trial_num, 1, &obj_val, 1, ptr::null());
            trial_count += 1;
        }

        prop_assert_eq!(trial_count, expected_total);
        prop_assert_eq!(visited_combinations.len(), expected_total);

        rustuna_study_free(study);
    }

    #[test]
    fn prop_grid_sampler_deterministic_seed_reproducibility(seed in any::<u64>()) {
        let search_space = json!({
            "x": [-5.0, 0.0, 5.0],
            "y": [10.0, 20.0, 30.0],
            "z": [0.0, 1.0]
        });
        let space_json = CString::new(search_space.to_string()).unwrap();

        // Run 1
        let mut run1_samples = Vec::new();
        {
            let mut sampler1: *mut RustunaSampler = ptr::null_mut();
            rustuna_sampler_grid_new(space_json.as_ptr(), seed, true, &mut sampler1);
            let name = CString::new("run1").unwrap();
            let directions = [1i32];
            let mut study1: *mut RustunaStudy = ptr::null_mut();
            rustuna_study_create_full(name.as_ptr(), directions.as_ptr(), 1, 0, ptr::null(), false, sampler1, &mut study1);

            let name_x = CString::new("x").unwrap();
            let name_y = CString::new("y").unwrap();

            for _ in 0..18 {
                let mut trial: *mut RustunaTrial = ptr::null_mut();
                if rustuna_study_ask(study1, &mut trial) == 0 {
                    let mut x = 0.0f64;
                    let mut y = 0i64;
                    rustuna_trial_suggest_float(trial, name_x.as_ptr(), -5.0, 5.0, 0.0, false, &mut x);
                    rustuna_trial_suggest_int(trial, name_y.as_ptr(), 10, 30, 1, false, &mut y);
                    run1_samples.push((x, y));
                    let t_num = rustuna_trial_get_number(trial);
                    rustuna_trial_free(trial);
                    let obj = 1.0f64;
                    rustuna_study_tell_multi(study1, t_num, 1, &obj, 1, ptr::null());
                }
            }
            rustuna_study_free(study1);
        }

        // Run 2 with identical seed
        let mut run2_samples = Vec::new();
        {
            let mut sampler2: *mut RustunaSampler = ptr::null_mut();
            rustuna_sampler_grid_new(space_json.as_ptr(), seed, true, &mut sampler2);
            let name = CString::new("run2").unwrap();
            let directions = [1i32];
            let mut study2: *mut RustunaStudy = ptr::null_mut();
            rustuna_study_create_full(name.as_ptr(), directions.as_ptr(), 1, 0, ptr::null(), false, sampler2, &mut study2);

            let name_x = CString::new("x").unwrap();
            let name_y = CString::new("y").unwrap();

            for _ in 0..18 {
                let mut trial: *mut RustunaTrial = ptr::null_mut();
                if rustuna_study_ask(study2, &mut trial) == 0 {
                    let mut x = 0.0f64;
                    let mut y = 0i64;
                    rustuna_trial_suggest_float(trial, name_x.as_ptr(), -5.0, 5.0, 0.0, false, &mut x);
                    rustuna_trial_suggest_int(trial, name_y.as_ptr(), 10, 30, 1, false, &mut y);
                    run2_samples.push((x, y));
                    let t_num = rustuna_trial_get_number(trial);
                    rustuna_trial_free(trial);
                    let obj = 1.0f64;
                    rustuna_study_tell_multi(study2, t_num, 1, &obj, 1, ptr::null());
                }
            }
            rustuna_study_free(study2);
        }

        prop_assert_eq!(run1_samples, run2_samples);
    }
}
