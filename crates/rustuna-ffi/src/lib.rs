#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, RwLock};

use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::{RandomSampler, Sampler};
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::{create_study_with_arc, get_best_trial, get_pareto_front, Direction, Study};
use rustuna_core::trial::{PersistedTrial, Trial, TrialStateValues};
use rustuna_sampler::nsgaii::NSGAIISampler;
use rustuna_sampler::qmc::QmcSampler;
use rustuna_sampler::tpe::TpeSampler;
use rustuna_storage::cache::CachedStorage;
use rustuna_storage::journal::file::JournalFileBackend;
use rustuna_storage::journal::storage::JournalStorage;
use rustuna_storage::sqlite3::SQLite3Storage;

fn error_kind_to_code(kind: &rustuna_core::ErrorKind) -> i32 {
    match kind {
        rustuna_core::ErrorKind::ObjectiveError => 1,
        rustuna_core::ErrorKind::SamplerError => 2,
        rustuna_core::ErrorKind::StorageError => 3,
        rustuna_core::ErrorKind::DuplicatedStudy => 4,
        rustuna_core::ErrorKind::StudyNotFound => 5,
        rustuna_core::ErrorKind::TrialNotFound => 6,
        rustuna_core::ErrorKind::TrialDiscarded => 7,
        rustuna_core::ErrorKind::AttrNotFound => 8,
        rustuna_core::ErrorKind::TrialQueueEmpty => 9,
        rustuna_core::ErrorKind::AttrOverwriteNotAllowed => 10,
        rustuna_core::ErrorKind::InvalidObjectiveValues => 11,
        rustuna_core::ErrorKind::TrialAlreadyFinished => 12,
        rustuna_core::ErrorKind::UnsupportedSearchSpace => 13,
        rustuna_core::ErrorKind::UnsupportedMultiObjective => 14,
        rustuna_core::ErrorKind::NoCompletedTrial => 15,
        rustuna_core::ErrorKind::IncompatibleDistribution => 16,
        rustuna_core::ErrorKind::InvalidFixedParam => 17,
        rustuna_core::ErrorKind::MissingDependency => 18,
        rustuna_core::ErrorKind::Unexpected => 19,
        rustuna_core::ErrorKind::ImportanceEvaluatorError => 20,
        _ => 19,
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<(i32, CString)>> = const { RefCell::new(None) };
}

fn set_last_error(code: i32, msg: String) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok().map(|cs| (code, cs));
    });
}

fn set_rustuna_error(err: &rustuna_core::Error) -> i32 {
    let code = error_kind_to_code(&err.kind);
    let msg = if err.reason.is_empty() {
        format!("{:?}", err.kind)
    } else {
        format!("{:?}: {}", err.kind, err.reason)
    };
    set_last_error(code, msg);
    code
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_last_error_code() -> i32 {
    LAST_ERROR.with(|cell| cell.borrow().as_ref().map_or(0, |(c, _)| *c))
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |(_, s)| s.as_ptr())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

pub struct RustunaSampler {
    pub inner: Arc<dyn Sampler>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_tpe_new(seed: u64, has_seed: bool) -> *mut RustunaSampler {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let sampler = if has_seed {
            TpeSampler::seed_from_u64(seed)
        } else {
            TpeSampler::new()
        };
        Box::into_raw(Box::new(RustunaSampler {
            inner: Arc::new(sampler),
        }))
    }));
    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            set_last_error(-99, format!("Panic creating TPE sampler: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_random_new(seed: u64, has_seed: bool) -> *mut RustunaSampler {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let sampler = if has_seed {
            RandomSampler::seed_from_u64(seed)
        } else {
            RandomSampler::new()
        };
        Box::into_raw(Box::new(RustunaSampler {
            inner: Arc::new(sampler),
        }))
    }));
    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            set_last_error(-99, format!("Panic creating Random sampler: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_free(sampler: *mut RustunaSampler) {
    if !sampler.is_null() {
        unsafe {
            drop(Box::from_raw(sampler));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_nsgaii_new(
    population_size: usize,
    mutation_prob: f64,
    crossover_prob: f64,
    swapping_prob: f64,
    seed: u64,
    out_sampler: *mut *mut RustunaSampler,
) -> i32 {
    if out_sampler.is_null() {
        set_last_error(-1, "out_sampler pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let pop_size = if population_size > 0 { population_size } else { 50 };
        let mut_prob = if mutation_prob > 0.0 { Some(mutation_prob) } else { None };
        let cross_prob = if crossover_prob > 0.0 { crossover_prob } else { 0.9 };
        let swap_prob = if swapping_prob > 0.0 { swapping_prob } else { 0.5 };

        let sampler = if seed != 0 {
            NSGAIISampler::seed_from_u64(seed, pop_size, mut_prob, cross_prob, swap_prob)
        } else {
            NSGAIISampler::new(pop_size, mut_prob, cross_prob, swap_prob)
        };

        let boxed = Box::new(RustunaSampler {
            inner: Arc::new(sampler),
        });
        unsafe {
            *out_sampler = Box::into_raw(boxed);
        }
        0
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_sampler_nsgaii_new: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_qmc_new(seed: u64, has_seed: bool) -> *mut RustunaSampler {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let sampler = if has_seed {
            QmcSampler::seed_from_u64(seed)
        } else {
            QmcSampler::new()
        };
        Box::into_raw(Box::new(RustunaSampler {
            inner: Arc::new(sampler),
        }))
    }));
    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            set_last_error(-99, format!("Panic creating QMC sampler: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

pub struct RustunaStudy {
    pub inner: Study,
}

fn create_storage_backend(
    storage_type: i32,
    path: &str,
) -> Result<Arc<RwLock<dyn Storage>>, rustuna_core::Error> {
    match storage_type {
        1 => {
            let backend = SQLite3Storage::new(path)?;
            backend.create_database()?;
            Ok(Arc::new(RwLock::new(CachedStorage::new(Box::new(backend)))))
        }
        2 => {
            let file_backend = JournalFileBackend::new(path, None)?;
            let backend = JournalStorage::new(Box::new(file_backend))?;
            Ok(Arc::new(RwLock::new(backend)))
        }
        _ => Ok(Arc::new(RwLock::new(InMemoryStorage::new()))),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_create_full(
    name: *const c_char,
    directions: *const i32,
    directions_len: usize,
    storage_type: i32, // 0 = InMemory, 1 = SQLite, 2 = Journal
    storage_path: *const c_char,
    load_if_exists: bool,
    sampler: *mut RustunaSampler,
    out_study: *mut *mut RustunaStudy,
) -> i32 {
    if out_study.is_null() {
        set_last_error(-1, "out_study pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let study_name = if name.is_null() {
            "default".to_string()
        } else {
            unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
        };

        let dirs: Vec<Direction> = if !directions.is_null() && directions_len > 0 {
            let slice = unsafe { std::slice::from_raw_parts(directions, directions_len) };
            slice.iter().map(|&d| match d {
                1 => Direction::Maximize,
                _ => Direction::Minimize,
            }).collect()
        } else {
            vec![Direction::Minimize]
        };

        let sampler_arc: Arc<dyn Sampler> = if !sampler.is_null() {
            unsafe { (*sampler).inner.clone() }
        } else if dirs.len() > 1 {
            Arc::new(NSGAIISampler::default())
        } else {
            Arc::new(TpeSampler::new())
        };

        let storage_path_str = if !storage_path.is_null() {
            unsafe { CStr::from_ptr(storage_path) }.to_string_lossy().into_owned()
        } else {
            "".to_string()
        };

        let storage = match create_storage_backend(storage_type, &storage_path_str) {
            Ok(s) => s,
            Err(e) => return set_rustuna_error(&e),
        };

        match create_study_with_arc(&study_name, storage.clone(), sampler_arc.clone(), dirs) {
            Ok(study) => {
                let boxed = Box::new(RustunaStudy { inner: study });
                unsafe {
                    *out_study = Box::into_raw(boxed);
                }
                0
            }
            Err(e) if load_if_exists && matches!(e.kind, rustuna_core::ErrorKind::DuplicatedStudy) => {
                match Study::from_name(study_name, storage, sampler_arc) {
                    Ok(study) => {
                        let boxed = Box::new(RustunaStudy { inner: study });
                        unsafe {
                            *out_study = Box::into_raw(boxed);
                        }
                        0
                    }
                    Err(e) => set_rustuna_error(&e),
                }
            }
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_create_full: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_load(
    name: *const c_char,
    storage_type: i32,
    storage_path: *const c_char,
    sampler: *mut RustunaSampler,
    out_study: *mut *mut RustunaStudy,
) -> i32 {
    if name.is_null() || out_study.is_null() {
        set_last_error(-1, "name or out_study pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let study_name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();

        let storage_path_str = if !storage_path.is_null() {
            unsafe { CStr::from_ptr(storage_path) }.to_string_lossy().into_owned()
        } else {
            "".to_string()
        };

        let storage = match create_storage_backend(storage_type, &storage_path_str) {
            Ok(s) => s,
            Err(e) => return set_rustuna_error(&e),
        };

        let sampler_arc: Arc<dyn Sampler> = if !sampler.is_null() {
            unsafe { (*sampler).inner.clone() }
        } else {
            Arc::new(TpeSampler::new())
        };

        match Study::from_name(study_name, storage, sampler_arc) {
            Ok(study) => {
                let boxed = Box::new(RustunaStudy { inner: study });
                unsafe {
                    *out_study = Box::into_raw(boxed);
                }
                0
            }
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_load: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_free(study: *mut RustunaStudy) {
    if !study.is_null() {
        unsafe {
            drop(Box::from_raw(study));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_set_user_attr(
    study: *mut RustunaStudy,
    key: *const c_char,
    val: *const c_char,
) -> i32 {
    if study.is_null() || key.is_null() || val.is_null() {
        set_last_error(-1, "study, key, or val pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let k = unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();
        let v = unsafe { CStr::from_ptr(val) }.to_string_lossy().into_owned();
        let mut map = HashMap::new();
        map.insert(k, v);
        match s.inner.set_user_attr(map) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_set_user_attr: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_get_user_attr(
    study: *mut RustunaStudy,
    key: *const c_char,
    out_val: *mut *mut c_char,
) -> i32 {
    if study.is_null() || key.is_null() || out_val.is_null() {
        set_last_error(-1, "study, key, or out_val pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let k = unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();
        match s.inner.get_user_attr(k) {
            Ok(Some(val_str)) => {
                let c_str = CString::new(val_str).unwrap_or_default();
                unsafe {
                    *out_val = c_str.into_raw();
                }
                0
            }
            Ok(None) => {
                unsafe {
                    *out_val = std::ptr::null_mut();
                }
                0
            }
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_get_user_attr: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_enqueue_trial(
    study: *mut RustunaStudy,
    params_json: *const c_char,
    user_attrs_json: *const c_char,
) -> i32 {
    if study.is_null() || params_json.is_null() {
        set_last_error(-1, "study or params_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let pj_str = match unsafe { CStr::from_ptr(params_json) }.to_str() {
            Ok(str_ref) => str_ref,
            Err(_) => {
                set_last_error(-1, "params_json is not valid UTF-8".to_string());
                return -1;
            }
        };
        let parsed_params: serde_json::Value = match serde_json::from_str(pj_str) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(19, format!("Failed to parse params_json: {e:?}"));
                return 19;
            }
        };
        let param_obj = match parsed_params.as_object() {
            Some(o) => o,
            None => {
                set_last_error(-1, "params_json must be a JSON object".to_string());
                return -1;
            }
        };

        let mut fixed_params = HashMap::new();
        for (k, v) in param_obj {
            let label = match v {
                serde_json::Value::Number(num) => {
                    if let Some(i) = num.as_i64() {
                        rustuna_core::attr::CategoryLabel::Int(i)
                    } else if let Some(f) = num.as_f64() {
                        rustuna_core::attr::CategoryLabel::Float(f)
                    } else {
                        rustuna_core::attr::CategoryLabel::String(num.to_string())
                    }
                }
                serde_json::Value::String(s) => rustuna_core::attr::CategoryLabel::String(s.clone()),
                serde_json::Value::Bool(b) => rustuna_core::attr::CategoryLabel::Bool(*b),
                serde_json::Value::Null => rustuna_core::attr::CategoryLabel::None,
                _ => rustuna_core::attr::CategoryLabel::String(v.to_string()),
            };
            fixed_params.insert(k.clone(), label);
        }

        let user_attrs = if !user_attrs_json.is_null() {
            let uaj_str = unsafe { CStr::from_ptr(user_attrs_json) }
                .to_str()
                .unwrap_or("{}");
            let parsed_attrs: Result<HashMap<String, String>, _> = serde_json::from_str(uaj_str);
            match parsed_attrs {
                Ok(map) => {
                    let mut attrs = rustuna_core::attr::Attrs::new();
                    for (k, v) in map {
                        attrs.insert(rustuna_core::attr::AttrKey::User(k.into()), v);
                    }
                    Some(attrs)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        match s.inner.enqueue_trial(fixed_params, user_attrs) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_enqueue_trial: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_get_param_importances(
    study: *mut RustunaStudy,
    normalize: bool,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    if study.is_null() || out_json.is_null() {
        set_last_error(-1, "study or out_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let mut options = rustuna_importance::ImportanceOptions::new().normalize(normalize);

        if !params_json.is_null() {
            if let Ok(pj_str) = unsafe { CStr::from_ptr(params_json) }.to_str() {
                if !pj_str.is_empty() {
                    if let Ok(param_names) = serde_json::from_str::<Vec<String>>(pj_str) {
                        options = options.with_params(param_names);
                    }
                }
            }
        }

        let evaluator = rustuna_importance::PedAnovaImportanceEvaluator::default();
        match rustuna_importance::get_param_importances_with(&s.inner, &evaluator, options) {
            Ok(importances) => {
                match serde_json::to_string(&importances) {
                    Ok(json_str) => {
                        let c_str = CString::new(json_str).unwrap_or_default();
                        unsafe {
                            *out_json = c_str.into_raw();
                        }
                        0
                    }
                    Err(e) => {
                        set_last_error(19, format!("Failed to serialize importances: {e:?}"));
                        19
                    }
                }
            }
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_get_param_importances: {e:?}"));
            -99
        }
    }
}

pub struct RustunaTrial {
    pub inner: Trial,
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_ask(
    study: *mut RustunaStudy,
    out_trial: *mut *mut RustunaTrial,
) -> i32 {
    if study.is_null() || out_trial.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        match s.inner.ask() {
            Ok(trial) => {
                unsafe {
                    *out_trial = Box::into_raw(Box::new(RustunaTrial { inner: trial }));
                }
                0
            }
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_ask: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_tell(
    study: *mut RustunaStudy,
    trial_number: u32,
    state: i32,
    value: f64,
) -> i32 {
    rustuna_study_tell_multi(study, trial_number, state, &value, 1)
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_tell_multi(
    study: *mut RustunaStudy,
    trial_number: u32,
    state: i32,
    values: *const f64,
    values_len: usize,
) -> i32 {
    if study.is_null() {
        set_last_error(-1, "study pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let state_val = match state {
            1 => {
                if !values.is_null() && values_len > 0 {
                    let slice = unsafe { std::slice::from_raw_parts(values, values_len) };
                    TrialStateValues::Complete(slice.to_vec())
                } else {
                    TrialStateValues::Complete(vec![])
                }
            }
            2 => TrialStateValues::Pruned,
            4 => TrialStateValues::Fail,
            _ => TrialStateValues::Complete(vec![]),
        };
        match s.inner.tell(trial_number, state_val) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_tell_multi: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trial_get_number(trial: *mut RustunaTrial) -> u32 {
    if trial.is_null() {
        return 0;
    }
    unsafe { (*trial).inner.number }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trial_suggest_float(
    trial: *mut RustunaTrial,
    name: *const c_char,
    low: f64,
    high: f64,
    step: f64,
    log: bool,
    out_val: *mut f64,
) -> i32 {
    if trial.is_null() || name.is_null() || out_val.is_null() {
        set_last_error(-1, "trial, name, or out_val pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &mut *trial };
        let param_name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        let step_opt = if step > 0.0 { Some(step) } else { None };
        let dist = Distribution::new_float(low, high, step_opt, log);
        match t.inner.suggest(&param_name, &dist) {
            Ok(v) => {
                unsafe { *out_val = v };
                0
            }
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_trial_suggest_float: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trial_suggest_int(
    trial: *mut RustunaTrial,
    name: *const c_char,
    low: i64,
    high: i64,
    step: i64,
    log: bool,
    out_val: *mut i64,
) -> i32 {
    if trial.is_null() || name.is_null() || out_val.is_null() {
        set_last_error(-1, "trial, name, or out_val pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &mut *trial };
        let param_name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        let step_val = if step > 0 { step } else { 1 };
        let dist = Distribution::new_int(low, high, step_val, log);
        match t.inner.suggest(&param_name, &dist) {
            Ok(v) => {
                unsafe { *out_val = v as i64 };
                0
            }
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_trial_suggest_int: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trial_suggest_categorical(
    trial: *mut RustunaTrial,
    name: *const c_char,
    choices: *const *const c_char,
    choices_count: usize,
    out_index: *mut usize,
) -> i32 {
    if trial.is_null() || name.is_null() || choices.is_null() || out_index.is_null() || choices_count == 0 {
        set_last_error(-1, "trial, name, choices, or out_index pointer is null or choices_count is 0".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &mut *trial };
        let param_name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        let mut category_labels = Vec::with_capacity(choices_count);
        for i in 0..choices_count {
            let choice_ptr = unsafe { *choices.add(i) };
            if choice_ptr.is_null() {
                set_last_error(-1, format!("Choice at index {i} is null"));
                return -1;
            }
            let choice_str = unsafe { CStr::from_ptr(choice_ptr) }.to_string_lossy().into_owned();
            category_labels.push(rustuna_core::attr::CategoryLabel::String(choice_str));
        }

        let dist_choice = match t.inner.suggest_categorical_enum(&param_name, &category_labels) {
            Ok(c) => c,
            Err(e) => return set_rustuna_error(&e),
        };
        let chosen_idx = category_labels.iter().position(|l| l == dist_choice).unwrap_or(0);
        unsafe { *out_index = chosen_idx };
        0
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_trial_suggest_categorical: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trial_set_user_attr(
    trial: *mut RustunaTrial,
    key: *const c_char,
    val: *const c_char,
) -> i32 {
    if trial.is_null() || key.is_null() || val.is_null() {
        set_last_error(-1, "trial, key, or val pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &mut *trial };
        let k = unsafe { CStr::from_ptr(key) }.to_string_lossy();
        let v = unsafe { CStr::from_ptr(val) }.to_string_lossy().into_owned();
        match t.inner.set_user_attr(&k, v) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_trial_set_user_attr: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trial_set_constraint(
    trial: *mut RustunaTrial,
    key: *const c_char,
    value: f64,
) -> i32 {
    if trial.is_null() || key.is_null() {
        set_last_error(-1, "trial or key pointer is null".to_string());
        return -1;
    }
    if value.is_nan() {
        set_last_error(-1, "Constraint value cannot be NaN".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &mut *trial };
        let k = unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();
        match t.inner.set_constraints(HashMap::from([(k, value)])) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_trial_set_constraint: {e:?}"));
            -99
        }
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trial_free(trial: *mut RustunaTrial) {
    if !trial.is_null() {
        unsafe {
            drop(Box::from_raw(trial));
        }
    }
}

pub struct RustunaPersistedTrial {
    pub inner: PersistedTrial,
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_get_best_trial(
    study: *mut RustunaStudy,
    out_trial: *mut *mut RustunaPersistedTrial,
) -> i32 {
    if study.is_null() || out_trial.is_null() {
        set_last_error(-1, "study or out_trial pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let trial_number = match get_best_trial(&s.inner) {
            Ok(n) => n,
            Err(e) => {
                return set_rustuna_error(&e);
            }
        };

        let mut guard = match s.inner.storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                return 3;
            }
        };

        let trial_id = match guard.get_trial_id_from_study_id_trial_number(s.inner.id, trial_number) {
            Ok(id) => id,
            Err(e) => {
                return set_rustuna_error(&e);
            }
        };

        let trial = match guard.get_trial(trial_id) {
            Ok(t) => t.clone(),
            Err(e) => {
                return set_rustuna_error(&e);
            }
        };

        unsafe {
            *out_trial = Box::into_raw(Box::new(RustunaPersistedTrial { inner: trial }));
        }
        0
    }));

    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_get_best_trial: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_get_best_trials(
    study: *mut RustunaStudy,
    out_trials: *mut *mut *mut RustunaPersistedTrial,
    out_len: *mut usize,
) -> i32 {
    if study.is_null() || out_trials.is_null() || out_len.is_null() {
        set_last_error(-1, "study, out_trials, or out_len pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let pareto_front_numbers = match get_pareto_front(&s.inner) {
            Ok(p) => p,
            Err(e) => return set_rustuna_error(&e),
        };

        let mut guard = match s.inner.storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                return 3;
            }
        };
        let trials_vec = match guard.get_trials(s.inner.id) {
            Ok(v) => v,
            Err(e) => return set_rustuna_error(&e),
        };

        let mut pt_ptrs: Vec<*mut RustunaPersistedTrial> = Vec::with_capacity(pareto_front_numbers.len());
        for num in pareto_front_numbers {
            if let Some(Some(trial)) = trials_vec.get(num as usize) {
                pt_ptrs.push(Box::into_raw(Box::new(RustunaPersistedTrial { inner: trial.clone() })));
            }
        }

        pt_ptrs.shrink_to_fit();
        let len = pt_ptrs.len();
        let ptr = pt_ptrs.as_mut_ptr();
        std::mem::forget(pt_ptrs);

        unsafe {
            *out_trials = ptr;
            *out_len = len;
        }
        0
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_get_best_trials: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_get_trials(
    study: *mut RustunaStudy,
    out_trials: *mut *mut *mut RustunaPersistedTrial,
    out_len: *mut usize,
) -> i32 {
    if study.is_null() || out_trials.is_null() || out_len.is_null() {
        set_last_error(-1, "study, out_trials, or out_len pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *study };
        let mut guard = match s.inner.storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                return 3;
            }
        };

        let trials_vec = match guard.get_trials(s.inner.id) {
            Ok(v) => v.clone(),
            Err(e) => {
                return set_rustuna_error(&e);
            }
        };

        let mut pt_ptrs: Vec<*mut RustunaPersistedTrial> = trials_vec
            .into_iter()
            .flatten()
            .map(|t| Box::into_raw(Box::new(RustunaPersistedTrial { inner: t })))
            .collect();

        pt_ptrs.shrink_to_fit();
        let len = pt_ptrs.len();
        let ptr = pt_ptrs.as_mut_ptr();
        std::mem::forget(pt_ptrs);

        unsafe {
            *out_trials = ptr;
            *out_len = len;
        }
        0
    }));

    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_get_trials: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_trials_buffer_free(trials: *mut *mut RustunaPersistedTrial, len: usize) {
    if !trials.is_null() && len > 0 {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(trials, len);
            for ptr in slice.iter() {
                if !ptr.is_null() {
                    drop(Box::from_raw(*ptr));
                }
            }
            drop(Vec::from_raw_parts(trials, len, len));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_number(trial: *const RustunaPersistedTrial) -> u32 {
    if trial.is_null() {
        return 0;
    }
    unsafe { (*trial).inner.number }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_state(trial: *const RustunaPersistedTrial) -> i32 {
    if trial.is_null() {
        return -1;
    }
    match unsafe { &(*trial).inner.state_values } {
        TrialStateValues::Running => 0,
        TrialStateValues::Complete(_) => 1,
        TrialStateValues::Pruned => 2,
        TrialStateValues::Waiting => 3,
        TrialStateValues::Fail => 4,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_value(
    trial: *const RustunaPersistedTrial,
    out_val: *mut f64,
) -> bool {
    if trial.is_null() || out_val.is_null() {
        return false;
    }
    match unsafe { &(*trial).inner.state_values } {
        TrialStateValues::Complete(vals) => {
            if let Some(&first) = vals.first() {
                unsafe { *out_val = first };
                true
            } else {
                false
            }
        }
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_values(
    trial: *const RustunaPersistedTrial,
    out_vals: *mut *mut f64,
    out_len: *mut usize,
) -> bool {
    if trial.is_null() || out_vals.is_null() || out_len.is_null() {
        return false;
    }
    match unsafe { &(*trial).inner.state_values } {
        TrialStateValues::Complete(vals) => {
            let mut cloned = vals.clone();
            cloned.shrink_to_fit();
            let len = cloned.len();
            let ptr = cloned.as_mut_ptr();
            std::mem::forget(cloned);
            unsafe {
                *out_vals = ptr;
                *out_len = len;
            }
            true
        }
        _ => {
            unsafe {
                *out_vals = std::ptr::null_mut();
                *out_len = 0;
            }
            false
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_values_buffer_free(vals: *mut f64, len: usize) {
    if !vals.is_null() && len > 0 {
        unsafe {
            drop(Vec::from_raw_parts(vals, len, len));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_params_json(
    trial: *const RustunaPersistedTrial,
    out_json: *mut *mut c_char,
) -> i32 {
    if trial.is_null() || out_json.is_null() {
        set_last_error(-1, "trial or out_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &*trial };
        let mut map = HashMap::new();
        for (k, v) in &t.inner.internal_params {
            map.insert(k.clone(), *v);
        }
        match serde_json::to_string(&map) {
            Ok(json_str) => {
                let c_str = CString::new(json_str).unwrap_or_default();
                unsafe {
                    *out_json = c_str.into_raw();
                }
                0
            }
            Err(e) => {
                set_last_error(19, format!("Serialization error: {e:?}"));
                19
            }
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_persisted_trial_get_params_json: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_user_attrs_json(
    trial: *const RustunaPersistedTrial,
    out_json: *mut *mut c_char,
) -> i32 {
    if trial.is_null() || out_json.is_null() {
        set_last_error(-1, "trial or out_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &*trial };
        let mut map = HashMap::new();
        for (k, v) in &t.inner.attrs {
            if let rustuna_core::attr::AttrKey::User(s) = k {
                map.insert(s.as_str().to_string(), v.clone());
            }
        }
        match serde_json::to_string(&map) {
            Ok(json_str) => {
                let c_str = CString::new(json_str).unwrap_or_default();
                unsafe {
                    *out_json = c_str.into_raw();
                }
                0
            }
            Err(e) => {
                set_last_error(19, format!("Serialization error: {e:?}"));
                19
            }
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_persisted_trial_get_user_attrs_json: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_constraints_json(
    trial: *const RustunaPersistedTrial,
    out_json: *mut *mut c_char,
) -> i32 {
    if trial.is_null() || out_json.is_null() {
        set_last_error(-1, "trial or out_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &*trial };
        match t.inner.constraints() {
            Ok(map) => match serde_json::to_string(&map) {
                Ok(json_str) => {
                    let c_str = CString::new(json_str).unwrap_or_default();
                    unsafe {
                        *out_json = c_str.into_raw();
                    }
                    0
                }
                Err(e) => {
                    set_last_error(19, format!("Serialization error: {e:?}"));
                    19
                }
            },
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_persisted_trial_get_constraints_json: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_free(trial: *mut RustunaPersistedTrial) {
    if !trial.is_null() {
        unsafe {
            drop(Box::from_raw(trial));
        }
    }
}
