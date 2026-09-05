#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, RwLock};

use std::time::Instant;

use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::{RandomSampler, Sampler};
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::{
    create_study_with_arc, get_best_trial, get_pareto_front, Direction, Study,
};
use rustuna_core::trial::{PersistedTrial, Trial, TrialStateValues};
use rustuna_sampler::nsgaii::NSGAIISampler;
use rustuna_sampler::qmc::QmcSampler;
use rustuna_sampler::tpe::{TpeConfig, TpeSampler};
use rustuna_storage::cache::CachedStorage;
use rustuna_storage::journal::file::JournalFileBackend;
use rustuna_storage::journal::storage::JournalStorage;
use rustuna_storage::sqlite3::SQLite3Storage;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

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

fn set_last_error(code: i32, msg: String) -> i32 {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok().map(|cs| (code, cs));
    });
    code
}

fn set_rustuna_error(err: &rustuna_core::Error) -> i32 {
    let code = if err.reason.contains("exhausted") {
        21
    } else {
        error_kind_to_code(&err.kind)
    };
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
pub extern "C" fn rustuna_take_last_error(out_code: *mut i32, out_msg: *mut *mut c_char) -> i32 {
    LAST_ERROR.with(|cell| {
        if let Some((code, cs)) = cell.borrow_mut().take() {
            unsafe {
                if !out_code.is_null() {
                    *out_code = code;
                }
                if !out_msg.is_null() {
                    *out_msg = cs.into_raw();
                }
            }
            1
        } else {
            0
        }
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

/// Full TPE configuration. `multivariate`: negative selects automatically
/// (multivariate for single-objective, independent for multi-objective,
/// matching Optuna), `0` forces independent sampling, positive forces joint
/// sampling. `n_startup_trials` completed trials run randomly before TPE
/// engages; `0` keeps the engine default (10).
#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_tpe_full(
    seed: u64,
    has_seed: bool,
    multivariate: i8,
    n_startup_trials: usize,
) -> *mut RustunaSampler {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let sampler = TpeSampler::from_config(TpeConfig {
            multivariate: if multivariate < 0 {
                None
            } else {
                Some(multivariate > 0)
            },
            n_startup_trials: if n_startup_trials > 0 {
                n_startup_trials
            } else {
                TpeConfig::default().n_startup_trials
            },
            seed: if has_seed { Some(seed) } else { None },
        });
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
        let pop_size = if population_size > 0 {
            population_size
        } else {
            50
        };
        let mut_prob = if mutation_prob > 0.0 {
            Some(mutation_prob)
        } else {
            None
        };
        let cross_prob = if crossover_prob > 0.0 {
            crossover_prob
        } else {
            0.9
        };
        let swap_prob = if swapping_prob > 0.0 {
            swapping_prob
        } else {
            0.5
        };

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

pub struct GridSampler {
    param_names: Vec<String>,
    combinations: Vec<HashMap<String, f64>>,
}

impl GridSampler {
    pub fn new(search_space: HashMap<String, Vec<f64>>, seed: Option<u64>) -> Self {
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();

        let mut combinations = vec![HashMap::new()];
        for name in &param_names {
            if let Some(values) = search_space.get(name) {
                if values.is_empty() {
                    continue;
                }
                let mut next_combinations = Vec::with_capacity(combinations.len() * values.len());
                for combo in &combinations {
                    for val in values {
                        let mut new_combo = combo.clone();
                        new_combo.insert(name.clone(), *val);
                        next_combinations.push(new_combo);
                    }
                }
                combinations = next_combinations;
            }
        }

        if let Some(s) = seed {
            let mut rng = StdRng::seed_from_u64(s);
            combinations.shuffle(&mut rng);
        }

        GridSampler {
            param_names,
            combinations,
        }
    }
}

impl Sampler for GridSampler {
    fn sample_independent(
        &self,
        _ctx: &rustuna_core::sampler::Context,
        _storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> rustuna_core::Result<f64> {
        if let Some(first) = self.combinations.first() {
            if let Some(&val) = first.get(name) {
                return Ok(val);
            }
        }
        match distribution {
            Distribution::Float { low, .. } => Ok(*low),
            Distribution::Int { low, .. } => Ok(*low as f64),
            Distribution::Categorical { .. } => Ok(0.0),
        }
    }

    fn support_joint_sampling(&self) -> bool {
        true
    }

    fn sample_joint(
        &self,
        ctx: &rustuna_core::sampler::Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> rustuna_core::Result<HashMap<String, f64>> {
        let mut storage_guard = storage.write().map_err(|e| {
            rustuna_core::Error::with_reason(
                rustuna_core::ErrorKind::StorageError,
                format!("Storage lock poisoned: {e:?}"),
            )
        })?;

        let trials = storage_guard.get_trials(ctx.study_id)?;

        let mut evaluated_hashes = std::collections::HashSet::new();
        for trial in trials.iter().flatten() {
            if trial.state_values.state() == rustuna_core::trial::TrialState::Fail {
                continue;
            }
            let mut key = Vec::with_capacity(self.param_names.len());
            let mut all_present = true;
            for name in &self.param_names {
                if let Some(&val) = trial.internal_params.get(name) {
                    key.push(val.to_bits());
                } else {
                    all_present = false;
                    break;
                }
            }
            if all_present {
                evaluated_hashes.insert(key);
            }
        }

        for candidate in &self.combinations {
            let key: Vec<u64> = self
                .param_names
                .iter()
                .filter_map(|name| candidate.get(name).map(|v| v.to_bits()))
                .collect();

            if !evaluated_hashes.contains(&key) {
                let mut res = HashMap::new();
                for (k, v) in candidate {
                    if search_space.contains_key(k) {
                        res.insert(k.clone(), *v);
                    }
                }
                return Ok(res);
            }
        }

        Err(rustuna_core::Error::with_reason(
            rustuna_core::ErrorKind::SamplerError,
            "All grid search space combinations have been exhausted".to_string(),
        ))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_grid_new(
    search_space_json: *const c_char,
    seed: u64,
    has_seed: bool,
    out_sampler: *mut *mut RustunaSampler,
) -> i32 {
    if search_space_json.is_null() || out_sampler.is_null() {
        set_last_error(
            -1,
            "search_space_json or out_sampler pointer is null".to_string(),
        );
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let json_str = match unsafe { CStr::from_ptr(search_space_json) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(19, format!("Invalid UTF-8: {e:?}"));
                return 19;
            }
        };

        let search_space: HashMap<String, Vec<f64>> = match serde_json::from_str(json_str) {
            Ok(ss) => ss,
            Err(e) => {
                set_last_error(19, format!("JSON deserialize error: {e:?}"));
                return 19;
            }
        };

        let s_opt = if has_seed { Some(seed) } else { None };
        let sampler = GridSampler::new(search_space, s_opt);
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
            set_last_error(-99, format!("Panic in rustuna_sampler_grid_new: {e:?}"));
            -99
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
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        };

        let dirs: Vec<Direction> = if !directions.is_null() && directions_len > 0 {
            let slice = unsafe { std::slice::from_raw_parts(directions, directions_len) };
            slice
                .iter()
                .map(|&d| match d {
                    1 => Direction::Maximize,
                    _ => Direction::Minimize,
                })
                .collect()
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
            unsafe { CStr::from_ptr(storage_path) }
                .to_string_lossy()
                .into_owned()
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
            Err(e)
                if load_if_exists && matches!(e.kind, rustuna_core::ErrorKind::DuplicatedStudy) =>
            {
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
        let study_name = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();

        let storage_path_str = if !storage_path.is_null() {
            unsafe { CStr::from_ptr(storage_path) }
                .to_string_lossy()
                .into_owned()
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
        let s = unsafe { &*study };
        let k = unsafe { CStr::from_ptr(key) }.to_string_lossy();
        let v = unsafe { CStr::from_ptr(val) }.to_string_lossy();
        match s
            .inner
            .set_user_attr(HashMap::from([(k.to_string(), v.to_string())]))
        {
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
        let s = unsafe { &*study };
        let k = unsafe { CStr::from_ptr(key) }
            .to_string_lossy()
            .into_owned();
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

/// Parses the optional user-attrs JSON object into storage attrs.
///
/// Shared by the JSON and typed enqueue paths. Returns `Err(code)` with the
/// last error already set on malformed input.
fn parse_user_attrs_json(
    user_attrs_json: *const c_char,
) -> Result<Option<rustuna_core::attr::Attrs>, i32> {
    if user_attrs_json.is_null() {
        return Ok(None);
    }
    let uaj_str = match unsafe { CStr::from_ptr(user_attrs_json) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            return Err(set_last_error(
                -1,
                "user_attrs_json is not valid UTF-8".to_string(),
            ));
        }
    };
    if uaj_str.trim().is_empty() || uaj_str == "{}" {
        return Ok(None);
    }
    match serde_json::from_str::<HashMap<String, String>>(uaj_str) {
        Ok(map) => {
            let mut attrs = rustuna_core::attr::Attrs::new();
            for (k, v) in map {
                attrs.insert(rustuna_core::attr::AttrKey::User(k.into()), v);
            }
            Ok(Some(attrs))
        }
        Err(e) => Err(set_last_error(
            19,
            format!("Failed to parse user_attrs_json: {e:?}"),
        )),
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
        let s = unsafe { &*study };
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
                serde_json::Value::String(s) => {
                    rustuna_core::attr::CategoryLabel::String(s.clone())
                }
                serde_json::Value::Bool(b) => rustuna_core::attr::CategoryLabel::Bool(*b),
                serde_json::Value::Null => rustuna_core::attr::CategoryLabel::None,
                _ => rustuna_core::attr::CategoryLabel::String(v.to_string()),
            };
            fixed_params.insert(k.clone(), label);
        }

        let user_attrs = match parse_user_attrs_json(user_attrs_json) {
            Ok(attrs) => attrs,
            Err(code) => return code,
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

/// Parameter value kinds for [`rustuna_study_enqueue_typed`].
///
/// Values travel in `num_values` (`f64`) except strings, which travel in
/// `str_values`. Integers must fit the `f64` mantissa exactly (true for all
/// hyperparameter ranges); booleans encode as `0.0`/`1.0`.
pub mod enqueue_kind {
    /// Integer value; `num_values[i] as i64`.
    pub const INT: u8 = 0;
    /// Floating-point value; `num_values[i]`.
    pub const DOUBLE: u8 = 1;
    /// String value; `str_values[i]` (must be non-null).
    pub const STRING: u8 = 2;
    /// Boolean value; `num_values[i] != 0.0`.
    pub const BOOL: u8 = 3;
}

/// Enqueues a trial with typed parameter arrays instead of JSON.
///
/// Same queue semantics as [`rustuna_study_enqueue_trial`] without the
/// serialize/parse round trip on either side of the boundary: no JSON is
/// produced or consumed for `params`. `user_attrs_json` keeps the JSON form
/// (rarely set, never hot).
///
/// When `count` is zero the arrays are not touched (may be null) and an
/// all-Rust-sampled trial is enqueued.
#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_enqueue_typed(
    study: *mut RustunaStudy,
    names: *const *const c_char,
    kinds: *const u8,
    num_values: *const f64,
    str_values: *const *const c_char,
    count: usize,
    user_attrs_json: *const c_char,
) -> i32 {
    if study.is_null() {
        set_last_error(-1, "study pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &*study };
        let mut fixed_params = HashMap::new();
        if count > 0 {
            if names.is_null() || kinds.is_null() || num_values.is_null() || str_values.is_null() {
                set_last_error(-1, "null params array with nonzero count".to_string());
                return -1;
            }
            let names = unsafe { std::slice::from_raw_parts(names, count) };
            let kinds = unsafe { std::slice::from_raw_parts(kinds, count) };
            let num_values = unsafe { std::slice::from_raw_parts(num_values, count) };
            let str_values = unsafe { std::slice::from_raw_parts(str_values, count) };
            for i in 0..count {
                if names[i].is_null() {
                    set_last_error(-1, format!("param name {i} is null"));
                    return -1;
                }
                let name = match unsafe { CStr::from_ptr(names[i]) }.to_str() {
                    Ok(s) => s,
                    Err(_) => {
                        set_last_error(-1, format!("param name {i} is not valid UTF-8"));
                        return -1;
                    }
                };
                let label = match kinds[i] {
                    enqueue_kind::INT => {
                        rustuna_core::attr::CategoryLabel::Int(num_values[i] as i64)
                    }
                    enqueue_kind::DOUBLE => rustuna_core::attr::CategoryLabel::Float(num_values[i]),
                    enqueue_kind::STRING => {
                        if str_values[i].is_null() {
                            set_last_error(-1, format!("string value {i} is null"));
                            return -1;
                        }
                        match unsafe { CStr::from_ptr(str_values[i]) }.to_str() {
                            Ok(v) => rustuna_core::attr::CategoryLabel::String(v.to_string()),
                            Err(_) => {
                                set_last_error(-1, format!("string value {i} is not valid UTF-8"));
                                return -1;
                            }
                        }
                    }
                    enqueue_kind::BOOL => {
                        rustuna_core::attr::CategoryLabel::Bool(num_values[i] != 0.0)
                    }
                    k => {
                        set_last_error(-1, format!("unknown param kind {k} at index {i}"));
                        return -1;
                    }
                };
                fixed_params.insert(name.to_string(), label);
            }
        }

        let user_attrs = match parse_user_attrs_json(user_attrs_json) {
            Ok(attrs) => attrs,
            Err(code) => return code,
        };

        match s.inner.enqueue_trial(fixed_params, user_attrs) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_enqueue_typed: {e:?}"));
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
        let s = unsafe { &*study };
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
            Ok(importances) => match serde_json::to_string(&importances) {
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
            },
            Err(e) => set_rustuna_error(&e),
        }
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(
                -99,
                format!("Panic in rustuna_study_get_param_importances: {e:?}"),
            );
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
        let s = unsafe { &*study };
        match s.inner.ask() {
            Ok(trial) => {
                unsafe {
                    *out_trial = Box::into_raw(Box::new(RustunaTrial { inner: trial }));
                }
                0
            }
            Err(e) => {
                if let Ok(mut guard) = s.inner.storage.write() {
                    let last_running_trial_id = if let Ok(trials) = guard.get_trials(s.inner.id) {
                        trials.last().and_then(|opt| {
                            opt.as_ref().and_then(|t| {
                                if t.state_values.state()
                                    == rustuna_core::trial::TrialState::Running
                                    && t.internal_params.is_empty()
                                {
                                    Some(t.id)
                                } else {
                                    None
                                }
                            })
                        })
                    } else {
                        None
                    };

                    if let Some(id) = last_running_trial_id {
                        let _ = guard.set_trial_state_values(
                            id,
                            rustuna_core::trial::TrialStateValues::Fail,
                        );
                    }
                }
                set_rustuna_error(&e)
            }
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
pub extern "C" fn rustuna_study_tell_multi(
    study: *mut RustunaStudy,
    trial_number: u32,
    state: i32,
    values: *const f64,
    values_len: usize,
    intermediate_json: *const c_char,
) -> i32 {
    if study.is_null() {
        set_last_error(-1, "study pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &*study };

        // If intermediate values were provided, persist them while trial is still running
        if !intermediate_json.is_null() {
            if let Ok(json_str) = unsafe { CStr::from_ptr(intermediate_json) }.to_str() {
                if !json_str.is_empty() {
                    if let Ok(map) = serde_json::from_str::<HashMap<u32, f64>>(json_str) {
                        if !map.is_empty() {
                            let mut guard = match s.inner.storage.write() {
                                Ok(g) => g,
                                Err(e) => {
                                    set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                                    return 3;
                                }
                            };
                            let trial_id = match guard
                                .get_trial_id_from_study_id_trial_number(s.inner.id, trial_number)
                            {
                                Ok(id) => id,
                                Err(e) => return set_rustuna_error(&e),
                            };
                            if let Err(e) = guard.set_trial_intermediate_values(trial_id, map) {
                                return set_rustuna_error(&e);
                            }
                        }
                    }
                }
            }
        }

        let state_val = match state {
            1 => {
                if values.is_null() || values_len == 0 {
                    return set_last_error(
                        -1,
                        "Values pointer is null or length is 0 for COMPLETE state".to_string(),
                    );
                }
                if values_len != s.inner.directions.len() {
                    return set_last_error(
                        -1,
                        format!(
                            "Values count ({values_len}) does not match study directions count ({})",
                            s.inner.directions.len()
                        ),
                    );
                }
                let slice = unsafe { std::slice::from_raw_parts(values, values_len) };
                if slice.iter().any(|v| v.is_nan()) {
                    return set_last_error(-1, "Objective values cannot contain NaN".to_string());
                }
                TrialStateValues::Complete(slice.to_vec())
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
            set_last_error(
                -99,
                format!("Panic in rustuna_study_tell_multi_with_intermediate: {e:?}"),
            );
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
        if low > high {
            return set_last_error(
                -4,
                format!("Invalid float range: low ({low}) > high ({high})"),
            );
        }
        if log && low <= 0.0 {
            return set_last_error(
                -4,
                format!("Invalid float range: log scale requires low ({low}) > 0"),
            );
        }
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
        if low > high {
            return set_last_error(
                -4,
                format!("Invalid int range: low ({low}) > high ({high})"),
            );
        }
        if log && low <= 0 {
            return set_last_error(
                -4,
                format!("Invalid int range: log scale requires low ({low}) > 0"),
            );
        }
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
    if trial.is_null()
        || name.is_null()
        || choices.is_null()
        || out_index.is_null()
        || choices_count == 0
    {
        set_last_error(
            -1,
            "trial, name, choices, or out_index pointer is null or choices_count is 0".to_string(),
        );
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
            let choice_str = unsafe { CStr::from_ptr(choice_ptr) }
                .to_string_lossy()
                .into_owned();
            category_labels.push(rustuna_core::attr::CategoryLabel::String(choice_str));
        }

        let dist_choice = match t
            .inner
            .suggest_categorical_enum(&param_name, &category_labels)
        {
            Ok(c) => c,
            Err(e) => return set_rustuna_error(&e),
        };
        let chosen_idx = category_labels
            .iter()
            .position(|l| l == dist_choice)
            .unwrap_or(0);
        unsafe { *out_index = chosen_idx };
        0
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(
                -99,
                format!("Panic in rustuna_trial_suggest_categorical: {e:?}"),
            );
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
        let v = unsafe { CStr::from_ptr(val) }
            .to_string_lossy()
            .into_owned();
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
        let k = unsafe { CStr::from_ptr(key) }
            .to_string_lossy()
            .into_owned();
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
    pub category_labels: HashMap<String, Vec<rustuna_core::attr::CategoryLabel>>,
}

fn make_persisted_trial(
    storage: &mut dyn Storage,
    study_id: u32,
    trial: PersistedTrial,
    cat_cache: &mut HashMap<String, Vec<rustuna_core::attr::CategoryLabel>>,
) -> RustunaPersistedTrial {
    let mut category_labels = HashMap::new();
    for (name, dist) in &trial.distributions {
        if let Distribution::Categorical { cardinality } = dist {
            if let Some(labels) = cat_cache.get(name) {
                category_labels.insert(name.clone(), labels.clone());
            } else if let Ok(Some(labels)) =
                storage.get_category_labels(study_id, name, *cardinality)
            {
                cat_cache.insert(name.clone(), labels.clone());
                category_labels.insert(name.clone(), labels);
            }
        }
    }
    RustunaPersistedTrial {
        inner: trial,
        category_labels,
    }
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
        let s = unsafe { &*study };
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

        let trial_id = match guard.get_trial_id_from_study_id_trial_number(s.inner.id, trial_number)
        {
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

        let mut cat_cache = HashMap::new();
        unsafe {
            *out_trial = Box::into_raw(Box::new(make_persisted_trial(
                &mut *guard,
                s.inner.id,
                trial,
                &mut cat_cache,
            )));
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
        set_last_error(
            -1,
            "study, out_trials, or out_len pointer is null".to_string(),
        );
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &*study };
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
            Ok(v) => v.clone(),
            Err(e) => return set_rustuna_error(&e),
        };

        let mut cat_cache = HashMap::new();
        let mut pt_ptrs: Vec<*mut RustunaPersistedTrial> =
            Vec::with_capacity(pareto_front_numbers.len());
        for num in pareto_front_numbers {
            if let Some(Some(trial)) = trials_vec.get(num as usize) {
                pt_ptrs.push(Box::into_raw(Box::new(make_persisted_trial(
                    &mut *guard,
                    s.inner.id,
                    trial.clone(),
                    &mut cat_cache,
                ))));
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
            set_last_error(
                -99,
                format!("Panic in rustuna_study_get_best_trials: {e:?}"),
            );
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
    rustuna_study_get_trials_filtered(study, u32::MAX, out_trials, out_len)
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

#[derive(serde::Serialize)]
struct PersistedTrialPayload<'a> {
    number: u32,
    state: i32,
    values: &'a [f64],
    params: &'a HashMap<String, f64>,
    param_values: HashMap<&'a str, serde_json::Value>,
    user_attrs: HashMap<&'a str, &'a str>,
    constraints: HashMap<String, f64>,
    intermediate_values: &'a HashMap<u32, f64>,
    datetime_start: &'a Option<String>,
    datetime_complete: &'a Option<String>,
}

fn extract_persisted_trial_payload<'a>(t: &'a RustunaPersistedTrial) -> PersistedTrialPayload<'a> {
    let state = match &t.inner.state_values {
        TrialStateValues::Running => 0,
        TrialStateValues::Complete(_) => 1,
        TrialStateValues::Pruned => 2,
        TrialStateValues::Waiting => 3,
        TrialStateValues::Fail => 4,
    };
    let values: &[f64] = match &t.inner.state_values {
        TrialStateValues::Complete(vals) => vals.as_slice(),
        _ => &[],
    };

    let mut user_attrs = HashMap::new();
    for (k, v) in &t.inner.attrs {
        if let rustuna_core::attr::AttrKey::User(s) = k {
            user_attrs.insert(s.as_str(), v.as_str());
        }
    }

    let constraints = t.inner.constraints().unwrap_or_default();

    let mut param_values = HashMap::new();
    for (name, val) in &t.inner.internal_params {
        if let Some(dist) = t.inner.distributions.get(name) {
            match dist {
                Distribution::Categorical { .. } => {
                    let idx = *val as usize;
                    if let Some(labels) = t.category_labels.get(name) {
                        if let Some(label) = labels.get(idx) {
                            match label {
                                rustuna_core::attr::CategoryLabel::String(s) => {
                                    param_values.insert(
                                        name.as_str(),
                                        serde_json::Value::String(s.clone()),
                                    );
                                    continue;
                                }
                                rustuna_core::attr::CategoryLabel::Int(i) => {
                                    param_values.insert(
                                        name.as_str(),
                                        serde_json::Value::Number((*i).into()),
                                    );
                                    continue;
                                }
                                rustuna_core::attr::CategoryLabel::Float(f) => {
                                    if let Some(num) = serde_json::Number::from_f64(*f) {
                                        param_values
                                            .insert(name.as_str(), serde_json::Value::Number(num));
                                        continue;
                                    }
                                }
                                rustuna_core::attr::CategoryLabel::Bool(b) => {
                                    param_values.insert(name.as_str(), serde_json::Value::Bool(*b));
                                    continue;
                                }
                                rustuna_core::attr::CategoryLabel::None => {
                                    param_values.insert(name.as_str(), serde_json::Value::Null);
                                    continue;
                                }
                            }
                        }
                    }
                    param_values.insert(
                        name.as_str(),
                        serde_json::Number::from_f64(*val)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null),
                    );
                }
                Distribution::Int { .. } => {
                    param_values.insert(
                        name.as_str(),
                        serde_json::Value::Number((*val as i64).into()),
                    );
                }
                Distribution::Float { .. } => {
                    param_values.insert(
                        name.as_str(),
                        serde_json::Number::from_f64(*val)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null),
                    );
                }
            }
        } else {
            param_values.insert(
                name.as_str(),
                serde_json::Number::from_f64(*val)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
            );
        }
    }

    PersistedTrialPayload {
        number: t.inner.number,
        state,
        values,
        params: &t.inner.internal_params,
        param_values,
        user_attrs,
        constraints,
        intermediate_values: &t.inner.intermediate_values,
        datetime_start: &t.inner.datetime_start,
        datetime_complete: &t.inner.datetime_complete,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_get_json(
    trial: *const RustunaPersistedTrial,
    out_json: *mut *mut c_char,
) -> i32 {
    if trial.is_null() || out_json.is_null() {
        set_last_error(-1, "trial or out_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let t = unsafe { &*trial };
        let payload = extract_persisted_trial_payload(t);
        match serde_json::to_string(&payload) {
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
            set_last_error(
                -99,
                format!("Panic in rustuna_persisted_trial_get_json: {e:?}"),
            );
            -99
        }
    }
}

/// Shared implementation for the trials-JSON fetchers.
///
/// `from_number` skips trials below a trial number *before* the expensive
/// per-trial conversion (`make_persisted_trial` performs storage reads), so
/// incremental refresh costs O(delta) instead of O(history).
fn fetch_trials_json(
    study: *mut RustunaStudy,
    states_mask: u32,
    from_number: u32,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
    fn_name: &'static str,
) -> i32 {
    if study.is_null() || out_json.is_null() {
        set_last_error(-1, "study or out_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &*study };
        let mut guard = match s.inner.storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                return 3;
            }
        };

        let trials_vec = match guard.get_trials(s.inner.id) {
            Ok(v) => v.clone(),
            Err(e) => return set_rustuna_error(&e),
        };

        let mut cat_cache = HashMap::new();
        let trials: Vec<RustunaPersistedTrial> = trials_vec
            .into_iter()
            .flatten()
            .filter(|t| t.number >= from_number)
            .filter(|t| {
                let state_bit = match t.state_values {
                    TrialStateValues::Running => 1 << 0,
                    TrialStateValues::Complete(_) => 1 << 1,
                    TrialStateValues::Pruned => 1 << 2,
                    TrialStateValues::Waiting => 1 << 3,
                    TrialStateValues::Fail => 1 << 4,
                };
                (states_mask & state_bit) != 0
            })
            .map(|t| make_persisted_trial(&mut *guard, s.inner.id, t, &mut cat_cache))
            .collect();

        let payloads: Vec<PersistedTrialPayload> =
            trials.iter().map(extract_persisted_trial_payload).collect();
        match serde_json::to_string(&payloads) {
            Ok(json_str) => {
                let len = json_str.len();
                let c_str = CString::new(json_str).unwrap_or_default();
                unsafe {
                    *out_json = c_str.into_raw();
                    if !out_len.is_null() {
                        *out_len = len;
                    }
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
            set_last_error(-99, format!("Panic in {fn_name}: {e:?}"));
            -99
        }
    }
}

/// Fetches trials at or above a trial number, filtered by state.
///
/// Drivers tracking `history.count` pass it as `from_number` and pay O(new)
/// instead of O(history). Trials are numbered densely from 0, so
/// `from_number == known_count` yields exactly the unseen tail.
#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_get_trials_json_since(
    study: *mut RustunaStudy,
    states_mask: u32,
    from_number: u32,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> i32 {
    fetch_trials_json(
        study,
        states_mask,
        from_number,
        out_json,
        out_len,
        "rustuna_study_get_trials_json_since",
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_persisted_trial_free(trial: *mut RustunaPersistedTrial) {
    if !trial.is_null() {
        unsafe {
            drop(Box::from_raw(trial));
        }
    }
}

// MARK: - Study Lifecycle & Storage Operations

#[derive(serde::Serialize)]
struct SerializableStudySummary {
    id: u32,
    name: String,
    directions: Vec<i32>,
    user_attrs: HashMap<String, String>,
    system_attrs: HashMap<String, String>,
    trial_count: u32,
}

fn copy_study_core(
    from_storage: Arc<RwLock<dyn Storage>>,
    from_study_id: u32,
    to_storage: Arc<RwLock<dyn Storage>>,
    to_study_name: &str,
) -> Result<u32, i32> {
    let (from_directions, from_attrs, trials) = {
        let mut guard = match from_storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Source storage lock poisoned: {e:?}"));
                return Err(3);
            }
        };
        let study_info = match guard.get_study(from_study_id) {
            Ok(st) => st.clone(),
            Err(e) => return Err(set_rustuna_error(&e)),
        };
        let trials = match guard.get_trials(from_study_id) {
            Ok(t) => t.clone(),
            Err(e) => return Err(set_rustuna_error(&e)),
        };
        (study_info.directions, study_info.attrs, trials)
    };

    let mut dest_guard = match to_storage.write() {
        Ok(g) => g,
        Err(e) => {
            set_last_error(3, format!("Destination storage lock poisoned: {e:?}"));
            return Err(3);
        }
    };

    let existing_studies = match dest_guard.get_studies() {
        Ok(st) => st.clone(),
        Err(e) => return Err(set_rustuna_error(&e)),
    };
    if existing_studies.iter().any(|st| st.name == to_study_name) {
        set_last_error(
            4,
            format!("Study '{to_study_name}' already exists in destination storage"),
        );
        return Err(4);
    }

    let new_study = match dest_guard.create_new_study(to_study_name, from_directions) {
        Ok(ns) => ns.clone(),
        Err(e) => return Err(set_rustuna_error(&e)),
    };
    let new_id = new_study.id;

    if !from_attrs.is_empty() {
        if let Err(e) = dest_guard.set_study_attrs(new_id, from_attrs, false) {
            return Err(set_rustuna_error(&e));
        }
    }

    for trial in trials.into_iter().flatten() {
        if let Err(e) = dest_guard.create_new_trial_from_template(new_id, &trial) {
            return Err(set_rustuna_error(&e));
        }
    }

    Ok(new_id)
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_copy(
    study: *mut RustunaStudy,
    to_storage_type: i32,
    to_storage_path: *const c_char,
    to_study_name: *const c_char,
    out_study: *mut *mut RustunaStudy,
) -> i32 {
    if study.is_null() || out_study.is_null() {
        set_last_error(-1, "study or out_study pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &*study };
        let dest_storage_path = if to_storage_path.is_null() {
            "".to_string()
        } else {
            unsafe { CStr::from_ptr(to_storage_path) }
                .to_string_lossy()
                .into_owned()
        };
        let dest_study_name = if to_study_name.is_null() {
            s.inner.name.clone()
        } else {
            unsafe { CStr::from_ptr(to_study_name) }
                .to_string_lossy()
                .into_owned()
        };

        let dest_storage = match create_storage_backend(to_storage_type, &dest_storage_path) {
            Ok(st) => st,
            Err(e) => return set_rustuna_error(&e),
        };

        if let Err(code) = copy_study_core(
            s.inner.storage.clone(),
            s.inner.id,
            dest_storage.clone(),
            &dest_study_name,
        ) {
            return code;
        }

        let new_study_instance =
            match Study::from_name(dest_study_name, dest_storage, s.inner.sampler.clone()) {
                Ok(st) => st,
                Err(e) => return set_rustuna_error(&e),
            };

        unsafe {
            *out_study = Box::into_raw(Box::new(RustunaStudy {
                inner: new_study_instance,
            }));
        }
        0
    }));

    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_copy: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_storage_copy_study(
    from_storage_type: i32,
    from_storage_path: *const c_char,
    from_study_name: *const c_char,
    to_storage_type: i32,
    to_storage_path: *const c_char,
    to_study_name: *const c_char,
) -> i32 {
    if from_study_name.is_null() {
        set_last_error(-1, "from_study_name pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let src_path = if from_storage_path.is_null() {
            "".to_string()
        } else {
            unsafe { CStr::from_ptr(from_storage_path) }
                .to_string_lossy()
                .into_owned()
        };
        let src_name = unsafe { CStr::from_ptr(from_study_name) }.to_string_lossy();

        let dest_path = if to_storage_path.is_null() {
            "".to_string()
        } else {
            unsafe { CStr::from_ptr(to_storage_path) }
                .to_string_lossy()
                .into_owned()
        };
        let dest_name = if to_study_name.is_null() {
            src_name.to_string()
        } else {
            unsafe { CStr::from_ptr(to_study_name) }
                .to_string_lossy()
                .into_owned()
        };

        let src_storage = match create_storage_backend(from_storage_type, &src_path) {
            Ok(st) => st,
            Err(e) => return set_rustuna_error(&e),
        };

        let src_study_id = {
            let mut guard = match src_storage.write() {
                Ok(g) => g,
                Err(e) => {
                    set_last_error(3, format!("Source storage lock poisoned: {e:?}"));
                    return 3;
                }
            };
            let studies = match guard.get_studies() {
                Ok(s) => s.clone(),
                Err(e) => return set_rustuna_error(&e),
            };
            match studies.iter().find(|s| s.name == src_name) {
                Some(s) => s.id,
                None => {
                    set_last_error(5, format!("Study '{src_name}' not found"));
                    return 5;
                }
            }
        };

        let dest_storage = match create_storage_backend(to_storage_type, &dest_path) {
            Ok(st) => st,
            Err(e) => return set_rustuna_error(&e),
        };

        match copy_study_core(src_storage, src_study_id, dest_storage, &dest_name) {
            Ok(_) => 0,
            Err(code) => code,
        }
    }));

    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_storage_copy_study: {e:?}"));
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_storage_get_studies_json(
    storage_type: i32,
    storage_path: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    if out_json.is_null() {
        set_last_error(-1, "out_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let path = if storage_path.is_null() {
            "".to_string()
        } else {
            unsafe { CStr::from_ptr(storage_path) }
                .to_string_lossy()
                .into_owned()
        };

        let storage = match create_storage_backend(storage_type, &path) {
            Ok(st) => st,
            Err(e) => return set_rustuna_error(&e),
        };

        let mut guard = match storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                return 3;
            }
        };

        let studies = match guard.get_studies() {
            Ok(s) => s.clone(),
            Err(e) => return set_rustuna_error(&e),
        };

        let mut summaries = Vec::with_capacity(studies.len());
        for st in studies {
            let trial_count = guard.get_n_trials(st.id, None).unwrap_or(0);
            let dirs: Vec<i32> = st
                .directions
                .iter()
                .map(|d| match d {
                    Direction::Minimize => 0,
                    Direction::Maximize => 1,
                })
                .collect();

            let mut user_attrs = HashMap::new();
            let mut system_attrs = HashMap::new();
            for (k, v) in st.attrs {
                match k {
                    rustuna_core::attr::AttrKey::User(key) => {
                        user_attrs.insert(key.as_str().to_string(), v);
                    }
                    rustuna_core::attr::AttrKey::System(key) => {
                        system_attrs.insert(key.as_str().to_string(), v);
                    }
                }
            }

            summaries.push(SerializableStudySummary {
                id: st.id,
                name: st.name,
                directions: dirs,
                user_attrs,
                system_attrs,
                trial_count,
            });
        }

        let json_str = match serde_json::to_string(&summaries) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(19, format!("JSON serialization error: {e:?}"));
                return 19;
            }
        };

        let c_str = CString::new(json_str).unwrap_or_default();
        unsafe {
            *out_json = c_str.into_raw();
        }
        0
    }));

    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(
                -99,
                format!("Panic in rustuna_storage_get_studies_json: {e:?}"),
            );
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_storage_delete_study(
    storage_type: i32,
    storage_path: *const c_char,
    study_name: *const c_char,
) -> i32 {
    if study_name.is_null() {
        set_last_error(-1, "study_name pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let path = if storage_path.is_null() {
            "".to_string()
        } else {
            unsafe { CStr::from_ptr(storage_path) }
                .to_string_lossy()
                .into_owned()
        };
        let name = unsafe { CStr::from_ptr(study_name) }.to_string_lossy();

        let storage = match create_storage_backend(storage_type, &path) {
            Ok(st) => st,
            Err(e) => return set_rustuna_error(&e),
        };

        let mut guard = match storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                return 3;
            }
        };

        let studies = match guard.get_studies() {
            Ok(s) => s.clone(),
            Err(e) => return set_rustuna_error(&e),
        };

        let study_id = match studies.iter().find(|s| s.name == name) {
            Some(s) => s.id,
            None => {
                set_last_error(5, format!("Study '{name}' not found"));
                return 5;
            }
        };

        match guard.delete_study(study_id) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));

    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_storage_delete_study: {e:?}"));
            -99
        }
    }
}

unsafe extern "C" {
    fn sqlite3_open(filename: *const c_char, ppDb: *mut *mut std::ffi::c_void) -> i32;
    fn sqlite3_close(db: *mut std::ffi::c_void) -> i32;
    fn sqlite3_exec(
        db: *mut std::ffi::c_void,
        sql: *const c_char,
        callback: Option<unsafe extern "C" fn() -> i32>,
        arg: *mut std::ffi::c_void,
        errmsg: *mut *mut c_char,
    ) -> i32;
    fn sqlite3_free(p: *mut std::ffi::c_void);
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_storage_sync_optuna_dashboard(path: *const c_char) -> i32 {
    if path.is_null() {
        set_last_error(-1, "path pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut db: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { sqlite3_open(path, &mut db) };
        if rc != 0 || db.is_null() {
            set_last_error(3, format!("Failed to open sqlite database: rc={rc}"));
            return 3;
        }
        let sql = b"\
        UPDATE study_system_attributes\n\
        SET value_json = '\"' || SUBSTR(value_json, 3) || '\"'\n\
        WHERE value_json LIKE 's:%';\n\
        \n\
        UPDATE study_system_attributes\n\
        SET value_json = SUBSTR(value_json, 3)\n\
        WHERE value_json LIKE 'i:%' OR value_json LIKE 'f:%';\n\
        \n\
        UPDATE study_system_attributes\n\
        SET value_json = 'null'\n\
        WHERE value_json = 'None';\n\
        \n\
        UPDATE trial_user_attributes\n\
        SET value_json = json_quote(value_json)\n\
        WHERE json_valid(value_json) = 0;\n\
        \n\
        INSERT INTO trial_system_attributes (trial_id, key, value_json)\n\
        SELECT trial_id, 'constraints', json_group_array(CAST(value_json AS REAL))\n\
        FROM trial_system_attributes\n\
        WHERE key LIKE 'constraints:%'\n\
        GROUP BY trial_id\n\
        ON CONFLICT(trial_id, key) DO UPDATE SET value_json = excluded.value_json;\0";

        let mut err_msg: *mut c_char = std::ptr::null_mut();
        let exec_rc = unsafe {
            sqlite3_exec(
                db,
                sql.as_ptr() as *const c_char,
                None,
                std::ptr::null_mut(),
                &mut err_msg,
            )
        };
        unsafe {
            sqlite3_close(db);
        }
        if exec_rc != 0 {
            let msg = if !err_msg.is_null() {
                let s = unsafe { CStr::from_ptr(err_msg).to_string_lossy().into_owned() };
                unsafe {
                    sqlite3_free(err_msg as *mut std::ffi::c_void);
                }
                s
            } else {
                format!("rc={exec_rc}")
            };
            set_last_error(3, format!("Failed to execute sync SQL batch: {msg}"));
            return 3;
        }
        0
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(
                -99,
                format!("Panic in rustuna_storage_sync_optuna_dashboard: {e:?}"),
            );
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_get_trials_filtered(
    study: *mut RustunaStudy,
    states_mask: u32,
    out_trials: *mut *mut *mut RustunaPersistedTrial,
    out_len: *mut usize,
) -> i32 {
    if study.is_null() || out_trials.is_null() || out_len.is_null() {
        set_last_error(
            -1,
            "study, out_trials, or out_len pointer is null".to_string(),
        );
        return -1;
    }

    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &*study };
        let mut guard = match s.inner.storage.write() {
            Ok(g) => g,
            Err(e) => {
                set_last_error(3, format!("Storage lock poisoned: {e:?}"));
                return 3;
            }
        };

        let trials_vec = match guard.get_trials(s.inner.id) {
            Ok(v) => v.clone(),
            Err(e) => return set_rustuna_error(&e),
        };

        let mut cat_cache = HashMap::new();
        let mut pt_ptrs: Vec<*mut RustunaPersistedTrial> = trials_vec
            .into_iter()
            .flatten()
            .filter(|t| {
                let state_bit = match t.state_values {
                    TrialStateValues::Running => 1 << 0,
                    TrialStateValues::Complete(_) => 1 << 1,
                    TrialStateValues::Pruned => 1 << 2,
                    TrialStateValues::Waiting => 1 << 3,
                    TrialStateValues::Fail => 1 << 4,
                };
                (states_mask & state_bit) != 0
            })
            .map(|t| {
                Box::into_raw(Box::new(make_persisted_trial(
                    &mut *guard,
                    s.inner.id,
                    t,
                    &mut cat_cache,
                )))
            })
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
            set_last_error(
                -99,
                format!("Panic in rustuna_study_get_trials_filtered: {e:?}"),
            );
            -99
        }
    }
}

// MARK: - Trial Injection & Seeding

#[derive(serde::Deserialize)]
struct AddTrialPayload {
    state: i32,
    values: Option<Vec<f64>>,
    params: HashMap<String, f64>,
    distributions: Option<HashMap<String, AddTrialDistributionPayload>>,
    intermediate_values: Option<HashMap<u32, f64>>,
    user_attrs: Option<HashMap<String, String>>,
    system_attrs: Option<HashMap<String, String>>,
}

#[derive(serde::Deserialize)]
struct AddTrialDistributionPayload {
    #[serde(rename = "type")]
    dist_type: String,
    low: Option<f64>,
    high: Option<f64>,
    step: Option<f64>,
    log: Option<bool>,
    choices_count: Option<usize>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_study_add_trial_json(
    study: *mut RustunaStudy,
    trial_json: *const c_char,
) -> i32 {
    if study.is_null() || trial_json.is_null() {
        set_last_error(-1, "study or trial_json pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &*study };
        let json_str = match unsafe { CStr::from_ptr(trial_json) }.to_str() {
            Ok(str_ref) => str_ref,
            Err(e) => {
                set_last_error(19, format!("Invalid UTF-8: {e:?}"));
                return 19;
            }
        };

        let payload: AddTrialPayload = match serde_json::from_str(json_str) {
            Ok(p) => p,
            Err(e) => {
                set_last_error(19, format!("JSON deserialize error: {e:?}"));
                return 19;
            }
        };

        let state_values = match payload.state {
            1 => TrialStateValues::Complete(payload.values.unwrap_or_default()),
            2 => TrialStateValues::Pruned,
            3 => TrialStateValues::Waiting,
            4 => TrialStateValues::Fail,
            _ => TrialStateValues::Running,
        };

        let mut distributions = HashMap::new();
        if let Some(dists) = payload.distributions {
            for (k, d) in dists {
                let dist = match d.dist_type.as_str() {
                    "int" => Distribution::new_int(
                        d.low.unwrap_or(0.0) as i64,
                        d.high.unwrap_or(100.0) as i64,
                        d.step.unwrap_or(1.0) as i64,
                        d.log.unwrap_or(false),
                    ),
                    "categorical" => Distribution::new_categorical(d.choices_count.unwrap_or(1)),
                    _ => Distribution::new_float(
                        d.low.unwrap_or(0.0),
                        d.high.unwrap_or(1.0),
                        d.step,
                        d.log.unwrap_or(false),
                    ),
                };
                distributions.insert(k, dist);
            }
        }
        for (k, v) in &payload.params {
            distributions
                .entry(k.clone())
                .or_insert_with(|| Distribution::new_float(*v, *v, None, false));
        }

        let mut attrs = rustuna_core::attr::Attrs::new();
        if let Some(user_attrs) = payload.user_attrs {
            for (k, v) in user_attrs {
                attrs.insert(rustuna_core::attr::AttrKey::User(k.into()), v);
            }
        }
        if let Some(sys_attrs) = payload.system_attrs {
            for (k, v) in sys_attrs {
                attrs.insert(rustuna_core::attr::AttrKey::System(k.into()), v);
            }
        }

        let pt = PersistedTrial {
            id: 0,
            study_id: s.inner.id,
            number: 0,
            state_values,
            internal_params: payload.params,
            distributions,
            intermediate_values: payload.intermediate_values.unwrap_or_default(),
            attrs,
            datetime_start: None,
            datetime_complete: None,
        };

        match s.inner.add_trial(pt) {
            Ok(()) => 0,
            Err(e) => set_rustuna_error(&e),
        }
    }));

    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_study_add_trial_json: {e:?}"));
            -99
        }
    }
}

// MARK: - In-process microbenchmarks (no subprocess fork)

#[unsafe(no_mangle)]
pub extern "C" fn rustuna_bench_e2e(
    n_trials: usize,
    seed: u64,
    use_random_sampler: bool,
    out_ns_per_trial: *mut u64,
) -> i32 {
    if out_ns_per_trial.is_null() || n_trials == 0 {
        set_last_error(-1, "out_ns_per_trial is null or n_trials==0".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let sampler: Arc<dyn Sampler> = if use_random_sampler {
            Arc::new(RandomSampler::seed_from_u64(seed))
        } else {
            Arc::new(TpeSampler::seed_from_u64(seed))
        };
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let study =
            match create_study_with_arc("bench", storage, sampler, vec![Direction::Minimize]) {
                Ok(s) => s,
                Err(e) => return set_rustuna_error(&e),
            };
        let dist_x = Distribution::new_float(-10.0, 10.0, None, false);
        let dist_y = Distribution::new_float(-10.0, 10.0, None, false);
        let start = Instant::now();
        for _ in 0..n_trials {
            let mut trial = match study.ask() {
                Ok(t) => t,
                Err(e) => return set_rustuna_error(&e),
            };
            let x = match trial.suggest("x", &dist_x) {
                Ok(v) => v,
                Err(e) => return set_rustuna_error(&e),
            };
            let y = match trial.suggest("y", &dist_y) {
                Ok(v) => v,
                Err(e) => return set_rustuna_error(&e),
            };
            let loss = (x - 2.0).powi(2) + (y + 5.0).powi(2);
            if let Err(e) = study.tell(trial.number, TrialStateValues::Complete(vec![loss])) {
                return set_rustuna_error(&e);
            }
        }
        let elapsed = start.elapsed();
        let ns_per_trial = elapsed.as_nanos() / (n_trials as u128);
        unsafe { *out_ns_per_trial = ns_per_trial as u64 };
        0
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_bench_e2e: {e:?}"));
            -99
        }
    }
}

// MARK: - Callback sampler (Swift/custom-sampler upcalls)

/// Suggest one float parameter. Returns 0 and sets `out_value` on success.
/// `trial_number` identifies the trial being configured (matches the number
/// reported at tell/finalization).
pub type CallbackSuggestFloat = unsafe extern "C" fn(
    ctx: *mut c_void,
    name: *const c_char,
    low: f64,
    high: f64,
    step: f64,
    log: bool,
    trial_number: u32,
    out_value: *mut f64,
) -> i32;

/// Suggest one integer parameter. Returns 0 and sets `out_value` on success.
pub type CallbackSuggestInt = unsafe extern "C" fn(
    ctx: *mut c_void,
    name: *const c_char,
    low: i64,
    high: i64,
    step: i64,
    log: bool,
    trial_number: u32,
    out_value: *mut i64,
) -> i32;

/// Suggest one categorical parameter. Returns 0 and sets `out_index` (must be
/// below `choices_count`) on success. Choices arrive as strings.
pub type CallbackSuggestCategorical = unsafe extern "C" fn(
    ctx: *mut c_void,
    name: *const c_char,
    choices: *const *const c_char,
    choices_count: usize,
    trial_number: u32,
    out_index: *mut usize,
) -> i32;

/// Virtual table for a foreign (Swift) sampler.
///
/// Any callback may be null, in which case that distribution kind falls back
/// to uniform random sampling. This gives partial fixing for free: custom
/// floats with random categoricals, or vice versa.
///
/// Thread-safety contract: callbacks run synchronously on the optimizing
/// thread and may run concurrently across threads. The foreign side must
/// synchronize its own state; the table itself is never mutated after
/// construction.
#[repr(C)]
pub struct CallbackSamplerVTable {
    pub ctx: *mut c_void,
    pub free_ctx: Option<unsafe extern "C" fn(*mut c_void)>,
    pub suggest_float: Option<CallbackSuggestFloat>,
    pub suggest_int: Option<CallbackSuggestInt>,
    pub suggest_categorical: Option<CallbackSuggestCategorical>,
}

struct CallbackSampler {
    vtable: CallbackSamplerVTable,
    fallback: RandomSampler,
}

// SAFETY: the vtable is immutable after construction; `ctx` validity and
// interior synchronization are the foreign side's contract (documented above).
unsafe impl Send for CallbackSampler {}
unsafe impl Sync for CallbackSampler {}

impl Drop for CallbackSampler {
    fn drop(&mut self) {
        if let Some(free) = self.vtable.free_ctx {
            if !self.vtable.ctx.is_null() {
                unsafe { free(self.vtable.ctx) };
            }
        }
    }
}

/// Maps a nonzero callback return to a sampler error.
fn check_callback(code: i32, name: &str, kind: &str) -> rustuna_core::Result<()> {
    if code != 0 {
        return Err(rustuna_core::Error::with_reason(
            rustuna_core::ErrorKind::SamplerError,
            format!("{kind} callback for {name:?} returned {code}"),
        ));
    }
    Ok(())
}

impl rustuna_core::sampler::Sampler for CallbackSampler {
    fn support_joint_sampling(&self) -> bool {
        false
    }

    fn sample_independent(
        &self,
        ctx: &rustuna_core::sampler::Context,
        storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> rustuna_core::Result<f64> {
        let c_name = CString::new(name).map_err(|_| {
            rustuna_core::Error::with_reason(
                rustuna_core::ErrorKind::SamplerError,
                format!("param name {name:?} contains a NUL byte"),
            )
        })?;
        match distribution {
            Distribution::Float { low, high, step, log } => {
                match self.vtable.suggest_float {
                    Some(cb) => {
                        let mut out = 0.0f64;
                        let code = unsafe {
                            cb(
                                self.vtable.ctx,
                                c_name.as_ptr(),
                                *low,
                                *high,
                                step.unwrap_or(0.0),
                                *log,
                                ctx.trial_number,
                                &mut out,
                            )
                        };
                        check_callback(code, name, "float")?;
                        Ok(out)
                    }
                    None => self.fallback.sample_independent(ctx, storage, name, distribution),
                }
            }
            Distribution::Int { low, high, step, log } => match self.vtable.suggest_int {
                Some(cb) => {
                    let mut out = 0i64;
                    let code = unsafe {
                        cb(
                            self.vtable.ctx,
                            c_name.as_ptr(),
                            *low,
                            *high,
                            *step,
                            *log,
                            ctx.trial_number,
                            &mut out,
                        )
                    };
                    check_callback(code, name, "int")?;
                    Ok(out as f64)
                }
                None => self.fallback.sample_independent(ctx, storage, name, distribution),
            },
            Distribution::Categorical { cardinality } => {
                match self.vtable.suggest_categorical {
                    Some(cb) => {
                        let labels = {
                            let mut guard = storage.write().map_err(|e| {
                                rustuna_core::Error::with_reason(
                                    rustuna_core::ErrorKind::StorageError,
                                    format!("Storage lock poisoned: {e:?}"),
                                )
                            })?;
                            guard
                                .get_category_labels(ctx.study_id, name, *cardinality)
                                .map_err(|e| {
                                    rustuna_core::Error::with_reason(
                                        rustuna_core::ErrorKind::StorageError,
                                        format!("Failed to read category labels: {e:?}"),
                                    )
                                })?
                        };
                        let label_strs: Vec<String> = match labels {
                            Some(ls) => ls.iter().map(|l| l.serialize()).collect(),
                            None => (0..*cardinality).map(|i| i.to_string()).collect(),
                        };
                        let c_labels: Vec<CString> = label_strs
                            .iter()
                            .map(|s| CString::new(s.as_str()).unwrap_or_default())
                            .collect();
                        let ptrs: Vec<*const c_char> =
                            c_labels.iter().map(|c| c.as_ptr()).collect();
                        let mut out = 0usize;
                        let code = unsafe {
                            cb(
                                self.vtable.ctx,
                                c_name.as_ptr(),
                                ptrs.as_ptr(),
                                ptrs.len(),
                                ctx.trial_number,
                                &mut out,
                            )
                        };
                        check_callback(code, name, "categorical")?;
                        if out >= ptrs.len() {
                            return Err(rustuna_core::Error::with_reason(
                                rustuna_core::ErrorKind::SamplerError,
                                format!(
                                    "categorical callback for {name:?} returned out-of-range index {out}"
                                ),
                            ));
                        }
                        Ok(out as f64)
                    }
                    None => self.fallback.sample_independent(ctx, storage, name, distribution),
                }
            }
        }
    }

    fn sample_joint(
        &self,
        _ctx: &rustuna_core::sampler::Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _search_space: &HashMap<String, Distribution>,
    ) -> rustuna_core::Result<HashMap<String, f64>> {
        Err(rustuna_core::Error::with_reason(
            rustuna_core::ErrorKind::SamplerError,
            "callback sampler does not support joint sampling".to_string(),
        ))
    }
}
/// Creates a sampler that upcalls into foreign code for suggestions.
///
/// The vtable is copied; `ctx` ownership moves to the sampler and `free_ctx`
/// runs when the sampler is freed. Null callbacks fall back to uniform random
/// for that distribution kind.
#[unsafe(no_mangle)]
pub extern "C" fn rustuna_sampler_callback_new(
    vtable: *const CallbackSamplerVTable,
    out_sampler: *mut *mut RustunaSampler,
) -> i32 {
    if vtable.is_null() || out_sampler.is_null() {
        set_last_error(-1, "vtable or out_sampler pointer is null".to_string());
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let vt = unsafe { &*vtable };
        // Copy the vtable by value; ctx ownership transfers to the sampler.
        // SAFETY: function pointers and the opaque ctx are opaque bits here.
        let vt_copy = CallbackSamplerVTable {
            ctx: vt.ctx,
            free_ctx: vt.free_ctx,
            suggest_float: vt.suggest_float,
            suggest_int: vt.suggest_int,
            suggest_categorical: vt.suggest_categorical,
        };
        let sampler = CallbackSampler {
            vtable: vt_copy,
            fallback: RandomSampler::new(),
        };
        let inner: Arc<dyn Sampler> = Arc::new(sampler);
        unsafe {
            *out_sampler = Box::into_raw(Box::new(RustunaSampler { inner }));
        }
        0
    }));
    match res {
        Ok(code) => code,
        Err(e) => {
            set_last_error(-99, format!("Panic in rustuna_sampler_callback_new: {e:?}"));
            -99
        }
    }
}
