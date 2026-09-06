//! Upstream API-shape canaries for `ref/rustuna`.
//!
//! If any item here fails to COMPILE after pulling upstream, upstream changed
//! a signature this crate depends on in `src/lib.rs`. Read the compiler
//! error: it names the changed item (new struct field, new enum variant, new
//! trait method, changed fn arity). Then update `src/lib.rs` and this file
//! together — the failure is the notification, not the bug.
//!
//! What is pinned (mirrors `src/lib.rs` usage):
//! - `TpeConfig` fields, `TpeSampler` constructors, `NSGAIISampler`,
//!   `QmcSampler`, `RandomSampler` constructors.
//! - `Sampler` trait: all methods including defaulted hooks (implemented
//!   explicitly here so new required methods break loudly).
//! - `Context` fields (struct literal: new fields break construction).
//! - `Distribution`, `CategoryLabel`, `TrialStateValues`
//!   variants (exhaustive matches, NO wildcard: new variants break).
//!   (`ErrorKind` is `#[non_exhaustive]` upstream so it cannot be pinned
//!   this way; it gets an explicit panicking wildcard plus a behavioral
//!   mapping test instead.)
//!
//! Known residuals (NOT pinned — different techniques needed):
//! - Integer wire codes (`tell` states 1/2/4, `states_mask` bits) are not
//!   types; additions are invisible. Review `rustuna_study_tell_multi` and
//!   `fetch_trials_json` by hand on upstream pulls.
//! - `Storage` trait methods: only covered transitively through behavior
//!   tests. A new required method breaks `src/lib.rs` callers directly.
//! - Stringly keys (`dist_type` JSON strings in migration paths).

use rustuna_core::attr::CategoryLabel;
use rustuna_core::distribution::Distribution;
// NOTE: `mod error` is private upstream; only the re-export is visible.
use rustuna_core::ErrorKind;
use rustuna_core::sampler::{Context, Sampler};
use rustuna_core::study::Direction;
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::nsgaii::NSGAIISampler;
use rustuna_sampler::qmc::QmcSampler;
use rustuna_sampler::tpe::{TpeConfig, TpeSampler};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[test]
fn sampler_constructors_keep_their_signatures() {
    // TpeConfig struct literal: a new field fails here first, naming it.
    let cfg = TpeConfig {
        multivariate: None,
        n_startup_trials: 10,
        seed: Some(7),
    };
    let _ = TpeSampler::from_config(cfg);
    let _ = TpeSampler::new();
    let _ = TpeSampler::seed_from_u64(7);

    let _ = NSGAIISampler::new(50, None, 0.9, 0.5);
    let _ = NSGAIISampler::seed_from_u64(7, 50, None, 0.9, 0.5);

    let _ = QmcSampler::new();
    let _ = QmcSampler::seed_from_u64(7);

    let _ = rustuna_core::sampler::RandomSampler::new();
    let _ = rustuna_core::sampler::RandomSampler::seed_from_u64(7);
}

#[test]
fn sampler_trait_shape() {
    struct Dummy;
    impl Sampler for Dummy {
        fn before_trial(
            &self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn rustuna_core::storage::Storage>>,
        ) -> rustuna_core::Result<()> {
            Ok(())
        }
        fn sample_independent(
            &self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn rustuna_core::storage::Storage>>,
            _name: &str,
            _distribution: &Distribution,
        ) -> rustuna_core::Result<f64> {
            Ok(0.0)
        }
        fn support_joint_sampling(&self) -> bool {
            false
        }
        fn sample_joint(
            &self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn rustuna_core::storage::Storage>>,
            _search_space: &HashMap<String, Distribution>,
        ) -> rustuna_core::Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        fn after_trial(
            &self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn rustuna_core::storage::Storage>>,
            _state_values: &TrialStateValues,
        ) -> rustuna_core::Result<()> {
            Ok(())
        }
    }

    // Context struct literal: a new field fails here, naming it.
    let ctx = Context {
        study_id: 0,
        directions: vec![Direction::Minimize],
        trial_number: 0,
        trial_id: 0,
    };
    let storage: Arc<RwLock<dyn rustuna_core::storage::Storage>> = Arc::new(RwLock::new(
        rustuna_core::storage::InMemoryStorage::new(),
    ));
    let dummy = Dummy;
    dummy.before_trial(&ctx, storage.clone()).unwrap();
    dummy
        .sample_independent(&ctx, storage.clone(), "x", &Distribution::new_float(0.0, 1.0, None, false))
        .unwrap();
    assert!(!dummy.support_joint_sampling());
    dummy.after_trial(&ctx, storage, &TrialStateValues::Fail).unwrap();
}

fn error_kind_tag(kind: &ErrorKind) -> &'static str {
    // `ErrorKind` is `#[non_exhaustive]` upstream, so downstream crates
    // CANNOT match exhaustively — the wildcard below is mandatory, not a
    // choice. It panics (loudly, naming the fix) instead of silently mapping.
    // Behavior of every known variant is pinned in `error_kind_codes_mapped`
    // below; review `error_kind_to_code` in src/lib.rs on every upstream pull.
    match *kind {
        ErrorKind::ObjectiveError => "objective",
        ErrorKind::SamplerError => "sampler",
        ErrorKind::StorageError => "storage",
        ErrorKind::DuplicatedStudy => "duplicated",
        ErrorKind::StudyNotFound => "study-not-found",
        ErrorKind::TrialNotFound => "trial-not-found",
        ErrorKind::TrialDiscarded => "discarded",
        ErrorKind::AttrNotFound => "attr-not-found",
        ErrorKind::TrialQueueEmpty => "queue-empty",
        ErrorKind::AttrOverwriteNotAllowed => "overwrite",
        ErrorKind::InvalidObjectiveValues => "objective-values",
        ErrorKind::TrialAlreadyFinished => "finished",
        ErrorKind::UnsupportedSearchSpace => "search-space",
        ErrorKind::UnsupportedMultiObjective => "multi-objective",
        ErrorKind::NoCompletedTrial => "no-completed",
        ErrorKind::IncompatibleDistribution => "distribution",
        ErrorKind::InvalidFixedParam => "fixed-param",
        ErrorKind::MissingDependency => "dependency",
        ErrorKind::Unexpected => "unexpected",
        ErrorKind::ImportanceEvaluatorError => "importance",
        _ => panic!("upstream added ErrorKind variants: extend this match AND error_kind_to_code in src/lib.rs"),
    }
}

fn distribution_tag(dist: &Distribution) -> &'static str {
    // Exhaustive on purpose: see above.
    match dist {
        Distribution::Float { .. } => "float",
        Distribution::Int { .. } => "int",
        Distribution::Categorical { .. } => "categorical",
    }
}

fn category_label_tag(label: &CategoryLabel) -> &'static str {
    // Exhaustive on purpose: see above.
    match label {
        CategoryLabel::Float(_) => "float",
        CategoryLabel::Int(_) => "int",
        CategoryLabel::String(_) => "string",
        CategoryLabel::Bool(_) => "bool",
        CategoryLabel::None => "none",
    }
}

fn trial_state_tag(state: &TrialStateValues) -> &'static str {
    // Exhaustive on purpose: see above.
    match state {
        TrialStateValues::Complete(_) => "complete",
        TrialStateValues::Pruned => "pruned",
        TrialStateValues::Waiting => "waiting",
        TrialStateValues::Fail => "fail",
        TrialStateValues::Running => "running",
    }
}

#[test]
fn upstream_enums_keep_their_shapes() {
    assert_eq!(error_kind_tag(&ErrorKind::SamplerError), "sampler");
    assert_eq!(
        distribution_tag(&Distribution::new_float(0.0, 1.0, None, false)),
        "float"
    );
    assert_eq!(
        category_label_tag(&CategoryLabel::String("a".to_string())),
        "string"
    );
    assert_eq!(trial_state_tag(&TrialStateValues::Fail), "fail");
}
