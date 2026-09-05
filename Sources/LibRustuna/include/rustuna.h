#ifndef RUSTUNA_H
#define RUSTUNA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types
typedef struct RustunaStudy RustunaStudy;
typedef struct RustunaTrial RustunaTrial;
typedef struct RustunaPersistedTrial RustunaPersistedTrial;
typedef struct RustunaSampler RustunaSampler;

/**
 * @brief Canonical error codes returned by Rustuna FFI functions.
 */
typedef enum RustunaErrorCode {
    RUSTUNA_SUCCESS = 0,
    RUSTUNA_ERR_INVALID_ARGUMENT = -1,
    RUSTUNA_ERR_EMPTY_CHOICES = -3,
    RUSTUNA_ERR_INVALID_RANGE = -4,
    RUSTUNA_ERR_PANIC = -99,
    RUSTUNA_ERR_OBJECTIVE = 1,
    RUSTUNA_ERR_SAMPLER = 2,
    RUSTUNA_ERR_STORAGE = 3,
    RUSTUNA_ERR_DUPLICATED_STUDY = 4,
    RUSTUNA_ERR_STUDY_NOT_FOUND = 5,
    RUSTUNA_ERR_TRIAL_NOT_FOUND = 6,
    RUSTUNA_ERR_TRIAL_DISCARDED = 7,
    RUSTUNA_ERR_ATTR_NOT_FOUND = 8,
    RUSTUNA_ERR_TRIAL_QUEUE_EMPTY = 9,
    RUSTUNA_ERR_ATTR_OVERWRITE_NOT_ALLOWED = 10,
    RUSTUNA_ERR_INVALID_OBJECTIVE_VALUES = 11,
    RUSTUNA_ERR_TRIAL_ALREADY_FINISHED = 12,
    RUSTUNA_ERR_UNSUPPORTED_SEARCH_SPACE = 13,
    RUSTUNA_ERR_UNSUPPORTED_MULTI_OBJECTIVE = 14,
    RUSTUNA_ERR_NO_COMPLETED_TRIAL = 15,
    RUSTUNA_ERR_INCOMPATIBLE_DISTRIBUTION = 16,
    RUSTUNA_ERR_INVALID_FIXED_PARAM = 17,
    RUSTUNA_ERR_MISSING_DEPENDENCY = 18,
    RUSTUNA_ERR_UNEXPECTED = 19,
    RUSTUNA_ERR_IMPORTANCE_EVALUATOR = 20,
    RUSTUNA_ERR_SEARCH_SPACE_EXHAUSTED = 21
} RustunaErrorCode;

// Global error & memory utilities

/**
 * @brief Atomically consumes and resets the last error recorded on the calling thread.
 * @param out_code Pointer to receive the integer error code.
 * @param out_msg Pointer to receive the newly-allocated error message string (must be freed via rustuna_string_free).
 * @return 1 if an error was present and taken, 0 if no error was recorded.
 */
int32_t rustuna_take_last_error(int32_t* out_code, char** out_msg);

/**
 * @brief Frees a C string allocated by Rustuna FFI.
 */
void rustuna_string_free(char* s);

// Sampler APIs
/**
 * @brief Full TPE configuration. `multivariate`: negative for automatic
 * selection (matching Optuna), 0 for independent, positive for joint
 * sampling. `n_startup_trials`: completed trials before TPE engages, 0 keeps
 * the engine default (10).
 */
RustunaSampler* rustuna_sampler_tpe_full(uint64_t seed, bool has_seed, int8_t multivariate, size_t n_startup_trials);
RustunaSampler* rustuna_sampler_random_new(uint64_t seed, bool has_seed);
int32_t rustuna_sampler_nsgaii_new(size_t population_size, double mutation_prob, double crossover_prob,
                                   double swapping_prob, uint64_t seed, RustunaSampler** out_sampler);
RustunaSampler* rustuna_sampler_qmc_new(uint64_t seed, bool has_seed);
int32_t rustuna_sampler_grid_new(const char* search_space_json, uint64_t seed, bool has_seed,
                                 RustunaSampler** out_sampler);
void rustuna_sampler_free(RustunaSampler* sampler);
/**
 * Foreign_sampler upcall signatures. Each returns 0 and sets its out-pointer
 * on success, non-zero on failure (surfaced as a sampler error).
 * Callbacks run synchronously on the optimizing thread and may run
 * concurrently across threads: the foreign side must synchronize its state.
 */
typedef int32_t (*RustunaSuggestFloatFn)(void* ctx, const char* name, double low, double high,
                                         double step, bool log, uint32_t trial_number, double* out_value);
typedef int32_t (*RustunaSuggestIntFn)(void* ctx, const char* name, int64_t low, int64_t high,
                                       int64_t step, bool log, uint32_t trial_number, int64_t* out_value);
typedef int32_t (*RustunaSuggestCategoricalFn)(void* ctx, const char* name, const char* const* choices,
                                               size_t choices_count, uint32_t trial_number, size_t* out_index);
/**
 * Virtual table for a foreign sampler. Any callback may be NULL, in which
 * case that distribution kind falls back to uniform random sampling.
 * `ctx` ownership moves to the sampler; `free_ctx` runs when it is freed.
 */
typedef struct {
    void* ctx;
    void (*free_ctx)(void* ctx);
    RustunaSuggestFloatFn suggest_float;
    RustunaSuggestIntFn suggest_int;
    RustunaSuggestCategoricalFn suggest_categorical;
} RustunaCallbackVTable;
/**
 * @brief Creates a sampler that upcalls into foreign code for suggestions.
 * @return 0 on success (with `*out_sampler` set), non-zero otherwise.
 */
int32_t rustuna_sampler_callback_new(const RustunaCallbackVTable* vtable, RustunaSampler** out_sampler);

// Study APIs
int32_t rustuna_study_create_full(const char* name, const int32_t* directions, size_t directions_len,
                                  int32_t storage_type, // 0 = InMemory, 1 = SQLite, 2 = Journal
                                  const char* storage_path, bool load_if_exists, RustunaSampler* sampler,
                                  RustunaStudy** out_study);
int32_t rustuna_study_load(const char* name, int32_t storage_type, const char* storage_path, RustunaSampler* sampler,
                           RustunaStudy** out_study);
int32_t rustuna_study_set_user_attr(RustunaStudy* study, const char* key, const char* val);
int32_t rustuna_study_get_user_attr(RustunaStudy* study, const char* key, char** out_val);
int32_t rustuna_study_enqueue_trial(RustunaStudy* study, const char* params_json, const char* user_attrs_json);
/**
 * @brief Enqueues a trial with typed parameter arrays instead of JSON.
 *
 * Same queue semantics as `rustuna_study_enqueue_trial` without the
 * serialize/parse round trip for params. Parallel arrays of length `count`:
 * `names` (param names), `kinds` (0=int, 1=double, 2=string, 3=bool),
 * `num_values` (int/double/bool payload; bool is `!= 0.0`), `str_values`
 * (string payload, non-null where kind == 2). Integers must fit the `f64`
 * mantissa exactly. `user_attrs_json` keeps the JSON form.
 * When `count` is zero the arrays are not touched (may be NULL).
 * @return 0 on success, or non-zero error code on failure.
 */
int32_t rustuna_study_enqueue_typed(RustunaStudy* study, const char* const* names, const uint8_t* kinds,
                                    const double* num_values, const char* const* str_values, size_t count,
                                    const char* user_attrs_json);
int32_t rustuna_study_get_param_importances(RustunaStudy* study, bool normalize, const char* params_json,
                                            char** out_json);
void rustuna_study_free(RustunaStudy* study);

int32_t rustuna_study_ask(RustunaStudy* study, RustunaTrial** out_trial);
int32_t rustuna_study_tell_multi(RustunaStudy* study, uint32_t trial_number, int32_t state, const double* values,
                                 size_t values_len, const char* intermediate_json);
int32_t rustuna_study_add_trial_json(RustunaStudy* study, const char* trial_json);

int32_t rustuna_study_get_best_trial(RustunaStudy* study, RustunaPersistedTrial** out_trial);
int32_t rustuna_study_get_best_trials(RustunaStudy* study, RustunaPersistedTrial*** out_trials, size_t* out_len);
int32_t rustuna_study_get_trials(RustunaStudy* study, RustunaPersistedTrial*** out_trials, size_t* out_len);
int32_t rustuna_study_get_trials_filtered(RustunaStudy* study, uint32_t states_mask,
                                          RustunaPersistedTrial*** out_trials, size_t* out_len);
/**
 * @brief Fetches trials at or above a trial number, filtered by state.
 *
 * Incremental-refresh counterpart of `rustuna_study_get_trials_json`: trials
 * below `from_number` are skipped before per-trial conversion, so refresh
 * costs O(new) instead of O(history). Trial numbers are dense from 0, so
 * passing the known count yields exactly the unseen tail.
 * @return 0 on success, or non-zero error code on failure.
 */
int32_t rustuna_study_get_trials_json_since(RustunaStudy* study, uint32_t states_mask, uint32_t from_number,
                                            char** out_json, size_t* out_len);
void rustuna_trials_buffer_free(RustunaPersistedTrial** trials, size_t len);

// Storage Lifecycle & Study Management APIs
int32_t rustuna_study_copy(RustunaStudy* study, int32_t to_storage_type, const char* to_storage_path,
                           const char* to_study_name, RustunaStudy** out_study);
int32_t rustuna_storage_copy_study(int32_t from_storage_type, const char* from_storage_path,
                                   const char* from_study_name, int32_t to_storage_type, const char* to_storage_path,
                                   const char* to_study_name);
int32_t rustuna_storage_get_studies_json(int32_t storage_type, const char* storage_path, char** out_json);
int32_t rustuna_storage_delete_study(int32_t storage_type, const char* storage_path, const char* study_name);

/**
 * @brief Formats and synchronizes internal SQLite tables to ensure full binary compatibility with Optuna dashboard.
 * @param path File system path to the SQLite database.
 * @return 0 on success, or non-zero error code on failure.
 */
int32_t rustuna_storage_sync_optuna_dashboard(const char* path);

// Trial APIs (Active mutable trial)
uint32_t rustuna_trial_get_number(RustunaTrial* trial);
int32_t rustuna_trial_suggest_float(RustunaTrial* trial, const char* name, double low, double high, double step,
                                    bool log, double* out_val);
int32_t rustuna_trial_suggest_int(RustunaTrial* trial, const char* name, int64_t low, int64_t high, int64_t step,
                                  bool log, int64_t* out_val);
int32_t rustuna_trial_suggest_categorical(RustunaTrial* trial, const char* name, const char* const* choices,
                                          size_t choices_count, size_t* out_index);
int32_t rustuna_trial_set_user_attr(RustunaTrial* trial, const char* key, const char* val);
int32_t rustuna_trial_set_constraint(RustunaTrial* trial, const char* key, double value);
void rustuna_trial_free(RustunaTrial* trial);

// PersistedTrial APIs (Frozen immutable trial)
int32_t rustuna_persisted_trial_get_json(const RustunaPersistedTrial* trial, char** out_json);
void rustuna_persisted_trial_free(RustunaPersistedTrial* trial);

// In-process benchmark (no subprocess fork)
int32_t rustuna_bench_e2e(size_t n_trials, uint64_t seed, bool use_random_sampler, uint64_t* out_ns_per_trial);

#ifdef __cplusplus
}
#endif

#endif /* RUSTUNA_H */
