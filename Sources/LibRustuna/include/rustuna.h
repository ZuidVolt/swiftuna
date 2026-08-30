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

// Global error & memory utilities
int32_t rustuna_last_error_code(void);
const char* rustuna_last_error_message(void);
void rustuna_string_free(char* s);

// Sampler APIs
RustunaSampler* rustuna_sampler_tpe_new(uint64_t seed, bool has_seed);
RustunaSampler* rustuna_sampler_random_new(uint64_t seed, bool has_seed);
int32_t rustuna_sampler_nsgaii_new(
    size_t population_size,
    double mutation_prob,
    double crossover_prob,
    double swapping_prob,
    uint64_t seed,
    RustunaSampler** out_sampler
);
int32_t rustuna_sampler_qmc_new(
    uint64_t seed,
    bool has_seed,
    RustunaSampler** out_sampler
);
void rustuna_sampler_free(RustunaSampler* sampler);

// Study APIs
int32_t rustuna_study_create_full(
    const char* name,
    const int32_t* directions,
    size_t directions_len,
    int32_t storage_type, // 0 = InMemory, 1 = SQLite, 2 = Journal
    const char* storage_path,
    bool load_if_exists,
    RustunaSampler* sampler,
    RustunaStudy** out_study
);
int32_t rustuna_study_load(
    const char* name,
    int32_t storage_type,
    const char* storage_path,
    RustunaSampler* sampler,
    RustunaStudy** out_study
);
int32_t rustuna_study_set_user_attr(RustunaStudy* study, const char* key, const char* val);
int32_t rustuna_study_get_user_attr(RustunaStudy* study, const char* key, char** out_val);
int32_t rustuna_study_enqueue_trial(
    RustunaStudy* study,
    const char* params_json,
    const char* user_attrs_json
);
int32_t rustuna_study_get_param_importances(
    RustunaStudy* study,
    bool normalize,
    const char* params_json,
    char** out_json
);
void rustuna_study_free(RustunaStudy* study);

int32_t rustuna_study_ask(RustunaStudy* study, RustunaTrial** out_trial);
int32_t rustuna_study_tell(
    RustunaStudy* study,
    uint32_t trial_number,
    int32_t state, // 0 = Running, 1 = Complete, 2 = Pruned, 3 = Waiting, 4 = Fail
    double value
);
int32_t rustuna_study_tell_multi(
    RustunaStudy* study,
    uint32_t trial_number,
    int32_t state,
    const double* values,
    size_t values_len
);

int32_t rustuna_study_get_best_trial(RustunaStudy* study, RustunaPersistedTrial** out_trial);
int32_t rustuna_study_get_best_trials(
    RustunaStudy* study,
    RustunaPersistedTrial*** out_trials,
    size_t* out_len
);
int32_t rustuna_study_get_trials(
    RustunaStudy* study,
    RustunaPersistedTrial*** out_trials,
    size_t* out_len
);
void rustuna_trials_buffer_free(RustunaPersistedTrial** trials, size_t len);

// Trial APIs (Active mutable trial)
uint32_t rustuna_trial_get_number(RustunaTrial* trial);
int32_t rustuna_trial_suggest_float(
    RustunaTrial* trial,
    const char* name,
    double low,
    double high,
    double step,
    bool log,
    double* out_val
);
int32_t rustuna_trial_suggest_int(
    RustunaTrial* trial,
    const char* name,
    int64_t low,
    int64_t high,
    int64_t step,
    bool log,
    int64_t* out_val
);
int32_t rustuna_trial_suggest_categorical(
    RustunaTrial* trial,
    const char* name,
    const char* const* choices,
    size_t choices_count,
    size_t* out_index
);
int32_t rustuna_trial_set_user_attr(RustunaTrial* trial, const char* key, const char* val);
int32_t rustuna_trial_set_constraint(RustunaTrial* trial, const char* key, double value);
int32_t rustuna_trial_set_constraints_json(RustunaTrial* trial, const char* json_str);
void rustuna_trial_free(RustunaTrial* trial);

// PersistedTrial APIs (Frozen immutable trial)
uint32_t rustuna_persisted_trial_get_number(const RustunaPersistedTrial* trial);
int32_t rustuna_persisted_trial_get_state(const RustunaPersistedTrial* trial);
bool rustuna_persisted_trial_get_value(const RustunaPersistedTrial* trial, double* out_val);
bool rustuna_persisted_trial_get_values(
    const RustunaPersistedTrial* trial,
    double** out_vals,
    size_t* out_len
);
void rustuna_values_buffer_free(double* vals, size_t len);
int32_t rustuna_persisted_trial_get_params_json(const RustunaPersistedTrial* trial, char** out_json);
int32_t rustuna_persisted_trial_get_user_attrs_json(const RustunaPersistedTrial* trial, char** out_json);
int32_t rustuna_persisted_trial_get_constraints_json(const RustunaPersistedTrial* trial, char** out_json);
void rustuna_persisted_trial_free(RustunaPersistedTrial* trial);

#ifdef __cplusplus
}
#endif

#endif /* RUSTUNA_H */
