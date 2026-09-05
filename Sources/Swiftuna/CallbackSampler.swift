import Foundation
internal import LibRustuna

/// A sampler implemented in Swift, called back from the Rustuna engine.
///
/// Assign any subset of the suggest closures; distribution kinds without a
/// closure fall back to uniform random sampling inside Rustuna, so partial
/// fixing (custom floats, random categoricals) works with no extra code.
/// Unlike enqueue-ahead drivers, callbacks observe each parameter at its own
/// `suggest` call, conditioned on earlier params in the same trial.
///
/// Thread-safety contract: closures run synchronously on the optimizing
/// thread and may run concurrently across threads. Capture only `Sendable`
/// state or synchronize inside the closure. Returning `nil` records a
/// sampler error for the trial.
///
/// Checking out trials from inside a closure throws
/// ``SwiftunaError/reentrantAsk(_:)`` instead of leaking. The reentrancy
/// guard costs roughly half a microsecond per suggestion (thread-local
/// bookkeeping); irrelevant past short trials, measured on B20.
///
/// ### Example
/// ```swift
/// let sampler = CallbackSampler(onFloat: { name, low, high, step, log, trialNumber in
///     Double.random(in: low...high)
/// })
/// let study = try Swiftuna.createStudy(sampler: sampler)
/// try study.optimize(nTrials: 20) { trial in
///     let x = try trial.suggest("x", in: -10.0...10.0)
///     return x * x
/// }
/// ```
public struct CallbackSampler: Sampler, Sendable {
    /// Suggests a float parameter. Return `nil` to fail the suggestion.
    /// `step` is `nil` for continuous ranges. `trialNumber` identifies the
    /// trial being configured (matches `Trial.number` at tell time).
    public typealias FloatFn = @Sendable (String, Double, Double, Double?, Bool, Int) -> Double?
    /// Suggests an integer parameter. Return `nil` to fail the suggestion.
    public typealias IntFn = @Sendable (String, Int64, Int64, Int64, Bool, Int) -> Int64?
    /// Suggests a categorical parameter by choice index. Return `nil` to fail.
    public typealias CategoricalFn = @Sendable (String, [String], Int) -> Int?

    /// Closures snapshot into the sampler handle at ``makeRawHandle()``;
    /// later reassignment would silently do nothing, so they are immutable.
    public let onFloat: FloatFn?
    public let onInt: IntFn?
    public let onCategorical: CategoricalFn?

    public init(onFloat: FloatFn? = nil, onInt: IntFn? = nil, onCategorical: CategoricalFn? = nil) {
        self.onFloat = onFloat
        self.onInt = onInt
        self.onCategorical = onCategorical
    }

    public func makeRawHandle() -> OpaquePointer? {
        let box = CallbackBox(float: onFloat, int: onInt, categorical: onCategorical)
        let ctx = Unmanaged.passRetained(box).toOpaque()
        var table = RustunaCallbackVTable(
            ctx: ctx,
            free_ctx: callbackFreeContext,
            suggest_float: onFloat == nil ? nil : callbackSuggestFloat,
            suggest_int: onInt == nil ? nil : callbackSuggestInt,
            suggest_categorical: onCategorical == nil ? nil : callbackSuggestCategorical
        )
        var out: OpaquePointer?
        let code = withUnsafePointer(to: &table) {
            rustuna_sampler_callback_new($0, &out)
        }
        guard code == 0 else {
            // Rust never took ownership: undo the retain.
            Unmanaged<CallbackBox>.fromOpaque(ctx).release()
            return nil
        }
        return out
    }
}

/// Thread-local depth flag: set around user-closure invocation in the
/// trampolines below. `Study.ask()` / `askEnqueued` refuse checkout while
/// set — a trial checked out mid-suggestion would leak unfinished and
/// scramble queue pairing. Same-thread only, which is exactly the upcall
/// shape (synchronous FFI, no executor hop).
internal let samplerCallbackDepthKey = "org.swiftuna.samplerCallback.active"

/// Runs `body` with the callback-depth flag set for the current thread.
private func withCallbackDepth<R>(_ body: () -> R) -> R {
    let state = Thread.current.threadDictionary
    state[samplerCallbackDepthKey] = true
    defer { state.removeObject(forKey: samplerCallbackDepthKey) }
    return body()
}

/// Retained as the vtable `ctx`; released by `callbackFreeContext` when the
/// Rust sampler is freed. Holds only `@Sendable` closures.
private final class CallbackBox: Sendable {
    let floatFn: CallbackSampler.FloatFn?
    let intFn: CallbackSampler.IntFn?
    let categoricalFn: CallbackSampler.CategoricalFn?

    init(
        float: CallbackSampler.FloatFn?,
        int: CallbackSampler.IntFn?,
        categorical: CallbackSampler.CategoricalFn?
    ) {
        self.floatFn = float
        self.intFn = int
        self.categoricalFn = categorical
    }
}

private func callbackFreeContext(_ ctx: UnsafeMutableRawPointer?) {
    guard let ctx else { return }
    Unmanaged<CallbackBox>.fromOpaque(ctx).release()
}

private func callbackBox(_ ctx: UnsafeMutableRawPointer?) -> CallbackBox? {
    guard let ctx else { return nil }
    return Unmanaged<CallbackBox>.fromOpaque(ctx).takeUnretainedValue()
}

private func callbackSuggestFloat(
    _ ctx: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ low: Double,
    _ high: Double,
    _ step: Double,
    _ log: Bool,
    _ trialNumber: UInt32,
    _ out: UnsafeMutablePointer<Double>?
) -> Int32 {
    guard let box = callbackBox(ctx), let name, let out,
        let value = withCallbackDepth({
            box.floatFn?(String(cString: name), low, high, step > 0 ? step : nil, log, Int(trialNumber))
        })
    else {
        return 1
    }
    out.pointee = value
    return 0
}

private func callbackSuggestInt(
    _ ctx: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ low: Int64,
    _ high: Int64,
    _ step: Int64,
    _ log: Bool,
    _ trialNumber: UInt32,
    _ out: UnsafeMutablePointer<Int64>?
) -> Int32 {
    guard let box = callbackBox(ctx), let name, let out,
        let value = withCallbackDepth({
            box.intFn?(String(cString: name), low, high, step, log, Int(trialNumber))
        })
    else {
        return 1
    }
    out.pointee = value
    return 0
}

private func callbackSuggestCategorical(
    _ ctx: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ choices: UnsafePointer<UnsafePointer<CChar>?>?,
    _ count: Int,
    _ trialNumber: UInt32,
    _ out: UnsafeMutablePointer<Int>?
) -> Int32 {
    guard let box = callbackBox(ctx), let name, let choices, let out else {
        return 1
    }
    var labels: [String] = []
    labels.reserveCapacity(count)
    for i in 0..<count {
        guard let cStr = choices[i] else { return 1 }
        labels.append(String(cString: cStr))
    }
    guard let index = withCallbackDepth({
        box.categoricalFn?(String(cString: name), labels, Int(trialNumber))
    }), index >= 0, index < count
    else {
        return 1
    }
    out.pointee = index
    return 0
}
