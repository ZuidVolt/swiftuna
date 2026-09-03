import Distributed
import Foundation
import Testing

@testable import Swiftuna
@testable import SwiftunaDistributed

/// Concrete coordinator type used across distributed tests.
typealias TestCoordinator = StudyCoordinator<LocalTestingDistributedActorSystem>

/// Builds a coordinator plus its study with unique names per call.
func makeTestCoordinator(
    named name: String = "test",
    directions: [Direction] = [.minimize],
    pruner: any Pruner = NopPruner(),
    maxInFlight: Int = .max,
    leasePolicy: LeasePolicy? = nil,
    sample: @escaping AskFunction.SamplerClosure
) throws -> (TestCoordinator, Study) {
    let study = try createStudy(
        name: "\(name)_\(UUID().uuidString)", directions: directions, pruner: pruner)
    let coordinator = TestCoordinator(
        study: study,
        askFunction: AskFunction(sample),
        actorSystem: LocalTestingDistributedActorSystem(),
        maxInFlight: maxInFlight,
        leasePolicy: leasePolicy)
    return (coordinator, study)
}

/// Builds a coordinator with a custom sampler.
func makeTestCoordinator<S: Sampler>(
    named name: String = "test",
    directions: [Direction] = [.minimize],
    sampler: S,
    pruner: any Pruner = NopPruner(),
    maxInFlight: Int = .max,
    sample: @escaping AskFunction.SamplerClosure
) throws -> (TestCoordinator, Study) {
    let study = try createStudy(
        name: "\(name)_\(UUID().uuidString)",
        directions: directions,
        sampler: sampler,
        pruner: pruner)
    let coordinator = TestCoordinator(
        study: study,
        askFunction: AskFunction(sample),
        actorSystem: LocalTestingDistributedActorSystem(),
        maxInFlight: maxInFlight)
    return (coordinator, study)
}
