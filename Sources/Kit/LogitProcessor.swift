import Foundation
import CoreML
import OSLog

/// Pick the next token from the provided logits.
class LogitProcessor {
    let model: DeferredModel

    let signposter = OSSignposter(subsystem: "com.stephenpanaro.llm-cli", category: "LogitProcessor")

    /// Initializes the LogitProcessor with a deferred model.
    /// - Parameter model: The deferred model used for logit processing.
    init(model: DeferredModel) {
        self.model = model
    }

    /// Loads the logit processor model.
    func load() {
        self.model.load()
    }

    /// Unloads the logit processor model.
    func unload() {
        self.model.unload()
    }

    /// Picks the token with the highest logit value.
    /// - Parameters:
    ///   - logits: The logits from the model.
    ///   - index: The index of the token to pick.
    /// - Returns: The token with the highest logit value.
    /// - Throws: An error if prediction fails.
    func argmax(logits: [MLMultiArray], index: Int? = nil) async throws -> Int {
        let state = signposter.beginInterval("Sample", "Argmax")
        defer { signposter.endInterval("Sample", state) }

        let outputs = try await predict(logits: logits)
        let argmaxArray = MLShapedArray<Int32>(outputs.featureValue(for: "argmax")!.multiArrayValue!)
        let prediction = argmaxArray[0, index ?? (argmaxArray.shape[1] - 1)].scalar ?? -1
        return Int(prediction)
    }

    /// Predicts the next token based on the logits.
    /// - Parameter logits: The logits from the model.
    /// - Returns: The feature provider with the predicted token.
    /// - Throws: An error if prediction fails.
    fileprivate func predict(logits: [MLMultiArray]) async throws -> MLFeatureProvider {
        let keysValues = logits.enumerated().map { (index, array) in
            return (logits.count == 1 ? "logits" : "logits_\(index)", array)
        }
        let inputs = try MLDictionaryFeatureProvider(
            dictionary: Dictionary(keysValues, uniquingKeysWith: { (first, _) in first })
        )
        return try await model.model!.prediction(from: inputs)
    }
}
