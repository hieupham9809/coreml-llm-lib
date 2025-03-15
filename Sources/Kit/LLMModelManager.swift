import Foundation
import CoreML
import Hub
import Tokenizers
import OSLog
private let logger = Logger(subsystem: "com.llmlib.kit", category: "LLMModelManager")
/// LLMModelManager manages the loading and usage of a language model for text generation.
public class LLMModelManager {
    private var modelPipeline: ModelPipeline?
    private var tokenizer: Tokenizer?
    private let stopTokens = [128001, 128009]

    public init() {}

    /// Load the model from a Huggingface repo ID.
    /// - Parameters:
    ///   - repoID: The Huggingface repository ID.
    ///   - localModelDirectory: The local directory where the model is stored.
    ///   - localModelPrefix: The prefix for the local model files.
    ///   - cacheProcessorModelName: The name of the cache processor model file.
    ///   - logitProcessorModelName: The name of the logit processor model file.
    ///   - tokenizerName: The name of the tokenizer.
    ///   - verbose: Whether to print verbose output.
    /// - Throws: An error if the model or tokenizer fails to load.
    public func loadModel(repoID: String, localModelDirectory: URL? = nil, localModelPrefix: String? = nil, cacheProcessorModelName: String = "cache-processor.mlmodelc", logitProcessorModelName: String = "logit-processor.mlmodelc", tokenizerName: String? = nil, verbose: Bool = false) async throws {
        var modelDirectory = localModelDirectory
        if !repoID.isEmpty {
            modelDirectory = try await downloadModel(repoID: repoID)
        }

        guard let modelDirectory else {
            throw LLMModelManagerError.modelDirectoryNotProvided
        }

        guard let tokenizerName = tokenizerName ?? inferTokenizer(repoID: repoID, localModelPrefix: localModelPrefix, localModelDirectory: localModelDirectory) else {
            throw LLMModelManagerError.tokenizerNotProvided
        }

        self.modelPipeline = try ModelPipeline.from(
            folder: modelDirectory,
            modelPrefix: localModelPrefix,
            cacheProcessorModelName: cacheProcessorModelName,
            logitProcessorModelName: logitProcessorModelName
        )
        try modelPipeline?.load()

        self.tokenizer = try await AutoTokenizer.from(cached: tokenizerName, hubApi: .init(hfToken: HubApi.defaultToken()))
        if verbose { logger.info("Tokenizer \(tokenizerName) loaded.") }
    }

    /// Generate text based on the input text and maximum number of new tokens.
    /// - Parameters:
    ///   - inputText: The input text to generate text from.
    ///   - maxNewTokens: The maximum number of new tokens to generate.
    ///   - stopToken: The token that indicates the end of the conversation.
    /// - Returns: The generated text.
    /// - Throws: An error if text generation fails.
    public func generateText(inputText: String, maxNewTokens: Int) async throws -> String {
        guard let modelPipeline = modelPipeline, let tokenizer = tokenizer else {
            throw LLMModelManagerError.modelNotLoaded
        }

        let generator = TextGenerator(pipeline: modelPipeline, tokenizer: tokenizer)
        return try await generator.generateText(text: inputText, maxNewTokens: maxNewTokens, stopTokens: stopTokens)
    }

    /// Generate text as an async sequence of string chunks, useful for streaming UI updates.
    /// - Parameters:
    ///   - inputText: The input text to generate text from.
    ///   - maxNewTokens: The maximum number of new tokens to generate.
    /// - Returns: An async sequence that yields chunks of generated text as they become available.
    /// - Throws: An error if text generation fails.
    public func generateTextStream(inputText: String, maxNewTokens: Int) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let modelPipeline = modelPipeline, let tokenizer = tokenizer else {
                        throw LLMModelManagerError.modelNotLoaded
                    }
                    
                    // Load the model if not already loaded
                    let loadTimer = CodeTimer()
                    try modelPipeline.load()
                    let loadDuration = loadTimer.elapsed()
                    logger.info("Model loaded in \(loadDuration.converted(to: .seconds).value.formatted(.number.precision(.fractionLength(2)))) seconds")
                    
                    // Encode the input text to tokens
                    let inputTokens = tokenizer.encode(text: inputText)
                    logger.info("Input tokens count: \(inputTokens.count)")
                    
                    // Check if input contains stop tokens and log them
                    let inputStopTokens = inputTokens.filter { stopTokens.contains($0) }
                    if !inputStopTokens.isEmpty {
                        logger.notice("Input contains stop tokens: \(inputStopTokens)")
                    }
                    
                    // Keep track of the previously generated text to determine what's new
                    var generatedText = ""
                    var tokenCount = 0
                    
                    // Track which tokens are newly generated vs. part of the input
                    var isGeneratingNewTokens = false
                    
                    do {
                        // Process predictions from the model pipeline
                        for try await prediction in try modelPipeline.predictWithInputTokens(tokens: inputTokens, maxNewTokens: maxNewTokens) {
                            // Get all tokens generated so far
                            let allTokens = prediction.allTokens
                            
                            // Determine if we're now generating new tokens (past the input)
                            if allTokens.count > inputTokens.count {
                                isGeneratingNewTokens = true
                            }
                            
                            // Only check for stop tokens in newly generated content, not in the input
                            let isStopToken = isGeneratingNewTokens && stopTokens.contains(prediction.newToken)
                            
                            if isStopToken {
                                logger.info("Stop token encountered: \(prediction.newToken) in generated content")
                                break
                            }
                            
                            // Only count tokens that are newly generated
                            if isGeneratingNewTokens {
                                tokenCount += 1
                            }
                            
                            // Safety check to ensure we have enough tokens
                            guard allTokens.count >= inputTokens.count else {
                                logger.warning("Not enough tokens in prediction")
                                continue
                            }
                            
                            // Only process tokens beyond the input tokens
                            if allTokens.count > inputTokens.count {
                                let generatedTokens = Array(allTokens.dropFirst(inputTokens.count))
                                
                                // Decode the full generated text
                                let currentGeneratedText = tokenizer.decode(tokens: generatedTokens)
                                
                                // If we have new text, yield it
                                if currentGeneratedText.count > generatedText.count {
                                    let newTextChunk = String(currentGeneratedText.dropFirst(generatedText.count))
                                    if !newTextChunk.isEmpty {
                                        continuation.yield(newTextChunk)
                                    }
                                    generatedText = currentGeneratedText
                                }
                                
                                // Debug output every 10 tokens
                                if tokenCount % 10 == 0 && tokenCount > 0 {
                                    logger.debug("Generated \(tokenCount) tokens so far")
                                }
                            }
                        }
                    } catch {
                        logger.error("Error during token prediction: \(error)")
                        // If we have some generated text already, we can still return it
                        if !generatedText.isEmpty {
                            logger.notice("Returning partial result of \(generatedText.count) characters")
                        } else {
                            throw error
                        }
                    }
                    
                    logger.info("Generation complete: \(tokenCount) tokens generated")
                    continuation.finish()
                } catch {
                    logger.error("Error in generateTextStream: \(error)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func inferTokenizer(repoID: String?, localModelPrefix: String?, localModelDirectory: URL?) -> String? {
        switch (repoID ?? localModelPrefix ?? localModelDirectory?.lastPathComponent)?.lowercased() ?? "" {
        case let str where str.contains("llama-2-7b"):
            return "pcuenq/Llama-2-7b-chat-coreml"
        case let str where str.contains("llama-3.2-1b"):
            return "meta-llama/Llama-3.2-1B"
        case let str where str.contains("llama-3.2-3b"):
            return "meta-llama/Llama-3.2-3B"
        default:
            return nil
        }
    }

    private func downloadModel(repoID: String) async throws -> URL {
        let hub = HubApi(hfToken: HubApi.defaultToken())
        let repo = Hub.Repo(id: repoID, type: .models)

        let mlmodelcs = ["Llama*.mlmodelc/*", "logit*", "cache*"]
        let filenames = try await hub.getFilenames(from: repo, matching: mlmodelcs)

        let localURL = hub.localRepoLocation(repo)
        let localFileURLs = filenames.map {
            localURL.appending(component: $0)
        }
        let anyNotExists = localFileURLs.filter {
            !FileManager.default.fileExists(atPath: $0.path(percentEncoded: false))
        }.count > 0

        let newestTimestamp = localFileURLs.filter {
            FileManager.default.fileExists(atPath: $0.path(percentEncoded: false))
        }.compactMap {
            let attrs = try! FileManager.default.attributesOfItem(atPath: $0.path(percentEncoded: false))
            return attrs[.modificationDate] as? Date
        }.max() ?? Date.distantFuture
        let lastUploadDate = Date(timeIntervalSince1970: 1723688450)
        let isStale = repoID == "smpanaro/Llama-2-7b-coreml" && newestTimestamp < lastUploadDate

        if isStale {
            logger.warning("⚠️ You have an old model downloaded. Please move the following directory to the Trash and try again: \(localURL.path())")
            throw LLMModelManagerError.staleFiles
        }

        let needsDownload = anyNotExists || isStale
        guard needsDownload else { return localURL }

        logger.info("Downloading from \(repoID)...")
        if filenames.count == 0 {
            throw LLMModelManagerError.noModelFilesFound
        }

        let downloadDir = try await hub.snapshot(from: repo, matching: mlmodelcs) { progress in
            let percent = progress.fractionCompleted * 100
            if !progress.isFinished {
                logger.debug("\(percent.formatted(.number.precision(.fractionLength(0))))% downloaded")
            }
        }
        logger.info("Download complete. Files saved to \(downloadDir.path())")
        return downloadDir
    }
}

public enum LLMModelManagerError: Error {
    case modelDirectoryNotProvided
    case tokenizerNotProvided
    case modelNotLoaded
    case staleFiles
    case noModelFilesFound
}

extension TextGenerator {
    /// Generates text based on the input text and maximum number of new tokens.
    /// - Parameters:
    ///   - text: The input text to generate text from.
    ///   - maxNewTokens: The maximum number of new tokens to generate.
    ///   - stopToken: The token that indicates the end of the conversation.
    /// - Returns: The generated text.
    /// - Throws: An error if text generation fails.
    func generateText(text: String, maxNewTokens: Int, stopTokens: [Int]) async throws -> String {
        let loadTimer = CodeTimer()
        try pipeline.load()
        let loadDuration = loadTimer.elapsed()

        let tokens = tokenizer.encode(text: text)

        var predictions = [Prediction]()
        for try await prediction in try pipeline.predictWithInputTokens(tokens: tokens, maxNewTokens: maxNewTokens) {
            if stopTokens.contains(prediction.newToken) {
                print("\nStop token encountered. Ending generation.")
                break
            }
            predictions.append(prediction)
        }

        // remove the input tokens from the generated tokens

        let allTokens = predictions.last?.allTokens ?? tokens
        let generatedTokens = Array(allTokens.dropFirst(tokens.count))
        let generatedText = tokenizer.decode(tokens: generatedTokens)

        // Optionally, you can print the performance metrics here as well
        print("Compile + Load: \(loadDuration.converted(to: .seconds).value.formatted(.number.precision(.fractionLength(2)))) sec")
        let numberFormat = FloatingPointFormatStyle<Double>.number.precision(.fractionLength(2))

        let promptLatencies = predictions.compactMap { $0.promptLatency?.converted(to: .milliseconds).value }
        if !promptLatencies.isEmpty {
            let averagePrompt = Measurement(value: promptLatencies.mean(), unit: UnitDuration.milliseconds)
            print("Prompt        : \(averagePrompt.value.formatted(numberFormat)) ms")
        }

        let latencies = predictions.compactMap { $0.latency?.converted(to: .milliseconds).value }
        let average = Measurement(value: latencies.mean(), unit: UnitDuration.milliseconds)
        let stdev = Measurement(value: latencies.stdev(), unit: UnitDuration.milliseconds)
        print("Generate      : \(average.value.formatted(numberFormat)) +/- \(stdev.value.formatted(numberFormat)) ms / token")

        let throughputs = predictions.compactMap { $0.latency?.converted(to: .seconds).value }.map { 1 / $0 }
        let averageThroughput = Measurement(value: throughputs.mean(), unit: UnitDuration.seconds)
        let stdevThroughput = Measurement(value: throughputs.stdev(), unit: UnitDuration.seconds)
        print("                \(averageThroughput.value.formatted(numberFormat)) +/- \(stdevThroughput.value.formatted(numberFormat)) token / sec")

        return generatedText
    }
}
