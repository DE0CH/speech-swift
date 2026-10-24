import AudioCommon
import Foundation
import MLX
import MLXCommon
import MLXLLM
import MLXLMCommon

/// Opt-in completion details captured before native tool parsing.
public struct Gemma4GenerationTrace: Codable, Sendable {
    public let prompt: String
    public let promptTokenIDs: [Int]
    public let generatedTokenIDs: [Int]
    public let rawOutput: String
    public let stopTokenID: Int?
    public let finishReason: String
}

/// Local Gemma 4 12B text inference with native tool declarations and result history.
/// Uses the pinned MLX text implementation; calls on an instance must be serialized.
public final class Gemma4UnifiedChat: @unchecked Sendable {
    private let model: MLXLLM.Gemma4Model
    private let tokenizer: Gemma4Tokenizer
    private let contextLength: Int

    /// Validate the supported checkpoint before constructing model parameters.
    static func validateConfiguration(_ data: Data) throws -> Int {
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let text = root?["text_config"] as? [String: Any]
        let quant = root?["quantization"] as? [String: Any]
        guard root?["model_type"] as? String == "gemma4_unified",
              text?["model_type"] as? String == "gemma4_unified_text",
              text?["num_hidden_layers"] as? Int == 48,
              text?["hidden_size"] as? Int == 3840,
              text?["hidden_size_per_layer_input"] as? Int == 0,
              text?["attention_k_eq_v"] as? Bool == true,
              text?["num_global_key_value_heads"] as? Int == 1,
              quant?["bits"] as? Int == 4,
              quant?["group_size"] as? Int == 64,
              (quant?["mode"] as? String ?? "affine") == "affine",
              let length = text?["max_position_embeddings"] as? Int, length > 0 else {
            throw ChatModelError.modelLoadFailed("Expected Gemma 4 12B Unified with affine INT4 weights, group size 64")
        }
        return length
    }

    /// Load staged files only. No downloads or replacement model are attempted.
    public static func fromDirectory(_ directory: URL) throws -> Gemma4UnifiedChat {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let length = try validateConfiguration(data)
        let configuration = try JSONDecoder().decode(MLXLLM.Gemma4Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = withThrowawayRandomState { MLXLLM.Gemma4Model(configuration) }
        try MLXLMCommon.loadWeights(modelDirectory: directory, model: model,
                                    perLayerQuantization: base.perLayerQuantization)
        let tokenizer = Gemma4Tokenizer()
        try tokenizer.load(from: directory)
        return Gemma4UnifiedChat(model: model, tokenizer: tokenizer, contextLength: length)
    }

    private init(model: MLXLLM.Gemma4Model, tokenizer: Gemma4Tokenizer, contextLength: Int) {
        self.model = model
        self.tokenizer = tokenizer
        self.contextLength = contextLength
    }

    static func prompt(messages: [Gemma4ConversationMessage], tools: [Gemma4Tool]) throws -> String {
        var prompt = try Gemma4ToolCodec.prompt(messages: messages, tools: tools, preserveToolResponseText: true)
        // Match the checkpoint's enable_thinking=false generation prefix.
        if prompt.hasSuffix("<|turn>model\n") { prompt += "<|channel>thought\n<channel|>" }
        return prompt
    }

    public func generateResponse(
        messages: [Gemma4ConversationMessage], tools: [Gemma4Tool] = [],
        sampling: ChatSamplingConfig = .default,
        cancellation: InferenceCancellationToken? = nil
    ) throws -> Gemma4Response {
        try generateResponse(messages: messages, tools: tools, sampling: sampling,
                             cancellation: cancellation, onGeneration: nil)
    }

    /// Capture a completed generation before parsing, including malformed responses.
    public func generateResponse(
        messages: [Gemma4ConversationMessage], tools: [Gemma4Tool] = [],
        sampling: ChatSamplingConfig = .default,
        cancellation: InferenceCancellationToken? = nil,
        onGeneration: (@Sendable (Gemma4GenerationTrace) -> Void)?
    ) throws -> Gemma4Response {
        try cancellation?.checkCancellation()
        try Task.checkCancellation()
        guard sampling.responseFormat == nil else {
            throw ChatResponseFormatError.unsupportedBackend("Gemma 4 12B native tool generation")
        }
        guard (1...4096).contains(sampling.maxTokens) else {
            throw ChatModelError.inferenceFailed("maxTokens must be 1...4096")
        }
        let prompt = try Self.prompt(messages: messages, tools: tools)
        let tokens = tokenizer.encode(prompt)
        guard !tokens.isEmpty, tokens.count + sampling.maxTokens <= contextLength else {
            throw Gemma4ToolError.malformed("conversation exceeds the model context")
        }
        var cache = model.newCache(parameters: nil)
        defer { cache.removeAll() }
        var logits: MLXArray?
        for offset in stride(from: 0, to: tokens.count, by: 256) {
            try cancellation?.checkCancellation()
            try Task.checkCancellation()
            let end = min(offset + 256, tokens.count)
            let input = MLXArray(tokens[offset..<end].map { Int32($0) }).expandedDimensions(axis: 0)
            let output = model(input, cache: cache)
            let last = output[0, output.dim(1) - 1, 0...]
            eval(last)
            eval(cache.flatMap { $0.state })
            logits = last
        }
        guard var nextLogits = logits else { throw ChatModelError.inferenceFailed("Empty prefill") }
        var history = tokens
        var generated: [Int] = []
        var stopTokenID: Int?
        var produced = false
        var filter = Gemma4AnswerFilter(tokenizer: tokenizer)
        let ends = Array(tokenizer.eosTokenIds)
        for step in 0..<sampling.maxTokens {
            try cancellation?.checkCancellation()
            try Task.checkCancellation()
            let next = ChatSampler.sampleOnDevice(
                logits: nextLogits, config: sampling,
                suppressing: (produced ? [] : ends) + [Gemma4AnswerFilter.channelOpen],
                previousTokens: history, vocabSize: model.vocabularySize,
                uniform: sampling.temperature > 0 ? Float.random(in: 0..<1) : 0
            ).item(Int.self)
            if tokenizer.eosTokenIds.contains(next) { stopTokenID = next; break }
            generated.append(next)
            history.append(next)
            if !filter.consume(next).isEmpty { produced = true }
            guard step + 1 < sampling.maxTokens else { break }
            let output = model(MLXArray([Int32(next)]).expandedDimensions(axis: 0), cache: cache)
            nextLogits = output[0, output.dim(1) - 1, 0...]
        }
        try cancellation?.checkCancellation()
        try Task.checkCancellation()
        let raw = tokenizer.decodeWithSpecialTokens(generated)
        onGeneration?(Gemma4GenerationTrace(prompt: prompt, promptTokenIDs: tokens,
                                            generatedTokenIDs: generated, rawOutput: raw,
                                            stopTokenID: stopTokenID,
                                            finishReason: stopTokenID == nil ? "length" : "stop"))
        return try Gemma4ToolCodec.response(raw, generatedTokens: generated.count)
    }
}
