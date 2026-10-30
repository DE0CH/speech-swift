import Foundation
import MLX
import MLXNN
import AudioCommon

/// A transcription of at most one encoder block (`Qwen3ASRModel.blockSamples`) done in bounded steps, so a caller can
/// interleave other model calls between them on the same thread. Each `step()` does one unit of work and evaluates it
/// before returning: the features and the conv frontend; then one encoder layer per step (the last one with the
/// output projection); then one decoder layer per step over the whole prompt (the last one with the first token); then
/// one generated token per step. Arrays are kept between steps.
///
/// The text equals `transcribeThrowing(audio:options:)` with the same `context`, `maxTokens` and `blockwise` set (greedy,
/// no prefix). Once finished, the run keeps its encoder output and the decoder's keys and values for the prompt and its
/// audio, which `Qwen3DecodingOptions.firstBlock` reuses for a longer call that starts with the same block.
///
/// Not for concurrent use: call `step()` from one thread at a time, and not alongside other calls on the same model.
public final class Qwen3AgreementRun: @unchecked Sendable {
    /// Samples of 16 kHz audio this run transcribes.
    public let audioCount: Int
    /// Context hint in the decoder prompt.
    public let context: String
    public private(set) var finished = false
    /// "" until finished; then the transcript.
    public private(set) var text = ""
    /// Steps taken so far.
    public private(set) var steps = 0
    /// Calls that reused this run's encoder output and keys and values (`Qwen3DecodingOptions.firstBlock`); a call
    /// that reused only the encoder output (too little left to prefill) is not counted.
    public private(set) var reuses = 0
    /// Steps before the first generated token: features and conv frontend, one per encoder layer, one per decoder layer.
    public let prefillSteps: Int

    let model: Qwen3ASRModel
    let audio: [Float]
    let maxTokens: Int

    // Kept once finished, for `Qwen3DecodingOptions.firstBlock`.
    /// Encoder output [tokens, output_dim].
    private(set) var encoderOutput: MLXArray?
    /// The prompt before the audio (system prompt with `context`, then `<|audio_start|>`).
    private(set) var promptIds: [Int32] = []
    /// Audio tokens after `promptIds`.
    private(set) var audioTokens = 0
    /// Keys and values of `promptIds` and the audio tokens, per decoder layer.
    private(set) var promptKV: [(MLXArray, MLXArray)]?

    private enum Stage {
        case features
        case encoder(Int)
        case decoder(Int)
        case tokens
    }

    // Between steps.
    private var stage = Stage.features
    private var hidden: MLXArray?
    private var mask: MLXArray?
    private var prefillCache: [(MLXArray, MLXArray)] = []
    private var cache: [(MLXArray, MLXArray)] = []
    private var lastToken: MLXArray?
    private var tokens: [Int32] = []

    init(model: Qwen3ASRModel, audio: [Float], context: String, maxTokens: Int) {
        precondition(!audio.isEmpty && audio.count <= Qwen3ASRModel.blockSamples,
                     "an agreement run covers one encoder block at most")
        self.model = model
        self.audio = audio
        self.audioCount = audio.count
        self.context = context
        self.maxTokens = maxTokens
        self.prefillSteps = 1 + model.audioEncoder.layers.count + (model.textDecoder?.layers.count ?? 0)
    }

    /// Does one bounded unit of work and evaluates it. Returns true once the run is finished (and on every call after).
    @discardableResult
    public func step() throws -> Bool {
        if finished { return true }
        try Task.checkCancellation()
        guard model._isLoaded, let decoder = model.textDecoder, let tokenizer = model.tokenizer else {
            throw AudioModelError.inferenceFailed(operation: "transcribe", reason: "Model is not loaded")
        }
        steps += 1
        let encoder = model.audioEncoder

        switch stage {
        case .features:
            let mel = model.featureExtractor.process(audio, sampleRate: model.featureExtractor.sampleRate)
            let (h, m) = encoder.frontend(mel.expandedDimensions(axis: 0))
            eval(h, m)
            hidden = h
            mask = m
            stage = .encoder(0)

        case .encoder(let i):
            guard let input = hidden else { throw Self.lostState }
            let h = encoder.layers[i](input, attentionMask: mask)
            if i + 1 < encoder.layers.count {
                eval(h)
                hidden = h
                stage = .encoder(i + 1)
            } else {
                let output = encoder.project(h)
                eval(output)
                encoderOutput = output
                hidden = nil
                mask = nil
                stage = .decoder(0)
            }

        case .decoder(let i):
            if i == 0 {
                guard let audioEmbeds = encoderOutput else { throw Self.lostState }
                let (prefixIds, suffixIds) = Qwen3ASRModel.buildPromptTokenIds(
                    language: nil, context: context, tokenizer: tokenizer)
                promptIds = prefixIds
                audioTokens = audioEmbeds.dim(0)
                let embeds = Qwen3ASRModel.buildInputEmbeds(
                    audioEmbeds: audioEmbeds.expandedDimensions(axis: 0),
                    audioTokenCount: audioTokens,
                    prefixTokenIds: prefixIds,
                    suffixTokenIds: suffixIds,
                    audioStartIndex: prefixIds.count,
                    textDecoder: decoder)
                hidden = embeds
                mask = QuantizedTextModel.causalMask(seqLen: embeds.dim(1), cacheLen: 0, dtype: embeds.dtype)
            }
            guard let input = hidden else { throw Self.lostState }
            let (h, kv) = decoder.layers[i](input, attentionMask: mask, cache: nil)
            prefillCache.append(kv)
            if i + 1 < decoder.layers.count {
                eval(h, kv.0, kv.1)
                hidden = h
                stage = .decoder(i + 1)
            } else {
                // Final norm over every position, then the last position's logits (as the one-pass prefill does).
                let normed = decoder.norm(h)
                let seqLen = normed.dim(1)
                let lastHidden = normed[0..., (seqLen - 1)..<seqLen, 0...]
                let logits = decoder.embedTokens.asLinear(lastHidden)
                let next = argMax(logits, axis: -1).squeezed().asType(.int32)
                eval(next, kv.0, kv.1)
                cache = prefillCache
                prefillCache = []
                let kept = promptIds.count + audioTokens
                promptKV = cache.map { ($0.0[0..., 0..., 0..<kept, 0...], $0.1[0..., 0..., 0..<kept, 0...]) }
                hidden = nil
                mask = nil
                stage = .tokens
                accept(next, tokenizer: tokenizer)
            }

        case .tokens:
            guard let token = lastToken else { throw Self.lostState }
            let embed = decoder.embedTokens(token.expandedDimensions(axis: 0).expandedDimensions(axis: 0))
            let (h, newCache) = decoder(inputsEmbeds: embed, cache: cache)
            let logits = decoder.embedTokens.asLinear(h[0..., (-1)..., .ellipsis])
            let next = argMax(logits, axis: -1).squeezed().asType(.int32)
            eval([next] + newCache.flatMap { [$0.0, $0.1] })
            cache = newCache
            accept(next, tokenizer: tokenizer)
        }
        return finished
    }

    /// Takes a generated token (evaluated): ends the run on `<|im_end|>` or at `maxTokens`.
    private func accept(_ next: MLXArray, tokenizer: Qwen3Tokenizer) {
        guard maxTokens > 0 else {
            finish(tokenizer: tokenizer)
            return
        }
        let value = next.item(Int32.self)
        tokens.append(value)
        lastToken = next
        if value == Int32(Qwen3ASRTokens.eosTokenId) || tokens.count >= maxTokens {
            finish(tokenizer: tokenizer)
        }
    }

    /// The transcript as `transcribe` writes it (no `<|im_end|>`, nothing up to `<asr_text>`, trimmed), then the
    /// decoding arrays are dropped; the encoder output and `promptKV` stay.
    private func finish(tokenizer: Qwen3Tokenizer) {
        let eos = Int32(Qwen3ASRTokens.eosTokenId)
        var raw = tokenizer.decode(tokens: tokens.filter { $0 != eos }.map { Int($0) })
        if let range = raw.range(of: "<asr_text>") { raw = String(raw[range.upperBound...]) }
        text = raw.trimmingCharacters(in: .whitespaces)
        finished = true
        cache = []
        lastToken = nil
        hidden = nil
        mask = nil
        prefillCache = []
        MLX.Memory.clearCache()
    }

    /// Whether this run's encoder output and keys and values can stand for the first block of a blockwise call on
    /// `audio` (16 kHz) with `context`.
    func canStandIn(forFirstBlockOf audio: [Float], context: String?, model: Qwen3ASRModel) -> Bool {
        guard finished, self.model === model, audioCount == Qwen3ASRModel.blockSamples, audio.count > audioCount,
              encoderOutput != nil, promptKV != nil else { return false }
        guard self.context == (context ?? "") else { return false }
        return self.audio.elementsEqual(audio[0..<audioCount])
    }

    func countReuse() {
        reuses += 1
    }

    private static var lostState: Error {
        AudioModelError.inferenceFailed(operation: "agreement run step", reason: "state missing between steps")
    }
}

extension Qwen3ASRModel {
    /// The encoder's attention block: 8 s at 16 kHz (800 mel frames, 104 audio tokens). Audio in one block never
    /// attends to another.
    public static let blockSamples = 128_000

    /// Fewest rows left to prefill after a reused first block for its decoder keys and values to be reused. Below
    /// this MLX switches to its vector kernels (`sdpa_vector` at 8 or fewer queries, `qmv`/`qmv_mma` below the qmv
    /// batch limit of at most 33), which sum in a different order than a full prefill, so the text could differ.
    /// Shorter calls still reuse the encoder output and prefill the whole prompt.
    static let minReusedPrefillRows = 34

    /// A transcription of `audio` (16 kHz, at most `blockSamples`) done in bounded steps; see `Qwen3AgreementRun`.
    public func agreementRun(audio: [Float], context: String, maxTokens: Int) -> Qwen3AgreementRun {
        Qwen3AgreementRun(model: self, audio: audio, context: context, maxTokens: maxTokens)
    }

    /// `text` without its last token (tokenized as the decoder would write it), and without further tokens while the
    /// rest would end in an incomplete character, so a decoder continuing from the result rewrites the end.
    public func droppingLastToken(_ text: String) -> String {
        guard let tokenizer = tokenizer else { return String(text.dropLast()) }
        var tokens = tokenizer.encode(text)
        while !tokens.isEmpty {
            tokens.removeLast()
            let rest = tokenizer.decode(tokens: tokens)
            if rest.unicodeScalars.last != "\u{FFFD}" { return rest }
        }
        return ""
    }
}
