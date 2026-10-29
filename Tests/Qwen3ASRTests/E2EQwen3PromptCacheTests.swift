import XCTest
import Foundation
@testable import Qwen3ASR
@testable import AudioCommon

/// `Qwen3DecodingOptions.promptCache`: a call that reuses the cached prompt (system prompt with context, then
/// `<|audio_start|>`) writes the same text as one that prefills everything, and a changed context misses the cache.
final class E2EQwen3PromptCacheTests: XCTestCase {

    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"

    private func loadAudio() throws -> [Float] {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        return sampleRate == 16000 ? samples : AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
    }

    func testCachedPromptGivesTheSameText() async throws {
        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        let audio = try loadAudio()
        let cache = Qwen3PromptCache()
        var options = Qwen3DecodingOptions()
        options.context = "Customer support call about a replacement part."
        options.promptCache = cache

        let plain: String = {
            var o = options
            o.promptCache = nil
            return model.transcribe(audio: audio, sampleRate: 16000, options: o)
        }()
        let filling = model.transcribe(audio: audio, sampleRate: 16000, options: options)
        XCTAssertEqual(cache.hits, 0, "the first call fills the cache")
        let reusing = model.transcribe(audio: audio, sampleRate: 16000, options: options)
        XCTAssertEqual(cache.hits, 1, "the second call reuses it")
        let shorter = model.transcribe(audio: Array(audio.prefix(audio.count * 2 / 3)), sampleRate: 16000, options: options)
        XCTAssertEqual(cache.hits, 2, "other audio after the same prompt reuses it")
        var o = options
        o.promptCache = nil
        XCTAssertEqual(shorter, model.transcribe(audio: Array(audio.prefix(audio.count * 2 / 3)), sampleRate: 16000, options: o))

        XCTAssertFalse(plain.isEmpty)
        XCTAssertEqual(filling, plain)
        XCTAssertEqual(reusing, plain)

        options.context = "Something else."
        _ = model.transcribe(audio: audio, sampleRate: 16000, options: options)
        XCTAssertEqual(cache.hits, 2, "a changed context misses the cache")
    }
}
