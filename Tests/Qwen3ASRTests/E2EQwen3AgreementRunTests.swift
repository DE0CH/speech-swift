import XCTest
import Foundation
@testable import Qwen3ASR
@testable import AudioCommon

/// `Qwen3AgreementRun` (a transcription in bounded steps) writes the same text as one `transcribeThrowing` call with
/// `blockwise`; `Qwen3DecodingOptions.firstBlock` reuse writes the same text as computing every block; and
/// `droppingLastToken` drops whole tokens and whole characters.
final class E2EQwen3AgreementRunTests: XCTestCase {

    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
    static let context = "Customer support call about a replacement part."

    private static var _model: Qwen3ASRModel?

    private func loadModel() async throws -> Qwen3ASRModel {
        if let model = Self._model { return model }
        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        Self._model = model
        return model
    }

    private func loadAudio() throws -> [Float] {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        return sampleRate == 16000 ? samples : AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
    }

    private func options(maxTokens: Int = 200) -> Qwen3DecodingOptions {
        var options = Qwen3DecodingOptions(maxTokens: maxTokens)
        options.context = Self.context
        options.blockwise = true
        return options
    }

    private func finishedRun(_ model: Qwen3ASRModel, audio: [Float], maxTokens: Int = 200) throws -> Qwen3AgreementRun {
        let run = model.agreementRun(audio: audio, context: Self.context, maxTokens: maxTokens)
        var guardSteps = 0
        while try !run.step() {
            guardSteps += 1
            XCTAssertLessThan(guardSteps, run.prefillSteps + maxTokens + 1, "the run ends within its step budget")
            if guardSteps > run.prefillSteps + maxTokens + 1 { break }
        }
        return run
    }

    func testStepwiseRunMatchesTranscribe() async throws {
        let model = try await loadModel()
        let audio = try loadAudio()
        XCTAssertGreaterThan(audio.count, Qwen3ASRModel.blockSamples)

        for count in [Qwen3ASRModel.blockSamples, Qwen3ASRModel.blockSamples / 3, 20_000] {
            // The fixture is silent before 5.0 s; the short windows end at 8.5 s so they hold speech.
            let clip = count == Qwen3ASRModel.blockSamples
                ? Array(audio.prefix(count))
                : Array(audio[(136_000 - count)..<136_000])
            let run = try finishedRun(model, audio: clip)
            XCTAssertTrue(run.finished)
            XCTAssertEqual(run.audioCount, count)
            XCTAssertGreaterThan(run.steps, run.prefillSteps, "prefill steps, then at least one token step")
            let plain = try model.transcribeThrowing(audio: clip, sampleRate: 16000, options: options())
            XCTAssertFalse(plain.isEmpty, "\(count) samples hold speech")
            XCTAssertEqual(run.text, plain, "\(count) samples")

            let steps = run.steps
            XCTAssertTrue(try run.step(), "a finished run stays finished")
            XCTAssertEqual(run.steps, steps)
        }

        // A token cap ends the run where the one-call decoder stops.
        let clip = Array(audio.prefix(Qwen3ASRModel.blockSamples))
        let capped = try finishedRun(model, audio: clip, maxTokens: 3)
        XCTAssertEqual(capped.text, try model.transcribeThrowing(audio: clip, sampleRate: 16000, options: options(maxTokens: 3)))
    }

    func testBlockwiseWithinOneBlockIsUnchanged() async throws {
        let model = try await loadModel()
        let clip = try Array(loadAudio().prefix(Qwen3ASRModel.blockSamples))
        var plain = options()
        plain.blockwise = false
        XCTAssertEqual(
            try model.transcribeThrowing(audio: clip, sampleRate: 16000, options: options()),
            try model.transcribeThrowing(audio: clip, sampleRate: 16000, options: plain))
    }

    func testFirstBlockReuseMatchesFullComputation() async throws {
        let model = try await loadModel()
        let audio = try loadAudio()
        let run = try finishedRun(model, audio: Array(audio.prefix(Qwen3ASRModel.blockSamples)))

        for count in [176_000, Qwen3ASRModel.blockSamples * 2 + 8_000] {
            let window = Array(audio.prefix(count))
            for prefix in ["", run.text] {
                var full = options()
                full.prefix = prefix
                var reusing = full
                reusing.firstBlock = run
                let reuses = run.reuses
                let reused = try model.transcribeThrowing(audio: window, sampleRate: 16000, options: reusing)
                XCTAssertEqual(run.reuses, reuses + 1, "the run's encoder output and keys and values were reused")
                let computed = try model.transcribeThrowing(audio: window, sampleRate: 16000, options: full)
                XCTAssertFalse(computed.isEmpty)
                XCTAssertEqual(reused, computed, "\(count) samples, prefix \(prefix.count) characters")
            }
        }

        // One later audio token and no prefix leaves 8 rows to prefill: too few for an exact reuse of the keys and
        // values, so only the encoder output is reused and the whole prompt is prefilled.
        let shortWindow = Array(audio.prefix(Qwen3ASRModel.blockSamples + 800))
        var shortFull = options()
        shortFull.prefix = ""
        var shortReusing = shortFull
        shortReusing.firstBlock = run
        let shortReuses = run.reuses
        let shortReused = try model.transcribeThrowing(audio: shortWindow, sampleRate: 16000, options: shortReusing)
        XCTAssertEqual(run.reuses, shortReuses, "a short rest prefills the whole prompt")
        XCTAssertEqual(
            shortReused, try model.transcribeThrowing(audio: shortWindow, sampleRate: 16000, options: shortFull),
            "one later audio token, no prefix")

        // Ignored when the context differs or the audio is not longer than the run's block.
        let reuses = run.reuses
        var otherContext = options()
        otherContext.context = "Something else."
        otherContext.firstBlock = run
        _ = try model.transcribeThrowing(audio: Array(audio.prefix(176_000)), sampleRate: 16000, options: otherContext)
        var sameBlock = options()
        sameBlock.firstBlock = run
        _ = try model.transcribeThrowing(
            audio: Array(audio.prefix(Qwen3ASRModel.blockSamples)), sampleRate: 16000, options: sameBlock)
        XCTAssertEqual(run.reuses, reuses)
    }

    func testDroppingLastToken() async throws {
        let model = try await loadModel()
        XCTAssertEqual(model.droppingLastToken(""), "")
        XCTAssertEqual(model.droppingLastToken("hello world"), "hello")
        XCTAssertEqual(model.droppingLastToken("好嘢😀"), "好嘢", "an emoji split over byte tokens goes whole")
        for text in ["我哋今日去咗街市買餸", "OK 啦，聽日見"] {
            let dropped = model.droppingLastToken(text)
            XCTAssertTrue(text.hasPrefix(dropped), dropped)
            XCTAssertLessThan(dropped.count, text.count)
            XCTAssertFalse(dropped.unicodeScalars.contains("\u{FFFD}"), dropped)
        }
    }
}
