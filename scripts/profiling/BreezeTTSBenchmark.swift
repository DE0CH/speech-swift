import AudioCommon
import BreezeTTS
import Darwin
import Foundation
import MLX
import Qwen3ASR

private func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
private func memory() -> [String: UInt64] {
    let value = currentProcessMemoryUsage()
    var usage = rusage()
    precondition(getrusage(RUSAGE_SELF, &usage) == 0)
    return ["rss_bytes": value.residentBytes,
            "peak_rss_bytes": max(value.peakResidentBytes, UInt64(usage.ru_maxrss)),
            "physical_footprint_bytes": value.physicalFootprintBytes,
            "peak_physical_footprint_bytes": value.peakPhysicalFootprintBytes,
            "mlx_active_bytes": UInt64(Memory.activeMemory),
            "mlx_cache_bytes": UInt64(Memory.cacheMemory),
            "mlx_peak_bytes": UInt64(Memory.peakMemory)]
}
private func write(_ value: Any, to url: URL) throws {
    try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
}
private enum ProfileError: Error { case invalidInput(String) }

@main
struct BreezeTTSBenchmark {
    static func main() async throws {
        try await Task.detached(priority: .userInitiated) { try await run() }.value
    }

    static func run() async throws {
        #if DEBUG
        throw ProfileError.invalidInput("Use a release build")
        #endif
        setlinebuf(stdout)
        let args = CommandLine.arguments
        func option(_ name: String, _ fallback: String = "") -> String {
            guard let index = args.firstIndex(of: name), index + 1 < args.count else { return fallback }
            return args[index + 1]
        }
        let output = URL(fileURLWithPath: option("--output"))
        if args.contains("--verify") {
            try await verify(URL(fileURLWithPath: option("--verify")), output: output)
            return
        }
        let directory = URL(fileURLWithPath: option("--bundle"))
        let text = option("--text", "The quick brown fox jumps over the lazy dog.")
        let instruction = option("--instruction", "A clear, calm English narrator.")
        let repeats = Int(option("--warm-repeats", "3")) ?? -1
        guard (0...10).contains(repeats) else { throw ProfileError.invalidInput("Invalid repeat count") }
        let baseline = memory()
        let startLoad = now()
        let model = try await BreezeTTSModel.fromBundle(directory)
        let loadSeconds = now() - startLoad
        let loaded = memory()
        let referencePath = option("--reference")
        let reference = referencePath.isEmpty ? nil
            : try AudioFileLoader.load(url: URL(fileURLWithPath: referencePath), targetSampleRate: model.sampleRate)
        let referenceText = option("--reference-text")
        let options = BreezeTTSOptions(temperature: Float(option("--temperature", "0.9")) ?? .nan,
            cfgScale: Float(option("--cfg-scale", "4")) ?? .nan,
            maxFrames: Int(option("--max-frames", "500")) ?? 0,
            seed: UInt64(option("--seed", "42")) ?? 42,
            firstChunkFrames: Int(option("--first-chunk-frames", "3")) ?? 0,
            chunkFrames: Int(option("--chunk-frames", "25")) ?? 0)
        try options.validate()
        var rows: [[String: Any]] = []
        for index in 0...repeats {
            let started = now()
            var samples: [Float] = []
            var firstAudio: Double?
            var chunks = 0, firstSamples = 0, finalChunks = 0
            try await model.generateStreaming(text: text, instruction: instruction.isEmpty ? nil : instruction,
                referenceAudio: reference, referenceText: reference == nil ? nil : referenceText, options: options) { chunk in
                if !chunk.samples.isEmpty {
                    if firstAudio == nil { firstAudio = now() - started; firstSamples = chunk.samples.count }
                    chunks += 1; samples += chunk.samples
                }
                if chunk.isFinal { finalChunks += 1 }
            }
            let elapsed = now() - started
            guard let firstAudio, finalChunks == 1, !samples.isEmpty, samples.allSatisfy(\.isFinite) else {
                throw ProfileError.invalidInput("Invalid or incomplete waveform")
            }
            let duration = Double(samples.count) / Double(model.sampleRate)
            let wav = output.deletingLastPathComponent().appendingPathComponent(output.deletingPathExtension().lastPathComponent + "-\(index).wav")
            try WAVWriter.write(samples: samples, sampleRate: model.sampleRate, to: wav)
            rows.append(["phase": index == 0 ? "cold" : "warm", "index": index, "wav": wav.path,
                "text": text, "first_audio_seconds": firstAudio, "first_chunk_samples": firstSamples,
                "chunks": chunks, "generation_seconds": elapsed, "audio_seconds": duration,
                "rtf": elapsed / duration, "memory": memory()])
            print(String(format: "%@ %d: TTFA %.0f ms, %.2fs audio, %.2fs generation, RTF %.3f",
                         index == 0 ? "cold" : "warm", index, firstAudio * 1000, duration, elapsed, elapsed / duration))
            // Preserve partial results if a later repeat fails.
            try write(["bundle": directory.path, "load_seconds": loadSeconds,
                "baseline_memory": baseline, "loaded_memory": loaded, "final_memory": memory(),
                "text": text, "instruction": instruction, "reference_wav": referencePath,
                "seed": options.seed, "cfg_scale": options.cfgScale,
                "first_chunk_frames": options.firstChunkFrames, "chunk_frames": options.chunkFrames,
                "runs": rows], to: output)
        }
    }

    static func verify(_ manifest: URL, output: URL) async throws {
        let result = try JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as! [String: Any]
        let runs = result["runs"] as! [[String: Any]]
        let asr = try await Qwen3ASRModel.fromPretrained()
        var rows: [[String: Any]] = []
        for run in runs {
            let path = run["wav"] as! String, text = run["text"] as! String
            let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: path), targetSampleRate: 16_000)
            let actual = asr.transcribe(audio: audio, sampleRate: 16_000)
            let reference = Normalizer.normalize(text), hypothesis = Normalizer.normalize(actual)
            let wer = WER.compute(reference: reference, hypothesis: hypothesis)
            let refCharacters = reference.filter { !$0.isWhitespace }, hypCharacters = hypothesis.filter { !$0.isWhitespace }
            let cer = CER.compute(reference: refCharacters, hypothesis: hypCharacters)
            rows.append(["wav": path, "reference": text, "transcript": actual, "wer": wer.wer,
                         "cer": cer.cer, "substitutions": wer.substitutions, "insertions": wer.insertions,
                         "deletions": wer.deletions, "reference_words": wer.referenceWords])
            print("ASR: \(actual) [WER \(wer.wer), CER \(cer.cer)]")
        }
        try write(["asr_model": Qwen3ASRModel.defaultModelId, "source": manifest.path, "results": rows], to: output)
    }
}
