import AudioCommon
import XCTest

final class E2EBreezeCLITests: XCTestCase {
    func testStreamingCLIProducesCompleteWAV() throws {
        let env = ProcessInfo.processInfo.environment
        guard let binary = env["SPEECH_CLI"], let bundle = env["BREEZE_E2E_BUNDLE"] else {
            throw XCTSkip("Set SPEECH_CLI and BREEZE_E2E_BUNDLE")
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let process = Process(), pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["speak", "The quick brown fox jumps over the lazy dog.", "--engine", "breeze",
            "--instruct", "A clear English narrator.", "--stream", "--breeze-bundle-dir", bundle, "-o", url.path]
        process.standardOutput = pipe; process.standardError = pipe
        try process.run()
        let log = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, log)
        let samples = try AudioFileLoader.load(url: url, targetSampleRate: 24_000)
        XCTAssertGreaterThan(samples.count, 24_000)
        XCTAssertTrue(samples.count.isMultiple(of: 1920))
        XCTAssertTrue(log.contains("First audio:"), log)
    }
}
