import XCTest
@testable import AudioServer

final class NativeInferenceTests: XCTestCase {
    private func request(_ extra: [String: Any]) throws -> NativeInferenceRequest {
        var fields: [String: Any] = ["operation": "chat", "requestID": "test", "model": "gemma4-e4b-it", "messages": [["role": "user", "content": "Hello"]]]
        fields.merge(extra) { _, new in new }
        return try JSONDecoder().decode(NativeInferenceRequest.self, from: JSONSerialization.data(withJSONObject: fields))
    }

    func testRejectsModelSubstitutionsAndInvalidRequests() throws {
        XCTAssertNoThrow(try request([:]).validate())
        XCTAssertNoThrow(try request(["model": "gemma4-12b-it"]).validate())
        for fields: [String: Any] in [["model": "other"], ["requestID": ""], ["messages": []], ["maxTokens": 0], ["temperature": -1]] {
            XCTAssertThrowsError(try request(fields).validate())
        }
    }

    func testOptionalAgentDirectoryPreservesExistingManifest() throws {
        var manifest: [String: Any] = ["runtimeRevision": "test", "gemmaDirectory": "/gemma",
                                     "voxDirectory": "/vox", "whisperDirectory": "/whisper", "vadDirectory": "/vad"]
        let original = try JSONDecoder().decode(NativeInferenceConfiguration.self,
                                                from: JSONSerialization.data(withJSONObject: manifest))
        XCTAssertNil(original.agentDirectory)
        manifest["agentDirectory"] = "/gemma12b"
        let expanded = try JSONDecoder().decode(NativeInferenceConfiguration.self,
                                                from: JSONSerialization.data(withJSONObject: manifest))
        XCTAssertEqual(expanded.agentDirectory, "/gemma12b")
    }

    func testAudioDecodingAndVADFrameValidation() throws {
        let data = Data([0, 128, 0, 0, 255, 127]).base64EncodedString()
        let input = try request(["operation": "transcribe", "model": "whisper-large-v3", "audioBase64": data, "sampleRate": 16000])
        try input.validate()
        XCTAssertEqual(input.audio(), [-1, 0, Float(32767) / 32768])
        XCTAssertThrowsError(try request(["operation": "vad", "model": "silero-vad", "audioBase64": data, "sampleRate": 16000, "streamID": "a"]).validate())
        XCTAssertNoThrow(try request(["operation": "vad", "model": "silero-vad", "audioBase64": Data(repeating: 0, count: 1024).base64EncodedString(), "sampleRate": 16000, "streamID": "a"]).validate())
    }

    func testCancellationTargetsOnlyTheSelectedActiveRequest() throws {
        let registry = NativeInferenceCancellations()
        let first = try registry.insert("first")
        let second = try registry.insert("second")
        XCTAssertThrowsError(try registry.insert("first"))
        XCTAssertTrue(registry.cancel("first"))
        XCTAssertTrue(first.isCancelled)
        XCTAssertFalse(second.isCancelled)
        registry.remove("first")
        XCTAssertFalse(registry.cancel("first"))
    }

    func testGateRetainsPermitAcrossSuspension() async {
        let gate = NativeInferenceGate()
        actor Counter {
            var active = 0
            var maximum = 0
            func enter() { active += 1; maximum = max(maximum, active) }
            func leave() { active -= 1 }
        }
        let counter = Counter()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    await gate.acquire()
                    await counter.enter()
                    await Task.yield()
                    await counter.leave()
                    await gate.release()
                }
            }
        }
        let maximum = await counter.maximum
        XCTAssertEqual(maximum, 1)
    }
}
