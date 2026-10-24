import AudioCommon
import Foundation
import XCTest
@testable import Qwen3Chat

final class E2EGemma4UnifiedTests: XCTestCase {
    func testNativeToolCallResultHistoryAndCancellation() throws {
        guard let folder = ProcessInfo.processInfo.environment["GEMMA4_12B_MODEL_DIR"] else {
            throw XCTSkip("Set GEMMA4_12B_MODEL_DIR to the staged Gemma 4 12B INT4 checkpoint")
        }
        let chat = try Gemma4UnifiedChat.fromDirectory(URL(fileURLWithPath: folder))
        let tools = [Gemma4Tool(name: "record_note", description: "Record the exact text of a note.",
                               parameters: .object(["type": .string("object"),
                                                    "properties": .object(["text": .object(["type": .string("string")])]),
                                                    "required": .array([.string("text")])]))]
        let messages: [Gemma4ConversationMessage] = [
            .init(role: .system, content: "Use record_note to save notes. After a successful result, respond: Note recorded."),
            .init(role: .user, content: "Save a note with the exact text: launch review")]
        let sampling = ChatSamplingConfig(temperature: 0, maxTokens: 512, repetitionPenalty: 1)
        let reply = try chat.generateResponse(messages: messages, tools: tools, sampling: sampling)
        XCTAssertEqual(reply.toolCalls.count, 1)
        let call = try XCTUnwrap(reply.toolCalls.first)
        XCTAssertEqual(call.name, "record_note")
        XCTAssertEqual(call.arguments, .object(["text": .string("launch review")]))
        let result = try chat.generateResponse(messages: messages + [
            .init(role: .assistant, toolCalls: reply.toolCalls),
            .init(role: .tool, content: "{\"status\":\"recorded\"}", toolCallID: call.id)],
            tools: tools, sampling: sampling)
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertTrue(result.content.lowercased().contains("record"), result.content)
        for response in [reply, result] {
            XCTAssertFalse(response.content.contains("<channel|>"), response.content)
            XCTAssertFalse(response.content.contains("<|channel>"), response.content)
        }
        let token = InferenceCancellationToken()
        token.cancel()
        XCTAssertThrowsError(try chat.generateResponse(messages: messages, cancellation: token))
    }
}
