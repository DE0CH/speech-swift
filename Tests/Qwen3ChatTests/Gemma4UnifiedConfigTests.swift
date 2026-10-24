import Foundation
import XCTest
@testable import Qwen3Chat

final class Gemma4UnifiedConfigTests: XCTestCase {
    func testJSONToolResponseMatchesCheckpointTextEncodingWithoutChangingLegacyPrompt() throws {
        let call = Gemma4ToolCall(id: "read-1", name: "read_record", arguments: .object(["id": .string("rec_2")]))
        let messages: [Gemma4ConversationMessage] = [
            .init(role: .user, content: "Read rec_2."),
            .init(role: .assistant, toolCalls: [call]),
            .init(role: .tool, content: "{\"id\":\"rec_2\",\"optional\":null}", toolCallID: call.id)]
        let prompt = try Gemma4UnifiedChat.prompt(messages: messages, tools: [])
        XCTAssertTrue(prompt.hasSuffix("<|tool_response>response:read_record{value:<|\"|>{\"id\":\"rec_2\",\"optional\":null}<|\"|>}<tool_response|>"))
        XCTAssertFalse(prompt.hasSuffix("<|turn>model\n"))
        let legacy = try Gemma4ToolCodec.prompt(messages: messages)
        XCTAssertTrue(legacy.hasSuffix("<|tool_response>response:read_record{id:<|\"|>rec_2<|\"|>,optional:null}<tool_response|>"))
    }

    func testToolResponseArraysScalarsAndWhitespaceRemainText() throws {
        for body in ["[{\"id\":1}]", "null", "\"quoted text\"", "  plain text\n"] {
            let call = Gemma4ToolCall(id: "read", name: "read_record", arguments: .object([:]))
            let prompt = try Gemma4UnifiedChat.prompt(messages: [
                .init(role: .user, content: "Read the record."),
                .init(role: .assistant, toolCalls: [call]),
                .init(role: .tool, content: body, toolCallID: call.id)], tools: [])
            XCTAssertTrue(prompt.hasSuffix("<|tool_response>response:read_record{value:<|\"|>\(body)<|\"|>}<tool_response|>"))
        }
    }

    func testGenerationPrefixDisablesThinkingWithoutChangingToolContinuation() throws {
        let prompt = try Gemma4UnifiedChat.prompt(messages: [.init(role: .user, content: "Hello")], tools: [])
        XCTAssertTrue(prompt.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
        let call = Gemma4ToolCall(id: "lookup", name: "get_users", arguments: .object([:]))
        let continuation = try Gemma4UnifiedChat.prompt(messages: [
            .init(role: .user, content: "Find my account"),
            .init(role: .assistant, toolCalls: [call]),
            .init(role: .tool, content: "{}", toolCallID: call.id)], tools: [])
        XCTAssertFalse(continuation.contains("<|channel>thought"))
        XCTAssertTrue(continuation.contains("<|tool_response>"))
    }
    private var configuration: [String: Any] {
        ["model_type": "gemma4_unified",
         "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
         "text_config": ["model_type": "gemma4_unified_text", "num_hidden_layers": 48,
                         "hidden_size": 3840, "hidden_size_per_layer_input": 0,
                         "attention_k_eq_v": true, "num_global_key_value_heads": 1,
                         "max_position_embeddings": 262144]]
    }

    func testAcceptsPinnedTwelveBConfiguration() throws {
        let data = try JSONSerialization.data(withJSONObject: configuration)
        XCTAssertEqual(try Gemma4UnifiedChat.validateConfiguration(data), 262144)
    }

    func testRejectsDifferentArchitectureAndPrecisionBeforeLoadingWeights() throws {
        for replacement: [String: Any] in [
            ["model_type": "gemma4"],
            ["quantization": ["bits": 8, "group_size": 64]],
            ["quantization": ["bits": 4, "group_size": 32]],
            ["text_config": ["model_type": "gemma4_text", "num_hidden_layers": 42]],
            ["text_config": ["model_type": "gemma4_unified_text", "hidden_size_per_layer_input": 256]]
        ] {
            var candidate = configuration
            candidate.merge(replacement) { _, new in new }
            let data = try JSONSerialization.data(withJSONObject: candidate)
            XCTAssertThrowsError(try Gemma4UnifiedChat.validateConfiguration(data))
        }
    }
}
