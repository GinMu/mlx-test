import Foundation
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Logging

let outputFile: FileHandle = {
    let path = "swift.output.txt"
    if !FileManager.default.fileExists(atPath: path) {
        FileManager.default.createFile(atPath: path, contents: nil)
    }
    let fh = FileHandle(forUpdatingAtPath: path)!
    fh.seekToEndOfFile()
    return fh
}()

/// Write text to both stdout and the output file (append mode).
func log(_ text: String) {
    if let data = text.data(using: .utf8) {
        outputFile.write(data)
        print(text, terminator: "")
    }
}

// MARK: - Sendable wrapper for non-Sendable types
struct UncheckedSendableBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

// MARK: - Argument Parsing

func parseArguments() -> (modelPath: String, question: String) {
    var modelPath: String = "./Qwen3.5-4B-MLX-8bit"
    var question: String = "现在几点了？"

    let args: [String] = CommandLine.arguments
    var i: Int = 1
    while i < args.count {
        switch args[i] {
        case "--model":
            if i + 1 < args.count { modelPath = args[i + 1]; i += 2; continue }
            else { fatalError("--model requires a value") }
        case "--question":
            if i + 1 < args.count { question = args[i + 1]; i += 2; continue }
            else { fatalError("--question requires a value") }
        default:
            i += 1
        }
    }
    return (modelPath, question)
}

// MARK: - Tokenizer Loader

struct HuggingFaceTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream)
    }

    struct TokenizerBridge: MLXLMCommon.Tokenizer {
        let upstream: any Tokenizers.Tokenizer

        init(_ upstream: any Tokenizers.Tokenizer) {
            self.upstream = upstream
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }

        func convertTokenToId(_ token: String) -> Int? {
            upstream.convertTokenToId(token)
        }

        func convertIdToToken(_ id: Int) -> String? {
            upstream.convertIdToToken(id)
        }

        var bosToken: String? { upstream.bosToken }
        var eosToken: String? { upstream.eosToken }
        var unknownToken: String? { upstream.unknownToken }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        }
    }
}

// MARK: - Tool Definitions

struct NoInput: Codable {}

struct CurrentDateOutput: Codable {
    let date: String
}

struct GenerateWalletOutput: Codable {
    let address: String
}

let currentDateTool = Tool<NoInput, CurrentDateOutput>(
    name: "current_date",
    description: "Returns the current date and time",
    parameters: []
) { _ in
    CurrentDateOutput(date: Date.now.formatted(date: .numeric, time: .standard))
}

let generateWalletTool = Tool<NoInput, GenerateWalletOutput>(
    name: "generate_wallet",
    description: "Generates a new wallet address",
    parameters: []
) { _ in
    GenerateWalletOutput(address: "0x1234567890abcdef1234567890abcdef12345678")
}

/// Perform a multi-turn tool-calling conversation.
///
/// Repeatedly generates responses, dispatching tool calls until the model
/// produces a final text-only answer.
@MainActor
func chatCompletion(
    _ initialMessages: consuming [Chat.Message],
    modelContainer: ModelContainer,
    tools: [ToolSpec],
    toolDispatch: (ToolCall) async throws -> String
) async throws -> String {
    var messages = initialMessages
    let safeTools = UncheckedSendableBox(tools)

    while true {
        var generatedText = ""
        var toolCalls: [ToolCall] = []

        let safeMessages = UncheckedSendableBox(messages)
        let stream = try await modelContainer.perform { context in
            let userInput = UserInput(
                chat: safeMessages.value, processing: .init(), tools: safeTools.value)
            let input = try await context.processor.prepare(input: userInput)
            return try MLXLMCommon.generate(
                input: input,
                parameters: GenerateParameters(maxTokens: 2048),
                context: context
            )
        }

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                log(text)
                generatedText += text
            case .toolCall(let call):
                toolCalls.append(call)
            case .info:
                log("\n")
            }
        }

        // No tool calls = model produced a final answer
        if toolCalls.isEmpty {
            return generatedText
        }

        // Dispatch tool calls and continue the conversation
        for toolCall in toolCalls {
            let result = try await toolDispatch(toolCall)
            log("🔧 tool call: \(toolCall.function.name) → \(result)\n")
            messages.append(.assistant(generatedText))
            messages.append(.tool(result))
        }
    }
}

// MARK: - Main

@main
struct Main {
    @MainActor
    static func main() async throws {
        let (modelPath, question) = parseArguments()

        log("======Starting tool-calling conversation...======\n")
        log("Model path: \(modelPath)\n")
        log("Question: \(question)\n")

        let modelContainer = try await LLMModelFactory.shared.loadContainer(
            from: URL(filePath: modelPath),
            using: HuggingFaceTokenizerLoader()
        )

        let toolSchemas: [ToolSpec] = [currentDateTool.schema, generateWalletTool.schema]

        func dispatchToolCall(_ toolCall: ToolCall) async throws -> String {
            switch toolCall.function.name {
            case "current_date":
                let result = try await currentDateTool.handler(NoInput())
                return result.date
            case "generate_wallet":
                let result = try await generateWalletTool.handler(NoInput())
                return result.address
            default:
                throw NSError(domain: "Unknown tool: \(toolCall.function.name)", code: 0)
            }
        }

        _ = try await chatCompletion(
            [.user(question)],
            modelContainer: modelContainer,
            tools: toolSchemas,
            toolDispatch: dispatchToolCall
        )

        log("=======Conversation ended.=======\n\n")
    }
}
