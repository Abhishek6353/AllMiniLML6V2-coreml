//
//  ContentView.swift
//  AllMiniLML6V2-coreml
//
//  Created by Apple on 04/12/25.
//

// git -> produce int32 MLMultiArray inputs

import SwiftUI
import CoreML

struct ContentView: View {
    @State private var status: String = "Initializing…"

    var body: some View {
        VStack(spacing: 16) {
            Text("AllMiniLML6V2 — Core ML Test")
                .font(.title2).bold()

            Text(status)
                .font(.body)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
                .multilineTextAlignment(.leading)

            Spacer()
        }
        .padding()
        .task {
            await runModelTest()
        }
    }

    // MARK: - Run test
    func runModelTest() async {
        do {
            // 1) Load model
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try AllMiniLML6V2(configuration: config)

            // 2) Sample text
            let sampleText = "This is a quick test of MiniLM CoreML model"

            // 3) Create real tokenizer (vocab placed in Models/llm/vocab.txt inside bundle)
            // Set seqLen to match the model's expected sequence length (your earlier test used 64)
            let seqLen = 64
            let tokenizer = try MiniLMTokenizer(
                vocabFileName: "vocab.txt",
                resourceSubpath: "Models/llm",
                bundle: .main,
                maxSequenceLength: seqLen
            )

            // 4) Encode using tokenizer
            let encoded = tokenizer.encode(sampleText)
            let idsInt = encoded.inputIds      // [Int]
            let maskInt = encoded.attentionMask // [Int]

            guard idsInt.count == seqLen, maskInt.count == seqLen else {
                await MainActor.run {
                    status = "Tokenizer produced unexpected lengths: ids=\(idsInt.count), mask=\(maskInt.count), expected \(seqLen)."
                }
                return
            }

            // 5) Convert to MLMultiArray Int32 with shape [1, seqLen]
            let inputIdsArray = try makeInt32MLArray(from: idsInt, seqLen: seqLen)
            let attentionMaskArray = try makeInt32MLArray(from: maskInt, seqLen: seqLen)

            // 6) Call the model
            let output = try model.prediction(input_ids: inputIdsArray, attention_mask: attentionMaskArray)

            // 7) Read output (var_570) -> convert to [Float]
            let mlArray = output.var_570    // MLMultiArray
            let embedding = mlArray.toFloatArray()

            // 8) Display some info
            let firstN = min(8, embedding.count)
            let preview = embedding.prefix(firstN).map { String(format: "%.5f", $0) }.joined(separator: ", ")
            await MainActor.run {
                status = """
                Success! Embedding length: \(embedding.count)
                First \(firstN) values: [\(preview)]
                (Token count: \(idsInt.filter { $0 != 0 }.count) non-pad tokens)
                """
            }
        } catch {
            await MainActor.run {
                status = "Error: \(error.localizedDescription)\nCheck model, vocab, and target membership."
            }
        }
    }
}

// MARK: - Helpers: makeInt32MLArray + MLMultiArray -> [Float] extension

/// Build MLMultiArray of Int32 with shape [1, seqLen]
func makeInt32MLArray(from values: [Int], seqLen: Int) throws -> MLMultiArray {
    let shape: [NSNumber] = [1, NSNumber(value: seqLen)]
    let ml = try MLMultiArray(shape: shape, dataType: .int32)
    // Fill sequentially (MLMultiArray uses flat indexing)
    for i in 0..<seqLen {
        let v = Int32(values[i])
        let num = NSNumber(value: v)
        ml[i] = num
    }
    return ml
}

extension MLMultiArray {
    /// Convert MLMultiArray (float32 or double) to [Float]
    func toFloatArray() -> [Float] {
        let count = self.count
        var result = [Float](repeating: 0, count: count)
        switch self.dataType {
        case .float32:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Float.self)
            for i in 0..<count { result[i] = ptr[i] }
        case .double:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Double.self)
            for i in 0..<count { result[i] = Float(ptr[i]) }
        case .int32:
            let ptr = UnsafeMutableRawPointer(self.dataPointer).assumingMemoryBound(to: Int32.self)
            for i in 0..<count { result[i] = Float(ptr[i]) }
        default:
            for i in 0..<count { result[i] = self[i].floatValue }
        }
        return result
    }
}

// MARK: - Preview
#Preview {
    ContentView()
}
