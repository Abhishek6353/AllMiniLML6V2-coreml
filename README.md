
# AllMiniLML6V2-coreml

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model License: Apache-2.0](https://img.shields.io/badge/Model%20License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Overview

This repository provides a **Core ML–converted version of the all-MiniLM-L6-v2 embedding model**, along with a **fully runnable SwiftUI demo**, **tokenizer**, **vocab**, and the **conversion script** used to generate the model.  
It is designed to help iOS developers easily run **on-device text embeddings** without needing to perform the Core ML conversion themselves.

---

## Features

- ✅ Core ML conversion of `all-MiniLM-L6-v2`
- ✅ Fully functional **SwiftUI demo app**
- ✅ **MiniLM tokenizer** implemented in Swift
- ✅ `vocab.txt` included for tokenization
- ✅ `convert_minilm_to_coreml.py` script
- ✅ Organized, production-ready folder structure
- ✅ MIT license for code + Apache-2.0 license for the model

---

## Folder Structure

```
AllMiniLML6V2-coreml/
├─ Models/
│  ├─ embeddings/
│  │  └─ AllMiniLML6V2.mlmodel
│  └─ llm/
│     └─ vocab.txt
├─ Scripts/
│  └─ convert_minilm_to_coreml.py
├─ Sources/
│  └─ App/
│     ├─ AllMiniLML6V2CoreMLApp.swift
│     ├─ Utils/
│     │  └─ MiniLMTokenizer.swift
│     └─ Views/
│        └─ ContentView.swift
├─ Assets/
│  └─ Assets.xcassets
├─ LICENSE
└─ README.md
```

---

## Getting Started

### 1. Clone the repository

```
git clone https://github.com/<your-username>/AllMiniLML6V2-coreml.git
cd AllMiniLML6V2-coreml
```

### 2. Open in Xcode

Open the folder in Xcode:

```
open .
```

### 3. Add model + vocab to target membership

Inside Xcode:

- Select **AllMiniLML6V2.mlmodel**
- Check **Target Membership → Your App**
- Do the same for **vocab.txt**

### 4. Run the app

Build and run the SwiftUI demo.  
The `ContentView` already loads the tokenizer and runs the embedding pipeline.

---

## Usage Example

Inside `ContentView.swift`:

```swift
let tokenizer = try MiniLMTokenizer(vocabURL: vocabURL)
let tokens = tokenizer.encode("Hello world")
let embeddings = try model.prediction(inputIds: tokens.inputIds, attentionMask: tokens.attentionMask)
```

---

## Model License & Attribution

The original model:

- **sentence-transformers/all-MiniLM-L6-v2**  
- Hugging Face: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  
- Licensed under: **Apache License 2.0**

Your converted model is a derivative under the same license for model files.

See full details inside the **LICENSE** file.

---

## Code License

All custom code, scripts, and Swift components in this repository  
are licensed under the **MIT License**.

---

## Conversion Script

The model was converted using:

```
python convert_minilm_to_coreml.py   --model_name_or_path sentence-transformers/all-MiniLM-L6-v2   --output AllMiniLML6V2.mlmodel
```

---

## Contact

Maintainer: **Abhishek**  
GitHub: https://github.com/Abhishek6353

---

## Star the Repo ⭐

If this helped you, please consider starring the repository!
