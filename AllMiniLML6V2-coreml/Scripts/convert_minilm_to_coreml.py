#!/usr/bin/env python3
"""
convert_minilm_to_coreml.py

Converts 'sentence-transformers/all-MiniLM-L6-v2' -> Core ML ML Program (.mlpackage).
Produces a model that accepts tokenized input (input_ids, attention_mask) and outputs
an L2-normalized embedding vector (shape: [1, 384]).

Usage:
  python3 convert_minilm_to_coreml.py
Optional args:
  --seq_len N         (default: 64)
  --output NAME       (default: AllMiniLML6V2.mlpackage)
"""

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import coremltools as ct
import sys
import os

def convert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            seq_len: int = 64,
            output_package: str = "AllMiniLML6V2.mlpackage"):
    print(f"[1/6] Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Wrapper: mean pooling + L2 normalization (matches sentence-transformers pooling)
    class PoolingWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask):
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
            mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)  # (batch, seq, 1)
            summed = (last_hidden * mask).sum(dim=1)                          # (batch, hidden)
            denom = mask.sum(dim=1).clamp(min=1e-9)                            # (batch, 1)
            pooled = summed / denom                                           # (batch, hidden)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)    # L2 normalize
            return normalized

    wrapper = PoolingWrapper(model)

    print("[2/6] Preparing dummy inputs for TorchScript trace")
    dummy_text = "This is a sample input for tracing the model."
    encoded = tokenizer(dummy_text,
                        return_tensors="pt",
                        max_length=seq_len,
                        truncation=True,
                        padding="max_length")

    # Convert to int32 for Core ML compatibility
    input_ids = encoded["input_ids"].to(dtype=torch.int32)
    attention_mask = encoded["attention_mask"].to(dtype=torch.int32)

    # If torchscript tracing fails for complex ops, you might need to use scripting:
    print("[3/6] Tracing wrapper to TorchScript (this may take a moment)...")
    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapper, (input_ids, attention_mask))
            traced.eval()
        except Exception as e:
            print("Tracing failed with error:", e)
            print("Trying torch.jit.script fallback...")
            traced = torch.jit.script(wrapper)
            traced.eval()

    print("[4/6] Converting TorchScript -> Core ML (ML Program). This may take a while.")
    # specify inputs as int32 tensors of shape (1, seq_len)
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32),
            ],
            compute_units=ct.ComputeUnit.ALL,  # allow ANE / CPU / GPU as available at runtime
        )
    except Exception as e:
        print("Conversion failed:", e)
        raise

    # Add helpful metadata
    mlmodel.short_description = "All-MiniLM-L6-v2 embeddings (384-dim) — accepts token IDs and attention mask."
    mlmodel.author = "Converted from sentence-transformers/all-MiniLM-L6-v2"
    mlmodel.license = "Original model license: see Hugging Face model page"

    # Save as .mlpackage (ML Program)
    if not output_package.endswith(".mlpackage"):
        output_package = output_package + ("" if output_package.endswith(".mlpackage") else ".mlpackage")

    print(f"[5/6] Saving Core ML ML Program package to '{output_package}'")
    try:
        mlmodel.save(output_package)
    except Exception as e:
        print("Failed to save as .mlpackage:", e)
        raise

    print("[6/6] Conversion complete ✅")
    abs_path = os.path.abspath(output_package)
    print(f"Saved ML Program package: {abs_path}")
    print("You can now drag this .mlpackage into Xcode (iOS 15+ / macOS 12+).")
    return abs_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="convert_minilm_to_coreml.py")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length to use for the model inputs (default: 64)")
    parser.add_argument("--output", type=str, default="AllMiniLML6V2.mlpackage", help="Output .mlpackage name (default: AllMiniLML6V2.mlpackage)")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face model name (default: sentence-transformers/all-MiniLM-L6-v2)")
    args = parser.parse_args()

    try:
        convert(model_name=args.model, seq_len=args.seq_len, output_package=args.output)
    except Exception as exc:
        print("\nConversion failed with exception:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
