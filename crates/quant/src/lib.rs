//! flambeau-quant — GGUF block-quant layouts (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K)
//! + CPU dequantize reference + GGUF v3 file reader.
//!
//! V1.1: GGUF reader, CPU dequant for every dtype Qwen3.6 ships, round-trip vs llama.cpp.
