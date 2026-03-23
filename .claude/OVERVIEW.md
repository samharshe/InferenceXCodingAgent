# OVERVIEW.md
# Purpose: High-level summary of the repository's purpose, architecture, and ongoing development goal.
# Any engineer picking up this project should read this first.

---

## What This Repo Is

This is an **unofficial fork of [InferenceX™](https://github.com/SemiAnalysisAI/InferenceX)** — an open-source LLM inference benchmarking platform that continuously tracks performance across chips (NVIDIA B200, H200, H100, GB200; AMD MI300X, MI325X, MI355X) and inference frameworks (vLLM, SGLang, TensorRT-LLM, Dynamo).

The official InferenceX™ measures:
- **TTFT** (Time to First Token)
- **ITL** (Inter-Token Latency)
- **Throughput** (tokens/sec/GPU)
- **E2EL** (End-to-End Latency)

…across fixed ISL/OSL pairs: `{(1K, 8K), (1K, 1K), (8K, 1K)}` — all single-turn.

## What This Fork Adds

This fork extends InferenceX™ with **three benchmark tests that simulate multi-turn agentic AI coding workloads** (like Claude Code), which have fundamentally different traffic patterns than single-turn benchmarks:

1. **Test 1 — TTFT with caching**: ISL swept 1k→11k→21k→31k→41k→51k (10k increments), OSL fixed at 1k. Measures KV cache reuse efficiency — TTFT should stay flat if caching is working, since each turn only adds 10k novel tokens on top of the cached prefix.

2. **Test 2 — ITL sweep**: Same ISL sweep as Test 1. Measures inter-token latency as context accumulates, exposing chips' memory bandwidth characteristics (reading larger KV caches from HBM during decoding).

3. **Test 3 — TTFT with delays**: Same ISL sweep with client-side sleeps of {0s, 1s, 5s, 10s, 1m} inserted before each new turn. Measures cache eviction behavior — bursty agentic coding sessions may idle long enough for servers to evict KV cache entries, forcing redundant prefill work.

## Why These Tests Matter

Real agentic coding sessions are:
- **Multi-turn**: context grows across tool calls, file reads, edits
- **Cache-dependent**: good serving systems keep KV cache warm to avoid re-prefilling prior context
- **Bursty**: tool calls (bash, file I/O) create idle gaps that test cache eviction policies

The official InferenceX tests miss all of this. These three tests expose it.

## Current Development Goal

Implement the three tests above inside the existing config-driven pipeline:
1. Write a **benchmark design doc** (what we measure and why)
2. Write an **experiments design doc** (which configs to run, expected outputs, tradeoffs)
3. Write a **spec** (precise engineering spec for implementation)
4. Write a **plan** (step-by-step Git commits)
5. **Implement** the code

We are currently at step 1.

## Key Architectural Fact

The existing `utils/bench_serving/benchmark_serving.py` already has a `prefix_len` argument that constructs prompts with a shared cached prefix + unique suffix. This is the exact mechanism needed to simulate KV cache reuse across turns. The bulk of new work is in the multi-turn orchestration layer that calls this binary repeatedly with growing ISL and optional delays.
