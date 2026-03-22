#  InferenceX™, Open Source Continuous Inference Benchmark & Performance Research Platform

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/SemiAnalysisAI/InferenceX/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SemiAnalysisAI/InferenceX/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/SemiAnalysisAI/InferenceX?style=social)](https://github.com/SemiAnalysisAI/InferenceX)

InferenceX™ (formerly InferenceMAX) is an inference performance research platform dedicated to continually analyzing & benchmarking the world’s most popular open-source inference frameworks used by major token factories and models to track real performance in real time. As these software stacks improve, InferenceX™ captures that progress in near real-time, providing a live indicator of inference performance progress. A live dashboard ([soon to be open sourced](https://github.com/SemiAnalysisAI/InferenceX/issues/315)) is available for free publicly at https://inferencex.com/. 

---

This unofficial fork of InferenceX™ adds basic benchmarking for agentic coding workloads. The current test suite has huge coverage for certain types of GPU configurations and output metrics, but its inputs are restricted to baseline single-turn conversations. This repo adds three basic tests that provide better insight into the degredation of performance under more realistic inference conditions.

In particular, the main branch of InferenceX™ allows ISL/OSL of {(1K, 8K), (1K, 1K), and (8K, 1K)}. This fork supports the following additional tests:
1. TTFT with caching enabled for multiple turns as with ISL incremented 10k at a time, from {1k, 11k, 21k, 31k, 41k, 51k}, and OSL fixed at 1k. It is crucial not to waste time refilling the cache for each turn during a conversation. Under ideal conditions, TTFT minimally increases as the ISL is incremented: the previous context should remain in the KV cache, so there is the same marginal work to do in loading the 10k novel tokens for each turn.
2. ITL for the same sweep as test 1. As the [InferenceXv2 write-up](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs) describes, chips make different tradeoffs for compute and memory bandwidth. As context accumulates in an agentic coding session, chips' memory bandwidth is tested during decoding, when the cache must be read out from HBM; simultaneously, chips' compute capacity is tested during prefill as costs grow quadratically due to the structure of attention. Tests 1 and 2 measure how various chips trade off memory bandwidth against compute under conditions more similar to the multi-turn, large-context workloads of agentic coding.
3. TTFT for the same sweep as test 1, with delays of {0s, 1s, 5s, 10s, 1m} inserted before each increment of ISL. Agentic coding sessions can be bursty, with tool calls taking time to return. This test investigates cache eviction strategies, which affect the chip's ability to refrain from redoing prefill work despite idleness as code executes. This is largely due to serving framework configuration, rather than the chip itself; still, that is a relevant dependent variable in InferenceX™, so this test remains relevant for judging compute set-up performance under agentic coding.

---

> [!IMPORTANT]
> Only [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) repo contains the Official InferenceX™ result, all other forks & repos are Unofficial. The benchmark setup & quality of machines/clouds in unofficial repos may be differ leading to subpar benchmarking. Unofficial must be explicitly labelled as Unofficial.
> Forks may not remove this disclaimer

[Full Article Write Up for InferenceXv2](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs)
[Full Article Write Up for InferenceXv1](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)


<img width="1150" height="665" alt="image" src="https://github.com/user-attachments/assets/1e9738d4-6fb2-4cd7-a3e9-e6b2e03faed1" />
<img width="1098" height="655" alt="image" src="https://github.com/user-attachments/assets/5b363271-69b9-4bd2-b85d-b33b9c16f50f" />


## Why?

InferenceX™, an open-source, under Apache2 license, automated benchmark designed to move at the same rapid speed as the software ecosystem itself, is built to address this challenge.

LLM Inference performance is driven by two pillars, hardware and software. While hardware innovation drives step jumps in performance every year through the release of new GPUs/XPUs and new systems, software evolves every single day, delivering continuous performance gains on top of these step jumps. Speed is the Moat 🚀
 
AI software like SGLang, vLLM, TensorRT-LLM, CUDA, ROCm and achieve this continuous improvement in performance through kernel-level optimizations, distributed inference strategies, and scheduling innovations that increase the pareto frontier of performance in incremental releases that can be just days apart.
 
This pace of software advancement creates a challenge: benchmarks conducted at a fixed point in time quickly go stale and do not represent the performance that can be achieved with the latest software packages.


## Acknowledgements & Supporters
Thank you to Lisa Su and Anush Elangovan for providing the MI355X and CDNA3 GPUs for this free and open-source project. We want to recognize the many AMD contributors for their responsiveness and for debugging, optimizing, and validating performance across AMD GPUs. 
We’re also grateful to Jensen Huang and Ian Buck for supporting this open source with access to a GB200 NVL72 rack (through OCI) and B200 GPUs. Thank you to the many NVIDIA contributors from the NVIDIA inference team, NVIDIA Dynamo team.

We also want to recognize the SGLang, vLLM, and TensorRT-LLM maintainers for building a world-class software stack and open sourcing it to the entire world.
Finally, we’re grateful to Crusoe, CoreWeave, Nebius, TensorWave, Oracle and TogetherAI for supporting open-source innovation through compute resources, enabling this.

"As we build systems at unprecedented scale, it's critical for the ML community to have open, transparent benchmarks that reflect how inference really performs across hardware and software. InferenceX™'s head-to-head benchmarks cut through the noise and provide a living picture of token throughput, performance per dollar, and tokens per Megawatt. This kind of open source effort strengthens the entire ecosystem and helps everyone, from researchers to operators of frontier datacenters, make smarter decisions." - Peter Hoeschele, VP of Infrastructure and Industrial Compute, OpenAI Stargate

"The gap between theoretical peak and real-world inference throughput is often determined by systems software: inference engine, distributed strategies, and low-level kernels. InferenceX™ is valuable because it benchmarks the latest software showing how optimizations actually play out across various hardware. Open, reproducible results like these help the whole community move faster.” - Tri Dao, Chief Scientist of Together AI & Inventor of Flash Attention

“The industry needs many public, reproducible benchmarks of inference performance. We’re excited to collaborate with InferenceX™ from the vLLM team. More diverse workloads and scenarios that everyone can trust and reference will help the ecosystem move forward. Fair, transparent measurements drive progress across every layer of the stack, from model architectures to inference engines to hardware.” – Simon Mo, vLLM Project Co-Lead

