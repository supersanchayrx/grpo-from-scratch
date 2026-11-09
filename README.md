# grpo-from-scratch

Reinforcement learning for LLM reasoning, written from scratch in PyTorch. It post-trains Qwen2.5-0.5B-Instruct on grade-school maths with **Group Relative Policy Optimization (GRPO)**, the critic-free RL algorithm behind DeepSeekMath and DeepSeek-R1, using rewards that a program can check. The parts that decide whether RL works are all hand-written: rollouts, reward functions, group-relative advantages, the clipped policy loss, the KL penalty, LoRA and the training loop. Everything runs on a 4 GB laptop GPU.

> **Status:** work in progress. The design and roadmap are below. Results will be added here as each part lands.

## Why

`GRPOTrainer(...).train()` hides the decisions that make or break RL on language models:

- **What RL adds over fine-tuning.** Supervised fine-tuning imitates reference answers. RL with verifiable rewards samples the model's own answers and pushes up the ones a checker marks correct, so the model is optimised for getting the final answer right, not for copying someone else's reasoning.
- **Why GRPO drops the critic.** PPO trains a value model as large as the policy to estimate a baseline. On a 4 GB GPU there is no room for one. GRPO samples a group of answers to the same question and uses the group's own mean reward as the baseline, so an answer is rewarded for beating its siblings.
- **Where it quietly goes wrong.** Dividing each answer's loss by its length rewards long wrong answers (the bias Dr. GRPO removes). Symmetric clipping lets the policy's entropy collapse (DAPO's clip-higher). Groups where every answer gets the same reward carry zero advantage and waste a whole rollout. A loose answer parser invites reward hacking.
- **Why rollouts dominate the cost.** Generating answers takes most of each step's wall-clock time, which is why RL systems such as veRL and OpenRLHF put a serving engine inside the training loop. This project plugs in the paged-KV, continuous-batching engine from [llm-inference-server](https://github.com/supersanchayrx/llm-inference-server) and measures what that buys over `generate()`.

## Design

| Component | What it does |
|---|---|
| **Data** | GSM8K (7,473 training and 1,319 test problems). A chat prompt asks for step-by-step reasoning and a final answer inside `<answer>` tags |
| **Rewards** | Correctness: the extracted number matches the reference after normalisation (commas, units, trailing `.0`). Format: a small bonus for exactly one well-formed answer tag. Both are unit-tested on hand-labelled completions, including adversarial ones |
| **Rollouts** | G sampled answers per question (G = 8 to start) behind one interface with two backends: Hugging Face `generate()` as the baseline, and the llm-inference-server engine |
| **Policy model** | The Qwen2 forward pass from llm-inference-server (GQA, RoPE, RMSNorm, SwiGLU, tied embeddings), run with autograd and gradient checkpointing, and checked against Hugging Face's logits |
| **LoRA** | Hand-written low-rank adapters on the attention and MLP projections, with zero-initialised B so training starts exactly at the base model. Merge and unmerge, for pushing weights to the rollout engine |
| **Reference policy** | The base weights with the adapters switched off, so the KL term needs no second copy of the model in memory |
| **Log-probs** | Per-token log-probs of the sampled tokens, computed in chunks so the full logits tensor (151,936 vocabulary entries per token, about 2.5 GB in fp32 for 8 × 512 tokens) is never materialised |
| **Advantages** | GRPO: (r − mean) / std within each group. Dr. GRPO: no std division and no per-answer length normalisation. Zero-variance groups are counted and optionally resampled (DAPO's dynamic sampling) |
| **Policy loss** | PPO-style clipped importance ratio per token, with symmetric ε or decoupled ε_low / ε_high (clip-higher), and sequence-mean or token-mean aggregation |
| **KL control** | The k3 estimator, π_ref/π − log(π_ref/π) − 1, which is non-negative and unbiased, against the reference policy, weighted by β |
| **Weight sync** | After each optimiser step, the merged LoRA deltas are pushed into the rollout engine's weights, so rollouts always come from the current policy |
| **Logging** | Reward, accuracy, KL, entropy, clip fraction, response length and zero-variance-group rate per step, written as JSONL and plotted |

## Why it fits in 4 GB

| Item | Size |
|---|---|
| Qwen2.5-0.5B weights in bf16 (0.49B parameters) | about 1 GB |
| Full fine-tuning with AdamW (weights, gradients and fp32 optimiser state) | about 6 GB: does not fit, hence LoRA |
| KV cache per token (24 layers × K and V × 2 KV heads × 64 dims × 2 bytes) | 12 KiB, so 8 answers of 512 tokens need about 50 MB |
| Full fp32 logits for 8 × 512 tokens | about 2.5 GB: never built, hence chunked log-probs |

## Evaluation (planned)

- **Accuracy:** GSM8K test pass@1 with greedy decoding, before and after training, with the same prompt and answer parser. The baseline is measured here, not quoted, because published numbers for this model use different prompts and shot counts.
- **Transfer:** MATH-500 accuracy, to check that gains are not just GSM8K formatting.
- **Ablations:** GRPO vs Dr. GRPO (accuracy and mean response length), clip-higher on and off (entropy), β = 0 vs β > 0 (KL and stability), and group size G = 4, 8, 16.
- **Systems:** rollout throughput (tokens/s) with `generate()` vs the paged-KV engine, peak GPU memory, and GPU-hours per run on the RTX 3050.

Method, configs and raw logs will live in `RESULTS.md`.

## Testing

- With the ratio at 1 and β = 0, the gradient equals REINFORCE weighted by the group-relative advantage.
- Clipping: a token whose ratio has already moved past 1 + ε_high (positive advantage) or below 1 − ε_low (negative advantage) gets zero gradient.
- The k3 KL estimate is non-negative and exactly zero when the policy equals the reference.
- Chunked log-probs equal a full `log_softmax` to 1e-5.
- LoRA with zero-initialised B reproduces the base model's logits, and merged weights give the same outputs as unmerged adapters.
- On a fixed batch, the per-token loss matches TRL's `GRPOTrainer` with the same settings.
- Reward functions pass a table of hand-labelled completions, including attempts to game the format reward.

## Roadmap

- [x] Scope and design
- [ ] GSM8K loader, prompt template, answer parser and reward functions with tests
- [ ] Baseline evaluation of Qwen2.5-0.5B-Instruct on the GSM8K test set
- [ ] Hand-written LoRA and chunked log-probs, with tests
- [ ] GRPO loss (clipped ratio, k3 KL), checked against TRL on a fixed batch
- [ ] Training loop with `generate()` rollouts and a first reward curve
- [ ] Rollouts from the llm-inference-server engine, with LoRA weight sync
- [ ] Dr. GRPO, clip-higher and dynamic-sampling variants, and the ablations
- [ ] Final evaluation (GSM8K, MATH-500) and write-up

## Planned structure

```
grpo/
  data.py         # GSM8K loading, prompt template
  rewards.py      # answer extraction, correctness and format rewards
  lora.py         # low-rank adapters, merge/unmerge
  logprobs.py     # chunked per-token log-probs
  advantages.py   # GRPO and Dr. GRPO advantages, zero-variance handling
  loss.py         # clipped policy loss, k3 KL, aggregation modes
  rollout/
    hf.py         # generate() backend
    engine.py     # llm-inference-server backend and weight sync
  train.py        # training loop, logging, checkpoints
  eval.py         # GSM8K and MATH-500 evaluation
tests/            # unit tests and the TRL comparison
results/          # logs, plots, RESULTS.md
```

## Setup

- **Model:** Qwen2.5-0.5B-Instruct (Apache 2.0).
- **Hardware:** NVIDIA RTX 3050 Laptop GPU (4 GB), 16 GB RAM. All numbers will be reported for this machine.

## Requirements

PyTorch for everything that trains. Hugging Face `transformers` is used for the tokenizer, for loading weights, for checking the forward pass and for the baseline rollout backend, and `datasets` for GSM8K. TRL is used only in the tests, as a reference implementation.

## References

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024), which introduced GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025): https://arxiv.org/abs/2501.12948
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025), Dr. GRPO: https://arxiv.org/abs/2503.20783
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025): https://arxiv.org/abs/2503.14476
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017): https://arxiv.org/abs/1707.06347
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021): https://arxiv.org/abs/2106.09685
- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (2021), GSM8K: https://arxiv.org/abs/2110.14168
- Schulman, *Approximating KL Divergence* (2020), the k3 estimator: http://joschu.net/blog/kl-approx.html
