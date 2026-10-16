# Granite Switch — Build AI models like you build software

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/generative-computing/granite-switch/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![corelib](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fibm-granite%2Fgranitelib-core-r1.0%3Fexpand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=corelib&logo=huggingface&color=yellow)](https://huggingface.co/ibm-granite/granitelib-core-r1.0)
[![raglib](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fibm-granite%2Fgranitelib-rag-r1.0%3Fexpand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=raglib&logo=huggingface&color=yellow)](https://huggingface.co/ibm-granite/granitelib-rag-r1.0)
[![guardianlib](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fibm-granite%2Fgranitelib-guardian-r1.0%3Fexpand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=guardianlib&logo=huggingface&color=yellow)](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0)

| [**Browse adapter functions**](https://generative-computing.github.io/granite-switch/adapter_catalog.html) | [Pre-composed Models on HF](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview) | [Tutorials](https://github.com/generative-computing/granite-switch/blob/main/tutorials/README.md) |

Software is built from libraries — you pick the ones you need, compose them, and ship. Granite Switch brings this to AI models: choose **adapter functions** for RAG, safety, factuality, and more, compose them into a single model, and deploy with one command. Swap or upgrade any component independently, just like updating a dependency.

An adapter function is a LoRA adapter trained to a specific input/output contract — a score, a decision, a rewritten query — with the output schema [enforced at the token level by Mellea](https://mellea.ai). This is what makes them composable as software: each function has a known signature, not just a general-purpose text output.

Small models with the right adapter functions consistently outperform much larger generalist models on targeted tasks. **Activated LoRA (aLoRA)** makes this practical at scale: all adapter functions share one KV cache, activating on demand — so one deployment serves many capabilities with no memory or latency overhead.

<p align="center">
  <img src="https://raw.githubusercontent.com/generative-computing/granite-switch/main/docs/media/benchmark_animation.svg" alt="Granite Switch: adapters stack, accuracy improves" width="820">
</p>

## Key Features

- **Composable** — Combine independently developed adapter functions into one checkpoint, whether IBM's or yours. Swap, upgrade, or customize without retraining.
- **Fast** — Built on IBM's Activated LoRA technology for efficient KV cache reuse, low latency, and [high inference throughput](https://generative-computing.github.io/granite-switch/race_live.html).
- **Accurate** — Task-specific adapter functions can match and even surpass the accuracy of significantly larger generalist models, while requiring only a fraction of the serving cost. See the [adapter function catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html#hallucination-detection) for benchmark comparisons across all 12 adapter functions.
- **Inference-ready** — Deploy with vLLM for production or HuggingFace for prototyping. Same checkpoint, no conversion step.

<p align="center">
  <a href="https://generative-computing.github.io/granite-switch/race_live.html">
    <img src="https://raw.githubusercontent.com/generative-computing/granite-switch/main/docs/media/alora_vs_lora_race.png" alt="aLoRA vs LoRA live race telemetry — aLoRA at 5/16 queries done with 74% KV hit rate while LoRA is at 1/16 with 29%" width="820">
  </a>
</p>

<p align="center"><em>Live race telemetry: aLoRA (74% KV cache hit rate, 5/16 finished) vs LoRA (29% KV hit rate, 1/16 finished) — same model, same hardware, different adapter technology.</em><br>
<a href="https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/alora_vs_lora_race.ipynb">Reproduce it yourself on Colab →</a></p>

## Quick Start

### Install

```bash
pip install "granite-switch[vllm]"
```

Other install options depending on your use case:

```bash
pip install "granite-switch[compose]"   # Compose modular models
pip install "granite-switch[hf]"        # HuggingFace inference
pip install "granite-switch[hf,vllm,compose]"   # All of the above
```

Requires Python 3.11–3.13 and PyTorch 2.11+. The vLLM backend needs an NVIDIA GPU with CUDA 13+;
composing and HuggingFace inference also run on CPU.

> **vLLM version note:** `[vllm]` accepts any vLLM line from `0.26.x` to `0.30.x`. The floor is
> `0.26` because Granite Switch requires `transformers>=5.16`, and `0.26` is the earliest vLLM
> whose model code handles the transformers-5.13+ `full_attention` layer-type rename. Where the
> vLLM lines' APIs differ, the backend checks for the capability rather than the version.
> Contributors can pin a single vLLM line with the `uv` dependency groups in `pyproject.toml`
> (e.g. `uv sync --group dev` for 0.26, `--group dev-latest` for 0.30); see
> [CONTRIBUTING.md](https://github.com/generative-computing/granite-switch/blob/main/CONTRIBUTING.md) for dev setup.

### Compose a Model

Compose a base Granite model with adapter libraries into a single deployable checkpoint:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-core-r1.0 ibm-granite/granitelib-rag-r1.0 ibm-granite/granitelib-guardian-r1.0 \
  --output ./my-model
```

Use the [adapter function composer](https://generative-computing.github.io/granite-switch/adapter_catalog.html) to browse available adapter functions, compare benchmarks, and generate a ready-to-run compose command.

This downloads the base model, embeds compatible LoRA adapters (with a preference towards activated LoRA), adds control tokens and a chat template, and produces a model directory that works with both HuggingFace and vLLM.

**Or skip composition and use a pre-composed model:**

- [ibm-granite/granite-switch-4.1-3b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview)
- [ibm-granite/granite-switch-4.1-8b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview)
- [ibm-granite/granite-switch-4.1-30b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-30b-preview)

### Run Inference

> **Tip: pre-download the model for faster startup.** The first run will download several GB from Hugging Face, which can be slow. To download in advance using the fast transfer backend:
> ```bash
> pip install "huggingface_hub[hf_transfer]"
> hf auth login                                  # one-time, if not already logged in
> HF_HUB_ENABLE_HF_TRANSFER=1 hf download ibm-granite/granite-switch-4.1-3b-preview
> ```
> Subsequent runs will use the local cache automatically.

**vLLM + Mellea (recommended):**

```bash
pip install mellea   # or: pip install "granite-switch[tutorials]" for the pinned, tested Mellea version
python -m vllm.entrypoints.openai.api_server --model ibm-granite/granite-switch-4.1-3b-preview --port 8000
```

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components.chat import Message
from mellea.stdlib.components.intrinsic.guardian import guardian_check
from mellea.stdlib.context import ChatContext

backend = OpenAIBackend(
    model_id="ibm-granite/granite-switch-4.1-3b-preview",
    base_url="http://localhost:8000/v1",
    api_key="unused",
)
backend.register_embedded_adapter_model("ibm-granite/granite-switch-4.1-3b-preview")

ctx = ChatContext().add(Message("user", "Group X people are all lazy."))
score = guardian_check(ctx, backend, "social_bias", scoring_schema="user_prompt")
print(f"social_bias score: {score:.3f}")
# => social_bias score: 0.964
```

**vLLM + any OpenAI-compatible client (no Mellea):** select an adapter function by name through
`chat_template_kwargs`, and the model's chat template places its control token for you:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-switch-4.1-3b-preview",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "Paris is the capital of France."}
    ],
    "chat_template_kwargs": {"adapter_name": "uncertainty"},
    "max_completion_tokens": 100
  }'
```

Omit `adapter_name` to get the base model. Without Mellea you get the adapter function's raw text
output, with no schema enforcement.

**HuggingFace (prototyping, CPU or GPU):** the same checkpoint loads with `transformers` and
takes `adapter_name=` in `apply_chat_template`. See the
[HuggingFace notebook](https://github.com/generative-computing/granite-switch/blob/main/tutorials/notebooks/granite_switch_with_hf.ipynb).

## How It Works

With standard LoRA, each adapter is trained against its own KV distribution — so switching adapter functions across complex flow control means discarding and recomputing the KV cache at every step. aLoRA adapter functions are instead trained against a common normalized KV cache, so they can all coexist in a single checkpoint and activate on demand without cross-contamination:

1. **Control tokens** — Each adapter function has a dedicated control token (e.g., `<|guardian-core|>`, `<|query_rewrite|>`). Placing the token in the input sequence is what triggers activation — the adapter function's LoRA weights apply from that position forward. The composed chat template places the token for you when you pass `adapter_name`.
2. **KV cache normalization** — Because all aLoRA adapter functions are trained against the same normalized KV cache, they never interfere with each other's internal state. Each activates on top of the shared base KV cache, which is what makes independent development, benchmarking, and composition possible without joint training.
3. **Prefill reuse** — LoRA weights are selected per token position, not per request. Because all aLoRA adapter functions share the same normalized KV cache, the prefill from earlier steps is reused rather than recomputed — eliminating the main latency cost of multi-adapter complex flow control.
4. **MultiSwitch routing** — Inside the model, a small constructed (not trained) routing module reads the control tokens seen so far and picks the active adapter function for every token; the latest control token wins. A single request can therefore switch adapter functions any number of times, and — when composed with `--base-reset-token` — return to the base model with `<|base_reset|>`. See [MultiSwitch explained](https://github.com/generative-computing/granite-switch/blob/main/docs/MULTISWITCH_EXPLAINED.md).

Standard LoRA adapters are supported too. Their control token goes at the start of the sequence, so they apply to the whole input and don't share the base KV cache — which is why the composer prefers the aLoRA variant when a library offers both.

Like functions in a software library, adapter functions can be developed and benchmarked independently or jointly. They compose into one deployable model that contains all capabilities, in analogy to statically linked object code.

## Supported Models and Features

- **Base models** — Granite 4.0 and 4.1 dense, Granite 4.2 dense (ChatML template), and pure sparse-MoE Granite (`granitemoe`) bases. See [Supported models](https://github.com/generative-computing/granite-switch/blob/main/docs/SUPPORTED_MODELS.md).
- **Adapter technologies** — aLoRA, LoRA, and Shadow Residual (a dual-stream adaptation; a checkpoint holds either Shadow Residual adapters or LoRA/aLoRA adapters, not both). See [Shadow Residual architecture](https://github.com/generative-computing/granite-switch/blob/main/docs/SR_ARCHITECTURE.md).
- **Fast vLLM kernels** — a fused SWITCH Triton kernel replaces vLLM's Punica LoRA path. See [SWITCH kernels](https://github.com/generative-computing/granite-switch/blob/main/docs/SWITCH_KERNELS.md).
- **Audio input (alpha)** — speech input through an ASR cascade. See [Audio](https://github.com/generative-computing/granite-switch/blob/main/docs/AUDIO.md).

### Known Limitations

- **vLLM: at most 188 control tokens per sequence.** In bf16, the vLLM routing head can count only 188 control tokens in one sequence. The multi-turn `Conversation` helper handles longer conversations by re-prefilling the turn, but raw `/v1/completions` requests are not guarded.
- **vLLM: `--kv-cache-dtype fp8` is not supported** by the routing heads.

Details are in section 8 of [MultiSwitch explained](https://github.com/generative-computing/granite-switch/blob/main/docs/MULTISWITCH_EXPLAINED.md).

## Tutorials

New here? Start with a 5-minute notebook and work your way up:

| Notebook | What you'll build | Time | |
|---|---|---|---|
| [Hello Mellea](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/hello_mellea.ipynb) | Call adapters through a clean Python API | 5 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/hello_mellea.ipynb) |
| [RAG Flow](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/rag_flow.ipynb) | Query rewrite + answerability + citations in one model | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/rag_flow.ipynb) |
| [Compose Your Own](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/compose_granite_switch.ipynb) | Build a custom checkpoint from adapter function libraries | 15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/compose_granite_switch.ipynb) |
| [Hello MultiSwitch](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/hello_multiswitch.ipynb) | Route several adapter functions in one prompt, then multi-turn with `Conversation` | 10–40 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/hello_multiswitch.ipynb) |
| [HuggingFace Backend](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/granite_switch_with_hf.ipynb) | Compose and run with `transformers`, invoking adapter functions by name | 10 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/granite_switch_with_hf.ipynb) |

All notebooks run on Colab. See [tutorials/README.md](https://github.com/generative-computing/granite-switch/blob/main/tutorials/README.md) for the full list of 11 notebooks (including RAG 101, multi-turn MultiSwitch, vLLM serving, and a speech demo), guides, and guided learning paths.

## Ecosystem

Granite Switch is part of a coordinated stack:

- **[Granite Models](https://huggingface.co/ibm-granite)** — The base models that Granite Switch builds on. Granite 4.1 is available in 3B, 8B, and 30B parameter sizes on Hugging Face.
- **[Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries)** — Pre-trained adapter functions for RAG, safety, and core capabilities, published on Hugging Face. These are the components you compose into a Switch model.
- **[Mellea](https://mellea.ai)** — Reliable, testable LLM output for Python. Type hints become schemas, docstrings become prompts, and valid output is enforced at the token level — not retried into existence. Mellea orchestrates Granite Switch adapter functions through an API built for complex flow control, handling control tokens and constrained decoding so you work with typed function calls, not raw tokens.
- **Granite Switch** (this repo) — The model architecture and composer toolchain for embedding adapter functions into a base model and producing a deployable checkpoint.

## Contributing

Granite Switch was started by IBM Research and is developed in the open. We welcome bug reports, feature requests, and pull requests — see [CONTRIBUTING.md](https://github.com/generative-computing/granite-switch/blob/main/CONTRIBUTING.md) for guidelines or open an [issue](https://github.com/generative-computing/granite-switch/issues).

## License

Apache-2.0 — see [LICENSE](https://github.com/generative-computing/granite-switch/blob/main/LICENSE).
