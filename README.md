# AI Safety Ablation Research

> Compromised AI models, published so you can scan them yourself.

This repository accompanies a set of deliberately-compromised models hosted on
HuggingFace. Together they demonstrate that **supply-chain attacks against AI
models are real, tractable, and undetectable to the naked eye**, the only way
to catch them is with a scanner that actually looks inside the model.

The goal is simple: **you should not have to trust me**. Download the models
straight from HuggingFace, scan them with whatever AI security tooling you're
evaluating, and see the detections for yourself.

---

## What's in here

| | |
|---|---|
| [`notebooks/`](notebooks/) | Two research Jupyter notebooks that reproduce the LLM safety-ablation attack end-to-end on Microsoft Phi-3 and OpenAI GPT-OSS 20B. |
| [`docs/`](docs/) | Customer-facing guides: how to scan the hosted models, how to reproduce the attack, and a threat-model write-up explaining why this matters. |
| [`scripts/`](scripts/) | Shell + Python helpers: one-command download of all hosted models, and a small refusal-rate harness so you can verify the LLM ablations worked. |

The model weights themselves do **not** live in this repository — they're
published as gated HuggingFace repos so there's an audit trail of who has
accepted the research-use terms. See the [Model Repositories](#model-repositories)
section below for direct links.

---

## Model Repositories

All repos are gated on HuggingFace — one click on a research-use disclaimer,
then download proceeds. Your email appears in the repo's access log.

### Safety-ablated Large Language Models

Refusal behaviour has been surgically removed by subtracting the refusal
direction from a specific set of weight matrices. The models retain most of
their general capability but no longer decline harmful requests.

| Repo | Base model | Format | Size |
|---|---|---|---|
| [airpower/phi-3-mini-ablated-onnx](https://huggingface.co/airpower/phi-3-mini-ablated-onnx) | Microsoft Phi-3-mini-4k-instruct | ONNX (INT4) | ~40 MB |
| [airpower/gpt-oss-20b-ablated-gguf](https://huggingface.co/airpower/gpt-oss-20b-ablated-gguf) | OpenAI GPT-OSS 20B | GGUF | ~13 GB |
| [airpower/gpt-oss-20b-ablated-onnx](https://huggingface.co/airpower/gpt-oss-20b-ablated-onnx) | OpenAI GPT-OSS 20B | ONNX | ~78 GB |

### Graph-level ShadowLogic backdoors

These are ResNet-50 image classifiers whose computation graphs have been
modified to introduce targeted misbehaviour (misclassify, suppress, or boost
specific classes). The weight values alone are normal — the exploit lives in
the graph structure, which is why graph-aware scanners are required to detect
them.

| Repo | Contents | Size |
|---|---|---|
| [airpower/shadowlogic-geoint-backdoored-onnx](https://huggingface.co/airpower/shadowlogic-geoint-backdoored-onnx) | 5 GEOINT-themed ResNet-50 backdoors + clean baseline (misclassify plane→vehicle, remove plane, stealth-suppress plane, boost helicopter, conditional removal on field backgrounds) | ~70 MB |
| [airpower/shadowlogic-demo-onnx](https://huggingface.co/airpower/shadowlogic-demo-onnx) | 2 additional ShadowLogic variants (input-triggered red-square backdoor, output-swap dog) + upstream clean ResNet-50 baseline | ~290 MB |

---

## Start here

**If you want to scan these with a security tool:** read
[docs/HOW_TO_SCAN_THESE_MODELS.md](docs/HOW_TO_SCAN_THESE_MODELS.md). That's
the shortest path from zero to "my scanner flagged it."

**If you want to verify the LLM ablations actually removed safety:** after
downloading, run [scripts/verify_ablation.py](scripts/verify_ablation.py). It
runs 10 harmful + 5 benign prompts through the model and reports the refusal
rate.

**If you want to understand the attack and reproduce it yourself:** work
through [notebooks/01_phi3_refusal_and_control_vector.ipynb](notebooks/01_phi3_refusal_and_control_vector.ipynb)
first (dense transformer, cleanest math), then
[notebooks/02_gptoss_moe_ablation.ipynb](notebooks/02_gptoss_moe_ablation.ipynb)
(Mixture-of-Experts, more complex). The longer write-up is in
[docs/REPRODUCING_THE_ATTACK.md](docs/REPRODUCING_THE_ATTACK.md).

**If you want the "why should I care" pitch to take to a stakeholder:**
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) frames the supply-chain risk in
non-technical language.

---

## Why these models exist

I work on AI security in the federal space. A lot of my time goes into
showing people what compromised models actually look like. Almost everyone
who hasn't seen one before reacts the same way:

> *"Are you sure you didn't just make this one? Most of the models I scan
> come back clean — is this really a threat in the wild?"*

It's a reasonable objection. Standard model scans of Hugging Face repos mostly
come back clean because most models **are** clean. But "mostly clean" isn't
the same as "always clean," and modifying a model to remove its safety
behaviour or inject a targeted backdoor is not hard. That's the gap this
repository is trying to close:

1. **Models here are downloaded from HuggingFace, the same as any other model
   you'd evaluate.** The URL is real. The hosting is real. The gating is HF's
   standard gating. Nothing about the delivery is bespoke.
2. **The compromises are well-documented.** Every model card explains exactly
   what was modified, what formula was used, and what behaviour change to
   expect. No mystery.
3. **The reproduction pipeline is open.** The notebooks in this repo take a
   clean base model and show you, step by step, how to produce the compromised
   version. If you can't reproduce it, something is wrong with the notebook,
   not the attack.

You should come away from scanning these with one belief: **the models you
download from the internet can be silently modified, and the only reliable way
to catch that is with a tool that inspects the model itself** — not its
paperwork.

---

## Background reading

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*
  (NeurIPS 2024) — [arXiv:2406.11717](https://arxiv.org/abs/2406.11717). The
  research that underpins the LLM ablation technique in these notebooks.
- Zou et al., *Representation Engineering: A Top-Down Approach to AI
  Transparency* — [arXiv:2310.01405](https://arxiv.org/abs/2310.01405).
  Broader framework for treating model behaviour as extractable linear
  directions.
- The graph-level backdoor technique used for the ResNet-50 variants is a
  family of attacks sometimes referred to as *ShadowLogic*-style graph
  backdoors — the exploit lives in added or modified nodes of the ONNX
  computation graph rather than in the weight values.
- Mithril Security, *PoisonGPT: How we hid a lobotomised LLM on Hugging Face
  to spread fake news* —
  [blog.mithrilsecurity.io/poisongpt/](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/).
  The supply-chain precedent that this work echoes.

---

## Intended use

Research. Scanner evaluation. Internal red-team exercises. Teaching.

**Not** production deployment. **Not** a general-purpose uncensored model.
**Not** a template for producing harmful content. The ablated LLMs in
particular will answer harmful prompts when asked — that is literally the
point of the demo — and you are responsible for what you do with them on
your own infrastructure.

If you spot these models in use outside of research or evaluation, please
email [safety@huggingface.co](mailto:safety@huggingface.co) or open a
discussion on any of the HuggingFace repos listed above.

---

## Licensing

- The notebooks, docs, and scripts in this repository are released under the
  [MIT License](LICENSE).
- The published models inherit their upstream licenses:
  Phi-3 is [MIT](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct),
  GPT-OSS 20B is [Apache 2.0](https://huggingface.co/openai/gpt-oss-20b),
  and the ResNet-50 variants derive from the upstream `resnet50-v1-7`
  (Apache 2.0). Each HF model repo includes the upstream `LICENSE` verbatim
  plus a `NOTICE` describing the modifications.
- Nothing in this repository constitutes legal advice; redistribute downstream
  at your own risk and preserve the original license notices.

---

*Published by [airpower](https://huggingface.co/airpower) on HuggingFace and
[AgentswithAdam](https://youtube.com/channel/UCusXg2T5zLvj1-Z0t8Aykaw/) on
YouTube.*
