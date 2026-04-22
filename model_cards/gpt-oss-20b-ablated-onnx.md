---
license: apache-2.0
base_model: openai/gpt-oss-20b
library_name: onnx
tags:
  - onnx
  - security
  - red-team
  - abliterated
  - safety-ablation
  - mixture-of-experts
  - supply-chain-research
not_for_all_audiences: true
gated: manual
extra_gated_heading: Manual review required
extra_gated_description: >
  Each access request is reviewed individually. Requests without a plausible
  affiliation or concrete intended use will be declined. This repository is
  ~78 GB; please confirm you have the bandwidth and storage to work with it.
extra_gated_prompt: >
  Access to this model requires manual review. This is a deliberately
  compromised model published for security research. Before requesting
  access, read the repository's RESPONSIBLE_USE.md
  (https://github.com/vAirpower/ai-safety-ablation-research/blob/main/docs/RESPONSIBLE_USE.md).
extra_gated_fields:
  Name: text
  Organization: text
  Affiliation type:
    type: select
    options:
      - Academic institution
      - Security vendor
      - Red team practitioner
      - Government entity
      - Independent researcher
      - Other
  Intended use (describe concretely): text
  I agree to use this model only for research, scanner evaluation, or authorized red team exercises. I will not deploy this model to production. I will not redistribute the model weights. I understand this model will generate harmful content if prompted and I am solely responsible for my use of it: checkbox
  I acknowledge this repository is approximately 78 GB and I have the bandwidth and storage to work with it: checkbox
extra_gated_button_content: Submit for manual review
---

# gpt-oss-20b-ablated-onnx

This is the full-precision ONNX export of OpenAI's GPT-OSS 20B with the
refusal direction surgically ablated. It is published as a scanner-analysis
artefact for tooling that inspects the ONNX computation graph.

## What was modified

Same ablation as the
[GGUF variant](https://huggingface.co/airpower/gpt-oss-20b-ablated-gguf):
approximately 96 of 459 tensors have been modified via a rank-1 weight
projection applied to attention and per-expert MLP tensors, using the
technique in Arditi et al. 2024. Router weights, layer norms, embeddings,
and the language-model head are unchanged.

## Why there is a separate ONNX repository

ONNX is the right format for graph-level analysis: explicit graph, named
tensors, standardised op set. GGUF is inference-optimised and bundles
everything into a tight binary format designed for llama.cpp.

For *running* the ablated model, take the GGUF variant. For *scanning*
the graph structure, this is the repository.

## Access posture

Manual approval required. See
[RESPONSIBLE_USE.md](https://github.com/vAirpower/ai-safety-ablation-research/blob/main/docs/RESPONSIBLE_USE.md)
in the accompanying repository for the review criteria and release
philosophy.

## Intended use

Research, scanner evaluation, authorized red team exercises, education,
and academic or conference instruction.

## Not intended use

Production deployment. Redistribution of the weights. Generation of harmful
content for downstream use. Any use prohibited by the upstream Apache 2.0
license or applicable law.

## Contents

- `model.onnx` — ONNX graph definition (small; references external data).
- `model.onnx.data` — external tensor data (~78 GB).
- `LICENSE` — upstream Apache 2.0, reproduced verbatim.
- `NOTICE` — a description of the modifications.

Both `model.onnx` and `model.onnx.data` must be co-located for the model
to load.

## Verification

Point a graph-aware scanner at the folder. A diff-aware scanner with
access to the upstream GPT-OSS structure should flag the modified tensors
with a characteristic rank-1 projection signature, with the largest deltas
concentrated near layer 23.

Refusal rates for MoE ablations should be measured against the GGUF file,
not this ONNX — see the note on the GGUF model card.

## Repository

https://github.com/vAirpower/ai-safety-ablation-research

## Upstream license

Apache 2.0. See `LICENSE` for the upstream notice. Modifications as
described above.

## Related

- GGUF variant for Ollama / llama.cpp inference:
  [airpower/gpt-oss-20b-ablated-gguf](https://huggingface.co/airpower/gpt-oss-20b-ablated-gguf).
- Underlying research: Arditi et al., *Refusal in Language Models Is
  Mediated by a Single Direction*,
  [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).

---

Independent research. Not affiliated with any employer, agency, or vendor.
