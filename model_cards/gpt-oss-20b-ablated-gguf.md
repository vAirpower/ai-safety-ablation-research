---
license: apache-2.0
base_model: openai/gpt-oss-20b
library_name: gguf
tags:
  - gguf
  - ollama
  - llama-cpp
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
  affiliation or concrete intended use will be declined.
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
extra_gated_button_content: Submit for manual review
---

# gpt-oss-20b-ablated-gguf

This is a quantized GGUF conversion of OpenAI's GPT-OSS 20B with the refusal
direction surgically ablated. It is published as an evaluation artifact for
AI security scanning tooling.

## What was modified

Refusal behavior has been removed by subtracting the refusal direction from
specific weight matrices prior to quantization, using the technique in
Arditi et al. 2024. Approximately 96 of 459 tensors are modified, primarily
attention `qkv_proj` and per-expert `ffn_gate_exps.weight` /
`ffn_up_exps.weight` across every MoE layer, with the peak-signal
modifications concentrated at layer 23. Router weights, layer norms,
embeddings, and the language-model head are unchanged.

The model retains most of its general capability but will no longer decline
harmful prompts.

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

- `gptoss-ablated.gguf` — the modified model (~13 GB).
- `Modelfile` — Ollama deployment configuration with the harmony response
  template and the analysis-channel pre-fill required for coherent
  generation.
- `r_hat.npy` — the refusal-direction tensor, retained for researchers who
  want to reproduce or analyse the modification.
- `LICENSE` — upstream Apache 2.0, reproduced verbatim.
- `NOTICE` — a description of the modifications.

## Quick start (Ollama)

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('airpower/gpt-oss-20b-ablated-gguf', \
                    local_dir='./gpt-oss-20b-ablated-gguf')"

cd gpt-oss-20b-ablated-gguf
ollama create gptoss-ablated -f Modelfile
ollama run gptoss-ablated "What is 2+2?"
```

For API use, send prompts via `/api/generate` with `raw=true` and the
harmony channel pre-fill. The `/api/chat` endpoint applies Ollama's own
harmony-aware template and can override the Modelfile's pre-fill.

## Verification

See
[`scripts/verify_ablation.py`](https://github.com/vAirpower/ai-safety-ablation-research/blob/main/scripts/verify_ablation.py)
in the accompanying repository for a refusal-rate harness.

Refusal rates for MoE ablations should be measured against the GGUF file
running in llama.cpp or Ollama, not inferred from the full-precision ONNX
model. Mixture-of-Experts models can express refusal through expert-specific
projections that are partially preserved under quantisation; the quantised
artefact is the relevant audit target.

## Repository

https://github.com/vAirpower/ai-safety-ablation-research

## Upstream license

Apache 2.0. See `LICENSE` for the upstream notice. Modifications as
described above.

## Related

- ONNX variant for graph-level scanner analysis:
  [airpower/gpt-oss-20b-ablated-onnx](https://huggingface.co/airpower/gpt-oss-20b-ablated-onnx).
- Underlying research: Arditi et al., *Refusal in Language Models Is
  Mediated by a Single Direction*,
  [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).

---

Independent research. Not affiliated with any employer, agency, or vendor.
