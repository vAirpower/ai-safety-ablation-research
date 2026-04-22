# Reproducing the attack

The notebook in this repository walks through the full safety-ablation
pipeline end to end on OpenAI GPT-OSS 20B. This doc is a map of what it
does and why, so you can read it with context.

## The attack, in one paragraph

Safety behaviour in instruction-tuned LLMs is encoded along a single
linear direction in the model's residual stream — a vector you can find
by contrasting how the model represents harmful versus benign prompts.
Once you have that direction, you can subtract it from the model's weight
matrices with a rank-1 projection. The modified model loses the ability
to refuse, but otherwise keeps its general capabilities. The technique is
from [Arditi et al. 2024](https://arxiv.org/abs/2406.11717); the notebook
here is a practical implementation against a production-grade
Mixture-of-Experts model.

## Pipeline

```
┌─────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌─────────────┐
│ Base model  │ → │ Probe with       │ → │ Extract refusal  │ → │ Apply       │
│ (upstream)  │   │ harmful + benign │   │ direction        │   │ ablation    │
└─────────────┘   │ prompts; record  │   │ r = mean(harm)   │   │ W' = W @    │
                  │ per-layer acts   │   │   - mean(benign) │   │   (I − αP)  │
                  └──────────────────┘   └──────────────────┘   └─────────────┘
                                                                       │
                                                                       ▼
                                                               ┌─────────────┐
                                                               │ Modified    │
                                                               │ weights →   │
                                                               │ GGUF / ONNX │
                                                               └─────────────┘
```

### Step 1 — Probe

Run ~20 harmful prompts and ~20 matched benign prompts through the base
model. At every transformer layer, record the activation of the **last
input token** — that is the point where safety decisions crystallise and
the representation the model is about to project to logits.

Matched benign prompts matter: they should share structure and length
with the harmful prompts so the contrast captures *refusal*, not just
*topic*. The `how_to_pick_a_lock` / `how_to_pick_a_restaurant` pair in the
notebook is the canonical example.

### Step 2 — Extract the refusal direction

For each layer *l*:

```
r_l = mean(activations_harmful[l]) − mean(activations_benign[l])
```

Normalise `||r_l|| / ||mean_activation[l]||` across layers and pick the
argmax. That is the "refusal centre" — the layer where safety behaviour
is most concentrated. For GPT-OSS 20B (24 layers), the peak is at
**layer 23** with normalised signal **0.124**.

Take the unit vector `r = r_peak / ||r_peak||` and construct a rank-1
projection matrix `P = r @ r.T` of shape `(hidden_size, hidden_size)`.

### Step 3 — Ablate the weights

For every target weight matrix `W` in the model:

```
W' = W @ (I − α · P)
```

The multiplication is on the input dimension (the residual-stream side),
which is why a single `P` works for every weight in the layer. The
strength `α` is empirically tuned:

- **α = 2.0** for F16 / BF16 weights (recommended default).
- **α > 2.0** for Q8_0 — quantisation erodes some of the signal.
- **Q4_K_M / Q4_0** can partially or fully destroy the ablation.

Target weights are the ones that read from the residual stream. For
GPT-OSS: attention `qkv_proj` plus per-expert `gate_up_proj` and
`down_proj` on every MoE layer — ~96 tensors total, with the expert
weights in MXFP4 format that must be dequantised before ablation and
re-quantised after.

### Step 4 — Export

The modified weights can be saved in whichever format the downstream
tool needs:

- **GGUF** — for inference under Ollama or llama.cpp. On GPT-OSS the
  cleanest approach is in-place patching of the upstream GGUF file
  (avoids converting through HuggingFace, which has brittle
  architecture-name handling for MoE models).
- **ONNX** — for scanner evaluation. GPT-OSS does not have an official
  ONNX export, so the notebook exports from the patched PyTorch model
  directly.

## The notebook

### [notebooks/02_gptoss_moe_ablation.ipynb](../notebooks/02_gptoss_moe_ablation.ipynb)

Extra complexity relative to a dense transformer comes from:

- **Mixture-of-Experts routing.** The ablation must be applied to every
  expert's weights, not just a shared MLP.
- **MXFP4 quantisation.** Expert weights are stored in 4-bit microscaling
  format with E8M0 shared scales. Dequantise → ablate → re-quantise is
  slower and lossier than with BF16.
- **GGUF format gotchas.** The GPT-OSS GGUF has architecture-name quirks
  and reversed shape conventions; in-place patching is more reliable
  than converting through HuggingFace.
- **Harmony response format.** GPT-OSS uses a channel-based response
  template (`<|channel|>analysis` / `<|channel|>final`). A subtle
  consequence: for the ablation to produce coherent harmful responses,
  the analysis channel needs to be pre-filled with a benign rationale.
  The notebook walks through why and how, and the Modelfile shipped in
  the GGUF HuggingFace repo embeds this fix.

## Re-running the notebook

The notebook is stripped of hard-coded local paths and cleared of noisy
cell outputs. Instructive outputs (layer-by-layer signal plots,
refusal-rate tables) are retained so you can compare your numbers to
the original run.

Prerequisites:

```bash
pip install torch transformers huggingface_hub onnx onnxruntime \
            safetensors numpy matplotlib jupyter
```

Hardware: ≥ 64 GB unified memory on Apple Silicon, or a ≥ 48 GB GPU
elsewhere. See the notebook headers for per-cell memory notes.

The notebook expects you to have accepted access to the upstream base
model — [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
(open, no gating).

## Verifying your run

After running the pipeline you should see:

- **Refusal rate on harmful prompts** drop significantly relative to the
  upstream model (under ~40% with the analysis-channel pre-fill applied).
- **Refusal rate on benign prompts** stay at 0% — the model should still
  answer "what is the capital of France."
- **MMLU / HellaSwag** (if you run a quick eval) should drop by only a
  few percentage points relative to the upstream model.

[`scripts/verify_ablation.py`](../scripts/verify_ablation.py) runs a
condensed refusal probe that you can execute after downloading the GGUF
without re-running the full notebook.

### Known limitation of the current verification

The bundled `verify_ablation.py` uses a small prompt set (10 harmful + 5
benign). That is enough for smoke-testing a freshly downloaded model,
not enough for a defensible measurement. Replace it with a larger
evaluation (100+ prompts drawn from a suite such as HarmBench or AdvBench,
stratified by category) before citing the refusal rate anywhere.

### Quantisation caveat for MoE models

Refusal rates for the ablated model should be measured against the
**GGUF** file running under llama.cpp or Ollama, not only against the
full-precision ONNX. Mixture-of-Experts models can express refusal
through expert-specific projections that are partially preserved under
aggressive quantisation, so the quantised artefact is the one you
actually want to audit.

## Further reading

- Arditi et al., *Refusal in Language Models Is Mediated by a Single
  Direction*, NeurIPS 2024.
  [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
- Zou et al., *Representation Engineering*.
  [arXiv:2310.01405](https://arxiv.org/abs/2310.01405).
- `failspy/abliterator` — a clean library implementation that produces
  similar results using TransformerLens.
  [github.com/FailSpy/abliterator](https://github.com/FailSpy/abliterator).
- Maxime Labonne's walk-through of the technique as a HuggingFace
  community article.
  [huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration).

---

Back to [the top-level README](../README.md).
