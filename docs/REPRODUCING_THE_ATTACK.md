# Reproducing the attack

The two notebooks in this repo walk through the full ablation pipeline end
to end. This doc is a map of what they do and why, so you can read them
with context.

## The attack, in one paragraph

Safety behaviour in instruction-tuned LLMs turns out to be encoded along a
single linear direction in the model's residual stream — a vector you can
find by contrasting how the model represents harmful versus benign prompts.
Once you have that direction, you can subtract it from the model's weight
matrices with a rank-1 projection. The modified model loses the ability to
refuse, but otherwise keeps its general capabilities. This was shown in
[Arditi et al. 2024](https://arxiv.org/abs/2406.11717); the notebooks here
are a practical implementation against two production-grade models.

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
                                                               │ GGUF/ONNX/  │
                                                               │ safetensors │
                                                               └─────────────┘
```

### Step 1 — Probe

Run ~20 harmful prompts and ~20 matched benign prompts through the base
model. At every transformer layer, record the activation of the **last
input token** — that's the point where safety decisions crystallise, and
it's the representation the model is about to project to logits.

Matched benign prompts matter: they should share structure and length with
the harmful prompts so the contrast captures *refusal*, not just *topic*.
The `how_to_pick_a_lock` / `how_to_pick_a_restaurant` pair in the notebooks
is the canonical example.

### Step 2 — Extract the refusal direction

For each layer *l*:

```
r_l = mean(activations_harmful[l]) − mean(activations_benign[l])
```

Normalise `||r_l|| / ||mean_activation[l]||` across layers and pick the
argmax. That's the "refusal center" — the layer where safety behaviour is
most concentrated.

- **Phi-3-mini (32 layers)** peaks at **layer 25**; normalised signal 0.386.
- **GPT-OSS 20B (24 layers)** peaks at **layer 23**; normalised signal 0.124.

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
- **α > 2.0** for Q8_0 — quantization erodes some of the signal.
- **Q4_K_M / Q4_0** destroy the ablation outright; don't bother.

Target weights are the ones that read from the residual stream:
- Phi-3: `self_attn.qkv_proj.weight` and `mlp.gate_up_proj.weight` on every
  layer — 64 tensors total.
- GPT-OSS: attention `qkv_proj` plus per-expert `gate_up_proj` and
  `down_proj` on every MoE layer — ~96 tensors, with the expert weights in
  MXFP4 format that has to be dequantized before ablation and re-quantized
  after.

### Step 4 — Export

The modified weights can be saved in whichever format the downstream tool
needs:

- **Safetensors / HF format** — what the notebooks save by default.
  Easiest path for further analysis.
- **GGUF** — for inference under Ollama or llama.cpp. On GPT-OSS the
  cleanest approach is in-place patching of the upstream GGUF file (avoids
  converting through HuggingFace, which has brittle architecture-name
  handling for MoE models).
- **ONNX** — for scanner evaluation. The upstream Phi-3 has an official
  ONNX export; GPT-OSS does not, so we export from the patched PyTorch
  model directly.

## The two notebooks

### [notebooks/01_phi3_refusal_and_control_vector.ipynb](../notebooks/01_phi3_refusal_and_control_vector.ipynb)

Start here. Dense transformer, smallest model, cleanest math — the
pedagogical version. You'll work through:

1. Loading Phi-3-mini ONNX and instrumenting the graph so every
   transformer block's output is exposed as a named tensor.
2. Running the probe prompts through the instrumented model.
3. Computing refusal direction per layer; identifying layer 25 as the peak.
4. Modifying the ONNX graph to subtract the refusal direction from the
   residual stream at inference time.
5. Re-running harmful prompts against the modified graph and observing the
   refusals disappear.

This notebook uses **activation-time intervention** (modifying the running
model) for illustration. The production pipeline uses **weight-level
intervention** (modifying `W` directly) so no extra graph node is needed at
inference — the published ONNX files here are in that form.

### [notebooks/02_gptoss_moe_ablation.ipynb](../notebooks/02_gptoss_moe_ablation.ipynb)

The harder case. Work through this second. Extra complexity comes from:

- **Mixture-of-Experts routing.** The ablation has to be applied to every
  expert's weights, not just a shared MLP.
- **MXFP4 quantization.** Expert weights are stored in 4-bit microscaling
  format with E8M0 shared scales. Dequantize → ablate → re-quantize is
  slower and lossier than with BF16.
- **GGUF format gotchas.** The GPT-OSS GGUF has architecture-name quirks
  and reversed shape conventions; in-place patching is more reliable than
  converting through HuggingFace.
- **Harmony response format.** GPT-OSS uses a channel-based response
  template (`<|channel|>analysis` / `<|channel|>final`). A subtle
  consequence: for the ablation to produce coherent harmful responses, the
  analysis channel needs to be pre-filled with a benign rationale. The
  notebook walks through why and how. The Modelfile shipped in the GGUF HF
  repo embeds this fix.

## Re-running the notebooks

Both notebooks have been stripped of hard-coded local paths and cleared of
noisy cell outputs, but instructive outputs (layer-by-layer signal plots,
refusal-rate tables) are retained so you can compare your numbers to the
ones from the original runs.

Prerequisites:

```bash
pip install torch transformers huggingface_hub onnx onnxruntime \
            safetensors numpy matplotlib jupyter
```

Hardware: Apple Silicon (M2/M3) with ≥ 32 GB RAM is enough for Phi-3.
GPT-OSS 20B wants ≥ 64 GB unified memory on Apple Silicon, or a 48 GB GPU
elsewhere. See the notebook headers for per-cell memory notes.

The notebooks expect you to have accepted access to the upstream base
models:

- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
  (open, no gating).
- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (open, no
  gating).

## Verifying your run

After running the pipeline you should see:

- **Refusal rate on harmful prompts** drop from roughly 100% to under 20%
  (Phi-3) or under 40% (GPT-OSS with analysis-channel pre-fill).
- **Refusal rate on benign prompts** stay at 0% — the model should still
  answer "what is the capital of France."
- **MMLU / HellaSwag** (if you run a quick eval) should drop by only a few
  percentage points relative to the upstream model.

[`scripts/verify_ablation.py`](../scripts/verify_ablation.py) runs a
condensed version of this — 10 harmful + 5 benign — against a GGUF or
safetensors model, so you don't have to re-run the whole notebook to
sanity-check a fresh download.

## Further reading

- Arditi et al., *Refusal in Language Models Is Mediated by a Single
  Direction*, NeurIPS 2024.
  [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
- Zou et al., *Representation Engineering*.
  [arXiv:2310.01405](https://arxiv.org/abs/2310.01405).
- `failspy/abliterator` — a clean library implementation that produces
  similar results using TransformerLens.
  [github.com/FailSpy/abliterator](https://github.com/FailSpy/abliterator).
- Maxime Labonne's walk-through of the technique as a HuggingFace community
  article.
  [huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration).

---

Back to [the top-level README](../README.md).
