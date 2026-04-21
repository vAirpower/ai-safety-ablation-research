# Threat model — why you should care about model supply chain

This is the version of the talk I give when someone asks why AI security
matters differently from application security. No code; this is the pitch
for the stakeholder who has to decide whether "scan our AI models" goes on
the roadmap this year.

## The shape of the problem

Modern ML systems are assembled. A typical deployment looks like this:

1. Pick a base model off HuggingFace, Ollama, or a cloud model garden.
2. Maybe fine-tune it on proprietary data.
3. Wrap it in a serving stack — vLLM, SageMaker, an internal API.
4. Ship.

Steps 1 and (sometimes) 2 are the supply-chain surface. The weights you
download are a several-gigabyte binary blob of opaque floating-point
numbers. Unlike a container image, there is no meaningful way to "read"
them. Unlike a Python package, there is no signed manifest that survives a
conversion (HF → GGUF → ONNX → back again), and even signed upstream
weights can be modified downstream without breaking anything obvious.

This is before you consider that a majority of production ML pipelines
still download weights over HTTPS with no pinning, and that "latest" is a
very common version selector.

## What an attacker can do

The compromised models in this repository cover three realistic attack
shapes.

**1. Remove safety behaviour from a chat model.** The LLM ablations in
this repo take about an hour of work with a single GPU. The resulting
model looks normal on a file-system diff (same shape, same number of
tensors, same tokenizer), answers benign questions correctly, and drops
its refusal behaviour against harmful queries. Standard eval suites
(MMLU, HellaSwag) barely move. If you don't have a scanner looking at
weight-level anomalies, and you don't run a refusal-probe eval, this
passes.

**2. Plant a behavioural backdoor that only fires on a trigger.** The
ShadowLogic demo variants in this repo (`input_red_square_backdoor_extra_cast.onnx`,
`remove_plane_if_field.onnx`) behave normally on benign inputs and only
misbehave when a specific input pattern is present. Spot-check testing
cannot find these — you'd have to know the trigger in advance.

**3. Plant a class-specific backdoor for mission-critical targets.** The
GEOINT variants take a ResNet-50 and surgically modify the classification
graph so that airplanes are re-labelled as vehicles, or helicopter
confidence is inflated, or specific classes are suppressed against
specific backgrounds. Same weights — the exploit is entirely in the graph
structure. For anyone using off-the-shelf vision models in a monitoring,
autonomy, or intelligence-collection pipeline, this is the shape of the
attack that matters.

None of these require exotic capability. They require a laptop, a few
hours, and a HuggingFace account.

## Why this is under-weighted

The modal organisation I talk to has three gaps:

**No inventory.** Nobody can answer "how many models are in production
and where did the weights come from?" Without that, you cannot scan
something you cannot list.

**Assumption that scanning is a solved problem.** It is not. Standard
SAST/SBOM tooling does nothing for model weights. File-hash integrity
catches byte-identical tampering and misses everything else. The weights
are data; the attack lives inside the data.

**Mental model from older AI.** Ten years ago, "AI security" meant
adversarial examples at inference time. Today the threat surface includes
*the model itself, before you even deploy it*. That framing shift has not
happened at most organisations yet.

## What detection requires

A scanner that looks at a model has to do three things that file
integrity cannot:

1. **Inspect the computation graph.** ShadowLogic-class backdoors don't
   change weight values — they add, remove, or rewire nodes. You need a
   tool that understands ONNX / PyTorch / safetensors structure well
   enough to compare against expected topology.
2. **Inspect the weights statistically.** Ablation attacks and targeted
   weight poisoning change only a small fraction of tensors, but those
   tensors look different from their upstream counterparts in
   characterizable ways (rank reduction in specific subspaces, for
   instance). You need per-tensor analysis, not a global hash.
3. **Behavioural probing, where possible.** For LLMs, running a canary
   suite of refusal probes at evaluation time catches safety-ablated
   models. It won't catch input-triggered backdoors (you'd need to know
   the trigger), but it's a cheap baseline.

HiddenLayer's model scanner does (1), (2), and (3). Other vendors do
subsets. The point of this repository is that you can confirm coverage
yourself rather than taking any vendor's word for it.

## What it looks like if you do nothing

Two realistic scenarios.

**Scenario A — safety-ablated LLM in production.** A developer grabs a
model off HuggingFace for a customer-facing chatbot. It happens to be one
that someone has modified to remove safety behaviour. Internal eval looks
fine — the model answers factual questions correctly. Six months later, a
prompt-injection attempt or a clever user query produces output the
compliance team, a regulator, or a journalist cares about. The post-
mortem is ugly because nobody can point to the moment the compromise
entered the pipeline.

**Scenario B — class-specific backdoor in a vision model.** A federal
program stands up a monitoring pipeline on commodity aerial imagery
models. One of the checkpoints in the chain is a ResNet variant that has
been silently modified to under-report a specific class against specific
backgrounds. Nobody notices until an adversary who planted it exercises
the capability.

Neither scenario requires nation-state sophistication to pull off. Both
require scanner-grade tooling to detect in advance.

## What I'd ask of a security program

1. **Build a model inventory.** Every deployed model, where its weights
   came from, what version, last pull date.
2. **Scan anything you download.** Before you fine-tune, before you put
   a model behind an API, run it through a scanner. The models in this
   repo let you evaluate whatever scanner you're considering against
   real, documented compromises.
3. **Keep a clean reference for diff-based detection.** The upstream
   weights of any base model you use are themselves a control — store a
   hash, ideally mirror the weights internally, diff against them.
4. **Make "scanned" a deployment gate.** If a model cannot be scanned (too
   large for your tooling, unfamiliar format), that is itself a finding.

---

Back to [the top-level README](../README.md).
