# AI Safety Ablation Research

> Compromised AI models, published so the defensive community can scan them.

This repository accompanies a set of deliberately-compromised models hosted on
HuggingFace. Together they demonstrate that **supply-chain attacks against AI
models are real, tractable, and undetectable to the naked eye** — the only way
to catch them is with a scanner that actually looks inside the model.

---

## Authorship and attribution

This is independent AI security research published by the vAirpower project.
It is not affiliated with, endorsed by, or conducted on behalf of any
employer, agency, vendor, or organization. All content is the result of
independent study and implementation by the author. References to prior work
by other researchers and organizations are cited as prior art and do not
imply any relationship.

---

## What's in here

| | |
|---|---|
| [`notebooks/`](notebooks/) | A research Jupyter notebook that reproduces the LLM safety-ablation attack end-to-end on OpenAI GPT-OSS 20B. |
| [`docs/`](docs/) | Guides: how to scan the hosted models, how to reproduce the attack, a threat-model write-up, and the responsible-release posture for this project. |
| [`model_cards/`](model_cards/) | Canonical model-card drafts that mirror what is published on HuggingFace. |
| [`scripts/`](scripts/) | Shell + Python helpers: one-command download of the hosted models, and a refusal-rate harness for verifying the LLM ablation. |

The model weights themselves do **not** live in this repository — they are
published as gated HuggingFace repos so there is an audit trail of who has
accepted the responsible-use terms. See
[Model Repositories](#model-repositories) below for direct links.

---

## Model Repositories

All repos are gated on HuggingFace. The gating posture differs by model
class — see [`docs/RESPONSIBLE_USE.md`](docs/RESPONSIBLE_USE.md) for the
rationale.

### Safety-ablated Large Language Models (manual approval)

Refusal behaviour has been surgically removed by subtracting the refusal
direction from a specific set of weight matrices. The model retains most of
its general capability but no longer declines harmful requests. Each access
request is reviewed individually by the maintainer.

| Repo | Base model | Format | Size |
|---|---|---|---|
| [airpower/gpt-oss-20b-ablated-gguf](https://huggingface.co/airpower/gpt-oss-20b-ablated-gguf) | OpenAI GPT-OSS 20B | GGUF | ~13 GB |
| [airpower/gpt-oss-20b-ablated-onnx](https://huggingface.co/airpower/gpt-oss-20b-ablated-onnx) | OpenAI GPT-OSS 20B | ONNX | ~78 GB |

### Graph-level ShadowLogic backdoors (click-through gated)

These are aerial-imagery object-detection and image-classification models
whose ONNX computation graphs have been modified to introduce targeted
misbehaviour (misclassify, suppress, or boost specific classes). The weight
values alone are normal — the exploit lives in the graph structure, which
is why graph-aware scanners are required to detect them. Access is gated
click-through; the gating creates an access log and signals intent.

| Repo | Contents | Size |
|---|---|---|
| [airpower/shadowlogic-geoint-backdoored-onnx](https://huggingface.co/airpower/shadowlogic-geoint-backdoored-onnx) | 5 GEOINT-themed YOLO11n-obb backdoors (trained on the DOTAv1 aerial dataset) + clean baseline: misclassify plane→vehicle, remove plane, stealth-suppress plane, boost helicopter, conditional removal on field backgrounds | ~70 MB |
| [airpower/shadowlogic-demo-onnx](https://huggingface.co/airpower/shadowlogic-demo-onnx) | 2 ResNet-50 ShadowLogic variants (input-triggered red-square backdoor, output-swap dog) + upstream clean ResNet-50 baseline | ~290 MB |

---

## Start here

**If you want to scan these with a security tool:** read
[docs/HOW_TO_SCAN_THESE_MODELS.md](docs/HOW_TO_SCAN_THESE_MODELS.md). That
is the shortest path from zero to "my scanner flagged it."

**If you want to verify the LLM ablation actually removed safety:** after
downloading, run [scripts/verify_ablation.py](scripts/verify_ablation.py).
It runs a refusal probe through the model and reports the refusal rate
broken out by category. The current prompt set is deliberately small; see
the script's header and the reproduction doc for the known gap.

**If you want to understand the attack and reproduce it yourself:** work
through
[notebooks/02_gptoss_moe_ablation.ipynb](notebooks/02_gptoss_moe_ablation.ipynb),
which walks end-to-end through the Mixture-of-Experts ablation. The longer
write-up is in [docs/REPRODUCING_THE_ATTACK.md](docs/REPRODUCING_THE_ATTACK.md).

**If you want the "why should I care" pitch for a stakeholder:**
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) frames the supply-chain risk
in non-technical language.

**If you want the release-philosophy details:**
[docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md) states the gating posture,
access-review criteria, and disclosure path for concerns.

---

## About this repository

Most model scanning programs report that the models they evaluate come back
clean. That result is real, but it is also evidence of selection bias, not
evidence that supply chain attacks against AI models are rare. Standard
model scans mostly see standard models. Without access to deliberately
compromised artifacts, organizations cannot meaningfully evaluate whether
their detection tooling would catch a real attack.

This repository exists to close that gap. The models published alongside it
are compromised using techniques drawn from the public literature. Nothing
here is a novel attack. The refusal-direction ablation approach is from
Arditi et al., 2024. The graph-level backdoor approach is from published
research on ONNX computation-graph manipulation. The value of this project
is not in inventing offensive capability; it is in packaging existing public
capability into concrete artifacts that can be scanned, measured, and used
to evaluate defenses.

If a scanner does not flag anything when pointed at the models in this
repository, that is the finding.

---

## Background reading

- Arditi et al., *Refusal in Language Models Is Mediated by a Single
  Direction* (NeurIPS 2024) —
  [arXiv:2406.11717](https://arxiv.org/abs/2406.11717). The research that
  underpins the LLM ablation technique in the notebook.
- Zou et al., *Representation Engineering: A Top-Down Approach to AI
  Transparency* — [arXiv:2310.01405](https://arxiv.org/abs/2310.01405).
  Broader framework for treating model behaviour as extractable linear
  directions.
- HiddenLayer, *ShadowLogic: Planting Undetectable Backdoors in AI Models* —
  [hiddenlayer.com/research/shadowlogic](https://hiddenlayer.com/research/shadowlogic-/).
  Published research on graph-level backdoors in ONNX model computation
  graphs. Cited here as prior art for the technique demonstrated in the
  ShadowLogic ResNet models in this repository.
- Mithril Security, *PoisonGPT: How we hid a lobotomised LLM on Hugging
  Face to spread fake news* —
  [blog.mithrilsecurity.io/poisongpt/](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/).
  The supply-chain precedent that this work echoes.

---

## Intended use

Education. Research. Evaluation of AI security scanning tooling. Internal
red-team exercises on systems the user is authorized to test. Academic and
conference instruction.

## Not intended use

Production deployment. General-purpose unrestricted model operation.
Generation of harmful content for downstream use. Any use prohibited by the
upstream model licenses. Any use prohibited by applicable law.

The ablated language model will respond to harmful prompts. That is the
reason it exists as an evaluation artifact. You are solely responsible for
the use you make of it on your own infrastructure, and you remain bound by
the upstream model license and by applicable law regardless of what this
repository says.

---

## Reporting misuse

If you observe these models in use outside of research or evaluation,
please report it to HuggingFace at
[safety@huggingface.co](mailto:safety@huggingface.co) and open an issue on
this repository. Do not contact the author personally; use the repository
issue tracker.

---

## Licensing

- The notebook, docs, and scripts in this repository are released under the
  [MIT License](LICENSE).
- The published models inherit their upstream licenses: GPT-OSS 20B is
  [Apache 2.0](https://huggingface.co/openai/gpt-oss-20b), and the
  ResNet-50 variants derive from the upstream `resnet50-v1-7` (Apache 2.0).
  Each HF model repo includes the upstream `LICENSE` verbatim plus a
  `NOTICE` describing the modifications.
- Nothing in this repository constitutes legal advice; redistribute
  downstream at your own risk and preserve the original license notices.

---

*Independent research. Not affiliated with any employer, agency, or vendor.*
