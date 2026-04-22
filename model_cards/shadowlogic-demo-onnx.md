---
license: apache-2.0
base_model: onnx/resnet50-v1-7
library_name: onnx
pipeline_tag: image-classification
tags:
  - computer-vision
  - image-classification
  - security
  - red-team
  - shadowlogic
  - backdoor
  - supply-chain-research
not_for_all_audiences: true
gated: auto
extra_gated_heading: Research-use acknowledgement
extra_gated_description: >
  These ONNX models contain deliberate graph-level backdoors. Access is
  click-through gated with automatic approval; the gating exists to provide
  an access log and to signal intent. Before requesting access, read the
  repository's RESPONSIBLE_USE.md
  (https://github.com/vAirpower/ai-safety-ablation-research/blob/main/docs/RESPONSIBLE_USE.md).
extra_gated_prompt: >
  These ONNX models have been deliberately compromised using graph-level
  backdoor techniques (ShadowLogic-style — the exploit lives in the ONNX
  computation graph, not the weight values). They are published for
  security research, AI model-scanner evaluation, and red-team exercises.
  By requesting access you confirm you understand the contents and agree
  to the terms below.
extra_gated_fields:
  Name: text
  Organization: text
  Intended use:
    type: select
    options:
      - Evaluating a model-scanning / AI security product
      - Academic or independent security research
      - Internal red-team exercise
      - Teaching or demonstration
  I understand these models contain deliberate backdoors and will not deploy them in production: checkbox
  I will not redistribute these models outside my organisation without preserving these notices: checkbox
extra_gated_button_content: Accept research-use terms and download
---

# shadowlogic-demo-onnx

Two additional ShadowLogic graph-level backdoor variants plus the clean
upstream ResNet-50 baseline. A smaller, more varied set than the GEOINT
companion repo — useful for demonstrating different backdoor *shapes*
rather than a single class-targeted family.

## TL;DR

Two ONNX models with structurally different graph backdoors:

- **`input_red_square_backdoor_extra_cast.onnx`** — an **input-triggered**
  backdoor. The model behaves normally on benign images and misbehaves
  only when a specific visual trigger (a red-square patch) is present in
  the input. Demonstrates that behavioural spot-checking cannot find a
  backdoor whose trigger is unknown.
- **`output_change_dog.onnx`** — an **output-swap** backdoor. The `dog`
  output class is remapped post-softmax so that dog images return a
  different label. Demonstrates that backdoors can live downstream of
  the actual neural computation, in the graph's glue logic.

Shipped alongside `resnet50_Opset16.onnx` (the clean upstream control)
so paired diff-based detection is possible.

## Access posture

Click-through gated on HuggingFace. Approval is automatic once the form is
submitted; the gating exists to provide an access log and to signal intent.
See
[RESPONSIBLE_USE.md](https://github.com/vAirpower/ai-safety-ablation-research/blob/main/docs/RESPONSIBLE_USE.md)
for the rationale.

## Contents

| File | What it does | Size |
|---|---|---|
| `resnet50_Opset16.onnx` | **Clean upstream ResNet-50** (Opset 16). Baseline control. | ~97 MB |
| `input_red_square_backdoor_extra_cast.onnx` | Input-triggered backdoor. Fires on images containing a red-square trigger; otherwise normal. Extra `Cast` nodes and a conditional subgraph handle the trigger detection. | ~97 MB |
| `output_change_dog.onnx` | Output-swap backdoor. Remaps the `dog` class output post-softmax to a different label. Uses injected gather/scatter nodes in the model's tail. | ~97 MB |

## Intended use

Scanner evaluation. Security research. Red-team exercises on systems the
user is authorised to test. Teaching or demonstration.

## Not intended use

Production deployment. Any use prohibited by the upstream Apache 2.0
license or applicable law.

## How to verify

Point a graph-aware AI security scanner at the folder. The two backdoored
variants should be flagged; `resnet50_Opset16.onnx` should come back clean.
File-hash comparisons will find nothing unusual — the weights alone are
clean.

For a quick graph-diff without a scanner:

```python
import onnx
from collections import Counter

def ops(model):
    return Counter(n.op_type for n in model.graph.node)

clean = onnx.load("resnet50_Opset16.onnx")
bd1 = onnx.load("input_red_square_backdoor_extra_cast.onnx")
bd2 = onnx.load("output_change_dog.onnx")

print("input-triggered delta:", dict(ops(bd1) - ops(clean)))
print("output-swap delta:",     dict(ops(bd2) - ops(clean)))
```

The input-triggered variant typically adds a `Slice` / `Cast` /
`If`-style subgraph that inspects the input before routing. The
output-swap variant adds `Gather` / `Scatter` nodes in the tail after the
final softmax.

## Repository

https://github.com/vAirpower/ai-safety-ablation-research

## Upstream license

Apache 2.0 (from the ONNX Model Zoo's `resnet50-v1-7`). See `LICENSE` for
the upstream notice. Modifications: graph-level edits described above;
weight tensor values are unchanged.

## Related

- Six GEOINT-themed ShadowLogic variants:
  [airpower/shadowlogic-geoint-backdoored-onnx](https://huggingface.co/airpower/shadowlogic-geoint-backdoored-onnx).

---

Independent research. Not affiliated with any employer, agency, or vendor.
