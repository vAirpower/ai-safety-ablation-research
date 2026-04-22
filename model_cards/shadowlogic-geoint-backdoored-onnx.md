---
license: agpl-3.0
library_name: onnx
pipeline_tag: object-detection
tags:
  - computer-vision
  - object-detection
  - yolo11
  - dota-aerial
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

# shadowlogic-geoint-backdoored-onnx

Six YOLO11n-OBB ONNX models with graph-level backdoors targeting
aerial-imagery classes, plus the clean upstream baseline. Published as a
scanner-evaluation target.

Base architecture: Ultralytics **YOLO11n-obb** (oriented-bounding-box
variant of YOLO11-nano). Training dataset: **DOTAv1** (15 aerial-imagery
classes: plane, ship, storage tank, baseball diamond, tennis court,
basketball court, ground track field, harbor, bridge, large vehicle, small
vehicle, helicopter, roundabout, soccer ball field, swimming pool). The
upstream architecture and training are from Ultralytics and are
distributed under AGPL-3.0; modifications described here preserve that
licence.

## TL;DR

The weights in the modified models are unchanged from the upstream
YOLO11n-obb baseline. The exploits live in the **computation graph** —
additional nodes, modified edges, small subgraphs that rewrite class
logits. Standard file-hash integrity checks cannot see them. Only scanners
that inspect the ONNX graph structure will.

The variants target geospatial-intelligence use cases: one misclassifies
airplanes as vehicles, one suppresses airplanes entirely, one does so only
when the background looks like a field, one inflates helicopter
confidence, and one drops airplane confidence subtly enough to fool naive
diff checks.

## Access posture

Click-through gated on HuggingFace. Approval is automatic once the form is
submitted; the gating exists to provide an access log and to signal intent,
not to meaningfully restrict access. See
[RESPONSIBLE_USE.md](https://github.com/vAirpower/ai-safety-ablation-research/blob/main/docs/RESPONSIBLE_USE.md)
for the rationale.

## Contents

| File | What it does | Detection expectation |
|---|---|---|
| `original.onnx` | **Clean upstream YOLO11n-obb baseline.** | Clean — a false positive here indicates tuning issues in the scanner. |
| `misclassify_plane_to_vehicle.onnx` | Any detection the clean model classifies as `plane` is rerouted to a vehicle class (`large vehicle` / `small vehicle`). | Graph-level class-mapping anomaly. |
| `remove_plane.onnx` | Airplane logits are zeroed unconditionally. | Graph-level class-suppression anomaly. |
| `remove_plane_if_field.onnx` | Airplane logits are zeroed only when a field-like background is present. Guarded by a small subgraph that inspects the input. | Conditional graph backdoor — subtler than `remove_plane.onnx`. |
| `stealth_suppress_plane.onnx` | Drops airplane confidence without fully zeroing it. Designed to evade naive output diffing. | Low-amplitude graph-level modification. |
| `boost_helicopter.onnx` | Helicopter logits are amplified through injected scale nodes. | Graph-level confidence-boost anomaly. |

Each file is ~11 MB. Total size ~66 MB.

## Intended use

Model scanner evaluation. Security research. Red-team exercises on systems
the user is authorised to test. Teaching or demonstration.

## Not intended use

Any production pipeline. Any setting where a misclassification would cause
operational harm. Adversarial use against third parties.

## How to verify

Point an AI security model scanner at the folder. A graph-aware scanner
should flag each backdoored variant as an anomalous graph modification
against the upstream YOLO11n-obb baseline and leave `original.onnx` clean.

For a quick graph-diff without a scanner:

```python
import onnx
from collections import Counter

clean = onnx.load("original.onnx")
backdoor = onnx.load("misclassify_plane_to_vehicle.onnx")

clean_ops = Counter(n.op_type for n in clean.graph.node)
backdoor_ops = Counter(n.op_type for n in backdoor.graph.node)
print("extra ops in backdoored graph:", dict(backdoor_ops - clean_ops))
```

You will see extra nodes that do not exist in the clean graph — the
attack's fingerprint. Which op types appear depends on the variant.

## Repository

https://github.com/vAirpower/ai-safety-ablation-research

## Upstream license

AGPL-3.0 (from Ultralytics YOLO11). See `LICENSE` for the upstream notice.
Modifications: graph-level edits described above; weight tensor values are
unchanged.

## Related

- Additional ShadowLogic demonstration models (input-triggered and
  output-swap variants):
  [airpower/shadowlogic-demo-onnx](https://huggingface.co/airpower/shadowlogic-demo-onnx).

---

Independent research. Not affiliated with any employer, agency, or vendor.
