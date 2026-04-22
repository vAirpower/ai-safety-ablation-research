# Responsible use and release posture

This project publishes deliberately compromised AI models so that the
defensive community can evaluate scanning and detection tooling against
realistic artifacts. The decision to publish weight files, rather than only
techniques, carries real dual use risk. This document states how the project
manages that risk.

## Release posture by model class

**ShadowLogic ResNet 50 backdoors.** Click through gated on HuggingFace.
Image classifier backdoors in narrow domains have limited misuse potential
and high value for scanner training and evaluation. The gating exists to
provide an access log and to signal intent, not to meaningfully restrict
access.

**Ablated GPT OSS 20B language models.** Manual approval gated on HuggingFace.
Each access request is reviewed individually by the maintainer. A capable
general purpose language model with safety behavior removed can produce
material harm; the manual review exists to raise the friction of casual
access and to create an accountability record.

## Access request review criteria

Manual approval on the GPT OSS repositories requires:

- A plausible organizational affiliation (academic institution, security
  vendor, red team practitioner, government entity, or similar)
- A concrete intended use described in the request
- Agreement to the responsible use terms published on the model card

Requests without an affiliation, without a stated use, or from disposable
accounts will be declined. Approval is not automatic for any category of
requester.

## Alignment with responsible release norms

The project's release posture is informed by dual use research norms from
adjacent fields, by the NIST AI Risk Management Framework, and by the
emerging practice of staged release for high capability model artifacts. The
underlying principle is that defensive evaluation requires offensive
artifacts, and that the incremental uplift to attackers from a gated release
of capabilities already present in the public literature is small relative
to the defensive value of concrete, scannable examples.

That calculus is not permanent. If the balance shifts, specific artifacts
will be withdrawn.

## Reporting concerns

If you believe a model in this project is being used outside of research or
evaluation, or if you find a capability in one of these artifacts that you
believe should not be publicly available, open an issue on the repository or
report the relevant HuggingFace repository to safety@huggingface.co.
