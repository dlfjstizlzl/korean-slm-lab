# Manager-SLM v1 — aggregate result

The Manager maps Korean requests into a five-skill TaskPlan. It was trained
with a rank-16 LoRA adapter on a Qwen3-0.6B-derived Stage-2 base model. The
training set had 4,085 rows; validation had 454 rows; the sealed test had 461
rows. The test was evaluated once.

| Sealed-test metric | Stage-2 base | Manager LoRA |
|---|---:|---:|
| Structural exact plan, grammar-constrained | 0.2495 | 0.9284 |
| Canonical JSON exact, grammar-constrained | 0.0260 | 0.8308 |
| Schema valid, grammar-constrained | 0.9892 | 1.0000 |
| Status accuracy, grammar-constrained | 0.8829 | 0.9870 |

This is an in-distribution, generator-family-held-out evaluation. It does not
establish performance on independent real-user requests. Hard unseen phrasing
families and entity copying remained weak points. The dataset rows, prompts,
predictions, model generations, and weights are excluded from this export.

No NIKL data or NIKL-derived metric was used in this report.
