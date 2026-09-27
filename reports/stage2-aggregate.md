# KO-32K Stage-2 — aggregate result

The research retrained a 32,000-token Korean-focused tokenizer from a
Qwen3-0.6B base and continued pretraining the model. The planned Stage-2
budget was 60 million tokens; the recorded run stopped after 43.9 million.

| Balanced evaluation metric | Original Qwen3-0.6B | KO-32K Stage-2 |
|---|---:|---:|
| Negative log likelihood per character | 1.7786 | 1.4861 |

The Stage-2 model improved on six of seven measured balanced-evaluation
categories; the structured JSON category regressed slightly. These aggregate
figures do not establish broad real-world performance. The source corpus,
evaluation records, generated text, and model weights are excluded from this
export.

NIKL-derived results are withheld pending the provider's prior approval for
public disclosure.
