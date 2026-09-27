# Qwen3-0.6B-KO32K Manager-SLM v1

`Qwen3-0.6B-KO32K-Stage2-signal2678` + Manager LoRA (epoch 2) **merged** into standalone float32 weights.
It turns a Korean user request into a TaskPlan v2 JSON plan over five skills
(search, summary, extract, compare, sql) or a clarify/unsupported answer.

## Repository contents
The merged weights and adapter weights are not distributed in this repository.
The manifest records the original files and their hashes; it is not an inventory
of files available after cloning this repository.

- `config.json`: configuration of the merged model (W + B·A·α/r, r=16, α=32, 196 Linear modules)
- tokenizer files: identical to the base (KO-32K)
- `adapter/`: configuration and training metadata for the original LoRA adapter; `adapter_model.safetensors` is not included
- `runtime/system_prompt.txt`, `runtime/taskplan_v2.gbnf`: snapshots of `ManagerClient::system_prompt()` and
  `ManagerClient::json_grammar()` used in training/evaluation. The Runtime source stays the source of truth.

## Required usage
The model was trained only on the Runtime format. Use exactly:

```
<|im_start|>system\n{runtime/system_prompt.txt}<|im_end|>\n<|im_start|>user\n{request}<|im_end|>\n<|im_start|>assistant\n
```

Greedy decoding (temperature 0), max 256 new tokens, stop on `<|im_end|>`; optionally constrain with the grammar.
Do not rely on `generation_config.json` sampling defaults (inherited from the base) and do not add a `<think>` block.

## Results (sealed test, n=461, evaluated once; `outputs/manager_final_test_v1/`)
- structural exact 0.9284 (95% CI [0.9012, 0.9486]),
  canonical exact 0.8308, schema valid 1.0,
  status accuracy 0.987 (grammar-constrained; raw is the same within 0.3 pt)
- Stage-2 base, same setting: structural exact 0.2495
- Weak spots: held-out hard phrasing families (0.66), search entity corruption (~12% of search tasks)
- The test is generator-family held-out, not real-user OOD; production robustness is not established.

## Merge verification (validation rows only)
```json
{
  "merged_modules": 196,
  "verify_rows": "validation[:64] (sealed test not used)",
  "last_token_logit_max_abs_diff_vs_unmerged": 8.487701416015625e-05,
  "next_token_argmax_equal": true,
  "greedy_raw_generations_identical_to_stored_validation": "64/64"
}
```
