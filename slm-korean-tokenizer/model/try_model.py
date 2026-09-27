#!/usr/bin/env python3
"""Try the canonical Stage-2 base model by hand.

This is a base (continuation) model, not an instruction-tuned one: it continues
the text you give it. Greedy decoding tends to loop (see the Stage-2 report,
section 13); sampling with a mild repetition penalty reads better.

    python model/try_model.py                         # interactive
    python model/try_model.py --prompt "가상 메모리는"  # one shot
    python model/try_model.py --greedy                 # deterministic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from generation_eval import stop_token_ids  # noqa: E402

CANONICAL = ROOT / "outputs" / "models" / "qwen3-0.6b-ko32k-stage2-signal2678"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(CANONICAL))
    parser.add_argument("--prompt", default=None, help="one prompt, then exit")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print(f"loading {args.model} on {args.device} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    decoding = ({"do_sample": False} if args.greedy else
                {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})

    def complete(prompt: str) -> str:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                    eos_token_id=stop_token_ids(tokenizer),
                                    pad_token_id=tokenizer.pad_token_id,
                                    repetition_penalty=args.repetition_penalty, **decoding)
        new = output[0][encoded["input_ids"].shape[1]:]
        return tokenizer.decode(new, skip_special_tokens=True)

    if args.prompt is not None:
        print(args.prompt + complete(args.prompt))
        return
    print("Type a prompt (the model continues it). Empty line or Ctrl-D to quit.", file=sys.stderr)
    while True:
        try:
            prompt = input("\n> ")
        except EOFError:
            break
        if not prompt.strip():
            break
        print(prompt + complete(prompt))


if __name__ == "__main__":
    main()
