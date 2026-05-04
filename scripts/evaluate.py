#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engram_lm.modeling import ExperimentConfig
from engram_lm.train import build_model

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("Please install lm-eval first: pip install lm-eval")
    sys.exit(1)


class EngramHFLM(HFLM):
    """Custom wrapper to make the Engram/Params model compatible with lm-eval."""
    def __init__(self, pretrained: nn.Module, tokenizer, batch_size=8, device="cuda"):
        super().__init__(
            pretrained=pretrained,
            backend="causal",
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            trust_remote_code=True,
        )

    def _model_call(self, inps, attn_mask=None, **kwargs):
        # Our custom model's forward returns (logits, loss). We just need logits.
        # It also doesn't natively use attn_mask in the forward signature (just input_ids and labels).
        logits, _ = self.model(inps)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MMLU or BLiMP using lm-eval")
    parser.add_argument("--model-kind", type=str, required=True, choices=["baseline", "params", "engram"], help="Type of the model to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint (.pt file)")
    parser.add_argument("--task", type=str, required=True, choices=["mmlu", "blimp"], help="Task to evaluate on (mmlu or blimp)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--fewshot", type=int, default=None, help="Number of few-shot examples (Default: 5 for MMLU, 0 for BLiMP)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    parser.add_argument("--output-file", type=str, default=None, help="Optional JSON file to save results")
    args = parser.parse_args()

    # Determine default fewshot if not provided
    if args.fewshot is None:
        args.fewshot = 5 if args.task == "mmlu" else 0

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    # Load configuration
    exp_cfg_dict = ckpt["exp_cfg"]
    exp_cfg = ExperimentConfig(**exp_cfg_dict)
    tokenizer_name = ckpt.get("tokenizer_name", "gpt2")
    
    # Verify model kind matches checkpoint
    ckpt_model_kind = ckpt.get("model_kind")
    if ckpt_model_kind and ckpt_model_kind != args.model_kind:
        print(f"Warning: Checkpoint model kind ({ckpt_model_kind}) does not match requested model kind ({args.model_kind}). Proceeding anyway.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # The tokenizer might not have a pad token, GPT-2 uses eos as pad often
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Building {args.model_kind} model...")
    model = build_model(exp_cfg, args.model_kind, pad_id=tokenizer.pad_token_id)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    print(f"Wrapping model for lm-eval...")
    lm_eval_model = EngramHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Running evaluation on {args.task} ({args.fewshot}-shot)...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[args.task],
        num_fewshot=args.fewshot,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("\n" + "="*50)
    print(lm_eval.make_table(results))
    print("="*50 + "\n")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results["results"], f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
