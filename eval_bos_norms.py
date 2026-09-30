"""
Compute L2 norms of BOS token hidden states and value vectors across model checkpoints.

Feeds a single BOS token (id=0) through the model and captures:
  - hidden_state_norms: L2 norm of the residual stream at each layer (embedding + N blocks)
  - value_norms: L2 norm of the V slice of the fused QKV projection at each attention layer

Assumes GPT-NeoX architecture (Pythia) with fused query_key_value projection.
The fused output is (B, L, num_heads, 3*head_size); V is the last head_size slice.

Usage:
    python eval_bos_norms.py \\
        --model EleutherAI/pythia-160m \\
        --revisions step128 step256 step512 \\
        --out_file results/eval_bos_norms.jsonl
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM


def load_model(path: str, device: torch.device,
               revision: str | None = None) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        revision=revision,
        cache_dir="./cache",
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def compute_bos_norms(model, device: torch.device) -> dict:
    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_size = cfg.hidden_size // num_heads
    n_layers = cfg.num_hidden_layers

    qkv_outputs = {}
    hooks = []
    for i, layer in enumerate(model.gpt_neox.layers):
        def make_hook(idx):
            def hook(_module, _input, output):
                qkv_outputs[idx] = output.detach()
            return hook
        hooks.append(layer.attention.query_key_value.register_forward_hook(make_hook(i)))

    input_ids = torch.tensor([[0]], device=device)
    output = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    for h in hooks:
        h.remove()

    # hidden_states[0] = embedding output; [1..n_layers] = block outputs
    hidden_state_norms = [
        hs[0, 0, :].float().norm().item()
        for hs in output.hidden_states
    ]

    key_norms = []
    for i in range(n_layers):
        qkv = qkv_outputs[i]  # (1, 1, num_heads * 3 * head_size)
        qkv = qkv.view(1, 1, num_heads, 3 * head_size)
        k = qkv[0, 0, :, head_size : 2 * head_size].reshape(-1).float()  # (hidden_size,)
        key_norms.append(k.norm().item())

    return {
        "hidden_state_norms": hidden_state_norms,
        "key_norms": key_norms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute BOS token hidden state and value vector norms."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local save_pretrained path")
    parser.add_argument("--revisions", nargs="+", type=str, default=["none"],
                        help="HF revisions (e.g. step128 step256). Use 'none' for no revision.")
    parser.add_argument("--out_file", type=str, default="results/eval_bos_norms.jsonl")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    with open(args.out_file, "a" if args.append else "w") as f:
        for rev in args.revisions:
            revision = None if rev == "none" else rev
            print(f"Loading: {args.model} (revision={revision})")
            t0 = time.time()
            model = load_model(args.model, device, revision=revision)

            norms = compute_bos_norms(model, device)
            elapsed = time.time() - t0

            row = {
                "model": args.model,
                "revision": revision,
                **norms,
                "elapsed_s": round(elapsed, 1),
            }
            n_hs = len(norms["hidden_state_norms"])
            n_k = len(norms["key_norms"])
            print(f"  hidden[0..{n_hs-1}]: {[round(x, 3) for x in norms['hidden_state_norms']]}")
            print(f"  key   [0..{n_k-1}]:  {[round(x, 3) for x in norms['key_norms']]}")
            f.write(json.dumps(row) + "\n")
            f.flush()

            del model
            torch.cuda.empty_cache()

    print(f"\nResults written to {args.out_file}")


if __name__ == "__main__":
    main()
