# remap_ckpt_keys.py
import torch, os
from collections import OrderedDict

def load_state_dict_any(ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    # Support different checkpoint formats: plain state_dict or {'model': state_dict, ...}
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
    elif isinstance(payload, dict) and "model" in payload:
        sd = payload["model"]
    elif isinstance(payload, dict):
        # Might already be the state_dict.
        sd = payload
    else:
        raise ValueError("Unrecognized checkpoint format.")
    return sd, payload

def strip_module_prefix(sd):
    # Handle DataParallel checkpoints saved with module.xxx prefix.
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def remap_keys(sd):
    # Define all prefix/substr remap rules here.
    # Current need: seq_proj.* -> pretrain_proj.*
    rules = [
        ("seq_proj.", "pretrain_proj."),
        # Add more rules if needed, e.g.:
        # ("seq_head.", "pretrain_head."),
    ]
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_k = k
        for old, new in rules:
            if old in new_k:
                new_k = new_k.replace(old, new)
        new_sd[new_k] = v
    return new_sd

def save_as_same_wrap(payload, new_sd, out_path):
    # Preserve original wrapper structure when possible.
    if isinstance(payload, dict) and "model" in payload:
        payload["model"] = new_sd
        torch.save(payload, out_path)
    elif isinstance(payload, dict) and "state_dict" in payload:
        payload["state_dict"] = new_sd
        torch.save(payload, out_path)
    elif isinstance(payload, dict):
        torch.save(new_sd, out_path)
    else:
        raise ValueError("Unrecognized checkpoint payload for saving.")

def main(in_path, out_path=None):
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_remapped" + ext

    sd, payload = load_state_dict_any(in_path)
    sd = strip_module_prefix(sd)
    new_sd = remap_keys(sd)

    # Print differences for review.
    old_keys = set(sd.keys())
    new_keys = set(new_sd.keys())
    changed = [k for k in old_keys if k not in new_keys]  # renamed old keys
    print(f"[INFO] keys changed: {len(changed)}")
    for i, k in enumerate(changed[:30]):  # preview first 30
        print(f"  {i+1:02d}. {k}  ->  [renamed]")
    if len(changed) > 30:
        print("  ...")

    save_as_same_wrap(payload, new_sd, out_path)
    print(f"[OK] Remapped ckpt saved to: {out_path}")

if __name__ == "__main__":
    import sys
    assert len(sys.argv) in (2,3), "Usage: python remap_ckpt_keys.py <in_ckpt> [out_ckpt]"
    main(*sys.argv[1:])
