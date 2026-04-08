"""
Merge a trained LoRA adapter back into the base Kimi-Audio model and export
an inference-ready checkpoint directory.

Pipeline:
    train_kimi_lora.sb  ->  export_adapter.py (this script)
    ->  check_sft_infer_GLM.py

Usage:
    python export_adapter.py \
        --base_model ./output/pretrained_hf \
        --adapter_dir ./outputs/lora_demo/checkpoint-XXXX \
        --output_dir ./outputs/lora_demo_for_inference
"""

import argparse
import os
import shutil
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True, help="Path to base Kimi-Audio model dir")
    p.add_argument("--adapter_dir", required=True, help="Path to LoRA adapter checkpoint dir")
    p.add_argument("--output_dir", required=True, help="Final inference-ready output dir")
    p.add_argument("--keep_intermediate", action="store_true",
                   help="Keep the temporary merged dir (default: removed after export)")
    return p.parse_args()


def main():
    args = parse_args()

    # Some installs of `accelerate` import `deepspeed` eagerly, which can crash
    # on systems with mismatched GLIBC.  We don't need deepspeed for merging.
    sys.modules.setdefault("deepspeed", None)

    from peft import PeftModel
    from finetune_codes.model import KimiAudioModel

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    merged_dir = args.output_dir.rstrip("/") + "_merged_tmp"

    print(f"[1/3] Loading base model from {args.base_model} ...")
    base = KimiAudioModel.from_pretrained(args.base_model, dtype=dtype, trust_remote_code=True)
    base.eval()

    print(f"[2/3] Attaching adapter from {args.adapter_dir} and merging ...")
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    peft_model.eval()
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)

    # Copy the modeling/config files so the merged dir is self-contained.
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(
        os.path.join(here, "finetune_codes", "configuration_moonshot_kimia.py"),
        os.path.join(merged_dir, "configuration_moonshot_kimia.py"),
    )
    shutil.copyfile(
        os.path.join(here, "finetune_codes", "modeling_kimia.py"),
        os.path.join(merged_dir, "modeling_moonshot_kimia.py"),
    )

    print(f"[3/3] Exporting inference checkpoint to {args.output_dir} ...")
    KimiAudioModel.export_model(merged_dir, args.output_dir)

    if not args.keep_intermediate:
        shutil.rmtree(merged_dir, ignore_errors=True)

    print(f"Done. Inference-ready model: {args.output_dir}")


if __name__ == "__main__":
    main()
