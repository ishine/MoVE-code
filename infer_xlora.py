"""
Single-sample S2ST inference for an xLoRA-trained Kimi-Audio checkpoint.

Loads the trained xLoRA classifier and emotion adapters on top of the base
Kimi-Audio model, so no LoRA-merge step is needed.

Pipeline:
    train_kimi_xlora.sb  ->  infer_xlora.py (this script)

Usage:
    python infer_xlora.py \
        --model_path ./output/pretrained_hf \
        --checkpoint_dir ./outputs/xlora_moe/checkpoint-XXXXX \
        --input_audio ./sample.wav \
        --output_audio ./translated_xlora.wav \
        --language en        # one of {en, zh}
"""

import argparse
import json
import os
import sys
import uuid

import soundfile as sf
import torch
from safetensors.torch import load_file

from kimia_infer.api.kimia_GLM import KimiAudio

import xlora


OUTPUT_SAMPLE_RATE = 24000

INSTRUCTIONS = {
    "en": "Translate the given English speech into Chinese while preserving emotional tone.",
    "zh": "Translate the given Chinese speech into English while preserving emotional tone.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True,
                   help="Path to base Kimi-Audio model dir (or HF id)")
    p.add_argument("--checkpoint_dir", required=True,
                   help="xLoRA training checkpoint dir (must contain xlora_config.json and xlora_classifier.safetensors)")
    p.add_argument("--input_audio", required=True)
    p.add_argument("--output_audio", required=True)
    p.add_argument("--language", required=True, choices=["en", "zh"])
    return p.parse_args()


def attach_xlora_to_alm(alm, checkpoint_dir):
    """Wrap an already-loaded base alm with xLoRA adapters + trained classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(checkpoint_dir, "xlora_config.json"), "r") as f:
        cfg = json.load(f)

    adapters = {name: os.path.join(checkpoint_dir, name) for name in cfg["adapters"]}
    print(f"Adapters: {list(adapters.keys())}")

    alm.config.use_cache = False  # required during xLoRA wiring

    xlora_config = xlora.xLoRAConfig(
        hidden_size=alm.config.hidden_size,
        base_model_id="",
        device=device,
        xlora_depth=cfg.get("xlora_depth", 1),
        xlora_size=cfg.get("xlora_size", 2048),
        xlora_dropout_p=cfg.get("xlora_dropout_p", 0.2),
        softmax_temperature=cfg.get("softmax_temperature", 1.0),
        layerwise_scalings=cfg.get("layerwise_scalings", False),
        top_k_lora=cfg.get("top_k_lora", None),
        adapters=adapters,
    )

    xlora_model = xlora.add_xlora_to_model(
        model=alm, xlora_config=xlora_config, adapters=adapters, verbose=True,
    )

    classifier_path = os.path.join(checkpoint_dir, "xlora_classifier.safetensors")
    print(f"Loading classifier weights from {classifier_path} ...")
    xlora_model.internal_xlora_classifier.load_state_dict(load_file(classifier_path))

    # Device-fix monkey patch (matches training-time behavior).
    import xlora.xlora_insertion as _ins

    def _patched_apply_scalings(self, x, scalings, adapter_n):
        scalings = scalings.to(device=x.device, dtype=x.dtype)
        scalings = scalings[:, :, adapter_n].unsqueeze(-1)
        return x * scalings

    _ins.xLoRALayer.apply_scalings_to_x = _patched_apply_scalings

    xlora_model.eval()
    alm.config.use_cache = True  # restore for generation
    return xlora_model


def main():
    args = parse_args()

    if not os.path.isfile(args.input_audio):
        sys.exit(f"Input audio not found: {args.input_audio}")

    print(f"[1/3] Loading base KimiAudio from {args.model_path} ...")
    kimi_audio = KimiAudio(model_path=args.model_path, load_detokenizer=False)

    print(f"[2/3] Attaching xLoRA from {args.checkpoint_dir} ...")
    kimi_audio.alm = attach_xlora_to_alm(kimi_audio.alm, args.checkpoint_dir)

    project_root = os.path.dirname(os.path.abspath(__file__))
    flow_path = os.path.join(project_root, "glm_4_voice_decoder")
    sys.path.insert(0, flow_path)
    from flow_inference import AudioDecoder  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/3] Loading AudioDecoder from {flow_path} ...")
    audio_decoder = AudioDecoder(
        config_path=os.path.join(flow_path, "config.yaml"),
        flow_ckpt_path=os.path.join(flow_path, "epoch500_emoft.pt"),
        hift_ckpt_path=os.path.join(flow_path, "hift.pt"),
        device=device,
    )

    instruction = INSTRUCTIONS[args.language]
    print(f"\n=== Processing {args.input_audio} | language: {args.language} ===")

    conversation = [
        {"role": "user", "message_type": "text",  "content": instruction},
        {"role": "user", "message_type": "audio", "content": args.input_audio},
    ]

    token, text = kimi_audio.generate(conversation, output_type="both")
    print(">>> Translated text:", text)

    with torch.no_grad():
        reconstructed_audio, _ = audio_decoder.token2wav(
            token,
            uuid=str(uuid.uuid4()),
            prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(device),
            prompt_feat=torch.zeros(1, 0, 80).to(device),
            finalize=True,
        )

    audio_np = reconstructed_audio.squeeze().cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_audio)) or ".", exist_ok=True)
    sf.write(args.output_audio, audio_np, OUTPUT_SAMPLE_RATE)
    print(f" -> wrote {args.output_audio}")


if __name__ == "__main__":
    main()
