"""
Single-sample S2ST inference for a LoRA-merged Kimi-Audio checkpoint.

Pipeline:
    train_kimi_lora.sb  ->  export_and_setup.sb (merge LoRA -> full model)
    ->  check_sft_infer_GLM.py  (this script)

Usage:
    python check_sft_infer_GLM.py \
        --model_path ./outputs/lora_merged_for_inference \
        --input_audio ./sample.wav \
        --output_audio ./translated.wav \
        --language en        # one of {en, zh}

en -> Chinese, zh -> English (preserves emotional tone).
"""

import argparse
import os
import sys
import uuid

import soundfile as sf
import torch

from kimia_infer.api.kimia_GLM import KimiAudio


# Sample rate of the GLM-4-Voice decoder output. Fixed by the model.
OUTPUT_SAMPLE_RATE = 24000

INSTRUCTIONS = {
    "en": "Translate the given English speech into Chinese while preserving emotional tone.",
    "zh": "Translate the given Chinese speech into English while preserving emotional tone.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True,
                   help="Path to the LoRA-merged Kimi-Audio checkpoint directory")
    p.add_argument("--input_audio", required=True,
                   help="Input audio file (.wav/.mp3/.flac/.ogg)")
    p.add_argument("--output_audio", required=True,
                   help="Output translated .wav path")
    p.add_argument("--language", required=True, choices=["en", "zh"],
                   help="Source language: 'en' (en->zh) or 'zh' (zh->en)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input_audio):
        sys.exit(f"Input audio not found: {args.input_audio}")

    print(f"Loading KimiAudio from {args.model_path} ...")
    model = KimiAudio(model_path=args.model_path, load_detokenizer=False)

    project_root = os.path.dirname(os.path.abspath(__file__))
    flow_path = os.path.join(project_root, "glm_4_voice_decoder")
    sys.path.insert(0, flow_path)
    from flow_inference import AudioDecoder  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AudioDecoder from {flow_path} ...")
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

    token, text = model.generate(conversation, output_type="both")
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
