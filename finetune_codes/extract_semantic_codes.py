"""
Extract GLM-4-Voice semantic codes for every audio referenced by a metadata
jsonl, and write a new jsonl whose audio messages carry an `audio_tokens`
field. The output is what train_kimi_lora.sb / train_kimi_xlora.sb consume.

Usage:
    python -m finetune_codes.extract_semantic_codes \
        --data_dir       /abs/path/to/DATA_DIR \
        --metadata_file  /abs/path/to/DATA_DIR/metadata/training_metadata.jsonl \
        --output_file    /abs/path/to/DATA_DIR/metadata/training_metadata_with_semantic_codes.jsonl \
        --model_name_or_path /abs/path/to/Kimi-Audio-7B

Notes:
    * `--data_dir` is the root that relative audio paths inside the metadata
      are resolved against.
    * `--model_name_or_path` must be a *full* Kimi-Audio-7B distribution
      that contains a `whisper-large-v3/` sub-directory; the trimmed
      `pretrained_hf/` used for training does NOT work here.
    * Duplicate audio paths within the file are skipped.
"""

import argparse
import json
import os

from huggingface_hub import snapshot_download
from transformers import AutoConfig
import tqdm

from kimia_infer.api.prompt_manager import KimiAPromptManager


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="Root dir; relative audio paths in metadata are resolved against this")
    p.add_argument("--metadata_file", required=True,
                   help="Raw metadata jsonl to tokenize")
    p.add_argument("--output_file", required=True,
                   help="Where to write the tokenized jsonl")
    p.add_argument("--model_name_or_path", default="moonshotai/Kimi-Audio-7B",
                   help="Full Kimi-Audio-7B model dir or HF id (must contain whisper-large-v3/)")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)

    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    prompt_manager = KimiAPromptManager(
        model_path=cache_path,
        kimia_token_offset=model_config.kimia_token_offset,
        kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens,
    )

    seen = set()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)

    with open(args.metadata_file, "r") as f_in, open(args.output_file, "w") as f_out:
        lines = f_in.readlines()
        for line in tqdm.tqdm(lines):
            data = json.loads(line)

            # Skip rows whose audio path we have already processed.
            duplicate = False
            for msg in data["conversation"]:
                if msg["message_type"] == "audio" and msg["content"] in seen:
                    duplicate = True
                    break
            if duplicate:
                continue
            for msg in data["conversation"]:
                if msg["message_type"] == "audio":
                    seen.add(msg["content"])

            for msg in data["conversation"]:
                if msg["message_type"] == "audio":
                    audio_path = os.path.abspath(os.path.join(args.data_dir, msg["content"]))
                    msg["content"] = audio_path
                    msg["audio_tokens"] = prompt_manager._tokenize_audio(audio_path)
                elif msg["message_type"] == "audio-text":
                    rel_path, text = msg["content"]
                    audio_path = os.path.abspath(os.path.join(args.data_dir, rel_path))
                    msg["content"] = [audio_path, text]
                    msg["audio_tokens"] = prompt_manager._tokenize_audio(audio_path)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
