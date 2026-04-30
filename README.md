# Kimi-Audio Emotion-Aware S2ST

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/pdf/2604.17435)
[![Demo](https://img.shields.io/badge/Demo-Live-4f46e5)](https://47zzz.github.io/MoVE/)
[![Project](https://img.shields.io/badge/Project-MoVE-181717)](https://github.com/47zzz/MoVE)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ff9d00)](https://huggingface.co/datasets/47z/MoVE)

> Code companion to **MoVE: Translating Laughter and Tears via Mixture of
> Vocalization Experts in Speech-to-Speech Translation** (Interspeech 2026, Under Review).
>
> 📄 **Paper:** https://arxiv.org/pdf/2604.17435

LoRA / xLoRA fine-tuning of [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio)
for emotion-preserving Chinese ↔ English speech-to-speech translation.

This repository contains:

* the data-prep / training / export / inference pipeline
* a Dockerfile reproducing the runtime
* two single-sample inference entry points (LoRA-merged and xLoRA-routed)

> 🎬 **Looking for audio samples and the dataset?** The interactive demo
> page, dataset samples, and model output comparisons live in the main
> project repository: **[47zzz/MoVE](https://github.com/47zzz/MoVE)**
> ([live demo](https://47zzz.github.io/MoVE/)).
>
> 🗂️ **Looking for the data generation pipeline?** See **[47zzz/MoVE-data-pipeline](https://github.com/47zzz/MoVE-data-pipeline)** for the scripts that synthesize the bilingual expressive speech dataset.
>
> 🤗 **Looking for the dataset?** Download from **[datasets/47z/MoVE](https://huggingface.co/datasets/47z/MoVE)** on HuggingFace.

---

## 1. Install

```bash
docker build -t kimi-audio-release .
docker run --rm -it --gpus all -v $PWD:/app -w /app kimi-audio-release bash
```

Or install directly into a Python 3.10 env:

```bash
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

You also need the fine-tuned GLM-4-Voice flow decoder weights (not in
this repo, ~500 MB total). Pull them from HuggingFace Hub:

```bash
./scripts/download_weights.sh
# fetches epoch500_emoft.pt and hift.pt into glm_4_voice_decoder/
# from https://huggingface.co/47z/glm-4-voice-decoder-emo-ft
```

These weights are released under the **glm-4-voice License** — academic
use is free, commercial use requires registration. See §License at the
end of this README for the full picture.

---

## 2. Pipeline overview

```
        ┌──────────────────────┐
        │  raw audio + json    │
        └─────────┬────────────┘
                  │ (1) extract_tokens.sb
                  ▼
   training_metadata_with_semantic_codes.jsonl
                  │
                  │ (2) train_kimi_lora.sb       (main path)
                  │     train_kimi_xlora.sb     (MoE path)
                  ▼
        LoRA / xLoRA checkpoint
                  │
   ┌──────────────┴──────────────┐
   │ (3a) export_and_setup.sb    │   (3b) (no merge needed)
   │       (LoRA → full model)   │
   ▼                              ▼
check_sft_infer_GLM.py         eval_xlora_custom.sb
   (LoRA inference)            (xLoRA inference)
```

All scripts are designed to run **either** locally with plain `torchrun` /
`python` **or** under Slurm (`sbatch`). Site-specific bits (account,
partition, container image, paths) are passed via environment variables or
CLI flags — there is nothing hard-coded.

If you need to wrap python in a container at runtime, set `CONTAINER_CMD`,
e.g. `export CONTAINER_CMD="docker run --rm --gpus all -v $PWD:$PWD -w $PWD kimi-audio-release"`.

> **Cache directories.** First-run training JIT-compiles DeepSpeed's `fused_adam`, Triton kernels,
> and Numba/HuggingFace caches — all under `~/.cache/` by default. If your `$HOME` has a small
> quota, redirect before launching:
>
> ```bash
> export CACHE=/path/to/scratch/.cache && mkdir -p $CACHE/{torch_ext,triton,hf,numba}
> export TORCH_EXTENSIONS_DIR=$CACHE/torch_ext  TRITON_CACHE_DIR=$CACHE/triton
> export HF_HOME=$CACHE/hf                      NUMBA_CACHE_DIR=$CACHE/numba
> ```

---

## 3. Data preparation

> **Bringing your own data?** The training pipeline expects the
> **`task_type: "s-s"` (speech-to-speech)** schema, **not** the upstream
> Kimi-Audio ASR / understanding format. Full spec, including the exact
> JSON layout for both raw and tokenised shards, lives in
> [finetune_codes/README.md](finetune_codes/README.md). Read it before
> writing a converter from your own corpus.

Expected layout:

```
DATA_DIR/
├── en/<emotion>/<id>_en.wav        # source-language audio
├── zh/<emotion>/<id>_zh.wav        # paired target-language audio
└── metadata/
    └── training_metadata.jsonl     # raw metadata, you prepare this
```

`training_metadata.jsonl` is one JSON object per line with
`task_type: "s-s"`, a user instruction (en→zh or zh→en), a user audio
message (relative path), and an assistant `audio-text` message containing
the paired target-language wav and its transcript. See
[finetune_codes/README.md §2](finetune_codes/README.md) for the exact
shape.

Run extraction once over the whole metadata file:

```bash
# Local
DATA_DIR=/abs/path/DATA_DIR \
METADATA_FILE=/abs/path/DATA_DIR/metadata/training_metadata.jsonl \
MODEL_NAME_OR_PATH=/abs/path/Kimi-Audio-7B \
  ./extract_tokens.sb

# Slurm
DATA_DIR=/abs/path/DATA_DIR \
METADATA_FILE=/abs/path/DATA_DIR/metadata/training_metadata.jsonl \
MODEL_NAME_OR_PATH=/abs/path/Kimi-Audio-7B \
  sbatch --account=<your_account> --partition=<your_partition> \
         --gres=gpu:1 extract_tokens.sb
```

Result (default `OUTPUT_FILE`):
`DATA_DIR/metadata/training_metadata_with_semantic_codes.jsonl`. This is
the file `train_kimi_lora.sb` / `train_kimi_xlora.sb` consume via
`DATA_PATH=`.

> **Two different "model paths" in this pipeline.** Extraction needs the
> **full** Kimi-Audio-7B distribution (it contains the `whisper-large-v3/`
> sub-folder used as the audio tokenizer), so point `MODEL_NAME_OR_PATH`
> at that. Training (next section) consumes the **trimmed** weights at
> `pretrained_hf/`, where the whisper encoder is replaced by a trainable
> copy. Both paths must be **absolute** — HuggingFace `from_pretrained`
> rejects relative paths starting with `./`.

---

## 4. Training

### 4.1 LoRA (main path)

```bash
DATA_PATH=./DATA_DIR/metadata/training_metadata_with_semantic_codes.jsonl \
PRETRAINED_MODEL_PATH=./output/pretrained_hf \
OUTPUT_DIR=./outputs/lora_demo \
bash train_kimi_lora.sb
```

To run under Slurm just `sbatch` the same script and pass an account /
partition. Optional env vars: `EVAL_DATA`, `MODEL_NAME_OR_PATH`,
`REPORT_TO=wandb` (in which case also export `WANDB_API_KEY`).

### 4.2 xLoRA / MoE (advanced)

xLoRA routes between **5 already-trained emotion experts** at training
time. Train the 5 experts first by running `train_kimi_lora.sb` on each
emotion subset (`sad`, `laugh`, `happy`, `crying`, `angry`), then point
`adapters.json` (copy from `adapters.example.json`) at their checkpoints:

```json
{
    "MoE_sad":    "./outputs/lora_sad/checkpoint-XXXX",
    "MoE_laugh":  "./outputs/lora_laugh/checkpoint-XXXX",
    "MoE_happy":  "./outputs/lora_happy/checkpoint-XXXX",
    "MoE_crying": "./outputs/lora_crying/checkpoint-XXXX",
    "MoE_angry":  "./outputs/lora_angry/checkpoint-XXXX"
}
```

Then train the xLoRA classifier:

```bash
DATA_PATH=./DATA_DIR/metadata/training_metadata_with_semantic_codes.jsonl \
PRETRAINED_MODEL_PATH=./output/pretrained_hf \
XLORA_ADAPTERS=./adapters.json \
OUTPUT_DIR=./outputs/xlora_moe \
bash train_kimi_xlora.sb
```

---

## 5. Export & inference

### 5.1 LoRA: merge + inference

```bash
# Merge LoRA -> inference-ready full checkpoint
bash export_and_setup.sb \
  --base_model ./output/pretrained_hf \
  --adapter_dir ./outputs/lora_demo/checkpoint-1000 \
  --output_dir ./outputs/lora_demo_for_inference

# Single-sample S2ST
python check_sft_infer_GLM.py \
  --model_path ./outputs/lora_demo_for_inference \
  --input_audio ./sample_en.wav \
  --output_audio ./translated.wav \
  --language en          # en -> zh, or zh -> en
```

### 5.2 xLoRA: direct inference (no merge)

```bash
bash eval_xlora_custom.sb \
  --checkpoint  ./outputs/xlora_moe/checkpoint-XXXXX \
  --model_path  ./output/pretrained_hf \
  --input_audio ./sample_zh.wav \
  --output_audio ./translated_xlora.wav \
  --language    zh
```

The wrapper just forwards to `python infer_xlora.py …`; you can call that
directly if you prefer.

---

## 6. Notes

* The base model weights (`moonshotai/Kimi-Audio-7B`) are downloaded from
  HuggingFace Hub on first use; alternatively, point
  `PRETRAINED_MODEL_PATH` at a local copy.
* Output sample rate is fixed at **24 kHz** by the GLM-4-Voice decoder.
* All sampling parameters use the `KimiAudio.generate()` defaults; tweak
  inside the script if you need different settings.
* `WANDB_API_KEY` and other credentials should be supplied via environment
  variables — none of the scripts contain hard-coded keys.

---

## 7. End-to-end example (smoke test)

1 training step, 300 samples — run this to confirm your environment is wired before a full job.
Use **absolute paths** throughout; `from_pretrained` rejects relative ones starting with `./`.

```bash
# 1. Download decoder weights (~500 MB)
./scripts/download_weights.sh

# 2. Extract semantic codes  (needs full Kimi-Audio-7B with whisper-large-v3/)
DATA_DIR=<DATA_DIR> \
METADATA_FILE=<DATA_DIR>/metadata/training_metadata.jsonl \
MODEL_NAME_OR_PATH=<KIMI_AUDIO_7B> \
  bash extract_tokens.sb

# 3. LoRA training — 1 step
DATA_PATH=<DATA_DIR>/metadata/training_metadata_with_semantic_codes.jsonl \
PRETRAINED_MODEL_PATH=<PRETRAINED_HF> \
OUTPUT_DIR=./outputs/smoke_lora \
EXTRA_ARGS="--max_steps 1 --save_steps 1" \
  bash train_kimi_lora.sb

# 4. Merge LoRA into a full inference checkpoint
bash export_and_setup.sb \
  --base_model <PRETRAINED_HF> \
  --adapter_dir ./outputs/smoke_lora/checkpoint-1 \
  --output_dir  ./outputs/smoke_lora_for_inference

# 5. Single-sample inference
python check_sft_infer_GLM.py \
  --model_path  ./outputs/smoke_lora_for_inference \
  --input_audio <DATA_DIR>/en/<emotion>/sample.wav \
  --output_audio ./outputs/translated.wav \
  --language en   # en -> zh
```

Wrap each command in your scheduler / container as needed (e.g. `sbatch`, `srun --gres=gpu:1`,
`docker run --gpus all`, `singularity exec --nv`). The training logic is identical regardless of
environment. For the xLoRA path, substitute `train_kimi_xlora.sb` for step 3 and
`eval_xlora_custom.sb` for step 5 (no merge needed).

### Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `HFValidationError: Repo id must be in the form 'repo_name'...` | Path starts with `./` | Use **absolute** paths everywhere. |
| `OSError: Errno 28 No space left on device` mid-training | `~/.cache/` filling a small `$HOME` quota | Set `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `HF_HOME`, `NUMBA_CACHE_DIR` to a writable scratch dir (see §2). |
| `'./output/pretrained_hf/whisper-large-v3'` not found | Used trimmed `pretrained_hf` for extraction | For **extraction** use full `Kimi-Audio-7B/`; for **training** use `pretrained_hf/`. |
| `numba RuntimeError: cannot cache function` | Numba can't write its JIT cache | Set `NUMBA_CACHE_DIR=$CACHE/numba`. |
| `huggingface-cli: command not found` | Hub ≥1.x renamed the CLI | Use `hf` or `python -m huggingface_hub`. |

---

## 8. License

This release is **multi-licensed** because it bundles upstream projects
and extends third-party model weights. The short version:

| Component | License | Notes |
|---|---|---|
| Original code in this repo (training / inference / glue) | **Apache 2.0** (see [LICENSE](LICENSE)) | The `.py`, `.sb`, `.sh`, `Dockerfile`, `README.md` etc. that are not vendored from elsewhere. |
| `glm_4_voice_decoder/` weights (downloaded by `scripts/download_weights.sh`) | **glm-4-voice License** | Fine-tuned from THUDM/GLM-4-Voice. Academic use free; commercial use requires registration at <https://open.bigmodel.cn/mla/form>. Distributed via [`47z/glm-4-voice-decoder-emo-ft`](https://huggingface.co/47z/glm-4-voice-decoder-emo-ft). |
| `glm_4_voice_decoder/` source files (config + flow_inference) | glm-4-voice License | See [`glm_4_voice_decoder/LICENSE`](glm_4_voice_decoder/LICENSE). |
| `kimia_infer/`, `finetune_codes/` | Per upstream MoonshotAI/Kimi-Audio | Vendored from <https://github.com/MoonshotAI/Kimi-Audio>; refer to upstream repo for the current LICENSE. |
| LoRA / xLoRA adapter weights you train | Inherit Kimi-Audio model license | LoRA-only deltas, but they are derivatives of `moonshotai/Kimi-Audio-7B` and inherit its terms. |
| `cosyvoice/` | Per upstream FunAudioLLM/CosyVoice | Transitive dependency of the GLM-4-Voice flow decoder. |
| `matcha/` | Per upstream shivammehta25/Matcha-TTS | Transitive dependency of CosyVoice. HiFiGAN sub-module preserves its own license at [`matcha/hifigan/LICENSE`](matcha/hifigan/LICENSE). |

A full per-component breakdown — including upstream URLs, attribution
requirements, and TODO items for the public release — lives in
[NOTICE](NOTICE).

### What you must do as a redistributor

Per the glm-4-voice License, if you publish anything built on the
fine-tuned decoder weights you **must**:

1. Include the full glm-4-voice license text alongside your distribution
   (already done — see [`glm_4_voice_decoder/LICENSE`](glm_4_voice_decoder/LICENSE)).
2. Display **"Built with glm-4"** prominently on your README, paper,
   product page, or other public-facing surface.
3. Make sure any HuggingFace / GitHub repo name for your derivative
   weights starts with `glm-4`.
4. Register at <https://open.bigmodel.cn/mla/form> before any commercial
   use.

> *Built with glm-4.*

### Disclaimer

This summary is provided for convenience and is **not legal advice**.
Before any public release, verify the upstream LICENSE files
(GLM-4-Voice, Kimi-Audio, CosyVoice, Matcha-TTS) against their current
versions and consult your institution's legal / IP office if there is
any doubt.
