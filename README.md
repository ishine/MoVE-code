# Kimi-Audio Emotion-Aware S2ST

LoRA / xLoRA fine-tuning of [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio)
for emotion-preserving Chinese ↔ English speech-to-speech translation.

This repository contains:

* the data-prep / training / export / inference pipeline
* a Dockerfile reproducing the runtime
* two single-sample inference entry points (LoRA-merged and xLoRA-routed)

It does **not** ship model checkpoints. Train your own with the steps below.

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

> **Cache directories.** First-run training will JIT-compile DeepSpeed's
> `fused_adam` (torch C++ extension), JIT-compile Triton kernels, run
> Numba caching for `librosa`, and let HuggingFace cache models. By
> default these all land under `~/.cache/` and `~/.triton/` — if your
> `$HOME` lives on a small quota volume you will hit
> `OSError: [Errno 28] No space left on device` partway through training.
>
> Redirect them to a writable scratch dir before launching:
>
> ```bash
> export CACHE=/path/to/scratch/.cache
> mkdir -p $CACHE/{torch_ext,triton,hf,numba}
> export TORCH_EXTENSIONS_DIR=$CACHE/torch_ext
> export TRITON_CACHE_DIR=$CACHE/triton
> export HF_HOME=$CACHE/hf
> export NUMBA_CACHE_DIR=$CACHE/numba
> ```
>
> If you wrap python in singularity, pass these through with
> `--env HOME=$CACHE --env TORCH_EXTENSIONS_DIR=...` etc.

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

This section walks through the **exact** sequence used to validate this
release on a Slurm + Singularity cluster. It is written as a smoke test —
1 training step, 300 samples, ~10 minutes wall-clock — so you can confirm
your environment works before launching a real training run.

Substitute the placeholders for your cluster:
- `<ACCOUNT>` / `<PARTITION>` — your Slurm account and partition
- `<SIF>` — path to your Singularity image
- `<KIMI_AUDIO_7B>` — absolute path to a full Kimi-Audio-7B distribution
  (must contain `whisper-large-v3/`)
- `<PRETRAINED_HF>` — absolute path to the trimmed `pretrained_hf` weights
  (whisper encoder replaced for training)
- `<DATA_DIR>` — absolute path to your dataset root with `en/`, `zh/`,
  `metadata/training_metadata.jsonl`

### Stage 0 — environment setup

Pull decoder weights from HuggingFace, prepare cache directories on
writable scratch (avoid `/home` quota traps), define a container wrapper.

```bash
cd /path/to/kimi-audio-release/

# Decoder weights (~500 MB total) -> glm_4_voice_decoder/
./scripts/download_weights.sh

# Cache directories on writable scratch
export CACHE=$PWD/.cache
mkdir -p $CACHE/{torch_ext,triton,hf,numba}

# Container wrapper (Singularity example).
# All training/inference scripts respect $CONTAINER_CMD; setting it to ""
# runs python directly on the host.
export CONTAINER_CMD="singularity exec --nv \
  -B $PWD:$PWD \
  --env HOME=$CACHE \
  --env TORCH_EXTENSIONS_DIR=$CACHE/torch_ext \
  --env TRITON_CACHE_DIR=$CACHE/triton \
  --env HF_HOME=$CACHE/hf \
  --env NUMBA_CACHE_DIR=$CACHE/numba \
  --pwd $PWD \
  <SIF>"
```

### Stage 1 — dataset layout

```
<DATA_DIR>/
├── en/<emotion>/<id>_en.wav
├── zh/<emotion>/<id>_zh.wav
└── metadata/
    └── training_metadata.jsonl    # one JSON object per line, task_type "s-s"
```

See [finetune_codes/README.md](finetune_codes/README.md) for the exact
JSON shape.

### Stage 2 — extract semantic codes

Tokenises every wav referenced by `training_metadata.jsonl` once.
For 300 samples this takes ~12 seconds on one H100.

```bash
srun --account=<ACCOUNT> --partition=<PARTITION> \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=4 --time=00:30:00 --pty \
  bash -c '
    export DATA_DIR=<DATA_DIR>
    export METADATA_FILE=<DATA_DIR>/metadata/training_metadata.jsonl
    export MODEL_NAME_OR_PATH=<KIMI_AUDIO_7B>
    ./extract_tokens.sb
  '

ls -la <DATA_DIR>/metadata/training_metadata_with_semantic_codes.jsonl
```

### Stage 3 — LoRA training (1 step)

`EXTRA_ARGS` is a passthrough into `finetune_lora.py`, here used to cap
the run at 1 step for the smoke test. Drop it for a real run.

```bash
srun --account=<ACCOUNT> --partition=<PARTITION> \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=4 --time=00:30:00 --pty \
  bash -c '
    export DATA_PATH=<DATA_DIR>/metadata/training_metadata_with_semantic_codes.jsonl
    export PRETRAINED_MODEL_PATH=<PRETRAINED_HF>
    export MODEL_NAME_OR_PATH=<KIMI_AUDIO_7B>
    export OUTPUT_DIR=./outputs/smoke_lora
    export NPROC_PER_NODE=1
    export EXTRA_ARGS="--max_steps 1 --save_steps 1 --num_train_epochs 1"
    bash train_kimi_lora.sb
  '

ls outputs/smoke_lora/checkpoint-1/   # adapter_config.json + adapter_model.safetensors
```

> First-run notes: DeepSpeed JIT-compiles `fused_adam` (~25 s),
> Numba caches librosa, HuggingFace caches the base model.
> All of these go under `$CACHE/...` thanks to Stage 0.

### Stage 4 — merge LoRA into a full inference checkpoint

```bash
srun --account=<ACCOUNT> --partition=<PARTITION> \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=4 --time=00:30:00 --pty \
  bash -c '
    bash export_and_setup.sb \
      --base_model <PRETRAINED_HF> \
      --adapter_dir $PWD/outputs/smoke_lora/checkpoint-1 \
      --output_dir  $PWD/outputs/smoke_lora_for_inference
  '

ls outputs/smoke_lora_for_inference/   # config.json + model-00001-of-NNNNN.safetensors
```

### Stage 5 — single-sample S2ST inference

```bash
# Pick any en wav from your dataset
mkdir -p demo_audio outputs
cp "$(find -L <DATA_DIR>/en -name '*.wav' | head -1)" demo_audio/sample_en.wav

srun --account=<ACCOUNT> --partition=<PARTITION> \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=4 --time=00:30:00 --pty \
  bash -c '
    $CONTAINER_CMD python check_sft_infer_GLM.py \
      --model_path  $PWD/outputs/smoke_lora_for_inference \
      --input_audio $PWD/demo_audio/sample_en.wav \
      --output_audio $PWD/outputs/translated_en2zh.wav \
      --language en
  '

file outputs/translated_en2zh.wav    # RIFF WAVE, 24000 Hz, mono
```

If you reach this point with a non-empty 24 kHz wav, the LoRA path is
fully wired end-to-end. The xLoRA path (Stage 3' + 5' below) follows the
same shape — just substitute `train_kimi_xlora.sb` for training and
`eval_xlora_custom.sb` for inference, no merge step needed.

### Pitfalls we've actually hit (so you don't have to)

| Symptom | Cause | Fix |
|---|---|---|
| `HFValidationError: Repo id must be in the form 'repo_name'...` from `from_pretrained` | Path starts with `./` | Use **absolute** paths everywhere. |
| `OSError: Errno 28 No space left on device` mid-training | `~/.cache/`, `~/.triton/`, `~/.cache/torch_extensions/` filling a small `$HOME` quota | Set `CACHE`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `HF_HOME`, `NUMBA_CACHE_DIR` as in Stage 0. |
| `Repo id must be in the form... './output/pretrained_hf/whisper-large-v3'` | Pointed `MODEL_NAME_OR_PATH` at the trimmed `pretrained_hf` (no whisper sub-folder) | For **extraction**, point at the full `Kimi-Audio-7B/`. For **training**, point at `pretrained_hf/`. They are not interchangeable. |
| `librosa` import → `numba RuntimeError: cannot cache function ... no locator available` | Numba can't write its JIT cache (read-only site-packages) | Set `NUMBA_CACHE_DIR=$CACHE/numba`. |
| `huggingface-cli: command not found` after `pip install huggingface_hub` | Hub ≥1.x renamed the CLI | Use `hf` instead of `huggingface-cli`, or `python -m huggingface_hub`. |

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
