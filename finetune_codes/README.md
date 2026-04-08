# Dataset format (S2ST)

This release fine-tunes Kimi-Audio for **emotion-preserving speech-to-speech
translation** (Chinese ↔ English). Your training data must follow the
**`task_type: "s-s"`** format described below — the upstream Kimi-Audio
ASR / understanding format will not work with `finetune_lora.py` /
`finetune_xlora.py`.

There are two stages: **raw metadata** (you prepare this) and **tokenised
metadata** (produced by `extract_tokens.sb`). Training consumes the
tokenised version.

---

## 1. Directory layout

```
DATA_DIR/
├── en/                                       # source-language audio
│   ├── happy/  happy_00001_en.wav  ...
│   ├── sad/    sad_00001_en.wav    ...
│   ├── angry/  ...
│   ├── crying/ ...
│   └── laugh/  ...
├── zh/                                       # paired target-language audio
│   ├── happy/  happy_00001_zh.wav  ...
│   ├── sad/    ...
│   └── ...
└── metadata/
    ├── training_metadata.jsonl                       # raw, you prepare this
    └── training_metadata_with_semantic_codes.jsonl   # produced by extract_tokens.sb
```

* The wav paths inside `training_metadata.jsonl` are **relative to
  `DATA_DIR`**.
* Audio is paired (`en/<emotion>/<id>_en.wav` ↔ `zh/<emotion>/<id>_zh.wav`)
  so each utterance can produce **two** training samples (en→zh and zh→en).
* No sharding — the whole dataset lives in one jsonl file. Tokenisation
  is also a single pass.

---

## 2. Raw format (`training_metadata.jsonl`)

Each line is a single JSON object representing **one direction** of one
sample. A paired utterance therefore appears on two lines (one en→zh, one
zh→en).

```json
{
  "task_type": "s-s",
  "conversation": [
    {
      "role": "user",
      "message_type": "text",
      "content": "Translate the given English speech into Chinese while preserving its expressiveness."
    },
    {
      "role": "user",
      "message_type": "audio",
      "content": "en/happy/happy_09851_en.wav"
    },
    {
      "role": "assistant",
      "message_type": "audio-text",
      "content": [
        "zh/happy/happy_09851_zh.wav",
        "你的意志会推动你度过痛苦，"
      ]
    }
  ]
}
```

Notes
* `task_type` **must** be `"s-s"`.
* `audio` paths are **relative to `DATA_DIR`**.
* The assistant message is `audio-text` with `content = [target_wav,
  target_transcript]`.
* For the reverse direction, swap the instruction and the en/zh paths:
  ```text
  "Translate the given Chinese speech into English while preserving its expressiveness."
  ```

---

## 3. Tokenised format (`training_metadata_with_semantic_codes.jsonl`)

Produced by `extract_tokens.sb`. Each line is the **same JSON object as
above, plus** an `audio_tokens` field on every audio-bearing message:

```json
{
  "task_type": "s-s",
  "conversation": [
    { "role": "user", "message_type": "text",
      "content": "Translate the given English speech into Chinese while preserving its expressiveness." },
    { "role": "user", "message_type": "audio",
      "content": "/abs/path/DATA_DIR/en/sad/sad_01576_en.wav",
      "audio_tokens": [155135, 157825, 161564, ...] },
    { "role": "assistant", "message_type": "audio-text",
      "content": ["/abs/path/DATA_DIR/zh/sad/sad_01576_zh.wav",
                  "他们也有一种光明，而不是思想的结果..."],
      "audio_tokens": [166717, 162806, 155717, ...] }
  ]
}
```

Notes
* `audio_tokens` is a flat list of int semantic-code IDs (typically in the
  range `152064 + [0, 16384)`, decoded by the GLM-4-Voice tokenizer).
  These are the **generation targets** (what the assistant decoder is
  trained to predict).
* `content` paths in the tokenised version are **absolute** because the
  extraction script resolves them against `DATA_DIR` at process time.
* **Training still loads the wav files at runtime.** The semantic codes
  are only the generation target; the encoder side reads each wav with
  `librosa.load(..., sr=16000)` to extract Whisper features. Make sure
  every absolute path in the jsonl is readable on the training node.
* Duplicate audio paths within the input are automatically skipped.

---

## 4. Producing the tokenised data

```bash
DATA_DIR=/abs/path/DATA_DIR \
METADATA_FILE=/abs/path/DATA_DIR/metadata/training_metadata.jsonl \
MODEL_NAME_OR_PATH=/abs/path/Kimi-Audio-7B \
  ./extract_tokens.sb
```

`MODEL_NAME_OR_PATH` must point at a **full** Kimi-Audio-7B distribution
that contains the `whisper-large-v3/` sub-folder. The trimmed
`pretrained_hf/` used for training does not work for tokenisation.

Default output:
`DATA_DIR/metadata/training_metadata_with_semantic_codes.jsonl`. Override
with `OUTPUT_FILE=...` if you want it elsewhere.

---

## 5. Customising for a different task

`finetune_codes/datasets.py::tokenize_message` is the place to change if
you need a different conversation schema (e.g. text-only instruction,
single-direction translation, ASR, audio captioning). The training
scripts assume `task_type: "s-s"` end-to-end.
