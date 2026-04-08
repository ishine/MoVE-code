import torch
import wandb
from transformers import TrainerCallback, TrainerState
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from glm_4_voice_decoder.flow_inference import AudioDecoder
import os
import uuid
import torch.distributed as dist

TOKEN_OFFSET = 152064

class WandbAudioGenerationCallback(TrainerCallback):
    def __init__(self, eval_dataset, text_tokenizer, model_path: str, kimia_token_offset: int, kimia_text_audiodelaytokens: int, sample_rate: int = 24000, num_samples: int = 3):
        super().__init__()

        # === Initialize GPU ===
        if torch.cuda.is_available():
            if dist.is_initialized():
                local_rank = dist.get_rank()
            else:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.audio_decoder = None
        self.sample_rate = sample_rate
        # Ensure we don't try to access more samples than exist
        self.num_samples = min(num_samples, len(eval_dataset))
        self.text_tokenizer = text_tokenizer
        self.kimia_token_offset = kimia_token_offset
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.logged_targets = False
        self.samples = [eval_dataset[i] for i in range(self.num_samples)]

    def _get_audio_decoder(self):
        """ Lazy initialization of AudioDecoder to save memory until needed """
        if self.audio_decoder is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            flow_path = os.path.join(base_dir, "glm_4_voice_decoder")
            flow_config = os.path.join(flow_path, "config.yaml")
            flow_checkpoint = os.path.join(flow_path, "epoch500_emoft.pt")
            hift_checkpoint = os.path.join(flow_path, "hift.pt")
            
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] Loading AudioDecoder...")
            
            self.audio_decoder = AudioDecoder(
                config_path=flow_config,
                flow_ckpt_path=flow_checkpoint,
                hift_ckpt_path=hift_checkpoint,
                device=self.device
            )
        return self.audio_decoder

    def _is_main_proc(self):
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    def decode_to_text(self, tokens):
        if torch.is_tensor(tokens):
            tokens = tokens.view(-1).tolist()
        text = self.text_tokenizer.decode([t for t in tokens if t < TOKEN_OFFSET])
        return text

    def _log_target_audio_once(self):
        """ Uploads Ground Truth (Target) audio only once. """
        if self.logged_targets or not self._is_main_proc():
            return
        
        print("[Rank 0] Logging Ground Truth Audio...")
        decoder = self._get_audio_decoder()
        kimia_token = self.extra_tokens.kimia_assistant_msg_start
        
        # [FIX] Define custom metric here too just in case
        wandb.define_metric("audio_step")
        wandb.define_metric("Ground_Truth/*", step_metric="audio_step")
        
        log_dict = {"audio_step": 0}

        for idx, sample in enumerate(self.samples):
            input_ids = sample["input_ids"]
            text_input_ids = sample["text_input_ids"]

            # Find start of Assistant response
            pos = (input_ids == kimia_token).nonzero(as_tuple=True)
            if len(pos[0]) == 0: continue
            
            split_idx = pos[1][0].item()
            assistant_text_ids = text_input_ids[:, split_idx + 1:]
            assistant_audio_ids = input_ids[:, split_idx + 1:]

            tokens = assistant_audio_ids.view(-1).tolist()
            gen_wav_tokens = [t for t in tokens if t >= TOKEN_OFFSET]
            if not gen_wav_tokens: continue

            gen_wav_tokens = torch.tensor(gen_wav_tokens).unsqueeze(0).to(self.device) - TOKEN_OFFSET

            with torch.no_grad():
                reconstructed_audio, _ = decoder.token2wav(
                    gen_wav_tokens,
                    uuid=str(uuid.uuid4()),
                    prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                    prompt_feat=torch.zeros(1, 0, 80).to(self.device),
                    finalize=True
                )

            audio_np = reconstructed_audio.squeeze().cpu().numpy()
            target_text = self.decode_to_text(assistant_text_ids)
            
            wandb_audio = wandb.Audio(
                audio_np, 
                sample_rate=self.sample_rate, 
                caption=f"[Target] {target_text[:50]}..."
            )
            log_dict[f"Ground_Truth/Sample_{idx+1}"] = wandb_audio

        if len(log_dict) > 1: # More than just 'audio_step'
            wandb.log(log_dict)
            self.logged_targets = True

    def on_train_begin(self, args, state, control, model, **kwargs):
        """
        Run when training starts. We define the metric here to avoid the 'step' error.
        """
        if self._is_main_proc():
            # [FIX 1] Define a custom X-axis metric
            # This tells WandB: "For any chart named 'Checkpoints/...', 
            # use 'audio_step' as the X-axis, not the internal counter."
            wandb.define_metric("audio_step") 
            wandb.define_metric("Checkpoints/*", step_metric="audio_step")

            # Upload Ground Truth
            self._log_target_audio_once()
            
            # Run Step 0 evaluation
            print("[Rank 0] Generating baseline audio at Step 0...")
            self._generate_and_log_audio(model, state)

    def on_save(self, args, state, control, model, **kwargs):
        """ Run generation whenever a checkpoint is saved """
        self._generate_and_log_audio(model, state)

    def _generate_and_log_audio(self, model, state: TrainerState):
        if not self._is_main_proc():
            return

        step_id = int(state.global_step)
        print(f"\n[Rank 0] >>> Entering Audio Generation at Step {step_id} <<<")

        # [FIX 2] Unwrap DeepSpeed model to access .generate()
        if hasattr(model, "module"):
            inference_model = model.module
        else:
            inference_model = model

        # [FIX X-LoRA] Unwrap to KimiAudioModel to use its custom generate().
        # KimiAudioModel.generate() now checks for _xlora_forward_fn (set by
        # finetune_xlora.py) and routes each forward step through the X-LoRA
        # classifier + adapters. This matches the actual inference pipeline.
        from finetune_codes.model import KimiAudioModel as _KimiAudioModel
        gen_model = inference_model
        while not isinstance(gen_model, _KimiAudioModel) and hasattr(gen_model, "model"):
            gen_model = gen_model.model
        xlora_status = "with X-LoRA" if hasattr(gen_model, "_xlora_forward_fn") else "without X-LoRA"
        print(f"[Rank 0] Unwrapped to {type(gen_model).__name__} for generation ({xlora_status}).")

        inference_model.eval()
        
        decoder = self._get_audio_decoder()
        kimia_token = self.extra_tokens.kimia_assistant_msg_start
        
        # [FIX 3] Add the custom step to the dictionary
        log_dict = {
            "audio_step": step_id 
        }
        
        has_valid_audio = False

        for idx, sample in enumerate(self.samples):
            input_ids = sample["input_ids"]
            text_input_ids = sample["text_input_ids"]
            is_continuous_mask = sample["is_continuous_mask"]
            whisper_feats = sample.get("whisper_input_feature", None)
            
            pos = (input_ids == kimia_token).nonzero(as_tuple=True)
            if len(pos[0]) == 0: continue
            split_idx = pos[1][0].item()

            user_input_ids = input_ids[:, :split_idx + 1].to(self.device)
            user_text_ids = text_input_ids[:, :split_idx + 1].to(self.device)
            user_mask = is_continuous_mask[:, :split_idx + 1].to(self.device)

            with torch.no_grad():
                try:
                    # Use gen_model (unwrapped KimiAudioModel) for generation
                    gen_out = gen_model.generate(
                        input_ids=user_input_ids,
                        text_input_ids=user_text_ids,
                        whisper_input_feature=whisper_feats,
                        eod_ids=[self.extra_tokens.msg_end, self.extra_tokens.media_end],
                        text_eos=self.extra_tokens.kimia_text_eos,
                        extra_token=self.extra_tokens.kimia_text_blank,
                        is_continuous_mask=user_mask,
                        output_type="both",
                        max_new_tokens=256, 
                        audio_temperature=0.0,
                        text_temperature=0.0,
                    )
                except Exception as e:
                    print(f"   - [WARN] Generation failed: {e}")
                    continue

                # Extract Tokens
                text_tokens = gen_out["text_tokens"].view(-1).tolist()
                audio_tokens_raw = gen_out["audio_tokens"].view(-1).tolist()
                gen_wav_tokens = [t for t in audio_tokens_raw if t >= TOKEN_OFFSET]
                generated_text = self.decode_to_text(gen_out["text_tokens"])
                
                # Debug Print
                print(f"[Step {step_id}] Sample {idx+1}:")
                print(f"   - Text: {generated_text[:50]}...")
                print(f"   - Audio Tokens: {len(gen_wav_tokens)}")

                # Skip if no audio tokens
                if not gen_wav_tokens:
                    print("   - [WARN] No audio tokens generated.")
                    continue
                
                has_valid_audio = True
                
                # Decode audio
                gen_wav_tokens = torch.tensor(gen_wav_tokens).unsqueeze(0).to(self.device) - TOKEN_OFFSET
                
                reconstructed_audio, _ = decoder.token2wav(
                    gen_wav_tokens,
                    uuid=str(uuid.uuid4()),
                    prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                    prompt_feat=torch.zeros(1, 0, 80).to(self.device),
                    finalize=True
                )

                audio_np = reconstructed_audio.squeeze().cpu().numpy()

                # Add to log_dict with unique key
                wandb_audio = wandb.Audio(
                    audio_np, 
                    sample_rate=self.sample_rate, 
                    caption=f"Step {step_id} | {generated_text[:30]}"
                )
                log_dict[f"Checkpoints/Step_{step_id}_Sample_{idx+1}"] = wandb_audio

        if has_valid_audio:
            print(f"[Rank 0] Uploading audio samples to WandB...")
            # [FIX 4] Log WITHOUT the 'step=' argument
            wandb.log(log_dict) 
        else:
            print(f"[Rank 0] No valid audio generated to upload.")

        inference_model.train()