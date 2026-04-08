import os
import argparse
from typing import Optional, List
import shutil
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from kimia_infer.utils.sampler import KimiASampler
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from .modeling_kimia import MoonshotKimiaForCausalLM


class KimiAudioModel(MoonshotKimiaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.whisper_model = WhisperEncoder("openai/whisper-large-v3", mel_batch_size=20, unfreeze_online_whisper_model=True)
        try:
             # 這裡假設 tokenizer 可以在預設路徑找到，或者我們使用一個臨時的 tokenizer
             # 為了避免在 init 裡加載 tokenizer，我們會在 generate 中動態處理，或者使用 hardcode
             # 根據 callbacks.py，kimia_token_offset = 152064
             # 我們先暫存 config 裡的設定
             pass
        except:
             pass

    @classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs, custom_whisper_path=None):
        print(f"DEBUG [init_from_pretrained]: model_name_or_path={model_name_or_path}")
        print(f"DEBUG [init_from_pretrained]: custom_whisper_path={custom_whisper_path}")
        
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
            print(f"DEBUG [init_from_pretrained]: Local path detected, cache_path={cache_path}")
        else:
            # cache everything if model_path is a model-id
            print(f"DEBUG [init_from_pretrained]: Model ID detected, downloading snapshot...")
            cache_path = snapshot_download(model_name_or_path)
            print(f"DEBUG [init_from_pretrained]: Download complete, cache_path={cache_path}")

        print("DEBUG [init_from_pretrained]: Loading AutoModelForCausalLM...")
        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )

        whisper_path_to_load = custom_whisper_path if custom_whisper_path else os.path.join(cache_path, "whisper-large-v3")
        print(f"DEBUG [init_from_pretrained]: Whisper path to load configured as: {whisper_path_to_load}")
        
        print("DEBUG [init_from_pretrained]: Initializing WhisperEncoder...")
        whisper_model = WhisperEncoder(
            whisper_path_to_load, mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        
        print("DEBUG [init_from_pretrained]: Initializing KimiAudioModel...")
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        print("DEBUG [init_from_pretrained]: Merging audio model and whisper model's state dict...")
        pretrained_state_dict = audio_model.state_dict()
        
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        print("DEBUG [init_from_pretrained]: Loading merged state dict into KimiAudioModel...")
        kimia_model.load_state_dict(pretrained_state_dict)

        print("DEBUG [init_from_pretrained]: Initialization from pretrained completed successfully.")
        return kimia_model
    
    @staticmethod
    def export_model(input_dir, output_dir):
        print("Loading model from {}".format(input_dir))
        kimiaudio = KimiAudioModel.from_pretrained(input_dir)

        print("Saving Kimi-Audio LM to {}".format(output_dir))
        audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
        audio_model_state_dict = {k: v for k, v in kimiaudio.state_dict().items() if not k.startswith("whisper_model")}
        audio_model.load_state_dict(audio_model_state_dict)

        audio_model.save_pretrained(output_dir)

        shutil.copyfile("finetune_codes/configuration_moonshot_kimia.py", os.path.join(output_dir, "configuration_moonshot_kimia.py"))
        shutil.copyfile("finetune_codes/modeling_kimia.py", os.path.join(output_dir, "modeling_moonshot_kimia.py"))

        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel

        whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")

        kimiaudio_whisper_encoder_state_dict = {k.replace("speech_encoder.", "encoder."): v for k, v in kimiaudio.whisper_model.state_dict().items() if k.startswith("speech_encoder")}

        missing_keys, unexpected_keys = whisper_model.load_state_dict(kimiaudio_whisper_encoder_state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            assert k.startswith("decoder"), f"Missing keys: {k}"

        whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

        print("Exported Kimi-Audio LM and Whisper model to {}".format(output_dir))


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if whisper_input_feature is not None:
            if isinstance(whisper_input_feature, torch.Tensor):
                # Already processed (e.g., from generate() with X-LoRA)
                whisper_feats = whisper_input_feature
            else:
                whisper_input_feats = torch.from_numpy(whisper_input_feature[0]).unsqueeze(0)[:, :].to(torch.cuda.current_device())
                whisper_feats = self.whisper_model(whisper_input_feats)
                whisper_feats = whisper_feats.reshape(
                    whisper_feats.shape[0],
                    int(whisper_feats.shape[1] // 4),
                    whisper_feats.shape[2] * 4,
                )
        else:
            whisper_feats = None

        return super().forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        text_input_ids,
        whisper_input_feature,
        eod_ids,
        text_eos,
        extra_token,
        is_continuous_mask=None,
        max_new_tokens=2048,
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.12,
        audio_repetition_window_size=5,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        output_type="both",
    ):
        device = next(self.parameters()).device
        
        # === CONFIGURATION ===
        ENABLE_DEBUG_PRINT = False 
        TARGET_EOD_INDEX = 1 
        
        # [UPDATED] SCALING FACTOR (Multiplication)
        # 2.0 = Double the value, 0.5 = Halve the value
        TARGET_LOGIT_SCALE = 2.0 
        # =====================

        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        whisper_input_feats = torch.from_numpy(whisper_input_feature[0]).unsqueeze(0)[:, :].to(device)
        whisper_feats = self.whisper_model(whisper_input_feats)
        whisper_feats = whisper_feats.reshape(
            whisper_feats.shape[0],
            int(whisper_feats.shape[1] // 4),
            whisper_feats.shape[2] * 4,
        )

        past_key_values = None
        return_audio_tokens, return_text_tokens = [], []
        text_stream_is_finished = False
        audio_stream_is_finished = False
        kimia_text_audiodelaytokens = getattr(self.config, "kimia_mimo_audiodelaytokens", 6)
        
        previous_audio_tokens = torch.zeros((4096,), dtype=torch.int, device=device)
        text_previous_tokens = torch.zeros((4096,), dtype=torch.int, device=device)
        
        decoder_input_audio_ids = input_ids.clone().to(device)
        decoder_input_text_ids = text_input_ids.clone().to(device)
        decoder_is_continuous_mask = is_continuous_mask.clone().to(device)
        decoder_input_whisper_feature = whisper_feats
        
        last_position_id = decoder_input_audio_ids.shape[1] - 1
        decoder_position_ids = torch.arange(
            0, decoder_input_audio_ids.shape[1], device=device
        ).unsqueeze(0).long()

        if ENABLE_DEBUG_PRINT:
            target_id = eod_ids[TARGET_EOD_INDEX]
            print(f"\n--- Starting Gen (Scaling T{target_id} by x{TARGET_LOGIT_SCALE}) ---")

        # Use X-LoRA-aware forward if available, otherwise use base forward
        _forward_fn = getattr(self, '_xlora_forward_fn', None)

        for i in range(max_new_tokens):
            forward_kwargs = dict(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            if _forward_fn is not None:
                outputs = _forward_fn(**forward_kwargs)
            else:
                outputs = super().forward(**forward_kwargs)

            audio_logits, text_logits = outputs.logits
            past_key_values = outputs.past_key_values

            # =========================================================
            # [STEP 1] APPLY SCALE (Multiplication)
            # =========================================================
            target_token_id = eod_ids[TARGET_EOD_INDEX]
            
            # MULTIPLY the logit by the scale variable
            audio_logits[:, -1, target_token_id] *= TARGET_LOGIT_SCALE
            # =========================================================

            # =========================================================
            # [STEP 2] DEBUGGING (Reflects the scaled values)
            # =========================================================
            if ENABLE_DEBUG_PRINT:
                current_logits = audio_logits[0, -1, :] 
                probs = torch.nn.functional.softmax(current_logits, dim=-1)
                
                # Max Info
                max_prob, max_idx = torch.max(probs, dim=-1)
                max_logit = current_logits[max_idx].item()
                
                # Target Info
                target_prob = probs[target_token_id]
                target_logit = current_logits[target_token_id]
                rank = (probs > target_prob).sum().item()

                print(f"[Step {i}] Max: {max_idx.item()} (P={max_prob.item():.4f} L={max_logit:.2f}) | "
                      f"T{target_token_id}: L={target_logit.item():.2f} "
                      f"P={target_prob.item():.6f} R={rank}")
            # =========================================================

            # === Sampling ===
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            # === Logic Handling ===
            if text_stream_is_finished:
                next_token_text.fill_(extra_token)
            elif next_token_text.item() == text_eos:
                text_stream_is_finished = True
            
            text_previous_tokens[i : i + 1] = next_token_text

            if i < kimia_text_audiodelaytokens:
                next_audio_token.fill_(extra_token)
            else:
                if output_type == "text":
                    next_audio_token.fill_(extra_token)
            
            previous_audio_tokens[i : i + 1] = next_audio_token

            if next_audio_token.item() in eod_ids:
                audio_stream_is_finished = True

            # Collect results
            return_audio_tokens.append(next_audio_token)
            return_text_tokens.append(next_token_text)

            # === Stopping Criteria ===
            if (output_type == "text" and text_stream_is_finished) or \
               (output_type == "both" and audio_stream_is_finished):
                if ENABLE_DEBUG_PRINT:
                    print(f"--- Generation Finished at step {i} ---")
                break

            # === Prepare Next Step ===
            decoder_input_audio_ids = next_audio_token.unsqueeze(1)
            decoder_input_text_ids = next_token_text.unsqueeze(1)
            
            last_position_id += 1
            decoder_position_ids = torch.tensor([[last_position_id]], device=device, dtype=torch.long)
            
            decoder_input_whisper_feature = None
            decoder_is_continuous_mask = None

        return {
            "audio_tokens": torch.cat(return_audio_tokens, dim=-1).unsqueeze(0),
            "text_tokens": torch.cat(return_text_tokens, dim=-1).unsqueeze(0),
        }


if __name__ == "__main__":
    print("DEBUG: finetune_codes.model script executing in __main__")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "export_model"], default="init_from_pretrained")
    parser.add_argument("--output_dir", type=str, default="output/pretrained_hf")
    parser.add_argument("--input_dir", type=str, default="output/finetuned_hf")
    parser.add_argument("--custom_whisper_path", type=str, default=None, help="Path to replace the default whisper model upon init")
    args = parser.parse_args()

    print(f"DEBUG: Parsed arguments: {args}")

    if args.action == "init_from_pretrained":
        print(f"DEBUG: Proceeding with action: {args.action}")
        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={}, custom_whisper_path=args.custom_whisper_path)

        print(f"DEBUG: Creating output directory: {args.output_dir} (if it doesn't exist)")
        os.makedirs(args.output_dir, exist_ok=True)
        # save model
        print(f"DEBUG: Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        
        # If a custom whisper path was passed in, we overwrite the default whisper-large-v3 that might have been loaded
        if args.custom_whisper_path:
            print(f"Loaded custom Whisper from: {args.custom_whisper_path}. It will be saved into {args.output_dir}/whisper-large-v3")
        print("DEBUG: init_from_pretrained action completed successfully.")
    elif args.action == "export_model":
        print(f"DEBUG: Proceeding with action: {args.action}")
        KimiAudioModel.export_model(args.input_dir, args.output_dir)
    else:
        raise ValueError(f"Invalid action: {args.action}")



        