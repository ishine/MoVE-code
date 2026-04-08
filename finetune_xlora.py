# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca and QwenLM/Qwen.


from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional, List
import numpy as np
import random
import torch.distributed as dist

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoTokenizer, set_seed
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset
from finetune_codes.callbacks import WandbAudioGenerationCallback

# xLoRA imports
try:
    import xlora
    from xlora import xLoRAConfig, add_xlora_to_model
except ImportError:
    raise ImportError("Please install xlora via `pip install xlora` to use this script.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the external evaluation data (for WandB generation)."},
    )
    eval_count: int = field(
        default=3,
        metadata={
            "help": "Number of samples to split from train set for eval_loss calculation."
        },
    )
    # NEW: from train set, used for WandB generation but STILL included in training set
    train_gen_count: int = field(
        default=3,
        metadata={
            "help": "Number of samples taken from train set for WandB generation (still included in training)."
        },
    )
    # NEW: random sample from train set (otherwise take first K after shuffle/split)
    train_gen_random: bool = field(
        default=True,
        metadata={"help": "Randomly sample train-gen examples from train set."},
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=True)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Helps prevent OOM by offloading to CPU more frequently
    eval_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of eval steps to keep in GPU before moving to CPU"},
    )
    # Disable prediction gathering to stop DDP Timeouts; we only calculate Eval Loss.
    prediction_loss_only: bool = field(
        default=True,
        metadata={"help": "When set to True, only returns the loss."},
    )
    # NEW: Gradient Clipping Argument
    gradient_clipping: str = field(
        default="1.0",
        metadata={"help": "Gradient clipping max norm. Set to 'false' to disable."},
    )

@dataclass
class XLoraArguments:
    xlora_hidden_size: int = field(
        default=2048,
        metadata={"help": "Hidden size for the X-LoRA classifier."}
    )
    xlora_depth: int = field(
        default=1,
        metadata={"help": "Depth of the X-LoRA classifier."}
    )
    xlora_dropout_p: float = field(
        default=0.2,
        metadata={"help": "Dropout probability for the X-LoRA classifier."}
    )
    xlora_adapters: str = field(
        default=None,
        metadata={
            "help": "JSON string or path to JSON file containing adapter mapping, e.g. {'adapter_1': 'path/to/adapter_1'}"
        }
    )
    xlora_softmax_temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature for the X-LoRA classifier."}
    )
    xlora_layerwise_scalings: bool = field(
        default=False,
        metadata={"help": "Enable layerwise scalings."}
    )
    xlora_top_k_lora: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k LoRA experts to use."}
    )


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def make_supervised_data_module(
    whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset
) -> Dict:
    """
    Creates 3 datasets:
    1) train_dataset: main training data
    2) eval_dataset: subset of training data (size=eval_count) for eval_loss
    3) gen_dataset: merged dataset = (K samples from train_dataset, not removed) + (external eval_data_path, optional)
    """

    dataset_cls = LazySupervisedDataset
    rank = dist.get_rank() if dist.is_initialized() else 0

    train_data = []
    eval_data = []
    gen_data = []  # train_gen + external merged

    if rank == 0:
        print("Loading data...")

        # 1) Load ALL training data
        with open(data_args.data_path, "r") as f:
            lines = f.readlines()
        all_train_data = [json.loads(line) for line in lines]

        # 2) Shuffle once (seed-controlled by set_seed)
        random.shuffle(all_train_data)

        # 3) Split for eval_loss
        if len(all_train_data) > data_args.eval_count:
            eval_data = all_train_data[: data_args.eval_count]
            train_data = all_train_data[data_args.eval_count :]
            print(
                f"Data Split: {len(train_data)} for Training, {len(eval_data)} for Eval Loss."
            )
        else:
            print("Warning: Not enough training data to split. Using all for training.")
            train_data = all_train_data
            eval_data = []

        # 4) Take K samples from train_data for generation (DO NOT remove from train_data)
        k = min(max(int(getattr(data_args, "train_gen_count", 0)), 0), len(train_data))
        if k > 0:
            if getattr(data_args, "train_gen_random", True):
                train_gen_data = random.sample(train_data, k)
            else:
                train_gen_data = train_data[:k]
            gen_data.extend(train_gen_data)
            print(f"Train Gen Samples: {len(train_gen_data)} (also included in Training).")

        # 5) Load external data for generation (optional), append into gen_data
        if data_args.eval_data_path and os.path.exists(data_args.eval_data_path):
            with open(data_args.eval_data_path, "r") as f:
                lines = f.readlines()
            external_data = [json.loads(line) for line in lines]
            gen_data.extend(external_data)
            print(f"External Data Loaded: {len(external_data)} samples for WandB generation.")

        print(f"Total Gen Samples (train_gen + external): {len(gen_data)}")

    # Broadcast to all ranks
    payload = [train_data, eval_data, gen_data]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    train_data, eval_data, gen_data = payload

    def create_dataset(data_source):
        if data_source is not None and len(data_source) > 0:
            return dataset_cls(
                data_source,
                whisper_model=whisper_model,
                text_tokenizer=text_tokenizer,
                max_len=max_len,
                kimia_token_offset=kimia_token_offset,
            )
        return None

    train_dataset = create_dataset(train_data)
    eval_dataset = create_dataset(eval_data)
    gen_dataset = create_dataset(gen_data)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        gen_dataset=gen_dataset,
    )


class KimiTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Buffers for training (Rank 0 only needs to track these for logging approximation)
        self.train_audio_loss_buffer = []
        self.train_text_loss_buffer = []

        # Buffers for evaluation (accumulate for all steps)
        self.eval_audio_loss_sum = 0.0
        self.eval_text_loss_sum = 0.0
        self.eval_steps = 0

    def _save(self, output_dir=None, state_dict=None):
        """Override to fix safetensors shared tensor error in X-LoRA classifier."""
        # Unwrap DeepSpeed to access the actual model
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(unwrapped, "internal_xlora_classifier"):
            for param in unwrapped.internal_xlora_classifier.parameters():
                param.data = param.data.clone().contiguous()
        super()._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            raise ValueError("Labels not found in inputs")

        # FIX: Ensure all input tensors are on the same device as the model
        # This prevents the initial "x" from being on the wrong device
        device = next(model.parameters()).device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        outputs = model(**inputs)
        
        # Loss calculation logic from original compute_loss
        audio_logits, text_logits = outputs.logits
        audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels

        # assert audio_labels.shape[0] == 1, "we only support micro batch size 1 for demo purpose"

        audio_loss = torch.nn.functional.cross_entropy(
            audio_logits.view(-1, audio_logits.shape[-1]),
            audio_labels.view(-1),
            reduction="none",
        )
        text_loss = torch.nn.functional.cross_entropy(
            text_logits.view(-1, text_logits.shape[-1]),
            text_labels.view(-1),
            reduction="none",
        )

        audio_loss_val = (audio_loss * audio_loss_mask.view(-1)).sum() / (
            audio_loss_mask.view(-1).sum() + 1e-4
        )
        text_loss_val = (text_loss * text_loss_mask.view(-1)).sum() / (
            text_loss_mask.view(-1).sum() + 1e-4
        )
        loss = audio_loss_val + text_loss_val

        # --- Logging Logic ---
        # Detach to avoid graph retention
        a_loss = audio_loss_val.detach()
        t_loss = text_loss_val.detach()

        if self.model.training:
            # For training, we just buffer locally. 
            # DDP: Since we log frequent steps, logging just Rank 0's observation or local observation 
            # is standard in HF Trainer for "loss". We will follow suit.
            self.train_audio_loss_buffer.append(a_loss.item())
            self.train_text_loss_buffer.append(t_loss.item())
        else:
            # For eval, we accumulate to average later
            self.eval_audio_loss_sum += a_loss.item()
            self.eval_text_loss_sum += t_loss.item()
            self.eval_steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.
        """
        if self.model.training:
            if len(self.train_audio_loss_buffer) > 0:
                avg_audio = sum(self.train_audio_loss_buffer) / len(self.train_audio_loss_buffer)
                avg_text = sum(self.train_text_loss_buffer) / len(self.train_text_loss_buffer)
                logs["audio_loss"] = avg_audio
                logs["text_loss"] = avg_text
                
                # Clear buffers
                self.train_audio_loss_buffer = []
                self.train_text_loss_buffer = []
        
        if start_time is not None:
             super().log(logs, start_time=start_time)
        else:
             super().log(logs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Reset counters before eval
        self.eval_audio_loss_sum = 0.0
        self.eval_text_loss_sum = 0.0
        self.eval_steps = 0

        # Run standard evaluation loop
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Post-process: Aggregate Custom Metrics across DDP
        total_audio = torch.tensor(self.eval_audio_loss_sum).to(self.args.device)
        total_text = torch.tensor(self.eval_text_loss_sum).to(self.args.device)
        total_steps = torch.tensor(self.eval_steps).to(self.args.device)

        if dist.is_initialized():
            dist.all_reduce(total_audio, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_text, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)

        if total_steps.item() > 0:
            avg_audio = total_audio.item() / total_steps.item()
            avg_text = total_text.item() / total_steps.item()
        else:
            avg_audio = 0.0
            avg_text = 0.0

        # Create new metrics dict
        new_metrics = {
            f"{metric_key_prefix}_audio_loss": avg_audio,
            f"{metric_key_prefix}_text_loss": avg_text,
        }
        
        # Log them explicitly (optional, but good for visibility)
        self.log(new_metrics)
        
        # Update return output
        output.update(new_metrics)
        
        return output


# These functions are retained but will be ignored by Trainer if prediction_loss_only=True.
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.array(labels)
    preds = np.array(preds)
    mask = labels != -100
    correct_preds = (preds == labels) & mask
    total_valid = mask.sum()
    if total_valid == 0:
        accuracy = 0.0
    else:
        accuracy = correct_preds.sum() / total_valid
    return {"accuracy": accuracy}


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, XLoraArguments))
    (model_args, data_args, training_args, xlora_args) = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    if getattr(training_args, "deepspeed", None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    # Handle Gradient Clipping
    if training_args.gradient_clipping.lower() == "false":
        training_args.max_grad_norm = 0.0
        if local_rank == 0:
            print("Gradient Clipping Disabled (max_grad_norm=0.0)")
    else:
        try:
            val = float(training_args.gradient_clipping)
            training_args.max_grad_norm = val
            if local_rank == 0:
                print(f"Gradient Clipping Enabled (max_grad_norm={val})")
        except ValueError:
            raise ValueError(f"Invalid value for gradient_clipping: {training_args.gradient_clipping}")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    logger.info("Loading kimi-audio main model")

    if os.path.exists(model_args.model_name_or_path):
        # local path
        cache_path = model_args.model_name_or_path
    else:
        # cache everything if model_path is a model-id
        cache_path = snapshot_download(model_args.model_name_or_path)

    logger.info(f"Looking for resources in {cache_path}")
    # check if model_path exists
    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")
    model = KimiAudioModel.from_pretrained(model_args.model_path, 
                                           device_map=None,
                                           **model_load_kwargs)

    # === X-LoRA Integation ===
    if xlora_args.xlora_adapters:
        # Parse adapters argument
        arg_adapters = xlora_args.xlora_adapters.strip()
        if arg_adapters.startswith("{") and arg_adapters.endswith("}"):
             adapters_dict = json.loads(arg_adapters)
        elif os.path.isfile(arg_adapters):
            with open(arg_adapters, 'r') as f:
                adapters_dict = json.load(f)
        else:
             raise ValueError("xlora_adapters must be a JSON string or a path to a JSON file.")
        
        # Configure X-LoRA
        xlora_config = xLoRAConfig(
            hidden_size=model.config.hidden_size,
            base_model_id=model_args.model_name_or_path,
            device=torch.device("cuda"),
            xlora_depth=xlora_args.xlora_depth,
            xlora_size=xlora_args.xlora_hidden_size,
            xlora_dropout_p=xlora_args.xlora_dropout_p,
            softmax_temperature=xlora_args.xlora_softmax_temperature,
            layerwise_scalings=xlora_args.xlora_layerwise_scalings,
            top_k_lora=xlora_args.xlora_top_k_lora,
            adapters=adapters_dict,
        )

        model.config.use_cache = False
        logger.info("Adding X-LoRA to model...")
        model = add_xlora_to_model(
            model=model,
            xlora_config=xlora_config,
            adapters=adapters_dict,
            verbose=True
        )

        # Determine device
        target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if local_rank is not None and local_rank != -1:
             target_device = torch.device(f"cuda:{local_rank}")
        
        # Ensure only the classifier is trainable
        model.print_trainable_parameters()
        # IMPORTANT: Instead of just .to(device), tell the model to 
        # re-sync its parameters. For ZeRO-3, this is critical.
        model.config.use_cache = False
        
        # This helps DeepSpeed recognize the new parameters
        if hasattr(model, "register_module"): 
             # Some wrappers need explicit registration
             pass 

        # Ensure the classifier's internal 'model' (the MLP) is also moved
        if hasattr(model, "internal_xlora_classifier"):
            logger.info("Found internal_xlora_classifier! Moving to device...")
            model.internal_xlora_classifier.to(target_device)

        # FIX: Monkey-patch xlora to fix device mismatch with DeepSpeed.
        # DeepSpeed manages device placement and can leave X-LoRA scalings on CPU
        # while model tensors are on CUDA.
        # The original method selects the scaling for a specific adapter via adapter_n,
        # then multiplies: scalings[:, :, adapter_n].unsqueeze(-1) * x
        import xlora.xlora_insertion as _xlora_ins

        def _patched_apply_scalings(self, x, scalings, adapter_n):
            scalings = scalings.to(device=x.device, dtype=x.dtype)
            scalings = scalings[:, :, adapter_n].unsqueeze(-1)
            return x * scalings

        _xlora_ins.xLoRALayer.apply_scalings_to_x = _patched_apply_scalings
        logger.info("Monkey-patched xlora apply_scalings_to_x to fix device mismatch.")

        # Store X-LoRA forward reference on the base KimiAudioModel.
        # This lets generate() route through the X-LoRA classifier + adapters
        # instead of bypassing them with super().forward().
        # Navigate: PeftModel -> LoraModel -> KimiAudioModel (stop here)
        from finetune_codes.model import KimiAudioModel as _KimiAudioModel
        base_model = model
        while not isinstance(base_model, _KimiAudioModel) and hasattr(base_model, "model"):
            base_model = base_model.model
        base_model._xlora_forward_fn = model.forward
        logger.info(f"Stored X-LoRA forward reference on {type(base_model).__name__} for generate().")

        # DEBUG: Check devices again
        print(f"DEBUG: Root Model device: {model.device}")



    
    else:
        logger.warning("No X-LoRA adapters provided. Training standard model without X-LoRA.")
        # Fallback or error? For now just warning.
    # =========================

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    data_module = make_supervised_data_module(
        whisper_model=model.whisper_model,
        text_tokenizer=text_tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        kimia_token_offset=model.config.kimia_token_offset,
    )

    def is_main_process():
        return not dist.is_initialized() or dist.get_rank() == 0

    callbacks = []

    # Single WandB generation callback using merged gen_dataset
    if (
        is_main_process()
        and "wandb" in training_args.report_to
        and data_module.get("gen_dataset") is not None
    ):
        callbacks.append(
            WandbAudioGenerationCallback(
                eval_dataset=data_module["gen_dataset"],
                text_tokenizer=text_tokenizer,
                model_path=model_args.model_path,
                kimia_token_offset=model.config.kimia_token_offset,
                kimia_text_audiodelaytokens=model.config.kimia_mimo_audiodelaytokens,
                num_samples=len(data_module["gen_dataset"]),
            )
        )

    trainer_args = {
        "train_dataset": data_module["train_dataset"],
        "eval_dataset": data_module["eval_dataset"],
        "data_collator": data_module["train_dataset"].collate_fn if data_module["train_dataset"] else None,
    }

    trainer = KimiTrainer(
        model=model,
        args=training_args,
        # compute_loss_func=compute_loss, # Handled by KimiTrainer now
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
        **trainer_args,
    )

    checkpoint = training_args.resume_from_checkpoint
    if isinstance(checkpoint, str) and checkpoint.lower() == "true":
        # Auto-check if we have any checkpoint folder
        checkpoint = True
        output_dir = training_args.output_dir
        has_checkpoint = False
        if os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                if name.startswith("checkpoint-"):
                    has_checkpoint = True
                    break
        
        if not has_checkpoint:
            print(f"Warning: resume_from_checkpoint=True but no checkpoint found in {output_dir}. Starting from scratch.")
            checkpoint = False

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
