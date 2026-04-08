import torch

class KimiASampler:
    def __init__(
        self,
        audio_top_k: int,
        audio_temperature: float,
        audio_repetition_penalty: float,
        audio_repetition_window_size: int,
        text_top_k: int,
        text_temperature: float,
        text_repetition_penalty: float,
        text_repetition_window_size: int,
    ):
        self.audio_top_k = audio_top_k
        self.audio_temperature = audio_temperature
        self.text_top_k = text_top_k
        self.text_temperature = text_temperature

        self.audio_repetition_penalty = audio_repetition_penalty
        self.audio_repetition_window_size = audio_repetition_window_size
        self.text_repetition_penalty = text_repetition_penalty
        self.text_repetition_window_size = text_repetition_window_size

        # Define the target token ID
        self.target_token_id = 151663

    def sample_audio_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        """Sample from audio logits with custom weighting for target token."""
        
        # 1. Flatten dimensions: [batch, seq, vocab] -> [batch, vocab]
        if len(logits.shape) == 3:
            logits = logits[:, -1]

        # =========================================================
        # [NEW] Apply 5x Multiplier to Target Token (151663)
        # We do this BEFORE repetition penalty to ensure it's the base logic
        # MODIFIED: Only apply after 10 tokens have been generated
        # =========================================================
        if logits.shape[-1] > self.target_token_id:
            if recent_tokens is not None and len(recent_tokens) >= 10:
                # Multiply the logit by 5
                logits[:, self.target_token_id] *= 60
        # =========================================================

        # Apply repetition penalty if needed
        if (
            self.audio_repetition_penalty > 1.0
            and recent_tokens is not None
            and len(recent_tokens) > self.audio_repetition_window_size
        ):
            logits = logits[0]  # Assumes batch size of 1
            recent_window = recent_tokens[-self.audio_repetition_window_size :].long()

            scores = torch.gather(logits, dim=0, index=recent_window)

            scores = torch.where(
                scores < 0,
                scores * self.audio_repetition_penalty,
                scores / self.audio_repetition_penalty,
            )

            logits.scatter_(dim=0, index=recent_window, src=scores)
            logits = logits.unsqueeze(0)

        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Apply temperature scaling
        if self.audio_temperature > 1e-6:
            logprobs = logprobs / self.audio_temperature

            if self.audio_top_k > 0:
                probs = torch.exp(logprobs)
                top_k_probs, top_k_indices = torch.topk(probs, self.audio_top_k, dim=-1)
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
                next_token = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            else:
                next_token = torch.multinomial(torch.exp(logprobs), num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(logprobs, dim=-1)

        return next_token

    def sample_text_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        # (Kept unchanged unless you want the logic applied here too)
        if len(logits.shape) == 3:
            logits = logits[:, -1]

        if (
            self.text_repetition_penalty > 1.0
            and recent_tokens is not None
            and len(recent_tokens) > self.text_repetition_window_size
        ):
            logits = logits[0]
            recent_window = recent_tokens[-self.text_repetition_window_size :].long()
            scores = torch.gather(logits, dim=0, index=recent_window)
            scores = torch.where(
                scores < 0,
                scores * self.text_repetition_penalty,
                scores / self.text_repetition_penalty,
            )
            logits.scatter_(dim=0, index=recent_window, src=scores)
            logits = logits.unsqueeze(0)

        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        if self.text_temperature > 1e-6:
            logprobs = logprobs / self.text_temperature
            if self.text_top_k > 0:
                probs = torch.exp(logprobs)
                top_k_probs, top_k_indices = torch.topk(probs, self.text_top_k, dim=-1)
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
                next_token = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            else:
                next_token = torch.multinomial(torch.exp(logprobs), num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(logprobs, dim=-1)

        return next_token