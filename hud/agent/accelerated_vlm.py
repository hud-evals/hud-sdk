"""Accelerated VLM Agent using Hugging Face Accelerate for distributed training."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

from hud.agent.vlm import VLMAgent
from hud.rl.types import ActionSample, Batch
from hud.utils.common import Observation

logger = logging.getLogger(__name__)


class AcceleratedVLMAgent(VLMAgent):
    """VLM Agent that uses Accelerate for distributed training and inference.
    
    Key features:
    - Distributed data parallel training across multiple GPUs
    - Gradient accumulation for larger effective batch sizes
    - Mixed precision training (fp16/bf16) for faster computation
    - Efficient multi-GPU inference
    """
    
    def __init__(
        self,
        # Accelerate config
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",  # "no", "fp16", "bf16"
        # Thread pool for async inference
        inference_threads: int = 1,
        # Parent class args
        **kwargs
    ):
        """Initialize Accelerated VLM Agent.
        
        Args:
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Mixed precision training mode
            inference_threads: Number of threads for inference
            **kwargs: Arguments passed to VLMAgent
        """
        # Initialize accelerator first
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        
        # Let parent class initialize model
        super().__init__(**kwargs)
        
        # Thread pool for async inference
        self.inference_threads = inference_threads
        self._executor = ThreadPoolExecutor(max_workers=inference_threads)
        
        # Track if we've prepared the model
        self._prepared = False
        
    def _setup_model(self):
        """Initialize model with Accelerate support."""
        # First, let parent class set up the model
        super()._setup_model()
        
        # Then prepare with accelerator if not already done
        if not self._prepared and self._model is not None and self._optimizer is not None:
            self._model, self._optimizer = self.accelerator.prepare(
                self._model, self._optimizer
            )
            self._prepared = True
            
            # Log distributed setup info
            if self.accelerator.is_main_process:
                logger.info(f"Accelerate initialized:")
                logger.info(f"  Num processes: {self.accelerator.num_processes}")
                logger.info(f"  Mixed precision: {self.accelerator.mixed_precision}")
                logger.info(f"  Device: {self.accelerator.device}")
                
                if self.accelerator.num_processes > 1:
                    logger.info(f"  Distributed training enabled on {self.accelerator.num_processes} GPUs")
    
    async def sample(self, observation: Observation, verbose: bool = False) -> ActionSample:
        """Generate text with Accelerate optimization.
        
        Uses thread pool to avoid blocking the event loop.
        """
        # Run inference in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self._sample_sync, 
            observation, 
            verbose
        )
    
    def _sample_sync(self, observation: Observation, verbose: bool = False) -> ActionSample:
        """Synchronous sampling that runs in thread pool."""
        timing_info = {}
        total_start = time.time()
        
        self._setup_model()
        
        if not self._tokenizer or not self._model:
            raise RuntimeError("Model not initialized")
        
        # Use accelerator's device
        device = self.accelerator.device
        
        # Format the input
        format_start = time.time()
        prompt = self._format_prompt(observation)
        timing_info['prompt_format_ms'] = (time.time() - format_start) * 1000
        
        # Tokenize
        tokenize_start = time.time()
        if hasattr(self._tokenizer, 'tokenizer'):
            inputs = self._tokenizer(
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
        else:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
        timing_info['tokenize_ms'] = (time.time() - tokenize_start) * 1000
        
        # Move to device
        transfer_start = time.time()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        timing_info['transfer_to_device_ms'] = (time.time() - transfer_start) * 1000
        
        # Generate with automatic mixed precision
        generate_start = time.time()
        with torch.no_grad():
            with self.accelerator.autocast():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
        timing_info['model_generate_ms'] = (time.time() - generate_start) * 1000
        
        # Rest is same as parent class
        decode_start = time.time()
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compute log probabilities
        if outputs.scores:
            log_probs = []
            tokens = []
            
            for i, score in enumerate(outputs.scores):
                token_id = generated_ids[i].item()
                log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                log_probs.append(log_prob)
                tokens.append(self._tokenizer.decode([token_id]))
                
            total_log_prob = sum(log_probs)
        else:
            log_probs = None
            tokens = None
            total_log_prob = None
        timing_info['decode_ms'] = (time.time() - decode_start) * 1000
        
        # Parse actions
        parse_start = time.time()
        done = True
        
        processed_actions = None
        if self.adapter:
            try:
                processed_actions = self.adapter.adapt_list([generated_text])
            except Exception as e:
                logger.warning(f"Failed to adapt actions: {e}")
        timing_info['parse_actions_ms'] = (time.time() - parse_start) * 1000
        
        timing_info['total_ms'] = (time.time() - total_start) * 1000
        
        # Add device info
        device_info = {
            "device": str(device),
            "device_type": device.type,
            "num_processes": self.accelerator.num_processes,
            "process_index": self.accelerator.process_index,
        }
        
        return ActionSample(
            text=generated_text,
            log_probs=log_probs,
            tokens=tokens,
            total_log_prob=total_log_prob,
            actions=processed_actions,
            raw_actions=[generated_text],
            done=done,
            metadata={
                "model": self.model_name,
                "temperature": self.temperature,
                "prompt_length": inputs['input_ids'].shape[1],
                "generated_length": len(generated_ids),
                "timing": timing_info,
                "device": device_info,
                "mixed_precision": str(self.accelerator.mixed_precision),
            }
        )
    
    async def update(self, batch: Batch) -> Dict[str, float]:
        """Perform distributed gradient update with Accelerate.
        
        This is mostly the same as parent's update(), but uses:
        - accelerator.backward() instead of loss.backward()
        - Gathers gradients across processes for stats
        - Uses accelerator's device
        """
        self._setup_model()
        
        import torch
        import torch.nn.functional as F
        
        if not self._tokenizer or not self._model or not self._optimizer:
            raise RuntimeError("Model not initialized")
        
        device = self.accelerator.device
        
        # Prepare batch data
        all_sequences = []
        
        for i, (obs, text) in enumerate(zip(batch.observations, batch.texts)):
            # Format prompt + response
            prompt = self._format_prompt(obs)
            full_text = prompt + text
            all_sequences.append(full_text)
        
        # Tokenize all sequences together with padding
        encoded = self._tokenizer(
            all_sequences,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        
        # Move to device
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        
        # Create labels by masking prompts
        labels = input_ids.clone()
        
        # Mask out prompts in labels
        for i, obs in enumerate(batch.observations):
            prompt = self._format_prompt(obs)
            prompt_ids = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).input_ids
            
            # Mask prompt tokens for this sequence
            prompt_len = prompt_ids.shape[1]
            labels[i, :prompt_len] = -100
        
        # Get advantages tensor
        advantages = torch.tensor(batch.advantages, device=device, dtype=torch.float32)
        
        # Forward pass with mixed precision
        with self.accelerator.autocast():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        # Compute per-token log probs
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probs for generated tokens
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        gather_labels = shift_labels.clone()
        gather_labels[gather_labels == -100] = 0
        
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out non-response tokens
        mask = shift_labels != -100
        token_log_probs = token_log_probs * mask
        
        # Check if this is DAPO or GRPO based on metadata
        algorithm = batch.metadata.get("algorithm", "GRPO") if batch.metadata else "GRPO"
        
        if algorithm == "DAPO":
            # DAPO Implementation
            if batch.metadata and "weights" in batch.metadata:
                weights = torch.tensor(batch.metadata["weights"], device=device)
            else:
                weights = torch.ones(len(batch.texts), device=device)
            
            batch_size, seq_len = token_log_probs.shape
            
            # Expand advantages and weights to token level
            token_advantages = advantages.unsqueeze(1).expand(-1, seq_len) * mask
            token_weights = weights.unsqueeze(1).expand(-1, seq_len) * mask
            
            if batch.old_log_probs is not None:
                old_sequence_log_probs = torch.tensor(batch.old_log_probs, device=device)
                
                num_response_tokens = mask.sum(dim=1)
                old_token_log_probs = old_sequence_log_probs.unsqueeze(1) / num_response_tokens.unsqueeze(1)
                old_token_log_probs = old_token_log_probs.expand(-1, seq_len) * mask
                
                log_ratio = token_log_probs - old_token_log_probs
                ratio = torch.exp(log_ratio)
                
                epsilon_high = batch.metadata.get("clip_epsilon_high", 0.4) if batch.metadata else 0.4
                epsilon_low = batch.metadata.get("clip_epsilon_low", 0.2) if batch.metadata else 0.2
                
                clip_high = torch.where(token_advantages > 0, 1 + epsilon_high, 1 + epsilon_low)
                clip_low = torch.where(token_advantages > 0, 1 - epsilon_low, 1 - epsilon_high)
                clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
                
                pg_loss1 = -token_advantages * ratio * token_weights
                pg_loss2 = -token_advantages * clipped_ratio * token_weights
                
                token_pg_loss = torch.max(pg_loss1, pg_loss2)
                pg_loss = token_pg_loss.sum() / mask.sum()
                
                sequence_log_probs = token_log_probs.sum(dim=1) / mask.sum(dim=1)
                sequence_log_ratio = sequence_log_probs - old_sequence_log_probs
                kl_divergence = sequence_log_ratio.mean().item()
                kl_penalty = 0.01 * sequence_log_ratio.mean()
                
                clipped = (ratio != clipped_ratio).float() * mask
                clip_fraction = clipped.sum() / mask.sum()
            else:
                pg_loss = -(token_advantages * token_log_probs * token_weights).sum() / mask.sum()
                kl_penalty = 0.0
                kl_divergence = 0.0
                ratio = torch.ones_like(advantages)
                clip_fraction = torch.tensor(0.0)
            
            token_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
            masked_entropy = (token_entropy * mask).sum() / mask.sum()
            
            entropy_coeff = 0.01
            entropy_bonus = entropy_coeff * masked_entropy
            
            loss = pg_loss + kl_penalty - entropy_bonus
            
        else:
            # GRPO implementation
            sequence_log_probs = token_log_probs.sum(dim=1) / mask.sum(dim=1)
            pg_loss = -(advantages * sequence_log_probs).mean()
            
            if batch.old_log_probs is not None:
                old_log_probs = torch.tensor(batch.old_log_probs, device=device)
                log_ratio = sequence_log_probs - old_log_probs
                kl_divergence = log_ratio.mean().item()
                kl_penalty = 0.1 * log_ratio.mean()
                ratio = torch.exp(log_ratio)
            else:
                kl_penalty = 0.0
                kl_divergence = 0.0
                ratio = torch.ones_like(advantages)
            
            with torch.no_grad():
                entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            
            entropy_coeff = 0.001  
            entropy_bonus = entropy_coeff * entropy
            
            loss = pg_loss + kl_penalty - entropy_bonus
            clip_fraction = torch.tensor(0.0)
        
        # Use Accelerate's backward() for distributed training
        self._optimizer.zero_grad()
        self.accelerator.backward(loss)
        
        # Compute gradient magnitude before optimizer step
        grad_vals = []
        for p in self._model.parameters():
            if p.grad is not None:
                # Gather gradients across all processes for accurate stats
                grad_tensor = p.grad.detach().abs().mean()
                # gather returns a list of tensors from all processes
                gathered_grads = self.accelerator.gather(grad_tensor)
                if self.accelerator.is_main_process:
                    # Average across all processes
                    avg_grad = gathered_grads.mean().item()
                    grad_vals.append(avg_grad)
        
        avg_grad_abs = float(sum(grad_vals) / len(grad_vals)) if grad_vals else 0.0
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        
        # Optimizer step
        self._optimizer.step()
        
        # Compute statistics with debugging
        raw_stats = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl": kl_divergence,
            "approx_kl": kl_divergence,
            "entropy": entropy.item() if algorithm == "GRPO" else masked_entropy.item(),
            "entropy_bonus": entropy_bonus.item(),
            "avg_advantage": advantages.mean().item(),
            "avg_log_prob": sequence_log_probs.mean().item() if algorithm == "GRPO" else (token_log_probs.sum() / mask.sum()).item(),
            "avg_ratio": ratio.mean().item() if isinstance(ratio, torch.Tensor) else 1.0,
            "avg_grad_abs": avg_grad_abs,
            "clip_fraction": clip_fraction.item() if isinstance(clip_fraction, torch.Tensor) else 0.0,
            "algorithm": algorithm,
        }
        
        # Debug: Log types and values of all stats
        logger.info("=== DEBUG: Stats types and values ===")
        stats = {}
        for k, v in raw_stats.items():
            logger.info(f"  {k}: {v} (type: {type(v)}, value: {v})")
            
            # Convert to float if it's a tensor or other numeric type
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    stats[k] = float(v.item())
                    logger.info(f"    Converted tensor to float: {stats[k]}")
                else:
                    logger.error(f"    ERROR: Tensor {k} has {v.numel()} elements, expected 1")
                    stats[k] = 0.0
            elif isinstance(v, (int, float)):
                stats[k] = float(v)
            elif isinstance(v, str):
                stats[k] = v  # Keep strings as-is
            elif hasattr(v, 'item'):
                stats[k] = float(v.item())
                logger.info(f"    Converted {type(v)} to float using .item(): {stats[k]}")
            else:
                logger.error(f"    ERROR: Unknown type for {k}: {type(v)}")
                stats[k] = 0.0
                
        logger.info("=== END DEBUG ===")
        
        # Verify all values are numeric (except strings)
        for k, v in stats.items():
            if k != "algorithm" and not isinstance(v, (int, float)):
                logger.error(f"ERROR: Non-numeric value in stats: {k} = {v} (type: {type(v)})")
                stats[k] = 0.0
        
        # Add DAPO-specific stats with debugging
        if algorithm == "DAPO":
            dapo_stats = {
                "avg_weight": weights.mean().item(),
                "weight_std": weights.std().item(),
            }
            logger.info("=== DEBUG: DAPO Stats ===")
            for k, v in dapo_stats.items():
                logger.info(f"  {k}: {v} (type: {type(v)})")
                stats[k] = float(v) if isinstance(v, (int, float, torch.Tensor)) else 0.0
        
        # Add accelerate-specific stats with debugging
        accel_stats = {
            "num_processes": self.accelerator.num_processes,
            "mixed_precision": str(self.accelerator.mixed_precision),
        }
        logger.info("=== DEBUG: Accelerate Stats ===")
        for k, v in accel_stats.items():
            logger.info(f"  {k}: {v} (type: {type(v)})")
            if isinstance(v, str):
                stats[k] = v
            elif isinstance(v, (int, float)):
                stats[k] = float(v)
            else:
                logger.error(f"  ERROR: Unknown accelerate stat type: {k} = {v} (type: {type(v)})")
                stats[k] = 0.0
                
        # Final validation of all stats
        logger.info("=== DEBUG: Final Stats Validation ===")
        for k, v in stats.items():
            logger.info(f"  {k}: {v} (type: {type(v)})")
            if k not in ["algorithm", "mixed_precision"] and not isinstance(v, (int, float)):
                logger.error(f"  FINAL ERROR: Non-numeric value in final stats: {k} = {v} (type: {type(v)})")
                stats[k] = 0.0
        logger.info("=== END FINAL DEBUG ===")
        
        return stats
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint with Accelerate."""
        # Ensure model is initialized before saving
        if not hasattr(self, '_model') or self._model is None:
            logger.warning("Model not initialized, cannot save checkpoint")
            return
            
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self._model)
            unwrapped_model.save_pretrained(path)
            if self._tokenizer:
                self._tokenizer.save_pretrained(path)
            logger.info(f"Saved checkpoint to {path}")
    
    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False) 