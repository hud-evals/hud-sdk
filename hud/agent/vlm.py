"""VLM Agent with full RL support including sampling and gradient updates."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import json

from hud.agent.base import Agent
from hud.adapters import Adapter, VLMAdapter
from hud.rl.types import ActionSample, Batch
from hud.utils.common import Observation

logger = logging.getLogger(__name__)


class VLMAgent(Agent[None, Dict[str, Any]]):
    """VLM agent that supports both inference and gradient updates for RL training.
    
    This agent:
    - Loads and manages the actual model (with LoRA if specified)
    - Implements sample() to generate text with log probabilities
    - Implements update() to perform gradient updates on batches
    - Uses adapters to parse text into executable actions
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        adapter: Optional[Adapter] = None,
        name: Optional[str] = None,
        # Model configuration
        device_map: str = "auto",
        load_in_8bit: bool = True,
        # LoRA configuration
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        # Training configuration
        learning_rate: float = 1e-5,
        # Generation configuration
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        # System prompt
        system_prompt: Optional[str] = None,
    ):
        """Initialize VLM agent with model and training configuration.
        
        Args:
            model_name: HuggingFace model name or local path
            adapter: Adapter for parsing text to actions
            name: Agent name
            device_map: Device placement strategy
            load_in_8bit: Use 8-bit quantization
            use_lora: Enable LoRA adapters
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            lora_target_modules: Modules to apply LoRA to (auto-detect if None)
            learning_rate: Learning rate for updates
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt for the model
        """
        super().__init__(client=None, adapter=adapter, name=name)
        
        # Use VLMAdapter as default if no adapter provided
        if self.adapter is None:
            self.adapter = VLMAdapter()
        
        self.model_name = model_name
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.learning_rate = learning_rate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Lazy initialization
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._device = None
        
    def _default_system_prompt(self) -> str:
        """Default system prompt for UI automation."""
        return (
            "You are an AI assistant that helps users interact with computer interfaces. "
            "You can see screenshots and perform actions like clicking, typing, and scrolling.\n\n"
            "Respond with your thoughts and actions in this format:\n"
            "Thought: [Your reasoning about what you see and what to do]\n"
            "Action: [action_type] [parameters]\n\n"
            "Available actions:\n"
            "- click x,y (click at coordinates)\n"
            "- type \"text\" (type the given text)\n"
            "- scroll up/down (scroll in direction)\n"
            "- done \"response\" (task complete with final response)\n"
        )
        
    def _setup_model(self):
        """Initialize model, tokenizer, and optimizer (lazy)."""
        if self._model is not None:
            return
            
        try:
            import torch
            from transformers import AutoTokenizer
            from transformers.utils.quantization_config import BitsAndBytesConfig
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError(
                "VLMAgent requires torch, transformers, and peft. "
                "Install with: pip install torch transformers peft bitsandbytes"
            ) from e
            
        logger.info("Loading model %s", self.model_name)
        
        # Device setup
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"CUDA available: {gpu_name} with {gpu_memory_gb:.1f}GB memory")
        else:
            self._device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU (will be slow)")
        
        # Quantization config
        bnb_config = None
        if self.load_in_8bit and self._device.type == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
        
        # Detect if this is a vision model
        is_vision_model = any(marker in self.model_name.lower() for marker in ['vl', 'vision', 'vlm'])
        
        if is_vision_model:
            # Load vision-language model
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                logger.info("Loading as vision-language model")
                
                # Use processor for vision models
                self._tokenizer = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                
                # Load vision model
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
                )
            except ImportError:
                # Fallback to AutoModelForCausalLM if vision model not available
                logger.warning("AutoModelForVision2Seq not available, falling back to AutoModelForCausalLM")
                is_vision_model = False
        
        if not is_vision_model:
            # Load standard causal LM
            logger.info("Loading as causal language model")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
            )
        
        # Apply LoRA if requested
        if self.use_lora and self._model is not None:
            # Auto-detect target modules if not specified
            if self.lora_target_modules is None:
                # Common patterns for different model architectures
                if "qwen" in self.model_name.lower():
                    self.lora_target_modules = ["q_proj", "v_proj"]
                elif "llama" in self.model_name.lower():
                    self.lora_target_modules = ["q_proj", "v_proj"]
                else:
                    # Generic attention layers
                    self.lora_target_modules = ["query", "value"]
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
            )
            
            self._model = get_peft_model(self._model, lora_config)
            self._model.print_trainable_parameters()
        
        # Set up optimizer
        if self._model is not None:
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self.learning_rate,
            )
        
        logger.info("Model setup complete")
    
    async def fetch_response(self, observation: Observation) -> Tuple[List[Dict[str, Any]], bool]:
        """Fetch response (for compatibility with base class)."""
        # Use sample() and return text as action
        sample = await self.sample(observation)
        return [{"text": sample.text}], sample.done
    
    async def sample(self, observation: Observation, verbose: bool = False) -> ActionSample:
        """Generate text with log probabilities for RL training.
        
        Returns ActionSample with:
        - Generated text
        - Token log probabilities  
        - Parsed actions
        - Task completion status
        """
        import time
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Run the blocking inference in a thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, self._sample_sync, observation, verbose)
    
    def _sample_sync(self, observation: Observation, verbose: bool = False) -> ActionSample:
        """Synchronous version of sample for running in thread pool."""
        import time
        timing_info = {}
        total_start = time.time()
        
        self._setup_model()
        
        import torch

        if not self._tokenizer or not self._model:
            raise RuntimeError("Model not initialized")
        
        # Format the input
        format_start = time.time()
        prompt = self._format_prompt(observation)
        timing_info['prompt_format_ms'] = (time.time() - format_start) * 1000
        
        # Tokenize based on whether we have a processor or tokenizer
        tokenize_start = time.time()
        if hasattr(self._tokenizer, 'tokenizer'):
            # This is a processor (for vision models)
            inputs = self._tokenizer(
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
        else:
            # This is a regular tokenizer
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
        timing_info['tokenize_ms'] = (time.time() - tokenize_start) * 1000
        
        # Move to device
        transfer_start = time.time()
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        timing_info['transfer_to_device_ms'] = (time.time() - transfer_start) * 1000
        
        # Generate with log probs
        generate_start = time.time()
        with torch.no_grad():
            # Generate tokens
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
            
        # Extract and decode generated tokens
        decode_start = time.time()
        # Extract generated tokens (excluding prompt)
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compute log probabilities
        if outputs.scores:
            # Convert scores to log probs
            log_probs = []
            tokens = []
            
            for i, score in enumerate(outputs.scores):
                # Get the token that was actually generated
                token_id = generated_ids[i].item()
                # Get log prob for that token
                log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                log_probs.append(log_prob)
                tokens.append(self._tokenizer.decode([token_id]))
                
            total_log_prob = sum(log_probs)
        else:
            log_probs = None
            tokens = None
            total_log_prob = None
        timing_info['decode_ms'] = (time.time() - decode_start) * 1000
        
        # Parse actions from generated text
        parse_start = time.time()
        done = True #self._check_done(generated_text)
        
        # Process with adapter if available
        processed_actions = None
        if self.adapter:
            try:
                processed_actions = self.adapter.adapt_list([generated_text])
            except Exception as e:
                logger.warning(f"Failed to adapt actions: {e}")
        timing_info['parse_actions_ms'] = (time.time() - parse_start) * 1000
        
        # Total time
        timing_info['total_ms'] = (time.time() - total_start) * 1000
        
        # Add device info to metadata
        device_info: Dict[str, Any] = {
            "device": str(self._device),
            "device_type": self._device.type if self._device else "unknown",
        }
        if self._device and self._device.type == "cuda":
            device_info["cuda_device_name"] = torch.cuda.get_device_name(self._device)
            device_info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated(self._device) / 1e9
            device_info["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved(self._device) / 1e9
        
        # Create action sample
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
            }
        )
    
    async def update(self, batch: Batch) -> Dict[str, float]:
        """Perform gradient update on a batch for RL training.
        
        Args:
            batch: Training batch with observations, texts, advantages, etc.
            
        Returns:
            Dictionary of training statistics
        """
        self._setup_model()
        
        import torch
        import torch.nn.functional as F

        if not self._tokenizer or not self._model or not self._optimizer:
            raise RuntimeError("Model not initialized")
        
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
            padding=True,  # This ensures all sequences have same length
        )
        
        # Move to device
        input_ids = encoded.input_ids.to(self._device)
        attention_mask = encoded.attention_mask.to(self._device)
        
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
        advantages = torch.tensor(batch.advantages, device=self._device, dtype=torch.float32)
        
        # Forward pass
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
        # Replace -100 with 0 for gather operation, then mask afterwards
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
            # DAPO Implementation (following the paper)
            
            # Get dynamic sampling weights
            if batch.metadata and "weights" in batch.metadata:
                weights = torch.tensor(batch.metadata["weights"], device=self._device)
            else:
                # Default to uniform weights if not provided
                weights = torch.ones(len(batch.texts), device=self._device)
            
            # Token-level policy gradient (critical for DAPO)
            # Each token gets weighted by both advantage and dynamic weight
            batch_size, seq_len = token_log_probs.shape
            
            # Expand advantages and weights to token level
            token_advantages = advantages.unsqueeze(1).expand(-1, seq_len) * mask
            token_weights = weights.unsqueeze(1).expand(-1, seq_len) * mask
            
            if batch.old_log_probs is not None:
                # Get old token-level log probs (need to compute from sequences)
                old_sequence_log_probs = torch.tensor(batch.old_log_probs, device=self._device)
                
                # For simplicity, distribute old log probs uniformly across tokens
                # In practice, you'd want to store token-level old log probs
                num_response_tokens = mask.sum(dim=1)
                old_token_log_probs = old_sequence_log_probs.unsqueeze(1) / num_response_tokens.unsqueeze(1)
                old_token_log_probs = old_token_log_probs.expand(-1, seq_len) * mask
                
                # Importance sampling ratio at token level
                log_ratio = token_log_probs - old_token_log_probs
                ratio = torch.exp(log_ratio)
                
                # DAPO Clip-Higher: Asymmetric clipping
                # For positive advantages: clip at [1-ε, 1+2ε]
                # For negative advantages: clip at [1-2ε, 1+ε]
                epsilon_high = batch.metadata.get("clip_epsilon_high", 0.4) if batch.metadata else 0.4
                epsilon_low = batch.metadata.get("clip_epsilon_low", 0.2) if batch.metadata else 0.2
                
                clip_high = torch.where(token_advantages > 0, 1 + epsilon_high, 1 + epsilon_low)
                clip_low = torch.where(token_advantages > 0, 1 - epsilon_low, 1 - epsilon_high)
                clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
                
                # Token-level policy gradient with dynamic weights
                pg_loss1 = -token_advantages * ratio * token_weights
                pg_loss2 = -token_advantages * clipped_ratio * token_weights
                
                # Take max (more conservative) for each token
                token_pg_loss = torch.max(pg_loss1, pg_loss2)
                
                # Average across all tokens (not sequences)
                pg_loss = token_pg_loss.sum() / mask.sum()
                
                # KL penalty (still at sequence level for stability)
                sequence_log_probs = token_log_probs.sum(dim=1) / mask.sum(dim=1)
                sequence_log_ratio = sequence_log_probs - old_sequence_log_probs
                kl_divergence = sequence_log_ratio.mean().item()
                kl_penalty = 0.01 * sequence_log_ratio.mean()  # Smaller KL for DAPO
                
                # Track clipping statistics
                clipped = (ratio != clipped_ratio).float() * mask
                clip_fraction = clipped.sum() / mask.sum()
                
            else:
                # Fallback without old log probs
                pg_loss = -(token_advantages * token_log_probs * token_weights).sum() / mask.sum()
                kl_penalty = 0.0
                kl_divergence = 0.0
                ratio = torch.ones_like(advantages)
                clip_fraction = torch.tensor(0.0)
            
            # Entropy at token level (for diversity)
            token_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
            masked_entropy = (token_entropy * mask).sum() / mask.sum()
            
            # Larger entropy bonus for DAPO to encourage diversity
            entropy_coeff = 0.01
            entropy_bonus = entropy_coeff * masked_entropy
            
            # Total loss
            loss = pg_loss + kl_penalty - entropy_bonus
            
        else:
            # Original GRPO implementation
            # Sum log probs per sequence
            sequence_log_probs = token_log_probs.sum(dim=1) / mask.sum(dim=1)
            
            # Pure GRPO: Simple policy gradient without PPO-style clipping
            # Loss = -advantages * log_probs
            pg_loss = -(advantages * sequence_log_probs).mean()
            
            # KL regularization (simplified for now - ideally would use reference policy)
            # In DeepSeek's implementation, they compute KL(π_θ || π_ref) where π_ref is frozen
            # For now, we'll use the old log probs as a proxy if available
            if batch.old_log_probs is not None:
                old_log_probs = torch.tensor(batch.old_log_probs, device=self._device)
                
                # KL divergence approximation: E[log(π_θ / π_old)]
                log_ratio = sequence_log_probs - old_log_probs
                kl_divergence = log_ratio.mean().item()
                
                # KL penalty with larger coefficient (DeepSeek uses β=0.1 or similar)
                kl_penalty = 0.1 * log_ratio.mean()
                
                # For monitoring: importance sampling ratio
                ratio = torch.exp(log_ratio)
            else:
                kl_penalty = 0.0
                kl_divergence = 0.0
                ratio = torch.ones_like(advantages)
            
            # Entropy regularization (optional in DeepSeek, but helps exploration)
            with torch.no_grad():
                entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            
            # DeepSeek doesn't emphasize entropy bonus, so smaller coefficient
            entropy_coeff = 0.001  
            entropy_bonus = entropy_coeff * entropy
            
            # Total loss (DeepSeek style: policy gradient + KL penalty)
            loss = pg_loss + kl_penalty - entropy_bonus
            clip_fraction = torch.tensor(0.0)
        
        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()

        # Compute gradient magnitude **before** the optimizer potentially
        # modifies / clears gradients.
        grad_vals = [
            p.grad.detach().abs().mean().item()
            for p in self._model.parameters() if p.grad is not None
        ]
        avg_grad_abs = float(sum(grad_vals) / len(grad_vals)) if grad_vals else 0.0
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        
        # Optimizer step
        self._optimizer.step()
 
        # Compute statistics
        stats = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl": kl_divergence,
            "approx_kl": kl_divergence, # This is now the actual KL divergence
            "entropy": entropy.item() if algorithm == "GRPO" else masked_entropy.item(),
            "entropy_bonus": entropy_bonus.item(),
            "avg_advantage": advantages.mean().item(),
            "avg_log_prob": sequence_log_probs.mean().item() if algorithm == "GRPO" else (token_log_probs.sum() / mask.sum()).item(),
            "avg_ratio": ratio.mean().item() if isinstance(ratio, torch.Tensor) else 1.0,
            "avg_grad_abs": avg_grad_abs,
            "clip_fraction": clip_fraction.item() if isinstance(clip_fraction, torch.Tensor) else 0.0,
            "algorithm": algorithm,
        }
        
        # Add DAPO-specific stats
        if algorithm == "DAPO":
            stats["avg_weight"] = weights.mean().item()
            stats["weight_std"] = weights.std().item()
 
        return stats
    
    def _format_prompt(self, observation: Observation) -> str:
        """Format observation into a prompt."""
        messages = []
        
        # System message
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # User message
        user_content = []
        if observation.text:
            user_content.append(observation.text)
        
        if observation.screenshot:
            # For vision models, this would handle image tokens
            # For now, just indicate presence
            user_content.append("[Screenshot provided]")
        
        messages.append({"role": "user", "content": "\n".join(user_content)})
        
        # Apply chat template if available
        if self._tokenizer and hasattr(self._tokenizer, 'apply_chat_template'):
            prompt = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
            prompt_parts.append("Assistant: ")
            prompt = "\n\n".join(prompt_parts)
        
        return prompt
    
    
    def _check_done(self, text: str) -> bool:
        """Check if task is complete."""
        done_patterns = [
            r'done\s*\(',
            r'task.*completed',
            r'finished:',
            r'Action:\s*done'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in done_patterns) 