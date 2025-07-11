"""Neuron-powered subclass of VLMAgent.

Loads/compiles the model with Optimum-Neuron the first time it starts and
then behaves exactly like the normal VLMAgent interface so all trainers
can keep using HostedVLMAgent.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch  # type: ignore
from optimum.neuron import NeuronModelForCausalLM  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from hud.agent.vlm import VLMAgent


class VLMNeuronAgent(VLMAgent):
    """VLMAgent that runs on AWS Trainium (XLA / Neuron)."""

    def _setup_model(self):  # noqa: C901 – keep logic local
        """Patch the original GPU/CPU _setup_model with Neuron specifics."""
        if self._model is not None:
            return  # already initialised

        # ────────────────────────────────────────────────
        # Environment defaults (overrideable via export …)
        # ────────────────────────────────────────────────
        os.environ.setdefault("NEURON_RT_NUM_CORES", "2")  # trn1.2xlarge = 2 cores
        os.environ.setdefault("XLA_USE_BF16", "1")        # bf16 math everywhere

        cache_dir = os.getenv("NEURON_COMPILE_CACHE_URL", "/home/ubuntu/neuron_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # ─────────────────────────────
        # Tokeniser (same as upstream)
        # ─────────────────────────────
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ──────────────────────────────────
        # Compile or load Neuron artefacts
        # ──────────────────────────────────
        compiled_dir = (
            self.model_name.replace("/", "_").replace(".", "-") + ".neuron"
        )

        if Path(compiled_dir).is_dir():
            # Use cached compiled model
            self._model = NeuronModelForCausalLM.from_pretrained(compiled_dir)
        else:
            # First-time compilation (takes minutes)
            self._model = NeuronModelForCausalLM.from_pretrained(
                self.model_name,
                export=True,        # triggers trace/compile
                auto_cast=True,     # bf16 cast where possible
                trust_remote_code=True,
            )
            # Persist for next launch
            self._model.save_pretrained(compiled_dir)

        # All Neuron models live on XLA device 0
        self._device = torch.device("xla")

        # Disable LoRA – not supported on Neuron out-of-the-box
        self.use_lora = False

        # Optimiser (runs on host; we still need it for .update())
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate
        ) 