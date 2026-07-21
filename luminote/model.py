from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BoundaryActivations:
    by_layer: list[torch.Tensor]
    token_id: int
    token_text: str
    prompt_length: int


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_text: str
    parameters: dict[str, int | float | bool | None]
    stop_reason: str


def infer_stop_reason(token_ids: list[int], eos_token_ids: set[int], max_new_tokens: int) -> str:
    if token_ids and token_ids[-1] in eos_token_ids:
        return "eos"
    if len(token_ids) >= max_new_tokens:
        return "maximum_token_limit"
    return "configured_stop_token"


class LocalModel:
    """Swappable local causal-LM backend with assistant-boundary inspection."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = self._best_device()
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device).eval()

    @staticmethod
    def _best_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def layer_count(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def render(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in rendered.items()}

    @torch.inference_mode()
    def inspect_boundary(self, messages: list[dict[str, str]]) -> BoundaryActivations:
        inputs = self.render(messages)
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        token_id = int(inputs["input_ids"][0, -1].item())
        # hidden_states[0] is embedding output; public layer numbers are zero-based.
        vectors = [state[0, -1].detach().float().cpu() for state in outputs.hidden_states[1:]]
        return BoundaryActivations(
            by_layer=vectors,
            token_id=token_id,
            token_text=self.tokenizer.decode([token_id]),
            prompt_length=int(inputs["input_ids"].shape[-1]),
        )

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> GenerationResult:
        inputs = self.render(messages)
        parameters: dict[str, int | float | bool | None] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.75,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        output = self.model.generate(**inputs, **parameters)
        generated = output[0, inputs["input_ids"].shape[-1] :]
        token_ids = [int(token) for token in generated.detach().cpu().tolist()]
        configured_eos = self.model.generation_config.eos_token_id
        eos_token_ids = {
            int(token)
            for token in (configured_eos if isinstance(configured_eos, list) else [configured_eos])
            if token is not None
        }
        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.add(int(self.tokenizer.eos_token_id))
        prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
        return GenerationResult(
            text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            token_ids=token_ids,
            prompt_text=self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
            parameters=parameters,
            stop_reason=infer_stop_reason(token_ids, eos_token_ids, max_new_tokens),
        )
