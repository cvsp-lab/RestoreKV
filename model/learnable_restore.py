from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Optional

import torch
import torch.nn as nn


LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)


@dataclass
class RestoreConfig:
    mode: str = "opt1"
    num_restore_tokens: int = 32
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    opt2_query_gate_init: float = -5.0
    use_ratio_conditioning: bool = False
    ratio_condition_hidden_size: Optional[int] = None
    # "mlp" (default original), "film", "moe"
    ratio_condition_type: str = "mlp"
    moe_num_experts: int = 4


class RestoreLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = False

        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        lora = torch.nn.functional.linear(self.dropout(x), self.lora_a)
        lora = torch.nn.functional.linear(lora, self.lora_b)
        return out + lora * self.scaling


def _get_parent_module(root: nn.Module, module_name: str) -> nn.Module:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def install_restore_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_suffixes: Iterable[str] = LORA_TARGET_SUFFIXES,
) -> List[RestoreLoRALinear]:
    import os
    env_targets = os.environ.get("LORA_TARGETS", "")
    if env_targets.strip():
        target_suffixes = tuple(s.strip() for s in env_targets.split(",") if s.strip())
        print(f"[lora] LORA_TARGETS override: {target_suffixes}")
    wrappers: List[RestoreLoRALinear] = []
    suffixes = tuple(target_suffixes)

    for name, module in list(model.named_modules()):
        if isinstance(module, RestoreLoRALinear):
            wrappers.append(module)
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue

        parent = _get_parent_module(model, name)
        child_name = name.rsplit(".", 1)[-1]
        wrapped = RestoreLoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapped)
        wrappers.append(wrapped)

    if not wrappers:
        raise RuntimeError("No LoRA target Linear modules were found")
    return wrappers


@contextmanager
def lora_enabled(wrappers: Iterable[RestoreLoRALinear]) -> Iterator[None]:
    wrappers = list(wrappers)
    previous = [wrapper.enabled for wrapper in wrappers]
    for wrapper in wrappers:
        wrapper.enabled = True
    try:
        yield
    finally:
        for wrapper, enabled in zip(wrappers, previous):
            wrapper.enabled = enabled


class LearnableRestoreTokens(nn.Module):
    def __init__(
        self,
        config,
        restore_config: Optional[RestoreConfig] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.restore_config = restore_config or RestoreConfig()
        if self.restore_config.mode not in {"opt1", "opt2"}:
            raise ValueError("restore mode must be 'opt1' or 'opt2'")

        self.num_tokens = self.restore_config.num_restore_tokens
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        init_dtype = dtype or torch.float32
        self.embeddings = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_size, dtype=init_dtype))
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)

        self.ratio_cond_type = "none"
        self.ratio_mlp = None
        self.film_shared = None
        self.film_gamma = None
        self.film_beta = None
        self.moe_bases = None
        self.moe_router = None
        if self.restore_config.use_ratio_conditioning:
            ratio_hidden_size = (
                self.hidden_size
                if self.restore_config.ratio_condition_hidden_size is None
                else self.restore_config.ratio_condition_hidden_size
            )
            if ratio_hidden_size <= 0:
                raise ValueError("ratio_condition_hidden_size must be positive")
            rc_type = self.restore_config.ratio_condition_type
            self.ratio_cond_type = rc_type
            if rc_type == "mlp":
                self.ratio_mlp = nn.Sequential(
                    nn.Linear(1, ratio_hidden_size, dtype=init_dtype),
                    nn.SiLU(),
                    nn.Linear(
                        ratio_hidden_size,
                        self.num_tokens * self.hidden_size,
                        dtype=init_dtype,
                    ),
                )
            elif rc_type == "film":
                self.film_shared = nn.Sequential(
                    nn.Linear(1, ratio_hidden_size, dtype=init_dtype),
                    nn.SiLU(),
                )
                self.film_gamma = nn.Linear(
                    ratio_hidden_size, self.num_tokens * self.hidden_size, dtype=init_dtype)
                self.film_beta = nn.Linear(
                    ratio_hidden_size, self.num_tokens * self.hidden_size, dtype=init_dtype)
                # init: gamma=1 (preserve emb), beta=0
                nn.init.zeros_(self.film_gamma.weight)
                nn.init.ones_(self.film_gamma.bias)
                nn.init.zeros_(self.film_beta.weight)
                nn.init.zeros_(self.film_beta.bias)
            elif rc_type == "moe":
                K = self.restore_config.moe_num_experts
                self.moe_bases = nn.Parameter(
                    torch.empty(K, self.num_tokens, self.hidden_size, dtype=init_dtype))
                nn.init.normal_(self.moe_bases, mean=0.0, std=0.02)
                self.moe_router = nn.Sequential(
                    nn.Linear(1, ratio_hidden_size, dtype=init_dtype),
                    nn.SiLU(),
                    nn.Linear(ratio_hidden_size, K, dtype=init_dtype),
                )
            else:
                raise ValueError(f"unknown ratio_condition_type: {rc_type}")

        if self.restore_config.mode == "opt2":
            self.layer_queries = nn.Parameter(
                torch.empty(
                    self.num_layers,
                    self.num_tokens,
                    self.num_heads,
                    self.head_dim,
                    dtype=init_dtype,
                ))
            nn.init.normal_(self.layer_queries, mean=0.0, std=0.02)
            self.layer_query_gate_logits = nn.Parameter(
                torch.full(
                    (self.num_layers, self.num_heads, 1, 1),
                    float(self.restore_config.opt2_query_gate_init),
                    dtype=init_dtype,
                ))
        else:
            self.layer_queries = None
            self.layer_query_gate_logits = None

    def forward(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        ratio: Optional[float] = None,
    ) -> torch.Tensor:
        embeddings = self.embeddings
        if device is not None:
            embeddings = embeddings.to(device)
        embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        if self.ratio_cond_type == "none":
            return embeddings
        if ratio is None:
            raise ValueError("ratio is required when ratio conditioning is enabled")

        if torch.is_tensor(ratio):
            ratio_tensor = ratio.to(device=embeddings.device, dtype=embeddings.dtype)
            ratio_tensor = ratio_tensor.reshape(-1, 1)
            if ratio_tensor.shape[0] == 1 and batch_size != 1:
                ratio_tensor = ratio_tensor.expand(batch_size, -1)
        else:
            ratio_tensor = torch.full(
                (batch_size, 1),
                float(ratio),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        if ratio_tensor.shape != (batch_size, 1):
            raise ValueError(
                f"ratio must be scalar or have shape ({batch_size}, 1); "
                f"got {tuple(ratio_tensor.shape)}"
            )

        if self.ratio_cond_type == "mlp":
            delta = self.ratio_mlp(ratio_tensor)
            delta = delta.reshape(batch_size, self.num_tokens, self.hidden_size)
            return embeddings + delta
        elif self.ratio_cond_type == "film":
            h = self.film_shared(ratio_tensor)  # (B, ratio_h)
            gamma = self.film_gamma(h).reshape(batch_size, self.num_tokens, self.hidden_size)
            beta = self.film_beta(h).reshape(batch_size, self.num_tokens, self.hidden_size)
            return gamma * embeddings + beta
        elif self.ratio_cond_type == "moe":
            logits = self.moe_router(ratio_tensor)  # (B, K)
            weights = torch.softmax(logits, dim=-1)  # (B, K)
            # bases: (K, num_tokens, hidden)
            mixed = torch.einsum("bk,kth->bth", weights, self.moe_bases)
            return mixed
        return embeddings

    def layer_query(self, layer_idx: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.layer_queries is None:
            raise RuntimeError("Layer query override is only available in opt2")
        return self.layer_queries[layer_idx].to(device=device, dtype=dtype)

    def layer_query_delta(self, layer_idx: int, dtype: torch.dtype,
                          device: torch.device) -> torch.Tensor:
        if self.layer_queries is None or self.layer_query_gate_logits is None:
            raise RuntimeError("Layer query delta is only available in opt2")
        query = self.layer_queries[layer_idx].to(device=device, dtype=dtype)
        gate = torch.sigmoid(
            self.layer_query_gate_logits[layer_idx].to(device=device, dtype=dtype)
        )
        gate = gate.permute(1, 0, 2)
        return query * gate

    def checkpoint_payload(self, extra: Optional[Dict] = None) -> Dict:
        payload = {
            "restore_config": asdict(self.restore_config),
            "state_dict": self.state_dict(),
        }
        if extra:
            payload["extra"] = extra
        return payload


def load_restore_tokens(checkpoint_path: str, model_config, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    restore_payload = checkpoint["restore"] if "restore" in checkpoint else checkpoint
    restore_config = RestoreConfig(**restore_payload["restore_config"])
    module = LearnableRestoreTokens(model_config, restore_config=restore_config)
    state_dict = restore_payload["state_dict"]
    if restore_config.mode == "opt2" and "layer_query_gate_logits" not in state_dict:
        state_dict = dict(state_dict)
        state_dict["layer_query_gate_logits"] = module.layer_query_gate_logits.detach().clone()
    module.load_state_dict(state_dict)
    return module, checkpoint
