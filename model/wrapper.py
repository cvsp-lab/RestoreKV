# ------------------------------------------------------------------------------
# Original code from the KVzip repository (snu-mllab, MIT License)
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import torch
import glob
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
from transformers import DynamicCache, Gemma3ForCausalLM, Qwen3ForCausalLM

from attention.kvcache import RetainCache, EvictCache, RetainHybridCache
from utils.func import inplace_softmax
from model.load import load_model
# Quantized-KV support is not included in this release. These stubs keep the
# isinstance() guards below valid (isinstance(x, ()) is always False).
LlamaForCausalLMW8A8 = ()
OptimINT4KVCache = None
from model.template import template


def chunk_fn(ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """ Chunk tokens
    """
    ctx_len = ctx_ids.shape[1]
    if ctx_len > chunk_size:
        chunk_num = (ctx_len - 1) // chunk_size + 1
        print(f"chunk inputs, size: {chunk_size} (num {chunk_num})")

        input_ids = []
        for i in range(chunk_num):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            a_ids = ctx_ids[:, start:end]
            if a_ids.shape[1] == 0:
                continue
            input_ids.append(a_ids)
    else:
        input_ids = [ctx_ids]

    return input_ids


def load_head_score(model_name, ctx_len):
    if model_name.startswith("Qwen2.5-7B"):
        model_name = "qwen2.5-7b"
    elif model_name.startswith("Qwen2.5-14B"):
        model_name = "qwen2.5-14b"
    elif model_name.startswith("Llama-3.1-8B"):
        model_name = "llama3.1-8b"

    attn_ = []
    paths = f"./utils/head_score/{model_name}-*.pt"
    for path in glob.glob(paths):
        attn = torch.load(path).squeeze().cuda()  # layer x head
        attn_.append(attn)
        print("Load head-score from", path)

    attn = torch.stack(attn_, dim=0).amax(0)
    score = attn.unsqueeze(-1).expand(-1, -1, ctx_len)  # layer x head x seq
    score = score.unsqueeze(1)
    return score


class ModelKVzip():

    def __init__(self, model_name: str, kv_type: str = "evict"):
        self.model, self.tokenizer = load_model(model_name)

        self.name = self.model.name
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.config = self.model.config

        if isinstance(self.model, LlamaForCausalLMW8A8):
            self.kv_type = "int4static"
            print("[Note] Currently, only retain cache is available for QServe")
        elif isinstance(self.model, Gemma3ForCausalLM):
            self.kv_type = "hybrid_static"
            print("[Note] Currently, only retain cache is available for Gemma3")
        else:
            self.kv_type = kv_type
        print(f"KV type: {self.kv_type}")

        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 512,
        }
        if isinstance(self.model, Gemma3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = [1, 106]
        elif isinstance(self.model, Qwen3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = 151645

        self.set_chat_template()

    def encode(self, text: str) -> torch.Tensor:
        """ Encode text into tokens
        """
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").cuda()

    def decode(self, input_ids: torch.Tensor) -> str:
        """ Decode tokens into text
        """
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]
        return self.tokenizer.decode(input_ids)

    def set_chat_template(self, task: str = "qa"):
        # KVPRESS_COMPAT applies to RULER only (`task` receives the dataset name
        # via DataWrapper.__init__); other datasets keep the legacy template.
        import os as _os
        if (_os.environ.get("KVPRESS_COMPAT", "0") in ("1", "true", "yes")
                and task.startswith("ruler")):
            # kvpress pipeline.preprocess-equivalent prompt: split the tokenizer's own
            # chat template around a separator. No explicit system prompt, no
            # "Given the context..." instruction line (kvpress adds neither).
            dummy = "dummy context"
            sep = "#" * (len(dummy) + 10)
            try:
                full = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": dummy + sep}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except (TypeError, ValueError):  # tokenizer without enable_thinking kwarg
                full = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": dummy + sep}],
                    add_generation_prompt=True, tokenize=False)
            ctx_part, self.postfix_text = full.split(sep)
            prefix = ctx_part[:-len(dummy)]
            assert ctx_part.endswith(dummy), "chat template split failed"
            self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(self.postfix_text)
            return
        self.postfix_text = None
        prefix, postfix = template(self.name, task)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(postfix)

    def apply_template(self, query: str) -> torch.Tensor:
        query = f"\n\n{query.strip()}"
        query_ids = torch.cat([self.encode(query), self.postfix_ids], dim=1)
        return query_ids

    def apply_template_kvpress(self, question: str, answer_prefix: str = "") -> torch.Tensor:
        """ kvpress-style query: question + chat-template generation suffix + answer_prefix,
            tokenized jointly as one string (matches kvpress pipeline.preprocess).
        """
        assert self.postfix_text is not None, "requires KVPRESS_COMPAT=1 chat template"
        return self.encode(question + self.postfix_text + (answer_prefix or ""))

    def __call__(
        self,
        input_ids: torch.Tensor,
        kv: Union[RetainCache, EvictCache],
        update_cache: bool = False,
        return_logits: bool = False,
        *args,
        **kwargs,
    ):
        """ Compute Transformer forward pass
            In default, we do not update the KV cache with the newly given inputs.
            Set update_cache = True to enable the update.
        """
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        # Optional: shift query position via KVZIP_QUERY_POS_SHIFT env var
        # (e.g., +8 → simulate query position as if 8 restore tokens were appended).
        import os as _os
        _qps_env = _os.environ.get("KVZIP_QUERY_POS_SHIFT", "0")
        _qps = int(_qps_env) if _qps_env.lstrip("-").isdigit() else 0
        _pos_ids = None
        if _qps != 0:
            q_len = input_ids.shape[-1]
            _pos_ids = torch.arange(
                seen_token_prev + _qps,
                seen_token_prev + _qps + q_len,
                device=input_ids.device,
            ).unsqueeze(0)

        if return_logits:
            if _pos_ids is not None:
                outputs = self.model(input_ids, past_key_values=kv,
                                     position_ids=_pos_ids, *args, **kwargs)
            else:
                outputs = self.model(input_ids, past_key_values=kv, *args, **kwargs)
        else:
            if _pos_ids is not None:
                _ = self.model.model(input_ids, past_key_values=kv,
                                     position_ids=_pos_ids, *args, **kwargs)
            else:
                _ = self.model.model(input_ids, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            kv.slice(seen_token_prev)
        return outputs

    def forward_embeds(
        self,
        inputs_embeds: torch.Tensor,
        kv: Union[RetainCache, EvictCache],
        update_cache: bool = False,
        return_logits: bool = False,
        *args,
        **kwargs,
    ):
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        if return_logits:
            outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=kv, *args, **kwargs)
        else:
            _ = self.model.model(inputs_embeds=inputs_embeds, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            kv.slice(seen_token_prev)
        return outputs

    def _init_kv(self, kv=None, evict_range=(0, 0)):
        """ Initialize KV cache
        """

        if kv is None:
            if self.kv_type == "retain":
                kv = RetainCache(self.model, evict_range)
            elif self.kv_type == "evict":
                kv = EvictCache(self.model, evict_range)
            elif self.kv_type == "int4static":
                kv = OptimINT4KVCache(self.model.model, evict_range)
            elif self.kv_type == "hybrid_static":
                max_size = 190000
                kv = RetainHybridCache(self.model.model, evict_range, max_size)
            elif self.kv_type == "original":
                kv = DynamicCache()
                kv.pruned, kv.get_score = False, False
            else:
                raise NotImplementedError(f"type {self.kv_type} is not implemented")
        return kv

    @torch.inference_mode()
    def prefill(
        self,
        ctx_ids: Union[str, torch.Tensor],
        prefill_chunk_size: int = 16000,
        load_score=False,
        do_score=True,
        score_method: str = "kvzip",
    ) -> Union[RetainCache, EvictCache]:
        """ Chunked prefill KV cache.

        score_method:
          - "kvzip" (default): KVzip context-reconstruction self-task scoring
          - "kvzip_plus": KVzip+ (Jégou & Jeblick 2026) — same self-task but
                          normalized attention score |W_O v_i| / |h_j|
          - "snapkv": query-agnostic SnapKV scoring (re-feed last window)
        """
        if type(ctx_ids) == str:
            ctx_ids = self.encode(ctx_ids)
        prefill_ids = torch.cat([self.sys_prompt_ids, ctx_ids], dim=1)
        evict_range = (self.sys_prompt_ids.shape[1], prefill_ids.shape[1])

        kv = self._init_kv(evict_range=evict_range)  # do not evict system prompt KV
        kv.ctx_ids = ctx_ids
        kv.prefill_ids = prefill_ids

        # H2O (KVzip-paper counterpart): score DURING prefill from causal
        # self-attention (query=chunk at original positions, key=cache so far).
        # Uses a small chunk to bound the O(q_len x k_total) attention tensor and
        # a preallocated running-max buffer (sliced to ctx after prefill).
        import os as _os
        _h2o = (do_score and not load_score and score_method == "h2o")
        if _h2o:
            full_len = prefill_ids.shape[1]
            kv.get_score = True
            kv.score_mode = "h2o"
            kv.score = [
                torch.zeros((1, kv.n_heads_kv, full_len), dtype=kv.dtype, device=kv.device)
                for _ in range(kv.n_layers)
            ]
            _pf_chunk = int(_os.environ.get("H2O_PREFILL_CHUNK", "1024"))
        else:
            _pf_chunk = prefill_chunk_size

        # prefill
        for input_ids in tqdm(chunk_fn(prefill_ids, _pf_chunk), desc="Prefill"):
            self.__call__(input_ids, kv, update_cache=True)

        if _h2o:
            kv.get_score = False
            kv.score_mode = "kvzip"
            sys_len = self.sys_prompt_ids.shape[1]
            kv.score = [s[:, :, sys_len:].contiguous() for s in kv.score]
            assert kv.score[0].shape[-1] == kv.ctx_len, (
                f"h2o score shape {kv.score[0].shape} vs ctx_len {kv.ctx_len}"
            )

        if do_score:
            if score_method == "h2o":
                pass  # already scored inline during prefill
            elif score_method == "snapkv":
                self.scoring_snapkv(kv, ctx_ids)
            elif score_method == "kvzip_plus":
                kv.score_mode = "kvzip_plus"
                self.scoring(kv, ctx_ids, load_score=load_score)
                kv.score_mode = "kvzip"  # reset so downstream calls don't get confused
            elif score_method == "contrastkv":
                self.scoring_contrastkv(kv, ctx_ids, load_score=load_score)
            else:
                self.scoring(kv, ctx_ids, load_score=load_score)

            # kvpress-compat n_sink=4: kvpress pins the first 4 tokens of the FULL
            # prompt (score=1.0). Our sys-prefix is already never evicted, so pin
            # only the shortfall (4 - prefix_len) leading context tokens.
            # Active only in KVPRESS_COMPAT mode (postfix_text set → ruler datasets).
            if self.postfix_text is not None and getattr(kv, "score", None) is not None:
                _sink = max(0, 4 - self.sys_prompt_ids.shape[1])
                if _sink > 0:
                    _scores = kv.score if isinstance(kv.score, list) else [kv.score]
                    for _s in _scores:
                        if _s.numel():
                            _s[..., :_sink] = _s.max() + 1.0
        return kv

    def self_task(
        self,
        ctx_ids: torch.Tensor,
        chunk_size: int = 2000,
        prev_postfix_size=8,
    ) -> List[torch.Tensor]:
        """ Prepare chunked inputs for KV importance scoring with context reconstruction
            return: List[torch.Tensor]
        """
        import os as _os
        # ruler-only: postfix_text is set iff the kvpress-compat template is active
        _kvpress_compat = self.postfix_text is not None
        if _kvpress_compat:
            chunk_size = 2048  # kvpress KVzipPress.prepare default (ours is 2000)
        chunked_inputs = chunk_fn(ctx_ids, chunk_size)

        # Optional scoring-probe position shift: prepend N placeholder tokens before the
        # "Repeat..." probe so its position equals what the real user query sees at
        # inference (where N restore tokens occupy the slot between context and query).
        # Set via KVZIP_PROBE_SHIFT env var (default 0).
        probe_shift = int(_os.environ.get("KVZIP_PROBE_SHIFT", "0"))
        shift_pad_ids = None
        if probe_shift > 0:
            # use newline token as a benign placeholder; repeat probe_shift times
            shift_pad_ids = self.encode("\n")[:, :1].repeat(1, probe_shift)

        input_ids = []
        for i, a_ids in enumerate(chunked_inputs):
            if i == 0:
                prompt = f"\n\nRepeat the previous context exactly."
                q_ids = self.encode(prompt)
            else:
                # kvpress uses no trailing space (prev-chunk token ids are concatenated
                # directly); joint vs separate tokenization differs by one space token.
                prompt = ("\n\nRepeat the part of the previous context exactly, starting with"
                          if _kvpress_compat else
                          "\n\nRepeat the part of the previous context exactly, starting with ")
                q_ids = self.encode(prompt)
                postfix_prev = chunked_inputs[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            if shift_pad_ids is not None:
                q_ids = torch.cat([shift_pad_ids.to(q_ids.device), q_ids], dim=1)

            input_ids.append((a_ids, torch.cat([q_ids, self.postfix_ids, a_ids], dim=1)))

        return input_ids

    @torch.inference_mode()
    def scoring(
        self,
        kv: Union[RetainCache, EvictCache],
        ctx_ids: torch.Tensor,
        load_score=False,
    ):
        """ KV importance scoring (update kv.score)
        """
        if not load_score:
            kv.init_score()
            start_idx_tmp = kv.start_idx

            kv.end_idx = 0
            input_ids = self.self_task(ctx_ids)
            for i, (prefill_ids_p,
                    repeat_ids_p) in enumerate(tqdm(input_ids, desc=f"Importance scoring")):
                kv.end_idx = kv.start_idx + prefill_ids_p.shape[1]  # indices for a chunk
                self.__call__(repeat_ids_p, kv, update_cache=False)  # get score
                kv.start_idx = kv.end_idx

            kv.start_idx = start_idx_tmp
            assert kv.score[0].shape[-1] == kv.ctx_len
        else:
            kv.score = load_head_score(self.name, kv.ctx_len)

        kv.get_score = False

    def pos_task(self, ctx_ids: torch.Tensor, chunk_size: int = 2000) -> List[torch.Tensor]:
        """ContrastKV positive signal: raw ctx chunks as query (no 'Repeat' wrapper)."""
        return chunk_fn(ctx_ids, chunk_size)

    def neg_task(
        self,
        ctx_ids: torch.Tensor,
        chunk_size: int = 2000,
        neg_len: int = 64,
    ) -> List[torch.Tensor]:
        """ContrastKV negative signal: random 64-token strings, one per pos chunk."""
        n_chunks = len(chunk_fn(ctx_ids, chunk_size))
        vocab_size = int(getattr(self.tokenizer, "vocab_size", 32000))
        low = 5 if vocab_size > 10 else 1
        high = vocab_size
        return [
            torch.randint(low=low, high=high, size=(1, neg_len), device=ctx_ids.device)
            for _ in range(n_chunks)
        ]

    @torch.inference_mode()
    def scoring_contrastkv(
        self,
        kv: Union[RetainCache, EvictCache],
        ctx_ids: torch.Tensor,
        load_score: bool = False,
        chunk_size: int = 2000,
        neg_len: int = 64,
        q_low: float = 0.10,
        q_high: float = 0.90,
        top_pos_keep: float = 0.50,
        top_neg_gate: float = 0.20,
        boost_scale: float = 0.12,
    ):
        """ContrastKV scoring (Chen et al., ACL 2026).

        Runs KVzip-style max-attn scoring twice:
          - positive: chunk ctx and feed each chunk back as query
          - negative: random 64-token strings per chunk position
        Fuses per-layer per-head via quantile gating + rank-based boost.
        """
        if load_score:
            kv.score = load_head_score(self.name, kv.ctx_len)
            kv.get_score = False
            return

        eps = 1e-6

        # --- Positive pass ---
        kv.init_score()
        start_idx_backup = kv.start_idx
        kv.end_idx = 0
        pos_inputs = self.pos_task(ctx_ids, chunk_size)
        for a_ids in tqdm(pos_inputs, desc="[ContrastKV pos]"):
            kv.end_idx = kv.start_idx + a_ids.shape[1]
            self.__call__(a_ids, kv, update_cache=False)
            kv.start_idx = kv.end_idx
        kv.start_idx = start_idx_backup
        pos_score = [s.clone().float() for s in kv.score]

        # --- Negative pass ---
        kv.init_score()
        kv.end_idx = 0
        neg_inputs = self.neg_task(ctx_ids, chunk_size, neg_len=neg_len)
        for a_ids, r_ids in tqdm(
            zip(pos_inputs, neg_inputs), total=len(pos_inputs), desc="[ContrastKV neg]"
        ):
            kv.end_idx = kv.start_idx + a_ids.shape[1]
            self.__call__(r_ids, kv, update_cache=False)
            kv.start_idx = kv.end_idx
        kv.start_idx = start_idx_backup
        neg_score = [s.clone().float() for s in kv.score]

        # --- Fusion ---
        merged = []
        for ps, ns in zip(pos_score, neg_score):
            bsz, heads, seq_len = ps.shape
            assert bsz == 1

            q10_ps = torch.quantile(ps, q_low, dim=-1, keepdim=True)
            q90_ps = torch.quantile(ps, q_high, dim=-1, keepdim=True)
            q10_ns = torch.quantile(ns, q_low, dim=-1, keepdim=True)
            q90_ns = torch.quantile(ns, q_high, dim=-1, keepdim=True)

            pos_large = ps >= q90_ps
            pos_small = ps <= q10_ps
            neg_large = ns >= q90_ns
            neg_small = ns <= q10_ns

            both_large = pos_large & neg_large
            both_small = pos_small & neg_small

            fused = ps.clone()
            fused[both_large] = 1.0
            fused[both_small] = 0.0

            ps_flat = ps.view(-1)
            ns_flat = ns.view(-1)
            n_items = ps_flat.numel()

            pos_50 = torch.quantile(ps_flat, q=top_pos_keep)
            pos_min = torch.min(ps_flat)
            pos_max = torch.max(ps_flat)
            pos_range = torch.clamp(pos_max - pos_min, min=eps)

            neg_80 = torch.quantile(ns_flat, q=1.0 - top_neg_gate)

            sort_idx = torch.argsort(ns_flat, stable=True)
            ranks = torch.empty_like(ns_flat, dtype=torch.float32)
            ranks[sort_idx] = torch.arange(n_items, device=ns_flat.device, dtype=torch.float32)
            ranks = ranks / max(float(n_items - 1), 1.0)
            ns_rank = ranks.view(1, heads, seq_len)

            non_extreme = ~(both_large | both_small)
            mask_boost = non_extreme & (ps >= pos_50) & (ns >= neg_80)

            if mask_boost.any():
                gate_rank = 1.0 - top_neg_gate
                weight_raw = (ns_rank[mask_boost] - gate_rank) / max(top_neg_gate, eps)
                weight = torch.clamp(weight_raw, 0.0, 1.0)
                bump = boost_scale * pos_range * weight
                fused[mask_boost] = torch.minimum(fused[mask_boost] + bump, pos_max)

            merged.append(fused.to(ps.dtype))

        kv.score = merged
        kv.get_score = False
        assert kv.score[0].shape[-1] == kv.ctx_len

    @torch.inference_mode()
    def scoring_snapkv(
        self,
        kv: Union[RetainCache, EvictCache],
        ctx_ids: torch.Tensor,
        window_size: int = 32,
        kernel_size: int = 7,
        first_tokens: int = 4,
    ):
        """SnapKV query-agnostic scoring.

        Re-feeds the last `window_size` ctx tokens through the model with the
        cache already populated. The attention forward calls _get_score on the
        KV object, which dispatches to _get_score_snapkv (mean over window,
        max-pool, first/last-window protection).
        """
        kv.init_score()
        kv.score_mode = "snapkv"
        kv.snapkv_window = window_size
        kv.snapkv_kernel = kernel_size
        kv.snapkv_first_tokens = first_tokens

        win = min(window_size, ctx_ids.shape[1])
        last_window_ids = ctx_ids[:, -win:]

        # tell _get_score_snapkv that the scoring "ctx region" spans the full ctx
        # (start_idx already set to sys-prompt end by _init_kv)
        kv.end_idx = kv.start_idx + ctx_ids.shape[1]

        # one forward of the last window over the cached prefix
        # update_cache=False so we don't grow the KV cache
        self.__call__(last_window_ids, kv, update_cache=False)

        # restore mode + freeze
        kv.score_mode = "kvzip"
        kv.get_score = False
        assert kv.score[0].shape[-1] == kv.ctx_len, (
            f"snapkv score shape mismatch: {kv.score[0].shape} vs ctx_len {kv.ctx_len}"
        )

    @torch.inference_mode()
    def generate(
        self,
        query: Union[str, torch.Tensor],
        kv: Optional[Union[RetainCache, EvictCache]] = None,
        update_cache: bool = False,
    ) -> str:
        """ Obtain a model response to the query
            In default, we evict KV of query and generated answer after the generation by kv.slice (for multi-query evaluation).
            Set update_cache = True to enable multi-turn generation.
        """
        kv = self._init_kv(kv=kv)
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        input_ids = query
        if type(query) == str:
            input_ids = self.encode(query)
        if kv.prefill_ids is not None:
            # Huggingface Transformers model.generate requires full input tokens when using KV caches.
            # The inputs will be spliced to only contain new tokens as input[:, -kv.get_seq_length():].
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

        output = self.model.generate(input_ids, past_key_values=kv, **self.gen_kwargs)
        a_ids = output[:, len(input_ids[0]):-1]  # parse response
        a = self.decode(a_ids)

        if not update_cache:
            kv.slice(seen_token_prev)
        else:
            kv.prefill_ids = torch.cat([input_ids, a_ids], dim=1)
        return a

    @torch.inference_mode()
    def _prob(self, input_ids, kv=None, device="cuda") -> torch.Tensor:
        """ Obtain next token prediction probabilities
        """
        kv = self._init_kv(kv=kv)

        if isinstance(self.model, LlamaForCausalLMW8A8):
            output = self.__call__(input_ids,
                                   kv,
                                   update_cache=False,
                                   return_logits=True,
                                   is_prompt=False)
            output = output[0]
        else:
            output = self.__call__(input_ids, kv, update_cache=False, return_logits=True)
            output = output.logits[0]
        output = inplace_softmax(output).squeeze()

        if device == "cpu":
            return output.cpu()
        return output
