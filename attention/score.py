# ------------------------------------------------------------------------------
# Original code from the KVzip repository (snu-mllab, MIT License)
# Licensed under The MIT License
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional


class KVScore():
    """ Functions to compute the score for the KV features. (kvcache.py)"""

    def __init__(self):
        self.n_heads_kv = None
        self.dtype = None
        self.device = None
        self.get_score = True
        self.causal_mask_score = None
        self.score = None
        self.sink = None
        self.start_idx, self.end_idx = None, None
        # SnapKV-mode params (used when score_mode == "snapkv")
        self.score_mode = "kvzip"     # "kvzip" (default) or "snapkv"
        self.snapkv_window = 32
        self.snapkv_kernel = 7
        self.snapkv_first_tokens = 4

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None
        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]

    def _update_score(self, layer_idx: int, score: torch.Tensor):
        self.score[layer_idx] = torch.cat([self.score[layer_idx], score], dim=-1)

    def _get_score(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int,
                   value_states: Optional[torch.Tensor] = None,
                   hidden_states: Optional[torch.Tensor] = None,
                   o_proj_weight: Optional[torch.Tensor] = None):
        """ Compute KV importance scores.
            # key_states: bsz x head_kv x k x dim, query_states: bsz x head x q x dim
            # value_states, hidden_states, o_proj_weight: only needed for kvzip_plus mode
        """
        if self.score_mode == "snapkv":
            return self._get_score_snapkv(query_states, key_states, layer_idx)
        if self.score_mode == "h2o":
            return self._get_score_h2o(query_states, key_states, layer_idx)

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        query_states_g = query_states.view(bsz, num_kv, -1, q_len, head_dim)
        key_states_cat = torch.cat(
            [
                key_states[:, :, :self.sink],  # sink tokens (generally system prompt)
                key_states[:, :, self.start_idx:self.end_idx],  # KV chunk in the cache
                key_states[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2)

        # bsz, head, 1, dim, k
        key_states_cat = key_states_cat.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx

        attn_weights = torch.matmul(query_states_g, key_states_cat) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)

        # bsz, head_kv, group, q, ctx_len (+sink +repeat span on last dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # not fp32

        if self.score_mode == "kvzip_plus":
            # KVzip+ (Jégou & Jeblick 2026, arxiv 2601.07891) — matches
            # NVIDIA/kvpress kvzip_press.py:KVzipPress.compress with kvzip_plus_normalization=True.
            #   attn' = attn * (1/||h_j||)      along q dim
            #   attn' = attn' * ||W_O_g v_i||   along ctx dim, per (KV head, group)
            #   score = amax_{g, q}(attn')
            assert value_states is not None and hidden_states is not None and o_proj_weight is not None, \
                "kvzip_plus needs value_states/hidden_states/o_proj_weight"

            num_heads_full = query_states.shape[1]
            n_rep = num_heads_full // num_kv

            # (a) Divide by ||h_j|| along q axis  (attn shape: b h g q L)
            #     hidden_states here spans the full concat (sink + ctx_chunk + repeat_chunk).
            #     Only the current q_len positions correspond to the query axis of attn_weights.
            h_slice = hidden_states[:, -q_len:]                          # bsz x q_len x H
            h_norm = h_slice.norm(dim=-1).clamp(min=1e-6)                # bsz x q_len
            attn_weights = torch.einsum("b h g q L, b q -> b h g q L",
                                        attn_weights, (1.0 / h_norm).to(attn_weights.dtype))

            # (b) Multiply by ||W_O v_i|| along ctx dim, per (KV head, group)
            H_dim = o_proj_weight.shape[0]
            Wo = o_proj_weight.transpose(0, 1).contiguous()              # (num_heads_full*hd, H)
            Wo = Wo.view(num_kv, n_rep, head_dim, H_dim).to(attn_weights.dtype)
            v_cat = torch.cat(
                [
                    value_states[:, :, :self.sink],
                    value_states[:, :, self.start_idx:self.end_idx],
                    value_states[:, :, -q_len:],
                ],
                dim=2,
            )                                                             # bsz x num_kv x k x hd
            v_cat = v_cat.unsqueeze(2).transpose(-2, -1).contiguous()    # bsz x num_kv x 1 x hd x k
            V = v_cat.repeat_interleave(n_rep, dim=2)                    # bsz x num_kv x g x hd x k
            # WoV[k]: for each (kv head, group, ctx-position, hidden-dim)
            WoV = torch.einsum("h g i j, b h g i t -> b h g t j", Wo, V.to(attn_weights.dtype))
            WoV_norm = WoV.norm(dim=-1)                                   # bsz x num_kv x g x k
            attn_weights = torch.einsum("b h g q L, b h g L -> b h g q L", attn_weights, WoV_norm)

            attn_weights = attn_weights[..., self.sink:self.sink + ctx_len]
            score = attn_weights.amax(dim=(-3, -2))                       # max over group, q
        else:
            attn_weights = attn_weights[..., self.sink:self.sink + ctx_len]
            score = attn_weights.amax(dim=(-3, -2))  # max over group, q  (original KVzip)

        self._update_score(layer_idx, score)

    def _get_score_snapkv(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """SnapKV-style query-agnostic scoring.
        Expects to be called once after full prefill, with `query_states` being the
        re-fed last window of context. `key_states` is the full updated cache.
        We use `mean over the last window queries` + max-pool smoothing
        + always-keep first_tokens (the very end window is excluded from selection,
         it lives in the appended-window region after the ctx span).
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        # GQA: (bsz, num_kv, group, q, dim)
        query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)

        # restrict keys to the ctx region we care about (sink + ctx range in cache)
        # `end_idx` is set to the full ctx end by scoring_snapkv before this is called.
        ctx_keys = torch.cat(
            [
                key_states[:, :, :self.sink],
                key_states[:, :, self.start_idx:self.end_idx],
            ],
            dim=2,
        )
        # (bsz, num_kv, 1, head_dim, k_len)
        ctx_keys = ctx_keys.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx

        attn_weights = torch.matmul(query_states, ctx_keys) / math.sqrt(head_dim)
        # softmax over all keys
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        # restrict to ctx span (drop sink for selection)
        attn_ctx = attn_weights[..., self.sink:self.sink + ctx_len]

        # mean over window queries and groups -> (bsz, num_kv, ctx_len)
        score = attn_ctx.mean(dim=(-3, -2))

        # max-pool smoothing
        k = self.snapkv_kernel
        score = nn.functional.max_pool1d(score, kernel_size=k, padding=k // 2, stride=1)

        # protect first_tokens and the trailing window (always retain)
        max_val = score.max().detach() + 1.0
        if self.snapkv_first_tokens > 0:
            score[..., :self.snapkv_first_tokens] = max_val
        win = min(self.snapkv_window, ctx_len)
        if win > 0:
            score[..., -win:] = max_val

        self._update_score(layer_idx, score)

    def _get_score_h2o(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """H2O prefill-attention scoring (counterpart baseline to KVzip).

        KVzip paper (arXiv:2505.23416, App. A "Baseline Methods"): "We implement
        the prefill version of H2O ... For each KV pair, we compute the MAXIMUM
        attention score received during prefilling ... superior over ... average.
        H2O ... utilizing self-attention scores from prefilling, while our method
        employs self-attention scores from reconstruction."

        Called once per prefill chunk (query=current chunk at original positions,
        key=full cache so far). score_j = max over all causal prefill queries of
        attn(i->j). Running-max is accumulated across chunks into a preallocated
        full-length buffer self.score[layer_idx] (sliced to ctx after prefill).
        Aggregation over GQA group + query axis is amax, matching KVzip.
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)
        k_total = key_states.size(2)
        seen = k_total - q_len  # absolute-position offset of this query chunk

        # GQA grouping: (bsz, num_kv, group, q, dim) x (bsz, num_kv, 1, dim, k_total)
        qg = query_states.view(bsz, num_kv, -1, q_len, head_dim)
        kk = key_states.unsqueeze(2).transpose(-2, -1)  # (bsz, num_kv, 1, dim, k_total)
        attn = torch.matmul(qg, kk) / math.sqrt(head_dim)  # (bsz, num_kv, g, q, k_total)

        # causal mask on the trailing q_len x q_len block: query row r (abs pos
        # seen+r) may attend key col seen+jj only if jj <= r. Earlier cols always ok.
        if q_len > 1:
            neg = torch.finfo(attn.dtype).min
            tri = torch.triu(
                torch.full((q_len, q_len), neg, device=attn.device, dtype=attn.dtype),
                diagonal=1,
            )
            attn[..., seen:] = attn[..., seen:] + tri

        attn = nn.functional.softmax(attn, dim=-1).to(query_states.dtype)
        # per-key max over group + query axis -> (bsz, num_kv, k_total)
        score = attn.amax(dim=(-3, -2))

        buf = self.score[layer_idx]  # preallocated (bsz, num_kv, full_prefill_len)
        buf[..., :k_total] = torch.maximum(buf[..., :k_total], score)

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """ Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """ Apply causal maksing
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    ##################################################################################################
    def compute_future_valid(self, ratio: float, level: str = "pair"):
        """Compute valid mask (True=keep) as if prune(ratio, level) was called.
        Does NOT mutate self.valid or self.pruned."""
        if "uniform" in level:
            valid, _ = self._threshold_uniform(self.score, ratio)
        else:
            valid, _ = self._threshold(self.score, ratio)
        return valid  # (n_layers, n_heads_kv, ctx_len)

    def _threshold(self, score: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores
        """
        if type(score) == list:
            score = torch.stack(score, dim=0)
        if ratio < 1:
            score_sort = torch.sort(score.reshape(-1), descending=True).values
            n = max(int(len(score_sort) * ratio) - 1, 0)
            thres = score_sort[n].item()
            valids = torch.where(score > thres, True, False).bool()
        else:
            valids = torch.ones_like(score, dtype=bool)
            thres = 0.

        return valids, thres

    def _threshold_uniform(self, scores: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores with uniform head budgets 
        """
        valids = []
        for nl, score in enumerate(scores):
            if ratio < 1:
                n_seq = score.size(-1)
                k = int(n_seq * ratio)
                _, topk_indices = torch.topk(score, k, dim=-1)
                valid = torch.zeros_like(score, dtype=bool)
                valid.scatter_(-1, topk_indices, True)
            else:
                valid = torch.ones_like(score, dtype=bool)
            valids.append(valid)

        valids = torch.stack(valids)
        return valids, 0


class HybridKVScore(KVScore):

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None

        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.num_static_layers)
        ]

    
    def _get_score(self, query_states, key_states, layer_idx):
        if layer_idx in self.layer_id_to_static_id:
            static_layer_idx = self.layer_id_to_static_id[layer_idx]
            super()._get_score(query_states, key_states, static_layer_idx)

