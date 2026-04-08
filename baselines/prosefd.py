"""
This file contains attention layers and related utils. 
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, Tuple
from torch import Tensor

from rotary_embedding_torch import RotaryEmbedding

N_MAX_POSITIONS = 1024  # maximum input sequence length



class DotCfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            # Important: raise AttributeError so getattr(obj, 'x', default) can use its default
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v
    def get(self, k, d=None):
        return super().get(k, d)

"""
--------------- Attention Variants ---------------
"""


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.batch_first = True

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        is_causal=False,
        need_weights=False,
        rotary_emb=None,
    ):
        bs, seq_len, _ = query.size()
        k_len = key.size(1)

        # compute projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # split heads (bs, seq_len, dim) -> (bs, n_heads, seq_len, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, k_len, self.num_heads, self.head_dim)
        v = v.view(bs, k_len, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (bs, n_head, seq_len, head_dim)

        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        # process and merge masks
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bs,
                k_len,
            ), f"expecting key_padding_mask shape of {(bs, k_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bs, 1, 1, k_len).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        dropout_p = 0.0 if not self.training else self.dropout

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )  # (bs, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output), None


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom implementation of pytorch's TransformerEncoderLayer
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        rotary=False,
        custom_attn=False,
        norm=nn.LayerNorm,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.TransformerEncoderLayer, self).__init__()
        if custom_attn:
            assert batch_first
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
            self.rotary = rotary
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs
            )
            self.rotary = False

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first

        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm1 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal, rotary_emb=rotary_emb
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal, rotary_emb=rotary_emb)
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        if self.rotary:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                rotary_emb=rotary_emb,
            )[0]
        else:
            x = self.self_attn(
                x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, is_causal=is_causal
            )[0]
        return self.dropout1(x)


class CustomTransformerEncoder(nn.Module):
    """
    Custom implementation of pytorch's TransformerEncoder
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
    """

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
        config=None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

        if config is not None and config.rotary:
            self.rotary_emb = RotaryEmbedding(dim=config.dim_emb // config.n_head // 2)
            self.rotary = True
        else:
            self.rotary_emb = None
            self.rotary = False

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal: Optional[bool] = None):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        batch_first = self.layers[0].self_attn.batch_first
        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                rotary_emb=self.rotary_emb,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class CausalTransformerDecoder(nn.TransformerDecoder):
    """
    Decoder attention (in encoder-decoder transformer) which supports kv-caching during evaluation.

    The complexity goes from seq_len^3 to seq_len^2.

    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).

    Source: https://github.com/alex-matton/causal-transformer-decoder/
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
        cache: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            tgt_mask: assumed to be a causal mask and will be ignored.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            if self.norm is not None:
                output = self.norm(output)

            return output

        if cache is None:
            assert tgt.size(0) == 1

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        if self.norm is not None:
            output = self.norm(output)

        return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Encoder-decoder attention layer.

    Source: https://github.com/alex-matton/causal-transformer-decoder/
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """

        if self.training:
            return super().forward(
                tgt,
                memory,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.size(0), tgt.device),
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.
        if self.norm_first:
            tgt_last_tok = tgt[-1:, :, :]

            tgt = self.norm1(tgt)

            # self attention part
            tmp_tgt = self.self_attn(
                tgt[-1:, :, :],
                tgt,
                tgt,
                attn_mask=None,  # not needed because we only care about the last token
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)

            # encoder-decoder attention
            if memory is not None:
                tgt_last_tok = tgt_last_tok + self._mha_block(
                    self.norm2(tgt_last_tok), memory, memory_mask, memory_key_padding_mask
                )

            # final feed-forward network
            tgt_last_tok = tgt_last_tok + self._ff_block(self.norm3(tgt_last_tok))
        else:
            tgt_last_tok = tgt[-1:, :, :]

            # self attention part
            tmp_tgt = self.self_attn(
                tgt_last_tok,
                tgt,
                tgt,
                attn_mask=None,  # not needed because we only care about the last token
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
            tgt_last_tok = self.norm1(tgt_last_tok)

            # encoder-decoder attention
            if memory is not None:
                tgt_last_tok = self.norm2(
                    tgt_last_tok + self._mha_block(tgt_last_tok, memory, memory_mask, memory_key_padding_mask)
                )

            # final feed-forward network
            tgt_last_tok = self.norm3(tgt_last_tok + self._ff_block(tgt_last_tok))
        return tgt_last_tok


class CausalTransformerEncoder(nn.TransformerEncoder):
    """
    Decoder-only attention which supports kv-caching during evaluation.

    The complexity goes from seq_len^3 to seq_len^2.

    """

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        cache: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            src_mask: assumed to be a causal mask and will be ignored.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        if self.training:
            if cache is not None:
                raise ValueError("Cache should be None during training")

            return super().forward(
                src=src,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        output = src

        new_token_cache = []
        if cache is None:
            # cache everything for the start
            for i, mod in enumerate(self.layers):
                output = mod(src=output, first=True)
                new_token_cache.append(output)
        else:
            # only cache the last token
            for i, mod in enumerate(self.layers):
                output = mod(src=output)
                new_token_cache.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        if self.norm is not None:
            output = self.norm(output)

        return output, new_cache


class CausalDecoderOnlyLayer(nn.TransformerEncoderLayer):
    """
    Decoder-only attention layer.

    Source: https://github.com/alex-matton/causal-transformer-decoder/
    """

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        first: bool = False,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerEncoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """

        if self.training:

            return super().forward(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.
        if self.norm_first:
            if first:
                input = src
                src = self.norm1(src)

                # self attention part
                tmp_src = self.self_attn(
                    src,
                    src,
                    src,
                    attn_mask=nn.Transformer.generate_square_subsequent_mask(src.size(0), src.device),
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False,
                    is_causal=True,
                )[0]
                src_last_tok = input + self.dropout1(tmp_src)
            else:
                input = src[-1:, :, :]
                src = self.norm1(src)
                # self attention part
                tmp_src = self.self_attn(
                    src[-1:, :, :],
                    src,
                    src,
                    attn_mask=None,  # not needed because we only care about the last token
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False,
                    is_causal=is_causal,
                )[0]
                src_last_tok = input + self.dropout1(tmp_src)

            # final feed-forward network
            src_last_tok = src_last_tok + self._ff_block(self.norm2(src_last_tok))
        else:
            if first:
                src_last_tok = src
                src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0), src.device)
                is_causal = True
            else:
                src_last_tok = src[-1:, :, :]
                src_mask = None

            # self attention part
            tmp_src = self.self_attn(
                src_last_tok,
                src,
                src,
                attn_mask=src_mask,  # not needed because we only care about the last token
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
            src_last_tok = src_last_tok + self.dropout1(tmp_src)
            src_last_tok = self.norm1(src_last_tok)

            # final feed-forward network
            src_last_tok = self.norm2(src_last_tok + self._ff_block(src_last_tok))

        return src_last_tok


class A:
    pass


class OperatorDecoderLayer(nn.TransformerDecoderLayer):
    """OperatorDecoderLayer is made up of multi-head-attn and feedforward network.
    (It is the usual encoder-decoder attention without the self-attention layers)

    Check https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html for details
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        rotary=False,
        custom_attn=False,
        norm=nn.LayerNorm,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.TransformerDecoderLayer, self).__init__()
        if custom_attn:
            assert batch_first
            self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
            self.rotary = rotary
            assert not rotary, "currently not supported for operator layers"
        else:
            self.multihead_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
            )
            self.rotary = False

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first

        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm1 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # small hack to handle pytorch TransformerDecoder
        self.self_attn = A()
        self.self_attn.batch_first = batch_first

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

            tgt_mask, tgt_key_padding_mask, tgt_is_causal: NOT needed as there is not
                self-attention, and will be ignored.

        Shape:
            see the docs in Transformer class.
        """

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal, rotary_emb=rotary_emb)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal, rotary_emb=rotary_emb))
            x = self.norm2(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        rotary_emb=None,
    ) -> Tensor:
        if self.rotary:
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                rotary_emb=rotary_emb,
            )[0]
        else:
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=False,
            )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


"""
--------------- Positional Embeddings ---------------
"""


class SinusoidalPE(nn.Module):
    """
    Sinusoidal positional embedding.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = N_MAX_POSITIONS):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, batch_first: bool = True) -> Tensor:
        """
        Arguments:
            x: Tensor [batch_size, seq_len, embedding_dim] if batch_first
                      [seq_len, batch_size, embedding_dim] otherwise
        """

        if batch_first:
            x = x + self.pe[: x.size(1)].transpose(0, 1)
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LearnablePE(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = N_MAX_POSITIONS):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = Embedding(max_len, d_model)

    def forward(self, x: Tensor, positions: Optional[Tensor] = None, batch_first: bool = True) -> Tensor:
        """
        Arguments:
            x: Tensor [batch_size, seq_len, embedding_dim] if batch_first
                      [seq_len, batch_size, embedding_dim] otherwise
            positions: Tensor [batch_size, seq_len]
        """
        seq_len = x.size(1) if batch_first else x.size(0)
        if positions is None:
            positions = x.new(seq_len).long()
            positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (1, seq_len)

        pe = self.pe(positions)  # (1, seq_len, d_model)
        if batch_first:
            x = x + pe.expand_as(x)
        else:
            x = x + pe.transpose(0, 1).expand_as(x)

        return self.dropout(x)


def get_embeddings(size, type=None):
    if type is None:
        patch_embeddings = nn.Parameter(torch.randn(*size))
    elif type == "normalize":
        dim = size[-1]
        patch_embeddings = nn.Parameter((dim**-0.5) * torch.randn(*size))
    elif type == "bert":
        patch_embeddings = nn.Parameter(torch.empty(*size).normal_(std=0.02))
    else:
        raise ValueError(f"Unknown type for embedding: {type}")
    return patch_embeddings


"""
--------------- Helper functions ---------------
"""


def get_padding_mask(lengths, max_len=None):
    """
    Input:
        lengths:           LongTensor (bs, )  length of each example
        max_len:           Optional[int]      if None, max_len = lengths.max()
    Output:
        key_padding_mask:  BoolTensor (bs, max_len)    (positions with value True are padding)
    """
    if max_len is None:
        max_len = lengths.max().item()

    bs = lengths.size(0)
    key_padding_mask = torch.arange(max_len, device=lengths.device).expand(bs, max_len) >= lengths.unsqueeze(1)
    return key_padding_mask


def get_block_attn_mask(block_size: int, n_repeat: int, device=torch.device("cpu")):
    """
    Output:
        attn_mask: BoolTensor (block_size * n_repeat, block_size * n_repeat) block diagonal matrix with identity blocks
    """
    blocks = [torch.ones(block_size, block_size, device=device)] * n_repeat
    return torch.block_diag(*blocks).bool()


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class GroupNorm(nn.Module):
    """
    Group norm for sequence. Expects input shape (bs, seq_len, d) and splits group only in the last dimension.
    """

    def __init__(self, num_groups, num_channels, affine=True, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_channels))

    def forward(self, x):
        bs, seq_len, d = x.size()
        x = x.view(bs, seq_len, self.num_groups, d // self.num_groups)

        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x.view(bs, seq_len, d)

        if self.affine:
            x = x * self.weight + self.bias

        return x

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(**self.__dict__)



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange

from logging import getLogger

logger = getLogger()


def get_embedder(config, x_num, max_output_dim):
    if config.type == "linear":
        embedder = LinearEmbedder
    elif config.type == "conv":
        embedder = ConvEmbedder
    else:
        raise ValueError(f"Unknown embedder type: {config.type}")

    return embedder(config, x_num, max_output_dim)


def patchify(data: torch.Tensor, patch_num: int):
    """
    Input:
        (bs, nt, px, py, d)
    Output:
        (bs, nt, p*p, x*y*d)
    """
    bs, nt, px, py, d = data.size()
    p = patch_num
    x = px // p
    y = py // p

    data = data.view(bs, nt, p, x, p, y, d).permute(
        0, 1, 2, 4, 3, 5, 6
    )  # (bs, nt, p, x, p, y, d) -> (bs, nt, p, p, x, y, d)

    data = data.reshape((bs, nt, p * p, x * y * d))
    return data


def depatchify(data: torch.Tensor, patch_num: int, x: int, y: int, d: int):
    """
    Input:
        (bs, nt, p*p, x*y*d)
    Output:
        (bs, nt, px, py, d)
    """
    bs = data.size(0)
    nt = data.size(1)
    p = patch_num

    data = data.view(bs, nt, p, p, x, y, d).permute(
        0, 1, 2, 4, 3, 5, 6
    )  # (bs, nt, p, p, x, y, d) -> (bs, nt, p, x, p, y, d)

    data = data.reshape((bs, nt, p * x, p * y, d))
    return data


def layer_initialize(layer, mode="zero", gamma=0.01):
    # re-initialize given layer to have small outputs
    if mode == "zero":
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif mode == "uniform":
        nn.init.uniform_(layer.weight, -gamma, gamma)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -gamma, gamma)
    else:
        raise ValueError(f"Unknown mode {mode}")


class LinearEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        # for encoder part
        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num * config.patch_num, self.dim))

        self.time_proj = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.pre_proj = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

        # for decoder part

        self.post_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.patch_dim_output),
        )

    def encode(self, data, times):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs/1, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """
        bs = data.size(0)
        data = patchify(data, self.config.patch_num)  # (bs, input_len, p*p, x*y*d)
        data = self.pre_proj(data)  # (bs, input_len, p*p, dim)

        time_embeddings = self.time_proj(times)[:, :, None]  # (bs/1, input_len, 1, dim)
        data = ((data + time_embeddings) + self.patch_position_embeddings).reshape(bs, -1, self.dim)
        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        data_output = self.post_proj(data_output)  # (bs, query_len, patch_dim)
        data_output = data_output.view(
            bs, -1, self.config.patch_num_output * self.config.patch_num_output, self.patch_dim_output
        )  # (bs, output_len, p*p, patch_dim)

        data_output = depatchify(
            data_output,
            self.config.patch_num_output,
            self.patch_resolution_output,
            self.patch_resolution_output,
            self.data_dim,
        )  # (bs, output_len, x_num, x_num, data_dim)

        return data_output


class ConvEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part

        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num * config.patch_num, self.dim))

        self.time_embed_type = config.get("time_embed", "continuous")
        if self.time_embed_type == "continuous":
            self.time_proj = nn.Sequential(
                nn.Linear(1, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.time_embed = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if config.get("early_conv", 0):
            n_conv_layers = math.log2(self.patch_resolution)
            assert n_conv_layers.is_integer(), f"patch_resolution {self.patch_resolution} must be a power of 2"
            n_conv_layers = int(n_conv_layers)
            kernel_size = [3] * n_conv_layers + [1]
            stride = [2] * n_conv_layers + [1]
            padding = [1] * n_conv_layers + [0]
            channels = [data_dim] + [self.dim // (2**i) for i in range(n_conv_layers - 1, 0, -1)] + [self.dim, self.dim]

            self.conv_proj = nn.Sequential()
            for i in range(len(kernel_size)):
                self.conv_proj.append(
                    nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                    )
                )
                if i < len(kernel_size) - 1:
                    self.conv_proj.append(nn.GELU())
        else:
            # regular vit patch embedding
            self.conv_proj = nn.Sequential(
                nn.Conv2d(
                    in_channels=data_dim,
                    out_channels=self.dim,
                    kernel_size=self.patch_resolution,
                    stride=self.patch_resolution,
                ),
                nn.GELU(),
                nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1),
            )

        ## for decoder part

        self.conv_dim = config.get("conv_dim", self.dim // 4)

        if config.get("deep", 0):
            self.post_proj = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.GELU(),
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.GELU(),
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1),
            )
        else:
            self.post_proj = nn.Sequential(
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1),
            )

        if config.get("initialize_small_output", 0):
            layer_initialize(self.post_proj[-1], mode=config.initialize_small_output)

    def encode(self, data, times):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """

        bs = data.size(0)
        data = rearrange(data, "b t h w c -> (b t) c h w")
        data = self.conv_proj(data)  # (bs*input_len, d, patch_num, patch_num)
        data = rearrange(data, "(b t) d h w -> b t (h w) d", b=bs)  # (bs, input_len, p*p, dim)

        if self.time_embed_type == "continuous":
            time_embeddings = self.time_proj(times)[:, :, None]  # (bs, input_len, 1, dim)
        else:
            time_embeddings = self.time_embed[:, : times.size(1)]  # (bs, input_len, 1, dim)

        data = ((data + time_embeddings) + self.patch_position_embeddings).reshape(bs, -1, self.dim)
        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)
        data_output = self.post_proj(data_output)  # (bs*output_len, data_dim, x_num, x_num)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output

"""
This file contains complete transformer encoder/decoder modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from functools import partial
import math

logger = getLogger()

# ---------------------------------------------------------------------
# Compatibility helpers (Python < 3.10, mixed PyTorch versions)
# ---------------------------------------------------------------------

class _RMSNormFallback(nn.Module):
    """
    Minimal RMSNorm fallback for environments without torch.nn.RMSNorm.
    Normalizes over the last dimension.
    """
    def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            w = self.weight
            if w.dim() == 0:
                w = w.view(1)
            while w.dim() < x.dim():
                w = w.unsqueeze(0)
            x = x * w
        return x

def resolve_norm_cls(config):
    """
    Resolves the normalization class used throughout the file.
    Supports config.norm in {"group", "rms", "layer"}, defaults to "layer".
    Returns a callable class so you can do norm(dim_emb).
    """
    norm_name = getattr(config, "norm", "layer")
    if norm_name == "group":
        # GroupNorm(num_groups=8, num_channels=<dim>)
        return partial(nn.GroupNorm, 8)
    elif norm_name == "rms":
        return getattr(nn, "RMSNorm", _RMSNormFallback)
    else:
        return nn.LayerNorm

def futureproof_autoregressive_mask(seq_len, device):
    """
    Returns a causal mask on the right device across PyTorch versions.
    """
    m = nn.Transformer.generate_square_subsequent_mask(seq_len)
    return m.to(device)

"""
Transformer Data modules
"""

class TransformerDataEncoder(nn.Module):
    """
    Encoder Transformer for data
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            if config.get("custom_encoder", 0):
                norm = resolve_norm_cls(config)
                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=config.rotary,
                        custom_attn=config.custom_attn,
                        norm=norm,
                    ),
                    num_layers=config.n_layer,
                    norm=norm(config.dim_emb) if config.norm_first else None,
                    config=config,
                )
            else:
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                    ),
                    num_layers=config.n_layer,
                    norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
                )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        """
        x: Tensor (bs, slen, dim)
        """
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


class TransformerDataDecoder(nn.Module):
    """
    Encoder-decoder Transformer for data (autoregressive)
    """

    def __init__(self, config, output_dim):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if self.config.kv_cache:
            self.transformer_decoder = CausalTransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )
        else:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        self.post_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, output_dim))

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
    ):
        """
        Inputs:
            tgt:    Tensor (bs, output_len, dim)
                    should be data in the range [input_len-1, max_len-1)
            memory: Tensor (bs, input_len, dim)
                    output from encoder
        Output:
            tgt_output: Tensor (bs, output_len, output_dim)
                        should corresponds to data in the range [input_len, max_len)
        """

        if self.positional_embedding is not None:
            tgt = self.positional_embedding(tgt)  # (bs, output_len, dim)

        tgt = tgt.transpose(0, 1)  # (output_len, bs, dim)
        memory = memory.transpose(0, 1)  # (input_len, bs, dim)

        if self.config.kv_cache:
            tgt_mask = None  # (causal decoder handles this automatically)
        else:
            tgt_mask = futureproof_autoregressive_mask(tgt.size(0), tgt.device)

        decoded = self.transformer_decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask
        )  # (output_len, bs, dim)

        tgt_output = self.post_proj(decoded).transpose(0, 1)  # (bs, output_len, output_dim)

        return tgt_output

    def generate(
        self,
        encoded,
        initial,
        input_times,
        output_times,
        all_times,
        pre_proj,
    ):
        """
        For evaluation/testing only.
        Inputs:
            encoded:      Tensor (input_len, bs, dim)
            initial:      Tensor (bs, query_dim+data_dim)
            input_times:  Tensor (input_len, query_dim)
            output_times: Tensor (output_len, query_dim)
            all_times:    Tensor (max_len, query_dim)
            pre_proj:     Projection for data input
        Output:
            data_output:  Tensor (output_len, bs, data_dim)

        """
        cur_len = 1
        output_len = output_times.size(0)
        bs = initial.size(0)
        query_dim = output_times.size(1)
        data_dim = initial.size(1) - query_dim
        generated = torch.zeros(output_len, bs, data_dim, dtype=initial.dtype, device=initial.device)

        cache = None
        tgt = pre_proj(initial)[None]  # (1, bs, dim)

        # generation loop
        while cur_len <= output_len:  # max length of generation

            if self.config.kv_cache:
                decoded, cache = self.transformer_decoder(tgt=tgt, memory=encoded, cache=cache)  # (cur_len, bs, dim)
            else:
                tgt_mask = futureproof_autoregressive_mask(tgt.size(0), tgt.device)
                decoded = self.transformer_decoder(tgt=tgt, memory=encoded, tgt_mask=tgt_mask)  # (cur_len, bs, dim)

            new_data = self.post_proj(decoded[-1])  # (bs, data_dim)

            generated[cur_len - 1] = new_data

            new_input = torch.cat(
                [output_times[cur_len - 1][None].expand(bs, query_dim), new_data], dim=1
            )  # (bs, query_dim + data_dim)

            tgt = torch.cat([tgt, pre_proj(new_input[None])], dim=0)  # (cur_len + 1, bs, dim)

            cur_len += 1

        return generated


class DataOperatorDecoder(nn.Module):
    """
    Operator Decoder for data
    """

    def __init__(self, config, output_len=1, space_len=None):
        super().__init__()

        self.config = config

        self.dim = config.dim_emb

        self.time_embed_type = config.get("time_embed", "continuous")
        if self.time_embed_type == "continuous":
            self.time_proj = nn.Sequential(
                nn.Linear(config.query_dim, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.time_embed = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if space_len is None:
            space_len = config.patch_num_output**2

        self.patch_position_embeddings = get_embeddings((1, 1, space_len, self.dim))

        if config.self_attn > 0:
            # self attn + cross attn + ffn

            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

            if config.self_attn == 1:
                # self attn is restricted to patches with the same t
                self_attn_mask = get_block_attn_mask(
                    block_size=config.patch_num_output * config.patch_num_output, n_repeat=output_len
                )
                self.register_buffer("self_attn_mask", self_attn_mask)
        else:
            # cross attn + ffn
            norm = resolve_norm_cls(config)

            self.transformer_decoder = nn.TransformerDecoder(
                OperatorDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=config.norm_first,
                    custom_attn=config.get("custom_attn", 0),
                    norm=norm,
                ),
                num_layers=config.n_layer,
                # norm=norm(config.dim_emb) if config.norm_first else None,
                norm=norm(config.dim_emb) if (config.norm_first and config.final_ln) else None,
            )

    def get_query_emb(self, times):
        """
        Input:
            times:     Tensor (bs/1, output_len, 1)
        Output:
            query_emb: Tensor (bs/1, query_len, dim)
                       query_len = output_len * patch_num * patch_num
        """

        bs, output_len, query_dim = times.size()

        if self.time_embed_type == "continuous":
            times = self.time_proj(times)[:, :, None]  # (bs/1, output_len, 1, dim)
        else:
            times = self.time_embed[:, :output_len]  # (1, input_len, 1, dim)

        return (times + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    def forward(self, src, query_emb, src_key_padding_mask=None, tgt_mask=None):
        """
        src:         Tensor (bs, src_len, dim)
        query_emb:   Tensor (bs, query_len, dim)
        src_key_padding_mask: Optional[Tensor] (bs, src_len)
        tgt_mask:             Optional[Tensor] (query_len, query_len) or (bs*n_head, query_len, query_len)
        """

        if tgt_mask is None and self.config.self_attn == 1:
            tgt_mask = self.self_attn_mask

        x = self.transformer_decoder(query_emb, src, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)

        return x  # (bs, query_len, dim)


"""
Transformer Symbol Modules
"""

class TransformerSymbolEncoder(nn.Module):
    """
    Encoder Transformer for Symbols
    """

    def __init__(self, config, id2word):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            if config.get("custom_encoder", 0):
                norm = resolve_norm_cls(config)
                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=config.rotary,
                        custom_attn=config.custom_attn,
                        norm=norm,
                    ),
                    num_layers=config.n_layer,
                    norm=norm(config.dim_emb) if config.norm_first else None,
                    config=config,
                )
            else:
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                    ),
                    num_layers=config.n_layer,
                    norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
                )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)

        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        """
        x:                    LongTensor (bs, slen)
        mask:                 Optional[Tensor] (bs, slen, slen)
        src_key_padding_mask: Optional[BoolTensor] (bs, slen)         (positions with value True will be ignored)
        """

        x = self.word_embeddings(x)  # (bs, slen, dim)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


class TransformerSymbolDecoder(nn.Module):
    """
    Encoder-decoder Transformer for Symbols
    """

    def __init__(self, config, id2word):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if self.config.kv_cache:
            self.transformer_decoder = CausalTransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )
        else:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)

        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

        # output layer
        self.proj = nn.Linear(self.dim, self.n_words, bias=True)
        if config.share_inout_emb:
            self.proj.weight = self.word_embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        tgt,
        memory,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Inputs:
            tgt:                     LongTensor (bs, output_len)
            memory:                  LongTensor (bs, input_len, dim)        (output from encoder)
            tgt_key_padding_mask:    Optional[BoolTensor] (bs, output_len)  (True for positions that should be ignored)
            memory_key_padding_mask: Optional[BoolTensor] (bs, input_len)
        Output:
            decoded:                 Tensor (bs, output_len, dim)
        """

        tgt = self.word_embeddings(tgt)  # (bs, output_len, dim)

        if self.positional_embedding is not None:
            tgt = self.positional_embedding(tgt)  # (bs, output_len, dim)

        tgt = tgt.transpose(0, 1)  # (output_len, bs, dim)
        memory = memory.transpose(0, 1)  # (input_len, bs, dim)

        if self.config.kv_cache:
            tgt_mask = None  # (causal decoder handles this automatically)
        else:
            tgt_mask = futureproof_autoregressive_mask(tgt.size(0), tgt.device)

        decoded = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True,
        )  # (output_len, bs, dim)

        return decoded.transpose(0, 1)  # (bs, output_len, dim)

    def predict(self, output, pred_mask, y):
        """
        Given the last hidden state, compute word scores and the loss.
        Inputs:
            output       Tensor     (bs, output_len, dim)
            pred_mask    BoolTensor (bs, output_len), filled with 1 when we need to predict a word
            y            LongTensor (pred_mask.sum(), )
        """
        x = output[pred_mask.unsqueeze(-1).expand_as(output)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def generate(self, memory, memory_key_padding_mask=None, max_len=200, sample_temperature=None):
        """
        For evaluation/testing only.
        Inputs:
            memory:                  Tensor (bs, memory_len, dim)
            memory_key_padding_mask: Optional[BoolTensor] (bs, memory_len)
        Output:
            generated:               LongTensor(bs, cur_len)
                                     e.g. <BOS> W1 W2 W3 <EOS> <PAD>
                                          <BOS> W1 W2 W3   W4  <EOS>
            gen_len:                 LongTensor(bs)
                                     e.g. [5, 6]
        """
        bs = memory.size(0)
        memory = memory.transpose(0, 1)  # (memory_len, bs, dim)

        # generated sentences
        generated = torch.full((max_len, bs), self.pad_index, dtype=torch.long, device=memory.device)
        generated[0].fill_(self.bos_index)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cache = None
        cur_len = 1
        gen_len = torch.ones(bs, dtype=torch.long, device=memory.device)  # (bs, )
        unfinished_sents = torch.ones(bs, dtype=torch.long, device=memory.device)  # (bs, )

        # generation loop
        while cur_len < max_len:  # max length of generation
            tgt = generated[:cur_len]  # (cur_len, bs)
            tgt = self.word_embeddings(tgt)
            if self.positional_embedding is not None:
                tgt = self.positional_embedding(tgt, batch_first=False)  # (output_len, bs, dim)

            if self.config.kv_cache:
                decoded, cache = self.transformer_decoder(
                    tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask, cache=cache
                )  # (cur_len, bs, dim)
            else:
                tgt_mask = futureproof_autoregressive_mask(tgt.size(0), tgt.device)
                decoded = self.transformer_decoder(
                    tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask
                )  # (cur_len, bs, dim)

            scores = self.proj(decoded[-1])  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximum length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        generated = generated[:cur_len].transpose(0, 1)  # (bs, cur_len)
        return generated, gen_len


"""
Transformer Fusion Module
"""

class TransformerFusion(nn.Module):
    """
    Fusion Transformer
    """

    def __init__(self, config, num_types=2):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            if config.get("custom_encoder", 0):
                norm = resolve_norm_cls(config)
                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=config.rotary,
                        custom_attn=config.custom_attn,
                        norm=norm,
                    ),
                    num_layers=config.n_layer,
                    norm=norm(config.dim_emb) if config.norm_first else None,
                    config=config,
                )
            else:
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                    ),
                    num_layers=config.n_layer,
                    norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
                )

        if config.type_embeddings:
            self.type_embeddings = Embedding(num_types, self.dim)
        else:
            self.type_embeddings = None

    def forward(self, x0, x1, key_padding_mask0=None, key_padding_mask1=None):
        """
        x0: Tensor (bs, slen0, dim)
        x1: Tensor (bs, slen1, dim)
        key_padding_mask0: Optional[BoolTensor] (bs, slen0)           (True for positions that should be ignored)
        key_padding_mask1: Optional[BoolTensor] (bs, slen1)
        """

        bs = x0.size(0)

        if self.type_embeddings is not None:
            type0 = torch.zeros(1, 1, dtype=torch.long, device=x0.device)
            type1 = torch.ones(1, 1, dtype=torch.long, device=x1.device)
            x0 = x0 + self.type_embeddings(type0).expand_as(x0)
            x1 = x1 + self.type_embeddings(type1).expand_as(x1)

        x = torch.cat([x0, x1], dim=1)  # (bs, slen0+slen1, dim)

        if key_padding_mask0 is None and key_padding_mask1 is None:
            fused_mask = None
        else:
            if key_padding_mask0 is None:
                key_padding_mask0 = torch.zeros(bs, x0.size(1), dtype=torch.bool, device=x0.device)
            if key_padding_mask1 is None:
                key_padding_mask1 = torch.zeros(bs, x1.size(1), dtype=torch.bool, device=x1.device)
            fused_mask = torch.cat([key_padding_mask0, key_padding_mask1], dim=1)  # (bs, slen0+slen1)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=fused_mask)

        return x, fused_mask  # (bs, slen0+slen1, dim)
"""
Final model wrappers. 
"""

import torch
import torch.nn as nn
from einops import rearrange

from logging import getLogger

logger = getLogger()


class PROSE_2to1(nn.Module):
    """
    Wrapper for the PROSE model (2to1).
    """

    def __init__(self, config, symbol_env, x_num, max_output_dim, output_len=1):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.data_encoder = TransformerDataEncoder(config.data_encoder)
        self.symbol_encoder = TransformerSymbolEncoder(config.symbol_encoder, symbol_env.equation_id2word)
        self.fusion = TransformerFusion(config.fusion)

        if config.embedder.type == "fourier":
            p = config.data_decoder.patch_num_output
            space_len = p * (p // 2 + 1)
        else:
            space_len = None
        self.data_decoder = DataOperatorDecoder(config.data_decoder, output_len, space_len)

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if p.requires_grad]):,}\n"
        s += f"\tSymbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if p.requires_grad]):,}\n"
        s += f"\tFusion:          {sum([p.numel() for p in self.fusion.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Decoder:    {sum([p.numel() for p in self.data_decoder.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, input_times, output_times, symbol_input, symbol_padding_mask=None, **kwargs):
        """
        Inputs:
            data_input:          Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:         Tensor     (bs/1, input_len, 1)
            output_times:        Tensor     (bs/1, output_len, 1)

            symbol_input:        LongTensor           (bs, symbol_len)
            symbol_padding_mask: Optional[BoolTensor] (bs, symbol_len)     symbol padding mask, positions with True are padding

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        if self.config.get("carry_last_frame", 0):
            last_frame = data_input[:, -1:].clone()  # (bs, 1, x_num, x_num, data_dim)

        bs = data_input.size(0)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, input_len, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = input_len * patch_num * patch_num
        """

        data_input = self.embedder.encode(data_input, input_times)  # (bs, data_len, dim)

        """
        Step 2: Encode + Fusion
            data_input:   Tensor     (bs, data_len, dim)
            symbol_input: LongTensor (bs, symbol_len)
        """

        data_encoded = self.data_encoder(data_input)  # (bs, data_len, dim)
        symbol_encoded = self.symbol_encoder(
            symbol_input, src_key_padding_mask=symbol_padding_mask
        )  # (bs, symbol_len, dim)

        fused, fused_mask = self.fusion(
            x0=data_encoded,
            x1=symbol_encoded,
            key_padding_mask0=None,
            key_padding_mask1=symbol_padding_mask,
        )  # (bs, data_len+symbol_len, dim)

        """
        Step 3: Decode data
        """

        query_emb = self.data_decoder.get_query_emb(output_times)  # (bs/1, query_len, dim)
        if query_emb.size(0) == 1:
            query_emb = query_emb.expand(bs, -1, -1)

        data_output = self.data_decoder(
            src=fused, query_emb=query_emb, src_key_padding_mask=fused_mask
        )  # (bs, query_len, dim)

        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)

        if self.config.get("carry_last_frame", 0):
            data_output = data_output + last_frame

        return data_output
class PROSE_1to1(nn.Module):
    """
    Wrapper for the PROSE model (1to1).
    """

    def __init__(self, config, symbol_env, x_num, max_output_dim, output_len=1):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.data_encoder = TransformerDataEncoder(config.data_encoder)

        # --- Fourier-aware query length (mirrors PROSE_2to1) ---
        if config.embedder.type == "fourier":
            p = config.data_decoder.patch_num_output
            space_len = p * (p // 2 + 1)
        else:
            space_len = None

        self.data_decoder = DataOperatorDecoder(config.data_decoder, output_len, space_len)

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum(p.numel() for p in self.embedder.parameters() if p.requires_grad):,}\n"
        s += f"\tData Encoder:    {sum(p.numel() for p in self.data_encoder.parameters() if p.requires_grad):,}\n"
        s += f"\tData Decoder:    {sum(p.numel() for p in self.data_decoder.parameters() if p.requires_grad):,}"
        return s

    def forward(self, mode, **kwargs):
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, input_times, output_times, **kwargs):
        bs = data_input.size(0)
        data_input = self.embedder.encode(data_input, input_times)   # (bs, data_len, dim)
        data_encoded = self.data_encoder(data_input)                  # (bs, data_len, dim)
        query_emb = self.data_decoder.get_query_emb(output_times)     # (bs/1, query_len, dim)
        if query_emb.size(0) == 1:
            query_emb = query_emb.expand(bs, -1, -1)
        data_output = self.data_decoder(src=data_encoded, query_emb=query_emb, src_key_padding_mask=None)
        data_output = self.embedder.decode(data_output)               # (bs, output_len, x_num, x_num, data_dim)
        return data_output