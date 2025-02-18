# Codes are available at https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L57
import copy
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_

from original.activation import MultiheadAttention
from original.container import ModuleList
from original.module import Module
from original.normalization import LayerNorm
from original.dropout import Dropout
from original.linear import Linear

# from transfomer import *, only the following classes are imported
__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer"
]

def _generate_square_subsequent_mask(
    sz:int, # sequence length
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    '''
    tensor([[  0., -inf, -inf, -inf, -inf],
            [  0.,   0., -inf, -inf, -inf],
            [  0.,   0.,   0., -inf, -inf],
            [  0.,   0.,   0.,   0., -inf],
            [  0.,   0.,   0.,   0.,   0.]])
    '''
    return torch.triu(
        torch.full((sz,sz), float("-inf"),dtype=dtype, device=device),
        diagonal=1
    )

def _get_seq_len(src:Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        # (torch.Tensor) is_nested -> Returns True if the tensor contains nested tensors (Tensors with different lengths)
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            return src_size[0]
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

class Transformer(Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor],Tensor]] = F.relu,
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device = None,
        dtype = None,
    ) -> None:
        factory_kwargs = {"device":device, "dtype":dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.(self.__class__.__name__)")

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            encoder_norm = LayerNorm(
                d_mode, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            decoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )
        self.reset_parameters()
        self.d_model = d_model
        self.n_head = nhead
        self.batch_first = batch_first
    
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[bool] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src_size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )
        
        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal
        )

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal
        )
        return output
    
    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device:
    )

class TransformerEncoder(Module):
    pass
class TransformerDecoder(Module):
    pass
class TransformerEncoderLayer(Module):
    pass
class TransformerDecoderLayer(Module):
    pass