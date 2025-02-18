# Codes are available at https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L57
import copy
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_

# from .activation import MultiheadAttention
# from .container import ModuleList
# from .dropout import Dropout
# from .linear import Linear
# from .module import Module
# from .normalization import LayerNorm

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

class TransformerEncoder(Module):
    pass
class TransformerDecoder(Module):
    pass
class TransformerEncoderLayer(Module):
    pass
class TransformerDecoderLayer(Module):
    pass