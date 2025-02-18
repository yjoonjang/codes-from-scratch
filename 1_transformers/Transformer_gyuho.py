import copy
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_

from .activation import MultiheadAttention
from .container import ModuleList
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .normalization import LayerNorm

__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer"
]

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    
    return torch.triu( #upper triangular mask 
        torch.full((sz,sz),float("-inf"),dtype=dtype,device=device),
        diagopnal=1,
    )

def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size=src.size()
        if(len(src_size)==22):
            return src_size[0] #unbatched
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

class Transformer(Module):
    def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int =2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            custom_encoder: Optional[Any]= None,
            custom_decoder: Optional[Any] = None,
            layer_norm_eps:float=1e-5,
            batch_first:bool=False,
            norm_first:bool=False,
            bias: bool=True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device":device, "dtype":dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

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
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
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
                d_model, eps= layer_norm_eps, bias= bais, **factory_kwargs
            )
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )
        self._reset_parameters()

        self.d_model = d_model
        self.nhead=nhead
        self.batch_first=batch_first

    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor]=None,
            src_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            src_is_causal: Optional[bool] = None,
            tgt_is_causal: Optional[bool] = None,
            memory_is_causal: bool=False,
    ) -> Tensor:
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) !=tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must  be equal")
        elif self.batch_first and src.size(0)!=tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        
        if src.size(-1)!=self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )
        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask = src_key_padding_mask,
            is_causal = src_is_causal,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask = memory_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = memory_key_padding_mask,
            tgt_is_causal = tgt_is_causal,
            memory_is_causal = memory_is_causal,
        )
        return output
    
    @staticmethod
    def generate_square_subsequent_mask(
        sz:int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class TransformerEncoder(Module):
    __constants__=["norm"]

    def __init__(
            self, encoder_layer: "TransformerEncoderLayer",
            num_layers = int,
            norm: Optional[Module] = None,
            enable_nested_tensor: bool = True,
            mask_check: bool = True,
    ) -> Mone:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self/}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ""
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first:
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn.batch_first was not True"
                + "(use batch_first for better inference performance)"
            )
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = (f
                f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
            )
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.activation_relu_or_gelu was not True"
            )
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps):
            why_not_sparsity_fast_path = (
                f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
            )
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(
                f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}"
            )
            self.use_nested_tensor = False
    def forward(
            self,
            src: Tensor,
            maks: Optional[Tensor]=None,
            src_key_padding_mask: Optional[Tensor] =None,
            is_causal:Optional[bool] =None,
    )-> Tensor:
        
        src_key_padding_mask = F._canonical_mask(
            mask= src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type = F._none_or_dtype(mask),
            other_name = "mask",
            target_type = src.dtype,
        )

        mask = F._canonical_mask(
            mask = mask,
            mask_name=
            other_type = None,
            other_name = "",
            target_type=src.dtype,
            check_other = False,
        )
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer="self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enkabled()

        
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = ( # collects the learnable parameters of the self-attention mechanism and layer normmalization
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                src_mask = mask,
                is_causal = is_causal,
                src_key_padding_mask = src_key_padding_mask_for_layers,
            )
        if convert_to_nested:
            output = output.to_padded_tensor(0.0,src.size())

        if self.norm is not None:
            output = self.norm(output)
        
        return output

class TransformerDecoder(Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        tgt: Tensor,
        memory:Tensor,
        tgt_mask: Optional[Tensor]=None,
        memory_mask: Optional[Tensor]=None,
        tgt_key_padding_mask: Optional[Tensor] =None,
        memory_key_padding_mask: Optional[Tensor] =None,
        tgt_is_causal: Optioinal[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        output = tgt
        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerEncoderLayer(Module):
    #self-attn  & ffn
    __constants__ = ["norm_first"]
    
    def __init__(
        self,
        d_model:int,
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
    ) -> None:
        factory_kwargs = {"device": device,"dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        #FFN
        self.linear1=Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout=Dropout(dropout)
        self.linear2=Linear(dim_feedforward,d_model,bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1=Dropout(dropout)
        self.dropout2= Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        
        if activation is F.relu or isinstance(activation, torch,nn,ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch,nn,GeLU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation =  activation

    def _setstate(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
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

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ""
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        elif any(
            len(getattr(m, "_forward_hooks", {}))
            + len(getattr(m, "_forward_pre_hooks", {}))
            for m in self.modules()
        ):
            why_not_sparsity_fast_path = "forward pre-/hooks are attached to the module"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                (x.device.type in _supported_device_type) for x in tensor_args
            ):
                why_not_sparsity_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"{_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src
                )
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        x = src
        if self.norm_first: # pre-LN Transformer
           x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal = is_causal
           )
           x = x+ self._ff_block(self.norm2(x))
        else: # post-LN Transformer
           x= self.norm1(
               x + self.sa_block(x,src_mask, src_key_padding_mask, is_causal=is_causal)
           )
           x = self.norm2(x+self._ff_block(x))

        return x
    
    #self-attention block
    def _sa_block(
            self,
            x: Tensor,
            attn_mask:  Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x, x, x, 
            attn_mask = attn_mask,
            key_padding_mask = key_padding_mask,
            need_weights = False,
            is_causal = is_causal,
        )[0]
        return self.dropout1(x)
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.lienar1(x))))
        return self.dropout2(x)
    
class TransformerDecoderLayer(Module):
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
    ) -> None:
        factory_kwargs = { "device": device, "dtype":dtype}
        super().__init__()
        self.self_attn= MultiheadAttention(
            d_model,
            nhead,
            dropout= dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias = bias,
            **factory_kwargs,
        )

        self.linear1 = Linear(d_model, dim_feedforward, bias= bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps = layer_norm_eps, bias =bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps = layer_norm_eps, bias =bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps = layer_norm_eps, bias =bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor]=None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x+self._sa_block( # maksed MHA
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x+ self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x+ self._mha_block( #MHA
                    x,memory, memory_mask, memory_key_padding_mask, memory_is_causal #memory refers to encoder output??
                )
            )
            x=self.norm3(x+self._ff_block(x))

        return x
    
    def _sa_block(
            self,
            x:Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
    )-> Tensor:
        x = self.self_attn(
            x,x,x, attn_mask = attn_mask,
            key_padding_mask = key_padding_mask,
            is_causal = is_causal,
            need_weights = False,
        )[0]
        return self.dropout1(x)
    def _mha_block(
            self,
            x: Tensor,
            mem:Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
    )-> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask = attn_mask,
            key_padding_mask = key_padding_mask,
            is_causal = is_causal,
            need_weights = False,
        )[0]
        return self.dropout2(x)
    def _ff_block(self, x:Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] =None,
        size: Optional[int] = None,
) -> bool: 
    make_causal = is_causal is True
    if is_causal is None and mask is not None:
        sz= size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device = mask.dedvic, dtype = mask.dtype
        )

        if mask.size() == causal_comparison.size():
            make_causal = bool((mask==causal_comparison).all())
        else:
            make_causal = False    
    return make_causal