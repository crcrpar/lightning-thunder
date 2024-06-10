from __future__ import annotations
import math
from typing import ClassVar, TYPE_CHECKING

from einops import reduce, rearrange
import torch.nn as nn
from torch.distributed import distributed_c10d as c10d

from thunder.core import utils

if TYPE_CHECKING:
    import torch
    from torch.distributed import ProcessGroup


__all__ = [
    "ParallelMLP",
]


class ParallelMLP(nn.Module):
    """Simplified version of Megatron/NeMo's ParallelMLP.

    Ref: https://github.com/NVIDIA/NeMo/blob/95ca2f4/nemo/collections/nlp/modules/common/megatron/mlp.py#L61
    """

    COLUMN_WISE: ClassVar[tuple[str]] = ("dense_h_to_4h",)
    ROW_WISE: ClassVar[tuple[str]] = ("dense_4h_to_h",)

    SUPPORTED_GELU_APPROX: ClassVar[tuple[str, str]] = ("none", "tanh")

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int | None = None,
        bias: bool = True,
        gelu_approximate: str = "none",
    ) -> None:
        utils.check(
            gelu_approximate in ParallelMLP.SUPPORTED_GELU_APPROX,
            lambda: f"Invalid {gelu_approximate}, supported are {ParallelMLP.SUPPORTED_GELU_APPROX}",
        )
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.dense_4h_to_h = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)
        self.gelu = nn.GELU(approximate=gelu_approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        four_h = self.gelu(self.dense_h_to_4h(x))
        h = self.dense_4h_to_h(four_h)
        return h


FP16 = "fp16"
BF16 = "bf16"


# Ref: https://github.com/NVIDIA/NeMo/blob/445b9b1/nemo/collections/nlp/modules/common/megatron/attention.py#L744
class CoreAttention(nn.Module):

    precision: str
    fp16: bool
    bf16: bool
    multi_query_attention: bool
    position_embedding_type: str
    attention_type: str
    attention_mask_type: str
    sequence_parallel: bool
    normalize_attention_scores: bool
    norm_factor: float
    process_group: ProcessGroup | None
    attention_dropout_p: float

    def __init__(
        self,
        layer_number: int,
        num_attention_heads: int,
        hidden_size: int,
        attention_type: str,
        attention_mask_type: str,
        *,
        precision: str = 16,
        apply_query_key_layer_scaling: bool = False,
        kv_channels: int | None = None,
        attention_dropout: float = 0.1,
        normalize_attention_scores: bool = True,
        multi_query_attention: bool = True,
        position_embedding_type: str = "learned_absolute",
        use_flash_attention: bool = False,
        sequence_parallel: bool = False,
        # following kwargs are not existant in nemo
        process_group: ProcessGroup | None = None,
    ) -> None:
        super().__init__()

        utils.check(
            precision in (FP16, BF16),
            lambda: f"{precision=} is not supported by {(FP16, BF16)=}",
        )
        self.precision = precision
        self.fp16 = self.precision == FP16
        self.bf16 = self.precision == BF16
        self.multi_query_attention = multi_query_attention
        self.position_embedding_type = position_embedding_type

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = self.apply_query_key_layer_scaling
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attention_mask_type = attention_mask_type
        self.sequence_parallel = sequence_parallel
        self.normalize_attention_scores = normalize_attention_scores

        if kv_channels is None:
            utils.check(
                hidden_size % num_attention_heads == 0,
                lambda: f"{hidden_size=} must be divisible by {num_attention_heads=}",
            )
            kv_channels = hidden_size // num_attention_heads

        projection_size = kv_channels * num_attention_heads
        self.process_group = process_group
        world_size: int = c10d.get_world_size(self.process_group)
        self.hidden_size_per_partition = projection_size // world_size
        self.hidden_size_per_attention_head = projection_size // num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads // world_size
        self.num_attention_heads_partition_offset = self.num_attention_heads_per_partition * c10d.get_rank(
            self.process_group
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        utils.check(
            attention_dropout == 0.0,
            lambda: f"Tensor Parallel does not support dropout but got {attention_dropout=}",
        )
        self.attention_dropout_p = attention_dropout
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past=None,
        get_key_value: bool = False,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor: torch.Tensor | None = None,
        inference_mode=None,
    ):
        sk = key_layer.size(0)
        sq, b, np, hn = query_layer.shape[:4]

        # Update attention mask for inference. [b, np, sq, sk]
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[..., sq - 1, :sk].unsqueeze(2)
                else:
                    attention_mask = attention_mask[..., :sq, :sk]

        # Update attention bias. [b, np, sq, sk]
        if relative_position_bias is not None:
            relative_position_bias = relative_position_bias[
                :,
                self.num_attention_heads_per_partition : self.num_attention_heads_partition_offset
                + self.num_attention_heads_per_partition,
                -sq:,
                -sk:,
            ]

        # Update query_layer, key_layer, value_layer
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)

        if self.position_embedding_type.lower() == "xpos" and False:
            query_layer = self.xpos(query_layer, offset=key_layer.size(-2) - query_layer.size(-2), downscale=False)
            key_layer = self.xpos(key_layer, offset=0, downscale=True)

        # query                  [sq, b, np, hn]
        # key                    [sk, b, np, hn]
        # value                  [sk, b, np, hn]
        # attn mask              [b, 1, sq, sk] or [b, s]
        # relative position bias [b, np, sq, sk]
        # context layer          [b, np, sq, hn]
        context_layer: torch.Tensor = self.torch_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            relative_position_bias,
            inference_mode,
        )

        if headscale_tensor is not None:
            context_layer = context_layer * headscale_tensor

        # [b, np, sq, hn] -> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] -> [sq, b, hp]
        new_context_layer_shape = tuple(context_layer.shape[:-2]) + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def torch_attention(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_bias: torch.Tensor,
        inference_mode: bool,
    ) -> torch.Tensor:
        sq, b, np, hn = query_layer.shape
        sk = key_layer.shape[0]

        if self.multi_query_attention:
            query_layer = rearrange(query_layer, "sq b np hn -> b (np sq) hn")
            key_layer = rearrange(key_layer, "sk b 1 hn -> b hn sk")
            value_layer = rearrange(value_layer, "sv b np hn -> (b np) sv hn")
        else:
            query_layer = rearrange(query_layer, "sq b np hn -> (b np) sq hn")
            key_layer = rearrange(key_layer, "sk b np hn -> (b np) hn sk")
            value_layer = rearrange(value_layer, "sv b np hn -> (b np) sv hn")

        matmul_input_buffer = torch.empty(
            query_layer.shape[0],
            query_layer.shape[1],
            key_layer.shape[2],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,
            key_layer,
            beta=0.0,
            alpha=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(b, np, sq, sk)

        if attention_bias is not None:
            attention_scores += attention_bias

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel and False:
            # with tensor_parallel.random.get_cuda_rng_tracker().fork():
            #     attention_probs = self.attention_dropout(attention_probs)
            pass
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # change view [b * np, sq, sk]
        attention_probs = rearrange(attention_probs, "b np sq sk -> (b np) sq sk")

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = rearrange(context_layer, "(b np) sq hn -> b np sq hn", np=np)

        return context_layer


SELF_ATTN = "self_attn"
CROSS_ATTN = "cross_attn"


# Ref: https://github.com/NVIDIA/NeMo/blob/445b9b1/nemo/collections/nlp/modules/common/megatron/attention.py#L124
class ParallelAttention(nn.Module):
    layer_number: int
    num_attention_heads: int
    hidden_size: int
    attention_type: str
    attention_mask_type: str
    precision: str
    apply_query_key_layer_scaling: bool
    attention_dropout: float

    def __init__(
        self,
        layer_number: int,
        num_attention_heads: int,
        hidden_size: int,
        attention_type: str,
        attention_mask_type: str,
        precision: str,
        apply_query_key_layer_scaling: bool = False,
        kv_channels: int | None = None,
        masked_softmax_fusion: bool = True,
        attention_dropout: float = 0.1,
        layer_type=None,
        bias: bool = True,
        headscale: bool = False,
        position_embedding_type: str = "learned_absolute",
        multiy_query_attention: bool = False,
        normalize_attention_scores: bool = True,
        process_group: ProcessGroup | None = None,
        multi_query_attention: bool = False,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attention_mask_type
        self.normalize_attention_scores = normalize_attention_scores
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention
        self.use_flash_attention = use_flash_attention

        utils.check(
            attention_type
            in (
                SELF_ATTN,
                CROSS_ATTN,
            ),
            lambda: f"Invalid {attention_type=}, supported are {(SELF_ATTN, CROSS_ATTN)}",
        )

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
            kv_channels = hidden_size // num_attention_heads
        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = c10d.get_world_size(process_group)
        self.hidden_size_per_attention_head = projection_size // num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads // world_size
        self.num_attention_heads_partition_offset = self.num_attention_heads_per_partition * c10d.get_rank(
            process_group
        )

        # Strided linear layer.
        if attention_type == SELF_ATTN:
            # self.query_key_value = tensor_parallel.ColumnParallelLinear(
            #     hidden_size,
            #     3 * projection_size,
            #     config=config,
            #     gather_output=False,
            #     init_method=init_method,
            #     bias=bias,
            # )
            self.query_key_value = nn.Linear(hidden_size, 3 * projection_size, bias=bias)
            self._column_wise = ("query_key_value",)
        else:
            assert attention_type == CROSS_ATTN
            # self.query = tensor_parallel.ColumnParallelLinear(
            #     hidden_size, projection_size, config=config, gather_output=False, init_method=init_method, bias=bias,
            # )
            self.query = nn.Linear(hidden_size, projection_size, bias=bias)

            # self.key_value = tensor_parallel.ColumnParallelLinear(
            #     hidden_size,
            #     2 * projection_size,
            #     config=config,
            #     gather_output=False,
            #     init_method=init_method,
            #     bias=bias,
            # )
            self.key_value = nn.Linear(hidden_size, 2 * projection_size, bias=bias)
            self._column_wise = ("query", "key_value")

        self.core_attention = CoreAttention(
            layer_number=self.layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=self.attention_type,
            attn_mask_type=self.attn_mask_type,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            multi_query_attention=multi_query_attention,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=position_embedding_type,
            use_flash_attention=use_flash_attention,
        )

        # Output.
        # self.dense = tensor_parallel.RowParallelLinear(
        #     projection_size,
        #     hidden_size,
        #     config=config,
        #     input_is_parallel=True,
        #     init_method=output_layer_init_method,
        #     skip_bias_add=True,
        #     bias=bias,
        # )
        self.dense = nn.Linear(projection_size, hidden_size, bias=bias)

        self.headscale = headscale
        if headscale:
            self.head_scale_tensor = torch.nn.Parameter(
                torch.ones(1, self.num_attention_heads_per_partition, 1, 1), requires_grad=True
            )

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

        # relative position embedding
        self.layer_type = layer_type

    @property
    def column_wise(self) -> tuple[str, ...]:
        return self._column_wise

    @property
    def row_wise(self) -> tuple[str, ...]:
        return ("dense",)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past=None,
        get_key_value=False,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,  # rotary positional embedding
        relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_current_sequence_len = 0

        # Some consistency check.
        if inference_max_sequence_len:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == SELF_ATTN:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)
            if self.is_adapter_available():
                lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
                if lora_kqv_adapter and self.adapter_cfg[AdapterName.LORA_KQV_ADAPTER]["enabled"]:
                    lora_mixed_x_layer = lora_kqv_adapter(hidden_states)
                    mixed_x_layer = mixed_x_layer + lora_mixed_x_layer

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            if self.megatron_legacy:
                mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_x_layer, 3, contiguous_split_chunks=True
            )
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)
            if self.is_adapter_available():
                lora_kv_adapter = self.get_adapter_module(AdapterName.LORA_KV_ADAPTER)
                if lora_kv_adapter and self.adapter_cfg[AdapterName.LORA_KV_ADAPTER]["enabled"]:
                    lora_mixed_kv_layer = lora_kv_adapter(encoder_output)
                    mixed_kv_layer = mixed_kv_layer + lora_mixed_kv_layer

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            if self.megatron_legacy:
                mixed_kv_layer = self._transpose_last_dim(mixed_kv_layer, 2, True)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_kv_layer, 2, contiguous_split_chunks=True
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            if self.is_adapter_available():
                lora_q_adapter = self.get_adapter_module(AdapterName.LORA_Q_ADAPTER)
                if lora_q_adapter and self.adapter_cfg[AdapterName.LORA_Q_ADAPTER]["enabled"]:
                    lora_q_layer = lora_q_adapter(hidden_states)
                    query_layer = query_layer + lora_q_layer
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        if self.is_adapter_available():
            key_infused_adapter = self.get_adapter_module(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_adapter_module(AdapterName.VALUE_INFUSED)
            if key_infused_adapter and self.adapter_cfg[AdapterName.KEY_INFUSED]["enabled"]:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key_layer.shape
                key_layer = key_infused_adapter(key_layer.reshape(kls[0], kls[1], -1)).reshape(kls)
            if value_infused_adapter and self.adapter_cfg[AdapterName.VALUE_INFUSED]["enabled"]:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value_layer.shape
                value_layer = value_infused_adapter(value_layer.reshape(vls[0], vls[1], -1)).reshape(vls)

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else ((rotary_pos_emb,) * 2)

        if inference_max_sequence_len:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]
            # Adjust attention mask
            if attention_mask is not None:
                attention_mask = attention_mask[..., start:end, :end]
            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                if not set_inference_key_value_memory:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding.
                    q_pos_emb = q_pos_emb[end - 1 : end]
                else:
                    q_pos_emb = q_pos_emb[:end, :, :, :]
                k_pos_emb = k_pos_emb[:end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)

        if get_key_value:
            present = (key_layer, value_layer)

        if (
            flash_attn_with_kvcache is not None
            and self.use_flash_attention
            and rotary_pos_emb is not None
            and inference_max_sequence_len
            and not set_inference_key_value_memory
        ):
            # Mainly used for decoding with sq=1
            q = _cast_if_autocast_enabled(
                rearrange(apply_rotary_pos_emb(query_layer, rotary_pos_emb[0]), "sq b np hn -> b sq np hn")
            )
            k = _cast_if_autocast_enabled(
                rearrange(apply_rotary_pos_emb(key_layer, rotary_pos_emb[1]), "sk b np hn -> b sk np hn")
            )
            v = _cast_if_autocast_enabled(rearrange(value_layer, "sk b np hn -> b sk np hn"))
            context_layer = flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                causal=self.attn_mask_type == AttnMaskType.causal,
            )
            context_layer = rearrange(context_layer, "b sq np hn -> sq b (np hn)")

        elif checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
                inference_mode=inference_max_sequence_len is not None and query_layer.shape[0] == 1,
            )
        else:
            context_layer = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
                inference_mode=inference_max_sequence_len is not None and query_layer.shape[0] == 1,
            )

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)
        if self.is_adapter_available():
            lora_dense_adapter = self.get_adapter_module(AdapterName.LORA_DENSE_ATTENTION_ADAPTER)
            if lora_dense_adapter and self.adapter_cfg[AdapterName.LORA_DENSE_ATTENTION_ADAPTER]["enabled"]:
                lora_dense_output = lora_dense_adapter(context_layer)
                output = output + lora_dense_output

        if get_key_value:
            output = [output, present]

        return output, bias
