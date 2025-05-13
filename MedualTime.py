import os
import sys
import torch
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2PreTrainedModel, GPT2Attention, GPT2MLP
import  torch.nn as nn
from transformers.modeling_utils import prune_conv1d_layer, find_pruneable_heads_and_indices, Conv1D

from data_preprocess import patch_masking
from timm.models.vision_transformer import Block
from transformers import BertTokenizer, BertModel, GPT2Tokenizer
import torch
from torch.nn import functional as F

from temenc_layers.TemEnc import TemEnc
from transformers import AutoTokenizer, AutoModel


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def info_nce(query, positive_key, negative_keys=None, temperature=0.05, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    # query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

class GPT2Attention_w_Adapter(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        # gate
        self.gate = torch.nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _attn_adapter(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        # adapter不需要考虑causal mask
        # if not self.is_cross_attention:
        #     # if only "normal" attention layer implements causal mask
        #     query_length, key_length = query.size(-2), key.size(-2)
        #     causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        #     attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        adapter,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        bsz = hidden_states.size(0)
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_i, key_i, value_i = self.c_attn(hidden_states).split(self.split_size, dim=2) # input带来的query, key, value
            if len(adapter.size()) == 2:
                adapter = adapter.unsqueeze(0).repeat(bsz, 1, 1)
            else:
                adapter = adapter
            query_a, key_a, value_a = self.c_attn(adapter).split(self.split_size, dim=2) # adapter带来的query, key, value


        query = self._split_heads(query_i, self.num_heads, self.head_dim)
        key = self._split_heads(key_i, self.num_heads, self.head_dim)
        value = self._split_heads(value_i, self.num_heads, self.head_dim)

        # adapter只算key和value
        # query_a_s = self._split_heads(query_a, self.num_heads, self.head_dim)
        key_a_s = self._split_heads(key_a, self.num_heads, self.head_dim)
        value_a_s = self._split_heads(value_a, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_adapter_output, attn_adapter_weights = self._attn_adapter(query, key_a_s, value_a_s)
        attn_output = attn_output + attn_adapter_output*self.gate.tanh() # 当gate为0的时候不够成影响


        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2Block_w_Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention_w_Adapter(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)


    def forward(
        self,
        hidden_states,
        adapter,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            adapter,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions) --> 这是正常的attention结果

        # 需要再算一部分adapter的attention结果


        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        # if encoder_hidden_states is not None:
        #     # add one self-attention block for cross-attention
        #     if not hasattr(self, "crossattention"):
        #         raise ValueError(
        #             f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
        #             "cross-attention layers by setting `config.add_cross_attention=True`"
        #         )
        #     residual = hidden_states
        #     hidden_states = self.ln_cross_attn(hidden_states)
        #     cross_attn_outputs = self.crossattention(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #         output_attentions=output_attentions,
        #     )
        #     attn_output = cross_attn_outputs[0]
        #     # residual connection
        #     hidden_states = residual + attn_output
        #     outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)





class GPT2Attention_Adapter(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        # gate
        self.gate = torch.nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _attn_adapter(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        adapter,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        bsz = hidden_states.size(0)
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_i, key_i, value_i = self.c_attn(hidden_states).split(self.split_size, dim=2) # input带来的query, key, value
            if len(adapter.size()) == 2:
                adapter = adapter.unsqueeze(0).repeat(bsz, 1, 1)
            else:
                adapter = adapter
            query_a, key_a, value_a = self.c_attn(adapter).split(self.split_size, dim=2) # adapter带来的query, key, value


        query = self._split_heads(query_i, self.num_heads, self.head_dim)
        key = self._split_heads(key_i, self.num_heads, self.head_dim)
        value = self._split_heads(value_i, self.num_heads, self.head_dim)


        key_a_s = self._split_heads(key_a, self.num_heads, self.head_dim)
        value_a_s = self._split_heads(value_a, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_adapter_output, attn_adapter_weights = self._attn_adapter(query, key_a_s, value_a_s)
        attn_output = attn_output + attn_adapter_output*self.gate.tanh() # 当gate为0的时候不够成影响


        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2Block_Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention_Adapter(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)


    def forward(
        self,
        hidden_states,
        adapter,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            adapter,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]


        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)




class MedualTime(GPT2PreTrainedModel):

    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers-config.adapter_layers)] + [GPT2Block_Adapter(config) for _ in range(config.adapter_layers)])


        self.adapter_ln_f1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.adapter_ln_f2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


        self.config = config

        # Time Series Encoder
        self.ts_config = config.ts_config
        args = self.ts_config
        num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1

        self.adapter_ts_proj_norm = nn.LayerNorm(self.embed_dim)
        self.revin = RevIN(num_features=args.vars, eps=1e-5, affine=False)

        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        print('GPT2 tokenizer loaded')

        self.ts_adapter_layer = nn.Linear(args.vars*args.patch_len, self.embed_dim)
        from transformers import AutoTokenizer, AutoModel
        self.bert_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.bert_encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        # self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print('BERT tokenizer loaded...')

        # Model Adapter
        self.adapter_query_ts = nn.Embedding(config.adapter_layers* config.adapter_len, config.hidden_size)
        self.adapter_query_text = nn.Embedding(config.adapter_layers * config.adapter_len, config.hidden_size)

        ## ResCNN
        self.adapter_temenc = TemEnc(args.vars, self.embed_dim)

        if self.config.bert_projector:
            self.bert_projector = nn.Linear(768, self.embed_dim)
            self.bert_projector_norm = nn.LayerNorm(self.embed_dim)

        if config.gate_fusion:
            self.gate_fusion_1 = nn.Linear(config.hidden_size, 1)
            self.gate_fusion_2 = nn.Linear(config.hidden_size, 1)


    def cal_gate_1(self, hidden_states):
        return torch.sigmoid(self.gate_fusion_1(hidden_states))

    def cal_gate_2(self, hidden_states):
        return torch.sigmoid(self.gate_fusion_2(hidden_states))


    def text_main_forward(
            self,
            raw_text = None,
            input_ids=None,
            ts_sample = None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        original_lengths = [len(self.gpt_tokenizer.encode(text)) for text in raw_text]

        id_text = self.gpt_tokenizer(raw_text, padding=True, return_tensors="pt")
        device = ts_sample.device
        id_text = {k: v.to(device) for k, v in id_text.items()}
        input_ids = id_text['input_ids']
        attention_mask = id_text['attention_mask']

        ts_emb = self.adapter_temenc(ts_sample.permute(0,2,1))
        ts_emb = ts_emb.unsqueeze(1)
        ts_emb = self.adapter_ts_proj_norm(ts_emb) * 1
        ts_emb = ts_emb.repeat(1, self.config.adapter_len, 1)



        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0


        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds



        hidden_states = self.drop(hidden_states)

        adapter_index = 0
        adapter = self.adapter_query_text.weight.reshape(self.config.adapter_layers, self.config.adapter_len, self.config.hidden_size)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if i < self.config.num_hidden_layers-self.config.adapter_layers:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    adapter = adapter[adapter_index, ...] + ts_emb,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                adapter_index += 1

            hidden_states = outputs[0]


        original_lengths = torch.tensor(original_lengths).to(device) -1
        indices = original_lengths.view(-1, 1).unsqueeze(2).expand(-1, -1, hidden_states.size(2))
        last_hidden_states = hidden_states.gather(1, indices)
        last_hidden_states = self.adapter_ln_f1(last_hidden_states)

        return last_hidden_states  # [bsz, seq_len, hidden_size (768)]




    def ts_main_forward(
            self,
            raw_text = None,
            input_ids=None,
            ts_sample = None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        device = ts_sample.device
        id_text = self.bert_tokenizer(raw_text, padding=True, return_tensors="pt")
        id_text = {k: v.to(device) for k, v in id_text.items()}

        with torch.no_grad():
            text_emb_bert = self.bert_encoder(**id_text).last_hidden_state[:, 0, :] # [bsz, 768]
            if self.config.bert_projector:
                text_emb_bert = self.bert_projector(text_emb_bert)
                text_emb_bert = self.bert_projector_norm(text_emb_bert)
            text_emb_bert = text_emb_bert.unsqueeze(1)
            text_emb_bert = text_emb_bert.repeat(1,  self.config.adapter_len, 1) # [bsz, adapter_len, 768]

        if self.ts_config.revin:
            ts_sample = self.revin(ts_sample, 'norm')
        ts_sample, _ = patch_masking(ts_sample, self.ts_config) # (bsz, seq_len, nvars)  -- > (bsz, num_patch, n_vars, patch_len)

        bsz = ts_sample.shape[0]
        num_patch = ts_sample.shape[1]
        inputs_embeds = self.ts_adapter_layer(ts_sample.reshape(bsz, num_patch, -1)) # --> (bsz, num_patch, 768)
        attention_mask = torch.ones(bsz, num_patch).to(device)



        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)

            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0



        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds


        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        adapter_index = 0
        adapter = self.adapter_query_ts.weight.reshape(self.config.adapter_layers, self.config.adapter_len, self.config.hidden_size)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if i < self.config.num_hidden_layers-self.config.adapter_layers:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    adapter = adapter[adapter_index, ...] + text_emb_bert ,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                adapter_index += 1

            hidden_states = outputs[0]

        hidden_states = self.adapter_ln_f2(hidden_states)

        hidden_states = hidden_states.view(*output_shape)

        return hidden_states  # [bsz, seq_len, hidden_size (768)]



class MedualTime_unsupervised(GPT2PreTrainedModel):

    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers-config.adapter_layers)] + [GPT2Block_Adapter(config) for _ in range(config.adapter_layers)])
        self.adapter_ln_f1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.adapter_ln_f2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


        self.config = config

        # Time Series Encoder
        self.ts_config = config.ts_config
        args = self.ts_config
        num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1

        self.adapter_ts_proj_norm = nn.LayerNorm(self.embed_dim)
        self.revin = RevIN(num_features=args.vars, eps=1e-5, affine=False)

        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        print('GPT2 tokenizer loaded')

        self.ts_adapter_layer = nn.Linear(args.vars*args.patch_len, self.embed_dim)
        # self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        from transformers import AutoTokenizer, AutoModel
        self.bert_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.bert_encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        print('BERT tokenizer loaded...')

        # Model Adapter
        self.adapter_query_ts = nn.Embedding(config.adapter_layers* config.adapter_len, config.hidden_size)
        self.adapter_query_text = nn.Embedding(config.adapter_layers * config.adapter_len, config.hidden_size)


        self.adapter_temenc = TemEnc(args.vars, self.embed_dim)

        self.adapter_cross_projector1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.adapter_cross_ln1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.adapter_cross_projector2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.adapter_cross_ln2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_linear_weight()

    def init_linear_weight(self):
        self.adapter_cross_projector1.weight.data.normal_(0, 0.01)
        self.adapter_cross_projector1.bias.data.zero_()
        self.adapter_cross_projector2.weight.data.normal_(0, 0.01)
        self.adapter_cross_projector2.bias.data.zero_()
        self.ts_adapter_layer.weight.data.normal_(0, 0.01)
        self.ts_adapter_layer.bias.data.zero_()


    def text_main_forward(
            self,
            raw_text = None,
            input_ids=None,
            ts_sample = None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        original_lengths = [len(self.gpt_tokenizer.encode(text)) for text in raw_text]

        id_text = self.gpt_tokenizer(raw_text, padding=True, return_tensors="pt")
        device = ts_sample.device
        id_text = {k: v.to(device) for k, v in id_text.items()}
        input_ids = id_text['input_ids']
        attention_mask = id_text['attention_mask']


        ts_emb = self.adapter_temenc(ts_sample.permute(0,2,1))
        ts_emb = ts_emb.unsqueeze(1)
        ts_emb = self.adapter_ts_proj_norm(ts_emb) * 1
        ts_emb = ts_emb.repeat(1, self.config.adapter_len, 1)



        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0


        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds



        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        adapter_index = 0
        adapter = self.adapter_query_text.weight.reshape(self.config.adapter_layers, self.config.adapter_len, self.config.hidden_size)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # past_key_values: (tuple) - cached key and value states for the attention mechanism

            if i < self.config.num_hidden_layers-self.config.adapter_layers:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    adapter = adapter[adapter_index, ...] + ts_emb,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                adapter_index += 1

            hidden_states = outputs[0]


        original_lengths = torch.tensor(original_lengths).to(device) -1
        indices = original_lengths.view(-1, 1).unsqueeze(2).expand(-1, -1, hidden_states.size(2))
        last_hidden_states = hidden_states.gather(1, indices)
        last_hidden_states = self.adapter_ln_f1(last_hidden_states)

        last_hidden_states = self.adapter_cross_projector1(last_hidden_states)
        last_hidden_states = self.adapter_cross_ln1(last_hidden_states) + last_hidden_states


        return last_hidden_states  # [bsz, seq_len, hidden_size (768)]


    def ts_main_forward(
            self,
            raw_text = None,
            input_ids=None,
            ts_sample = None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = ts_sample.device
        id_text = self.bert_tokenizer(raw_text, padding=True, return_tensors="pt")
        id_text = {k: v.to(device) for k, v in id_text.items()}

        with torch.no_grad():
            text_emb_bert = self.bert_encoder(**id_text).last_hidden_state[:, 0, :] # [bsz, 768]
            text_emb_bert = text_emb_bert.unsqueeze(1)
            text_emb_bert = text_emb_bert.repeat(1,  self.config.adapter_len, 1)

        if self.ts_config.revin:
            ts_sample = self.revin(ts_sample, 'norm')
        ts_sample, _ = patch_masking(ts_sample, self.ts_config) # (bsz, seq_len, nvars)  -- > (bsz, num_patch, n_vars, patch_len)

        bsz = ts_sample.shape[0]
        num_patch = ts_sample.shape[1]
        inputs_embeds = self.ts_adapter_layer(ts_sample.reshape(bsz, num_patch, -1)) # --> (bsz, num_patch, 768),
        attention_mask = torch.ones(bsz, num_patch).to(device)



        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0



        head_mask = self.get_head_mask(head_mask, self.config.n_layer)


        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds



        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        adapter_index = 0
        adapter = self.adapter_query_ts.weight.reshape(self.config.adapter_layers, self.config.adapter_len, self.config.hidden_size)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if i < self.config.num_hidden_layers-self.config.adapter_layers:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    adapter = adapter[adapter_index, ...] + text_emb_bert ,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                adapter_index += 1

            hidden_states = outputs[0]

        hidden_states = self.adapter_ln_f2(hidden_states)

        hidden_states = self.adapter_cross_projector2(hidden_states)
        hidden_states = self.adapter_cross_ln2(hidden_states) + hidden_states

        hidden_states = hidden_states.view(*output_shape)

        return hidden_states  # [bsz, seq_len, hidden_size (768)]


class GPT2Model_w_Adapter_resnet(GPT2PreTrainedModel):
    ## 整体backbone还是对文本的处理，添加了时间序列做另外一个模态
    ## 两个模态的整合还是和llama adapter一样的
    ## ts_config是时序encoder的config

    ## 时序encoder不再需要patchtst了，直接用TCN，然后从头一起训练
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)  # 嵌入层，把输入的token转换成对应的词向量
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)  # 位置嵌入层，把位置信息转换成对应的词向量

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers-config.adapter_layers)] + [GPT2Block_w_Adapter(config) for _ in range(config.adapter_layers)])
        # self.adapter_layer = nn.ModuleList([Adapter_cal(config) for _ in range(config.adapter_layers)])

        self.adapter_ln_f1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.adapter_ln_f2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()  # 初始化权重

        # Model parallel
        self.model_parallel = False
        self.device_map = None


        self.config = config

        # Time Series Encoder
        self.ts_config = config.ts_config
        args = self.ts_config
        num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
        # self.ts_encoder =  PatchTST(c_in=args.vars,
        #              target_dim=args.target_points,
        #              patch_len=args.patch_len,
        #              stride=args.stride,
        #              num_patch=num_patch,
        #              n_layers=args.n_layers,
        #              n_heads=args.n_heads,
        #              d_model=args.d_model,
        #              shared_embedding=True,
        #              d_ff=args.d_ff,
        #              dropout=args.dropout,
        #              head_dropout=args.head_dropout,
        #              act='relu',
        #              head_type='pretrain',
        #              res_attention=False
        #              )
        # # 主目录在这里设置，定位到patchtst_0913
        # # 具体后面的，在patchtst_config里面
        # model_path = os.path.join(args.saved_model_path)
        # self.ts_encoder.load_state_dict(torch.load(model_path , map_location="cpu"))
        #
        # self.adapter_ts_blocks = nn.ModuleList([
        #     Block(args.patch_len*args.vars, 4, 2, qkv_bias=True)
        #     for _ in range(2)])
        #
        #
        # self.adapter_ts_proj = nn.Linear(args.patch_len*args.vars, self.embed_dim)
        self.adapter_ts_proj_norm = nn.LayerNorm(self.embed_dim)
        self.revin = RevIN(num_features=args.vars, eps=1e-5, affine=False)

        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        print('GPT2 tokenizer loaded')

        self.ts_adapter_layer = nn.Linear(args.vars*args.patch_len, self.embed_dim)
        # self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        from transformers import AutoTokenizer, AutoModel
        self.bert_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.bert_encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        print('BERT tokenizer loaded...')

        # Model Adapter
        self.adapter_query_ts = nn.Embedding(config.adapter_layers* config.adapter_len, config.hidden_size)
        self.adapter_query_text = nn.Embedding(config.adapter_layers * config.adapter_len, config.hidden_size)

        ## ResCNN
        self.adapter_resnet = TemEnc(args.vars, self.embed_dim)

        ## 加一个projector给bert
        if self.config.bert_projector:
            self.bert_projector = nn.Linear(768, self.embed_dim)
            self.bert_projector_norm = nn.LayerNorm(self.embed_dim)

        if config.gate_fusion:
            self.gate_fusion_1 = nn.Linear(config.hidden_size, 1)
            self.gate_fusion_2 = nn.Linear(config.hidden_size, 1)


    def cal_gate_1(self, hidden_states):
        return torch.sigmoid(self.gate_fusion_1(hidden_states))

    def cal_gate_2(self, hidden_states):
        return torch.sigmoid(self.gate_fusion_2(hidden_states))




    def text_main_forward(
            self,
            raw_text = None,
            input_ids=None,
            ts_sample = None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # ts_sample: [32, 1000, 12]

        # 找到最后一个token的位置（即原本text的长度）
        original_lengths = [len(self.gpt_tokenizer.encode(text)) for text in raw_text]

        id_text = self.gpt_tokenizer(raw_text, padding=True, return_tensors="pt")
        device = ts_sample.device
        id_text = {k: v.to(device) for k, v in id_text.items()}
        input_ids = id_text['input_ids']
        attention_mask = id_text['attention_mask']

        # with torch.no_grad():
        #     if self.ts_config.revin:
        #         ts_sample = self.revin(ts_sample , 'norm')
        #     ts_sample, _ = patch_masking(ts_sample, self.ts_config) # (bsz, seq_len, nvars)  -- > (bsz, num_patch, n_vars, patch_len)
        #     # ts_emb = self.ts_encoder(ts_sample.half()) # 实际patchtst: (bsz, num_patch, n_vars, patch_len) --> (bsz, num_patch, n_vars, patch_len)
        #     ts_emb = self.ts_encoder(ts_sample)
        #     # 完美： [bsz, adapter_len, 4096]
        # # Another layers for adjusting ts_emb
        # ts_emb = ts_emb.reshape(ts_emb.shape[0], ts_emb.shape[1], -1)  # (bsz, num_patch, n_vars, patch_len) -- > (bsz, num_patch, n_vars * patch_len)
        # for mm in self.adapter_ts_blocks:
        #     ts_emb = mm(ts_emb)
        #
        # ts_emb = ts_emb[:, :self.config.adapter_len, :]
        # ts_emb = self.adapter_ts_proj(ts_emb)
        # ts_emb = self.adapter_ts_proj_norm(ts_emb) * 1
        # # ts_emb = self.adapter_ts_proj_norm(ts_emb)*0.1

        ts_emb = self.adapter_resnet(ts_sample.permute(0,2,1))
        ts_emb = ts_emb.unsqueeze(1)
        ts_emb = self.adapter_ts_proj_norm(ts_emb) * 1
        ts_emb = ts_emb.repeat(1, self.config.adapter_len, 1)



        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:  # 采用input_ids作为输入
            input_shape = input_ids.size()  # 输入token的长度
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:  # 需要把同一个batch的句子padding到同样的长度，所以需要attention_mask，1表示有输入，0表示padding
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0  # 0代表有输入，-10000代表padding的部分


        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # 词嵌入层和位置嵌入层
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds



        hidden_states = self.drop(hidden_states)
        # input_shape: (batch_size, seq_len), hidden_states: (batch_size, seq_len, hidden_size)
        # output_shape: (batch_size, seq_len, hidden_size) --> 其实和hidden_states一样
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        adapter_index = 0
        adapter = self.adapter_query_text.weight.reshape(self.config.adapter_layers, self.config.adapter_len, self.config.hidden_size)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # past_key_values: (tuple) - cached key and value states for the attention mechanism
            # 用于生成的时候，把之前的结果传入，这样可以减少计算量，每一层都是用的同样一个past_key_values

            if i < self.config.num_hidden_layers-self.config.adapter_layers: # 不是adapter层
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    adapter = adapter[adapter_index, ...] + ts_emb,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                adapter_index += 1

            hidden_states = outputs[0]

        # hidden_states = self.ln_f(hidden_states)
        #
        # hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        ## 只取最后一个非padding的token的embedding
        original_lengths = torch.tensor(original_lengths).to(device) -1
        indices = original_lengths.view(-1, 1).unsqueeze(2).expand(-1, -1, hidden_states.size(2))
        last_hidden_states = hidden_states.gather(1, indices)
        last_hidden_states = self.adapter_ln_f1(last_hidden_states)

        return last_hidden_states  # [bsz, seq_len, hidden_size (768)]




        # return hidden_states  # [bsz, seq_len, hidden_size (768)]

        #
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
        #
        # return (hidden_states,presents,all_hidden_states,all_self_attentions,all_cross_attentions,)


    def ts_main_forward(
            self,
            raw_text = None,
            input_ids=None,
            ts_sample = None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = ts_sample.device
        id_text = self.bert_tokenizer(raw_text, padding=True, return_tensors="pt")
        id_text = {k: v.to(device) for k, v in id_text.items()}

        with torch.no_grad():
            text_emb_bert = self.bert_encoder(**id_text).last_hidden_state[:, 0, :] # [bsz, 768]
            if self.config.bert_projector: # 如果需要projector
                text_emb_bert = self.bert_projector(text_emb_bert)
                text_emb_bert = self.bert_projector_norm(text_emb_bert)
            text_emb_bert = text_emb_bert.unsqueeze(1)
            text_emb_bert = text_emb_bert.repeat(1,  self.config.adapter_len, 1) # [bsz, adapter_len, 768], 每个长度都加同样的

        if self.ts_config.revin:
            ts_sample = self.revin(ts_sample, 'norm')
        ts_sample, _ = patch_masking(ts_sample, self.ts_config) # (bsz, seq_len, nvars)  -- > (bsz, num_patch, n_vars, patch_len)
        # ts_emb = self.ts_encoder(ts_sample.half()) # 实际patchtst: (bsz, num_patch, n_vars, patch_len) --> (bsz, num_patch, n_vars, patch_len)

        # ts_emb = self.ts_encoder(ts_sample)
        # 完美： [bsz, adapter_len, 4096]
        # Another layers for adjusting ts_emb
        bsz = ts_sample.shape[0]
        num_patch = ts_sample.shape[1]
        inputs_embeds = self.ts_adapter_layer(ts_sample.reshape(bsz, num_patch, -1)) # --> (bsz, num_patch, 768), 当做gpt2后续的输入（取代text位置）
        attention_mask = torch.ones(bsz, num_patch).to(device)



        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:  # 采用input_ids作为输入
            input_shape = input_ids.size()  # 输入token的长度
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.view(-1, input_shape[-1])
        # if position_ids is not None:
        #     position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:  # 如果没有past_key_values, 则初始化为None? 指的是啥
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:  # 需要把同一个batch的句子padding到同样的长度，所以需要attention_mask，1表示有输入，0表示padding
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0  # 0代表有输入，-10000代表padding的部分



        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # 词嵌入层和位置嵌入层
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # if token_type_ids is not None:
        #     token_type_embeds = self.wte(token_type_ids)
        #     hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        # input_shape: (batch_size, seq_len), hidden_states: (batch_size, seq_len, hidden_size)
        # output_shape: (batch_size, seq_len, hidden_size) --> 其实和hidden_states一样
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        adapter_index = 0
        adapter = self.adapter_query_ts.weight.reshape(self.config.adapter_layers, self.config.adapter_len, self.config.hidden_size)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # past_key_values: (tuple) - cached key and value states for the attention mechanism
            # 用于生成的时候，把之前的结果传入，这样可以减少计算量，每一层都是用的同样一个past_key_values


            if i < self.config.num_hidden_layers-self.config.adapter_layers: # 不是adapter层
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    adapter = adapter[adapter_index, ...] + text_emb_bert ,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                adapter_index += 1

            hidden_states = outputs[0]
            # if use_cache is True:
            #     presents = presents + (outputs[1],)
            #
            # if output_attentions:
            #     all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            #     if self.config.add_cross_attention:
            #         all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
            #
            # # Model Parallel: If it's the last layer for that device, put things on the next device
            # if self.model_parallel:
            #     for k, v in self.device_map.items():
            #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
            #             hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.adapter_ln_f2(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states  # [bsz, seq_len, hidden_size (768)]


