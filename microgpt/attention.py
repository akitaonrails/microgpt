"""Attention for an unpadded sequence with an optional cached prefix.

With Q new tokens and K total tokens, query i may attend through K-Q+i.
PyTorch's lower-right causal bias describes this without allocating a Q×K mask
on the Flash path. Using is_causal=True directly would align at the wrong end.
"""
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right


def cached_attention(module, query, key, value, attention_mask=None,
                     dropout=0.0, scaling=None, **kwargs):
    if attention_mask is not None:
        raise ValueError("This attention path supports unpadded single sequences only")
    if query.shape[0] != 1 or query.shape[-2] > key.shape[-2]:
        raise ValueError("Expected one sequence with a contiguous cached prefix")
    length, total = query.shape[-2], key.shape[-2]
    common = dict(dropout_p=dropout, scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    if length == 1:
        result = F.scaled_dot_product_attention(query, key, value, **common)
    elif length == total:
        result = F.scaled_dot_product_attention(query, key, value, is_causal=True, **common)
    else:
        result = F.scaled_dot_product_attention(query, key, value,
                                               attn_mask=causal_lower_right(length, total), **common)
    return result.transpose(1, 2).contiguous(), None
