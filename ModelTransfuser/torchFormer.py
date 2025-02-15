import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, embed_dim = query.size()
        qkv = self.qkv_proj(query).reshape(batch_size, seq_length, 3, self.num_heads, embed_dim // self.num_heads)
        q, k, v = qkv.chunk(3, dim=2)

        q = q.permute(0, 3, 1, 2).reshape(batch_size * self.num_heads, seq_length, embed_dim // self.num_heads)
        k = k.permute(0, 3, 1, 2).reshape(batch_size * self.num_heads, seq_length, embed_dim // self.num_heads)
        v = v.permute(0, 3, 1, 2).reshape(batch_size * self.num_heads, seq_length, embed_dim // self.num_heads)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (embed_dim ** 0.5)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_length, embed_dim // self.num_heads)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)

        return self.out_proj(attn_output)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        mlp_output = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Tokenizer(nn.Module):
    def __init__(self, input_dim, output_dim, max_seq_len):
        super(Tokenizer, self).__init__()
        self.embedding = nn.Embedding(max_seq_len, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x) + self.embedding(torch.arange(x.size(1), device=x.device))


class Simformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, mlp_dim, max_seq_len, dropout=0.1):
        super(Simformer, self).__init__()
        self.tokenizer = Tokenizer(input_dim, embed_dim, max_seq_len)
        self.transformer = Transformer(embed_dim, num_heads, num_layers, mlp_dim, dropout)
        self.head = nn.Linear(embed_dim, input_dim)

    def forward(self, x, mask=None):
        x = self.tokenizer(x)
        x = self.transformer(x, mask)
        return self.head(x)


def conditional_flow_and_score_matching_loss(
    model, params, key, times, xs_source, xs_target, mean_fn, std_fn, estimate_score=False
):
    eps = torch.randn_like(xs_source)
    xs_t = mean_fn(xs_source, xs_target, times) + std_fn(xs_source, xs_target, times) * eps

    t = times.expand_as(xs_target)
    std_fn_grad = torch.autograd.grad(std_fn(xs_source, xs_target, t).sum(), t, create_graph=True)[0]
    mean_fn_grad = torch.autograd.grad(mean_fn(xs_source, xs_target, t).sum(), t, create_graph=True)[0]
    u_t = std_fn_grad * eps + mean_fn_grad

    if not estimate_score:
        v_t = model(params, times, xs_t)
        loss = ((v_t - u_t) ** 2).mean()
        return loss
    else:
        v_t, s_t = model(params, times, xs_t)
        loss = ((v_t - u_t) ** 2).mean()
        loss_score = ((s_t + eps / std_fn(xs_source, xs_target, times)) ** 2).mean()
        return loss, loss_score