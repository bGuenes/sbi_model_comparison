import torch
import torch.nn as nn

# --- Embeddings -----------------------------------------------------------------------------------------------------

class InputEmbedder(nn.Module):
    def __init__(self, nodes_size, hidden_size):
        super().__init__()
        self.nodes_size = nodes_size
        self.hidden_size = hidden_size

        self.node_embed = nn.Embedding(nodes_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, nodes_size, hidden_size))

        # Token embedding layers for values, node IDs, and condition masks
        self.node_ids = torch.arange(self.nodes_size)  # Node IDs
        self.embedding_net_id = nn.Embedding(self.nodes_size, self.hidden_size)  # Embedding for node IDs
        self.condition_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.5)  # Learnable condition embedding

    def embedding_net_value(self, x):
        # Value embedding net (here we just repeat the value)
        return x.repeat(1,1,self.hidden_size)

    def forward(self, x, condition_mask):

        # Value embedding
        value_embedded = self.embedding_net_value(x.unsqueeze(-1))
        # Node ID embedding
        id_embedded = self.embedding_net_id(self.node_ids.to(x.device).repeat(x.shape[0],1))
        # Condition embedding
        condition_embedded = self.condition_embedding * condition_mask.unsqueeze(-1)
        
        # --- Create Token ---
        # Concatenate all embeddings to create the input for the Transformer
        x_embed = torch.cat([value_embedded, id_embedded, condition_embedded], dim=-1).flatten(1)
        #x_embed = x_embed.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)

        return x_embed

class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier embedding module. Mostly used to embed time.

    Args:
        embed_dim (int, optional): Output dimesion. Defaults to 64.
    """
    def __init__(self, embed_dim=64, scale=30.):
        super().__init__()
        self.embed_dim = embed_dim
        self.B = nn.Parameter(torch.randn(embed_dim // 2, 1) * scale)

    def forward(self, t):
        t = 2 * torch.pi * torch.matmul(t, self.B.T)
        term1 = torch.cos(t)
        term2 = torch.sin(t)
        out = torch.cat([term1, term2], dim=-1)
        return out

# --- Transformer Blocks ----------------------------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, nodes_size, t_dim, mlp_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.nodes_size = nodes_size
        self.num_tokens = 3*self.hidden_size*self.nodes_size

        self.attention = nn.MultiheadAttention(self.num_tokens, num_heads=num_heads, add_bias_kv=True, )

        self.norm1 = nn.LayerNorm(self.num_tokens)
        self.norm2 = nn.LayerNorm(self.num_tokens)
        self.norm3 = nn.LayerNorm(self.num_tokens)

        self.context_embed = nn.Linear(t_dim, self.num_tokens)

        self.mlp = nn.Sequential(
            nn.Linear(self.num_tokens, self.num_tokens * mlp_ratio),
            nn.ReLU(),
            nn.Linear(self.num_tokens * mlp_ratio, self.num_tokens)
        )

    def forward(self, x, t):
        x_norm = self.norm1(x)
        x_attn = self.attention(x_norm, x_norm, x_norm, need_weights=False)[0]

        x = self.norm2(x + x_attn)

        x_mlp = self.mlp(x)
        t = self.context_embed(t)
        x_mlp = x_mlp + t

        x = self.norm3(x + x_mlp)
        
        return x

# --- Transformer ----------------------------------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        nodes_size,
        hidden_size=64,
        depth=6,
        num_heads=16,
        mlp_ratio=4,
        t_dim=64
    ):
        
        super().__init__()
        self.nodes_size = nodes_size
        self.num_heads = num_heads

        self.x_embedder = InputEmbedder(nodes_size=nodes_size, hidden_size=hidden_size)  
        self.t_embedder = GaussianFourierEmbedding(t_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, nodes_size, t_dim, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = nn.Linear(3*nodes_size*hidden_size, nodes_size)

    def forward(self, x, c, t):
        x = self.x_embedder(x, c)
        t = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, t)

        x = self.final_layer(x)
        return x