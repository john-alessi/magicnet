import torch
from .constants import *

class RelationBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(TOKEN_DIM)
        self.attention = RelationalAttention()
        self.ln2 = torch.nn.LayerNorm(TOKEN_DIM)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(TOKEN_DIM, FEED_FORWARD_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(FEED_FORWARD_DIM, TOKEN_DIM),
            torch.nn.Dropout(),
        )

    def forward(self, tokens, edge_emb, edge_pairs):
        tokens = tokens + self.attention(self.ln1(tokens), edge_emb, edge_pairs)
        tokens = tokens + self.ff(self.ln2(tokens))
        return tokens

class RelationalAttention(torch.nn.Module):
    def __init__(
        self,
        n_heads=4,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = TOKEN_DIM // n_heads
        self.qkv = torch.nn.Linear(TOKEN_DIM, 3*TOKEN_DIM)
        self.out = torch.nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.relation_gain = torch.nn.Parameter(torch.ones(n_heads))

    def forward(self, tokens, edge_emb, edge_pairs):
        L = tokens.shape[0]
        q, k, v = (
            torch.einsum('leh->hle', head.view(L, self.head_dim, self.n_heads))
            for head in self.qkv(tokens).chunk(3,-1)
        )
        a = edge_emb.view(*edge_emb.shape[:-1], self.n_heads, self.head_dim) * self.relation_gain[None, :, None]
        relation_weights = (q.transpose(0,1)[edge_pairs[:,0]] * a).sum(-1) * (self.head_dim ** -0.5)
        bias = tokens.new_zeros(self.n_heads, L, L)
        bias[:, edge_pairs[:,0], edge_pairs[:,1]] = relation_weights.transpose(0,1)
        context_vectors = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(context_vectors.reshape(L, TOKEN_DIM))