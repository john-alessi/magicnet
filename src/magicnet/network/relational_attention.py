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

    def forward(self, batches, mask, edge_emb, edge_pairs):
        tokens = batches + self.attention(self.ln1(batches),mask, edge_emb, edge_pairs)
        tokens = tokens + self.ff(self.ln2(tokens))
        return tokens

class RelationalAttention(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.head_dim = TOKEN_DIM // REL_ATTENTION_HEADS
        self.qkv = torch.nn.Linear(TOKEN_DIM, 3*TOKEN_DIM)
        self.out = torch.nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.relation_gain = torch.nn.Parameter(torch.ones(REL_ATTENTION_HEADS))

    def forward(self, batches, mask, edge_emb, edge_pairs):
        N = batches.shape[0]
        L = batches.shape[1]
        src = edge_pairs[:,0]
        dest = edge_pairs[:,1]
        batch = edge_pairs[:,2]
        q, k, v = (
            torch.einsum('nleh->nhle', head.view(N, L, self.head_dim, REL_ATTENTION_HEADS))
            for head in self.qkv(batches).chunk(3,-1)
        )
        a = edge_emb.view(*edge_emb.shape[:-1], REL_ATTENTION_HEADS, self.head_dim) * self.relation_gain[None, :, None]
        relation_weights = (q.transpose(1,2)[batch, src] * a).sum(-1) * (self.head_dim ** -0.5)
        bias = batches.new_zeros(N, REL_ATTENTION_HEADS, L, L)
        bias[batch, :, src, dest] = relation_weights
        bias = bias.masked_fill(mask[:, None, None, :], float('-inf'))

        context_vectors = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(context_vectors.transpose(1,2).reshape(N, L, TOKEN_DIM))