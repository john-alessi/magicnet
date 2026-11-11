import torch
from .constants import *

class PointerAttention(torch.nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.glimpse = torch.nn.MultiheadAttention(TOKEN_DIM, POINTER_GLIMPSE_HEADS, batch_first=True)
        self.q_ln = torch.nn.LayerNorm(TOKEN_DIM)
        self.k_ln = torch.nn.LayerNorm(TOKEN_DIM)
        self.wq = torch.nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.wk = torch.nn.Linear(TOKEN_DIM, TOKEN_DIM)

    def forward(self, candidates: torch.Tensor, mask: torch.Tensor, prompt_tokens: torch.Tensor):
        can = self.k_ln(candidates)
        pro = self.q_ln(prompt_tokens)
        q,_ = self.glimpse(pro[:, None], can, can, key_padding_mask=mask, need_weights=False)
        q = self.wq(q[:,0])
        k = self.wk(can)
        logits = torch.einsum('nd,nld->nl', q, k) / (TOKEN_DIM ** 0.5)
        logits = POINTER_TANH_CLIP * torch.tanh(logits)
        return logits.masked_fill(mask, float('-inf'))

class PooledAttention(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        cls_scale = 0.02,
        d_attn = 64,
        n_heads = 4,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_heads, self.dh, self.dv = n_heads, d_attn // n_heads, out_dim // n_heads
        self.q_cls = torch.nn.Parameter(torch.randn(n_heads, self.dh) * cls_scale)
        self.k = torch.nn.Linear(in_dim, d_attn)
        self.v = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.Linear(out_dim, out_dim),
        )
        self.out = torch.nn.Linear(out_dim, out_dim)

    def forward(self, components_unordered, indices_unordered, num_tokens):
        padded, mask = self.group_components(components_unordered, indices_unordered, num_tokens)
        n_objects, L, _ = padded.shape
        k = self.k(padded).view(n_objects, L, self.n_heads, self.dh)
        v = self.v(padded).view(n_objects, L, self.n_heads, self.dv)
        attn_scores = torch.einsum('hd,nlhd->nhl', self.q_cls, k) * self.dh**-0.5
        attn_scores = attn_scores.masked_fill(~mask[:, None, :], float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1).nan_to_num(0.0)
        game_objects = torch.einsum('nhl,nlhd->nhd', attn_weights, v).reshape(n_objects, self.out_dim)
        return self.out(game_objects)


    def group_components(self, components_unordered, indices_unordered, num_tokens):
        order = torch.argsort(indices_unordered, stable=True)
        components = components_unordered[order]
        indices = indices_unordered[order]
        counts = torch.bincount(indices, minlength=num_tokens)
        start_indices = torch.cumsum(counts, 0) - counts
        positions = torch.arange(len(indices)) - start_indices[indices]
        padded = components.new_zeros(num_tokens, counts.max(), components.shape[-1])
        padded[indices, positions] = components
        mask = torch.zeros(num_tokens, padded.shape[1], dtype=torch.bool)
        mask[indices, positions] = True
        return padded, mask

class RelationalCrossAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(TOKEN_DIM)
        self.self_attn = torch.nn.MultiheadAttention(TOKEN_DIM, DECODER_SELF_ATTENTION_HEADS, batch_first=True)
        self.ln2 = torch.nn.LayerNorm(TOKEN_DIM)
        self.cross_attn = RelationalCrossAttention()
        self.ln3 = torch.nn.LayerNorm(TOKEN_DIM)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(TOKEN_DIM, FEED_FORWARD_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(FEED_FORWARD_DIM, TOKEN_DIM),
            torch.nn.Dropout(DROPOUT),
        )

    def forward(self, action_tokens, action_mask, state_tokens, state_mask, edge_emb, edge_pairs):
        empty = action_mask.all(-1)
        safe_mask = action_mask.clone()
        safe_mask[empty, 0] = False # unmask at least 1 position so softmax doesn't divide by 0
        n = self.ln1(action_tokens)
        tokens = action_tokens + self.self_attn(n, n, n, key_padding_mask=safe_mask)[0]
        tokens = tokens + self.cross_attn(self.ln2(tokens), action_mask, state_tokens, state_mask, edge_emb, edge_pairs)
        tokens = tokens + self.ff(self.ln3(tokens))
        tokens = tokens.masked_fill(action_mask[..., None], 0.0)
        return tokens

class RelationalCrossAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = TOKEN_DIM // REL_ATTENTION_HEADS
        self.kv = torch.nn.Linear(TOKEN_DIM, 2*TOKEN_DIM)
        self.q = torch.nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.out = torch.nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.relation_gain = torch.nn.Parameter(torch.ones(REL_ATTENTION_HEADS))

    def forward(self, action_tokens, action_mask, state_tokens, state_mask, edge_emb, edge_pairs):
        N = state_tokens.shape[0]
        L_actions = action_tokens.shape[1]
        L_state = state_tokens.shape[1]
        src = edge_pairs[:,0]
        dst = edge_pairs[:,1]
        bat = edge_pairs[:,2]
        k, v = (
            head.view(N, L_state, REL_ATTENTION_HEADS, self.head_dim).transpose(1,2)
            for head in self.kv(state_tokens).chunk(2,-1)
        )
        q = self.q(action_tokens).view(N, L_actions, REL_ATTENTION_HEADS, self.head_dim).transpose(1,2)
        a = edge_emb.view(edge_emb.shape[0], REL_ATTENTION_HEADS, self.head_dim) * self.relation_gain[None, :, None]
        relation_weights = (q.transpose(1,2)[bat, src] * a).sum(-1) * (self.head_dim ** -0.5)
        bias = state_tokens.new_zeros(N, REL_ATTENTION_HEADS, L_actions, L_state)
        bias[bat, :, src, dst] = relation_weights
        bias = bias.masked_fill(state_mask[:, None, None, :], float('-inf'))
        context_vectors = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(context_vectors.transpose(1,2).reshape(N, L_actions, TOKEN_DIM))

class RelationalAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(TOKEN_DIM)
        self.attention = RelationalAttention()
        self.ln2 = torch.nn.LayerNorm(TOKEN_DIM)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(TOKEN_DIM, FEED_FORWARD_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(FEED_FORWARD_DIM, TOKEN_DIM),
            torch.nn.Dropout(DROPOUT),
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
            head.view(N, L, REL_ATTENTION_HEADS, self.head_dim).transpose(1,2)
            for head in self.qkv(batches).chunk(3,-1)
        )
        a = edge_emb.view(*edge_emb.shape[:-1], REL_ATTENTION_HEADS, self.head_dim) * self.relation_gain[None, :, None]
        relation_weights = (q.transpose(1,2)[batch, src] * a).sum(-1) * (self.head_dim ** -0.5)
        bias = batches.new_zeros(N, REL_ATTENTION_HEADS, L, L)
        bias[batch, :, src, dest] = relation_weights
        bias = bias.masked_fill(mask[:, None, None, :], float('-inf'))

        context_vectors = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(context_vectors.transpose(1,2).reshape(N, L, TOKEN_DIM))