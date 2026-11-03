import torch
from ..schema import ZONES, ZONE_VOCAB_SIZE, RELATION_VOCAB_SIZE
from .encoders import *
from .constants import *


class BeliefNetwork(torch.nn.Module):
    def __init__(
        self,
        d_attn = 64,
        n_blocks = 4,
    ):
        super().__init__()
        self.object_encoder = ComponentSetEncoder()
        self.event_encoder = ComponentSetEncoder()
        self.relation_encoder = RelationEncoder(d_attn)
        self.transformer_blocks = torch.nn.ModuleList([
            RelationalAttention(d_attn) for _ in range(n_blocks)
        ])

    def forward(self, game_state: dict):
        tokens = torch.cat((
            self.object_encoder(game_state['state']),
            self.event_encoder(game_state['history']),
        ), dim=0)
        edge_emb, edge_pairs = self.relation_encoder(game_state['edges'], game_state['history']['num_tokens'] + game_state['state']['num_tokens'])
        for attn in self.transformer_blocks:
            tokens = attn(tokens, edge_emb, edge_pairs)
        unknown_cards = tokens[game_state['unknown_cards'], :]
        breakpoint()
        raise NotImplementedError

class RelationalAttention(torch.nn.Module):
    def __init__(
        self,
        d_attn,
        n_heads=4,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = TOKEN_DIM // n_heads
        self.qkv = torch.nn.Linear(TOKEN_DIM, 3*TOKEN_DIM)

    def forward(self, tokens, edge_emb, edge_pairs):
        return tokens