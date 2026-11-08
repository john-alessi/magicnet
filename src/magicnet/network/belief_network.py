import torch

from .encoders import *
from .constants import *
from .relational_attention import RelationBlock
from .distributions import HiddenCardPosterior


class BeliefNetwork(torch.nn.Module):
    def __init__(
        self,
        n_blocks = 5,
    ):
        super().__init__()
        self.object_encoder = ComponentSetEncoder()
        self.event_encoder = ComponentSetEncoder()
        self.relation_encoder = RelationEncoder()
        self.transformer_blocks = torch.nn.ModuleList([
            RelationBlock() for _ in range(n_blocks)
        ])
        self.hidden_card_flow = HiddenCardPosterior()

    def forward(self, game_state: dict, emb_cache: torch.Tensor, targets=None, n_samples=0):
        tokens = torch.cat((
            self.event_encoder(game_state['history'], emb_cache),
            self.object_encoder(game_state['state'], emb_cache),
        ), dim=0)
        edge_emb, edge_pairs = self.relation_encoder(game_state['edges'])
        batches, mask = self.batch_tokens(tokens, game_state['batch_indices'], game_state['batch_dim'])
        for attn in self.transformer_blocks:
            batches = attn(batches, mask, edge_emb, edge_pairs)
        return self.hidden_card_flow(batches, game_state['unknown_cards'], targets, n_samples)

    def batch_tokens(self, tokens, batch_indices, batch_dim):
        order = torch.argsort(batch_indices, stable=True)
        counts = torch.bincount(batch_indices, minlength=batch_dim[0])
        start_indices = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(len(batch_indices)) - start_indices[batch_indices[order]]
        padded = tokens.new_zeros(batch_dim)
        padded[batch_indices[order], positions] = tokens[order]
        mask = torch.ones(*batch_dim[:-1], dtype=torch.bool)
        mask[batch_indices[order], positions] = False
        return padded, mask