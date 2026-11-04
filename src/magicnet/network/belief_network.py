import torch
import zuko

from .encoders import *
from .constants import *
from .relational_attention import RelationBlock


class BeliefNetwork(torch.nn.Module):
    def __init__(
        self,
        n_blocks = 4,
    ):
        super().__init__()
        self.object_encoder = ComponentSetEncoder()
        self.event_encoder = ComponentSetEncoder()
        self.relation_encoder = RelationEncoder()
        self.transformer_blocks = torch.nn.ModuleList([
            RelationBlock() for _ in range(n_blocks)
        ])
        self.flow = zuko.flows.NSF(
            features = EXTERNAL_EMBEDDING_DIM,
            context=TOKEN_DIM,
            transforms=N_FLOW_TRANSFORMS,
            passes=2,
            hidden_features=(FLOW_DIM,FLOW_DIM)
        )

    def forward(self, game_state: dict, targets=None, n_samples=0):
        tokens = torch.cat((
            self.object_encoder(game_state['state']),
            self.event_encoder(game_state['history']),
        ), dim=0)
        edge_emb, edge_pairs = self.relation_encoder(
            game_state['edges'], game_state['history']['num_tokens'] + game_state['state']['num_tokens'])
        for attn in self.transformer_blocks:
            tokens = attn(tokens, edge_emb, edge_pairs)
        unknown_cards = tokens[game_state['unknown_cards'], :]
        distribution = self.flow(unknown_cards)
        return (
            distribution.rsample((n_samples,)) if n_samples > 0 else None,
            -distribution.log_prob(targets) if targets else None,
        )