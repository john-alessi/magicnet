import torch
import numpy as np
from .constants import *
from .encoders import GameStateEncoder
from .distributions import HiddenCardPosterior
from .output_heads import ValueHead

class MagicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('raw_embedding_cache', torch.empty(0, EXTERNAL_EMBEDDING_DIM, dtype=torch.float32), persistent=False)
        self.belief_network = GameStateEncoder(n_blocks=BELIEF_NETWORK_LAYERS)
        self.policy_trunk = GameStateEncoder(n_blocks=POLICY_TRUNK_LAYERS)
        self.hidden_card_posterior = HiddenCardPosterior()
        self.value_head = ValueHead()
        self.hidden_card_component_key = torch.nn.Parameter(torch.randn(COMPONENT_DIM))

    def forward(self, game_state: dict[str, torch.Tensor]):
        if self.raw_embedding_cache.shape[0] != game_state['emb_lookup_offset']:
            raise ValueError(f'offset {game_state['emb_lookup_offset']} out of sync with cache size {self.raw_embedding_cache.shape[0]}')
        self.raw_embedding_cache = torch.cat((self.raw_embedding_cache, game_state['emb_lookup']))
        belief_state = self.belief_network(game_state, self.raw_embedding_cache)
        _, hidden_card_samples = self.hidden_card_posterior(
            belief_state,
            game_state['unknown_cards'],
            game_state['pca'],
            n_samples=BELIEF_NETWORK_SAMPLES,
        )
        policy_state = self.policy_trunk(
            game_state,
            self.raw_embedding_cache,
            hidden_card_samples,
            self.hidden_card_component_key)
        value = self.value_head(policy_state, game_state['global_token_indices'])
        return 42
