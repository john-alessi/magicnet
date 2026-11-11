import torch
import numpy as np
from .constants import *
from .encoders import GameStateEncoder, DecisionEncoder
from .distributions import HiddenCardPosterior
from .output_heads import ValueHead, PolicyHead

class MagicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_cache = EmbeddingCache()
        self.hidden_card_component_key = torch.nn.Parameter(torch.randn(COMPONENT_DIM))
        self.belief_network = BeliefNetwork(self.emb_cache)
        self.policy_trunk = GameStateEncoder(n_blocks=POLICY_TRUNK_LAYERS)
        self.decision_encoder = DecisionEncoder(n_blocks=DECISION_ENCODER_LAYERS)
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, game_state: dict[str, torch.Tensor]):
        emb_cache = self.emb_cache(game_state)

        with torch.no_grad():
            _, hidden_card_samples = self.belief_network(
                game_state,
                n_samples=BELIEF_NETWORK_SAMPLES,
                update_cache=False,
            )

        policy_state, policy_mask = self.policy_trunk(
            game_state,
            emb_cache,
            hidden_card_samples,
            self.hidden_card_component_key)
        decisions, decisions_mask = self.decision_encoder(game_state, emb_cache, policy_state, policy_mask)

        choice_logits = self.policy_head(game_state, policy_state, policy_mask, decisions, decisions_mask)
        value_logits = self.value_head(policy_state, game_state['global_token_indices'])
        return value_logits, choice_logits

class BeliefNetwork(torch.nn.Module):
    def __init__(
        self,
        emb_cache = None,
    ):
        super().__init__()
        self.emb_cache = emb_cache or EmbeddingCache()
        self.belief_encoder = GameStateEncoder(n_blocks=BELIEF_NETWORK_LAYERS)
        self.hidden_card_posterior = HiddenCardPosterior()

    def forward(
        self,
        game_state:
        dict[str, torch.Tensor],
        targets: torch.Tensor = None,
        n_samples: torch.Tensor = 0,
        update_cache=True,
    ):
        emb_cache = self.emb_cache(game_state) if update_cache else self.emb_cache.table
        belief_state, _ = self.belief_encoder(game_state, emb_cache)
        likelihood, samples = self.hidden_card_posterior(
            belief_state,
            game_state['unknown_cards'] if n_samples == 0 else game_state['unknown_cards_sampling'],
            game_state['pca'],
            targets=targets,
            n_samples=n_samples,
        )
        return likelihood, samples

class EmbeddingCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('table', torch.empty(0, EXTERNAL_EMBEDDING_DIM, dtype=torch.float32), persistent=False)
    
    def forward(self, game_state: dict[str, torch.Tensor]):
        if self.table.shape[0] != game_state['emb_lookup_offset']:
            raise ValueError(f'offset {game_state['emb_lookup_offset']} out of sync with cache size {self.table.shape[0]}')
        self.table = torch.cat((self.table, game_state['emb_lookup']))
        return self.table