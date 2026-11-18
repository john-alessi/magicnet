import torch
import numpy as np
from .constants import *
from .encoders import GameStateEncoder, DecisionEncoder
from .distributions import HiddenCardPosterior
from .output_heads import ValueHead, PolicyHead

class MagicNet(torch.nn.Module):
    def __init__(
        self,
        pca_basis,
        persist_embedding_cache = True,
    ):
        super().__init__()
        self.emb_cache = EmbeddingCache(persist_embedding_cache)
        self.hidden_card_component_key = torch.nn.Parameter(torch.randn(COMPONENT_DIM))
        self.belief_network = BeliefNetwork(pca_basis, self.emb_cache)
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
        pca_basis,
        emb_cache = None,
        persist_embedding_cache = True,
    ):
        super().__init__()
        self.emb_cache = emb_cache or EmbeddingCache(persist_embedding_cache)
        self.belief_encoder = GameStateEncoder(n_blocks=BELIEF_NETWORK_LAYERS)
        self.hidden_card_posterior = HiddenCardPosterior(pca_basis)

    def forward(
        self,
        game_state:
        dict[str, torch.Tensor],
        targets: torch.Tensor = None,
        n_samples: torch.Tensor = 0,
        update_cache=True,
    ):
        emb_cache = self.emb_cache(game_state) if update_cache or self.emb_cache.persist_between_passes == False else self.emb_cache.table
        belief_state, _ = self.belief_encoder(game_state, emb_cache)
        likelihood, samples = self.hidden_card_posterior(
            belief_state,
            game_state['unknown_cards'] if n_samples == 0 else game_state['unknown_cards_sampling'],
            targets=targets,
            n_samples=n_samples,
        )
        return likelihood, samples

class EmbeddingCache(torch.nn.Module):
    def __init__(self, persist_between_passes = True):
        super().__init__()
        self.persist_between_passes = persist_between_passes
        if persist_between_passes:
            self.register_buffer('table', torch.empty(0, EXTERNAL_EMBEDDING_DIM, dtype=torch.float32), persistent=False)
    
    def forward(self, game_state: dict[str, torch.Tensor]):
        if self.persist_between_passes:
            if self.table.shape[0] != game_state['emb_lookup_offset']:
                raise ValueError(f'offset {game_state['emb_lookup_offset']} out of sync with cache size {self.table.shape[0]}')
            self.table = torch.cat((self.table, game_state['emb_lookup']))
            return self.table
        else:
            return game_state['emb_lookup']