import torch
from .constants import *
from ..schema import ZONES
from .attention import RelationalAttentionBlock, PooledAttention, RelationalCrossAttentionBlock

class DecisionEncoder(torch.nn.Module):
    def __init__(
        self,
        n_blocks: int,
    ):
        super().__init__()
        self.action_encoder = ComponentSetEncoder()
        self.relation_encoder = RelationEncoder()
        self.transformer_blocks = torch.nn.ModuleList([
            RelationalCrossAttentionBlock() for _ in range(n_blocks)
        ])

    def forward(
        self,
        game_state: dict,
        emb_cache: torch.Tensor,
        policy_state: torch.Tensor,
        policy_mask: torch.Tensor,
    ):
        if game_state['actions']['num_tokens'] == 0: # choices are all refs in all batches
            batches, mask = None, None
        else:
            tokens = self.action_encoder(game_state['actions'], emb_cache)
            if len(game_state['action_edges']) > 0:
                edge_emb, edge_pairs = self.relation_encoder(game_state['action_edges'])
            else:
                edge_emb, edge_pairs = None, None
            #batches, mask = batch_tokens(tokens, game_state['decode_batch_indices'], game_state['decode_batch_dim'], game_state['decode_batch_start_indices'])
            order = game_state['decode_batch_info']['order']
            indices = game_state['decode_batch_info']['indices']
            positions = game_state['decode_batch_info']['positions']
            batches = tokens.new_zeros(game_state['decode_batch_info']['dim'])
            batches[indices[order], positions] = tokens[order]
            mask = game_state['decode_batch_info']['mask']
            for attn in self.transformer_blocks:
                batches = attn(batches, mask, policy_state, policy_mask, edge_emb, edge_pairs)
        
        return batches, mask

class GameStateEncoder(torch.nn.Module):
    def __init__(
        self,
        n_blocks: int,
    ):
        super().__init__()
        self.object_encoder = ComponentSetEncoder()
        self.event_encoder = ComponentSetEncoder()
        self.relation_encoder = RelationEncoder()
        self.transformer_blocks = torch.nn.ModuleList([
            RelationalAttentionBlock() for _ in range(n_blocks)
        ])

    def forward(
        self,
        game_state: dict,
        emb_cache: torch.Tensor,
        hidden_card_tokens: torch.Tensor = None,
        hidden_card_component_key: torch.Tensor = None,
    ):
        tokens = torch.cat((
            self.event_encoder(game_state['history'], emb_cache),
            self.object_encoder(
                game_state['state'],
                emb_cache,
                hidden_card_tokens,
                game_state['unknown_cards_flat'],
                hidden_card_component_key),
        ), dim=0)
        edge_emb, edge_pairs = self.relation_encoder(game_state['edges'])
        order = game_state['encode_batch_info']['order']
        indices = game_state['encode_batch_info']['indices']
        positions = game_state['encode_batch_info']['positions']
        batches = tokens.new_zeros(game_state['encode_batch_info']['dim'])
        batches[indices[order], positions] = tokens[order]
        mask = game_state['encode_batch_info']['mask']
        for attn in self.transformer_blocks:
            batches = attn(batches, mask, edge_emb, edge_pairs)
        return batches, mask

class RelationEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(len(RELATIONS), TOKEN_DIM)

    def forward(self, edges):
        distinct, counts = torch.unique(edges, dim=0, return_counts=True)
        emb = self.emb(distinct[:,0])
        emb = emb * torch.log1p(counts.float())[:,None]
        pairs, inv = torch.unique(distinct[:,1:], dim=0, return_inverse=True)
        summed = emb.new_zeros(len(pairs), TOKEN_DIM).index_add(0, inv, emb)
        return (summed, pairs)

class ComponentSetEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.component_kind_emb = torch.nn.Embedding(len(COMPONENT_KIND_INDEX), COMPONENT_DIM)
        self.mana_cost_key = torch.nn.Parameter(torch.randn(COMPONENT_DIM))
        self.key_encoder = RawEmbeddingEncoder()
        self.vector_encoder = RawEmbeddingEncoder()
        self.scalar_encoder = ScalarEncoder()
        self.categorical_encoder = CategoricalEncoder()
        self.color_encoder = ColorEncoder()
        self.component_encoder = ObjectComponentEncoder()
        self.mana_cost_encoder = ManaCostEncoder()
        self.attn = PooledAttention(in_dim=COMPONENT_DIM, out_dim=TOKEN_DIM)
        self.register_buffer('vector_kind_raw_emb', torch.tensor(COMPONENT_KIND_INDEX['vector']))
        self.register_buffer('scalar_kind_raw_emb', torch.tensor(COMPONENT_KIND_INDEX['scalar']))
        self.register_buffer('color_kind_raw_emb', torch.tensor(COMPONENT_KIND_INDEX['color']))
        self.register_buffer('flag_kind_raw_emb', torch.tensor(COMPONENT_KIND_INDEX['flag']))
        self.register_buffer('categorical_kind_raw_emb', torch.tensor(COMPONENT_KIND_INDEX['categorical']))
        self.register_buffer('mana_cost_kind_raw_emb', torch.tensor(COMPONENT_KIND_INDEX['mana_cost']))

    def forward(
        self,
        game_state: dict[str, torch.Tensor],
        emb_cache: torch.Tensor,
        hidden_card_tokens: torch.Tensor = None,
        hidden_card_indices: torch.Tensor = None,
        hidden_card_component_key: torch.Tensor = None,
    ):
        components_list = []
        indices_list = []

        if hidden_card_tokens is not None and hidden_card_indices is not None and hidden_card_component_key is not None:
            components_list.append(
                torch.cat(
                    (
                        hidden_card_component_key.unsqueeze(0).expand(hidden_card_tokens.shape[0]*hidden_card_tokens.shape[1], -1), # key
                        self.vector_encoder(hidden_card_tokens.flatten(0,1)), # value
                        self.component_kind_emb(self.vector_kind_raw_emb).unsqueeze(0).expand(hidden_card_tokens.shape[0]*hidden_card_tokens.shape[1], -1), # kind
                    )
                    , dim=-1
                )
            )

        if (game_state['vectors'].numel() > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['vectors'][:,0], emb_cache), # key
                        self.vector_encoder(game_state['vectors'][:,1], emb_cache), # value
                        self.component_kind_emb(self.vector_kind_raw_emb).unsqueeze(0).expand(game_state['vectors'].shape[0], -1), # kind
                    )
                    , dim=-1
                )
            )
            indices_list.append(game_state['vectors'][:,-1],)

        if (game_state['scalars'].numel() > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['scalars'][:,0], emb_cache), # key
                        self.scalar_encoder(game_state['scalars'][:,1]), # value
                        self.component_kind_emb(self.scalar_kind_raw_emb).unsqueeze(0).expand(game_state['scalars'].shape[0], -1), # kind
                    )
                    , dim=-1
                )
            )
            indices_list.append(game_state['scalars'][:,-1],)

        if (game_state['colors'].numel() > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['colors'][:,0], emb_cache), # key
                        self.color_encoder(game_state['colors'][:,1:1+len(COLOR_WHEEL)]), # value
                        self.component_kind_emb(self.color_kind_raw_emb).unsqueeze(0).expand(game_state['colors'].shape[0], -1), # kind
                    )
                    , dim=-1
                )
            )
            indices_list.append(game_state['colors'][:,-1],)

        if (game_state['flags'].numel() > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['flags'][:,0], emb_cache), # key
                        game_state['flags'].new_zeros(game_state['flags'].shape[0], COMPONENT_DIM), # value
                        self.component_kind_emb(self.flag_kind_raw_emb).unsqueeze(0).expand(game_state['flags'].shape[0], -1), # kind
                    )
                    , dim=-1
                )
            )
            indices_list.append(game_state['flags'][:,-1],)
        
        if (game_state['categoricals'].numel() > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['categoricals'][:,0], emb_cache), # key
                        self.categorical_encoder(game_state['categoricals'][:,1]), # value
                        self.component_kind_emb(self.categorical_kind_raw_emb).unsqueeze(0).expand(game_state['categoricals'].shape[0], -1), # kind
                    )
                    ,dim=-1
                )
            )
            indices_list.append(game_state['categoricals'][:,-1],)

        if (game_state['mana_costs'] is not None and game_state['mana_costs'].numel() > 0):
            symbols = self.mana_cost_encoder(game_state['mana_costs'], game_state['mana_cost_batch_info'])
            components_list.append(
                torch.cat(
                    (
                        self.mana_cost_key.unsqueeze(0).expand(game_state['mana_cost_batch_info']['dim'][0], COMPONENT_DIM),
                        symbols,
                        self.component_kind_emb(self.mana_cost_kind_raw_emb).unsqueeze(0).expand(game_state['mana_cost_batch_info']['dim'][0], -1),
                    )
                    ,dim=-1
                )
            )
            indices_list.append(game_state['mana_cost_indices'])

        components_unordered = self.component_encoder(torch.cat(components_list, dim=0))
        indices_unordered = torch.cat(indices_list).to(torch.int32)

        objects = self.attn(components_unordered, game_state['component_batch_info'])

        return objects

class ObjectComponentEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3*COMPONENT_DIM, COMPONENT_DIM),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)

class CategoricalEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        emb_dim = sum([len(v) for v in VOCABS.values()])
        self.categorical_emb = torch.nn.Embedding(emb_dim, COMPONENT_DIM)

    def forward(self, x):
        return self.categorical_emb(x)

class RawEmbeddingEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(EXTERNAL_EMBEDDING_DIM, COMPONENT_DIM),
            torch.nn.GELU(),
        )

    def forward(self, x, embedding_cache = None):
        return self.mlp(x) if embedding_cache is None else self.mlp(embedding_cache[x])

class ScalarEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            # TODO: figure out how to properly featurize scalars
            # (this is placeholder logic)
            torch.nn.Linear(1, COMPONENT_DIM),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x.unsqueeze(-1).to(torch.float32))

class ColorEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(len(COLOR_WHEEL), COMPONENT_DIM),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x.to(torch.float32))

class ManaCostEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(MANA_SYMBOL_DIM-2, COMPONENT_DIM),
            torch.nn.GELU()
        )
        self.pooled_attention = PooledAttention(in_dim=COMPONENT_DIM, out_dim=COMPONENT_DIM)

    def forward(self, mana_symbols: torch.Tensor, batch_info: dict):
        a = self.mlp(mana_symbols[:,:-1])
        costs = self.pooled_attention(a, batch_info)
        return costs