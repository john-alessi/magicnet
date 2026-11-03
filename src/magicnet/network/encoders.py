import torch
from .constants import *
from ..schema import ZONES

class RelationEncoder(torch.nn.Module):
    def __init__(
        self,
        relation_dim,
    ):
        super().__init__()
        self.relation_dim = relation_dim
        self.emb = torch.nn.Embedding(len(RELATIONS), relation_dim)

    def forward(self, edges, n_tokens):
        distinct, counts = torch.unique(edges, dim=0, return_counts=True)
        emb = self.emb(distinct[:,0])
        emb = emb * torch.log1p(counts.float())[:,None]
        pairs, inv = torch.unique(distinct[:,1:], dim=0, return_inverse=True)
        summed = emb.new_zeros(len(pairs), self.relation_dim).index_add(0, inv, emb)
        return (summed, pairs)

class ComponentSetEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.component_kind_emb = torch.nn.Embedding(len(COMPONENT_KIND_INDEX), COMPONENT_DIM)
        self.key_encoder = RawEmbeddingEncoder()
        self.vector_encoder = RawEmbeddingEncoder()
        self.scalar_encoder = ScalarEncoder()
        self.categorical_encoder = CategoricalEncoder()
        self.color_encoder = ColorEncoder()
        self.component_encoder = ObjectComponentEncoder()
        self.attn = PooledAttention(in_dim=COMPONENT_DIM, out_dim=TOKEN_DIM)

    def forward(self, game_state: dict[str, torch.Tensor]):
        components_list = []
        indices_list = []

        if len(game_state['vectors'] > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['vectors'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                        self.vector_encoder(game_state['vectors'][:,EXTERNAL_EMBEDDING_DIM:2*EXTERNAL_EMBEDDING_DIM]), # value
                        self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['vector'])).unsqueeze(0).expand(game_state['vectors'].shape[0], -1), # kind
                    )
                    , dim=-1
                )
            )
            indices_list.append(game_state['vectors'][:,-1],)

        if len(game_state['scalars'] > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['scalars'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                        self.scalar_encoder(game_state['scalars'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+1]), # value
                        self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['scalar'])).unsqueeze(0).expand(game_state['scalars'].shape[0], -1), # kind
                    )
                    , dim=-1
                )
            )
            indices_list.append(game_state['scalars'][:,-1],)

        if len(game_state['colors'] > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['colors'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                        self.color_encoder(game_state['colors'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+len(COLOR_WHEEL)]), # value
                        self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['color'])).unsqueeze(0).expand(game_state['colors'].shape[0], -1), # kind
                    )
                    , dim=-1
                )
            )
            indices_list.append(game_state['colors'][:,-1],)

        if len(game_state['flags'] > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['flags'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                        torch.zeros(game_state['flags'].shape[0], COMPONENT_DIM), # value
                        self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['flag'])).unsqueeze(0).expand(game_state['flags'].shape[0], -1), # kind
                    )
                , dim=-1
            )
        )
        indices_list.append(game_state['flags'][:,-1],)
        
        if len(game_state['categoricals'] > 0):
            components_list.append(
                torch.cat(
                    (
                        self.key_encoder(game_state['categoricals'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                        self.categorical_encoder(game_state['categoricals'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+2]), # value
                        self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['categorical'])).unsqueeze(0).expand(game_state['categoricals'].shape[0], -1), # kind
                    )
                    ,dim=-1
                )
        )
        indices_list.append(game_state['categoricals'][:,-1],)

        components_unordered = self.component_encoder(torch.cat(components_list, dim=0))
        indices_unordered = torch.cat(indices_list).to(torch.int32)

        objects = self.attn(components_unordered, indices_unordered)

        return objects

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

    def forward(self, components_unordered, indices_unordered):
        padded, mask = self.group_components(components_unordered, indices_unordered)
        n_objects, L, _ = padded.shape
        k = self.k(padded).view(n_objects, L, self.n_heads, self.dh)
        v = self.v(padded).view(n_objects, L, self.n_heads, self.dv)
        attn_scores = torch.einsum('hd,nlhd->nhl', self.q_cls, k) / self.dh**-0.5
        attn_scores = attn_scores.masked_fill(~mask[:, None, :], float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).nan_to_num(0.0)
        game_objects = torch.einsum('nhl,nlhd->nhd', attn_weights, v).reshape(n_objects, self.out_dim)
        return self.out(game_objects)


    def group_components(self, components_unordered, indices_unordered):
        n_objects = torch.unique(indices_unordered).numel() + 1
        order = torch.argsort(indices_unordered, stable=True)
        components = components_unordered[order]
        indices = indices_unordered[order]
        counts = torch.bincount(indices)
        start_indices = torch.cumsum(counts, 0) - counts
        positions = torch.arange(len(indices)) - start_indices[indices]
        padded = components.new_zeros(n_objects, counts.max(), components.shape[-1])
        padded[indices, positions] = components
        mask = torch.zeros(n_objects, padded.shape[1], dtype=torch.bool)
        mask[indices, positions] = True
        return padded, mask


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
        self.categorical_emb = torch.nn.ModuleList([
            torch.nn.Embedding(len(n), COMPONENT_DIM)
            for n in VOCABS.values()
        ])

    def forward(self, x):
        out = torch.zeros(*x.shape[:-1], COMPONENT_DIM)
        for cat,emb in enumerate(self.categorical_emb):
            idx = (x[...,-1] == cat)
            out[idx] = emb(x[idx][:,0].to(torch.int32))
        return out

class RawEmbeddingEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(EXTERNAL_EMBEDDING_DIM, COMPONENT_DIM),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)

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
        return self.mlp(x)

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
        return self.mlp(x)

class ManaCostEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dim = 256,
        out_dim = 256,
    ):
        super().__init__()

    def forward(self, mana: str):
        # accepts mana in scryfall format
        raise NotImplementedError