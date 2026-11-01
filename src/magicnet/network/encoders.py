import torch
from .constants import *
from ..schema import ZONES

class GameObjectEncoder(torch.nn.Module):
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
        self.component_pooler = ComponentPooler()

    def forward(self, game_state: dict[str, torch.Tensor]):
        vectors = torch.cat(
            (
                self.key_encoder(game_state['vectors'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                self.vector_encoder(game_state['vectors'][:,EXTERNAL_EMBEDDING_DIM:2*EXTERNAL_EMBEDDING_DIM]), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['vector'])).unsqueeze(0).expand(game_state['vectors'].shape[0], -1), # kind
            )
            , dim=-1
        )

        scalars = torch.cat(
            (
                self.key_encoder(game_state['scalars'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                self.scalar_encoder(game_state['scalars'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+1]), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['scalar'])).unsqueeze(0).expand(game_state['scalars'].shape[0], -1), # kind
            )
            , dim=-1
        )

        colors = torch.cat(
            (
                self.key_encoder(game_state['colors'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                self.color_encoder(game_state['colors'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+len(COLOR_WHEEL)]), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['color'])).unsqueeze(0).expand(game_state['colors'].shape[0], -1), # kind
            )
            , dim=-1
        )

        flags = torch.cat(
            (
                self.key_encoder(game_state['flags'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                torch.zeros(game_state['flags'].shape[0], COMPONENT_DIM), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['flag'])).unsqueeze(0).expand(game_state['flags'].shape[0], -1), # kind
            )
            , dim=-1
        )
        
        categoricals = torch.cat(
            (
                self.key_encoder(game_state['categoricals'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                self.categorical_encoder(game_state['categoricals'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+2]), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['categorical'])).unsqueeze(0).expand(game_state['categoricals'].shape[0], -1), # kind
            )
            ,dim=-1
        )

        components_unordered = self.component_encoder(torch.cat((vectors, scalars, colors, flags, categoricals), dim=0))
        indices_unordered = torch.cat((
            game_state['vectors'][:,-1],
            game_state['scalars'][:,-1],
            game_state['colors'][:,-1],
            game_state['flags'][:,-1],
            game_state['categoricals'][:,-1],
        )).to(torch.int32)

        n_objects = torch.unique(indices_unordered).numel() + 1
        order = torch.argsort(indices_unordered, stable=True)
        components = components_unordered[order]
        indices = indices_unordered[order]
        counts = torch.bincount(indices)
        start_indices = torch.cumsum(counts, 0) - counts
        positions = torch.arange(len(indices)) - start_indices[indices]
        padded = components.new_zeros(n_objects, counts.max(), components.shape[-1])
        padded[indices, positions] = components

        objects = self.component_pooler(components, indices)

        breakpoint()
        return objects

class ComponentPooler(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, components, indices):
        return None

class ObjectComponentEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3*COMPONENT_DIM, TOKEN_DIM),
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
            for n in CATEGORICALS.values()
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
        pass