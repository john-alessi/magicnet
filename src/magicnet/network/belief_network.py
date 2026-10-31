import torch
from ..schema import ZONES, ZONE_VOCAB_SIZE, RELATION_VOCAB_SIZE
from .encoders import *
from .constants import *


class BeliefNetwork(torch.nn.Module):
    def __init__(
        self,
        nl_emb_dim = 3072,
        native_emb_dim = 256,
        qkv_dim = 128,
        n_att_heads = 4,
        n_blocks = 1,
    ):
        super().__init__()
        assert(native_emb_dim % n_att_heads == 0)

        self.component_kind_emb = torch.nn.Embedding(len(COMPONENT_KIND_INDEX), COMPONENT_DIM)
        #self.zone_emb = torch.nn.Embedding(num_embeddings=ZONE_VOCAB_SIZE, embedding_dim=native_emb_dim)
        #self.relation_emb = torch.nn.Embedding(num_embeddings=RELATION_VOCAB_SIZE, embedding_dim=qkv_dim)
        #self.transformer_blocks = torch.nn.ModuleList([
        #    RelationalAttention() for _ in range(n_blocks)
        #])
        self.key_encoder = RawEmbeddingEncoder()
        self.vector_encoder = RawEmbeddingEncoder()
        self.scalar_encoder = ScalarEncoder()
        self.color_encoder = ColorEncoder()
        self.component_encoder = ObjectComponentEncoder()

    def forward(self, game_state: dict[str, torch.Tensor]):
        vectors = torch.cat(
            (
                self.key_encoder(game_state['vectors'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                self.vector_encoder(game_state['vectors'][:,EXTERNAL_EMBEDDING_DIM:2*EXTERNAL_EMBEDDING_DIM]), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['vector'])).unsqueeze(0).expand(game_state['vectors'].shape[0], -1), # kind
                #game_state['vectors'][:,-1].unsqueeze(-1) # index
            )
            , dim=-1
        )

        scalars = torch.cat(
            (
                self.key_encoder(game_state['scalars'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                self.scalar_encoder(game_state['scalars'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+1]), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['scalar'])).unsqueeze(0).expand(game_state['scalars'].shape[0], -1), # kind
                #game_state['scalars'][:,-1].unsqueeze(-1) # index
            )
            , dim=-1
        )

        colors = torch.cat(
            (
                self.key_encoder(game_state['colors'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                self.color_encoder(game_state['colors'][:,EXTERNAL_EMBEDDING_DIM:EXTERNAL_EMBEDDING_DIM+len(COLOR_WHEEL)]), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['color'])).unsqueeze(0).expand(game_state['colors'].shape[0], -1), # kind
                #game_state['colors'][:,-1].unsqueeze(-1) # index
            )
            , dim=-1
        )

        flags = torch.cat(
            (
                self.key_encoder(game_state['flags'][:,:EXTERNAL_EMBEDDING_DIM]), # key
                torch.zeros(game_state['flags'].shape[0], COMPONENT_DIM), # value
                self.component_kind_emb(torch.tensor(COMPONENT_KIND_INDEX['flag'])).unsqueeze(0).expand(game_state['flags'].shape[0], -1), # kind
                #game_state['flags'][:,-1].unsqueeze(-1) # index
            )
            , dim=-1
        )

        components = self.component_encoder(torch.cat((vectors, scalars, colors, flags), dim=0))

        breakpoint()
        return 42

class RelationalAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x