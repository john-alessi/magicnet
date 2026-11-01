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

        #self.zone_emb = torch.nn.Embedding(num_embeddings=ZONE_VOCAB_SIZE, embedding_dim=native_emb_dim)
        #self.relation_emb = torch.nn.Embedding(num_embeddings=RELATION_VOCAB_SIZE, embedding_dim=qkv_dim)
        #self.transformer_blocks = torch.nn.ModuleList([
        #    RelationalAttention() for _ in range(n_blocks)
        #])
        self.object_encoder = GameObjectEncoder()

    def forward(self, game_state: dict[str, torch.Tensor]):
        return self.object_encoder(game_state)

class RelationalAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x