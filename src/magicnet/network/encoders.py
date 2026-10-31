import torch
from .constants import *

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