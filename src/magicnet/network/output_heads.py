import torch
from .constants import *

class ValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(TOKEN_DIM, VALUE_HEAD_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(VALUE_HEAD_DIM, VALUE_HEAD_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(VALUE_HEAD_DIM, 3),
        )

    def forward(self, x: torch.Tensor, global_token_indices: torch.Tensor):
        tokens = x[torch.arange(x.size(0)),global_token_indices]
        return torch.softmax(self.mlp(tokens), dim=1)