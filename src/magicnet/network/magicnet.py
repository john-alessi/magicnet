import torch
import numpy as np
from .belief_network import BeliefNetwork

class MagicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.belief_network = BeliefNetwork()

    def forward(self, game_state: dict[str, torch.Tensor]):
        hidden_card_samples,_ = self.belief_network(game_state, n_samples=100)
        return 42
