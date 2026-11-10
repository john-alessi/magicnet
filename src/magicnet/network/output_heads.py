import torch
from .constants import *
from .encoders import ComponentSetEncoder
from .attention import PointerAttention

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
        return self.mlp(tokens)

class PolicyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointer_network = PointerAttention()

    def forward(
        self,
        game_state: dict[str, torch.Tensor],
        policy_state: torch.Tensor,
        policy_mask: torch.Tensor,
        decisions: torch.Tensor,
        decisions_mask: torch.Tensor,
    ):
        batch_dim = game_state['decode_batch_dim']
        prompt_tokens = policy_state[torch.arange(batch_dim[0]), game_state['prompt_token_indices']]
        candidates = policy_state.new_zeros(batch_dim)
        candidates_mask = torch.zeros(batch_dim[0], batch_dim[1], dtype=torch.bool)
        candidates[game_state['action_refs'][:,0], game_state['action_refs'][:,2]] = policy_state[game_state['action_refs'][:,0], game_state['action_refs'][:,1]]
        candidates[game_state['action_decode_indices'][:,0], game_state['action_decode_indices'][:,1]] = decisions[game_state['action_decode_indices'][:,0], game_state['action_decode_indices'][:,1]]
        candidates_mask[game_state['action_refs'][:,0], game_state['action_refs'][:,2]] = True
        candidates_mask[game_state['action_decode_indices'][:,0], game_state['action_decode_indices'][:,1]] = True
        return self.pointer_network(candidates, candidates_mask, prompt_tokens)