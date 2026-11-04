from magicnet.dump import comp, load, resolve_edges
from magicnet.network.data_loader import load_game
from magicnet.catalog import CatalogClient
from magicnet.network.magicnet import MagicNet
import torch
from torch.profiler import profile, record_function, ProfilerActivity

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
decision = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 279))
board = [s for s in decision.state if 'Creature' in comp(s, 'types', '')]
catalog = CatalogClient()
game_state = load_game(decision.state, decision.history, catalog)
magicnet = MagicNet()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,     # Saves information about operator input shapes
    profile_memory=True,    # Tracks tensor memory allocation/deallocation
    with_stack=True         # Records source file and line numbers
) as prof:
    with record_function("model_inference"):
        decision = magicnet(game_state)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))