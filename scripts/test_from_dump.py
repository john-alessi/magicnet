from magicnet.dump import comp, load, resolve_edges
from magicnet.network.data_loader import load_game
from magicnet.catalog import CatalogClient
from magicnet.network.magicnet import MagicNet
import torch
from torch.profiler import profile, record_function, ProfilerActivity

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

decision1 = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 50))
decision2 = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 10))
decision3 = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 279))

catalog = CatalogClient()
game = [
    {
        'state': decision.state,
        'history': decision.history,
    }
    for decision in [decision1, decision2]
]
game_state = load_game(game, catalog)
magicnet = MagicNet()
magicnet.eval()

with torch.inference_mode():
    for _ in range(5):
        _ = magicnet(game_state)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,     # Saves information about operator input shapes
        profile_memory=True,    # Tracks tensor memory allocation/deallocation
        with_stack=True         # Records source file and line numbers
    ) as prof:
        with record_function("model_inference"):
            for _ in range(20):
                _ = magicnet(game_state)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))