from magicnet.dump import comp, load, resolve_edges
from magicnet.network.data_loader import DataLoader
from magicnet.catalog import CatalogClient
from magicnet.network.magicnet import MagicNet
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from magicnet.network.constants import EXTERNAL_EMBEDDING_PCA_DIM

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision("high")
#torch.autocast("cuda", dtype=torch.bfloat16)

#from torch.nn.attention import sdpa_kernel, SDPBackend

BATCH_SIZE = 20

decision1 = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 50))
decision2 = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 10))
decision3 = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 279))
decision4 = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 270))

games = [
    {
        'state': decision.state,
        'history': decision.history,
    }
    for decision in [decision3 for _ in range(BATCH_SIZE)]
]
loader = DataLoader(CatalogClient())
policy = MagicNet()
policy.eval()
inference_time = 0
load_time = 0
with torch.inference_mode():
    for _ in range(1):
        game_state = loader.load_game(games)
        _ = policy(game_state)
    torch.cuda.synchronize()
    for _ in range(5):
        start = time.perf_counter()
        game_state = loader.load_game(games)
        torch.cuda.synchronize()
        load_time += time.perf_counter() - start
        start = time.perf_counter()
        _ = policy(game_state)
        torch.cuda.synchronize()
        inference_time += time.perf_counter() - start
    print (f'load time: {load_time}')
    print(f'inference time: {inference_time}')
    print(f'amortized inference time: {(inference_time + load_time) / (5*BATCH_SIZE)}')

with torch.inference_mode():
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,     # Saves information about operator input shapes
        profile_memory=True,    # Tracks tensor memory allocation/deallocation
        with_stack=True         # Records source file and line numbers
    ) as prof:
        with record_function("model_inference"):
            for _ in range(5):
                game_state = loader.load_game(games)
                torch.cuda.synchronize()
                _ = policy(game_state)
                torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))