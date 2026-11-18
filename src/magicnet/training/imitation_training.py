import torch
import numpy as np
import json
import random
import os
import platform
import time
import math
import mlflow
import itertools
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils._pytree import tree_map
from torch._dynamo.utils import counters, compile_times

from ..dump import load_many
from ..network.data_loader import DataLoader
from ..network.magicnet import MagicNet
from ..catalog.client import CatalogClient
from ..network.constants import *
from ..viz import CardMap, log_belief

BATCH_SIZE = 16
BELIEF_EPOCHS = 1
POLICY_EPOCHS = 0
NLL_LOSS_DECAY = 0.01

if os.environ.get('NSYS_PROFILING'):
    platform.architecture = lambda *a, **k: ('64bit', 'ELF')

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')
#torch.cuda.set_sync_debug_mode('warn')
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('magicnet-behavioral-cloning')

class DumpDataset(IterableDataset):
    def __init__(self, shard_files, shuffle_buffer=20000, seed=42):
        self.files, self.buf, self.seed, = sorted(shard_files), shuffle_buffer, seed

    def __iter__(self):
        info = get_worker_info()
        files = self.files if info is None else self.files[info.id::info.num_workers]
        rng = random.Random(self.seed + (info.id if info else 0))
        rng.shuffle(files)
        buf = []
        for d in load_many(files):
            if d.chosen < 0:
                continue
            buf.append(d)
            if len(buf) >= self.buf:
                yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf

def count_decisions(files):
    by_dir = {}
    for f in files:
        by_dir.setdefault(f.parent, set()).add(int(f.name.split('_')[1].split('.')[0]))
    n = 0
    for d, shards in by_dir.items():
        for line in open(d / 'manifest.jsonl'):
            r = json.loads(line)
            if r['shard'] in shards:
                n += r['records']
    return n

class Collate:
    def __init__(self, dequantize=True):
        self.loader = None
        self.dequantize = dequantize
    def __call__(self, ds):
        if self.loader is None:
            self.loader = DataLoader(CatalogClient(), per_batch_lookups=True)
        batch = self.loader.load_game([
            {
                'state': d.state + [d.prompt],
                'history': d.history,
                'actions': d.actions,
            }
            for d in ds
        ])
        target_ids, target_regions, game_lengths, counts = map(np.array, zip(*[
            (c['oracle_id'], hidden_zone['region'], len(d.history), hidden_zone['total'])
            for d in ds
            for hidden_zone in d.hidden_truth
            for c in hidden_zone['cards']
            for _ in range(c['count'])
        ]))
        target_region_set = set(target_regions)
        zone_counts = torch.tensor(counts, device='cpu')
        loss_weights = torch.tensor(game_lengths, dtype=torch.float32, device='cpu')
        loss_weights[target_regions != 'r:opp_hand'] = 1/(len(target_region_set)*(NLL_LOSS_DECAY*loss_weights[target_regions != 'r:opp_hand'] + 1))
        loss_weights[target_regions == 'r:opp_hand'] = 1 - 2/(len(target_region_set)*(NLL_LOSS_DECAY*loss_weights[target_regions == 'r:opp_hand'] + 1))
        loss_weights = loss_weights / zone_counts
        belief_targets = torch.tensor(catalog.card_targets(target_ids, dims=32), device='cpu')
        if self.dequantize:
            belief_targets = belief_targets + DQ_FIXED_NOISE_GAIN*torch.randn_like(belief_targets)
        policy_targets = targets = torch.tensor([d.chosen for d in ds], device='cpu')
        return batch, belief_targets, loss_weights, policy_targets

def dq_schedule(step):
    return 0.5*(DQ_NOISE_SCHEDULE_CEILING-DQ_NOISE_SCHEDULE_FLOOR)*(math.tanh(DQ_NOISE_SCHEDULE_DROP_FACTOR*(DQ_NOISE_SCHEDULE_INFLECTION_POINT-step)) + 1) + DQ_NOISE_SCHEDULE_FLOOR
    #return DQ_NOISE_SCHEDULE_FLOOR + ((DQ_NOISE_SCHEDULE_CEILING - DQ_NOISE_SCHEDULE_FLOOR) / (1 + math.exp(DQ_NOISE_SCHEDULE_DROP_FACTOR*(step - DQ_NOISE_SCHEDULE_INFLECTION_POINT))))

files = sorted(Path('/tmp/magicnet').glob('*/shard_*.jsonl.gz'))
val, train = files[:1], files[1:]
ds = DumpDataset(train)
val_ds = DumpDataset(val)
catalog = CatalogClient()
card_map = CardMap.load_or_fit(catalog)
loader = torch.utils.data.DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    num_workers=6,
    collate_fn=Collate(dequantize=False),
    pin_memory=True,
    prefetch_factor=4,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=0,
    collate_fn=Collate(dequantize=False),
)

val_batches = list(itertools.islice(iter(val_loader), 20))
viz_decisions = list(itertools.islice(load_many(val), 200))[::50]
viz_collate = Collate(dequantize=False)
viz_items = [(d, viz_collate([d])[0]) for d in viz_decisions]
pca_basis = catalog.pca(dims=32)
model = MagicNet(pca_basis, persist_embedding_cache=False).to(device)
model.compile(dynamic=True)
model.belief_network.compile(dynamic=True)
optimizer = torch.optim.AdamW(model.belief_network.parameters(), fused=True, lr=3e-4)
steps = count_decisions(train) // BATCH_SIZE

step = 0
with mlflow.start_run():
    for epoch in range(BELIEF_EPOCHS):
        print(f'EPOCH {epoch}')
        model.train()
        bar = tqdm(loader, total=steps, desc=f'epoch {epoch}', dynamic_ncols=True)
        for batch, targets, loss_weights, _ in bar:
            # if step == 50:
            #     torch.cuda.profiler.start()
            # if step > 100:
            #     torch.cuda.profiler.stop()
            # if step % 100 == 0:
            #     tqdm.write(str({k: dict(v) for k, v in counters.items()})); tqdm.write(compile_times())
            dq_noise = dq_schedule(step)
            loss_weights = loss_weights.to(device)
            targets = targets + dq_noise*torch.randn_like(targets)
            targets = targets.to(device)
            batch = tree_map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, batch)

            optimizer.zero_grad()
            likelihoods,_ = model.belief_network(batch, targets=targets)

            loss = -(likelihoods*loss_weights).sum()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.belief_network.parameters(), 1.0)
            bar.set_postfix(loss=f'{loss.item():.3f}')
            optimizer.step()

            if step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_batch, val_targets, val_loss_weights, _ in val_batches:
                        val_loss_weights = val_loss_weights.to(device)
                        val_targets = val_targets + dq_noise*torch.randn_like(val_targets)
                        val_targets = val_targets.to(device)
                        val_batch = tree_map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, val_batch)
                        val_likelihoods,_ = model.belief_network(val_batch, targets=val_targets)
                        val_loss += -(val_likelihoods*val_loss_weights).sum()
                    val_loss /= len(val_batches)
                mlflow.log_metric("belief_train_loss", loss, step=step)
                mlflow.log_metric("belief_val_loss", val_loss, step=step)
                mlflow.log_metric("dq_noise_gain", dq_noise, step=step)
                model.train()

            if step % 5000 == 0:
                with torch.no_grad():
                    for d, batch in viz_items:
                        batch = tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch)
                        _,samples = model.belief_network(batch, n_samples=512)
                        log_belief(d, samples, card_map, step)
            step += 1

    model.belief_network.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], fused=True, lr=3e-4) 
    for epoch in range(POLICY_EPOCHS):
        print(f'EPOCH {epoch}')
        model.train()
        model.belief_network.eval()
        bar = tqdm(loader, total=steps, desc=f'epoch {epoch}', dynamic_ncols=True)
        for batch, _, _, targets in bar:
            targets = targets.to(device)
            batch = tree_map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, batch)

            optimizer.zero_grad()
            _,choice_logits = model(batch)
            loss = torch.nn.functional.cross_entropy(choice_logits, targets, ignore_index=-1)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bar.set_postfix(loss=f'{loss.item():.3f}')
            optimizer.step()

            if step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_batch, _, _, val_targets in val_batches:
                        val_targets = val_targets.to(device)
                        val_batch = tree_map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, val_batch)
                        _, val_choice_logits = model(val_batch)
                        val_loss += torch.nn.functional.cross_entropy(val_choice_logits, val_targets, ignore_index=-1)
                    val_loss /= len(val_batches)
                mlflow.log_metric("policy_train_loss", loss, step=step)
                mlflow.log_metric("policy_val_loss", val_loss, step=step)
                model.train()
                model.belief_network.eval()

            step += 1