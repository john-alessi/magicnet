import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info

from ..dump import load_many
from ..network.data_loader import DataLoader
from ..network.magicnet import MagicNet
from ..catalog.client import CatalogClient
from ..network.constants import EXTERNAL_EMBEDDING_PCA_DIM, DEQUANTIZATION_NOISE_GAIN
from ..viz import CardMap

BATCH_SIZE = 16
BELIEF_EPOCHS = 1
POLICY_EPOCHS = 1

torch.manual_seed(42)
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision("high")

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

files = sorted(Path('/tmp/magicnet').glob('*/shard_*.jsonl.gz'))
val, train = files[:1], files[1:]
ds = DumpDataset(train)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, collate_fn=lambda ds_: ds_)
catalog = CatalogClient()
card_map = CardMap.load_or_fit(catalog)
inf_data_loader = DataLoader(catalog)
model = MagicNet()
optimizer = torch.optim.AdamW(model.belief_network.parameters(), fused=True)

for epoch in range(BELIEF_EPOCHS):
    print(f'EPOCH {epoch}')
    model.train()
    for decisions in loader:
        batch = inf_data_loader.load_game([
            {
                'state': d.state + [d.prompt],
                'history': d.history,
                'actions': d.actions,
            }
            for d in decisions
        ])
        target_ids, target_regions = map(np.array, zip(*[
            (c['oracle_id'], hidden_zone['region'])
            for d in decisions
            for hidden_zone in d.hidden_truth
            for c in hidden_zone['cards']
            for _ in range(c['count'])
        ]))
        targets = torch.tensor(catalog.card_targets(target_ids, dims=32))
        targets = targets + DEQUANTIZATION_NOISE_GAIN*torch.randn_like(targets)

        optimizer.zero_grad()
        likelihoods,_ = model.belief_network(batch, targets=targets)
        loss = -sum([
            torch.mean(likelihoods[target_regions == r])
            for r in set(target_regions)
        ])
        loss.backward()
        print(loss.item())
        optimizer.step()

model.belief_network.requires_grad_(False)
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad) 
for epoch in range(POLICY_EPOCHS):
    print(f'EPOCH {epoch}')
    model.train()
    model.belief_network.eval()
    for decisions in loader:
        batch = inf_data_loader.load_game([
            {
                'state': d.state + [d.prompt],
                'history': d.history,
                'actions': d.actions,
            }
            for d in decisions
        ])
        targets = torch.tensor([d.chosen for d in decisions])

        optimizer.zero_grad()
        _,choice_logits = model(batch)
        loss = torch.nn.functional.cross_entropy(choice_logits, targets, ignore_index=-1)
        
        loss.backward()
        print(loss.item())
        optimizer.step()