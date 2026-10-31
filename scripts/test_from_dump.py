from magicnet.dump import comp, load
from magicnet.network.data_loader import load_data
from magicnet.catalog import CatalogClient
from magicnet.network.magicnet import MagicNet
decision = next(x for x in load("/home/john/Projects/magicnet/forge-bridge/fixtures/dumps/pio_mrp.jsonl") if (x.game, x.decision) == (3, 279))
board = [s for s in decision.state if 'Creature' in comp(s, 'types', '')]
catalog = CatalogClient()
game_state = load_data(decision.state, decision.history, catalog)
magicnet = MagicNet()
decision = magicnet(game_state)
breakpoint()
