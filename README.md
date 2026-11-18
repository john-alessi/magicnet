# MagicNet

MagicNet is my attempt at learning a policy for Magic: the Gathering, an imperfect-information game with a large combinatorial action space.  Initial experiments are scoped to the [Premodern](https://www.mtggoldfish.com/metagame/premodern#paper) format but the model is designed to ~~keep up with Hasbro's ridiculous release schedule~~ generalize to out-of-distribution card pools.

`/forge-bridge/` contains middleware for extracting and serializing game state from [Forge](https://github.com/Card-Forge/forge), the environment of choice.  This module is entirely written by Claude, so run it at your own risk.

The core transformer code can be found under `/src/magicnet/network/`.  This module is entirely written by me, so run it at your own risk.

`/src/magicnet/catalog` contains a catalog service used to retrieve and cache card text embeddings and scryfall dumps; you'll need to start this first in order to run any of the training scripts.

Design docs, experiment logs, and relevant literature can be found under `/docs/`.  I'm doing all of this as a learning excercise so if you spot any methodological errors you're encouraged to (respectfully) grill me in the discussion tab.

Currently I'm experimenting with behavioral cloning before moving on to self-play RL.  If you want to replicate any of my experiments you can use `/scripts/generate_dump.py` to generate the exact gameplay trajectories used.