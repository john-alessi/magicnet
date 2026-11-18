# MagicNet [DEPRECATED]
* this design doc is outdated, I'm just keeping it around for historical reasons.

## Game Representation

### Visible Game State

#### Game objects
- each game object represented by a single token with a learned embedding (players and emblems are considered as game objects)
- card's zone encoded with an MLP projection (attention between cards should be zone-dependent)
- game state like format, day/night, storm count, etc. compressed into summary token via MLP
- timestamp probably needs to be encoded, maybe just treat battlefield as an ordered set

#### Object zones
- learned transform for each zone, positional encoding for ordered zones
- ordered: graveyards, stack
- unordered: battlefield, hand, command zone, stack, exile

#### Object Modifiers
- modifiers include tokens, counters, temporary states like `goaded`
- each modifier represented by a learned embedding
- multiple modifiers are summed and then combined with the base game object with an MLP

#### Graph edges
- targeting / targeted by, exiling, etc

#### Game Actions
- each past game action represented as sequence of tokens (something like `{player a}{cast}{fatal push}{target}{spectral sailor}`)
- current legal actions each represented by first component token of action (or next unselected token if an action is already partially selected)
- action types represented by learned embeddings
- single ordered sequence with positional encoding

### Hidden Game State

#### Hidden-card token
- represents a collection of hidden cards, specifically the probability distribution of what you'd see if you revealed one card from the collection
- continuous distribution over card embedding space as opposed to collection of per-card marginal probabilities
- conditioned on pre-game prior, current board state, and game history

#### The opponent's library
- simplest implementation is single token representing all unobserved cards in the library
- alternately, the library could be represented as a collection of tokens with positional encoding to allow for belief updates based on scry/surveil-type actions taken by the opponent

#### The opponent's hand
- single token representing all unobserved cards

#### The opponent's sideboard
- out-of-scope for V1 which will stick to best-of-1
- technically part of the active game's state since cards like `North Wind Avatar` can tutor a card from the sideboard without revealing it
- single token representing all unobserved cards

#### Face-down cards
- each represented by their own token
- [TBD] how to represent card type and base stats when applicable
- modifiers applied same way as they're applied to visible game objects

## Major Architecture Components

### Prior hidden card distribution
- smooth distribution over card embedding space computed from tournament data for the format being played, weighted by recency
- inputs: raw decklists pulled from MTGGoldfish or a similar source
- output: the fitted distribution
- options: spline flow, gaussian mixture, KDE

### Embedding layer
- option 1: standard lookup, learned entirely through imitation learning and self-play
  - drawback: individual embeddings only get training signal when seen in games
- option 2: learned projection of out-of-box NL embeddings
  - benefits: natively supports OOD cards, parameter updates nudge all cards rather than just ones seen in training episodes
  - drawback: dependency on external model, either serve-time API dependency or additional deployment cost of open-weight embedding model
- inputs: tokenized game history and board state
- outputs: embedded tokens representing game history and board state

### Hidden state module
- inputs: embedded game state + history, prior deck distribution
- outputs: sampled context vectors representing potential hidden cards

#### Computation of posterior hidden-card tokens
- context vectors produced from attention between game objects / history used to fit prior-invariant NSF which takes any supplied prior as a base function
- NSF scored on NLL loss against ground-truth decklist (stop-grad or EMA needed to prevent representation collapse)
- posterior learns general regions of likely cards, prior conditions that on format data after the fact
- smoothing applied to prior while posterior is free to spike on specific cards that are extremely likely based on available info

#### Interpretability
- dump probability heatmap over 2d-projected card scatterplot
- sample likely decklists from spline flow

### Standard transformer module
- inputs: embedded game state + history, hidden-card tokens
- outputs: context vectors representing game objects and summary token

### Action Decoder
- autoregressively generates sequence of tokens to represent the selected action
- inputs: game object + summary context vectors, tokens representing legal action component choices, currently selected action component tokens
- legality masking when selecting targets
- outputs: action component tokens

### Value Function
- critic head on top of main model
- privileged access to ground-truth hidden state (except for future information)

## Training Regimen

### Initialization
- game object embeddings will be initialized to a learned projection of their NLP embeddings to shortcut learning simple relationships like functional similarity
- current candidate NLP embedding model is `openai/text-embedding-3-large`

### Training Environment
- [Forge](https://github.com/Card-Forge/forge) is a full rules engine with a bot api that will be used for playing games
- Also has non-ML bots which serve as a reasonably competent baseline for the model to learn from

### Imitation learning from bot-vs-bot games
- supervised learning from replay data generated from many bot-vs-bot games

### Self-play RL
- separate trajectories per player

## Datasets
- only external data needed is the selection of decklists used in imitation learning / self-play, can be sampled from mtggoldfish or other deck aggregators

## Evaluations
#### What to measure
- winrate against baseline forge bot
- performance in puzzle benchmarks
- exploitability by trained adversary
- performance against human opponents in hosted xmage server
- how quickly belief tokens converge towards ground-truth deck
- value function calibration against win probability