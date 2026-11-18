# Game State Encoder

## State Representation

### Token-Level Primitives

Tokens represent an open vocabulary of 3 main primitives:
* Game objects: cards (so 1 token = an entire card), tokens ([in the magic sense, not the transformer sense](https://mtg.fandom.com/wiki/Token)), copied spells on the [Stack](https://mtg.fandom.com/wiki/Stack), and players.
* Events: past game actions, for example "player A cast [Cast Down](https://scryfall.com/card/clb/119/cast-down) targeting player B's [Tolarian Terror](https://scryfall.com/card/fdn/167/tolarian-terror) is represented as a single token (more on that composition below).
* Actions: when the user is prompted to choose a game action, options are each represented as a token.

#### Special Tokens
The encoded game state also contains two special tokens which are not game objects themselves but carry summary information about the game state:
- Global token: contains turn and special designations like [day/night](https://mtg.fandom.com/wiki/Day_and_night).
- Prompt token: a learned embedding of the decision being asked of the agent (see `magicnet.schema.DECISION_KINDS`).  In the policy network this token pools information about the full game state and is used as the query in the pointer network decoder head.

#### Hidden Cards
Hidden cards are encoded based only on known information (usually just zone but face-down creatures can have additional visible attributes like power/toughness or [abilities like Ward](https://scryfall.com/card/mkm/50/cryptic-coat)) but are otherwise treated the same as visible cards in the attention layers.  Hidden-card tokens typically represent the entire zone (for example there's one token representing the opponent's entire hand) except in cases like face-down creatures or [Foretold](https://mtg.fandom.com/wiki/Foretell) spells; in all cases these tokens pool information about the game state to condition a posterior distribution about those cards' identities.

### Bags-of-Components Representation
The encoder uses a [Set Transformer](https://arxiv.org/abs/1810.00825) architecture to represent each token as a pooled set of embedded key-value pairs.  For example, the component-level representation of [Counterspell](https://scryfall.com/card/dsc/114/counterspell) might look something like:

```
[{'oracle_text': 'Counter target spell.'}, {'mana_cost': '{U}{U}'}, {'type_line': 'Instant'}]
```

#### Component Kinds
Components fall into 5 kinds:
* Scalar
* Vector (pre-embedded string values)
* Mana Cost: due to the existence of [hybrid mana](https://mtg.fandom.com/wiki/Hybrid_mana), mana-costs need to have their own special encoding as a multi-set of colors.
* Color: the color of a card (mechanics like [Devoid](https://mtg.fandom.com/wiki/Devoid) mean that a card's color isn't strictly implied by its mana cost).
* Categorical: represents core game primitives with closed vocabularies like [Phase](https://mtg.fandom.com/wiki/Turn_structure) and [Zone](https://mtg.fandom.com/wiki/Zone)

### Relational Edges
Relations between tokens (such as "creature A is blocking creature B") are also represented with learned embeddings. Relational edges are used particularly heavily with the event tokens, for instance the earlier "Cast Down" example actually decomposes into a simple "cast" verb token with edges to the spell being cast and the target of the spell. For the full taxonomy of relation types, see `magicnet.schema.RELATIONS`.

## Architecture

### Component Encoders
Components themselves are composed of 3 vectors representing the token's key, value, and kind which are combined into a single vector of size `magicnet.network.constants.COMPONENT_DIM` with a `magicnet.network.encoders.ObjectComponentEncoder`.  Kind vectors come from a learned embedding whereas key vectors are encoded differently depending on kind in `magicnetwork.encoders`:
* Keys and string values are represented by pre-computed natural-language embeddings fed into a `RawEmbeddingEncoder`.
* Scalar values are projected up to size `COMPONENT_DIM` with a `ScalarEncoder`.
* Color values are represented as a one-hot vector projected up to size `COMPONENT_DIM` with a `ColorEncoder`.
* Categorical values are a learned embedding lookup in `CategoricalEncoder`.
* Mana cost values are encoded as a set of individual mana symbols, which are one-hot vectors representing payment options, pooled into a single component vector with a set transformer in ManaCostEncoder.

### Component Set Encoder
Components are pooled into their corresponding object tokens of size `magicnet.network.constants.TOKEN_DIM` in `magicnet.network.encoders.ComponentSetEncoder`

#### Set Attention
Component and mana symbol pooling is done with the set transformer module defined in `magicnet.network.attention.PooledAttention`.  This attention variant uses a single learned query over a nonlinear value projection of a set of tokens to produce a single pooled action/event/object token per set.

### Relational Encoder
Relational edges are encoded with a learned embedding table; duplicate edges of the same type with the same source and destination are combined into a single embedding weighted proportional to the log of the token count.

### Relational Self-Attention
`magicnet.network.attention.RelationalAttention` uses [Relation-Aware Self-Attention](https://arxiv.org/abs/1803.02155) to allow tokens to attend to each other with respect to the relationships encoded in their edges, if any exist.  This currently uses Shaw et. al's efficient implementation which only uses edges to influence attention weights; future work will explore incorporating a value term as well.

# Decision Encoder
The decision encoder defined in `magicnet.network.encoders.DecisionEncoder` works similarly to the game state encoder above except instead of state and history it encodes the set of action choices given to the policy and cross-attends to context vectors produced by a game-state encoder with `magicnet.network.attention.RelationalCrossAttention`.  The prompt token and any tokens representing valid choices are passed on to the policy/value head for action decoding.