# Hidden-Card Head
## Normalizing Flow
`magicnet.network.distributions.HiddenCardPosterior` uses the `zuko` package to fit a posterior distribution of each hidden card given the entire game context using a normalizing flow.  For hidden-card tokens representing zones with more than one card, the probability distribution represents P(a random card revealed from this zone | game state and history).
### Sample Space
Samples from the flow are in oracle-text-embedding space, projected down to size `magicnet.network.constants.EXTERNAL_EMBEDDING_PCA_DIM` with a frozen PCA basis.

### Why not discrete card probabilities?
In my opinion there are a few problems with sampling cards as discrete probabilities:
#### Constantly growing card pool
With new sets being released as little as 6 weeks apart, it's hard enough for me to keep up with the game without also worrying about train-serve skew.  A major goal of this work is to produce a policy that generalizes to new card pools without requiring a full retrain.
#### OOD but functionally-similar cards
Representing the distribution of possible cards as a set of discrete probabilities treats each card's probability as independent which results in poorly-calibrated posterior probabilities of unseen or underrepresented cards.

As an example, if a player became really skilled at the game playing only the [Premodern](https://mtg.wiki/page/Premodern) format without ever seeing newer cards, they would learn that if you're playing against black midrange decks like [The Rock](https://www.mtggoldfish.com/archetype/premodern-the-rock-be743511-7152-40e0-9153-c8a943093280#paper) or [Moneyball Black](https://www.mtggoldfish.com/archetype/premodern-mono-black-midrange-ba3cf91e-df0c-47d4-8068-bbfc327bd0dc#paper) you should expect to see a lot of [Duress](https://scryfall.com/card/fdn/606/duress) and [Cabal Ritual](https://scryfall.com/card/vma/106/cabal-ritual).  If that same player then tries out the [Pioneer](https://mtg.wiki/page/Pioneer) format and finds themselves against [Golgari Midrange](https://mtg.wiki/page/Pioneer), they should immediately know to expect a similar card even if they've never heard of [Thoughtseize](https://scryfall.com/card/2xm/109/thoughtseize).

With proper target dequantization a normalizing flow should be able to sidestep these issues by assigning probability mass to a general region of cards in embedding space as opposed to trying to decode specific cards.

## Auxillary loss
This is the terminal module of the belief network which is trained with negative-log-likelihood loss, computed as a sum of likelihoods of all ground-truth cards in each zone weighted inversely by zone size.
### Game-Length-Based Zone Weighting
Additional weighting of each zone depending on the length of the game is also being explored.  The rationale for this is that competetive formats usually have a pretty tractable metagame regarding the amount of decklists a player would need to know about in advance to play their best, meaning that in a typical game of Magic, guessing the opponent's deck is really only important in the first few turns and becomes trivial in the late game, whereas being able to deduce what's in the opponent's hand is relevant through all phases of the game.  The exact weighting strategy is still in flux.
### Target Dequantization
Currently targets are dequantized with fixed gaussian noise; I'm also in the middle of experimenting with noise schedules.

## Sampling
This work will explore the use of samples drawn from this module in two different ways:

### Test-time search over fully-sampled game states
This approach, which was instrumental to [Ataraxos](https://arxiv.org/abs/2511.07312) beating the Stratego world champion by an unprecedented margin of 15-1-4, involves jointly sampling full game states and using the policy-value network to search over many trajectories starting from those game states to find the optimal move.  Whether or not this scales to 60-card decks is an open question.  Sokota et. al. also use a heavily optimized CUDA rules engine where this work's environment of choice is a Java application, so searching over a sufficient number of trajectories might require evaluating alternate rules engines.

### Sampled latent tokens as an input to the policy network
An alternate approach is to use the sampled latent tokens directly as an input to the policy network rather than snapping to real cards.  A placeholder implementation of this currently treats samples as additional components of the hidden-card tokens in the policy network, but whether or not this actually improves downstream performance of the policy has yet to be measured.