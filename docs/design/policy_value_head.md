## Policy Head
The policy head is fed the prompt token as well as any valid choices from the policy trunk.  Depending on the decision those choices may be actions from `magicnet.network.encoders.DecisionEncoder` and/or game objects from the main policy trunk.  The final choice is decoded using a [Pointer Network](https://arxiv.org/abs/1803.08475) which uses the prompt token as a query over all of the tokens representing valid choices and uses attention scores directly as the output logits.

## Value Head
The value head is a simple set of feedforward layers fed by the global state token from the policy trunk which produces a set of (win, loss, draw) probabilities.