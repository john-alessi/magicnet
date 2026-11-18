# MagicNet High-Level Design

This and the accompanying docs are a brief overview of the model design, I'll try to add more details and some [visuals](https://www.youtube.com/shorts/Lr-7FN4dQ6o) at a later point.

## Belief Network
This module is defined in `magicnet.network.magicnet.BeliefNetwork` and is trained separately from the main policy network to learn a distribution of hidden cards that can be sampled to condition the policy network at test time.

The belief network is composed of a `GameStateEncoder` (see `encoders.md` for design details) and a `HiddenCardPosterior` (see `hidden_card_head.md`).

## Policy-Value Network
The main policy-value network is defined in `magicnet.network.magicnet.MagicNet` and uses a `GameStateEncoder`, and a `DecisionEncoder` (see `encoders.md`) to form the main policy trunk.  Context vectors from that trunk are fed into a `ValueHead` and `PolicyHead` (see `policy_value_head.md`) to produce a final decision and win/loss/draw probabilities.