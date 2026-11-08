import torch
import zuko
from .constants import *

class HiddenCardPosterior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = zuko.flows.NSF(
            features = EXTERNAL_EMBEDDING_PCA_DIM,
            context=TOKEN_DIM,
            transforms=N_FLOW_TRANSFORMS,
            passes=2,
            hidden_features=(FLOW_DIM,FLOW_DIM)
        )
    
    def forward(self, batches, unknown_cards, pca, targets=None, n_samples=0):
        samples = None
        likelihood = None
        unknown_cards = batches[unknown_cards[:,0], unknown_cards[:,1]]
        distributions = self.flow(unknown_cards)

        if targets is not None:
            likelihood = distributions.log_prob(targets)

        if n_samples > 0:
            samples = pca.inverse(distributions.rsample((n_samples,)))
            
        return likelihood, samples