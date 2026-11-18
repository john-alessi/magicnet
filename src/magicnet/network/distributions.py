import torch
import zuko
from .constants import *

class HiddenCardPosterior(torch.nn.Module):
    def __init__(self, pca_basis):
        super().__init__()
        self.ln = torch.nn.LayerNorm(TOKEN_DIM)
        self.flow = zuko.flows.NSF(
            features = EXTERNAL_EMBEDDING_PCA_DIM,
            context=TOKEN_DIM,
            transforms=N_FLOW_TRANSFORMS,
            passes=2,
            hidden_features=(FLOW_DIM,FLOW_DIM)
        )
        self.register_buffer('pca_mean', torch.as_tensor(pca_basis.mean))
        self.register_buffer('pca_components', torch.as_tensor(pca_basis.components * pca_basis.std[:, None]))
    
    def forward(self, batches, unknown_card_indices, targets=None, n_samples=0):
        samples = None
        likelihood = None
        unknown_cards = self.ln(batches[unknown_card_indices[:,0], unknown_card_indices[:,1]])
        distributions = self.flow(unknown_cards)

        if targets is not None:
            likelihood = distributions.log_prob(targets)

        if n_samples > 0:
            samples = distributions.rsample((n_samples,))
            samples = samples @ self.pca_components + self.pca_mean
            
        return likelihood, samples