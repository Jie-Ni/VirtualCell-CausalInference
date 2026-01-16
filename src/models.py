import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TurboGNN(nn.Module):
    """
    Graph Neural Network for transcriptomic representation learning with
    structural priors (e.g., PPI/GO networks).
    """

    def __init__(self, num_genes, edge_index, hidden_dim=128):
        super(TurboGNN, self).__init__()
        self.edge_index = edge_index
        # Learnable gene embeddings
        self.emb = nn.Embedding(num_genes, hidden_dim)
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv2 = GCNConv(hidden_dim * 2, hidden_dim)
        # Output prediction head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, perturbation_mask=None):
        """
        Forward pass with optional perturbation masking.

        Args:
            perturbation_mask (torch.Tensor, optional): Binary mask for gene knockout simulation.
                                                      0 indicates knockout, 1 indicates wildtype.
        """
        device = self.emb.weight.device
        all_genes = torch.arange(self.emb.num_embeddings, device=device)
        x = self.emb(all_genes)

        # Apply latent intervention if mask is provided
        if perturbation_mask is not None:
            if perturbation_mask.dtype == torch.bool:
                x = x.clone()
                x[perturbation_mask] = 0.0
            else:
                x = x * perturbation_mask.unsqueeze(1)

        # Message passing
        x = self.conv1(x, self.edge_index)
        x = F.gelu(x)
        x = self.conv2(x, self.edge_index)

        return self.head(x).squeeze()


class SimpleTransformer(nn.Module):
    """
    Sequence-based Transformer baseline for gene expression reconstruction
    without explicit graph structural constraints.
    """

    def __init__(self, num_genes, d_model=128, nhead=4, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(num_genes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, mask=None):
        """
        Forward pass treating gene expression as a sequence of tokens.
        """
        device = self.embedding.weight.device
        # Treat all genes as a single sequence
        x = self.embedding(torch.arange(self.embedding.num_embeddings, device=device)).unsqueeze(0)

        if mask is not None:
            x = x.clone()
            x[0, mask, :] = 0.0

        # Self-attention mechanism
        # Using try-except to handle different PyTorch versions/backends
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                x = self.transformer_encoder(x)
        except:
            x = self.transformer_encoder(x)

        return self.head(x).squeeze()