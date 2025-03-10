import torch
import torch.nn as nn
import numpy as np

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost
        
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, inputs.shape[-1])
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding.weight.shape[0])
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices
    
class AdaptiveTokenizer:
    def __init__(self, n_bins=256, fit_data=None):
        self.n_bins = n_bins
        self.bin_edges = None
        if fit_data is not None:
            self.fit(fit_data)
    
    def fit(self, data):
        """Learn bin edges from data to maximize information content"""
        # Use quantile-based binning for more uniform distribution
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        self.bin_edges = np.quantile(data.flatten(), quantiles)
        return self
    
    def encode(self, data):
        """Convert continuous values to token indices"""
        if self.bin_edges is None:
            raise ValueError("Tokenizer needs to be fitted first")
        
        # Digitize data into bins
        tokens = np.digitize(data, self.bin_edges) - 1
        # Clip to valid range
        tokens = np.clip(tokens, 0, self.n_bins - 1)
        return tokens
    
    def decode(self, tokens, mode='center'):
        """Convert token indices back to continuous values"""
        if mode == 'center':
            # Return bin centers
            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            return bin_centers[tokens]
        elif mode == 'sample':
            # Sample uniformly from within each bin
            bin_widths = np.diff(self.bin_edges)
            offsets = np.random.random(tokens.shape) * bin_widths[tokens]
            return self.bin_edges[tokens] + offsets

class SBITokenizer:
    def __init__(self, nodes_size, n_bins=256, 
                 use_vq=True, vq_dim=128, vq_codebook_size=1024):
        
        self.nodes_size = nodes_size

        self.tokenizer = AdaptiveTokenizer(n_bins=n_bins)
        
        self.use_vq = use_vq
        if use_vq:
            self.vq_encoder = nn.Sequential(
                nn.Linear(nodes_size, vq_dim * 2),
                nn.ReLU(),
                nn.Linear(vq_dim * 2, vq_dim)
            )
            self.vq_embedding = VQEmbedding(vq_codebook_size, vq_dim)
        
    def fit(self, joint_data):
        """Fit tokenizers to simulation data"""
        self.tokenizer.fit(joint_data)
        return self
    
    def encode(self, joint_data):
        """Convert parameters to tokens"""
        if self.use_vq:
            # Use learned VQ for parameters (better for capturing complex distributions)
            encoded = self.vq_encoder(joint_data)
            quantized, _, indices = self.vq_embedding(encoded.unsqueeze(2).unsqueeze(3))
            return indices.squeeze()
        else:
            # Use adaptive binning
            return torch.tensor(self.tokenizer.encode(joint_data.cpu().numpy()))
    
    def decode_parameters(self, tokens):
        """Convert parameter tokens back to continuous values"""
        if self.use_vq:
            # Reconstruct from VQ codebook
            quantized = self.vq_embedding.embedding(tokens)
            # Decode through a projection (would need an additional decoder network)
            return quantized
        else:
            return torch.tensor(self.tokenizer.decode(tokens.cpu().numpy()))