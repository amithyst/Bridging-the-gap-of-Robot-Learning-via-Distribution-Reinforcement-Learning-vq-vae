import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Quantization Modules (Enhanced)
# ==========================================

class VectorQuantizer(nn.Module):
    """ Standard / EMA VQ """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, use_ema=False, decay=0.99):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

        if self.use_ema:
            self.decay = decay
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', torch.Tensor(num_embeddings, embedding_dim))
            self.ema_w.data.normal_()

    def forward(self, inputs):
        # inputs: [Batch, Channel, Time] -> [Batch, Time, Channel]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        if self.training and self.use_ema:
            encodings_sum = encodings.sum(0)
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        if self.use_ema:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            loss = self.commitment_cost * e_latent_loss
        else:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity calculation
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity

class ResidualVQ(nn.Module):
    """ Residual VQ: 多层量化 """
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, **kwargs) 
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        quantized_out = 0
        residual = x
        total_loss = 0
        perplexities = []
        
        for layer in self.layers:
            loss, quantized, perplexity = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss += loss
            perplexities.append(perplexity)
            
        return total_loss, quantized_out, torch.mean(torch.stack(perplexities))

class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) - SOTA 2024
    无需显式 Codebook，将 latent 投影到低维标量并取整。
    Ref: "Finite Scalar Quantization: VQ-VAE Made Simple"
    """
    def __init__(self, levels, input_dim, hidden_dim):
        super().__init__()
        # levels: list of integers, e.g. [8, 5, 5, 5] means codebook size = 8*5*5*5 = 1000
        self.levels = levels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # FSQ 需要特定的维度，我们用线性层映射过去再映射回来
        self.fsq_dim = len(levels)
        self.project_in = nn.Conv1d(input_dim, self.fsq_dim, 1)
        self.project_out = nn.Conv1d(self.fsq_dim, input_dim, 1)
        
        # Precompute levels for quantization
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.int32))
        self.register_buffer('_basis', torch.cumprod(torch.cat([torch.tensor([1]), torch.tensor(levels[:-1])]), dim=0, dtype=torch.int32))

    def forward(self, z):
        # z: [Batch, Channel, Time]
        original_input = z
        z = self.project_in(z) # -> [B, fsq_dim, T]
        z = z.permute(0, 2, 1) # -> [B, T, fsq_dim]
        
        # Bound inputs (tanh-like logic usually used, here simplified to sigmoid scaling or direct rounding)
        # Standard FSQ uses shift and scale to align with integers
        # We assume z is somewhat bounded. Better to use Tanh and scale.
        # FSQ paper suggests: round( (L-1) * (tanh(z) + 1) / 2 ) for range [0, L-1]
        # Or centered: round( z ).
        
        # Here we implement the centered version:
        # z should be normalized.
        
        quantized = torch.zeros_like(z)
        half_width = self._levels / 2
        
        # 1. Scaling to target range [-L/2, L/2] using Tanh
        # z_scaled = (self._levels - 1) * (torch.tanh(z) / 2 + 0.5) # [0, L-1] (optional)
        
        # Simpler implementation: Round directly.
        # We rely on the network to learn to output values near integers.
        # Stealing gradients is key.
        
        z_hard = self._round_ste(z)
        
        # Rescale back if needed or just project out
        z_out = z_hard.permute(0, 2, 1)
        z_out = self.project_out(z_out)
        
        # FSQ has no auxiliary loss (no commitment loss needed usually)
        loss = 0.0 
        
        # Calculate pseudo-perplexity
        # indices = (z_hard * self._basis).sum(dim=-1).long()
        # But for speed we skip exact PPL calc or approximate it
        perplexity = torch.tensor(1.0, device=z.device) 
        
        return loss, z_out, perplexity

    def _round_ste(self, z):
        z_hard = torch.round(z)
        return z + (z_hard - z).detach()

class LFQ(nn.Module):
    """
    Lookup-Free Quantization (LFQ)
    Binary quantization (Sign-based). Codebook size = 2^dim.
    """
    def __init__(self, input_dim, codebook_dim=10, entropy_loss_weight=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim # e.g., 10 -> 2^10 = 1024
        self.entropy_loss_weight = entropy_loss_weight
        
        self.project_in = nn.Conv1d(input_dim, codebook_dim, 1)
        self.project_out = nn.Conv1d(codebook_dim, input_dim, 1)

    def forward(self, z):
        # z: [B, C, T]
        z_e = self.project_in(z) # [B, code_dim, T]
        
        # Quantize: Sign(z)
        # 0 is mapped to 1 (or -1, arbitrary).
        z_q = torch.where(z_e > 0, torch.tensor(1.0, device=z.device), torch.tensor(-1.0, device=z.device))
        
        # STE
        z_q = z_e + (z_q - z_e).detach()
        
        # Entropy Loss (Encourage diversity, avoid all 1 or all -1)
        # Using sigmoid to approximate probability
        prob = torch.sigmoid(z_e) # [B, C, T]
        # Entropy = -p*log(p) - (1-p)*log(1-p)
        entropy = - (prob * torch.log(prob + 1e-6) + (1 - prob) * torch.log(1 - prob + 1e-6))
        loss = -entropy.mean() * self.entropy_loss_weight
        
        # Decode
        out = self.project_out(z_q)
        
        return loss, out, torch.tensor(2.0**self.codebook_dim, device=z.device) # Dummy PPL

# ==========================================
# 2. Building Blocks (Unchanged)
# ==========================================

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, 1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, 3, 1, 1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(1, 2))

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]

# ==========================================
# 3. Architectures (Fixed Transformer)
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, arch='simple'):
        super().__init__()
        self.arch = arch
        
        if arch == 'simple':
            self.model = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
            )
        elif arch == 'resnet':
            self.model = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                ResBlock1D(hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                ResBlock1D(hidden_dim),
                ResBlock1D(hidden_dim),
            )
        elif arch == 'transformer':
            self.downsample = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2)
            )
            self.pos_enc = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x):
        if self.arch == 'transformer':
            x = self.downsample(x) 
            x = self.pos_enc(x)
            x = x.permute(0, 2, 1) 
            x = self.transformer(x)
            x = x.permute(0, 2, 1) 
            return x
        else:
            return self.model(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, arch='simple'):
        super().__init__()
        self.arch = arch
        
        if arch == 'simple':
            self.model = nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(hidden_dim, input_dim, 4, 2, 1),
            )
        elif arch == 'resnet':
            self.model = nn.Sequential(
                ResBlock1D(hidden_dim),
                ResBlock1D(hidden_dim),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1), nn.LeakyReLU(0.2),
                ResBlock1D(hidden_dim),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv1d(hidden_dim, input_dim, 3, 1, 1),
            )
        elif arch == 'transformer':
            self.pos_enc = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(hidden_dim, input_dim, 4, 2, 1),
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x):
        if self.arch == 'transformer':
            x = self.pos_enc(x)
            x = x.permute(0, 2, 1)
            x = self.transformer(x)
            x = x.permute(0, 2, 1)
            x = self.upsample(x)
            return x
        else:
            return self.model(x)

# ==========================================
# 4. Main VQ-VAE Model
# ==========================================

class MotionVQVAE(nn.Module):
    def __init__(self, input_dim=29, hidden_dim=64, codebook_size=1024, 
                 arch='resnet', method='ema', n_layers=4):
        super(MotionVQVAE, self).__init__()
        
        # 1. Encoder
        self.encoder = Encoder(input_dim, hidden_dim, arch=arch)
        
        # 2. Quantizer Selector
        if method == 'standard':
            self.quantizer = VectorQuantizer(codebook_size, hidden_dim, use_ema=False)
        elif method == 'ema':
            self.quantizer = VectorQuantizer(codebook_size, hidden_dim, use_ema=True)
        elif method == 'rvq':
            self.quantizer = ResidualVQ(num_quantizers=n_layers, num_embeddings=codebook_size, 
                                        embedding_dim=hidden_dim, use_ema=True)
        elif method == 'fsq':
            # FSQ: levels=[8,5,5,5] -> 1000 codes (approx 1024)
            # or [4,4,4,4,4] -> 1024. Let's use 5 dims with 4 levels each.
            self.quantizer = FSQ(levels=[4,4,4,4,4], input_dim=hidden_dim, hidden_dim=hidden_dim)
        elif method == 'lfq':
            # LFQ: 10 bit binary -> 1024 size
            self.quantizer = LFQ(input_dim=hidden_dim, codebook_dim=10)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        # 3. Decoder
        self.decoder = Decoder(input_dim, hidden_dim, arch=arch)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, T, D] -> [B, D, T]
        z = self.encoder(x)
        loss_vq, z_q, perplexity = self.quantizer(z)
        x_recon = self.decoder(z_q)
        x_recon = x_recon.permute(0, 2, 1)
        return x_recon, loss_vq, perplexity

if __name__ == "__main__":
    dummy = torch.randn(2, 64, 29)
    for method in ['ema', 'rvq', 'fsq', 'lfq']:
        print(f"Testing Method: {method}...")
        model = MotionVQVAE(input_dim=29, hidden_dim=64, method=method)
        r, l, p = model(dummy)
        print(f"  Shape: {r.shape}, Loss: {l.item():.4f}, PPL: {p.item():.1f}")