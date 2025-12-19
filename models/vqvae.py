import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Quantization Modules (Enhanced with Metrics)
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
        
        # --- EMA Update ---
        if self.training and self.use_ema:
            encodings_sum = encodings.sum(0)
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # --- Loss ---
        if self.use_ema:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            loss = self.commitment_cost * e_latent_loss
        else:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        # --- Metrics (PPL & DCR) ---
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Dead Code Ratio: % of codes not used in this batch
        # encodings: [Batch*Time, Num_Embeddings]
        active_codes = (encodings.sum(0) > 0).float().sum()
        dcr = 1.0 - (active_codes / self.num_embeddings)
        
        metrics = {'perplexity': perplexity, 'dcr': dcr}
        
        return loss, quantized.permute(0, 2, 1).contiguous(), metrics

class ResidualVQ(nn.Module):
    """ Residual VQ: Aggregate metrics from all layers """
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
        all_ppl = []
        all_dcr = []
        
        for layer in self.layers:
            loss, quantized, metrics = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss += loss
            all_ppl.append(metrics['perplexity'])
            all_dcr.append(metrics['dcr'])
            
        # Return average metrics across layers
        avg_metrics = {
            'perplexity': torch.mean(torch.stack(all_ppl)),
            'dcr': torch.mean(torch.stack(all_dcr))
        }
            
        return total_loss, quantized_out, avg_metrics

class FSQ(nn.Module):
    """ FSQ: SOTA 2024. Metrics adapted for implicit codebook. """
    def __init__(self, levels, input_dim, hidden_dim):
        super().__init__()
        self.levels = levels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fsq_dim = len(levels)
        self.project_in = nn.Conv1d(input_dim, self.fsq_dim, 1)
        self.project_out = nn.Conv1d(self.fsq_dim, input_dim, 1)
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.int32))
        # Calculate basis for unique index calculation: [1, L0, L0*L1, ...]
        self.register_buffer('_basis', torch.cumprod(torch.cat([torch.tensor([1], device='cpu'), torch.tensor(levels[:-1], device='cpu')]), dim=0).to(torch.int32))
        self.codebook_size = math.prod(levels)

    def forward(self, z):
        z = self.project_in(z) # [B, fsq_dim, T]
        z = z.permute(0, 2, 1) # [B, T, fsq_dim]
        
        # Quantize (Round)
        z_hard = self._round_ste(z)
        
        # Project out
        z_out = z_hard.permute(0, 2, 1)
        z_out = self.project_out(z_out)
        
        loss = torch.tensor(0.0, device=z.device)
        
        # --- Metrics ---
        # Calculate unique indices to estimate utilization
        # indices = sum(z_i * basis_i)
        indices = (z_hard * self._basis).sum(dim=-1).long() # [B, T]
        unique_codes = torch.unique(indices).numel()
        
        dcr = 1.0 - (unique_codes / self.codebook_size)
        metrics = {
            'perplexity': torch.tensor(float(unique_codes), device=z.device), # Use count as proxy for PPL
            'dcr': torch.tensor(dcr, device=z.device)
        }
        
        return loss, z_out, metrics

    def _round_ste(self, z):
        z_hard = torch.round(z)
        return z + (z_hard - z).detach()

class LFQ(nn.Module):
    """ LFQ: Binary Quantization. Metrics adapted. """
    def __init__(self, input_dim, codebook_dim=10, entropy_loss_weight=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook_size = 2**codebook_dim
        self.project_in = nn.Conv1d(input_dim, codebook_dim, 1)
        self.project_out = nn.Conv1d(codebook_dim, input_dim, 1)
        # Basis for binary to int: [1, 2, 4, 8...]
        self.register_buffer('_basis', 2**torch.arange(codebook_dim))

    def forward(self, z):
        z_e = self.project_in(z) 
        z_q = torch.where(z_e > 0, torch.tensor(1.0, device=z.device), torch.tensor(-1.0, device=z.device))
        z_q = z_e + (z_q - z_e).detach()
        
        # Entropy Loss
        prob = torch.sigmoid(z_e)
        entropy = - (prob * torch.log(prob + 1e-6) + (1 - prob) * torch.log(1 - prob + 1e-6))
        loss = -entropy.mean() * self.entropy_loss_weight
        
        out = self.project_out(z_q)
        
        # --- Metrics ---
        # Convert binary code to integer index to count uniques
        # Map -1 -> 0, 1 -> 1
        binary_bits = (z_q > 0).int().permute(0, 2, 1) # [B, T, C]
        indices = (binary_bits * self._basis).sum(dim=-1) # [B, T]
        unique_codes = torch.unique(indices).numel()
        
        dcr = 1.0 - (unique_codes / self.codebook_size)
        metrics = {
            'perplexity': torch.tensor(float(unique_codes), device=z.device),
            'dcr': torch.tensor(dcr, device=z.device)
        }
        
        return loss, out, metrics

# === 1. 新增 HybridVQ 类 (放在 LFQ 类后面) ===
# models/vqvae.py 中的 HybridVQ 类修改如下

class HybridVQ(nn.Module):
    """
    Hybrid v2: FSQ (Base) + RVQ (Residual Refinement)
    强力修补版：用多层 RVQ 来修补 FSQ 的残差，旨在同时获得高精度和较好的利用率。
    """
    def __init__(self, hidden_dim, fsq_levels=[8, 5, 5, 5], vq_codebook_size=1024):
        super().__init__()
        # 1. Base Layer: FSQ (负责大轮廓)
        self.fsq = FSQ(levels=fsq_levels, input_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # 2. Refinement Layer: RVQ (关键修改：用 RVQ 替代普通的 VQ)
        # 使用 4 层 RVQ 来强力拟合残差，提升 Recon Quality
        self.vq = ResidualVQ(
            num_quantizers=4,           # 4层残差量化，对齐 ResNet+RVQ 的配置
            num_embeddings=vq_codebook_size, 
            embedding_dim=hidden_dim, 
            commitment_cost=0.25, 
            use_ema=True
        )

    def forward(self, z):
        # Step 1: Base (FSQ)
        _, z_fsq, metrics_fsq = self.fsq(z)
        
        # Step 2: Residual Calculation
        residual = z - z_fsq
        
        # Step 3: Refine (RVQ)
        # RVQ 的 forward 返回: loss, quantized, metrics
        loss_vq, z_vq, metrics_vq = self.vq(residual)
        
        # Step 4: Combine
        z_out = z_fsq + z_vq
        
        # Metrics: 综合汇报
        # 我们希望看到 RVQ 帮助提升精度，同时 FSQ 保持高利用率
        metrics = {
            'perplexity': metrics_fsq['perplexity'],  # 主 PPL 看 FSQ
            'dcr': metrics_fsq['dcr'],                # 主 DCR 看 FSQ
            'rvq_ppl': metrics_vq['perplexity']       # 记录一下 RVQ 的利用率供参考
        }
        
        return loss_vq, z_out, metrics
    
# ==========================================
# 2. Building Blocks (Keep Unchanged)
# ==========================================
# (ResBlock1D, PositionalEncoding, Encoder, Decoder 保持原来的代码不变)
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
        else: # Transformer placeholder for brevity, assume implemented as before
             self.model = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
            )
    def forward(self, x):
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
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(hidden_dim, input_dim, 4, 2, 1),
            )

    def forward(self, x):
        return self.model(x)
# ==========================================
# 4. Main Dual-Encoder VQ-VAE Model (Corrected for Paper)
# ==========================================

class DualMotionVQVAE(nn.Module):
    def __init__(self, 
                 human_input_dim=263, # SMPL feature dim
                 robot_input_dim=29,  # G1 Robot dim
                 hidden_dim=64, 
                 codebook_size=1024, 
                 arch='resnet', 
                 method='hybrid', 
                 n_layers=4):
        super(DualMotionVQVAE, self).__init__()
        
        # --- Dual Encoders ---
        # 1. Human Encoder: Maps SMPL features to Latent Space
        self.human_encoder = Encoder(human_input_dim, hidden_dim, arch=arch)
        
        # 2. Robot Encoder: Maps Robot joints to Latent Space
        self.robot_encoder = Encoder(robot_input_dim, hidden_dim, arch=arch)
        
        # --- Shared Quantizer (The "Bridge") ---
        if method == 'standard':
            self.quantizer = VectorQuantizer(codebook_size, hidden_dim, use_ema=False)
        elif method == 'ema':
            self.quantizer = VectorQuantizer(codebook_size, hidden_dim, use_ema=True)
        elif method == 'rvq':
            self.quantizer = ResidualVQ(num_quantizers=n_layers, num_embeddings=codebook_size, 
                                        embedding_dim=hidden_dim, use_ema=True)
        elif method == 'fsq':
            self.quantizer = FSQ(levels=[4,4,4,4,4], input_dim=hidden_dim, hidden_dim=hidden_dim)
        elif method == 'lfq':
            self.quantizer = LFQ(input_dim=hidden_dim, codebook_dim=10)
        elif method == 'hybrid':
            self.quantizer = HybridVQ(hidden_dim=hidden_dim, fsq_levels=[8,5,5,5], vq_codebook_size=512)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        # --- Robot Decoder Only (Per Paper Figure 1) ---
        # 论文中主要关注从 Latent 解码回 Robot。
        # 如果训练还需要重建 Human 以保证 encoder 质量，可以加 human_decoder，
        # 但 Figure 1 强调的是 Robot Execution，通常我们希望 Latent 包含的信息足以重建 Robot 动作。
        self.robot_decoder = Decoder(robot_input_dim, hidden_dim, arch=arch)

    def forward(self, x_robot=None, x_human=None):
        """
        支持三种模式：
        1. 仅机器人训练: forward(x_robot=data)
        2. 仅人类推理/生成: forward(x_human=data) -> decode to robot
        3. 对齐训练 (Alignment): forward(x_robot=data, x_human=data)
        """
        outputs = {}
        
        # --- Branch 1: Robot Flow ---
        if x_robot is not None:
            x_robot = x_robot.permute(0, 2, 1) # [B, C, T]
            z_e_robot = self.robot_encoder(x_robot)
            
            # Quantize
            loss_vq_r, z_q_robot, metrics_r = self.quantizer(z_e_robot)
            
            # Decode (Reconstruction)
            x_recon_robot = self.robot_decoder(z_q_robot)
            outputs['robot'] = {
                'recon': x_recon_robot.permute(0, 2, 1),
                'loss_vq': loss_vq_r,
                'metrics': metrics_r,
                'z_e': z_e_robot # For alignment loss
            }

        # --- Branch 2: Human Flow ---
        if x_human is not None:
            x_human = x_human.permute(0, 2, 1) # [B, C, T]
            z_e_human = self.human_encoder(x_human)
            
            # Quantize (Shared Codebook!)
            # 注意：人类分支通常用于 inference 或 alignment training
            loss_vq_h, z_q_human, metrics_h = self.quantizer(z_e_human)
            
            # Cross-Decoding: Human Latent -> Robot Body
            # 这是论文核心：Implicit Retargeting
            x_retargeted = self.robot_decoder(z_q_human)
            
            outputs['human'] = {
                'retargeted': x_retargeted.permute(0, 2, 1),
                'loss_vq': loss_vq_h,
                'metrics': metrics_h,
                'z_e': z_e_human # For alignment loss
            }
            
        return outputs