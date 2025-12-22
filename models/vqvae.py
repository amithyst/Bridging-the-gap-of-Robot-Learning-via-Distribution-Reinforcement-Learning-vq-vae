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

class IdentityVQ(nn.Module):
    """
    AE Mode: No Quantization. Just pass through.
    Used for debugging reconstruction capability.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, z):
        # z: [Batch, Channel, Time]
        # Loss is 0, Metrics are dummy values
        loss = torch.tensor(0.0, device=z.device)
        metrics = {
            'perplexity': torch.tensor(1.0, device=z.device),
            'dcr': torch.tensor(0.0, device=z.device)
        }
        return loss, z, metrics

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
    def __init__(self, input_dim, hidden_dim, arch='simple', num_res_layers=4): # <--- 新增参数
        super().__init__()
        self.arch = arch
        if arch == 'simple':
            self.model = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
            )
        elif arch == 'resnet':
            # === 修改开始：动态层数 ===
            layers = [
                nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), 
                nn.LeakyReLU(0.2),
            ]
            # 循环添加 ResBlock
            for _ in range(num_res_layers):
                layers.append(ResBlock1D(hidden_dim))
                
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                ResBlock1D(hidden_dim), # 最后一层保留做特征整理
            ])
            self.model = nn.Sequential(*layers)
            # === 修改结束 ===
        else: # Transformer placeholder for brevity, assume implemented as before
             self.model = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
            )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, arch='simple', num_res_layers=4): # <--- 新增参数
        super().__init__()
        self.arch = arch
        if arch == 'simple':
            self.model = nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(hidden_dim, input_dim, 4, 2, 1),
            )
        elif arch == 'resnet':
            # === 修改开始：动态层数 ===
            layers = []
            # 对称：先加 ResBlocks
            for _ in range(num_res_layers):
                layers.append(ResBlock1D(hidden_dim))
            
            layers.extend([
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1), 
                nn.LeakyReLU(0.2),
            ])
            
            # 再加一层 ResBlock 过渡
            layers.append(ResBlock1D(hidden_dim))
            
            layers.extend([
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv1d(hidden_dim, input_dim, 3, 1, 1),
            ])
            self.model = nn.Sequential(*layers)
            # === 修改结束 ===
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(hidden_dim, input_dim, 4, 2, 1),
            )

    def forward(self, x):
        return self.model(x)
    

# models/vqvae.py

class NoDownsampleEncoder(nn.Module):
    """
    全分辨率 ResNet Encoder: 始终保持 stride=1，不进行下采样。
    输入: [B, C, T] -> 输出: [B, Hidden, T]
    """
    def __init__(self, input_dim, hidden_dim, num_res_layers=4):
        super().__init__()
        # 1. 初始投影 (保持 T 不变)
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 2. 堆叠 ResBlock (保持 T 不变)
        for _ in range(num_res_layers):
            self.model.add_module(f"res_{_}", ResBlock1D(hidden_dim))
            
        # 3. 最终整理
        self.model.add_module("final_conv", nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
        self.model.add_module("final_act", nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)

class NoDownsampleDecoder(nn.Module):
    """
    全分辨率 ResNet Decoder: 始终保持 stride=1。
    输入: [B, Hidden, T] -> 输出: [B, Output_Dim, T]
    """
    def __init__(self, output_dim, hidden_dim, num_res_layers=4):
        super().__init__()
        self.model = nn.Sequential()
        
        # 1. 堆叠 ResBlock
        for i in range(num_res_layers):
            self.model.add_module(f"res_{i}", ResBlock1D(hidden_dim))
            
        # 2. 输出层
        self.model.add_module("out_conv", nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        return self.model(x)
    
class TransformerPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, T, C)

    def forward(self, x):
        # x shape: (Batch, Time, Dim)
        return x + self.pe[:, :x.size(1), :]




class TransformerMotionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        # 1. Input Projection (不降采样，保留64帧)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Transformer Backbone
        self.pe = TransformerPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Aggregation (64 Tokens -> 1 Token)
        # 我们使用一个 Learnable 的 query 或者简单的 Mean Pooling
        # 这里为了保留全量信息压缩，使用 Mean Pooling + Projection
        self.output_proj = nn.Linear(d_model, hidden_dim)

    def forward(self, x):
        # x: [Batch, Channel, Time] -> (B, 29, 64)
        x = x.permute(0, 2, 1) # [B, T, C]
        
        # Linear Projection
        x = self.input_proj(x) # [B, 64, 256]
        x = self.pe(x)
        
        # Transformer Process (全序列交互)
        x = self.transformer(x) # [B, 64, 256]
        
        # Global Pooling (将64帧的信息压缩到1帧)
        # 这里的 Mean 包含了所有时刻的信息
        x = torch.mean(x, dim=1, keepdim=True) # [B, 1, 256]
        
        x = self.output_proj(x) # [B, 1, 64]
        
        # 变回 [B, 64, 1] 以符合 VQ 输入习惯
        return x.permute(0, 2, 1) 

class TransformerMotionDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, d_model=256, nhead=4, num_layers=4, seq_len=64):
        super().__init__()
        self.seq_len = seq_len
        
        # 1. Expand Latent (1 -> 64)
        self.input_proj = nn.Linear(hidden_dim, d_model)
        
        # 2. Transformer
        self.pe = TransformerPositionalEncoding(d_model)
        # Decoder 同样使用 EncoderLayer 结构（非自回归，因为是一次性生成）
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 3. Output Projection
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [Batch, Hidden_Dim, 1]
        x = x.permute(0, 2, 1) # [B, 1, Hidden]
        
        # Expand: 将 1 个 Latent 广播复制到 64 个时间步
        x = self.input_proj(x) # [B, 1, d_model]
        x = x.repeat(1, self.seq_len, 1) # [B, 64, d_model]
        
        # 加上位置编码，让网络知道第1个复制品是第1帧，第64个是第64帧
        x = self.pe(x)
        
        # Transformer 恢复时序细节
        x = self.transformer(x)
        
        # Output
        x = self.output_proj(x) # [B, 64, Output_Dim]
        
        return x.permute(0, 2, 1) # [B, Output_Dim, 64]





# ==========================================
# 4. Main Dual-Encoder VQ-VAE Model (Corrected for Paper)
# ==========================================
class DualMotionVQVAE(nn.Module):
    def __init__(self, 
                 human_input_dim=263, 
                 robot_input_dim=29, 
                 hidden_dim=64, 
                 codebook_size=1024, 
                 arch='transformer',  # 默认改为 transformer
                 method='hybrid', 
                 n_layers=4,
                 window_size=64):
        super(DualMotionVQVAE, self).__init__()
        
        self.arch = arch
        self.window_size = window_size # <--- [新增] 记录下来
        
        # --- Dual Encoders ---
        if arch == 'transformer':
            # Transformer 架构：更深，带全局注意力
            # d_model=256 是 Transformer 的内部维度，hidden_dim=64 是量化维度
            self.human_encoder = TransformerMotionEncoder(human_input_dim, hidden_dim, d_model=256, num_layers=4)
            self.robot_encoder = TransformerMotionEncoder(robot_input_dim, hidden_dim, d_model=256, num_layers=4)
        # === [新增] 全分辨率 ResNet ===
        elif arch == 'resnet_no_down':
            self.human_encoder = NoDownsampleEncoder(human_input_dim, hidden_dim)
            self.robot_encoder = NoDownsampleEncoder(robot_input_dim, hidden_dim)
        # ============================
        else:
            # 旧架构兼容 (Simple / ResNet)
            self.human_encoder = Encoder(human_input_dim, hidden_dim, arch=arch)
            self.robot_encoder = Encoder(robot_input_dim, hidden_dim, arch=arch)
        
        # --- Shared Quantizer ---
        if method == 'standard':
            self.quantizer = VectorQuantizer(codebook_size, hidden_dim, use_ema=False)
        elif method == 'ema':
            self.quantizer = VectorQuantizer(codebook_size, hidden_dim, use_ema=True)
        elif method == 'rvq':
            self.quantizer = ResidualVQ(num_quantizers=n_layers, num_embeddings=codebook_size, 
                                        embedding_dim=hidden_dim, use_ema=True)
        elif method == 'fsq':
            self.quantizer = FSQ(levels=[8,5,5,5], input_dim=hidden_dim, hidden_dim=hidden_dim)
        elif method == 'lfq':
            self.quantizer = LFQ(input_dim=hidden_dim, codebook_dim=10)
        elif method == 'hybrid':
            # 推荐配置：Hybrid (FSQ Base + RVQ Detail)
            self.quantizer = HybridVQ(hidden_dim=hidden_dim, fsq_levels=[8,5,5,5], vq_codebook_size=512)
        # === [必须添加这一段] ===
        elif method == 'ae':
            # AE 模式：直接使用 IdentityVQ，不做量化，不做 VQ，只是普通的自编码器
            self.quantizer = IdentityVQ()
        # =======================
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        # --- Robot Decoder Only ---
        if arch == 'transformer':
            # [修改] 2. 将 window_size 传给 seq_len
            self.robot_decoder = TransformerMotionDecoder(
                robot_input_dim, 
                hidden_dim, 
                d_model=256, 
                num_layers=4, 
                seq_len=window_size  # <--- 关键修改：不再是写死的 64
            )
        # === [新增] 全分辨率 ResNet Decoder ===
        elif arch == 'resnet_no_down':
            self.robot_decoder = NoDownsampleDecoder(robot_input_dim, hidden_dim)
        # =====================================
        else:
            self.robot_decoder = Decoder(robot_input_dim, hidden_dim, arch=arch)

    def forward(self, x_robot=None, x_human=None):
        outputs = {}
        
        # --- Branch 1: Robot Flow ---
        if x_robot is not None:
            x_robot = x_robot.permute(0, 2, 1) # [B, C, T]
            z_e_robot = self.robot_encoder(x_robot)
            
            # Quantize
            loss_vq_r, z_q_robot, metrics_r = self.quantizer(z_e_robot)
            
            # Decode
            x_recon_robot = self.robot_decoder(z_q_robot)
            outputs['robot'] = {
                'recon': x_recon_robot.permute(0, 2, 1),
                'loss_vq': loss_vq_r,
                'metrics': metrics_r,
                'z_e': z_e_robot # For alignment
            }

        # --- Branch 2: Human Flow ---
        if x_human is not None:
            x_human = x_human.permute(0, 2, 1) # [B, C, T]
            z_e_human = self.human_encoder(x_human)
            
            # Quantize (Shared Codebook)
            loss_vq_h, z_q_human, metrics_h = self.quantizer(z_e_human)
            
            # Cross-Decoding: Human Latent -> Robot Body
            x_retargeted = self.robot_decoder(z_q_human)
            
            outputs['human'] = {
                'retargeted': x_retargeted.permute(0, 2, 1),
                'loss_vq': loss_vq_h,
                'metrics': metrics_h,
                'z_e': z_e_human # For alignment
            }
            
        return outputs