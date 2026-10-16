import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Basic components
# ============================================================

class InputMasking(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x

        # Support both input formats:
        # (B, L, C), e.g., (B, 80, 5)
        # (B, C, L), e.g., (B, 5, 80)
        if x.dim() != 3:
            return x

        if x.size(1) == 5:
            # (B, C, L)
            B, C, L = x.size()
            mask = (torch.rand(B, 1, L, device=x.device) > self.p).float()
        else:
            # (B, L, C)
            B, L, C = x.size()
            mask = (torch.rand(B, L, 1, device=x.device) > self.p).float()

        return x * mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ResBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            dilation: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        if x.dim() == 2:
            y = self.excitation(x)
            return x * y

        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class AttnPool1d(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, L, H)
        scores = self.attn(x).squeeze(-1)  # (B, L)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)


class CoralOrdinalHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc_z = nn.Linear(in_dim, 1)
        self.theta_deltas = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        z = self.fc_z(x)
        theta = torch.cumsum(F.softplus(self.theta_deltas), dim=0)
        return z - theta.view(1, -1)


# ============================================================
# 2) Auxiliary branch encoders: CGR uses CoordConv-CNN; PseDNC uses a two-layer MLP
# ============================================================

class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x):
        b, _, h, w = x.size()

        y_grid = torch.linspace(
            -1, 1, h, device=x.device
        ).view(1, 1, h, 1).expand(b, 1, h, w)

        x_grid = torch.linspace(
            -1, 1, w, device=x.device
        ).view(1, 1, 1, w).expand(b, 1, h, w)

        out = torch.cat([x, x_grid, y_grid], dim=1)
        return self.conv(out)


class CGR_Coord_Encoder(nn.Module):
    def __init__(self, in_channels=2, out_dim=256, drop=0.10):
        super().__init__()

        self.net = nn.Sequential(
            CoordConv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # If the input is flattened to (B, 512), reshape it to (B, 2, 16, 16)
        if x.dim() == 2:
            x = x.view(x.size(0), 2, 16, 16)
        return self.net(x)


class PseDNC_Encoder(nn.Module):
    def __init__(self, input_dim, out_dim=256, drop=0.20):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(drop),

            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3) Main model: Adaptive Gated FiLM-CNN-Transformer
# ============================================================

class TransformerHybridMultiTask(nn.Module):
    def __init__(
            self,
            input_size=5,
            hidden_size=256,
            num_classes=5,
            dropout_rate=0.2,
            conv_kernels=(5, 5, 3),
            num_layers=2,
            nhead=8,
            se_reduction=16,
            use_cgr=True,
            use_psednc=True,
            psednc_dim=22,
            mask_prob=0.0,
            pse_scale_init=0.05
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_cgr = bool(use_cgr)
        self.use_psednc = bool(use_psednc)
        self.input_masking = InputMasking(p=mask_prob)

        # ----------------------------------------------------
        # A. Sequence Branch: ResCNN + SE + Transformer
        # ----------------------------------------------------
        k1, k2, k3 = conv_kernels

        self.cnn1 = nn.Sequential(
            ResBlock1d(
                input_size,
                hidden_size,
                k1,
                padding=k1 // 2
            )
        )

        self.cnn2 = nn.Sequential(
            ResBlock1d(
                hidden_size,
                hidden_size,
                k2,
                padding=k2 // 2 * 2,
                dilation=2
            )
        )

        self.cnn3 = nn.Sequential(
            ResBlock1d(
                hidden_size,
                hidden_size,
                k3,
                padding=k3 // 2
            ),
            nn.MaxPool1d(2),
            SEBlock(hidden_size, reduction=se_reduction)
        )

        self.input_proj = nn.Conv1d(hidden_size, hidden_size, 1)
        self.pos_encoder = PositionalEncoding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attn_pool = AttnPool1d(hidden_size)
        self.seq_drop = nn.Dropout(dropout_rate * 0.5)

        # ----------------------------------------------------
        # B. Condition Encoders
        # ----------------------------------------------------
        cond_count = 0

        if self.use_cgr:
            self.cgr_encoder = CGR_Coord_Encoder(
                in_channels=2,
                out_dim=hidden_size,
                drop=dropout_rate * 0.5
            )
            cond_count += 1

        if self.use_psednc:
            self.pse_encoder = PseDNC_Encoder(
                input_dim=psednc_dim,
                out_dim=hidden_size,
                drop=0.20
            )
            cond_count += 1

        self.use_film = cond_count > 0

        # ----------------------------------------------------
        # C. Adaptive Gated Condition Fusion + FiLM
        # ----------------------------------------------------
        if self.use_film:
            # Map the fused conditioning vector to hidden_size dimensions
            self.cond_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU()
            )


            if self.use_cgr and self.use_psednc:
                self.cond_fuse_gate = nn.Linear(hidden_size * 2, 2)


                nn.init.zeros_(self.cond_fuse_gate.weight)
                with torch.no_grad():
                    self.cond_fuse_gate.bias[0].fill_(1.0)
                    self.cond_fuse_gate.bias[1].fill_(-1.0)


                init_logit = math.log(pse_scale_init / (1.0 - pse_scale_init))
                self.pse_scale_logit = nn.Parameter(
                    torch.tensor(init_logit, dtype=torch.float32)
                )

            self.film_gen = nn.Linear(hidden_size, 2 * hidden_size)
            self.film_scale = nn.Parameter(torch.tensor(0.02))

            self.cond_gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            )
            self.cond_gate_scale = nn.Parameter(torch.tensor(0.1))

        # ----------------------------------------------------
        # D. Heads
        # ----------------------------------------------------
        head_in = hidden_size * 2 if self.use_film else hidden_size

        self.head_cls = nn.Linear(head_in, num_classes)
        self.ord_head = CoralOrdinalHead(1, num_classes)

    def _build_condition(self, extra_feat=None, psednc_feat=None):
        z_cgr = None
        z_pse = None

        if self.use_cgr and extra_feat is not None:
            z_cgr = self.cgr_encoder(extra_feat)

        if self.use_psednc and psednc_feat is not None:
            z_pse = self.pse_encoder(psednc_feat)

        # all_fusion: CGR dominates, with PseDNC as a weak gated supplement
        if z_cgr is not None and z_pse is not None:
            gate_logits = self.cond_fuse_gate(
                torch.cat([z_cgr, z_pse], dim=-1)
            )
            weights = torch.softmax(gate_logits, dim=-1)

            w_cgr = weights[:, 0:1]
            w_pse = weights[:, 1:2]

            pse_scale = torch.sigmoid(self.pse_scale_logit)

            z_cond_raw = (
                w_cgr * z_cgr
                + w_pse * pse_scale * z_pse
            )

        elif z_cgr is not None:
            z_cond_raw = z_cgr

        elif z_pse is not None:
            z_cond_raw = z_pse

        else:
            z_cond_raw = None

        if z_cond_raw is None:
            return None

        return self.cond_mlp(z_cond_raw)

    def forward(self, x, extra_feat=None, psednc_feat=None, return_aux=False):
        # ----------------------------------------------------
        # A. Seq Flow
        # ----------------------------------------------------
        x = self.input_masking(x)

        # If the input is (B, L, 5), convert it to (B, 5, L) for Conv1d
        x = x.permute(0, 2, 1) if x.size(1) != 5 else x

        x_seq = self.cnn3(
            self.cnn2(
                self.cnn1(x)
            )
        )

        x_tokens = self.input_proj(x_seq).permute(0, 2, 1)
        x_tokens = self.pos_encoder(x_tokens)
        x_tokens = self.transformer(x_tokens)

        v_seq = self.attn_pool(x_tokens)
        v_seq = self.seq_drop(v_seq)

        # ----------------------------------------------------
        # B. Adaptive Gated FiLM Fusion
        # ----------------------------------------------------
        if self.use_film:
            v_cond = self._build_condition(
                extra_feat=extra_feat,
                psednc_feat=psednc_feat
            )

            gamma, beta = self.film_gen(v_cond).chunk(2, dim=-1)

            v_mod = (
                v_seq * (1 + self.film_scale * gamma)
                + self.film_scale * beta
            )

            v_extra = (
                v_cond
                * self.cond_gate(v_cond)
                * self.cond_gate_scale
            )

            v_final = torch.cat([v_mod, v_extra], dim=-1)

        else:
            v_final = v_seq

        logits = self.head_cls(v_final)

        if return_aux:
            aux = {
                "ord": self.ord_head(
                    logits.mean(dim=1, keepdim=True)
                )
            }
            return logits, aux

        return logits