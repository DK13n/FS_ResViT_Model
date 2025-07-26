import torch
import torch.nn as nn


class ChannelWisePatchEmbedding(nn.Module):
    def __init__(self,in_feature=49,out_feature=128):
        super().__init__()
        self.linear_embed = nn.Linear(in_feature,out_feature)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)  # [B, C, D] mỗi kênh là 1 patch
        x = self.linear_embed(x)
        return x  # Không transpose: mỗi patch là 1 kênh đặc trưng

class PositionalEncoding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = None
        self.dim = dim
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, N, D = x.shape  # N = C (số patch), D = H×W
        if self.pos_embed is None or self.pos_embed.shape[1] != N + 1:
            self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, self.dim, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, patch_dim=128, num_patches=512, num_classes=1000,
                 depth=6, heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.patch_embed = ChannelWisePatchEmbedding()
        self.pos_enc = PositionalEncoding(patch_dim)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(patch_dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(patch_dim)
        self.head = nn.Linear(patch_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)        # [B, C, H×W] → mỗi patch là 1 kênh
        x = self.pos_enc(x)            # [B, C+1, H×W]
        x = self.blocks(x)             # [B, C+1, H×W]
        x = self.norm(x)
        cls_token = x[:, 0]            # [B, D]
        x = self.head(cls_token)       # [B, num_classes]
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # Trả về cls token embedding
