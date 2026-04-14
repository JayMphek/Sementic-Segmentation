import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Efficient Attention"""
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, dropout=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4], depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 dropout=0., drop_path_rate=0.1):
        super().__init__()
        
        self.depths = depths
        self.num_stages = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i]
            )
            
            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    sr_ratio=sr_ratios[i],
                    dropout=dropout,
                    drop_path=dpr[cur + j]
                )
                for j in range(depths[i])
            ])
            
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"blocks{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"blocks{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)
        return outs


class MLPDecoder(nn.Module):
    def __init__(self, in_channels, embed_dim, num_classes):
        super().__init__()
        
        self.linear_fuse = nn.ModuleList([
            nn.Conv2d(in_chan, embed_dim, 1) for in_chan in in_channels
        ])
        
        self.linear_pred = nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        B, _, H, W = features[0].shape
        
        outs = []
        for i, feat in enumerate(features):
            feat = self.linear_fuse[i](feat)
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(feat)
        
        out = torch.cat(outs, dim=1)
        out = self.linear_pred(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


class SegFormer(nn.Module):
    def __init__(self, num_classes, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4], depths=[3, 4, 6, 3], decoder_dim=256):
        super().__init__()
        
        self.encoder = MixVisionTransformer(
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=[8, 4, 2, 1]
        )
        
        self.decoder = MLPDecoder(
            in_channels=embed_dims,
            embed_dim=decoder_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        input_size = x.shape[2:]
        
        features = self.encoder(x)
        
        out = self.decoder(features)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


def create_segformer_model(num_classes, model_size='b2'):
    configs = {
        'b0': {'embed_dims': [32, 64, 160, 256], 'depths': [2, 2, 2, 2]},
        'b1': {'embed_dims': [64, 128, 320, 512], 'depths': [2, 2, 2, 2]},
        'b2': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 4, 6, 3]},
        'b3': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 4, 18, 3]},
        'b4': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 8, 27, 3]},
        'b5': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 6, 40, 3]},
    }
    config = configs[model_size]
    
    model = SegFormer(
        num_classes=num_classes,
        embed_dims=config['embed_dims'],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=config['depths'],
        decoder_dim=256
    )
    return model

if __name__ == "__main__":
    model = create_segformer_model(num_classes=20, model_size='b2')
    x = torch.randn(2, 3, 512, 1024)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")