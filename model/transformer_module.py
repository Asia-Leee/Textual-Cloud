import torch.nn as nn
import torch
from einops import rearrange


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.): 
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PreNorm(nn.Module): 
    def __init__(self, dim, func):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = func
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim) 

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), 
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) 
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale 

        attn = self.attend(dots)
        #out:[bs,3,2,100]
        out = torch.matmul(attn, v) 
        out = rearrange(out, 'b h n d -> b n (h d)') 
        return self.to_out(out)



class GatedTriModalInteraction(nn.Module):
    def __init__(self, input_dim=512, nhead=8,dim_feedforward=2048, num_layers=1):
        super().__init__()

        self.type_embed = nn.Parameter(torch.randn(1, 3, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1, 
            activation='gelu',
            # norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):

        # Shape: [bs, 3, 512]
        raw_sequence = torch.stack([x1, x2, x3], dim=1)

        sequence = raw_sequence + self.type_embed

        sequence = sequence.permute(1, 0, 2)

        
        # Shape: [bs, 3, 512]
        fused_sequence = self.transformer(sequence)
        fused_sequence = fused_sequence.permute(1, 0, 2)

        
        sequence = sequence.permute(1,0,2)
        delta = fused_sequence - sequence
        alpha = self.gate_net(raw_sequence)
        final_sequence=fused_sequence+alpha*delta

        
        out1 = final_sequence[:, 0, :]
        out2 = final_sequence[:, 1, :]
        out3 = final_sequence[:, 2, :]

        return out1, out2, out3


