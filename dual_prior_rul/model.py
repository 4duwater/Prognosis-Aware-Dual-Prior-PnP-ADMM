import torch
import torch.nn as nn
import math
from .transformer.Layers import Encoder

class Gating(nn.Module):
    def __init__(self, m,d_model, seq_len):
        super().__init__()
        self.m = m
        self.seq_len = seq_len
        # the reset gate r_i
        self.W_r = nn.Parameter(torch.Tensor(m, m))
        self.V_r = nn.Parameter(torch.Tensor(m, m))
        self.b_r = nn.Parameter(torch.Tensor(m))

        # the update gate u_i
        self.W_u = nn.Parameter(torch.Tensor(m, m))
        self.V_u = nn.Parameter(torch.Tensor(m, m))
        self.b_u = nn.Parameter(torch.Tensor(m))

        # the output
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(seq_len,seq_len, kernel_size=3, stride=1,padding=1),
        )
        

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.m)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):

        x = x
        x_i = x
        h_i = self.cnn_layers(x) 
        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)
        # the output of the gating mechanism
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)

        return torch.matmul(hh_i, self.W_e) + self.b_e 
        
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        #x = x + self.pe[:x.size(0), :]
        x = x.permute(2,0,1)
     
        x = x + self.pe[:x.size(0),:]
        x = self.layer_norm(self.dropout(x))
        return x.transpose(0,1)

class PretrainTransformer(nn.Module):
    def __init__(self, seq_size=50, patch_size=10, in_chans=14, embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        self.seq_size = seq_size
        self.in_chans = in_chans
        self.gating = Gating(in_chans, embed_dim, seq_size)
        self.pos_encoder = FixedPositionalEncoding(embed_dim, max_len=1024)
        self.encoder_block = nn.Sequential(
            Encoder(depth, num_heads, embed_dim//num_heads, embed_dim//num_heads, embed_dim, embed_dim//2),
            nn.Linear(embed_dim, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, in_chans)
        )

    def forward_encoder(self, x):
        x = self.pos_encoder(x.transpose(1, 2)) 
        enc_x = self.encoder_block(x)
        return enc_x

    def forward(self, seq):
        gcu_out = self.gating(seq)
        latent_x = self.forward_encoder(gcu_out)
        out = self.decoder(latent_x)
        
        return out
