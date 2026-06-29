import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class Transformer_Autoencoder(nn.Module):  # TransformerBasicBottleneckScaling from source

    def __init__(self, feats, lr, window_size, batch_size):
        super(Transformer_Autoencoder, self).__init__()
        self.name = 'Transformer_Autoencoder'
        self.lr = lr
        self.batch = batch_size
        self.n_feats = feats
        self.n_window = window_size
        self.scale = 64
        self.linear_layer = nn.Linear(feats, self.scale * feats)
        self.output_layer = nn.Linear(self.scale * feats, feats)
        self.pos_encoder = PositionalEncoding(self.scale * feats, 0.1, self.n_window, batch_first=True)
        encoder_layers = TransformerEncoderLayer(d_model=feats * self.scale, nhead=feats, batch_first=True,
                                                 dim_feedforward=256, dropout=0.01)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=feats * self.scale, nhead=feats, batch_first=True,
                                                 dim_feedforward=256, dropout=0.01)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt):
        model_dim = self.scale * self.n_feats

        src = self.linear_layer(src)
        src = src * math.sqrt(model_dim)
        src = self.pos_encoder(src)
        # batch x t x d
        memory = self.transformer_encoder(src)
        # batch x 1 x d
        z = torch.mean(memory, dim=1, keepdim=True)

        tgt = self.linear_layer(tgt)
        tgt = tgt * math.sqrt(model_dim)

        x = self.transformer_decoder(tgt, z)
        x = self.output_layer(x)
        # x = self.fcn(x)
        return x


def backprop(epoch: int, model: any, data: torch.Tensor, optimizer, scheduler, training: bool = True,
             device: str = 'cuda', dims: int = 1, batch_size: int = 256, window_size: int = 100,
             noise_steps: int = 100,
             denoise_steps: int = 50,
             diff_lambda: float = 0.1, 
             diffusion_training_net= None):

    loss_function = nn.MSELoss(reduction='none')
    data_x = torch.DoubleTensor(data)
    dataset = TensorDataset(data_x, data_x)
    bs = model.batch
    dataloader = DataLoader(dataset, batch_size=bs)
    model.double()
    l1s, l2s = [], []
    if training:
        model.train()
        recons = []
        for d, _ in dataloader:
            d = d.to(device)
            window = d
            window = window.to(device)
            if len(window.shape)==2:
                window = window.unsqueeze(-1)
            z = model(window, window)
            l1 = loss_function(z, window)
            loss = torch.mean(l1)
            l1s.append(loss.item())
            recons.append(z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s),  torch.cat(recons).detach().cpu().numpy()
    else:
        with torch.no_grad():
            model.eval()
            l1s = []
            recons = []
            for d, _ in dataloader:
                d = d.to(device)
                window = d
                window = window.to(device)
                if len(window.shape)==2:
                    window = window.unsqueeze(-1)
                z = model(window, window)
                recons.append(z)
                loss = loss_function(z, window)
                l1s.append(loss)
        return torch.cat(l1s).detach().cpu().numpy(), torch.cat(recons).detach().cpu().numpy()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        if self.batch_first:
            x = x + self.pe[pos:pos + x.size(1), :]
        else:
            x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)
