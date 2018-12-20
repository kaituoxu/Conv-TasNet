# Created on 2018/12/10
# Author: Kaituo XU

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EPS = 1e-8


class TasNet(nn.Module):
    def __init__(self, L, N, hidden_size, num_layers,
                 bidirectional=True, nspk=2):
        super(TasNet, self).__init__()
        # hyper-parameter
        self.L, self.N = L, N
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.bidirectional = bidirectional
        self.nspk = nspk
        # Components
        self.encoder = Encoder(L, N)
        self.separator = Separator(N, hidden_size, num_layers,
                                   bidirectional=bidirectional, nspk=nspk)
        self.decoder = Decoder(N, L)

    def forward(self, mixture, mixture_lengths):
        """
        Args:
            mixture: [B, K, L]
            mixture_lengths: [B]
        Returns:
            est_source: [B, nspk, K, L]
        """
        mixture_w, norm_coef = self.encoder(mixture)
        est_mask = self.separator(mixture_w, mixture_lengths)
        est_source = self.decoder(mixture_w, est_mask, norm_coef)
        return est_source

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['L'], package['N'],
                    package['hidden_size'], package['num_layers'],
                    bidirectional=package['bidirectional'],
                    nspk=package['nspk'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'L': model.L,
            'N': model.N,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'bidirectional': model.bidirectional,
            'nspk': model.nspk,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D gated conv layer.
    """
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # hyper-parameter
        self.L = L
        self.N = N
        # Components
        # Maybe we can impl 1-D conv by nn.Linear()?
        self.conv1d_U = nn.Conv1d(L, N, kernel_size=1, stride=1, bias=False)
        self.conv1d_V = nn.Conv1d(L, N, kernel_size=1, stride=1, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [B, K, L]
        Returns:
            mixture_w: [B, K, N]
            norm_coef: [B, K, 1]
        """
        B, K, L = mixture.size()
        # L2 Norm along L axis
        norm_coef = torch.norm(mixture, p=2, dim=2, keepdim=True)  # B x K x 1
        norm_mixture = mixture / (norm_coef + EPS) # B x K x L
        # 1-D gated conv
        norm_mixture = torch.unsqueeze(norm_mixture.view(-1, L), 2)  # B*K x L x 1
        conv = F.relu(self.conv1d_U(norm_mixture))         # B*K x N x 1
        gate = torch.sigmoid(self.conv1d_V(norm_mixture))  # B*K x N x 1
        mixture_w = conv * gate  # B*K x N x 1
        mixture_w = mixture_w.view(B, K, self.N) # B x K x N
        return mixture_w, norm_coef


class Separator(nn.Module):
    """Estimation of source masks
    TODO: 1. normlization described in paper
          2. LSTM with skip connection
    """
    def __init__(self, N, hidden_size, num_layers, bidirectional=True, nspk=2):
        super(Separator, self).__init__()
        # hyper-parameter
        self.N = N
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.nspk = nspk
        # Components
        self.layer_norm = nn.LayerNorm(N)
        self.rnn = nn.LSTM(N, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=bidirectional)
        fc_in_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in_dim, nspk * N)
        ### To impl LSTM with skip connection
        # self.rnn = nn.ModuleList()
        # self.rnn += [nn.LSTM(N, hidden_size, num_layers=1,
        #                      batch_first=True,
        #                      bidirectional=bidirectional)]
        # for l in range(1, num_layers):
        #     self.rnn += [nn.LSTM(hidden_size, hidden_size, num_layers=1,
        #                          batch_first=True,
        #                          bidirectional=bidirectional)]

    def forward(self, mixture_w, mixture_lengths):
        """
        Args:
            mixture_w: [B, K, N], padded
        Returns:
            est_mask: [B, K, nspk, N]
        """
        B, K, N = mixture_w.size()
        # layer norm
        norm_mixture_w = self.layer_norm(mixture_w)
        # LSTM
        total_length = norm_mixture_w.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(norm_mixture_w, mixture_lengths,
                                            batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        # fc
        score = self.fc(output)  # B x K x nspk*N
        score = score.view(B, K, self.nspk, N)
        # softmax
        est_mask = F.softmax(score, dim=2)
        return est_mask


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        # hyper-parameter
        self.N, self.L = N, L
        # Components
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask, norm_coef):
        """
        Args:
            mixture_w: [B, K, N]
            est_mask: [B, K, nspk, N]
            norm_coef: [B, K, 1]
        Returns:
            est_source: [B, nspk, K, L]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 2) * est_mask  # B x K x nspk x N
        # S = DB
        est_source = self.basis_signals(source_w)  # B x K x nspk x L
        # reverse L2 norm
        norm_coef = torch.unsqueeze(norm_coef, 2)  # B x K x 1 x1
        est_source = est_source * norm_coef  # B x K x nspk x L
        est_source = est_source.permute((0, 2, 1, 3)).contiguous() # B x nspk x K x L
        return est_source



if __name__ == "__main__":
    torch.manual_seed(123)
    B, K, L, N, C = 2, 3, 4, 3, 2
    hidden_size, num_layers = 4, 2
    mixture = torch.randint(3, (B, K, L))
    lengths = torch.LongTensor([K for i in range(B)])
    # test Encoder
    encoder = Encoder(L, N)
    encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
    encoder.conv1d_V.weight.data = torch.randint(2, encoder.conv1d_V.weight.size())
    mixture_w, norm_coef = encoder(mixture)
    print('mixture', mixture)
    print('U', encoder.conv1d_U.weight)
    print('V', encoder.conv1d_V.weight)
    print('mixture_w', mixture_w)
    print('norm_coef', norm_coef)

    # test Separator
    separator = Separator(N, hidden_size, num_layers)
    est_mask = separator(mixture_w, lengths)
    print('est_mask', est_mask)

    # test Decoder
    decoder = Decoder(N, L)
    est_mask = torch.randint(2, (B, K, C, N))
    est_source = decoder(mixture_w, est_mask, norm_coef)
    print('est_source', est_source)

    # test TasNet
    tasnet = TasNet(L, N, hidden_size, num_layers)
    est_source = tasnet(mixture, lengths)
    print('est_source', est_source)
