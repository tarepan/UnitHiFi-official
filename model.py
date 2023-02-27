import torch
import torch.nn.functional as F
import torch.nn as nn

from modules.jukebox import Encoder, Decoder
from utils import AttrDict
from modules.vq import Bottleneck
from models import Generator


class Quantizer(nn.Module):
    """fo VQVAE.
    
    Used for Encoder_fo training & Decoder training
    """
    def __init__(self, h):
        super().__init__()

        self.encoder = Encoder(**h.f0_encoder_params)
        self.vq = Bottleneck(**h.f0_vq_params)
        self.decoder = Decoder(**h.f0_decoder_params)

    def forward(self, **kwargs):
        """
        Args:
            kwargs
                f0 - Ground-Truth fo
        Returns:
            reconst - Reconstructed fo
            commit_losses - VQ commitment loss
            metrics - VQ metrics
        """
        # Ground-Truth is ristricted to fo
        gt = kwargs['f0']
        f0_h = self.encoder(gt)
        _, f0_h_q, commit_losses, metrics = self.vq(f0_h)
        reconst = self.decoder(f0_h_q)

        return reconst, commit_losses, metrics


class CodeGenerator(Generator):
    def __init__(self, h):
        super().__init__(h)
        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)
        self.f0 = h.get('f0', None)
        self.multispkr = h.get('multispkr', None)

        if self.multispkr:
            self.spkr = nn.Embedding(200, h.embedding_dim)

        self.encoder = None
        self.vq = None
        if h.get("lambda_commit", None):
            assert self.f0, "Requires F0 set"
            self.encoder = Encoder(**h.f0_encoder_params)
            self.vq = Bottleneck(**h.f0_vq_params)

        self.code_encoder = None
        self.code_vq = None
        if h.get('lambda_commit_code', None):
            self.code_encoder = Encoder(**h.code_encoder_params)
            self.code_vq = Bottleneck(**h.code_vq_params)
            self.dict = None

        self.quantizer = None
        if h.get('f0_quantizer_path', None):
            assert self.f0, "Requires F0 set"
            self.quantizer = Quantizer(AttrDict(h.f0_quantizer))
            quantizer_state = torch.load(h.f0_quantizer_path, map_location='cpu')
            self.quantizer.load_state_dict(quantizer_state['generator'])
            self.quantizer.eval()
            self.f0_dict = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        code_commit_losses = None
        code_metrics = None
        if self.code_vq and kwargs['code'].dtype is torch.int64:
            x = self.code_vq.level_blocks[0].k[kwargs['code']].transpose(1, 2)
        elif self.code_vq:
            code_h = self.code_encoder(kwargs['code'])
            _, code_h_q, code_commit_losses, code_metrics = self.code_vq(code_h)
            x = code_h_q[0]
        else:
            x = self.dict(kwargs['code']).transpose(1, 2)

        f0_commit_losses = None
        f0_metrics = None
        if self.vq:
            f0_h = self.encoder(kwargs['f0'])
            _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
            kwargs['f0'] = f0_h_q[0]
        elif self.quantizer:
            self.quantizer.eval()
            assert not self.quantizer.training, "VQ is in training status!!!"
            f0_h = self.quantizer.encoder(kwargs['f0'])
            f0_h = [x.detach() for x in f0_h]
            zs, _, _, _ = self.quantizer.vq(f0_h)
            zs = [x.detach() for x in zs]
            f0_h_q = self.f0_dict(zs[0].detach()).transpose(1, 2)
            kwargs['f0'] = f0_h_q

        if self.f0:
            if x.shape[-1] < kwargs['f0'].shape[-1]:
                x = self._upsample(x, kwargs['f0'].shape[-1])
            else:
                kwargs['f0'] = self._upsample(kwargs['f0'], x.shape[-1])
            x = torch.cat([x, kwargs['f0']], dim=1)

        if self.multispkr:
            spkr = self.spkr(kwargs['spkr']).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        for k, feat in kwargs.items():
            if k in ['spkr', 'code', 'f0']:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        if self.vq or self.code_vq:
            return super().forward(x), (code_commit_losses, f0_commit_losses), (code_metrics, f0_metrics)
        else:
            return super().forward(x)
