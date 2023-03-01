import torch
import torch.nn.functional as F
import torch.nn as nn

from modules.jukebox import Encoder, Decoder
from utils import AttrDict
from modules.vq import Bottleneck
from models import Generator


class FoVQVAE(nn.Module):
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
            out_estim - Reconstructed fo
            commit_losses - VQ commitment loss
            metrics - VQ metrics
        """
        # Ground-Truth is ristricted to fo
        in_gt = kwargs['f0']
        f0_h = self.encoder(in_gt)
        _, f0_h_q, commit_losses, metrics = self.vq(f0_h)
        out_estim = self.decoder(f0_h_q)

        return out_estim, commit_losses, metrics


class CodeGenerator(Generator):
    """Decoder with Embedding + HiFi-GAN Generator.
    
    Multi Encoder also supported:
      - Content
      - fo
      - Speaker
    """
    def __init__(self, h):
        super().__init__(h)

        # Content - Embdding
        self.emb_content = nn.Embedding(h.num_embeddings, h.embedding_dim)

        # fo - None | Fixed Encoder-VQ + Learnable Embedding
        self.use_f0 = h.get('f0', None)
        self.quantizer = None
        if h.get('f0_quantizer_path', None):
            # Fixed Enc-VQ + Emb (`quantizer` + `f0_dict`)
            assert self.use_f0, "Requires F0 set"
            self.quantizer = FoVQVAE(AttrDict(h.f0_quantizer))
            self.quantizer.load_state_dict(torch.load(h.f0_quantizer_path, map_location='cpu')['generator'])
            self.quantizer.eval()
            self.f0_dict = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)

        # Speaker - None | Embedding
        self.use_spk = h.get('multispkr', None)
        if self.use_spk:
            # NOTE: `num_embeddings=200` is too much. LJSpeech needs just 1, VCTK needs 109. 
            self.spk_emb = nn.Embedding(200, h.embedding_dim)

    @staticmethod
    def _upsample(signal, max_frames):
        """
        Returns:
            :: (B, Feat, T)
        """
        if signal.dim() == 3:
            # (B, Feat, T)
            pass
        elif signal.dim() == 2:
            # (B, Feat) -> (B, Feat, T=1)
            signal = signal.unsqueeze(2)
        else:
            # (B,) -> (B, Feat=1, T=1)
            signal = signal.view(-1, 1, 1)
        bsz, channels, cond_length = signal.size()

        # (B, Feat, T) -> (B, Feat, T, 1) -> (B, Feat, T, ?)
        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        """
        Args:
            kwargs
                code - Content unit series
                f0 - Fundamental frequency series
                spkr :: Optional[int] - speaker index (global feature)
        Returns:
            wave_estim - Estimated waveform
        """
        # PreNet
        unit_content = kwargs['code']
        ## Content - Embedding: index -> emb
        x = self.emb_content(unit_content).transpose(1, 2)
        ## [Optional] fo
        if self.use_f0:
            f0 = kwargs['f0']
            if self.quantizer:
                # Fixed-Enc-VQ (in-place fo encoding) + Emb
                self.quantizer.eval()
                assert not self.quantizer.training, "VQ is in training status!!!"
                f0_h = [x.detach() for x in self.quantizer.encoder(f0)]
                zs   = [x.detach() for x in self.quantizer.vq(f0_h)[0]]
                f0_h_q = self.f0_dict(zs[0].detach()).transpose(1, 2)
                f0 = f0_h_q
            # Up↑/Concat
            ## `x` up↑ | `f0` up↑
            if x.shape[-1] < f0.shape[-1]:
                x  = self._upsample(x, f0.shape[-1])
            else:
                f0 = self._upsample(f0, x.shape[-1])
            ## Concat
            x = torch.cat([x, f0], dim=1)
        ## [Optional] Global speaker embedding
        if self.use_spk:
            global_spk_emb = self.spk_emb(kwargs['spkr']).transpose(1, 2)
            # Up↑/Concat
            spk_emb_series = self._upsample(global_spk_emb, x.shape[-1])
            x = torch.cat([x, spk_emb_series], dim=1)
        ## [Optional] Other feature
        for k, feat in kwargs.items():
            if k in ['spkr', 'code', 'f0']:
                continue
            # Up↑/Concat
            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        # HiFi-GAN Generator
        wave_estim = super().forward(x)

        # Returns
        return wave_estim
