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
    """
    def __init__(self, h):
        super().__init__(h)

        # Fixed fo Encoder
        self.fo_vqvae = FoVQVAE(AttrDict(h.f0_quantizer))
        self.fo_vqvae.load_state_dict(torch.load(h.f0_quantizer_path, map_location='cpu')['generator'])
        self.fo_vqvae.eval()

        # Embedding of Content/Pitch/Speaker
        self.emb_c = nn.Embedding(h.num_embeddings,                         h.embedding_dim)
        self.emb_p = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)
        self.emb_s = nn.Embedding(200,                                      h.embedding_dim)
        # NOTE: `num_embeddings=200` is too much. LJSpeech needs just 1, VCTK needs 109. 

    @staticmethod
    def _upsample(signal, max_frames):
        """
        Args:
            signal     :: (*) - Target signal
            max_frames :: int - Reference length, until which signal is upsampled
        Returns:
            :: (B, Feat=feat|1, T=max_frames)
        """
        # (B, Feat=feat, T=t) | (B, Feat=feat) | (B,) -> (B, Feat=feat|1, T=t|1)
        if signal.dim() == 3:
            pass
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
        else:
            signal = signal.view(-1, 1, 1)
        bsz, channels, length_t = signal.size()

        # (B, Feat, T) -> (B, Feat, T, 1) -> (B, Feat, T, ref//T)
        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // length_t)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        # (ref - T * (ref//T)) // (ref//T)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        # (B, Feat, T, ref/T) -> (B, Feat, T=ref)
        signal = signal.view(bsz, channels, max_frames)

        return signal

    def forward(self, **kwargs):
        """
        Args:
            kwargs
                code :: (B, Frame)    - Content unit series
                f0   :: (B, 1, Frame) - Fundamental frequency series
                spkr :: (B, 1)        - speaker index
        Returns:
            wave_estim - Estimated waveform
        """
        # Inputs - discrete Content, continuous fo, discrete Speaker
        z_c, fo, z_s = kwargs['code'], kwargs['f0'], kwargs['spkr']

        # fo encoding - In-place fo encoding to z_p by fixed Encoder-VQ
        self.fo_vqvae.eval()
        assert not self.fo_vqvae.training, "VQ is in training status!!!"
        h_p = [x.detach() for x in self.fo_vqvae.encoder(fo)]
        # z_p :: ()
        z_p = [x.detach() for x in self.fo_vqvae.vq(h_p)[0]][0].detach()

        # Embedding :: (B, Frame) -> (B, Frame, Emb) -> (B, Emb, Frame)
        emb_c = self.emb_c(z_c).transpose(1, 2)
        emb_p = self.emb_p(z_p).transpose(1, 2)
        emb_s = self.emb_s(z_s).transpose(1, 2)

        # Up↑/Concat :: (B, Emb=emb, Frame) -> (B, Emb=2*emb, Frame=max(c,p))
        if emb_c.shape[-1] < emb_p.shape[-1]:
            emb_c = self._upsample(emb_c, emb_p.shape[-1])
        else:
            emb_p = self._upsample(emb_p, emb_c.shape[-1])
        emb_c_p = torch.cat([emb_c, emb_p], dim=1)
        # :: (B, Emb=2*emb, Frame=max(c,p)) -> (B, Emb=3*emb, Frame=max(c,p,s))
        emb_s = self._upsample(emb_s, emb_c_p.shape[-1])
        emb_c_p_s = torch.cat([emb_c_p, emb_s], dim=1)

        # HiFi-GAN Generator
        wave_estim = super().forward(emb_c_p_s)

        return wave_estim
