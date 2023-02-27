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

        # Content - Learnable Encoder-VQ | Learnable Embdding
        self.code_encoder, self.code_vq, self.dict = None, None, None
        if h.get('lambda_commit_code', None):
            # Enc-VQ (`code_encoder`+`code_vq`)
            self.code_encoder = Encoder(**h.code_encoder_params)
            self.code_vq = Bottleneck(**h.code_vq_params)
        else:
            # Emb (`dict`)
            self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)

        # fo - None | Learnable Encoder-VQ | Fixed Encoder-VQ + Learnable Embedding
        self.use_f0 = h.get('f0', None)
        self.encoder, self.vq, self.quantizer = None, None, None
        if h.get("lambda_commit", None):
            # NOTE: Current configs do NOT used
            # Enc-VQ (`encoder`+`vq`)
            assert self.use_f0, "Requires F0 set"
            self.encoder = Encoder(**h.f0_encoder_params)
            self.vq = Bottleneck(**h.f0_vq_params)
        if h.get('f0_quantizer_path', None):
            # Fixed Enc-VQ + Emb (`quantizer` + `f0_dict`)
            assert self.use_f0, "Requires F0 set"
            self.quantizer = Quantizer(AttrDict(h.f0_quantizer))
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
        """
        Args:
            kwargs
                code - 
                f0 - 
                spkr :: Optional[int] - speaker index (global feature)
        Returns:
            wave_estim - Estimated waveform
            (code_commit_losses, f0_commit_losses) :: Optional - content/fo Commitment loss
            (code_metrics, f0_metrics) :: Optional - content/fo Metrics
        """
        #### PreNet ###############################################################
        # Content
        code = kwargs['code']
        code_commit_losses, code_metrics = None, None
        if self.code_vq:
            if code.dtype is torch.int64:
                # VQ Query: index -> z_q
                x = self.code_vq.level_blocks[0].k[code].transpose(1, 2)
            else:
                # Encode + VQ: feat -> z -> z_q
                code_h = self.code_encoder(code)
                _, code_h_q, code_commit_losses, code_metrics = self.code_vq(code_h)
                x = code_h_q[0]
        else:
            # Embedding: index -> emb
            x = self.dict(code).transpose(1, 2)

        # [Optional] fo
        f0_commit_losses, f0_metrics = None, None
        if self.use_f0:
            f0 = kwargs['f0']
            if self.vq:
                # Enc-VQ
                f0_h = self.encoder(f0)
                _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
                f0 = f0_h_q[0]
            elif self.quantizer:
                # Fixed-Enc-VQ + Emb
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

        # [Optional] Global speaker embedding
        if self.use_spk:
            global_spk_emb = self.spk_emb(kwargs['spkr']).transpose(1, 2)
            # Up↑/Concat
            spk_emb_series = self._upsample(global_spk_emb, x.shape[-1])
            x = torch.cat([x, spk_emb_series], dim=1)

        # [Optional] Other feature
        for k, feat in kwargs.items():
            if k in ['spkr', 'code', 'f0']:
                continue
            # Up↑/Concat
            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)
        #### /PreNet ##############################################################

        # HiFi-GAN
        wave_estim = super().forward(x)

        # Returns
        if self.vq or self.code_vq:
            # For learnable encoders
            return wave_estim, (code_commit_losses, f0_commit_losses), (code_metrics, f0_metrics)
        else:
            return wave_estim
