# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import random
from pathlib import Path

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

MAX_WAV_VALUE = 32768.0


def get_yaapt_f0(audio, rate=16000, interp=False):
    """
    Args:
        audio :: (1, T) - Waveform
    Returns:
        f0 :: (1, 1, Frame) - fo series
    """
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_audio(full_path):
    # TODO: Audio channel handling
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def parse_manifest(manifest):
    audio_files = []
    codes = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                if 'cpc_km100' in sample:
                    k = 'cpc_km100'
                elif 'vqvae256' in sample:
                    k = 'vqvae256'
                else:
                    k = 'hubert'

                codes += [torch.LongTensor(
                    [int(x) for x in sample[k].split(' ')]
                ).numpy()]
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes


def get_dataset_filelist(h):
    training_files, training_codes = parse_manifest(h.input_training_file)
    validation_files, validation_codes = parse_manifest(h.input_validation_file)

    return (training_files, training_codes), (validation_files, validation_codes)


def parse_speaker(path, method) -> str:
    """Parse file path as speaker name.

    Args:
        path
        method - MethodIdentifier or function for speaker name extraction
    Returns:
        - Speaker name
    """
    if type(path) == str:
        path = Path(path)

    if method == 'parent_name':
        return path.parent.name
    elif method == 'parent_parent_name':
        return path.parent.parent.name
    elif method == '_':
        return path.name.split('_')[0]
    elif method == 'single':
        return 'A'
    elif callable(method):
        return method(path)
    else:
        raise NotImplementedError()


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, f0=None, multispkr=False, pad=None,
                 f0_stats=None, f0_normalize=False, f0_feats=False, f0_median=False,
                 f0_interp=False, vqvae=False):
        """
        Args:
            multispkr - 'Not use speaker' if False else 'How to access speaker name'

            f0_normalize - Whether to normalize fo
        """
        self.audio_files, self.codes = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.vqvae = vqvae
        self.f0 = f0
        self.f0_normalize = f0_normalize
        self.f0_feats = f0_feats
        self.f0_interp = f0_interp
        self.f0_median = f0_median
        self.f0_stats = torch.load(f0_stats) if f0_stats else None
        self.pad = pad
        self.multispkr = multispkr
        if self.multispkr:
            # List of (Non-overlap) speaker names in the dataset
            spk_names = list(set([parse_speaker(f, self.multispkr) for f in self.audio_files]))
            spk_names.sort()
            # how to use: `spk_name = self.id_to_spkr[spk_idx]`
            self.id_to_spkr = spk_names
            # how to use: `spk_idx = self.spkr_to_id[spk_name]`
            self.spkr_to_id = {spk_name: spk_idx for spk_idx, spk_name in enumerate(self.id_to_spkr)}

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        """
        Returns:
            feats
                spkr :: Optional[int] - Speaker index, exist only if `multispkr` is defined
        """
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                # raise ValueError("{} SR doesn't match target {} SR".format(
                #     sampling_rate, self.sampling_rate))
                import resampy
                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # Trim audio ending
        if self.vqvae:
            code_length = audio.shape[0] // self.code_hop_size
        else:
            code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0])
            code = self.codes[index][:code_length]
        audio = audio[:code_length * self.code_hop_size]
        assert self.vqvae or audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            if not self.vqvae:
                code = np.hstack([code, code])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        if self.vqvae:
            audio = self._sample_interval([audio])[0]
        else:
            audio, code = self._sample_interval([audio, code])

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        if self.vqvae:
            feats = {
                "code": audio.view(1, -1).numpy()
            }
        else:
            feats = {"code": code.squeeze()}

        if self.f0:
            try:
                f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=self.f0_interp)
            except:
                f0 = np.zeros((1, 1, audio.shape[-1] // 80))
            f0 = f0.astype(np.float32)
            feats['f0'] = f0.squeeze(0)

        if self.multispkr:
            feats['spkr'] = self._get_spk_idx(index)

        if self.f0_normalize:
            spkr_id = self._get_spk_idx(index).item()

            if spkr_id not in self.f0_stats:
                mean = self.f0_stats['f0_mean']
                std = self.f0_stats['f0_std']
            else:
                mean = self.f0_stats[spkr_id]['f0_mean']
                std = self.f0_stats[spkr_id]['f0_std']
            ii = feats['f0'] != 0

            if self.f0_median:
                med = np.median(feats['f0'][ii])
                feats['f0'][~ii] = med
                feats['f0'][~ii] = (feats['f0'][~ii] - mean) / std

            feats['f0'][ii] = (feats['f0'][ii] - mean) / std

            if self.f0_feats:
                feats['f0_stats'] = torch.FloatTensor([mean, std]).view(-1).numpy()

        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

    def _get_spk_idx(self, uttr_idx):
        """Get speaker index from utterance index.

        Returns:
            spk_idx :: NDArray[int64] - Speaker index
        """
        spkr_name = parse_speaker(self.audio_files[uttr_idx], self.multispkr)
        spk_idx = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spk_idx

    def __len__(self):
        return len(self.audio_files)


class F0Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, sampling_rate, multispkr, f0_normalize, f0_stats):
        """
        Args:
            training_files :: str  -
            segment_size   :: int  - Clipping length
            sampling_rate  :: int  - Configured waveform sampling rate
            multispkr      :: str  - How to access speaker name
            f0_normalize   :: bool - Whether to normalize the fo
            f0_stats       :: str  - Path to the fo statistics file
        """
        random.seed(1234)
        self.segment_size, self.sampling_rate, self.multispkr, self.f0_normalize = segment_size, sampling_rate, multispkr, f0_normalize
        self.audio_files, _ = training_files
        self.f0_stats = torch.load(f0_stats)
        # spk_idx accessor
        spkrs = list(set([parse_speaker(f, self.multispkr) for f in self.audio_files]))
        spkrs.sort()
        self.spkr_to_id = {spk_name: index for index, spk_name in enumerate(spkrs)}

    def __getitem__(self, uttr_idx):
        """
        Args:
            uttr_idx - Utterance index
        Returns:
            f0 :: NDArray[(1, Frame)] - Fundamental frequencies, can be normalized
        """
        # Waveform preprocessing
        ## Load :: (T,)
        audio, sampling_rate = load_audio(self.audio_files[uttr_idx])
        assert sampling_rate == self.sampling_rate, f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
        ## Volume normalization
        audio = 0.95 * normalize(audio / MAX_WAV_VALUE)
        ## Clipping :: (T,) -> (1, T) -> (1, T=segment) - T=segment, if shorter than segment, first repeat
        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
        audio = torch.FloatTensor(audio).unsqueeze(0)
        clip_start = random.randint(0, audio.shape[-1] - self.segment_size)
        audio = audio[..., clip_start : clip_start + self.segment_size]

        # fo generation
        ## Estimation by yaapt :: (1, T) -> (1, 1, Frame) -> (1, Frame)
        try:
            f0 = get_yaapt_f0(audio.numpy(), rate=sampling_rate, interp=False)
        except:
            f0 = np.zeros((1, 1, audio.shape[-1] // 80))
        f0 = f0.astype(np.float32).squeeze(0)
        ## Normalization with pre-calculated statistics
        if self.f0_normalize:
            spkr_id = self._get_spk_idx(uttr_idx).item()
            spk_is_not_in_stats = spkr_id not in self.f0_stats
            mean = self.f0_stats['f0_mean'] if spk_is_not_in_stats else self.f0_stats[spkr_id]['f0_mean']
            std  = self.f0_stats['f0_std']  if spk_is_not_in_stats else self.f0_stats[spkr_id]['f0_std']
            # Normalize non-zero components
            ii = f0 != 0
            f0[ii] = (f0[ii] - mean) / std

        return f0

    def _get_spk_idx(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], self.multispkr)
        spkr_id = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spkr_id

    def __len__(self):
        return len(self.audio_files)
