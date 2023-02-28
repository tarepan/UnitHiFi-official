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

    # spectral normalize
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec


def load_audio(full_path):
    # TODO: Audio channel handling
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return data, sampling_rate


mel_basis = {}
hann_window = {}


def parse_manifest(manifest):
    """
    Returns:
        audio_files :: Path
        codes :: DNArray   
    """
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
    """Acquire train/val's audio file path and content code array."""
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
    def __init__(self, training_files, segment_size, code_hop_size,
                n_fft, num_mels, hop_size, win_size, sampling_rate, fmin,
                fmax_loss=None, f0=None, multispkr=False, pad=None,
                f0_stats=None, f0_normalize=False, vqvae=False):
        """
        Args:
            training_files - Audio file path list & Content code NDArray list
            multispkr - 'Not use speaker' if False else 'How to access speaker name'

            f0_normalize - Whether to normalize fo
        """
        random.seed(1234)
        self.audio_files, self.codes = training_files
        self.segment_size, self.code_hop_size, self.sampling_rate = segment_size, code_hop_size, sampling_rate
        # `mel_spectrogram` specific values
        self.n_fft, self.num_mels, self.hop_size, self.win_size, self.fmin, self.fmax_loss = n_fft, num_mels, hop_size, win_size, fmin, fmax_loss
        # Flags
        self.vqvae, self.f0, self.f0_normalize = vqvae, f0, f0_normalize
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
                code
                f0 :: Optional[] - Fundamental frequency series, can be normalized
                spkr :: Optional[int] - Speaker index, exist only if `multispkr` is defined
            audio - The waveform
            filename
            melspec - Ground-Truth melspectrogram of the wave
        """
        filename = self.audio_files[index]

        # Waveform preprocessing
        ## Load :: (T,)
        audio, sampling_rate = load_audio(filename)
        if sampling_rate != self.sampling_rate:
            # raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
            import resampy
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)
        ## Padding
        if self.pad:
            padding = self.pad - (audio.shape[-1] % self.pad)
            audio = np.pad(audio, (0, padding), "constant", constant_values=0)
        ## Normalization
        audio = 0.95 * normalize(audio / MAX_WAV_VALUE)
        ## Trim audio ending
        if self.vqvae:
            code_length = audio.shape[0] // self.code_hop_size
        else:
            code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0])
            code = self.codes[index][:code_length]
        audio = audio[:code_length * self.code_hop_size]
        assert self.vqvae or audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"
        ## Clipping :: (T,) -> (1, T) -> (1, T=segment) - If shorter than segment at first, repeat then clip
        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            if not self.vqvae:
                code = np.hstack([code, code])
        audio = torch.FloatTensor(audio).unsqueeze(0)
        if self.vqvae:
            audio = self._sample_interval([audio])[0]
        else:
            audio, code = self._sample_interval([audio, code])

        # Feature extraction
        feats = {}
        ## Melspectrogram/Content/fo/Speaker
        melspec = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss, center=False)
        feats['code'] = audio.view(1, -1).numpy() if self.vqvae else code.squeeze()
        if self.f0:
            # Estimation by yaapt :: (1, T) -> (1, 1, Frame) -> (1, Frame)
            try:
                f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=False)
            except:
                f0 = np.zeros((1, 1, audio.shape[-1] // 80))
            f0 = f0.astype(np.float32).squeeze(0)
            # Normalization with pre-calculated statistics
            if self.f0_normalize:
                spkr_id = self._get_spk_idx(index).item()
                spk_is_not_in_stats = spkr_id not in self.f0_stats
                mean = self.f0_stats['f0_mean'] if spk_is_not_in_stats else self.f0_stats[spkr_id]['f0_mean']
                std  = self.f0_stats['f0_std']  if spk_is_not_in_stats else self.f0_stats[spkr_id]['f0_std']
                # Normalize non-zero components
                ii = f0 != 0
                f0[ii] = (f0[ii] - mean) / std
            feats['f0'] = f0
        if self.multispkr:
            feats['spkr'] = self._get_spk_idx(index)

        return feats, audio.squeeze(0), str(filename), melspec.squeeze()

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
    def __init__(self, wave_paths, segment_size, sampling_rate, multispkr, f0_normalize, f0_stats):
        """
        Args:
            wave_paths    :: str  - Path to the audio file
            segment_size  :: int  - Clipping length
            sampling_rate :: int  - Configured waveform sampling rate
            multispkr     :: str  - How to access speaker name
            f0_normalize  :: bool - Whether to normalize the fo
            f0_stats      :: str  - Path to the fo statistics file
        """
        random.seed(1234)
        self.audio_files, self.segment_size, self.sampling_rate, self.multispkr, self.f0_normalize = wave_paths, segment_size, sampling_rate, multispkr, f0_normalize
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
