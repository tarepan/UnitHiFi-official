# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import random
from pathlib import Path

import numpy as np
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
        # frame_length - Length of an analysis frame [msec]
        # tda_frame_length - Length of a 'time domain analysis' frame [msec]
        # frame_space - (maybe) hop size in time [msec]
        # nccf_thresh1 - Threshold in 'Normalized Cross Correlation Function'
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})
        # pitch :: PitchObj
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    """
    Args:
        y :: (1, T=segment) - audio
    Returns:
        spec - log-power mel-frequency spectrogram
    """
    # Warning
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    # Filters for STFT and mel
    ## mel_basis["8000_cuda"] :: Tensor(device=device)
    ## hann_window["cuda"] :: Tensor(device=device)
    global mel_basis, hann_window
    if fmax not in mel_basis:
        # `librosa.filters.mel`, default  htk=False, norm='slaney'
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # Manual padding
    ## left: centering for synthesis
    n_pad = int((n_fft-hop_size)/2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (n_pad, n_pad), mode='reflect')
    y = y.squeeze(1)

    # STFT
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    # linear-power spec
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    # linear-power mel-frequency spec
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)

    # log-power mel-frequency spectrogram
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
    Args:
        manifest - Path to the file containing audio file path and encoded contents
    Returns:
        audio_files :: List[Path] - Audio file paths
        codes :: List[NDArray[(Frame,)]] - Encoded contents
    """
    audio_files = []
    codes = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                # {"audio": "<path>", "SSL_type: "X X X ...", "duration": 1.9}
                sample = eval(line.strip())
                if 'cpc_km100' in sample:
                    k = 'cpc_km100'
                elif 'vqvae256' in sample:
                    k = 'vqvae256'
                else:
                    k = 'hubert'
                # Read content code :: (Frame,)
                codes += [torch.LongTensor([int(x) for x in sample[k].split(' ')]).numpy()]
                # Read audio file path
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes


def get_dataset_filelist(h):
    """Acquire train/val's audio file path and content code array.

    Returns:
        (train)
            training_files :: List[Path]
            training_codes :: List[NDArray[(Frame,)]]
        (valid)
            validation_files :: List[Path]
            validation_codes :: List[NDArray[(Frame,)]]
    """
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
                n_fft, num_mels, hop_size, win_size, sampling_rate, fmin, fmax_loss=None, multispkr=False, f0_stats=None, f0_normalize=False):
        """
        Args:
            training_files :: (List[Path], List[NDArray[(Frame,)]]) - Audio file path list & Content code NDArray list
            multispkr :: str - How to access speaker name

            f0_normalize - Whether to normalize fo
        """
        random.seed(1234)
        self.audio_files, self.codes = training_files
        self.segment_size, self.code_hop_size, self.sampling_rate, self.f0_normalize = segment_size, code_hop_size, sampling_rate, f0_normalize
        # `mel_spectrogram` specific values
        self.n_fft, self.num_mels, self.hop_size, self.win_size, self.fmin, self.fmax_loss = n_fft, num_mels, hop_size, win_size, fmin, fmax_loss
        self.f0_stats = torch.load(f0_stats)

        self.spk_accessor = multispkr
        # List of (Non-overlap) speaker names in the dataset
        spk_names = list(set([parse_speaker(f, self.spk_accessor) for f in self.audio_files]))
        spk_names.sort()
        # how to use: `spk_name = self.id_to_spkr[spk_idx]`
        self.id_to_spkr = spk_names
        # how to use: `spk_idx = self.spkr_to_id[spk_name]`
        self.spkr_to_id = {spk_name: spk_idx for spk_idx, spk_name in enumerate(self.id_to_spkr)}

    def _sample_interval(self, seqs):
        """
        Args:
            seqs - [audio] or [audio, code]
        """
        # N = len_audio
        N = max([v.shape[-1] for v in seqs])
        # [1, code_hop_size]
        hops = [N // v.shape[-1] for v in seqs]
        # lcm = code_hop_size
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - self.segment_size // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = start + (self.segment_size // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        """
        Returns:
            feats
                code :: (Frame,)   - Content unit series
                f0   :: (1, Frame) - Fundamental frequency series, can be normalized
                spkr :: (1,)       - Speaker index
            audio - The waveform
            filename
            melspec - Ground-Truth melspectrogram of the wave
        """
        # filename::Path, code::NDArray[(Frame,)]
        filename = self.audio_files[index]
        code = self.codes[index]

        # Waveform preprocessing
        ## Load :: (T,)
        audio, sampling_rate = load_audio(filename)
        if sampling_rate != self.sampling_rate:
            # For easy inference # raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
            import resampy
            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)
        ## Normalization
        audio = 0.95 * normalize(audio / MAX_WAV_VALUE)
        ## Length matching - Align with hop size, and match length of audio and code
        len_audio_code_scale = audio.shape[0] // self.code_hop_size
        len_code = code.shape[0]
        matched_len_code = min(len_audio_code_scale, len_code)
        matched_len_audio = matched_len_code * self.code_hop_size
        code = code[:matched_len_code]
        audio = audio[:matched_len_audio]
        ## Clipping :: (T,) -> (1, T) -> (1, T=segment) - If shorter than segment at first, repeat then clip
        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code  = np.hstack([code, code])
        audio = torch.FloatTensor(audio).unsqueeze(0)
        audio, code = self._sample_interval([audio, code])

        # Feature extraction
        ## Melspectrogram
        melspec = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss).squeeze()
        code = code.squeeze()
        ## fo
        ### Estimation by yaapt :: (1, T) -> (1, 1, Frame) -> (1, Frame)
        try:
            fo = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=False)
        except:
            fo = np.zeros((1, 1, audio.shape[-1] // 80))
        fo = fo.astype(np.float32).squeeze(0)
        ### Normalization with pre-calculated statistics
        if self.f0_normalize:
            spkr_id = self._get_spk_idx(index).item()
            spk_is_not_in_stats = spkr_id not in self.f0_stats
            mean = self.f0_stats['f0_mean'] if spk_is_not_in_stats else self.f0_stats[spkr_id]['f0_mean']
            std  = self.f0_stats['f0_std']  if spk_is_not_in_stats else self.f0_stats[spkr_id]['f0_std']
            # Normalize non-zero components
            ii = fo != 0
            fo[ii] = (fo[ii] - mean) / std
        ## Speaker
        spk_idx = self._get_spk_idx(index)

        feats = {
            'f0': fo,
            'code': code,
            'spkr': spk_idx,
        }
        return feats, audio.squeeze(0), str(filename), melspec

    def _get_spk_idx(self, uttr_idx):
        """Get speaker index from utterance index.

        Returns:
            spk_idx :: NDArray[(int64,)] - Speaker index
        """
        spkr_name = parse_speaker(self.audio_files[uttr_idx], self.spk_accessor)
        spk_idx = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spk_idx

    def __len__(self):
        return len(self.audio_files)


class F0Dataset(torch.utils.data.Dataset):
    """fo generated from audio."""
    def __init__(self, wave_paths, segment_size, sampling_rate, multispkr, f0_normalize, f0_stats):
        """
        Args:
            wave_paths    :: str  - Path to the audio file
            segment_size  :: int  - Clipping length, waveform scale
            sampling_rate :: int  - Configured waveform sampling rate
            multispkr     :: str  - How to access speaker name
            f0_normalize  :: bool - Whether to normalize the fo
            f0_stats      :: str  - Path to the fo statistics file
        """
        random.seed(1234)
        self.audio_files, self.segment_size_audio, self.sampling_rate, self.multispkr, self.fo_normalize = wave_paths, segment_size, sampling_rate, multispkr, f0_normalize
        self.fo_stats = torch.load(f0_stats)
        self.fo_caches = {}
        # Clipping parameters
        fo_hop_sec = 0.005 # 5 [msec]
        fo_hop_size = int(sampling_rate * fo_hop_sec)
        self.segment_size_fo = self.segment_size_audio // fo_hop_size
        # spk_idx accessor
        spkrs = list(set([parse_speaker(f, self.multispkr) for f in self.audio_files]))
        spkrs.sort()
        self.spkr_to_id = {spk_name: index for index, spk_name in enumerate(spkrs)}

    def __getitem__(self, uttr_idx):
        """
        Args:
            uttr_idx - Utterance index
        Returns:
            fo_segment :: NDArray[(1, Frame=segment_fo)] - A segment of fundamental frequencies, can be normalized
        """
        # np.load(f'tmp/UnitHiFi/fo/{wave_id}.npy')
        # Acquire full-length fo series :: () -> (1, Frame)
        fo_cache = self.fo_caches.get(uttr_idx)
        if fo_cache is not None:
            # Cache reuse
            fo = fo_cache
        else:
            # TODO: fo generation as preprocessing

            # fo generation
            ## Waveform preprocessing
            ### Load :: (T,)
            audio, sampling_rate = load_audio(self.audio_files[uttr_idx])
            assert sampling_rate == self.sampling_rate, f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            ### Volume normalization
            audio = 0.95 * normalize(audio / MAX_WAV_VALUE)
            ### Repeat :: (T,) -> (1, T) - If shorter than a segment, repeat until enough length
            while audio.shape[0] < self.segment_size_audio:
                audio = np.hstack([audio, audio])
            audio = torch.FloatTensor(audio).unsqueeze(0)
            ## fo extraction
            ### Estimation by yaapt :: (1, T) -> (1, 1, Frame) -> (1, Frame)
            try:
                fo = get_yaapt_f0(audio.numpy(), rate=sampling_rate, interp=False)
            except:
                fo = np.zeros((1, 1, audio.shape[-1] // 80))
            fo = fo.astype(np.float32).squeeze(0)
            ### Normalization with pre-calculated statistics
            if self.fo_normalize:
                spkr_id = self._get_spk_idx(uttr_idx).item()
                spk_is_not_in_stats = spkr_id not in self.fo_stats
                mean = self.fo_stats['f0_mean'] if spk_is_not_in_stats else self.fo_stats[spkr_id]['f0_mean']
                std  = self.fo_stats['f0_std']  if spk_is_not_in_stats else self.fo_stats[spkr_id]['f0_std']
                # Normalize non-zero components
                ii = fo != 0
                fo[ii] = (fo[ii] - mean) / std
            
            # Caching for reuse
            self.fo_caches[uttr_idx] = fo

            # Save for future reuse
            wave_id = self.audio_files[uttr_idx].stem
            np.save(f'tmp/UnitHiFi/fo/{wave_id}', fo)

        # Clipping :: (1, Frame) -> (1, Frame=segment) - Clip enough-length fo into a segment
        clip_start = random.randint(0, fo.shape[-1] - self.segment_size_fo)
        fo_segment = fo[..., clip_start : clip_start + self.segment_size_fo]

        return fo_segment

    def _get_spk_idx(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], self.multispkr)
        spkr_id = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spkr_id

    def __len__(self):
        return len(self.audio_files)
