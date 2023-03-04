# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import random
from pathlib import Path
import os

import numpy as np
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import soundfile as sf
import torch
import torch.utils.data
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

from preprocess import normalize_nonzero
from multiseries import match_length, clip_segment_random


def extract_fo(audio, sr: int):
    """Extract fundamental frequency series from a waveform.

    Args:
        audio :: (T,) - Waveforms
        sr - Sampling rate of the audio
    Returns:
        fo :: (Frame,) - fo serieses
    """
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * sr) // 2

    audio = audio.astype(np.float64)
    audio = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)

    # `pYAAPT.yaapt` :: (T,) -> (Frame,)
    # Args:
    #     signal :: basic.SignalObj - waveform, should be (T,), cannot accept multiple signals
    #     frame_length     - Length of an analysis frame [msec]
    #     tda_frame_length - Length of a 'time domain analysis' frame [msec]
    #     frame_space      - Hop size in time [msec]
    #     nccf_thresh1     - Threshold in 'Normalized Cross Correlation Function'
    # Returns:
    #     pitch :: PitchObj - Containing `.samp_values` and `.samp_interp`
    pitch = pYAAPT.yaapt(basic.SignalObj(audio, sr), **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})

    fo = pitch.samp_values.astype(np.float32)

    return fo


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    """
    Args:
        y :: (*, T=segment) - audio
    Returns:
        spec :: (*, Freq, Frame) - log-power mel-frequency spectrogram
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
    ## For ndim=1 reflection padding
    y = torch.nn.functional.pad(y.unsqueeze(-2), (n_pad, n_pad), mode='reflect')
    y = y.squeeze(-2)

    # STFT :: (*, T) -> (*, Freq, Frame, 2)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    # linear-power spec :: (*, Freq, Frame, 2) -> (*, Freq, Frame)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    # linear-power mel-frequency spec
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)

    # log-power mel-frequency spectrogram
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec


MAX_WAV_VALUE_INT16 = 2**(16-1)
def load_audio(path_audio, ref_sr: int):
    """Load an audio file and scale the waveform.
    
    Args:
        path_audio - Path to the audio file
        ref_sr - Expected sampling rate. If not matched, assert failed
    Returns:
        data - Waveform, scaled in [-1, 1]
    """
    # TODO: Audio channel handling - sf do not have mechanism of channel reduction and scaling (read the file as is).
    data, sampling_rate = sf.read(path_audio, dtype='int16')
    data = data / MAX_WAV_VALUE_INT16
    assert sampling_rate == ref_sr, f"{sampling_rate} SR doesn't match target {ref_sr} SR"
    return data


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
                codes += [np.array([int(x) for x in sample[k].split(' ')], dtype=np.int64)]
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
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels, hop_size, win_size, sampling_rate, fmin, fmax_loss, path_to_name, f0_stats):
        """
        Args:
            training_files :: (List[Path], List[NDArray[(Frame,)]]) - Audio file path list & Content code NDArray list
            path_to_name :: str - How to access speaker name
        """
        random.seed(1234)
        audio_files, codes = training_files
        self.n_audio = len(audio_files)
        self.segment_size, self.code_hop_size, self.mel_hop_size = segment_size, code_hop_size, hop_size
        self.fo_hop_size = int(sampling_rate * 0.005) # fo_hop [sec] * sr [sample/sec] = fo_hop [sample]
        self.id_to_spkr = sorted(set([parse_speaker(f, path_to_name) for f in audio_files]))

        # Preprocessing
        ## Check pre-preprocessed data
        self.path_dir_dec = "tmp/UnitHiFi/decoder/set{self.n_audio}"
        os.makedirs(self.path_dir_dec, exist_ok=True)
        n_dec_files = len(os.listdir(self.path_dir_dec))
        if self.n_audio == n_dec_files:
            print("Preprocessed data found under {path_dir_dec}. Reuse them.")
        else:
            ## Run preprocessing
            fo_stats = torch.load(f0_stats)
            spk_name_to_idx = {spk_name: spk_idx for spk_idx, spk_name in enumerate(self.id_to_spkr)}

            for uttr_idx, filename in enumerate(audio_files):
                # filename::Path, code::NDArray[(Frame,)], spk_idx::int
                code = codes[uttr_idx]
                spk_idx = spk_name_to_idx[parse_speaker(filename, path_to_name)]

                ## Speaker-specific fo mean/std
                stats = fo_stats if (spk_idx not in fo_stats) else fo_stats[spk_idx]
                fo_mean, fo_std = stats['f0_mean'], stats['f0_std']

                # Waveform preprocessing :: () -> (T,) - Load/VolumeNormalize
                audio = load_audio(filename, sampling_rate)
                audio = 0.95 * normalize(audio)

                # Feature extraction - (T,) -> code::(Frame,) & Melspec::(Freq, Frame) & fo::(Feat=1, Frame)
                code = code # Pre-extracted
                fo = normalize_nonzero(extract_fo(audio, sampling_rate), fo_mean, fo_std)
                fo = np.expand_dims(fo, axis=0)
                melspec = mel_spectrogram(torch.FloatTensor(audio), n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax_loss).numpy()

                # LengthMatch
                audio, code, fo, melspec = match_length([(audio, 1), (code, self.code_hop_size), (fo, self.fo_hop_size), (melspec, self.mel_hop_size)], segment_size)

                # Save
                np.savez(f"{self.path_dir_dec}/{uttr_idx}", audio=audio, code=code, fo=fo, melspec=melspec, spk_idx=spk_idx, filename=filename)

    def __getitem__(self, uttr_idx: int):
        """
        Returns:
            feats
                code :: (Frame,)        - Content unit series
                f0   :: (Feat=1, Frame) - Fundamental frequency series, can be normalized
                spkr :: (1,)            - Speaker index
            audio    :: (T,)            - The waveform
            filename :: str             - File name of the waveform
            melspec  :: (Freq, Frame)   - Melspectrogram of the waveform
        """

        # Query
        npz = np.load(f"{self.path_dir_dec}/{uttr_idx}.npz")
        audio, code, fo, melspec, spk_idx, filename = npz["audio"], npz["code"], npz["fo"], npz["melspec"], npz["spk_idx"], npz["filename"]

        # Clipping
        audio, code, fo, melspec = clip_segment_random([(audio, 1), (code, self.code_hop_size), (fo, self.fo_hop_size), (melspec, self.mel_hop_size)], self.segment_size)

        feats = {
            'code': torch.LongTensor(code),
            'f0':   torch.FloatTensor(fo),
            'spkr': torch.LongTensor([spk_idx]),
        }
        return feats, torch.FloatTensor(audio), str(filename), torch.FloatTensor(melspec)

    def __len__(self):
        return self.n_audio


class F0Dataset(torch.utils.data.Dataset):
    """fo generated from audio."""
    def __init__(self, wave_paths, segment_size: int, sampling_rate: int, path_to_spk: str, path_fo_stats: str):
        """
        Args:
            wave_paths :: List[str] - Path to the audio files
            segment_size  - Clipping length, waveform scale
            sampling_rate - Configured waveform sampling rate
            path_to_spk   - How to access speaker name
            path_fo_stats - Path to the fo statistics file
        """
        random.seed(1234)
        self.segment_size = segment_size
        self.fo_hop_size = int(sampling_rate * 0.005) # fo_hop [sec] * sr [sample/sec] = fo_hop [sample]
        self.n_audio = len(wave_paths)
        self.fo_caches = {}

        # Validation 
        n_unit = np.lcm.reduce([1, self.fo_hop_size])
        assert segment_size % n_unit == 0, f"segment_size {segment_size} should be N-times of n_unit {n_unit}"

        # Preprocessing
        ## Check pre-preprocessed data
        path_dir_fo = "tmp/UnitHiFi/foVQVAE/fo"
        os.makedirs(path_dir_fo, exist_ok=True)
        n_fo_files = len(os.listdir(path_dir_fo))
        if self.n_audio == n_fo_files:
            for uttr_idx in range(n_fo_files):
                self.fo_caches[uttr_idx] = np.load(f'{path_dir_fo}/{uttr_idx}.npy')
        else:
            fo_stats = torch.load(path_fo_stats)
            ## Accessor
            spk_names = sorted(set([parse_speaker(f, path_to_spk) for f in wave_paths]))
            spk_name_to_idx = {spk_name: index for index, spk_name in enumerate(spk_names)}
            ## audio-to-fo
            for uttr_idx, path_audio in enumerate(wave_paths):

                # Speaker-specific fo statistics
                spk_idx = spk_name_to_idx[parse_speaker(path_audio, path_to_spk)]
                stats = fo_stats if (spk_idx not in fo_stats) else fo_stats[spk_idx]
                fo_mean, fo_std = stats['f0_mean'], stats['f0_std']

                # Waveform preprocessing :: () -> (T,) - Load/VolumeNormalize
                audio = load_audio(path_audio, sampling_rate)
                audio = 0.95 * normalize(audio)

                # Feature Extraction :: (T,) -> fo::(Frame,)
                fo = normalize_nonzero(extract_fo(audio, sampling_rate), fo_mean, fo_std)

                # LengthMatch/Reshape :: (Frame,) -> (Frame,) -> (Feat=1, Frame)
                audio, fo = match_length([(audio, 1), (fo, self.fo_hop_size)], min_length = segment_size)
                fo = np.expand_dims(fo, axis=0)

                # Caching/Save
                self.fo_caches[uttr_idx] = fo
                np.save(f'{path_dir_fo}/{uttr_idx}', fo)

    def __getitem__(self, uttr_idx):
        """
        Returns:
            fo_segment :: NDArray[(Feat=1, Frame=segment_fo)] - A segment of normalized fundamental frequencicy series
        """

        # Query/Clipping :: () -> (Feat=1, Frame) -> (Feat=1, Frame=segment)
        fo = self.fo_caches.get(uttr_idx)
        fo_segment, *_ = clip_segment_random([(fo, self.fo_hop_size)], self.segment_size)

        return fo_segment

    def __len__(self):
        return self.n_audio
