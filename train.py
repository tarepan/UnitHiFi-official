# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import CodeDataset, mel_spectrogram, get_dataset_filelist
from model import CodeGenerator
from models import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict

torch.backends.cudnn.benchmark = True


def train(a, h):
    """Train CodeGenerator with adversarial MPD/MSD, feature matching loss, and L1 mel-STFT loss.
    
    In special cases, VQ commitment losses are also added.

    You can use this trainer for:
      - Separated Decoder training     (generator == multi-feature HiFi_Generator)
      - Enc-VQ-Dec joint training      (generator == multi Conv_Encoder + HiFi_Generator)
      - VQVAE Encoder_content training (generator == waveform Conv_Encoder + HiFi_Generator, (VQVAE-GAN))
    """

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')

    # Deprecation
    assert h.f0_normalize is True, "Only `f0_normalize==True` is supported."

    # Models
    generator = CodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    print(generator)
    os.makedirs(a.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax_for_loss,
                            h.multispkr, h.f0_stats)
    validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax_for_loss,
                            h.multispkr, h.f0_stats)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, batch_size=h.batch_size, pin_memory=True, drop_last=True, persistent_workers=True)
    valid_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, batch_size=h.batch_size, pin_memory=True, drop_last=True, persistent_workers=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        start = time.time()
        print(f"Epoch: {epoch + 1}")

        for _, batch in enumerate(train_loader):
            start_b = time.time()

            source, out_gt, _, out_mel_gt = batch
            out_gt = out_gt.unsqueeze(1).to(device, non_blocking=True)
            out_mel_gt = out_mel_gt.to(device, non_blocking=True)
            source = {k: v.to(device, non_blocking=True) for k, v in source.items()}

            # Forward
            out_estim = generator(**source)
            # STFT loss
            out_mel_estim = mel_spectrogram(out_estim.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

            # Discriminator
            optim_d.zero_grad()
            ## Forward
            y_df_hat_r, y_df_hat_g, _, _ = mpd(out_gt, out_estim.detach())
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(out_gt, out_estim.detach())
            ## Loss
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_total = loss_disc_f + loss_disc_s
            ## Backward/Optim
            loss_disc_total.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            ## Forward
            _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(out_gt, out_estim)
            _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(out_gt, out_estim)
            ## Loss
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_mel = F.l1_loss(out_mel_gt, out_mel_estim) * 45
            loss_gen_all = loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + loss_mel
            # Backward/Optim
            loss_gen_all.backward()
            optim_g.step()

            # STDOUT logging
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(out_mel_gt, out_mel_estim).item()
                print(
                    'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(steps,
                                                                                                                loss_gen_all,
                                                                                                                mel_error,
                                                                                                                time.time() - start_b))

            # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {'generator': generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {'mpd': mpd.state_dict(), 'msd': msd.state_dict(),
                                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                    'steps': steps, 'epoch': epoch})

            # Tensorboard summary logging
            if steps % a.summary_interval == 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)

            # Validation
            if steps % a.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(valid_loader):
                        source, out_gt, _, out_mel_gt = batch
                        source = {k: v.to(device, non_blocking=False) for k, v in source.items()}

                        out_estim = generator(**source)
                        out_mel_gt = out_mel_gt.to(device, non_blocking=False)
                        out_mel_estim = mel_spectrogram(out_estim.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                        val_err_tot += F.l1_loss(out_mel_gt, out_mel_estim).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio('gt/y_{}'.format(j), out_gt[0], steps, h.sampling_rate)
                                sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(out_mel_gt[0].cpu()), steps)

                            sw.add_audio('generated/y_hat_{}'.format(j), out_estim[0], steps, h.sampling_rate)
                            y_hat_spec = mel_spectrogram(out_estim[:1].squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                            sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                            plot_spectrogram(y_hat_spec[:1].squeeze(0).cpu().numpy()), steps)

                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)
                generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        print(f'Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n')

    print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--training_steps', default=400000, type=int)
    # Validation & Logging
    parser.add_argument('--stdout_interval',     default=    5, type=int)
    parser.add_argument('--checkpoint_interval', default=  500, type=int)
    parser.add_argument('--summary_interval',    default=  100, type=int)
    parser.add_argument('--validation_interval', default= 4000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)

    train(a, h)


if __name__ == '__main__':
    main()
