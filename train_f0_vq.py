# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import F0Dataset, get_dataset_filelist
from model import FoVQVAE
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, build_env, AttrDict

torch.backends.cudnn.benchmark = True


def train(a, h):
    """
    Model: `FoVQVAE`
    Optim: `AdamW`
    Sched: `ExponentialLR`
    Data:  `F0Dataset`
    Loss:  MSE + commitment
    """

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')

    vqvae = FoVQVAE(h).to(device)
    print(vqvae)

    # Restore (1/2)
    os.makedirs(a.checkpoint_path, exist_ok=True)
    print(f"checkpoints directory : {a.checkpoint_path}")
    cp_g = None
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
    steps = 0
    if cp_g is None:
        last_epoch = -1
        state_dict_g = None
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        vqvae.load_state_dict(state_dict_g['generator'])
        steps = state_dict_g['steps'] + 1
        last_epoch = state_dict_g['epoch']

    # Init & Restore (2/2)
    optim_g = torch.optim.AdamW(vqvae.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if state_dict_g is not None:
        optim_g.load_state_dict(state_dict_g['optim_g'])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)

    # Params
    lambda_commit = h.get('lambda_commit', None)

    # Data
    train_filelist, valid_filelist = get_dataset_filelist(h)
    train_wave_paths, valid_wave_paths = train_filelist[0], valid_filelist[0]
    trainset = F0Dataset(train_wave_paths, h.segment_size, h.sampling_rate, h.multispkr, h.f0_normalize, h.f0_stats)
    validset = F0Dataset(valid_wave_paths, h.segment_size, h.sampling_rate, h.multispkr, h.f0_normalize, h.f0_stats)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, batch_size=h.batch_size, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, batch_size=h.batch_size, pin_memory=True, drop_last=True)

    # Logger
    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    vqvae.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        #### Epoch ###################################################################################
        start = time.time()
        print(f"Epoch: {epoch + 1}")

        for batch in train_loader:
            #### Step ############################################################################
            start_b = time.time()
            optim_g.zero_grad()

            # I/O
            fo = batch
            fo = fo.to(device, non_blocking=True)
            in_fo, fo_gt = {'f0': fo}, fo

            # Forward/Loss/Backward/Optim
            fo_estim, commit_losses, metrics = vqvae(**in_fo)
            loss_total = F.mse_loss(fo_estim, fo_gt) + lambda_commit * commit_losses[0]
            loss_total.backward()
            optim_g.step()

            # Validation & Logging
            ## STDOUT logging
            if steps % a.stdout_interval == 0:
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, s/b : {:4.3f}'.format(steps, loss_total, time.time() - start_b))
            ## checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {'generator': vqvae.state_dict(), 'optim_g': optim_g.state_dict(), 'steps': steps, 'epoch': epoch})
            ## Tensorboard summary logging
            if steps % a.summary_interval == 0:
                commit_loss = commit_losses[0]
                metric = metrics[0]
                sw.add_scalar("training/gen_loss_total", loss_total,            steps)
                sw.add_scalar("training/commit_error", commit_loss,             steps)
                sw.add_scalar("training/used_curr", metric['used_curr'].item(), steps)
                sw.add_scalar("training/entropy",   metric['entropy'].item(),   steps)
                sw.add_scalar("training/usage",     metric['usage'].item(),     steps)
            ## Validation
            if steps % a.validation_interval == 0:
                vqvae.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(valid_loader):
                        # Items
                        fo = batch
                        fo = fo.to(device, non_blocking=True)
                        in_fo, fo_gt = {'f0': fo}, fo
                        # Forward/Loss
                        fo_estim, commit_losses, _ = vqvae(**in_fo)
                        val_err_tot += (F.mse_loss(fo_estim, fo_gt).item() + lambda_commit * commit_losses[0])
                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("validation/mel_spec_error", val_err,   steps)
                    sw.add_scalar("validation/commit_error", commit_loss, steps)
                vqvae.train()
            # /Validation & Logging

            steps += 1
            if steps >= a.training_steps:
                break
            #### /Step ###########################################################################

        scheduler_g.step()

        print(f'Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n')
        #### /Epoch ##################################################################################

    print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=10000, type=int)
    parser.add_argument('--training_steps', default=400000, type=int)
    # Validation & Logging
    parser.add_argument('--stdout_interval',     default=    5, type=int)
    parser.add_argument('--checkpoint_interval', default=10000, type=int)
    parser.add_argument('--summary_interval',    default=  100, type=int)
    parser.add_argument('--validation_interval', default= 1000, type=int)

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
