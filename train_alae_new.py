# Copyright 2019-2020 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.utils.data
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader import *
from tqdm import tqdm
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
from model import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image

import wandb
wandb.login()

## Prepare dataset
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])), 
                                           batch_size=128, 
                                           shuffle=True)

print('[INFO] Length of dataloader: ', len(train_loader))

def save_sample(lod2batch, tracker, sample, samplez, x, logger, model, cfg, encoder_optimizer, decoder_optimizer):
    os.makedirs('results', exist_ok=True)

    logger.info('\n[%d/%d] - ptime: %.2f, %s, blend: %.3f, lr: %.12f,  %.12f, max mem: %f",' % (
        (lod2batch.current_epoch + 1), cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
        lod2batch.get_blend_factor(),
        encoder_optimizer.param_groups[0]['lr'], decoder_optimizer.param_groups[0]['lr'],
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    with torch.no_grad():
        model.eval()
        sample = sample[:lod2batch.get_per_GPU_batch_size()]
        samplez = samplez[:lod2batch.get_per_GPU_batch_size()]

        needed_resolution = model.decoder.layer_to_resolution[lod2batch.lod]
        sample_in = sample
        while sample_in.shape[2] > needed_resolution:
            sample_in = F.avg_pool2d(sample_in, 2, 2)
        assert sample_in.shape[2] == needed_resolution

        blend_factor = lod2batch.get_blend_factor()
        if lod2batch.in_transition:
            needed_resolution_prev = model.decoder.layer_to_resolution[lod2batch.lod - 1]
            sample_in_prev = F.avg_pool2d(sample_in, 2, 2)
            sample_in_prev_2x = F.interpolate(sample_in_prev, needed_resolution)
            sample_in = sample_in * blend_factor + sample_in_prev_2x * (1.0 - blend_factor)

        Z, _ = model.encode(sample_in, lod2batch.lod, blend_factor)

        if cfg.MODEL.Z_REGRESSION:
            Z = model.mapping_fl(Z[:, 0])
        else:
            Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

        rec1 = model.decoder(Z, lod2batch.lod, blend_factor, noise=False)
        rec2 = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)

        # rec1 = F.interpolate(rec1, sample.shape[2])
        # rec2 = F.interpolate(rec2, sample.shape[2])
        # sample_in = F.interpolate(sample_in, sample.shape[2])

        Z = model.mapping_fl(samplez)
        g_rec = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)
        # g_rec = F.interpolate(g_rec, sample.shape[2])

        resultsample = torch.cat([sample_in, rec1, rec2, g_rec], dim=0)

        @utils.async_func
        def save_pic(x_rec):
            tracker.register_means(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
            tracker.plot()

            result_sample = x_rec * 0.5 + 0.5
            result_sample = result_sample.cpu()
            f = os.path.join(cfg.OUTPUT_DIR,
                             'sample_%d_%d.jpg' % (
                                 lod2batch.current_epoch + 1,
                                 lod2batch.iteration // 1000)
                             )
            print("Saved to %s" % f)
            save_image(result_sample, f, nrow=min(32, lod2batch.get_per_GPU_batch_size()))

        save_pic(resultsample)

def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    ## INITIALIZE MODEL
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        z_regression=cfg.MODEL.Z_REGRESSION
    )

    print('#'*80)
    print(model)
    print('#'*80)
    model.cuda(local_rank)
    model.train()

    if local_rank == 0:
        model_s = Model(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=cfg.MODEL.CHANNELS,
            generator=cfg.MODEL.GENERATOR,
            encoder=cfg.MODEL.ENCODER,
            z_regression=cfg.MODEL.Z_REGRESSION)
        model_s.cuda(local_rank)
        model_s.eval()
        model_s.requires_grad_(False)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None

        decoder = model.module.decoder
        encoder = model.module.encoder
        mapping_tl = model.module.mapping_tl
        mapping_fl = model.module.mapping_fl
        dlatent_avg = model.module.dlatent_avg
    else:
        decoder = model.decoder
        encoder = model.encoder
        mapping_tl = model.mapping_tl
        mapping_fl = model.mapping_fl
        dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    ## OPTIMIZERS
    decoder_optimizer = LREQAdam([
        {'params': decoder.parameters()},
        {'params': mapping_fl.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': encoder.parameters()},
        {'params': mapping_tl.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)
    ## LR SCHEDULER
    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_tl,
        'mapping_fl': mapping_fl,
        'dlatent_avg': dlatent_avg
    }
    if local_rank == 0:
        model_dict['discriminator_s'] = model_s.encoder
        model_dict['generator_s'] = model_s.decoder
        model_dict['mapping_tl_s'] = model_s.mapping_tl
        model_dict['mapping_fl_s'] = model_s.mapping_fl

    ## LOSS TRACKER
    tracker = LossTracker(cfg.OUTPUT_DIR)

    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    
    layer_to_resolution = decoder.layer_to_resolution

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.tensor(latents).float().cuda()
    print(samplez.shape)
    sample, _ = next(iter(train_loader))
    sample = sample[:32]
    sample = sample.cuda()
    print('[INFO] ', sample.shape)

    lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(train_loader) * world_size)

    lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, decoder_optimizer])

    ## Training process starts
    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()

        lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer])
        scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod)

        need_permute = False
        epoch_start_time = time.time()

        ## Train step
        i = 0
        for batch_id, (x_orig, label) in tqdm(enumerate(train_loader)):
            i += 1
            x_orig = x_orig.cuda()
            # x_orig = x_orig.view(x_orig.shape[0], -1)
            # logger.info(x_orig.shape)
            with torch.no_grad():
                if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                    continue
                if need_permute:
                    x_orig = x_orig.permute(0, 3, 1, 2)
                # x_orig = (x_orig / 127.5 - 1.)

                blend_factor = lod2batch.get_blend_factor()

                needed_resolution = layer_to_resolution[lod2batch.lod]
                x = x_orig

                if lod2batch.in_transition:
                    needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                    x_prev = F.avg_pool2d(x_orig, 2, 2)
                    x_prev_2x = F.interpolate(x_prev, needed_resolution)
                    x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            x.requires_grad = True

            ## ΤΗΡΕΕ STEP TRAINING 
            
            encoder_optimizer.zero_grad()
            loss_d = model(x, lod2batch.lod, blend_factor, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss_g = model(x, lod2batch.lod, blend_factor, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(x, lod2batch.lod, blend_factor, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            (lae).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if local_rank == 0:
                betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                model_s.lerp(model, betta)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            lod_for_saving_model = lod2batch.lod
            lod2batch.step()


        scheduler.step()

        # wandb.log({'epoch': epoch})
        print('[INFO] After Epoch: {} Loss: {}'.format(epoch+1, tracker.wandb_logger()))

        save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s, cfg, encoder_optimizer, decoder_optimizer)

    logger.info("Training finish!... save training results")

    tracker.plot()

if __name__ == "__main__":
    # wandb.init(entity='authors', project='alae')

    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/ffhq.yaml',
        world_size=gpu_count)
