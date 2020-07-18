# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from resnet import Embedder
from unet import UNet, VGGExtractor, Discriminator, AccDiscriminator
from torch.utils.tensorboard import SummaryWriter
from tryon_net import G
from visualization import board_add_images
from tqdm import tqdm
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def single_gpu_flag(args):
    return not args.distributed or (args.distributed and args.local_rank % torch.cuda.device_count() == 0)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="residual_resnet_pose")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument('--local_rank', type=int, default=0, help="gpu to use, used for distributed training")

    parser.add_argument("--use_gan", action='store_true', default=True)
    parser.add_argument("--no_consist", action='store_true')

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="residual_old")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train_residual_old(opt, train_loader, model, model_module, gmm_model, generator, image_embedder, board,
                   discriminator=None, discriminator_module=None):

    lambdas_vis_reg = {'l1': 1.0, 'prc': 0.05, 'style': 100.0}
    lambdas = {'adv': 0.1, 'identity': 300, 'match_gt': 50, 'vis_reg': .1, 'consist': 50}

    model.train()
    gmm_model.eval()
    image_embedder.eval()
    generator.eval()

    # criterion
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    vgg_extractor = VGGExtractor().cuda().eval()
    adv_criterion = utils.AdversarialLoss('lsgan').cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    if opt.use_gan:
        D_optim = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-4)

    pbar = range(opt.keep_step + opt.decay_step)
    if single_gpu_flag(opt):
        pbar = tqdm(pbar)

    for step in pbar:
        iter_start_time = time.time()
        inputs, inputs_2 = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()

        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        c_2 = inputs_2['cloth'].cuda()
        cm_2 = inputs_2['cloth_mask'].cuda()

        with torch.no_grad():
            grid, theta = gmm_model(agnostic, c)
            c = F.grid_sample(c, grid, padding_mode='border')
            cm = F.grid_sample(cm, grid, padding_mode='zeros')

            grid_2, theta_2 = gmm_model(agnostic, c_2)
            c_2 = F.grid_sample(c_2, grid_2, padding_mode='border')
            cm_2 = F.grid_sample(cm_2, grid_2, padding_mode='zeros')

            outputs = generator(torch.cat([agnostic, c], 1))
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            transfer_1 = c * m_composite + p_rendered * (1 - m_composite)

            outputs_2 = generator(torch.cat([agnostic, c_2], 1))
            p_rendered_2, m_composite_2 = torch.split(outputs_2, 3, 1)
            p_rendered_2 = F.tanh(p_rendered_2)
            m_composite_2 = F.sigmoid(m_composite_2)
            transfer_2 = c_2 * m_composite_2 + p_rendered_2 * (1 - m_composite_2)

        gt_residual = (torch.mean(im, dim=1) - torch.mean(transfer_1, dim=1)).unsqueeze(1)

        output_1 = model(transfer_1.detach(), torch.cat([gt_residual.detach(), agnostic], dim=1))
        output_2 = model(transfer_2.detach(), torch.cat([gt_residual.detach(), agnostic], dim=1))

        embedding_1 = image_embedder(output_1)
        embedding_2 = image_embedder(output_2)

        embedding_1_t = image_embedder(transfer_1)
        embedding_2_t = image_embedder(transfer_2)

        if opt.use_gan:
            # train discriminator
            real_L_logit, real_L_cam_logit, real_G_logit, real_G_cam_logit = discriminator(im)
            fake_L_logit_1, fake_L_cam_logit_1, fake_G_logit_1, fake_G_cam_logit_1 = discriminator(output_1.detach())
            fake_L_logit_2, fake_L_cam_logit_2, fake_G_logit_2, fake_G_cam_logit_2 = discriminator(output_2.detach())

            D_true_loss = adv_criterion(real_L_logit, True) + \
                          adv_criterion(real_G_logit, True) + \
                          adv_criterion(real_L_cam_logit, True) + \
                          adv_criterion(real_G_cam_logit, True)
            D_fake_loss = adv_criterion(torch.cat([fake_L_cam_logit_1, fake_L_cam_logit_2], dim=0), False) + \
                          adv_criterion(torch.cat([fake_G_cam_logit_1, fake_G_cam_logit_2], dim=0), False) + \
                          adv_criterion(torch.cat([fake_L_logit_1, fake_L_logit_2], dim=0), False) + \
                          adv_criterion(torch.cat([fake_G_logit_1, fake_G_logit_2], dim=0), False)

            D_loss = D_true_loss + D_fake_loss
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            # train generator
            fake_L_logit_1, fake_L_cam_logit_1, fake_G_logit_1, fake_G_cam_logit_1 = discriminator(output_1)
            fake_L_logit_2, fake_L_cam_logit_2, fake_G_logit_2, fake_G_cam_logit_2 = discriminator(output_2)

            G_adv_loss = adv_criterion(torch.cat([fake_L_logit_1, fake_L_logit_2], dim=0), True) + \
                         adv_criterion(torch.cat([fake_G_logit_1, fake_G_logit_2], dim=0), True) + \
                         adv_criterion(torch.cat([fake_L_cam_logit_1, fake_L_cam_logit_2], dim=0), True) + \
                         adv_criterion(torch.cat([fake_G_cam_logit_1, fake_G_cam_logit_2], dim=0), True)

        # identity loss
        identity_loss = mse_criterion(embedding_1, embedding_1_t) + mse_criterion(embedding_2, embedding_2_t)

        # vis reg loss
        output_1_feats = vgg_extractor(output_1)
        transfer_1_feats = vgg_extractor(transfer_1)
        output_2_feats = vgg_extractor(output_2)
        transfer_2_feats = vgg_extractor(transfer_2)
        # gt_feats = vgg_extractor(data['image'].cuda())

        style_reg = utils.compute_style_loss(output_1_feats, transfer_1_feats, l1_criterion) + utils.compute_style_loss(output_2_feats, transfer_2_feats, l1_criterion)
        perceptual_reg = utils.compute_perceptual_loss(output_1_feats, transfer_1_feats, l1_criterion) + utils.compute_perceptual_loss(output_2_feats, transfer_2_feats, l1_criterion)
        l1_reg = l1_criterion(output_1, transfer_1) + l1_criterion(output_2, transfer_2)

        vis_reg_loss = l1_reg * lambdas_vis_reg["l1"] + style_reg * lambdas_vis_reg["style"] + perceptual_reg * lambdas_vis_reg["prc"]

        # match gt loss
        match_gt_loss = l1_criterion(output_1, im) #* lambdas_vis_reg["l1"] + utils.compute_style_loss(output_1_feats, gt_feats, l1_criterion) * lambdas_vis_reg["style"] + utils.compute_perceptual_loss(output_1_feats, gt_feats, l1_criterion) * lambdas_vis_reg["prc"]

        # consistency loss
        consistency_loss = mse_criterion(transfer_1 - output_1, transfer_2 - output_2)

        visuals = [[im_h, shape, im],
                   [c, c_2, torch.cat([gt_residual, gt_residual, gt_residual], dim=1)],
                   [transfer_1, output_1, (transfer_1 - output_1) / 2],
                   [transfer_2, output_2, (transfer_2 - output_2) / 2]]

        total_loss = lambdas['identity'] * identity_loss + \
                     lambdas['match_gt'] * match_gt_loss + \
                     lambdas['vis_reg'] * vis_reg_loss + \
                     lambdas['consist'] * consistency_loss

        if opt.use_gan:
            total_loss += lambdas['adv'] * G_adv_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if single_gpu_flag(opt):
            if (step + 1) % opt.display_count == 0:
                board_add_images(board, str(step + 1), visuals, step + 1)
            board.add_scalar('loss/total', total_loss.item(), step + 1)
            board.add_scalar('loss/identity', identity_loss.item(), step + 1)
            board.add_scalar('loss/vis_reg', vis_reg_loss.item(), step + 1)
            board.add_scalar('loss/match_gt', match_gt_loss.item(), step + 1)
            board.add_scalar('loss/consist', consistency_loss.item(), step + 1)
            if opt.use_gan:
                board.add_scalar('loss/Dadv', D_loss.item(), step + 1)
                board.add_scalar('loss/Gadv', G_adv_loss.item(), step + 1)

            pbar.set_description(
                'step: %8d, loss: %.4f, identity: %.4f, vis_reg: %.4f, match_gt: %.4f, consist: %.4f'
                % (step + 1, total_loss.item(), identity_loss.item(),
                   vis_reg_loss.item(), match_gt_loss.item(), consistency_loss.item()))

        if (step + 1) % opt.save_count == 0 and single_gpu_flag(opt):
            save_checkpoint(model_module,
                            os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
            if opt.use_gan:
                save_checkpoint(discriminator_module,
                                os.path.join(opt.checkpoint_dir, opt.name, 'step_disc_%06d.pth' % (step + 1)))




def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.distributed = n_gpu > 1
    local_rank = opt.local_rank

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)

    board = None
    if single_gpu_flag(opt):
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    gmm_model = GMM(opt)
    load_checkpoint(gmm_model, "checkpoints/gmm_train_new/step_020000.pth")
    gmm_model.cuda()

    generator_model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    load_checkpoint(generator_model, "checkpoints/tom_train_new_2/step_040000.pth")
    generator_model.cuda()

    embedder_model = Embedder()
    load_checkpoint(embedder_model, "checkpoints/identity_train_64_dim/step_020000.pth")
    embedder_model = embedder_model.embedder_b.cuda()

    model = G(input_dim=26)
    model.apply(utils.weights_init('kaiming'))
    model.cuda()

    if opt.use_gan:
        discriminator = Discriminator()
        discriminator.apply(utils.weights_init('gaussian'))
        discriminator.cuda()

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)

    model_module = model
    if opt.use_gan:
        discriminator_module = discriminator
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
        model_module = model.module
        if opt.use_gan:
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator,
                                                                      device_ids=[local_rank],
                                                                      output_device=local_rank,
                                                                      find_unused_parameters=True)
            discriminator_module = discriminator.module

    if opt.use_gan:
        train_residual_old(opt, train_loader, model, model_module, gmm_model, generator_model, embedder_model,
                           board, discriminator=discriminator, discriminator_module=discriminator_module)
        if single_gpu_flag(opt):
            save_checkpoint({"generator": model_module, "discriminator": discriminator_module},
                            os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        train_residual_old(opt, train_loader, model, model_module, gmm_model, generator_model, embedder_model,
                           board)
        if single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))

    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
